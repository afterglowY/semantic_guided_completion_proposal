"""
Training Script for Semantic-Guided Point Cloud Completion

Three-phase training schedule:
  Phase 1 (0–20% epochs): Geometric warmup — λ_sem=0.0, PTv3 fully frozen
  Phase 2 (20–60% epochs): Semantic intro  — λ_sem linear 0→0.3, PTv3 frozen
  Phase 3 (60–100% epochs): Joint finetune — λ_sem=0.3, PTv3 last 2 layers unfrozen

Usage:
    python train.py --config cfgs/sgc_default.yaml
    python train.py --config cfgs/sgc_default.yaml --resume ckpts/last.pth
    python train.py --config cfgs/sgc_default.yaml \
        --pretrained_ptv3 weights/ptv3.pth \
        --pretrained_adapointr weights/adapointr.pth
"""

import os
import sys
import time
import argparse
import yaml
import logging
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from addict import Dict as AttrDict

from models.sgc import SemanticGuidedCompletion
from datasets.dataloader import FacadeCompletionDataset, CurriculumScheduler, build_dataloaders


# ======================== CLI ========================

def parse_args():
    p = argparse.ArgumentParser(description='SGC Training')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--pretrained_ptv3', type=str, default=None)
    p.add_argument('--pretrained_adapointr', type=str, default=None)
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--val_freq', type=int, default=10)
    p.add_argument('--save_freq', type=int, default=50)
    p.add_argument('--log_freq', type=int, default=10)
    return p.parse_args()


# ======================== Helpers ========================

def load_config(path):
    with open(path) as f:
        return AttrDict(yaml.safe_load(f))


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(os.path.join(exp_dir, 'train.log')),
                  logging.StreamHandler()])
    return logging.getLogger(__name__)


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ======================== Training ========================

def train_one_epoch(model, loader, optimizer, epoch, max_epochs,
                    device, logger, args, scaler=None):
    model.train()
    model.set_epoch(epoch, max_epochs)

    meters = defaultdict(AverageMeter)
    t0 = time.time()

    for i, batch in enumerate(loader):
        partial = batch['partial'].to(device)
        gt      = batch['gt'].to(device)
        normals = batch.get('normals')
        if normals is not None:
            normals = normals.to(device)
        gt_sem = batch.get('gt_sem')
        if gt_sem is not None:
            gt_sem = gt_sem.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                ret = model(partial, normals=normals)
                loss, ld = model.get_loss(ret, gt, gt_sem=gt_sem, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            ret = model(partial, normals=normals)
            loss, ld = model.get_loss(ret, gt, gt_sem=gt_sem, epoch=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k, v in ld.items():
            meters[k].update(v)

        if i % args.log_freq == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}/{max_epochs}] [{i}/{len(loader)}] '
                f'phase={model.training_phase} lr={lr:.6f} '
                f'loss={meters["total"].avg:.4f} '
                f'cd_fine={meters["cd_fine"].avg:.6f} '
                f'sem={meters["sem_total"].avg:.4f} '
                f'time={elapsed:.1f}s')

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, epoch, max_epochs, device, logger):
    model.eval()
    model.set_epoch(epoch, max_epochs)
    meters = defaultdict(AverageMeter)

    for batch in loader:
        partial = batch['partial'].to(device)
        gt      = batch['gt'].to(device)
        normals = batch.get('normals')
        if normals is not None:
            normals = normals.to(device)
        gt_sem = batch.get('gt_sem')
        if gt_sem is not None:
            gt_sem = gt_sem.to(device)

        ret = model(partial, normals=normals)
        _, ld = model.get_loss(ret, gt, gt_sem=gt_sem, epoch=epoch)
        for k, v in ld.items():
            meters[k].update(v)

    m = {k: v.avg for k, v in meters.items()}
    cd_key = 'cd_fine' if 'cd_fine' in m else 'cd'
    logger.info(f'Val [{epoch}] CD={m.get(cd_key, 0):.6f} total={m.get("total",0):.4f}')
    return m


def save_ckpt(model, optim, sched, epoch, metrics, path):
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optim.state_dict(), 'scheduler': sched.state_dict(),
                'metrics': metrics}, path)


# ======================== Main ========================

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = args.exp_name or f'sgc_{datetime.now():%Y%m%d_%H%M%S}'
    exp_dir  = os.path.join('experiments', exp_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logging(exp_dir)
    logger.info(f'Config: {args.config}')
    logger.info(f'Device: {device}')

    # ---- Model ----
    model = SemanticGuidedCompletion(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Params: {n_params:,} total, {n_train:,} trainable')

    if args.pretrained_ptv3:
        model.load_pretrained_ptv3(args.pretrained_ptv3)
        logger.info(f'Loaded PTv3 from {args.pretrained_ptv3}')
    if args.pretrained_adapointr:
        model.load_pretrained_adapointr(args.pretrained_adapointr)
        logger.info(f'Loaded AdaPoinTr from {args.pretrained_adapointr}')

    # ---- Optimizer & scheduler ----
    base_lr = cfg.optimizer.kwargs.lr
    wd      = cfg.optimizer.kwargs.weight_decay
    param_groups = model.get_optimizer_params(base_lr, wd)
    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)

    max_epochs = cfg.max_epoch
    warmup_ep  = cfg.get('warmup_epochs', 10)

    def lr_lambda(ep):
        if ep < warmup_ep:
            return max(ep / warmup_ep, 0.001)
        prog = (ep - warmup_ep) / max(max_epochs - warmup_ep, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Data ----
    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg, args.num_workers)
    curriculum = CurriculumScheduler(train_ds, max_epochs)
    logger.info(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

    # ---- AMP ----
    use_amp = cfg.get('use_amp', True)
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Resume ----
    start_epoch = 0
    best_cd     = float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        best_cd = ck['metrics'].get('best_cd', float('inf'))
        logger.info(f'Resumed from epoch {start_epoch}')

    # ---- Training loop ----
    for epoch in range(start_epoch, max_epochs):
        curriculum.step(epoch)

        train_one_epoch(model, train_loader, optimizer, epoch, max_epochs,
                        device, logger, args, scaler)
        scheduler.step()

        if epoch % args.val_freq == 0 or epoch == max_epochs - 1:
            vm = validate(model, val_loader, epoch, max_epochs, device, logger)
            cd_key = 'cd_fine' if 'cd_fine' in vm else 'cd'
            val_cd = vm.get(cd_key, float('inf'))
            model.update_loss_safety(val_cd)

            if val_cd < best_cd:
                best_cd = val_cd
                save_ckpt(model, optimizer, scheduler, epoch,
                          {'best_cd': best_cd, **vm},
                          os.path.join(ckpt_dir, 'best.pth'))
                logger.info(f'New best CD: {best_cd:.6f}')

        if epoch % args.save_freq == 0:
            save_ckpt(model, optimizer, scheduler, epoch,
                      {'best_cd': best_cd},
                      os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pth'))

        save_ckpt(model, optimizer, scheduler, epoch,
                  {'best_cd': best_cd},
                  os.path.join(ckpt_dir, 'last.pth'))

    logger.info(f'Done. Best CD: {best_cd:.6f}')


if __name__ == '__main__':
    main()
