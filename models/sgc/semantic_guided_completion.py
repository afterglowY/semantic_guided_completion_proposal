"""
Semantic-Guided Completion (SGC) — Top-Level Model

Dual-branch architecture:
  Branch A: PTv3 (frozen semantic feature extractor)
  Branch B: Modified AdaPoinTr (semantic-aware completion backbone)

Three-phase training:
  Phase 1 (0-20%):  Geometric warmup — PTv3 frozen, lambda_sem=0
  Phase 2 (20-60%): Semantic intro  — PTv3 frozen, lambda_sem ramps up
  Phase 3 (60-100%): Joint finetune — PTv3 last 2 layers unfrozen, lambda_sem=0.3
"""

import torch
import torch.nn as nn

from addict import Dict as AttrDict

from models.sgc.ptv3_backbone import PTv3SemanticBackbone
from models.sgc.adapointr_semantic import SemanticAdaPoinTr
from models.sgc.losses import SemanticGuidedCompletionLoss


class SemanticGuidedCompletion(nn.Module):
    """
    Top-level model combining PTv3 semantic backbone with SemanticAdaPoinTr.

    Usage:
        model = SemanticGuidedCompletion(config)
        model.set_epoch(epoch, max_epochs)
        ret = model(partial_pc, normals=normals)
        loss, loss_dict = model.get_loss(ret, gt, gt_sem=labels, epoch=epoch)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = getattr(config, 'num_classes', 12)
        self.sem_feat_dim = getattr(config, 'sem_feat_dim', 256)
        self.use_normals = getattr(config, 'use_normals', True)

        # Phase tracking
        self.training_phase = 'warmup'
        self.current_epoch = 0
        self.max_epochs = 300

        # Branch A: PTv3 Semantic Backbone
        ptv3_cfg = getattr(config, 'ptv3_config', AttrDict())
        self.semantic_backbone = PTv3SemanticBackbone(
            num_classes=self.num_classes,
            sem_feat_dim=self.sem_feat_dim,
            in_channels=6 if self.use_normals else 3,
            grid_size=ptv3_cfg.get('grid_size', 0.05),
            enc_depths=ptv3_cfg.get('enc_depths', [2, 2, 2, 6, 2]),
            enc_channels=ptv3_cfg.get('enc_channels', [32, 64, 128, 256, 512]),
            enc_num_head=ptv3_cfg.get('enc_num_head', [2, 4, 8, 16, 32]),
            enc_patch_size=ptv3_cfg.get('enc_patch_size', [1024, 1024, 1024, 1024, 1024]),
            dec_depths=ptv3_cfg.get('dec_depths', [2, 2, 2, 2]),
            dec_channels=ptv3_cfg.get('dec_channels', [64, 64, 128, 256]),
            dec_num_head=ptv3_cfg.get('dec_num_head', [4, 4, 8, 16]),
            dec_patch_size=ptv3_cfg.get('dec_patch_size', [1024, 1024, 1024, 1024]),
            enable_flash=ptv3_cfg.get('enable_flash', True),
            frozen=ptv3_cfg.get('frozen', True),
        )

        # Branch B: Semantic-enhanced AdaPoinTr
        self.completion_backbone = SemanticAdaPoinTr(config)

        # Loss
        loss_cfg = getattr(config, 'loss_config', AttrDict())
        self.loss_module = SemanticGuidedCompletionLoss(loss_cfg)

    def set_epoch(self, epoch, max_epochs):
        """Update training phase based on epoch progress."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs
        self.loss_module.max_epochs = max_epochs
        progress = epoch / max(max_epochs, 1)

        if progress < 0.2:
            self.training_phase = 'warmup'
            # Ensure PTv3 is frozen
            if not self.semantic_backbone.frozen:
                self.semantic_backbone.freeze()
        elif progress < 0.6:
            self.training_phase = 'semantic_intro'
            # PTv3 still frozen
            if not self.semantic_backbone.frozen:
                self.semantic_backbone.freeze()
        else:
            self.training_phase = 'joint_finetune'
            # Unfreeze last 2 PTv3 decoder layers
            if self.semantic_backbone.frozen:
                self.semantic_backbone.unfreeze_last_layers(n_layers=2)

    def get_optimizer_params(self, base_lr, wd):
        """
        Return parameter groups with different learning rates.

        - Completion backbone (AdaPoinTr): full lr
        - Semantic heads (sem_head, feat_proj): full lr
        - PTv3 backbone (when unfrozen): 0.1x lr
        """
        completion_params = []
        head_params = []
        backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('semantic_backbone.ptv3'):
                backbone_params.append(param)
            elif name.startswith('semantic_backbone'):
                head_params.append(param)
            else:
                completion_params.append(param)

        groups = [
            {'params': completion_params, 'lr': base_lr, 'weight_decay': wd},
            {'params': head_params, 'lr': base_lr, 'weight_decay': wd},
        ]
        if backbone_params:
            groups.append({
                'params': backbone_params,
                'lr': base_lr * 0.1,
                'weight_decay': wd
            })
        return groups

    def load_pretrained_ptv3(self, path):
        """Load pretrained PTv3 weights."""
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        # Filter to only PTv3 keys
        ptv3_state = {}
        for k, v in state.items():
            # Try direct match
            if k.startswith('ptv3.'):
                ptv3_state[k.replace('ptv3.', '')] = v
            elif k.startswith('backbone.'):
                ptv3_state[k.replace('backbone.', '')] = v
            else:
                ptv3_state[k] = v
        missing, unexpected = self.semantic_backbone.ptv3.load_state_dict(
            ptv3_state, strict=False)
        if missing:
            print(f'PTv3 missing keys: {len(missing)}')
        if unexpected:
            print(f'PTv3 unexpected keys: {len(unexpected)}')

    def load_pretrained_adapointr(self, path):
        """Load pretrained AdaPoinTr weights."""
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        # Map keys from original AdaPoinTr to our SemanticAdaPoinTr
        mapped = {}
        for k, v in state.items():
            if k.startswith('base_model.'):
                mapped[k] = v
            elif k.startswith('increase_dim.') or k.startswith('reduce_map.') or k.startswith('decode_head.'):
                mapped[k] = v
        missing, unexpected = self.completion_backbone.load_state_dict(
            mapped, strict=False)
        if missing:
            print(f'AdaPoinTr missing keys: {len(missing)}')
        if unexpected:
            print(f'AdaPoinTr unexpected keys: {len(unexpected)}')

    def forward(self, partial, normals=None):
        """
        Args:
            partial: (B, N, 3) partial point cloud
            normals: (B, N, 3) point normals (optional)

        Returns:
            ret: dict from SemanticAdaPoinTr
        """
        B, N, _ = partial.shape

        # Prepare features for PTv3
        if normals is not None and self.use_normals:
            ptv3_feat = torch.cat([partial, normals], dim=-1)  # B, N, 6
        else:
            ptv3_feat = partial  # B, N, 3

        # Branch A: Extract semantic features
        # PTv3 expects (N_total, C) with batch indices
        # Flatten batch dimension
        batch_idx = torch.arange(B, device=partial.device).unsqueeze(1).expand(-1, N).reshape(-1)
        coord_flat = partial.reshape(-1, 3)
        feat_flat = ptv3_feat.reshape(-1, ptv3_feat.size(-1))
        offset = torch.arange(1, B + 1, device=partial.device) * N

        sem_feat, sem_pred = self.semantic_backbone(
            coord=coord_flat,
            feat=feat_flat,
            batch=batch_idx,
            offset=offset,
        )

        # Reshape back to (B, N, D)
        sem_feat = sem_feat.reshape(B, N, -1)

        # Branch B: Semantic-guided completion
        ret = self.completion_backbone(partial, sem_feat=sem_feat)

        # Attach backbone semantic predictions for auxiliary loss
        ret['backbone_sem_pred'] = sem_pred.reshape(B, N, -1)

        return ret

    def get_loss(self, ret, gt, gt_sem=None, epoch=None):
        """
        Compute multi-task loss.

        Args:
            ret: dict from forward()
            gt: (B, M, 3) ground truth
            gt_sem: (B, M) ground truth semantic labels
            epoch: current epoch (uses self.current_epoch if None)

        Returns:
            total_loss, loss_dict
        """
        if epoch is None:
            epoch = self.current_epoch

        factor = self.completion_backbone.factor

        return self.loss_module(ret, gt, gt_sem=gt_sem, epoch=epoch, factor=factor)

    def update_loss_safety(self, val_cd):
        """Update safety valve with current validation CD."""
        self.loss_module.update_safety_valve(val_cd, self.current_epoch)
