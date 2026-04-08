"""
Semantic-Guided Completion Loss

Multi-task loss with geometry absolute priority:
  L_total = lambda_geo * L_geo + lambda_sem * L_sem + lambda_denoise * L_denoise

Where:
  L_geo   = CD_coarse + CD_fine (Chamfer Distance L1)
  L_sem   = CE(sem_logits, gt_sem) + optional F-score bonus
  L_denoise = 0.5 * CD(denoised_fine, denoised_target)

Dynamic scheduling:
  Phase 1 (0-20%):  lambda_sem = 0.0
  Phase 2 (20-60%): lambda_sem linearly increases 0 -> max_lambda_sem
  Phase 3 (60-100%): lambda_sem = max_lambda_sem

Safety valve: if val CD degrades >20% from best, halve lambda_sem for 5 epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.chamfer_dist import ChamferDistanceL1
from models.adapointr.transformer_utils import knn_point, index_points


class SemanticGuidedCompletionLoss(nn.Module):
    """Multi-task loss for semantic-guided point cloud completion."""

    def __init__(self, config):
        super().__init__()
        self.max_epochs = getattr(config, 'max_epochs', 300)
        self.max_lambda_sem = 0.3
        self.f_score_threshold = getattr(config, 'f_score_threshold', 0.05)

        # Loss weights
        self.alpha = getattr(config, 'alpha', 1.0)   # coarse CD weight
        self.beta = getattr(config, 'beta', 0.5)      # denoise weight
        self.gamma = getattr(config, 'gamma', 1.0)    # fine CD weight

        self.chamfer = ChamferDistanceL1()

        # Safety valve state
        self.best_val_cd = float('inf')
        self.safety_halve_until = -1
        self.current_lambda_sem = 0.0

    def get_lambda_sem(self, epoch):
        """Compute lambda_sem based on training phase and safety valve."""
        progress = epoch / max(self.max_epochs, 1)

        if progress < 0.2:
            # Phase 1: warmup - no semantic loss
            lam = 0.0
        elif progress < 0.6:
            # Phase 2: linear ramp from 0 to max_lambda_sem
            phase_progress = (progress - 0.2) / 0.4
            lam = self.max_lambda_sem * phase_progress
        else:
            # Phase 3: full semantic loss
            lam = self.max_lambda_sem

        # Safety valve: halve if triggered
        if epoch < self.safety_halve_until:
            lam = lam * 0.5

        self.current_lambda_sem = lam
        return lam

    def update_safety_valve(self, val_cd, epoch):
        """
        Check if val CD has degraded >20% from best.
        If so, halve lambda_sem for next 5 epochs.
        """
        if val_cd < self.best_val_cd:
            self.best_val_cd = val_cd
        elif val_cd > self.best_val_cd * 1.2:
            self.safety_halve_until = epoch + 5

    def compute_f_score(self, pred, gt, threshold=None):
        """Compute F-Score at a given distance threshold."""
        if threshold is None:
            threshold = self.f_score_threshold

        dist1 = torch.cdist(pred, gt, p=2)  # B, N, M
        min_dist_pred, _ = dist1.min(dim=2)  # B, N
        min_dist_gt, _ = dist1.min(dim=1)    # B, M

        precision = (min_dist_pred < threshold).float().mean(dim=1)  # B
        recall = (min_dist_gt < threshold).float().mean(dim=1)       # B

        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        return f_score.mean()

    def forward(self, ret, gt, gt_sem=None, epoch=0, factor=None):
        """
        Compute multi-task loss.

        Args:
            ret: dict from SemanticAdaPoinTr forward
            gt: (B, M, 3) ground truth complete point cloud
            gt_sem: (B, M) ground truth semantic labels (optional)
            epoch: current epoch
            factor: points per query token (for denoise target)

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses (detached floats)
        """
        pred_coarse = ret['pred_coarse']
        pred_fine = ret['pred_fine']

        # Geometric losses
        loss_coarse = self.chamfer(pred_coarse, gt)
        loss_fine = self.chamfer(pred_fine, gt)
        loss_geo = self.alpha * loss_coarse + self.gamma * loss_fine

        loss_dict = {
            'cd_coarse': loss_coarse.item(),
            'cd_fine': loss_fine.item(),
            'geo': loss_geo.item(),
        }

        total_loss = loss_geo

        # Denoise loss (training only)
        if 'denoised_fine' in ret and 'denoised_coarse' in ret:
            denoise_length = ret.get('denoise_length', 0)
            if denoise_length > 0 and factor is not None:
                denoised_coarse = ret['denoised_coarse']
                denoised_fine = ret['denoised_fine']
                idx = knn_point(factor, gt, denoised_coarse)
                denoised_target = index_points(gt, idx)
                denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
                loss_denoise = self.chamfer(denoised_fine, denoised_target) * self.beta
                total_loss = total_loss + loss_denoise
                loss_dict['denoise'] = loss_denoise.item()

        # Semantic loss
        lambda_sem = self.get_lambda_sem(epoch)
        loss_dict['lambda_sem'] = lambda_sem

        if lambda_sem > 0 and gt_sem is not None and 'sem_logits' in ret:
            sem_logits = ret['sem_logits']  # B, N_pred, C
            B, N_pred, C = sem_logits.shape

            # Transfer GT semantic labels to predicted points via nearest neighbor
            with torch.no_grad():
                dist = torch.cdist(pred_fine, gt)  # B, N_pred, M
                nn_idx = dist.argmin(dim=2)  # B, N_pred
                pred_sem_gt = torch.gather(gt_sem, 1, nn_idx)  # B, N_pred

            loss_sem_ce = F.cross_entropy(
                sem_logits.reshape(-1, C),
                pred_sem_gt.reshape(-1).long(),
                ignore_index=-1,
            )
            loss_sem = lambda_sem * loss_sem_ce
            total_loss = total_loss + loss_sem

            loss_dict['sem_ce'] = loss_sem_ce.item()
            loss_dict['sem_total'] = loss_sem.item()

            # Semantic accuracy
            with torch.no_grad():
                pred_labels = sem_logits.argmax(dim=-1)
                valid = pred_sem_gt >= 0
                if valid.any():
                    acc = (pred_labels[valid] == pred_sem_gt[valid]).float().mean()
                    loss_dict['sem_acc'] = acc.item()
        else:
            loss_dict['sem_total'] = 0.0

        # F-Score
        with torch.no_grad():
            f_score = self.compute_f_score(pred_fine, gt)
            loss_dict['f_score'] = f_score.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
