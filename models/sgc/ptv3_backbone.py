"""
PTv3 Semantic Backbone Wrapper

Wraps PointTransformerV3 as a frozen semantic feature extractor.
Outputs per-point semantic embeddings (N x sem_feat_dim) and
semantic class predictions (N x num_classes).

The backbone is frozen during initial training phases, with optional
late-stage partial unfreezing of the last few decoder layers.
"""

import copy
import torch
import torch.nn as nn

from models.pointtransformerv3 import PointTransformerV3, Point


class PTv3SemanticBackbone(nn.Module):
    """
    Frozen PTv3 backbone for extracting semantic features from point clouds.

    Architecture:
        Input: incomplete point cloud (N x 3) + normals (N x 3) → N x 6
        PTv3 Encoder-Decoder → N x dec_channels[0] feature map
        Semantic Head → N x num_classes predictions
        Feature Projection → N x sem_feat_dim embeddings

    The backbone is frozen by default. Call `unfreeze_last_layers(n)` to
    enable fine-tuning of the last n decoder layers during late training.
    """

    def __init__(
        self,
        num_classes=12,
        sem_feat_dim=256,
        in_channels=6,
        grid_size=0.05,
        # PTv3 config
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        drop_path=0.3,
        enable_flash=True,
        frozen=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sem_feat_dim = sem_feat_dim
        self.grid_size = grid_size
        self.frozen = frozen

        # PTv3 backbone (segmentation mode: cls_mode=False)
        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            enable_flash=enable_flash,
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )

        out_channels = dec_channels[0]  # output feature dim from PTv3 decoder

        # Semantic classification head
        self.sem_head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

        # Feature projection to desired embedding dimension
        self.feat_proj = nn.Sequential(
            nn.Linear(out_channels, sem_feat_dim),
            nn.LayerNorm(sem_feat_dim),
        )

        if frozen:
            self.freeze()

    def freeze(self):
        """Freeze all PTv3 parameters."""
        for param in self.ptv3.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_last_layers(self, n_layers=2, lr_scale=0.1):
        """
        Unfreeze the last n decoder layers for fine-tuning.
        Returns parameter groups with scaled learning rate.
        """
        # Unfreeze last n decoder blocks
        dec_modules = list(self.ptv3.dec._modules.values())
        for module in dec_modules[-n_layers:]:
            for param in module.parameters():
                param.requires_grad = True
        self.frozen = False

    def get_finetune_params(self, base_lr, lr_scale=0.1):
        """Get parameter groups with different learning rates."""
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('ptv3'):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': backbone_params, 'lr': base_lr * lr_scale},
            {'params': head_params, 'lr': base_lr},
        ]

    def forward(self, coord, feat, batch=None, offset=None):
        """
        Args:
            coord: (N, 3) point coordinates
            feat: (N, C) point features (e.g., XYZ + normals = 6 channels)
            batch: (N,) batch indices
            offset: cumulative point counts per sample

        Returns:
            sem_feat: (N, sem_feat_dim) semantic feature embeddings
            sem_pred: (N, num_classes) semantic class predictions
        """
        data_dict = dict(
            coord=coord,
            feat=feat,
            grid_size=self.grid_size,
        )
        if batch is not None:
            data_dict['batch'] = batch
        if offset is not None:
            data_dict['offset'] = offset

        # Forward through PTv3
        if self.frozen:
            with torch.no_grad():
                point = self.ptv3(data_dict)
        else:
            point = self.ptv3(data_dict)

        ptv3_feat = point.feat  # (N, dec_channels[0])

        # Semantic predictions
        sem_pred = self.sem_head(ptv3_feat)

        # Semantic embeddings (soft features, not hard labels)
        sem_feat = self.feat_proj(ptv3_feat)

        return sem_feat, sem_pred
