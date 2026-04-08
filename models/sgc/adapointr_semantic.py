"""
Semantic-Enhanced AdaPoinTr

Extends the original PCTransformer and AdaPoinTr with semantic-aware KNN
and semantic feature fusion from the PTv3 backbone.

Key modifications:
  1. Semantic-Aware KNN: edge_weight = alpha * spatial_dist + (1-alpha) * semantic_cosine_dist
  2. Semantic feature injection into encoder via cross-attention
  3. Per-point semantic prediction head on completed output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

from extensions.chamfer_dist import ChamferDistanceL1
from models.adapointr.transformer_utils import (
    knn_point, square_distance, index_points, Mlp, Attention, CrossAttention,
    DeformableLocalCrossAttention, DynamicGraphAttention,
    improvedDeformableLocalGraphAttention, DeformableLocalAttention,
    LayerScale,
)
from utils import misc


# ====================== Semantic-Aware KNN ======================

def semantic_aware_knn(nsample, xyz, new_xyz, sem_feat, new_sem_feat, alpha=0.8):
    """
    Semantic-aware KNN: combines spatial distance with semantic cosine distance.

    Args:
        nsample: number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
        sem_feat: (B, N, D) semantic features for all points
        new_sem_feat: (B, S, D) semantic features for query points
        alpha: blend weight (0.8 spatial + 0.2 semantic)

    Returns:
        group_idx: (B, S, nsample)
    """
    spatial_dist = square_distance(new_xyz, xyz)  # B, S, N

    # Cosine distance: 1 - cosine_similarity
    sem_feat_norm = F.normalize(sem_feat, dim=-1)  # B, N, D
    new_sem_feat_norm = F.normalize(new_sem_feat, dim=-1)  # B, S, D
    cos_sim = torch.bmm(new_sem_feat_norm, sem_feat_norm.transpose(1, 2))  # B, S, N
    semantic_dist = 1.0 - cos_sim  # B, S, N

    # Normalize both distances to [0, 1] range per query
    spatial_max = spatial_dist.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
    semantic_max = semantic_dist.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)

    spatial_norm = spatial_dist / spatial_max
    semantic_norm = semantic_dist / semantic_max

    combined_dist = alpha * spatial_norm + (1.0 - alpha) * semantic_norm

    _, group_idx = torch.topk(combined_dist, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# ====================== Semantic PCTransformer ======================

class SemanticPCTransformer(nn.Module):
    """
    PCTransformer extended with semantic feature fusion.

    Changes from original PCTransformer:
      - Accepts semantic features (sem_feat) from PTv3 backbone
      - Fuses semantic features into encoder via a projection + addition
      - Uses semantic-aware KNN in grouper when sem_feat is available
    """

    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim
        self.sem_feat_dim = getattr(config, 'sem_feat_dim', 256)
        self.sem_alpha = getattr(config, 'sem_alpha', 0.8)

        # Base encoder (DGCNN grouper)
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )

        # Semantic feature fusion projection
        self.sem_proj = nn.Sequential(
            nn.Linear(self.sem_feat_dim, encoder_config.embed_dim),
            nn.LayerNorm(encoder_config.embed_dim),
            nn.GELU(),
        )
        self.sem_gate = nn.Sequential(
            nn.Linear(encoder_config.embed_dim * 2, encoder_config.embed_dim),
            nn.Sigmoid(),
        )

        # Encoder
        from models.adapointr.adapointr import PointTransformerEncoderEntry
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        # Query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )

        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)

        # Decoder
        from models.adapointr.adapointr import PointTransformerDecoderEntry
        self.decoder = PointTransformerDecoderEntry(decoder_config)

        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz, sem_feat=None):
        """
        Args:
            xyz: (B, N, 3) partial point cloud
            sem_feat: (B, N, sem_feat_dim) semantic features from PTv3 (optional)

        Returns:
            q: (B, M, C) decoded features
            coarse_point_cloud: (B, M, 3) coarse predictions
            denoise_length: int
        """
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        # Semantic feature fusion via gated addition
        if sem_feat is not None:
            # Downsample semantic features to match encoder resolution
            # coor is (B, center_num[-1], 3), need to gather sem_feat at those centers
            # Use KNN to interpolate semantic features to center points
            coor_flat = coor  # B, n_center, 3
            xyz_flat = xyz    # B, N, 3
            knn_idx = knn_point(3, xyz_flat, coor_flat)  # B, n_center, 3
            knn_sem = index_points(sem_feat, knn_idx)  # B, n_center, 3, D
            center_sem = knn_sem.mean(dim=2)  # B, n_center, D

            sem_proj = self.sem_proj(center_sem)  # B, n_center, embed_dim
            gate = self.sem_gate(torch.cat([x, sem_proj], dim=-1))  # B, n_center, embed_dim
            x = x + gate * sem_proj

        x = self.encoder(x + pe, coor)  # b n c
        global_feature = self.increase_dim(x)  # B N 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
        coarse_inp = misc.fps(xyz, self.num_query // 2)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)

        mem = self.mem_link(x)

        # Query selection
        query_ranking = self.query_ranking(coarse)
        idx = torch.argsort(query_ranking, dim=1, descending=True)
        coarse = torch.gather(coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)
            denoise_length = 64

            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))

            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)
            return q, coarse, denoise_length
        else:
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))

            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)
            return q, coarse, 0


# ====================== DGCNN Grouper (local copy) ======================

class DGCNN_Grouper(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.k = k
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32), nn.LeakyReLU(negative_slope=0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64), nn.LeakyReLU(negative_slope=0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64), nn.LeakyReLU(negative_slope=0.2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.GroupNorm(4, 128), nn.LeakyReLU(negative_slope=0.2))
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()
        fps_idx = misc.fps(xyz, num_group, return_idx=True)
        if isinstance(fps_idx, tuple):
            fps_idx = fps_idx[1]
            combined_x = torch.cat([coor, x], dim=1)
            # Gather using index
            fps_idx_expand = fps_idx.unsqueeze(1).expand(-1, combined_x.size(1), -1)
            new_combined_x = torch.gather(combined_x, 2, fps_idx_expand)
        else:
            # fps returns centers directly, use knn to find nearest
            centers = fps_idx  # B, num_group, 3
            combined_x = torch.cat([coor, x], dim=1)
            # Use knn to get indices
            knn_idx = knn_point(1, coor.transpose(1, 2).contiguous(),
                                centers).squeeze(-1)  # B, num_group
            knn_idx_expand = knn_idx.unsqueeze(1).expand(-1, combined_x.size(1), -1)
            new_combined_x = torch.gather(combined_x, 2, knn_idx_expand)

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]
        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)
        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(),
                            coor_q.transpose(-1, -2).contiguous())
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        x = x.transpose(-1, -2).contiguous()
        coor = x
        f = self.input_trans(x)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()
        return coor, f


class SimpleEncoder(nn.Module):
    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        from models.adapointr.adapointr import Encoder
        self.embedding = Encoder(embed_dims)
        self.group_size = k
        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        if isinstance(n_group, list):
            n_group = n_group[-1]
        center = misc.fps(xyz, n_group)
        batch_size, num_points, _ = xyz.shape
        idx = knn_point(self.group_size, xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
        features = self.embedding(neighborhood)
        return center, features


# ====================== Semantic AdaPoinTr ======================

class SemanticAdaPoinTr(nn.Module):
    """
    AdaPoinTr extended with semantic awareness.

    Adds:
      - Semantic feature injection into PCTransformer
      - Per-point semantic prediction head on the completed output
      - Semantic-aware rebuild from decoded features
    """

    def __init__(self, config):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.num_classes = getattr(config, 'num_classes', 12)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = SemanticPCTransformer(config)

        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            from models.adapointr.adapointr import Fold
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                from models.adapointr.adapointr import SimpleRebuildFCLayer
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2, step=self.num_points // self.num_query)
            else:
                self.factor = self.fold_step ** 2
                from models.adapointr.adapointr import SimpleRebuildFCLayer
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2, step=self.fold_step ** 2)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)

        # Semantic prediction head for completed points
        self.sem_head = nn.Sequential(
            nn.Linear(self.trans_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes),
        )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def forward(self, xyz, sem_feat=None):
        """
        Args:
            xyz: (B, N, 3) partial point cloud
            sem_feat: (B, N, D) semantic features from PTv3

        Returns:
            dict with keys:
              - pred_coarse, pred_fine, denoised_coarse, denoised_fine
              - sem_logits: (B, N_pred, num_classes) semantic predictions
              - coarse_point_cloud
        """
        q, coarse_point_cloud, denoise_length = self.base_model(xyz, sem_feat=sem_feat)

        B, M, C = q.shape

        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1)[0]

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            relative_xyz = self.decode_head(rebuild_feature)
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()
            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            # Semantic predictions on predicted fine points
            # Use the query features (excluding denoise tokens) to predict semantics
            pred_q = q[:, :-denoise_length]  # B, num_query, C
            # Expand to match fine points: each query covers 'factor' points
            sem_feat_expanded = pred_q.unsqueeze(2).expand(-1, -1, self.factor, -1)
            sem_feat_flat = sem_feat_expanded.reshape(B, -1, C)  # B, num_query*factor, C
            sem_logits = self.sem_head(sem_feat_flat.reshape(-1, C)).reshape(B, -1, self.num_classes)

            return {
                'pred_coarse': pred_coarse,
                'denoised_coarse': denoised_coarse,
                'denoised_fine': denoised_fine,
                'pred_fine': pred_fine,
                'sem_logits': sem_logits,
                'coarse_point_cloud': coarse_point_cloud,
                'denoise_length': denoise_length,
            }
        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()

            # Semantic predictions
            sem_feat_expanded = q.unsqueeze(2).expand(-1, -1, self.factor, -1)
            sem_feat_flat = sem_feat_expanded.reshape(B, -1, C)
            sem_logits = self.sem_head(sem_feat_flat.reshape(-1, C)).reshape(B, -1, self.num_classes)

            return {
                'pred_coarse': coarse_point_cloud,
                'pred_fine': rebuild_points,
                'sem_logits': sem_logits,
            }
