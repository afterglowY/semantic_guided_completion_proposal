"""
DataLoader for Semantic-Guided Point Cloud Completion

Reads NPZ files produced by dataset_generator.py:
  - {scene_id}_synthetic.npz: points(N,3), semantic_id(N,), colors(N,3), mesh_area
  - {scene_id}_annotated.npz: points(N,3), semantic_id(N,), confidence_dist(N,), colors(N,3)

Training pairs:
  - Incomplete (partial): annotated scan OR synthetically occluded version of synthetic
  - Complete (GT): synthetic point cloud (full surface sampling)
  - Semantic labels: from synthetic (face-level ground truth)

Supports 3 training phases:
  Phase 1 (warmup):       geometry only — no semantic labels returned
  Phase 2 (semantic_intro): geometry + semantic labels
  Phase 3 (joint_finetune): geometry + semantic labels (same as phase 2)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# ==================== Semantic Definition (matches dataset_generator.py) ====================

SEMANTIC_LABELS = {
    0: 'wall',      1: 'window',    2: 'door',      3: 'roof',
    4: 'banister',  5: 'equipment', 6: 'sign',      7: 'awning',
    8: 'stairs',    9: 'balcony',   10: 'eave',     11: 'other'
}
NUM_CLASSES = 12


# ==================== Data Transforms ====================

def voxel_downsample_np(points, labels=None, voxel_size=0.05):
    """Voxel grid downsampling with centroid + majority-vote labels."""
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx, inverse = np.unique(
        voxel_idx, axis=0, return_index=True, return_inverse=True
    )
    n_voxels = len(unique_idx)

    centroids = np.zeros((n_voxels, 3), dtype=np.float32)
    counts = np.zeros(n_voxels, dtype=np.int32)
    np.add.at(centroids, inverse, points)
    np.add.at(counts, inverse, 1)
    centroids /= counts[:, None].clip(min=1)

    if labels is not None:
        voxel_labels = np.zeros(n_voxels, dtype=np.int64)
        for vid in range(n_voxels):
            mask = inverse == vid
            vals, cnts = np.unique(labels[mask], return_counts=True)
            voxel_labels[vid] = vals[cnts.argmax()]
        return centroids, voxel_labels

    return centroids, None


def estimate_normals_np(points, k=30):
    """Estimate normals via local PCA (pure numpy, no open3d required)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))

    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]

    # Consistent orientation
    center = points.mean(axis=0)
    flip = ((points - center) * normals).sum(axis=1) < 0
    normals[flip] *= -1
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms.clip(min=1e-8)
    return normals.astype(np.float32)


def random_occlusion(points, labels=None, n_occluders=(2, 5), ratio=(0.1, 0.3)):
    """Simulate partial scan by removing points inside random ellipsoids."""
    n_occ = np.random.randint(n_occluders[0], n_occluders[1] + 1)
    bbox = points.max(0) - points.min(0)
    mask = np.ones(len(points), dtype=bool)

    for _ in range(n_occ):
        center = points.min(0) + np.random.rand(3) * bbox
        r = np.random.uniform(*ratio)
        radii = bbox * r * (0.5 + np.random.rand(3) * 0.5)
        inside = ((points - center) / radii.clip(min=1e-8)) ** 2
        mask &= inside.sum(axis=1) >= 1.0

    if mask.sum() < 128:
        mask[:128] = True

    result = points[mask]
    result_labels = labels[mask] if labels is not None else None
    return result, result_labels


def architectural_augment(points, labels=None):
    """Building-aware augmentation: discrete rotation, flip, jitter, dropout."""
    points = points.copy()
    if labels is not None:
        labels = labels.copy()

    # Discrete Z-rotation (0/90/180/270)
    angle = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    points = points @ rot.T

    # Random mirror (X or Y)
    if np.random.rand() > 0.5:
        ax = np.random.randint(2)
        points[:, ax] *= -1

    # Height scale
    points[:, 2] *= np.random.uniform(0.9, 1.1)

    # Jitter
    points += np.random.normal(0, 0.01, points.shape).astype(np.float32)

    # Dropout 5-15%
    rate = np.random.uniform(0.05, 0.15)
    keep = np.random.rand(len(points)) > rate
    if keep.sum() < 128:
        keep[:128] = True
    points = points[keep]
    if labels is not None:
        labels = labels[keep]

    return points, labels


def fixed_size_sample(points, n, labels=None, normals=None):
    """Resample to exactly n points."""
    if len(points) == 0:
        pts = np.zeros((n, 3), dtype=np.float32)
        lbl = np.zeros(n, dtype=np.int64) if labels is not None else None
        nrm = np.zeros((n, 3), dtype=np.float32) if normals is not None else None
        return pts, lbl, nrm

    if len(points) >= n:
        idx = np.random.choice(len(points), n, replace=False)
    else:
        idx = np.random.choice(len(points), n, replace=True)

    pts = points[idx]
    lbl = labels[idx] if labels is not None else None
    nrm = normals[idx] if normals is not None else None
    return pts, lbl, nrm


# ==================== Dataset ====================

class FacadeCompletionDataset(Dataset):
    """
    Dataset for building facade point cloud completion.

    Directory layout expected (output of dataset_generator.py):
        data_root/
            scene_001/
                scene_001_synthetic.npz
                scene_001_annotated.npz   (optional)
            scene_002/
                ...
        OR flat:
        data_root/
            scene_001_synthetic.npz
            scene_001_annotated.npz
            ...

    Each sample returns:
        partial:  (num_points, 3)  float32 — incomplete point cloud
        gt:       (num_points, 3)  float32 — complete point cloud
        normals:  (num_points, 3)  float32 — normals for partial cloud
        gt_sem:   (num_points,)    int64   — semantic labels for GT  (or None)
    """

    def __init__(
        self,
        data_root,
        split='train',
        num_points=16384,
        voxel_size=0.05,
        use_normals=True,
        augment=True,
        phase='warmup',          # 'warmup' | 'semantic_intro' | 'joint_finetune'
        use_annotated_as_partial=True,
        max_incompleteness=1.0,  # for curriculum learning
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.use_normals = use_normals
        self.augment = augment and (split == 'train')
        self.phase = phase
        self.use_annotated_as_partial = use_annotated_as_partial
        self.max_incompleteness = max_incompleteness

        self.samples = self._discover_samples()
        print(f"[FacadeCompletionDataset] {split}: {len(self.samples)} samples, "
              f"phase={phase}, augment={self.augment}")

    def _discover_samples(self):
        """Find all synthetic/annotated pairs."""
        samples = []
        root = os.path.join(self.data_root, self.split) \
            if os.path.isdir(os.path.join(self.data_root, self.split)) \
            else self.data_root

        # Find all synthetic NPZs (recursive)
        synth_files = sorted(glob.glob(os.path.join(root, '**', '*_synthetic.npz'), recursive=True))
        if not synth_files:
            synth_files = sorted(glob.glob(os.path.join(root, '*_synthetic.npz')))

        for synth_path in synth_files:
            scene_id = os.path.basename(synth_path).replace('_synthetic.npz', '')
            ann_path = synth_path.replace('_synthetic.npz', '_annotated.npz')

            sample = {
                'scene_id': scene_id,
                'synthetic': synth_path,
                'annotated': ann_path if os.path.exists(ann_path) else None,
            }
            samples.append(sample)

        return samples

    def set_phase(self, phase):
        """Switch training phase: controls whether semantic labels are returned."""
        assert phase in ('warmup', 'semantic_intro', 'joint_finetune')
        self.phase = phase

    def set_max_incompleteness(self, val):
        """For curriculum learning."""
        self.max_incompleteness = val

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- Load complete (GT) from synthetic ----
        synth = np.load(sample['synthetic'])
        gt_points = synth['points'].astype(np.float32)
        gt_labels = synth['semantic_id'].astype(np.int64)

        # ---- Load / generate partial ----
        if self.use_annotated_as_partial and sample['annotated'] is not None:
            ann = np.load(sample['annotated'])
            partial_points = ann['points'].astype(np.float32)
            # Optional: filter by confidence distance
            if 'confidence_dist' in ann:
                conf = ann['confidence_dist']
                reliable = conf < 1.0
                if reliable.sum() > 128:
                    partial_points = partial_points[reliable]
        else:
            # Generate partial from synthetic via random occlusion
            partial_points, _ = random_occlusion(gt_points, ratio=(0.15, 0.4))

        # ---- Augmentation (training only) ----
        if self.augment:
            gt_points, gt_labels = architectural_augment(gt_points, gt_labels)
            partial_points, _ = architectural_augment(partial_points)

        # ---- Voxel downsample ----
        gt_points, gt_labels = voxel_downsample_np(gt_points, gt_labels, self.voxel_size)
        partial_points, _ = voxel_downsample_np(partial_points, voxel_size=self.voxel_size)

        # ---- Normalize to local coordinates ----
        centroid = gt_points.mean(axis=0)
        scale = np.abs(gt_points - centroid).max()
        scale = max(scale, 1e-8)
        gt_points = (gt_points - centroid) / scale
        partial_points = (partial_points - centroid) / scale

        # ---- Estimate normals ----
        normals = None
        if self.use_normals and len(partial_points) >= 4:
            try:
                normals = estimate_normals_np(partial_points)
            except Exception:
                normals = np.zeros_like(partial_points)

        # ---- Fixed-size sampling ----
        partial_points, _, normals = fixed_size_sample(
            partial_points, self.num_points, normals=normals)
        gt_points, gt_labels, _ = fixed_size_sample(
            gt_points, self.num_points, labels=gt_labels)

        # ---- Build output dict ----
        result = {
            'partial': torch.from_numpy(partial_points).float(),
            'gt': torch.from_numpy(gt_points).float(),
        }

        if normals is not None:
            result['normals'] = torch.from_numpy(normals).float()

        # Semantic labels only in semantic-aware phases
        if self.phase in ('semantic_intro', 'joint_finetune') and gt_labels is not None:
            result['gt_sem'] = torch.from_numpy(gt_labels).long()

        return result


# ==================== Curriculum Sampler ====================

class CurriculumScheduler:
    """
    Manages curriculum learning across training.

    Phase 1 (0-30% epochs): Easy samples only (< 30% missing)
    Phase 2 (30-70% epochs): Gradually harder
    Phase 3 (70-100% epochs): Full data
    """

    def __init__(self, dataset, max_epochs):
        self.dataset = dataset
        self.max_epochs = max_epochs

    def step(self, epoch):
        progress = epoch / self.max_epochs

        # Curriculum: control incompleteness
        if progress < 0.3:
            self.dataset.set_max_incompleteness(0.3)
        elif progress < 0.7:
            frac = (progress - 0.3) / 0.4
            self.dataset.set_max_incompleteness(0.3 + frac * 0.4)
        else:
            self.dataset.set_max_incompleteness(1.0)

        # Phase: control semantic label availability
        if progress < 0.2:
            self.dataset.set_phase('warmup')
        elif progress < 0.6:
            self.dataset.set_phase('semantic_intro')
        else:
            self.dataset.set_phase('joint_finetune')


# ==================== Builder ====================

def build_dataloaders(config, num_workers=8):
    """
    Build train + val dataloaders from config.

    Args:
        config: AttrDict with keys: dataset.{train,val}.data_root, model.num_points, etc.
        num_workers: dataloader workers

    Returns:
        train_dataset, val_dataset, train_loader, val_loader
    """
    num_points = config.model.get('num_points', 16384)
    voxel_size = config.dataset.get('voxel_size', 0.05)
    batch_size = config.get('total_bs', 8)

    train_dataset = FacadeCompletionDataset(
        data_root=config.dataset.train.data_root,
        split='train',
        num_points=num_points,
        voxel_size=voxel_size,
        use_normals=config.model.get('use_normals', True),
        augment=True,
        phase='warmup',
    )

    val_dataset = FacadeCompletionDataset(
        data_root=config.dataset.val.data_root,
        split='val',
        num_points=num_points,
        voxel_size=voxel_size,
        use_normals=config.model.get('use_normals', True),
        augment=False,
        phase='semantic_intro',  # always evaluate with semantics
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader
