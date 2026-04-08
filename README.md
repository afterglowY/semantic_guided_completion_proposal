# Semantic-Guided Point Cloud Completion (SGC)

A semantic-guided point cloud completion system for building facades, combining
**PointTransformerV3** (semantic feature extraction) with **AdaPoinTr** (adaptive
point cloud completion) in a dual-branch architecture.

## Architecture

```
Partial PC ──┬──▶ PTv3 Backbone (frozen) ──▶ Semantic Features (N×256)
             │                                        │
             │                                        ▼
             └──▶ Semantic AdaPoinTr ◄── Gated Fusion ──▶ Complete PC + Semantic Labels
```

- **Branch A — PTv3 Semantic Backbone**: Frozen PointTransformerV3 extracts per-point
  semantic embeddings and class predictions (12 facade classes).
- **Branch B — Semantic AdaPoinTr**: Modified AdaPoinTr with semantic-aware KNN
  (`0.8×spatial + 0.2×semantic`) and gated semantic feature fusion.

### 12 Semantic Classes

wall, window, door, roof, banister, equipment, sign, awning, stairs, balcony, eave, other

## Three-Phase Training Schedule

| Phase | Epoch Range | PTv3 State | λ_sem | Description |
|-------|-------------|------------|-------|-------------|
| Warmup | 0–20% | Frozen | 0.0 | Geometric-only completion |
| Semantic Intro | 20–60% | Frozen | 0→0.3 | Gradual semantic loss introduction |
| Joint Finetune | 60–100% | Last 2 layers unfrozen | 0.3 | End-to-end refinement |

**Safety valve**: If validation Chamfer Distance degrades >20% from best, λ_sem is
halved for 5 epochs to protect geometric quality.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (recommended; CPU fallback available)

## Dataset Preparation

1. Generate training data using the dataset generator:

```bash
python scripts/dataset_generator.py \
    --obj_dir /path/to/obj_files \
    --json_dir /path/to/json_annotations \
    --las_dir /path/to/las_scans \
    --output_dir data/facade_completion \
    --num_samples 16384
```

This produces `{scene_id}_synthetic.npz` and `{scene_id}_annotated.npz` files.

2. Organize data:

```
data/facade_completion/
├── train/
│   ├── scene001_synthetic.npz
│   ├── scene001_annotated.npz
│   └── ...
└── val/
    ├── scene100_synthetic.npz
    ├── scene100_annotated.npz
    └── ...
```

## Training

```bash
# Basic training
python train.py --config cfgs/sgc_default.yaml

# With pretrained weights
python train.py --config cfgs/sgc_default.yaml \
    --pretrained_ptv3 weights/ptv3_scannet.pth \
    --pretrained_adapointr weights/adapointr_pcn.pth

# Resume from checkpoint
python train.py --config cfgs/sgc_default.yaml --resume experiments/sgc_xxx/checkpoints/last.pth

# Custom settings
python train.py --config cfgs/sgc_default.yaml \
    --exp_name my_experiment \
    --gpu 0 \
    --num_workers 8 \
    --val_freq 5
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Path to YAML config |
| `--pretrained_ptv3` | None | Pretrained PTv3 weights |
| `--pretrained_adapointr` | None | Pretrained AdaPoinTr weights |
| `--resume` | None | Resume from checkpoint |
| `--gpu` | 0 | GPU device ID |
| `--val_freq` | 10 | Validation frequency (epochs) |
| `--save_freq` | 50 | Checkpoint save frequency |

## Project Structure

```
semantic_guided_completion/
├── cfgs/
│   └── sgc_default.yaml          # Default configuration
├── datasets/
│   ├── __init__.py
│   └── dataloader.py             # NPZ dataset & curriculum scheduler
├── extensions/
│   └── chamfer_dist/
│       └── __init__.py           # Pure PyTorch Chamfer Distance
├── models/
│   ├── __init__.py
│   ├── adapointr/                # AdaPoinTr backbone
│   │   ├── __init__.py
│   │   ├── adapointr.py
│   │   └── transformer_utils.py
│   ├── pointtransformerv3/       # PTv3 backbone
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── serialization/
│   │       ├── __init__.py
│   │       ├── default.py
│   │       ├── hilbert.py
│   │       └── z_order.py
│   └── sgc/                      # Semantic-guided completion
│       ├── __init__.py
│       ├── ptv3_backbone.py      # Frozen PTv3 wrapper
│       ├── adapointr_semantic.py # Semantic-enhanced AdaPoinTr
│       ├── losses.py             # Multi-task loss with safety valve
│       └── semantic_guided_completion.py  # Top-level model
├── scripts/
│   └── dataset_generator.py      # Training data generator
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── misc.py                   # FPS, jitter, etc.
├── train.py                      # Training script
├── requirements.txt
└── README.md
```

## Loss Function

```
L_total = λ_geo × (α×CD_coarse + γ×CD_fine) + λ_sem × CE_semantic + β × CD_denoise
```

- **Chamfer Distance L1** for geometric reconstruction (coarse + fine)
- **Cross-Entropy** for semantic segmentation on predicted points
- **Denoise loss** from AdaPoinTr's adaptive denoising mechanism
- **F-Score** tracked as evaluation metric

## Citation

This project builds upon:
- [PointTransformerV3](https://arxiv.org/abs/2312.10035) (Wu et al., 2024)
- [AdaPoinTr](https://arxiv.org/abs/2301.04545) (Yu et al., 2023)

## License

MIT
