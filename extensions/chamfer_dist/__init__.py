import torch
import torch.nn as nn


class ChamferDistanceL1(nn.Module):
    """Chamfer Distance L1 - pure PyTorch implementation."""
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1: (B, N, 3) predicted
            xyz2: (B, M, 3) ground truth
        Returns:
            scalar chamfer distance
        """
        dist1 = torch.cdist(xyz1, xyz2, p=2)  # B, N, M
        min1, _ = dist1.min(dim=2)  # B, N
        min2, _ = dist1.min(dim=1)  # B, M
        return (min1.mean(dim=1) + min2.mean(dim=1)).mean()


class ChamferDistanceL2(nn.Module):
    """Chamfer Distance L2 - pure PyTorch implementation."""
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2):
        dist1 = torch.cdist(xyz1, xyz2, p=2)
        min1, _ = dist1.min(dim=2)
        min2, _ = dist1.min(dim=1)
        return ((min1 ** 2).mean(dim=1) + (min2 ** 2).mean(dim=1)).mean()
