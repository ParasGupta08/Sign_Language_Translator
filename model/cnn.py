# =============================================================================
# model/cnn.py — Lightweight CNN for spatial feature extraction
#
# Design rationale:
#   • 4 convolutional blocks with BatchNorm + ReLU + MaxPool
#   • Progressively increasing filters: 32 → 64 → 128 → 128
#   • Global Average Pooling instead of large FC layers → fewer params,
#     better generalisation under limited data
#   • Outputs a CNN_FEATURE_DIM-dimensional feature vector per frame
#   • All weights initialised from scratch (Xavier uniform)
# =============================================================================

import torch
import torch.nn as nn
from config import CNN_IN_CHANNELS, CNN_BASE_FILTERS, CNN_FEATURE_DIM


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool"""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GestureCNN(nn.Module):
    """
    Input  : (B, 1, 64, 64)   — single grayscale frame
    Output : (B, CNN_FEATURE_DIM)  — spatial feature vector
    """
    def __init__(self):
        super().__init__()
        f = CNN_BASE_FILTERS   # 32

        # ── Convolutional backbone ──
        # 64→32→16→8→8 spatial dims after pooling
        self.backbone = nn.Sequential(
            ConvBlock(CNN_IN_CHANNELS, f,       pool=True),   # 64 → 32
            ConvBlock(f,              f * 2,    pool=True),   # 32 → 16
            ConvBlock(f * 2,          f * 4,    pool=True),   # 16 →  8
            ConvBlock(f * 4,          f * 4,    pool=False),  #  8 →  8  (no pool)
        )

        # ── Global Average Pooling + projection ──
        self.gap  = nn.AdaptiveAvgPool2d(1)           # (B, f*4, 1, 1)
        self.proj = nn.Sequential(
            nn.Flatten(),                             # (B, f*4)
            nn.Linear(f * 4, CNN_FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = self.proj(x)
        return x                                      # (B, CNN_FEATURE_DIM)


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    model = GestureCNN()
    dummy = torch.zeros(4, 1, 64, 64)
    out   = model(dummy)
    print(f"GestureCNN output shape : {out.shape}")   # expect (4, 128)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters        : {total:,}")
