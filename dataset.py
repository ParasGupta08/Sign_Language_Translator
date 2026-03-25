# =============================================================================
# dataset.py — PyTorch Dataset for processed gesture clips
# =============================================================================

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from config import PROCESSED_DIR, SEQUENCE_LEN, FRAME_SIZE, TRAIN_SPLIT, BATCH_SIZE


class GestureDataset(Dataset):
    """
    Loads processed .npy clips from data/processed/ using the manifest CSV.

    Each item:
        frames : FloatTensor (T, 1, H, W)   — normalised grayscale sequence
        label  : LongTensor  scalar
    """

    def __init__(self, manifest_path=None):
        if manifest_path is None:
            manifest_path = os.path.join(PROCESSED_DIR, "manifest.csv")

        self.samples = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["rel_path"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(PROCESSED_DIR, rel_path)

        clip = np.load(full_path)           # (T, H, W) float32

        # ── Ensure correct sequence length ──
        T = clip.shape[0]
        if T < SEQUENCE_LEN:
            pad = np.stack([clip[-1]] * (SEQUENCE_LEN - T), axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        clip = clip[:SEQUENCE_LEN]          # (SEQUENCE_LEN, H, W)

        # ── Add channel dim ──
        clip = clip[:, np.newaxis, :, :]    # (T, 1, H, W)

        frames = torch.from_numpy(clip).float()
        return frames, torch.tensor(label, dtype=torch.long)


def get_dataloaders():
    """
    Returns train_loader, val_loader with an 80/20 split.
    """
    dataset    = GestureDataset()
    n_total    = len(dataset)
    n_train    = int(n_total * TRAIN_SPLIT)
    n_val      = n_total - n_train

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"Dataset   : {n_total} clips  →  train {n_train} / val {n_val}")
    return train_loader, val_loader


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    frames, labels = next(iter(train_loader))
    print(f"Batch frames : {frames.shape}")   # (B, T, 1, H, W)
    print(f"Batch labels : {labels.shape}")   # (B,)
