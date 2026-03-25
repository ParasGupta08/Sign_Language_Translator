# =============================================================================
# preprocess.py — Build the processed dataset from raw clips
#
# What it does:
#   1. Loads every .npy clip from data/raw/<SIGN>/
#   2. Normalises pixel values to [0, 1]
#   3. Applies augmentation (flips, brightness, rotation, noise)
#   4. Saves augmented clips to data/processed/
#   5. Produces a manifest CSV: data/processed/manifest.csv
#
# Usage:
#   python preprocess.py
# =============================================================================

import os
import numpy as np
import csv
import random
from PIL import Image, ImageEnhance
from config import (
    SIGNS, DATA_DIR, PROCESSED_DIR, FRAME_SIZE,
    FRAMES_PER_CLIP,
    AUG_HFLIP_PROB, AUG_BRIGHTNESS_RANGE,
    AUG_ROTATION_DEGREES, AUG_NOISE_STD,
    LABEL_MAP
)


# ─────────────────────────────────────────
# Per-frame augmentation helpers
# ─────────────────────────────────────────

def aug_hflip(frames):
    """Horizontal flip — applied consistently across the entire sequence."""
    if random.random() < AUG_HFLIP_PROB:
        return [np.fliplr(f) for f in frames]
    return frames


def aug_brightness(frames):
    """Random brightness shift — same factor for whole clip."""
    factor = random.uniform(*AUG_BRIGHTNESS_RANGE)
    out = []
    for f in frames:
        img = Image.fromarray((f * 255).astype(np.uint8), mode='L')
        img = ImageEnhance.Brightness(img).enhance(factor)
        out.append(np.array(img).astype(np.float32) / 255.0)
    return out


def aug_rotation(frames):
    """Random rotation — same angle for whole clip."""
    angle = random.uniform(-AUG_ROTATION_DEGREES, AUG_ROTATION_DEGREES)
    out = []
    for f in frames:
        img = Image.fromarray((f * 255).astype(np.uint8), mode='L')
        img = img.rotate(angle, resample=Image.BILINEAR)
        out.append(np.array(img).astype(np.float32) / 255.0)
    return out


def aug_noise(frames):
    """Additive Gaussian noise."""
    out = []
    for f in frames:
        noise = np.random.normal(0, AUG_NOISE_STD, f.shape).astype(np.float32)
        out.append(np.clip(f + noise, 0.0, 1.0))
    return out


def augment_clip(frames, n_augmented=4):
    """
    Given a list of float32 frames (H, W) in [0,1],
    return the original + n_augmented variants.
    """
    results = [frames]                          # always keep original
    augmenters = [aug_hflip, aug_brightness, aug_rotation, aug_noise]
    for _ in range(n_augmented):
        aug = frames.copy()
        random.shuffle(augmenters)
        for fn in augmenters[:random.randint(1, 3)]:   # apply 1–3 augmentations
            aug = fn(aug)
        results.append(aug)
    return results


# ─────────────────────────────────────────
# Main preprocessing pipeline
# ─────────────────────────────────────────

def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    manifest_rows = []   # (rel_path, label_idx)

    for sign in SIGNS:
        raw_sign_dir  = os.path.join(DATA_DIR, sign.replace(" ", "_"))
        proc_sign_dir = os.path.join(PROCESSED_DIR, sign.replace(" ", "_"))
        os.makedirs(proc_sign_dir, exist_ok=True)

        label_idx = LABEL_MAP[sign]
        clip_files = sorted([
            f for f in os.listdir(raw_sign_dir) if f.endswith(".npy")
        ]) if os.path.isdir(raw_sign_dir) else []

        if not clip_files:
            print(f"  [WARN] No clips found for sign '{sign}'. Skipping.")
            continue

        print(f"\n  Processing [{sign}] — {len(clip_files)} raw clips …")
        save_idx = 0

        for clip_file in clip_files:
            raw_path = os.path.join(raw_sign_dir, clip_file)
            clip     = np.load(raw_path)            # (T, H, W) uint8

            # ── 1. Normalise ──
            frames = [clip[t].astype(np.float32) / 255.0
                      for t in range(clip.shape[0])]

            # ── 2. Ensure correct length ──
            if len(frames) < FRAMES_PER_CLIP:
                # pad by repeating last frame
                while len(frames) < FRAMES_PER_CLIP:
                    frames.append(frames[-1])
            frames = frames[:FRAMES_PER_CLIP]

            # ── 3. Augment ──
            variants = augment_clip(frames, n_augmented=4)

            # ── 4. Save ──
            for variant in variants:
                arr       = np.stack(variant, axis=0)   # (T, H, W) float32
                save_name = f"clip_{save_idx:05d}.npy"
                save_path = os.path.join(proc_sign_dir, save_name)
                np.save(save_path, arr)
                manifest_rows.append((
                    os.path.join(sign.replace(" ", "_"), save_name),
                    label_idx
                ))
                save_idx += 1

        print(f"    → {save_idx} clips saved (including augmented).")

    # ── Write manifest ──
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_path", "label"])
        writer.writerows(manifest_rows)

    print(f"\n✓ Preprocessing complete. Manifest: {manifest_path}")
    print(f"  Total clips: {len(manifest_rows)}")


if __name__ == "__main__":
    preprocess()
