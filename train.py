# =============================================================================
# train.py — Full training pipeline (CNN + GRU end-to-end)
#
# Usage:
#   python train.py
#
# Outputs:
#   checkpoints/best_model.pth   ← saved whenever val accuracy improves
#   checkpoints/last_model.pth   ← always saved at end of each epoch
# =============================================================================

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, MODEL_SAVE_DIR,
    BEST_MODEL, CNN_FEATURE_DIM, SEQUENCE_LEN
)
from model.cnn      import GestureCNN
from model.temporal import TemporalGRU
from dataset        import get_dataloaders


# ─────────────────────────────────────────
# Combined forward pass helper
# ─────────────────────────────────────────

def forward_pass(cnn, gru, frames, device):
    """
    frames : (B, T, 1, H, W)
    Returns logits : (B, NUM_CLASSES)
    """
    B, T, C, H, W = frames.shape
    frames = frames.to(device)

    # Reshape to process all frames through CNN simultaneously
    frames_flat = frames.view(B * T, C, H, W)          # (B*T, 1, H, W)
    features    = cnn(frames_flat)                       # (B*T, CNN_FEATURE_DIM)
    features    = features.view(B, T, CNN_FEATURE_DIM)  # (B, T, F)

    logits      = gru(features)                          # (B, NUM_CLASSES)
    return logits


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── Models ──
    cnn = GestureCNN().to(device)
    gru = TemporalGRU().to(device)

    # ── Data ──
    train_loader, val_loader = get_dataloaders()

    # ── Optimiser & scheduler ──
    params    = list(cnn.parameters()) + list(gru.parameters())
    optimiser = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=LR_STEP_SIZE,
                                          gamma=LR_GAMMA)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── History ──
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_acc = 0.0

    print("\n" + "="*55)
    print("  TRAINING")
    print("="*55)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ────── TRAIN ──────
        cnn.train(); gru.train()
        running_loss, correct, total = 0.0, 0, 0

        for frames, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch:03d}/{EPOCHS} [train]",
                                   leave=False):
            labels = labels.to(device)
            optimiser.zero_grad()

            logits = forward_pass(cnn, gru, frames, device)
            loss   = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(params, max_norm=5.0)  # gradient clipping
            optimiser.step()

            running_loss += loss.item() * labels.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ────── VALIDATE ──────
        cnn.eval(); gru.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for frames, labels in val_loader:
                labels = labels.to(device)
                logits = forward_pass(cnn, gru, frames, device)
                loss   = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                preds         = logits.argmax(dim=1)
                val_correct  += (preds == labels).sum().item()
                val_total    += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        scheduler.step()

        # ── Log ──
        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:03d}/{EPOCHS} | "
              f"T-loss {train_loss:.4f}  T-acc {train_acc:.3f} | "
              f"V-loss {val_loss:.4f}  V-acc {val_acc:.3f} | "
              f"LR {scheduler.get_last_lr()[0]:.5f} | {elapsed:.1f}s")

        # ── Save best ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "cnn_state": cnn.state_dict(),
                "gru_state": gru.state_dict(),
                "epoch":     epoch,
                "val_acc":   val_acc,
            }, BEST_MODEL)
            print(f"    ✓ Best model saved (val_acc={val_acc:.3f})")

        # ── Save last ──
        torch.save({
            "cnn_state": cnn.state_dict(),
            "gru_state": gru.state_dict(),
        }, os.path.join(MODEL_SAVE_DIR, "last_model.pth"))

    # ── Plot training curves ──
    _plot_history(history)
    print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.3f}")


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join("checkpoints", "training_curves.png"), dpi=120)
    print("  Training curves saved → checkpoints/training_curves.png")


if __name__ == "__main__":
    train()
