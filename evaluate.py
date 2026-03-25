# =============================================================================
# evaluate.py — Evaluate trained model on validation set
#
# Outputs:
#   • Accuracy, per-class precision/recall/F1
#   • Confusion matrix plot → checkpoints/confusion_matrix.png
#
# Usage:
#   python evaluate.py
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import BEST_MODEL, IDX_TO_SIGN, CNN_FEATURE_DIM, SIGNS
from model.cnn      import GestureCNN
from model.temporal import TemporalGRU
from dataset        import get_dataloaders


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = GestureCNN().to(device)
    gru = TemporalGRU().to(device)
    ckpt = torch.load(BEST_MODEL, map_location=device)
    cnn.load_state_dict(ckpt["cnn_state"])
    gru.load_state_dict(ckpt["gru_state"])
    cnn.eval(); gru.eval()
    print(f"✓ Loaded model from epoch {ckpt.get('epoch', '?')} "
          f"(val_acc={ckpt.get('val_acc', '?'):.3f})")

    _, val_loader = get_dataloaders()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, labels in val_loader:
            B, T, C, H, W = frames.shape
            flat     = frames.view(B * T, C, H, W).to(device)
            features = cnn(flat).view(B, T, CNN_FEATURE_DIM)
            logits   = gru(features)
            preds    = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"\n  Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)\n")

    print(classification_report(
        all_labels, all_preds,
        target_names=SIGNS, digits=3
    ))

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SIGNS, yticklabels=SIGNS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("checkpoints/confusion_matrix.png", dpi=120)
    print("\n  Confusion matrix → checkpoints/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
