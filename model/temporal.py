# =============================================================================
# model/temporal.py — GRU-based temporal classifier
#
# Design rationale:
#   • Receives a sequence of CNN feature vectors (one per frame)
#   • A 2-layer GRU captures short- and medium-range temporal dynamics
#   • The final hidden state (last timestep) is used for classification
#   • Feature differencing is prepended to help the GRU focus on MOTION
#     rather than static appearance
# =============================================================================

import torch
import torch.nn as nn
from config import CNN_FEATURE_DIM, GRU_HIDDEN_DIM, GRU_LAYERS, DROPOUT, NUM_CLASSES


class TemporalGRU(nn.Module):
    """
    Input  : (B, T, CNN_FEATURE_DIM)  — sequence of CNN features
    Output : (B, NUM_CLASSES)          — class logits

    Motion encoding:
        We concatenate [f_t ; f_t - f_{t-1}] before feeding the GRU.
        The difference signal explicitly encodes frame-to-frame motion,
        making temporal dynamics easier to learn from limited data.
        This doubles the GRU input size to 2 * CNN_FEATURE_DIM.
    """
    def __init__(self):
        super().__init__()

        gru_input_dim = CNN_FEATURE_DIM * 2   # features + frame difference

        self.gru = nn.GRU(
            input_size  = gru_input_dim,
            hidden_size = GRU_HIDDEN_DIM,
            num_layers  = GRU_LAYERS,
            batch_first = True,
            dropout     = DROPOUT if GRU_LAYERS > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(GRU_HIDDEN_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)     # orthogonal init for recurrent weights
            elif "bias" in name:
                nn.init.zeros_(param.data)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _add_motion_signal(self, x):
        """
        x : (B, T, F)
        returns (B, T, 2F) where the extra F dims are frame differences.
        First-frame difference is set to zero.
        """
        diff          = torch.zeros_like(x)
        diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]   # f_t - f_{t-1}
        return torch.cat([x, diff], dim=-1)              # (B, T, 2F)

    def forward(self, x):
        # x : (B, T, CNN_FEATURE_DIM)
        x, _     = self.gru(self._add_motion_signal(x))  # (B, T, hidden)
        x        = x[:, -1, :]                            # last timestep → (B, hidden)
        logits   = self.classifier(x)                     # (B, NUM_CLASSES)
        return logits


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import SEQUENCE_LEN
    model  = TemporalGRU()
    dummy  = torch.zeros(4, SEQUENCE_LEN, CNN_FEATURE_DIM)
    logits = model(dummy)
    print(f"TemporalGRU output shape : {logits.shape}")   # expect (4, NUM_CLASSES)
    total  = sum(p.numel() for p in model.parameters())
    print(f"Total parameters         : {total:,}")
