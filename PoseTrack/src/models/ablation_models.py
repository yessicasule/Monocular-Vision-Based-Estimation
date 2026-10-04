"""
ablation_models.py — Phase 10: Ablation Study Model Variants

5 configurations to quantify what each module contributes:

  Config 1 (Full):          Transformer + CrossAttn + BiLSTM + MCDropout + GAN
  Config 2 (No Transformer): Linear embedding only → BiLSTM → Head
  Config 3 (No CrossAttn):  Transformer → BiLSTM (skip cross-attention)
  Config 4 (No Bayesian):   Standard deterministic forward pass (dropout=0)
  Config 5 (BiLSTM only):   Identical to baseline but trained on all 3 frameworks
"""

import torch
import torch.nn as nn
from fusion_network import (
    PositionalEncoding, TransformerEncoderBlock, CrossAttentionFusion,
    FusionDataset, train_fusion_model, evaluate_fusion_model
)


# ─────────────────────────────────────────────────────────────────────────────
# Config 2 — No Transformer (Linear embed → BiLSTM only)
# ─────────────────────────────────────────────────────────────────────────────

class FusionNoTransformer(nn.Module):
    """Removes the Transformer blocks and Cross-Attention entirely."""
    def __init__(self, input_dim=12, d_model=64, lstm_hidden=128,
                 lstm_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.bilstm(emb)
        return self.head(lstm_out)


# ─────────────────────────────────────────────────────────────────────────────
# Config 3 — No Cross-Attention (Transformer → BiLSTM, skip CrossAttn)
# ─────────────────────────────────────────────────────────────────────────────

class FusionNoCrossAttn(nn.Module):
    """Has Transformer but skips the Cross-Attention fusion layer."""
    def __init__(self, input_dim=12, d_model=64, n_heads=8,
                 n_transformer_blocks=2, d_ff=256,
                 lstm_hidden=128, lstm_layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.embedding   = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
              for _ in range(n_transformer_blocks)]
        )
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        emb = self.pos_enc(self.embedding(x))
        enc = self.transformer(emb)
        lstm_out, _ = self.bilstm(enc)          # directly into LSTM, no CrossAttn
        return self.head(lstm_out)


# ─────────────────────────────────────────────────────────────────────────────
# Config 4 — No Bayesian (dropout=0 → deterministic)
# ─────────────────────────────────────────────────────────────────────────────

class FusionNoBayesian(nn.Module):
    """Full architecture but dropout=0 → no MC Dropout uncertainty."""
    def __init__(self, input_dim=12, d_model=64, n_heads=8,
                 n_transformer_blocks=2, d_ff=256,
                 lstm_hidden=128, lstm_layers=2, output_dim=4):
        super().__init__()
        dropout = 0.0   # deterministic
        self.embedding   = nn.Sequential(nn.Linear(input_dim, d_model), nn.ReLU())
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
              for _ in range(n_transformer_blocks)]
        )
        self.cross_attn  = CrossAttentionFusion(d_model, n_heads, dropout)
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        raw_12 = x.clone()
        emb    = self.pos_enc(self.embedding(x))
        enc    = self.transformer(emb)
        fused  = self.cross_attn(enc, raw_12)
        lstm_out, _ = self.bilstm(fused)
        return self.head(lstm_out)


# ─────────────────────────────────────────────────────────────────────────────
# Config 5 — BiLSTM only on all 12 features (no Transformer, no CrossAttn)
# ─────────────────────────────────────────────────────────────────────────────

class FusionBiLSTMOnly(nn.Module):
    """Simplest fusion: all 12 features → BiLSTM → Head. No attention."""
    def __init__(self, input_dim=12, hidden=256, layers=2,
                 output_dim=4, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        return self.head(lstm_out)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner — trains + evaluates all 5 configs
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "A1_Full":           None,            # passed in (already trained)
    "A2_No_Transformer": FusionNoTransformer,
    "A3_No_CrossAttn":   FusionNoCrossAttn,
    "A4_No_Bayesian":    FusionNoBayesian,
    "A5_BiLSTM_Only":    FusionBiLSTMOnly,
}


def run_ablation(
    full_model,           # already-trained DeepFusionPoseModel
    train_loader,
    val_loader,
    eval_loader,
    output_dir,
    device="cpu",
    epochs=40,            # fewer epochs — just for comparison
):
    """
    Trains ablation variants (A2–A5) at reduced epochs and evaluates all
    5 configs including the already-trained full model (A1).

    Returns: dict {config_name: metrics_dict}
    """
    import json
    from pathlib import Path
    import numpy as np

    results = {}

    # A1: evaluate existing full model (no retraining)
    print("\n=== A1_Full (pre-trained, load + eval only) ===")
    full_model.eval()
    full_metrics = evaluate_fusion_model(full_model, eval_loader, device=device, n_mc=30)
    results["A1_Full"] = full_metrics
    avg = full_metrics["average"]
    print(f"  MAE={avg['MAE']:.3f}  RMSE={avg['RMSE']:.3f}  Jitter={avg['Jitter']:.3f}")

    # A2–A5: train ablation variants
    variant_classes = {
        "A2_No_Transformer": FusionNoTransformer,
        "A3_No_CrossAttn":   FusionNoCrossAttn,
        "A4_No_Bayesian":    FusionNoBayesian,
        "A5_BiLSTM_Only":    FusionBiLSTMOnly,
    }

    for name, cls in variant_classes.items():
        print(f"\n=== {name} ({epochs} epochs) ===")
        model   = cls()
        ckpt    = str(Path(output_dir) / f"ablation_{name}.pt")
        history = train_fusion_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=1e-4, device=device,
            checkpoint_path=ckpt,
        )
        model.load_state_dict(torch.load(ckpt, map_location=device))

        # Simple MAE/RMSE eval (no MC for non-Bayesian variants)
        model.eval()
        all_preds, all_targets = [], []
        import numpy as np
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device)
                all_preds.append(model(xb).cpu().numpy())
                all_targets.append(yb.numpy())
        preds   = np.concatenate(all_preds).reshape(-1, 4)
        targets = np.concatenate(all_targets).reshape(-1, 4)
        mae  = np.mean(np.abs(preds - targets), axis=0)
        rmse = np.sqrt(np.mean((preds - targets)**2, axis=0))

        from metrics import compute_jitter
        jitter = np.array([compute_jitter(preds[:, j].tolist()) for j in range(4)])
        joint_names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]
        metrics = {}
        for i, jn in enumerate(joint_names):
            metrics[jn] = {"MAE": float(mae[i]), "RMSE": float(rmse[i]), "Jitter": float(jitter[i])}
        metrics["average"] = {
            "MAE": float(mae.mean()), "RMSE": float(rmse.mean()), "Jitter": float(jitter.mean())
        }
        results[name] = metrics
        avg = metrics["average"]
        print(f"  MAE={avg['MAE']:.3f}  RMSE={avg['RMSE']:.3f}  Jitter={avg['Jitter']:.3f}")

    # Save combined results
    out_path = Path(output_dir) / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {out_path}")
    return results
