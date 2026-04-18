"""
04_train_model.py
=================
Step 4: Train a Neural Network to predict NCAA tournament game outcomes.

Architecture:
    Input (25 diff features)
    -> BatchNorm
    -> Linear(25, 64) -> ReLU -> Dropout(0.3)
    -> Linear(64, 32) -> ReLU -> Dropout(0.3)
    -> Linear(32, 1)  -> Sigmoid
    -> Binary cross-entropy loss

Training strategy:
    - Time-based train/val/test split (no random shuffle across seasons)
      Train : 2003–2019
      Val   : 2021–2022  (2020 was COVID — no tournament)
      Test  : 2023–2025
    - This mimics real deployment: the model never sees future seasons during training.

Baselines compared against:
    1. Always predict 50% (coin flip)
    2. Seed-based: lower seed always wins (traditional bracket logic)

Metrics: Accuracy, Log Loss (Brier score used by the Kaggle competition)

Outputs:
    models/nn_model.pt          — saved model weights
    figures/training_curves.png — loss & accuracy curves

Author: [Your Name]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

DATA_DIR    = "./data"
MODEL_DIR   = "./models"
FIGURES_DIR = "./figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURE_COLS = [f"diff_{c}" for c in [
    "SeedNum", "AvgPtsFor", "AvgPtsAgainst", "AvgPointDiff",
    "AvgFGPct", "AvgFG3Pct", "AvgFTPct", "AvgOppFGPct",
    "AvgReb", "AvgOR", "AvgTO", "AvgStl", "AvgBlk", "AvgAst",
    "PomRank", "MedianRank", "Wins", "Losses", "WinPct",
    "RecentWinPct", "RecentAvgPtDiff",
    "TourneyAppearances", "AvgPastRounds",
    "SOS_AvgOppWinPct", "SOS_AvgOppPomRank",
]]

# Time-based splits — never let future data leak into training
TRAIN_SEASONS = list(range(2003, 2020))          # 17 seasons
VAL_SEASONS   = [2021, 2022]                     # 2020 = COVID, no tournament
TEST_SEASONS  = [2023, 2024, 2025]


# ── Data loading & splitting ───────────────────────────────────────────────────
def load_splits():
    df = pd.read_csv(os.path.join(DATA_DIR, "matchup_dataset.csv"))

    train = df[df["Season"].isin(TRAIN_SEASONS)]
    val   = df[df["Season"].isin(VAL_SEASONS)]
    test  = df[df["Season"].isin(TEST_SEASONS)]

    print(f"[Split]  Train: {len(train)} rows ({len(TRAIN_SEASONS)} seasons)")
    print(f"         Val:   {len(val)} rows ({len(VAL_SEASONS)} seasons)")
    print(f"         Test:  {len(test)} rows ({len(TEST_SEASONS)} seasons)")

    # Fit scaler on train only — never fit on val/test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].values)
    X_val   = scaler.transform(val[FEATURE_COLS].values)
    X_test  = scaler.transform(test[FEATURE_COLS].values)

    y_train = train["Label"].values.astype(np.float32)
    y_val   = val["Label"].values.astype(np.float32)
    y_test  = test["Label"].values.astype(np.float32)

    # Save scaler for inference later
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # Keep test metadata for result analysis
    test_meta = test[["Season","TeamA_ID","TeamB_ID","Label"]].reset_index(drop=True)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test_meta


def to_tensors(X, y):
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1))


# ── Model definition ───────────────────────────────────────────────────────────
class MarchMadnessNN(nn.Module):
    """
    Feed-forward neural network for binary game outcome prediction.

    Design choices:
    - BatchNorm on input: normalizes features even after StandardScaler,
      helps with features on very different scales (rank vs. percentage).
    - Two hidden layers (64 -> 32): small enough to avoid overfitting
      on our ~3,000 training rows.
    - Dropout(0.3): regularization to prevent memorizing specific upsets.
    - Sigmoid output: produces a probability in [0, 1].
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Training loop ──────────────────────────────────────────────────────────────
def train_model(train_data, val_data, input_dim, epochs=150, lr=1e-3, batch_size=64):
    X_train, y_train = to_tensors(*train_data)
    X_val,   y_val   = to_tensors(*val_data)

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True)

    model     = MarchMadnessNN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Reduce LR if val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs):
        # -- train --
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # -- validate --
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            val_pred   = model(X_val)
            t_loss = criterion(train_pred, y_train).item()
            v_loss = criterion(val_pred,   y_val).item()
            t_acc  = ((train_pred > 0.5).float() == y_train).float().mean().item()
            v_acc  = ((val_pred   > 0.5).float() == y_val).float().mean().item()

        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        # Save best model by val loss
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train loss={t_loss:.4f} acc={t_acc:.3f} | "
                  f"val loss={v_loss:.4f} acc={v_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"\n  Best val loss: {best_val_loss:.4f}")
    return model, history


# ── Evaluation & baselines ─────────────────────────────────────────────────────
def evaluate(model, test_data, test_meta):
    X_test, y_test = to_tensors(*test_data)

    model.eval()
    with torch.no_grad():
        probs = model(X_test).squeeze().numpy()

    preds = (probs > 0.5).astype(int)
    labels = y_test.squeeze().numpy()

    nn_acc      = accuracy_score(labels, preds)
    nn_logloss  = log_loss(labels, probs)

    # Baseline 1: always 50%
    coin_logloss = log_loss(labels, np.full_like(probs, 0.5))
    coin_acc     = 0.5

    # Baseline 2: seed-based (lower seed number wins -> negative diff_SeedNum means TeamA better)
    # We stored diff_SeedNum as first feature; negative diff -> TeamA favored -> predict 1
    df = pd.read_csv(os.path.join(DATA_DIR, "matchup_dataset.csv"))
    test_df = df[df["Season"].isin(TEST_SEASONS)].reset_index(drop=True)
    seed_preds = (test_df["diff_SeedNum"] < 0).astype(int)    # TeamA is lower seed -> predict win
    seed_preds[test_df["diff_SeedNum"] == 0] = 1              # same seed: guess TeamA
    seed_acc     = accuracy_score(labels, seed_preds)
    seed_logloss = log_loss(labels, seed_preds.clip(0.01, 0.99))

    print("\n" + "="*55)
    print("EVALUATION ON TEST SET (2023–2025)")
    print("="*55)
    print(f"{'Model':<25} {'Accuracy':>10} {'Log Loss':>10}")
    print("-"*55)
    print(f"{'Coin flip (50%)':<25} {coin_acc:>10.3f} {coin_logloss:>10.4f}")
    print(f"{'Seed-based':<25} {seed_acc:>10.3f} {seed_logloss:>10.4f}")
    print(f"{'Neural Network':<25} {nn_acc:>10.3f} {nn_logloss:>10.4f}")
    print("="*55)

    # Per-season breakdown
    test_meta = test_meta.copy()
    test_meta["prob"]  = probs
    test_meta["pred"]  = preds
    test_meta["correct"] = (preds == labels.astype(int))
    print("\nPer-season accuracy (NN):")
    print(test_meta.groupby("Season")["correct"].mean().to_string())

    return probs, {"nn_acc": nn_acc, "nn_logloss": nn_logloss,
                   "seed_acc": seed_acc, "coin_logloss": coin_logloss}


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Neural Network Training Curves", fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Validation")
    axes[0].set_title("Loss (BCE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot saved] {path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading and splitting data...")
    train_data, val_data, test_data, test_meta = load_splits()

    input_dim = train_data[0].shape[1]
    print(f"\nInput dimension: {input_dim} features")

    print(f"\nTraining for 150 epochs...")
    model, history = train_model(train_data, val_data, input_dim)

    # Save model
    model_path = os.path.join(MODEL_DIR, "nn_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Saved] {model_path}")

    # Evaluate
    probs, metrics = evaluate(model, test_data, test_meta)
    plot_training(history)