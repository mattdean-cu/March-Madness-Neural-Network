"""
04b_baseline_models.py
======================
Trains and evaluates three models on the same train/val/test split
used by the neural network, enabling direct comparison:

  1. Logistic Regression  — linear baseline
  2. Random Forest        — tree-based ensemble baseline
  3. Neural Network       — loaded from saved weights (04_train_neural_network.py)

Metrics reported for each model:
  - Accuracy
  - Log Loss
  - ROC AUC
  - Brier Score

Output:
  figures/model_comparison.png  — side-by-side bar chart of all metrics

Author: [Your Name]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, log_loss,
                             roc_auc_score, roc_curve)
import torch
import torch.nn as nn
import joblib

DATA_DIR    = "./data"
MODEL_DIR   = "./models"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

TRAIN_SEASONS = list(range(2003, 2020))
VAL_SEASONS   = [2021, 2022]
TEST_SEASONS  = [2023, 2024, 2025]

FEATURE_COLS = [f"diff_{c}" for c in [
    "SeedNum", "AvgPtsFor", "AvgPtsAgainst", "AvgPointDiff",
    "AvgFGPct", "AvgFG3Pct", "AvgFTPct", "AvgOppFGPct",
    "AvgReb", "AvgOR", "AvgTO", "AvgStl", "AvgBlk", "AvgAst",
    "PomRank", "MedianRank", "Wins", "Losses", "WinPct",
    "RecentWinPct", "RecentAvgPtDiff",
    "TourneyAppearances", "AvgPastRounds",
    "SOS_AvgOppWinPct", "SOS_AvgOppPomRank",
]]


# ── Neural network definition (must match 04_train_neural_network.py) ──────────
class MarchMadnessNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),         nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


# ── Load and split data ────────────────────────────────────────────────────────
def load_splits():
    df = pd.read_csv(os.path.join(DATA_DIR, "matchup_dataset.csv"))

    train = df[df["Season"].isin(TRAIN_SEASONS)]
    val   = df[df["Season"].isin(VAL_SEASONS)]
    test  = df[df["Season"].isin(TEST_SEASONS)]

    # Combine train + val for sklearn models (they don't need a separate val set)
    trainval = pd.concat([train, val])

    scaler = StandardScaler()
    X_trainval = scaler.fit_transform(trainval[FEATURE_COLS].values)
    X_test     = scaler.transform(test[FEATURE_COLS].values)
    y_trainval = trainval["Label"].values
    y_test     = test["Label"].values

    # Also load the NN's scaler separately (fit on train only)
    nn_scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_test_nn  = nn_scaler.transform(test[FEATURE_COLS].values)

    print(f"Train+Val: {len(trainval)} rows | Test: {len(test)} rows")
    return X_trainval, y_trainval, X_test, y_test, X_test_nn


# ── Train baseline models ──────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression: linear model, interpretable coefficients.
    C=1.0 is standard regularization; max_iter=1000 ensures convergence.
    """
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    print("  Logistic Regression trained.")
    return lr


def train_random_forest(X_train, y_train):
    """
    Random Forest: ensemble of decision trees.
    n_estimators=300 for stability; max_depth=8 prevents overfitting
    on our relatively small dataset (~2,500 rows).
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("  Random Forest trained.")
    return rf


def load_neural_network(input_dim):
    model = MarchMadnessNN(input_dim)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "nn_model.pt"), weights_only=True))
    model.eval()
    print("  Neural Network loaded.")
    return model


# ── Evaluate all models ────────────────────────────────────────────────────────
def evaluate_all(models_and_probs, y_test):
    """
    models_and_probs: list of (name, probs) tuples
    Returns a DataFrame of metrics for each model.
    """
    rows = []
    for name, probs in models_and_probs:
        preds = (probs > 0.5).astype(int)
        rows.append({
            "Model":       name,
            "Accuracy":    accuracy_score(y_test, preds),
            "Log Loss":    log_loss(y_test, probs),
            "ROC AUC":     roc_auc_score(y_test, probs),
            "Brier Score": np.mean((probs - y_test) ** 2),
        })

    results = pd.DataFrame(rows)

    print("\n" + "="*62)
    print("MODEL COMPARISON — TEST SET (2023–2025)")
    print("="*62)
    print(f"{'Model':<22} {'Accuracy':>9} {'Log Loss':>9} {'AUC':>7} {'Brier':>7}")
    print("-"*62)
    for _, r in results.iterrows():
        print(f"{r['Model']:<22} {r['Accuracy']:>9.3f} {r['Log Loss']:>9.4f} "
              f"{r['ROC AUC']:>7.3f} {r['Brier Score']:>7.4f}")
    print("="*62)
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_comparison(results, models_and_probs, y_test):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Model Comparison — Logistic Regression vs Random Forest vs Neural Network",
                 fontsize=13, fontweight="bold")

    colors = ["#6B7280", "#10B981", "#2563EB"]
    models = results["Model"].tolist()

    # ── 1. Accuracy bar chart ──
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(models, results["Accuracy"], color=colors, edgecolor="white")
    ax1.set_ylim(0.5, 0.85)
    ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_ylabel("Accuracy")
    ax1.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Coin flip")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, results["Accuracy"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.tick_params(axis='x', labelsize=8)

    # ── 2. Log Loss bar chart (lower is better) ──
    ax2 = fig.add_subplot(2, 2, 2)
    bars = ax2.bar(models, results["Log Loss"], color=colors, edgecolor="white")
    ax2.set_title("Log Loss (lower = better)", fontweight="bold")
    ax2.set_ylabel("Log Loss")
    ax2.axhline(0.693, color="red", linestyle="--", linewidth=0.8, label="Coin flip (0.693)")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, results["Log Loss"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax2.tick_params(axis='x', labelsize=8)

    # ── 3. ROC AUC bar chart ──
    ax3 = fig.add_subplot(2, 2, 3)
    bars = ax3.bar(models, results["ROC AUC"], color=colors, edgecolor="white")
    ax3.set_ylim(0.5, 0.95)
    ax3.set_title("ROC AUC (higher = better)", fontweight="bold")
    ax3.set_ylabel("AUC")
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax3.legend(fontsize=8)
    for bar, val in zip(bars, results["ROC AUC"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.tick_params(axis='x', labelsize=8)

    # ── 4. ROC curves overlay ──
    ax4 = fig.add_subplot(2, 2, 4)
    for (name, probs), color in zip(models_and_probs, colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax4.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax4.plot([0,1],[0,1], "k--", linewidth=0.8, label="Random")
    ax4.set_title("ROC Curves", fontweight="bold")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.show()


def plot_roc_overlay(models_and_probs, y_test):
    """Standalone clean ROC curve for the report."""
    colors = ["#6B7280", "#10B981", "#2563EB"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, probs), color in zip(models_and_probs, colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", linewidth=0.8, label="Random baseline")
    ax.set_title("ROC Curve Comparison — All Models", fontweight="bold", fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    X_trainval, y_trainval, X_test, y_test, X_test_nn = load_splits()

    print("\nTraining models...")
    lr  = train_logistic_regression(X_trainval, y_trainval)
    rf  = train_random_forest(X_trainval, y_trainval)
    nn  = load_neural_network(input_dim=len(FEATURE_COLS))

    print("\nGenerating predictions...")
    lr_probs = lr.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]
    with torch.no_grad():
        nn_probs = nn(torch.tensor(X_test_nn, dtype=torch.float32)).squeeze().numpy()

    models_and_probs = [
        ("Logistic Regression", lr_probs),
        ("Random Forest",       rf_probs),
        ("Neural Network",      nn_probs),
    ]

    results = evaluate_all(models_and_probs, y_test)
    plot_comparison(results, models_and_probs, y_test)
    plot_roc_overlay(models_and_probs, y_test)

    # Save models for potential later use
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print(f"\n[Saved] models/logistic_regression.pkl")
    print(f"[Saved] models/random_forest.pkl")
    print("\n✓ Model comparison complete.")