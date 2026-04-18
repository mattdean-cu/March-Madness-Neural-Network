"""
05_evaluate.py
==============
Step 5: Deep evaluation and visualization for the project report.

Generates:
  figures/confusion_matrix.png       - prediction breakdown
  figures/calibration_curve.png      - are our probabilities trustworthy?
  figures/feature_importance.png     - which features matter most
  figures/upset_analysis.png         - how does the model handle upsets?
  figures/seed_diff_vs_prob.png      - does predicted probability scale with seed gap?

All figures are report-ready (labeled, titled, saved at 150dpi).

Author: [Your Name]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    log_loss, accuracy_score, roc_auc_score, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib

DATA_DIR    = "./data"
MODEL_DIR   = "./models"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

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

FEATURE_LABELS = [
    "Seed #", "Avg Pts For", "Avg Pts Against", "Avg Pt Diff",
    "FG%", "3P%", "FT%", "Opp FG%",
    "Rebounds", "Off Rebounds", "Turnovers", "Steals", "Blocks", "Assists",
    "Pomeroy Rank", "Median Rank", "Wins", "Losses", "Win%",
    "Recent Win%", "Recent Pt Diff",
    "Tourney Appearances", "Avg Past Rounds",
    "SOS Opp Win%", "SOS Opp Pom Rank",
]


# ── Load model & data ──────────────────────────────────────────────────────────
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


def load_model_and_data():
    df      = pd.read_csv(os.path.join(DATA_DIR, "matchup_dataset.csv"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    test_df = df[df["Season"].isin(TEST_SEASONS)].reset_index(drop=True)

    X = scaler.transform(test_df[FEATURE_COLS].values)
    y = test_df["Label"].values.astype(np.float32)

    model = MarchMadnessNN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "nn_model.pt"), weights_only=True))
    model.eval()

    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()

    return test_df, X, y, probs, scaler


# ── 1. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(y, probs):
    preds = (probs > 0.5).astype(int)
    cm    = confusion_matrix(y, preds)
    disp  = ConfusionMatrixDisplay(cm, display_labels=["Team B Wins", "Team A Wins"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set (2023–2025)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


# ── 2. Calibration curve ───────────────────────────────────────────────────────
def plot_calibration(y, probs):
    """
    A well-calibrated model: when it says 70% confidence, it should be right ~70% of the time.
    The closer the curve is to the diagonal, the better calibrated the model is.
    """
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "s-", color="#2563EB", label="Neural Network")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Win Rate)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "calibration_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


# ── 3. ROC curve ───────────────────────────────────────────────────────────────
def plot_roc(y, probs):
    fpr, tpr, _ = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", label=f"Neural Network (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
    ax.set_title("ROC Curve — Test Set", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


# ── 4. Feature importance via input perturbation ───────────────────────────────
def plot_feature_importance(model, X, y, scaler):
    """
    Permutation importance: for each feature, shuffle its values and measure
    how much the accuracy drops. Bigger drop = more important feature.
    This is model-agnostic and works well for neural networks.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        base_probs = model(X_tensor).squeeze().numpy()
    base_acc = accuracy_score(y, (base_probs > 0.5).astype(int))

    importances = []
    rng = np.random.default_rng(42)

    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        with torch.no_grad():
            perm_probs = model(torch.tensor(X_perm, dtype=torch.float32)).squeeze().numpy()
        perm_acc = accuracy_score(y, (perm_probs > 0.5).astype(int))
        importances.append(base_acc - perm_acc)

    imp_df = pd.DataFrame({
        "feature": FEATURE_LABELS,
        "importance": importances
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2563EB" if v >= 0 else "#EF4444" for v in imp_df["importance"]]
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Importance (Permutation Method)\n"
                 "Drop in accuracy when feature is shuffled",
                 fontweight="bold")
    ax.set_xlabel("Accuracy Drop (higher = more important)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()

    print("\nTop 5 most important features:")
    print(imp_df.sort_values("importance", ascending=False).head(5).to_string(index=False))


# ── 5. Upset analysis ──────────────────────────────────────────────────────────
def plot_upset_analysis(test_df, probs, y):
    """
    Break games into 'expected' (lower seed wins) vs 'upset' (higher seed wins)
    and show how the model performs on each category.
    """
    df = test_df.copy()
    df["prob"]   = probs
    df["pred"]   = (probs > 0.5).astype(int)
    df["correct"] = (df["pred"] == df["Label"].astype(int))
    df["upset"]   = (df["Label"] == 1) & (df["diff_SeedNum"] > 0)  # underdog won
    df["expected"]= (df["Label"] == 1) & (df["diff_SeedNum"] <= 0) # favorite won

    upset_acc    = df[df["upset"]]["correct"].mean()
    expected_acc = df[df["expected"]]["correct"].mean()
    n_upsets     = df["upset"].sum()
    n_expected   = df["expected"].sum()

    print(f"\n[Upset Analysis]")
    print(f"  Expected outcomes (favorite wins): {n_expected} games | "
          f"Model accuracy: {expected_acc:.1%}")
    print(f"  Upsets (underdog wins):            {n_upsets} games  | "
          f"Model accuracy: {upset_acc:.1%}")

    # Confidence distribution: upset games vs normal games
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Behavior on Upsets vs Expected Outcomes", fontweight="bold")

    axes[0].hist(df[df["upset"]]["prob"],    bins=15, alpha=0.7,
                 color="#EF4444", label=f"Upset games (n={n_upsets})")
    axes[0].hist(df[df["expected"]]["prob"], bins=15, alpha=0.7,
                 color="#2563EB", label=f"Expected games (n={n_expected})")
    axes[0].axvline(0.5, color="black", linestyle="--")
    axes[0].set_title("Predicted Probability Distribution")
    axes[0].set_xlabel("Predicted P(Team A wins)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Accuracy by seed difference bucket
    df["seed_diff_bucket"] = pd.cut(df["diff_SeedNum"],
                                     bins=[-16, -8, -4, -1, 0, 1, 4, 8, 16],
                                     labels=["A much better","A better","A slightly better",
                                             "Same","B slightly better","B better","B much better",
                                             "B much better+"])
    bucket_acc = df.groupby("seed_diff_bucket", observed=True)["correct"].mean()
    axes[1].bar(range(len(bucket_acc)), bucket_acc.values, color="#10B981")
    axes[1].set_xticks(range(len(bucket_acc)))
    axes[1].set_xticklabels(bucket_acc.index, rotation=30, ha="right", fontsize=8)
    axes[1].axhline(0.5, color="red", linestyle="--", label="50% baseline")
    axes[1].set_title("Accuracy by Seed Difference Bucket")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "upset_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


# ── 6. Summary metrics table ───────────────────────────────────────────────────
def print_summary(y, probs):
    preds = (probs > 0.5).astype(int)
    print("\n" + "="*50)
    print("FINAL METRICS SUMMARY")
    print("="*50)
    print(f"  Accuracy  : {accuracy_score(y, preds):.4f}")
    print(f"  Log Loss  : {log_loss(y, probs):.4f}")
    print(f"  ROC AUC   : {roc_auc_score(y, probs):.4f}")
    print(f"  Brier Score: {np.mean((probs - y)**2):.4f}")
    print("="*50)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model and test data...")
    test_df, X, y, probs, scaler = load_model_and_data()

    print_summary(y, probs)
    plot_confusion_matrix(y, probs)
    plot_calibration(y, probs)
    plot_roc(y, probs)
    plot_feature_importance(MarchMadnessNN(len(FEATURE_COLS)), X, y, scaler)
    plot_upset_analysis(test_df, probs, y)

    print("\n✓ Evaluation complete. All figures saved to ./figures/")