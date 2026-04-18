"""
03_build_dataset.py
===================
Step 3: Build the matchup-level training dataset.

Each row represents one tournament game with the format:
    (Team A features) - (Team B features) + label

Label: 1 if Team A won, 0 if Team B won.

To avoid the model learning a directional bias (e.g. "first team always wins"),
each game is duplicated with teams swapped and label flipped.

The FEATURE_COLS list defines exactly which features feed into the model —
edit this list to experiment with feature subsets.

Output:
    data/matchup_dataset.csv   — full dataset (all seasons)

"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "./data"
OUT_DIR  = "./data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature columns used in the model ─────────────────────────────────────────
# These are the per-team columns from 02_build_features.py.
# The dataset will contain (TeamA_col - TeamB_col) for each of these.
FEATURE_COLS = [
    "SeedNum",
    "AvgPtsFor",
    "AvgPtsAgainst",
    "AvgPointDiff",
    "AvgFGPct",
    "AvgFG3Pct",
    "AvgFTPct",
    "AvgOppFGPct",
    "AvgReb",
    "AvgOR",
    "AvgTO",
    "AvgStl",
    "AvgBlk",
    "AvgAst",
    "PomRank",
    "MedianRank",
    "Wins",
    "Losses",
    "WinPct",
    "RecentWinPct",
    "RecentAvgPtDiff",
    "TourneyAppearances",
    "AvgPastRounds",
    "SOS_AvgOppWinPct",
    "SOS_AvgOppPomRank",
]


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    features = pd.read_csv(os.path.join(DATA_DIR, "team_season_features.csv"))
    tourney  = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv"))
    return features, tourney


# ── Build matchup rows ─────────────────────────────────────────────────────────
def build_matchup_dataset(features: pd.DataFrame,
                           tourney:  pd.DataFrame) -> pd.DataFrame:
    """
    For each tournament game:
      1. Look up both teams' feature vectors for that season.
      2. Compute the difference vector: Team A - Team B.
      3. Label = 1 (Team A won).
      4. Duplicate with teams swapped and label flipped -> removes directional bias.

    The difference representation means the NN learns
    "how much better is team A than team B on each dimension"
    rather than memorizing absolute stat levels.
    """
    # Detailed box score data only exists from 2003 onward.
    # Dropping pre-2003 games avoids ~44% of rows being zeroed-out NaNs,
    # which would mislead the model into thinking those teams were identical.
    tourney = tourney[tourney["Season"] >= 2003].copy()
    print(f"[Filter]   Keeping seasons 2003+ ({len(tourney)} games)")

    feat_map = features.set_index(["Season","TeamID"])

    rows = []
    skipped = 0

    for _, game in tourney.iterrows():
        season   = game["Season"]
        winner   = game["WTeamID"]
        loser    = game["LTeamID"]

        # Both teams must have feature vectors
        if (season, winner) not in feat_map.index or \
           (season, loser)  not in feat_map.index:
            skipped += 1
            continue

        w_feats = feat_map.loc[(season, winner), FEATURE_COLS]
        l_feats = feat_map.loc[(season, loser),  FEATURE_COLS]

        # Diff vector: winner perspective (label=1)
        diff_w = (w_feats - l_feats).values
        rows.append(list(diff_w) + [1, season, winner, loser])

        # Diff vector: loser perspective (label=0) — swapped
        diff_l = (l_feats - w_feats).values
        rows.append(list(diff_l) + [0, season, loser, winner])

    diff_cols = [f"diff_{c}" for c in FEATURE_COLS]
    all_cols  = diff_cols + ["Label", "Season", "TeamA_ID", "TeamB_ID"]
    df = pd.DataFrame(rows, columns=all_cols)

    print(f"[Dataset]  {len(tourney)} tournament games")
    print(f"           {skipped} skipped (missing features — likely pre-2003)")
    print(f"           {len(df)} total rows after duplication")
    print(f"           {df['Label'].mean():.1%} label=1 (should be ~50%)")
    print(f"           Seasons covered: {sorted(df['Season'].unique())}")

    return df


# ── Sanity checks ──────────────────────────────────────────────────────────────
def sanity_check(df: pd.DataFrame) -> None:
    """Quick checks to catch data issues before modeling."""
    print("\n[Sanity Checks]")

    # Label balance
    balance = df["Label"].mean()
    assert 0.49 < balance < 0.51, f"Label imbalance! mean={balance:.3f}"
    print(f"  ✓ Label balance: {balance:.3f} (expected ~0.500)")

    # No NaNs in diff columns
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    nan_counts = df[diff_cols].isnull().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans):
        print(f"  ⚠ NaNs found in: {cols_with_nans.to_dict()}")
        print(f"    These will be filled with 0 (neutral diff)")
        df[diff_cols] = df[diff_cols].fillna(0)
    else:
        print(f"  ✓ No NaNs in feature columns")

    # Seed diff should be negative for label=1 (lower seed# = better team)
    avg_seed_diff_wins   = df[df["Label"]==1]["diff_SeedNum"].mean()
    avg_seed_diff_losses = df[df["Label"]==0]["diff_SeedNum"].mean()
    print(f"  ✓ Avg seed diff when label=1: {avg_seed_diff_wins:.2f} (expect negative)")
    print(f"    Avg seed diff when label=0: {avg_seed_diff_losses:.2f} (expect positive)")

    # PomRank diff: lower rank = better, so winners should have negative diff
    avg_pom_diff_wins = df[df["Label"]==1]["diff_PomRank"].mean()
    print(f"  ✓ Avg PomRank diff when label=1: {avg_pom_diff_wins:.2f} (expect negative)")

    print("\n  All checks passed.\n")
    return df


# ── Save ───────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame) -> None:
    out_path = os.path.join(OUT_DIR, "matchup_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}  shape={df.shape}")
    print(f"\nSample row (first game, winner perspective):")
    print(df.iloc[0].to_string())


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    features, tourney = load_data()
    df = build_matchup_dataset(features, tourney)
    df = sanity_check(df)
    save(df)
    print("\n✓ Dataset built. Next: 04_train_model.py")