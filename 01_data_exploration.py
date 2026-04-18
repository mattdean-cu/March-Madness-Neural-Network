"""
01_explore_data.py
==================
Step 1: Load and explore the March Madness dataset.

This script loads all key CSV files, prints basic stats,
and verifies the data is clean enough to use for modeling.

Author: [Matt Dean, Shona Doyle]
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"          # adjust if running from a different working directory
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Load files ─────────────────────────────────────────────────────────────────
def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all key CSV files into a dictionary of DataFrames."""
    files = {
        "teams":            "MTeams.csv",
        "seeds":            "MNCAATourneySeeds.csv",
        "tourney_compact":  "MNCAATourneyCompactResults.csv",
        "tourney_detailed": "MNCAATourneyDetailedResults.csv",
        "reg_detailed":     "MRegularSeasonDetailedResults.csv",
        "massey":           "MMasseyOrdinals.csv",
        "slots":            "MNCAATourneySlots.csv",
    }
    dfs = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        dfs[key] = pd.read_csv(path)
        print(f"  Loaded {filename:45s}  shape={dfs[key].shape}")
    return dfs


# ── Summary stats ──────────────────────────────────────────────────────────────
def summarize(dfs: dict[str, pd.DataFrame]) -> None:
    """Print a quick overview of each DataFrame."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    df = dfs["teams"]
    print(f"\n[Teams]  {len(df)} total teams | "
          f"seasons {df['FirstD1Season'].min()}–{df['LastD1Season'].max()}")
    active = df[df["LastD1Season"] == df["LastD1Season"].max()]
    print(f"         {len(active)} teams currently Division-I (LastD1Season=2026)")

    df = dfs["seeds"]
    seasons = sorted(df["Season"].unique())
    print(f"\n[Seeds]  {len(seasons)} tournament seasons: {seasons[0]}–{seasons[-1]}")
    print(f"         {df['Season'].value_counts().mean():.1f} teams per tournament on average")

    df = dfs["tourney_compact"]
    print(f"\n[Tourney] {len(df)} total tournament games across all seasons")
    print(f"          seasons: {df['Season'].min()}–{df['Season'].max()}")

    df = dfs["reg_detailed"]
    print(f"\n[Regular Season Detailed]  {len(df):,} games | "
          f"seasons {df['Season'].min()}–{df['Season'].max()}")

    df = dfs["massey"]
    systems = df["SystemName"].nunique()
    print(f"\n[Massey Ordinals]  {len(df):,} rows | "
          f"{systems} ranking systems | "
          f"seasons {df['Season'].min()}–{df['Season'].max()}")

    # Missing values check
    print("\n[Missing Values Check]")
    for name, df in dfs.items():
        nulls = df.isnull().sum().sum()
        flag = "  ✓" if nulls == 0 else f"  ⚠ {nulls} nulls"
        print(f"  {name:20s} {flag}")


# ── Score margin distribution ──────────────────────────────────────────────────
def plot_score_margins(dfs: dict[str, pd.DataFrame]) -> None:
    """Plot the distribution of win margins in tournament games."""
    df = dfs["tourney_compact"].copy()
    df["margin"] = df["WScore"] - df["LScore"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NCAA Tournament Game Score Distributions", fontsize=14, fontweight="bold")

    # Margin histogram
    axes[0].hist(df["margin"], bins=30, color="#2563EB", edgecolor="white", linewidth=0.5)
    axes[0].set_title("Win Margin Distribution")
    axes[0].set_xlabel("Winning Margin (points)")
    axes[0].set_ylabel("Number of Games")
    axes[0].axvline(df["margin"].median(), color="red", linestyle="--",
                    label=f"Median = {df['margin'].median():.1f}")
    axes[0].legend()

    # Games per season
    gpb = df.groupby("Season").size()
    axes[1].bar(gpb.index, gpb.values, color="#10B981", edgecolor="white", linewidth=0.3)
    axes[1].set_title("Tournament Games per Season")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Number of Games")
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "score_distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot saved] {out_path}")
    plt.show()


# ── Seed win rate analysis ─────────────────────────────────────────────────────
def analyze_seed_win_rates(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each tournament game, determine the seed of winner and loser,
    then compute historical upset rates by seed matchup.
    """
    seeds = dfs["seeds"].copy()
    # Extract numeric seed (strip region letter and any play-in suffix)
    seeds["SeedNum"] = seeds["Seed"].str.extract(r'(\d+)').astype(int)
    seed_map = seeds.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

    df = dfs["tourney_compact"].copy()
    df["WSeed"] = df.apply(lambda r: seed_map.get((r["Season"], r["WTeamID"]), None), axis=1)
    df["LSeed"] = df.apply(lambda r: seed_map.get((r["Season"], r["LTeamID"]), None), axis=1)
    df = df.dropna(subset=["WSeed", "LSeed"])

    # Lower seed number = stronger team; upset = winner had HIGHER seed number
    df["upset"] = df["WSeed"] > df["LSeed"]
    upset_rate = df.groupby("WSeed")["upset"].mean().reset_index()
    upset_rate.columns = ["WinningSeed", "FractionOfGamesTheyWereTheUnderdog"]

    print("\n[Seed Win Rate Summary]")
    print("  (An 'upset' here means the winning team had a worse/higher seed)")
    by_seed = df.groupby(["WSeed", "LSeed"]).size().reset_index(name="games")
    print(f"  Total seed-annotated games: {len(df)}")
    print(f"  Overall upset rate: {df['upset'].mean():.1%}")

    # Seed-by-seed win rate (classic 1 vs 16, etc.)
    wins = df.groupby("WSeed").size().rename("wins")
    losses = df.groupby("LSeed").size().rename("losses")
    seed_record = pd.concat([wins, losses], axis=1).fillna(0)
    seed_record["total"] = seed_record["wins"] + seed_record["losses"]
    seed_record["win_pct"] = seed_record["wins"] / seed_record["total"]
    seed_record = seed_record.sort_index()

    print("\n  Win % by seed number:")
    print(seed_record[["wins", "losses", "win_pct"]].to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(seed_record.index, seed_record["win_pct"], color="#F59E0B", edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", label="50% baseline")
    ax.set_title("Historical Win % by Tournament Seed (1985–2024)", fontweight="bold")
    ax.set_xlabel("Seed Number")
    ax.set_ylabel("Win %")
    ax.set_xticks(range(1, 17))
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "seed_win_rates.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot saved] {out_path}")
    plt.show()

    return seed_record


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    dfs = load_data(DATA_DIR)

    summarize(dfs)
    plot_score_margins(dfs)
    seed_record = analyze_seed_win_rates(dfs)