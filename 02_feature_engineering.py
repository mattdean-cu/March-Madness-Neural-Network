"""
02_build_features.py
====================
Step 2: Feature engineering — build a per-team season profile
that can be used to represent each side of a matchup.

For each (Season, TeamID) we compute:

  BOX SCORE (season averages):
    - AvgPtsFor, AvgPtsAgainst, AvgPointDiff
    - AvgFGPct, AvgFG3Pct, AvgFTPct
    - AvgOppFGPct (defensive shooting allowed)
    - AvgReb, AvgOR, AvgTO, AvgStl, AvgBlk, AvgAst

  RANKINGS:
    - PomRank    : Pomeroy pre-tournament ordinal rank
    - MedianRank : median across all 197 Massey systems

  SEED:
    - SeedNum : numeric tournament seed (1-16)

  SEASON RECORD:
    - Wins, Losses, WinPct

  RECENT FORM (last 10 regular season games):
    - RecentWinPct      : win % in final 10 games
    - RecentAvgPtDiff   : avg point differential in final 10 games

  TOURNAMENT EXPERIENCE (prior seasons only, no data leakage):
    - TourneyAppearances : number of prior tournament appearances
    - AvgPastRounds      : average rounds won per prior appearance

  STRENGTH OF SCHEDULE:
    - SOS_AvgOppWinPct  : average win % of all regular season opponents
    - SOS_AvgOppPomRank : average Pomeroy rank of all regular season opponents
                          (lower = harder schedule; missing for pre-2003 seasons)


Output: data/team_season_features.csv

"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "./data"
OUT_DIR  = "./data"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load raw files ─────────────────────────────────────────────────────────────
def load_raw():
    reg     = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
    massey  = pd.read_csv(os.path.join(DATA_DIR, "MMasseyOrdinals.csv"))
    seeds   = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
    tourney = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv"))
    return reg, massey, seeds, tourney


# ── 1. Box score season averages ───────────────────────────────────────────────
def build_box_score_features(reg: pd.DataFrame):
    """
    Stack winner/loser rows so every team appears on every game they played,
    then average across the full season.
    Returns both the aggregated features and the per-game stacked table
    (reused for recent form and SOS).
    """
    w = reg[["Season","WTeamID","WScore","LScore",
             "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA",
             "WOR","WDR","WAst","WTO","WStl","WBlk","WPF",
             "LFGM","LFGA"]].copy()
    w.columns = ["Season","TeamID","PtsFor","PtsAgainst",
                 "FGM","FGA","FGM3","FGA3","FTM","FTA",
                 "OR","DR","Ast","TO","Stl","Blk","PF",
                 "Opp_FGM","Opp_FGA"]

    l = reg[["Season","LTeamID","LScore","WScore",
             "LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA",
             "LOR","LDR","LAst","LTO","LStl","LBlk","LPF",
             "WFGM","WFGA"]].copy()
    l.columns = ["Season","TeamID","PtsFor","PtsAgainst",
                 "FGM","FGA","FGM3","FGA3","FTM","FTA",
                 "OR","DR","Ast","TO","Stl","Blk","PF",
                 "Opp_FGM","Opp_FGA"]

    games = pd.concat([w, l], ignore_index=True)

    games["FGPct"]    = games["FGM"]     / games["FGA"].replace(0, np.nan)
    games["FG3Pct"]   = games["FGM3"]    / games["FGA3"].replace(0, np.nan)
    games["FTPct"]    = games["FTM"]     / games["FTA"].replace(0, np.nan)
    games["OppFGPct"] = games["Opp_FGM"] / games["Opp_FGA"].replace(0, np.nan)
    games["RebTotal"] = games["OR"] + games["DR"]

    agg = games.groupby(["Season","TeamID"]).agg(
        Games         = ("PtsFor",    "count"),
        AvgPtsFor     = ("PtsFor",    "mean"),
        AvgPtsAgainst = ("PtsAgainst","mean"),
        AvgFGPct      = ("FGPct",     "mean"),
        AvgFG3Pct     = ("FG3Pct",    "mean"),
        AvgFTPct      = ("FTPct",     "mean"),
        AvgOppFGPct   = ("OppFGPct",  "mean"),
        AvgReb        = ("RebTotal",  "mean"),
        AvgOR         = ("OR",        "mean"),
        AvgTO         = ("TO",        "mean"),
        AvgStl        = ("Stl",       "mean"),
        AvgBlk        = ("Blk",       "mean"),
        AvgAst        = ("Ast",       "mean"),
    ).reset_index()

    agg["AvgPointDiff"] = agg["AvgPtsFor"] - agg["AvgPtsAgainst"]
    print(f"[Box scores]   {len(agg):,} (Season, Team) records")
    return agg, games


# ── 2. Massey / Pomeroy rankings ───────────────────────────────────────────────
def build_massey_features(massey: pd.DataFrame) -> pd.DataFrame:
    """Use last pre-tournament snapshot (RankingDayNum=133). Prefer POM, fallback to median."""
    pre = massey[massey["RankingDayNum"] == 133].copy()

    pom = (pre[pre["SystemName"] == "POM"]
           [["Season","TeamID","OrdinalRank"]]
           .rename(columns={"OrdinalRank": "PomRank"}))

    median_rank = (pre.groupby(["Season","TeamID"])["OrdinalRank"]
                   .median().reset_index()
                   .rename(columns={"OrdinalRank": "MedianRank"}))

    ranks = pd.merge(median_rank, pom, on=["Season","TeamID"], how="left")
    ranks["PomRank"] = ranks["PomRank"].fillna(ranks["MedianRank"])
    print(f"[Massey]       {len(ranks):,} (Season, Team) ranking records")
    return ranks


# ── 3. Tournament seed ─────────────────────────────────────────────────────────
def build_seed_features(seeds: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric seed (1-16) from seed string e.g. 'W01' -> 1."""
    s = seeds.copy()
    s["SeedNum"] = s["Seed"].str.extract(r'(\d+)').astype(int)
    s = s[["Season","TeamID","SeedNum"]]
    print(f"[Seeds]        {len(s):,} (Season, Team) seed records")
    return s


# ── 4. Season win/loss record ──────────────────────────────────────────────────
def build_record_features(reg: pd.DataFrame) -> pd.DataFrame:
    """Count wins and losses for each team each season."""
    wins   = reg.groupby(["Season","WTeamID"]).size().reset_index(name="Wins")
    wins   = wins.rename(columns={"WTeamID": "TeamID"})
    losses = reg.groupby(["Season","LTeamID"]).size().reset_index(name="Losses")
    losses = losses.rename(columns={"LTeamID": "TeamID"})

    record = pd.merge(wins, losses, on=["Season","TeamID"], how="outer").fillna(0)
    record["Wins"]   = record["Wins"].astype(int)
    record["Losses"] = record["Losses"].astype(int)
    record["WinPct"] = record["Wins"] / (record["Wins"] + record["Losses"])
    print(f"[Record]       {len(record):,} (Season, Team) win/loss records")
    return record


# ── 5. Recent form — last 10 regular season games ─────────────────────────────
def build_recent_form(reg: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, take their last 10 games by DayNum and compute:
      - RecentWinPct      : win rate over those 10 games
      - RecentAvgPtDiff   : average point differential over those 10 games
    """
    w = reg[["Season","WTeamID","DayNum","WScore","LScore"]].copy()
    w["TeamID"] = w["WTeamID"]
    w["Win"]    = 1
    w["PtDiff"] = w["WScore"] - w["LScore"]
    w = w[["Season","TeamID","DayNum","Win","PtDiff"]]

    l = reg[["Season","LTeamID","DayNum","WScore","LScore"]].copy()
    l["TeamID"] = l["LTeamID"]
    l["Win"]    = 0
    l["PtDiff"] = l["LScore"] - l["WScore"]
    l = l[["Season","TeamID","DayNum","Win","PtDiff"]]

    all_games = pd.concat([w, l], ignore_index=True)
    all_games = all_games.sort_values(["Season","TeamID","DayNum"])

    last10 = (all_games.groupby(["Season","TeamID"])
              .tail(10)
              .groupby(["Season","TeamID"])
              .agg(RecentWinPct    = ("Win",    "mean"),
                   RecentAvgPtDiff = ("PtDiff", "mean"))
              .reset_index())

    print(f"[Recent form]  {len(last10):,} (Season, Team) recent form records")
    return last10


# ── 6. Tournament experience (prior seasons only — no leakage) ─────────────────
def build_experience_features(tourney: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, TeamID), look back at ALL prior seasons only:
      - TourneyAppearances : number of prior tournament appearances
      - AvgPastRounds      : average wins (rounds) per prior appearance
    """
    wins = (tourney.groupby(["Season","WTeamID"])
            .size().reset_index(name="RoundsWon")
            .rename(columns={"WTeamID": "TeamID"}))

    appeared = seeds[["Season","TeamID"]].copy()
    appeared = pd.merge(appeared, wins, on=["Season","TeamID"], how="left")
    appeared["RoundsWon"] = appeared["RoundsWon"].fillna(0).astype(int)
    appeared_sorted = appeared.sort_values("Season")

    records = []
    for (season, team_id), _ in appeared.groupby(["Season","TeamID"]):
        prior = appeared_sorted[
            (appeared_sorted["TeamID"] == team_id) &
            (appeared_sorted["Season"] <  season)
        ]
        records.append({
            "Season":             season,
            "TeamID":             team_id,
            "TourneyAppearances": len(prior),
            "AvgPastRounds":      prior["RoundsWon"].mean() if len(prior) > 0 else 0.0,
        })

    exp = pd.DataFrame(records)
    print(f"[Experience]   {len(exp):,} (Season, Team) experience records")
    return exp


# ── 7. Strength of schedule ────────────────────────────────────────────────────
def build_sos_features(reg: pd.DataFrame, massey: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, TeamID), compute the average quality of opponents faced
    during the regular season using two signals:

      SOS_AvgOppWinPct  : average win % of all opponents that season
                          (available for all seasons in the dataset)

      SOS_AvgOppPomRank : average Pomeroy rank of all opponents that season
                          (only available for seasons 2003+, NaN otherwise)
                          Lower rank = tougher schedule.

    Method:
      1. Build a (Season, TeamID) -> WinPct lookup from the full regular season.
      2. For each game, identify the opponent and look up their win %.
      3. Average over all opponents faced.
      4. Repeat using Pomeroy rank as the quality signal.
    """

    # --- opponent win % lookup ---
    wins   = reg.groupby(["Season","WTeamID"]).size().reset_index(name="W")
    wins   = wins.rename(columns={"WTeamID": "TeamID"})
    losses = reg.groupby(["Season","LTeamID"]).size().reset_index(name="L")
    losses = losses.rename(columns={"LTeamID": "TeamID"})
    record = pd.merge(wins, losses, on=["Season","TeamID"], how="outer").fillna(0)
    record["WinPct"] = record["W"] / (record["W"] + record["L"])
    winpct_map = record.set_index(["Season","TeamID"])["WinPct"].to_dict()

    # --- opponent Pomeroy rank lookup ---
    pom = (massey[(massey["RankingDayNum"] == 133) & (massey["SystemName"] == "POM")]
           [["Season","TeamID","OrdinalRank"]]
           .set_index(["Season","TeamID"])["OrdinalRank"]
           .to_dict())

    # --- build opponent list for each game ---
    # Each row: (Season, TeamID, OpponentID)
    w_opp = reg[["Season","WTeamID","LTeamID"]].rename(
        columns={"WTeamID": "TeamID", "LTeamID": "OppID"})
    l_opp = reg[["Season","LTeamID","WTeamID"]].rename(
        columns={"LTeamID": "TeamID", "WTeamID": "OppID"})
    matchups = pd.concat([w_opp, l_opp], ignore_index=True)

    matchups["OppWinPct"]  = matchups.apply(
        lambda r: winpct_map.get((r["Season"], r["OppID"]), np.nan), axis=1)
    matchups["OppPomRank"] = matchups.apply(
        lambda r: pom.get((r["Season"], r["OppID"]), np.nan), axis=1)

    sos = matchups.groupby(["Season","TeamID"]).agg(
        SOS_AvgOppWinPct  = ("OppWinPct",  "mean"),
        SOS_AvgOppPomRank = ("OppPomRank", "mean"),
    ).reset_index()

    print(f"[SOS]          {len(sos):,} (Season, Team) strength-of-schedule records")
    pom_coverage = sos["SOS_AvgOppPomRank"].notna().mean()
    print(f"               PomRank SOS coverage: {pom_coverage:.1%} of rows")
    return sos


# ── Combine all features and save ──────────────────────────────────────────────
def build_and_save() -> pd.DataFrame:
    reg, massey, seeds, tourney = load_raw()

    box, _ = build_box_score_features(reg)
    ranks  = build_massey_features(massey)
    seed_f = build_seed_features(seeds)
    record = build_record_features(reg)
    recent = build_recent_form(reg)
    exp    = build_experience_features(tourney, seeds)
    sos    = build_sos_features(reg, massey)

    # Start from seed_feats — only tournament teams get a row
    features = seed_f.copy()
    for df in [box, ranks, record, recent, exp, sos]:
        features = pd.merge(features, df, on=["Season","TeamID"], how="left")

    # Fill missing ranking values with season worst (rare, early seasons)
    for col in ["PomRank", "MedianRank"]:
        worst = features.groupby("Season")[col].transform("max")
        features[col] = features[col].fillna(worst)

    out_path = os.path.join(OUT_DIR, "team_season_features.csv")
    features.to_csv(out_path, index=False)

    print(f"\n[Saved] {out_path}  shape={features.shape}")
    print(f"\nAll feature columns ({len(features.columns)}):")
    for c in features.columns:
        print(f"  {c}")
    print(f"\nSample (2024 Duke, TeamID=1181):")
    sample = features[(features["Season"]==2024) & (features["TeamID"]==1181)]
    if len(sample):
        print(sample.T.to_string(header=False))
    else:
        print("  (Duke not in 2024 tournament)")

    return features


if __name__ == "__main__":
    features = build_and_save()