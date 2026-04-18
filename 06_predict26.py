"""
06_predict_2026.py
==================
Generates model predictions for every Round 1 matchup in the
2026 NCAA Tournament and visualizes them as a bracket chart.

Since the 2026 tournament is complete (Michigan won), we also
mark actual Round 1 upsets so you can see where the model
was right and wrong — a strong visual for your presentation.

Key fix vs first attempt:
  - Builds 2026 team features on the fly from MRegularSeasonDetailedResults.csv
    since MNCAATourneySeeds.csv doesn't include 2026 seeds yet.
  - Uses correct TeamIDs verified against MTeams.csv.

Author: [Your Name]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import joblib

DATA_DIR    = "./data"
MODEL_DIR   = "./models"
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [f"diff_{c}" for c in [
    "SeedNum", "AvgPtsFor", "AvgPtsAgainst", "AvgPointDiff",
    "AvgFGPct", "AvgFG3Pct", "AvgFTPct", "AvgOppFGPct",
    "AvgReb", "AvgOR", "AvgTO", "AvgStl", "AvgBlk", "AvgAst",
    "PomRank", "MedianRank", "Wins", "Losses", "WinPct",
    "RecentWinPct", "RecentAvgPtDiff",
    "TourneyAppearances", "AvgPastRounds",
    "SOS_AvgOppWinPct", "SOS_AvgOppPomRank",
]]

RAW_COLS = [c.replace("diff_", "") for c in FEATURE_COLS]

# ── 2026 bracket — verified TeamIDs from MTeams.csv ───────────────────────────
# Format: (seed, team_name, team_id)
BRACKET_2026 = {
    "East": [
        (1,  "Duke",         1181), (2,  "UConn",        1163),
        (3,  "Michigan St",  1277), (4,  "Kansas",        1242),
        (5,  "St. John's",   1385), (6,  "Louisville",    1257),
        (7,  "UCLA",         1417), (8,  "Ohio St",       1326),
        (9,  "TCU",          1395), (10, "UCF",           1416),
        (11, "S Florida",    1378), (12, "N Iowa",        1320),
        (13, "Cal Baptist",  1465), (14, "N Dakota St",   1295),
        (15, "Furman",       1202), (16, "Siena",         1373),
    ],
    "West": [
        (1,  "Arizona",      1112), (2,  "Purdue",        1345),
        (3,  "Gonzaga",      1211), (4,  "Arkansas",      1116),
        (5,  "Wisconsin",    1458), (6,  "BYU",           1140),
        (7,  "Miami FL",     1274), (8,  "Villanova",     1437),
        (9,  "Utah St",      1429), (10, "Missouri",      1281),
        (11, "Texas",        1400), (12, "High Point",    1219),
        (13, "Hawaii",       1218), (14, "Kennesaw St",   1244),
        (15, "Queens NC",    1474), (16, "LIU Brooklyn",  1254),
    ],
    "Midwest": [
        (1,  "Michigan",     1276), (2,  "Iowa St",       1235),
        (3,  "Virginia",     1438), (4,  "Alabama",       1104),
        (5,  "Texas Tech",   1403), (6,  "Tennessee",     1397),
        (7,  "Kentucky",     1246), (8,  "Georgia",       1208),
        (9,  "St Louis",     1387), (10, "Santa Clara",   1365),
        (11, "Miami OH",     1275), (12, "Akron",         1103),
        (13, "Hofstra",      1220), (14, "Wright St",     1460),
        (15, "Tennessee St", 1398), (16, "UMBC",          1420),
    ],
    "South": [
        (1,  "Florida",      1196), (2,  "Houston",       1222),
        (3,  "Illinois",     1228), (4,  "Nebraska",      1304),
        (5,  "Vanderbilt",   1435), (6,  "N Carolina",    1314),
        (7,  "St Mary's CA", 1388), (8,  "Clemson",       1155),
        (9,  "Iowa",         1234), (10, "Texas A&M",     1401),
        (11, "VCU",          1433), (12, "McNeese St",    1270),
        (13, "Troy",         1407), (14, "Penn",          1335),
        (15, "Idaho",        1225), (16, "Prairie View",  1341),
    ],
}

# Round of 32 matchups — winners of R1 games face each other
# Format: (region, team_name, team_id, seed) for each pairing
# Seed matchups: winner of 1/16 vs winner of 8/9, 2/15 vs 7/10, 3/14 vs 6/11, 4/13 vs 5/12
R2_MATCHUPS = [
    # South
    ("South", "Florida",     1196, 1,  "Iowa",        1234, 9),
    ("South", "Nebraska",    1304, 4,  "Vanderbilt",  1435, 5),
    ("South", "Illinois",    1228, 3,  "VCU",         1433, 11),
    ("South", "Houston",     1222, 2,  "Texas A&M",   1401, 10),
    # Midwest
    ("Midwest", "Michigan",  1276, 1,  "St Louis",    1387, 9),
    ("Midwest", "Alabama",   1104, 4,  "Texas Tech",  1403, 5),
    ("Midwest", "Tennessee", 1397, 6,  "Virginia",    1438, 3),
    ("Midwest", "Iowa St",   1235, 2,  "Kentucky",    1246, 7),
    # East
    ("East", "Duke",         1181, 1,  "TCU",         1395, 9),
    ("East", "St. John's",   1385, 5,  "Kansas",      1242, 4),
    ("East", "Michigan St",  1277, 3,  "Louisville",  1257, 6),
    ("East", "UConn",        1163, 2,  "UCLA",        1417, 7),
    # West
    ("West", "Arizona",      1112, 1,  "Utah St",     1429, 9),
    ("West", "Arkansas",     1116, 4,  "High Point",  1219, 12),
    ("West", "Texas",        1400, 11, "Gonzaga",     1211, 3),
    ("West", "Purdue",       1345, 2,  "Miami FL",    1274, 7),
]

# Actual Round of 32 winners (team_id of winner)
R2_ACTUAL_WINNERS = {
    ("South",   1196): 1234,   # Iowa over Florida
    ("South",   1304): 1304,   # Nebraska over Vanderbilt
    ("South",   1228): 1228,   # Illinois over VCU
    ("South",   1222): 1222,   # Houston over Texas A&M
    ("Midwest", 1276): 1276,   # Michigan over St Louis
    ("Midwest", 1104): 1104,   # Alabama over Texas Tech
    ("Midwest", 1397): 1397,   # Tennessee over Virginia
    ("Midwest", 1235): 1235,   # Iowa St over Kentucky
    ("East",    1181): 1181,   # Duke over TCU
    ("East",    1385): 1385,   # St. John's over Kansas  (upset! 5 over 4)
    ("East",    1277): 1277,   # Michigan St over Louisville
    ("East",    1163): 1163,   # UConn over UCLA
    ("West",    1112): 1112,   # Arizona over Utah St
    ("West",    1116): 1116,   # Arkansas over High Point
    ("West",    1400): 1400,   # Texas over Gonzaga  (upset! 11 over 3)
    ("West",    1345): 1345,   # Purdue over Miami FL
}

# Upsets in Round of 32 (higher seed won)
R2_UPSETS = {
    ("South",  1196),   # Iowa (9) over Florida (1)
    ("East",   1385),   # St. John's (5) over Kansas (4) — marginal
    ("West",   1400),   # Texas (11) over Gonzaga (3)
    ("Midwest",1397),   # Tennessee (6) over Virginia (3)
}

# Actual Round 1 winners — used to show winner on left in the chart
# Key: (region, fav_seed), Value: True if favorite won, False if underdog won
R1_FAV_WON = {
    ("East",  1): True,  ("East",  2): True,  ("East",  3): True,  ("East",  4): True,
    ("East",  5): True,  ("East",  6): True,  ("East",  7): True,  ("East",  8): False,  # TCU upset Ohio St
    ("West",  1): True,  ("West",  2): True,  ("West",  3): True,  ("West",  4): True,
    ("West",  5): False, ("West",  6): False, ("West",  7): True,  ("West",  8): False,  # Utah St upset Villanova
    ("Midwest",1): True, ("Midwest",2): True, ("Midwest",3): True, ("Midwest",4): True,
    ("Midwest",5): True, ("Midwest",6): True, ("Midwest",7): True, ("Midwest",8): False, # St Louis upset Georgia
    ("South", 1): True,  ("South", 2): True,  ("South", 3): True,  ("South", 4): True,
    ("South", 5): True,  ("South", 6): False, ("South", 7): False, ("South", 8): False,  # Iowa, VCU, TexasA&M upsets
}

# Sweet 16 matchups — winners of R2 games
# Format: (region, name_a, id_a, seed_a, name_b, id_b, seed_b)
# Team A is always the lower seed (higher rank) going in
S16_MATCHUPS = [
    # East
    ("East",    "Duke",      1181, 1,  "St. John's", 1385, 5),
    ("East",    "UConn",     1163, 2,  "Michigan St",1277, 3),
    # West
    ("West",    "Arizona",   1112, 1,  "Arkansas",   1116, 4),
    ("West",    "Purdue",    1345, 2,  "Texas",      1400, 11),
    # Midwest
    ("Midwest", "Michigan",  1276, 1,  "Alabama",    1104, 4),
    ("Midwest", "Tennessee", 1397, 6,  "Iowa St",    1235, 2),
    # South
    ("South",   "Iowa",      1234, 9,  "Nebraska",   1304, 4),
    ("South",   "Illinois",  1228, 3,  "Houston",    1222, 2),
]

# Actual Sweet 16 winners
S16_ACTUAL_WINNERS = {
    ("East",    1181): 1181,   # Duke over St. John's
    ("East",    1163): 1163,   # UConn over Michigan St
    ("West",    1112): 1112,   # Arizona over Arkansas
    ("West",    1345): 1345,   # Purdue over Texas
    ("Midwest", 1276): 1276,   # Michigan over Alabama
    ("Midwest", 1397): 1397,   # Tennessee over Iowa St  (upset! 6 over 2)
    ("South",   1234): 1234,   # Iowa over Nebraska      (upset! 9 over 4)
    ("South",   1228): 1228,   # Illinois over Houston   (upset! 3 over 2)
}

S16_UPSETS = {
    ("Midwest", 1397),   # Tennessee (6) over Iowa St (2)
    ("South",   1234),   # Iowa (9) over Nebraska (4)
    ("South",   1228),   # Illinois (3) over Houston (2)
}

# Actual Round 1 upsets in 2026 (higher seed # won = upset)
# Format: (region, underdog_seed_that_won)
ACTUAL_UPSETS_R1 = {
    ("South",    9),   # Iowa over #8 Clemson
    ("South",   11),   # VCU over #6 N Carolina (OT)
    ("South",   10),   # Texas A&M over #7 Saint Mary's CA
    ("Midwest",  9),   # Saint Louis over #8 Georgia
    ("East",     9),   # TCU over #8 Ohio St
    ("West",    11),   # Texas over #6 BYU
    ("West",    12),   # High Point over #5 Wisconsin
    ("West",     9),   # Utah St over #8 Villanova
}

# Round 1 seed matchups: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
R1_PAIRS = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]

# Actual Round 1 winners — True if favorite (lower seed) won
R1_FAV_WON = {
    ("East",  1): True,  ("East",  2): True,  ("East",  3): True,  ("East",  4): True,
    ("East",  5): True,  ("East",  6): True,  ("East",  7): True,  ("East",  8): False,
    ("West",  1): True,  ("West",  2): True,  ("West",  3): True,  ("West",  4): True,
    ("West",  5): False, ("West",  6): False, ("West",  7): True,  ("West",  8): False,
    ("Midwest",1): True, ("Midwest",2): True, ("Midwest",3): True, ("Midwest",4): True,
    ("Midwest",5): True, ("Midwest",6): True, ("Midwest",7): True, ("Midwest",8): False,
    ("South", 1): True,  ("South", 2): True,  ("South", 3): True,  ("South", 4): True,
    ("South", 5): True,  ("South", 6): False, ("South", 7): False, ("South", 8): False,
}

# ── Elite Eight ────────────────────────────────────────────────────────────────
# South winner vs Midwest winner, East winner vs West winner
E8_MATCHUPS = [
    ("South/Midwest", "Michigan",  1276, 1, "Tennessee", 1397, 6),
    ("South/Midwest", "Illinois",  1228, 3, "Iowa",      1234, 9),
    ("East/West",     "UConn",     1163, 2, "Duke",      1181, 1),
    ("East/West",     "Arizona",   1112, 1, "Purdue",    1345, 2),
]
E8_ACTUAL_WINNERS = {
    ("South/Midwest", 1276): 1276,  # Michigan over Tennessee
    ("South/Midwest", 1228): 1228,  # Illinois over Iowa      (upset! 3 over 9 — fav wins)
    ("East/West",     1163): 1163,  # UConn over Duke         (upset! 2 over 1)
    ("East/West",     1112): 1112,  # Arizona over Purdue
}
E8_UPSETS = {
    ("East/West", 1163),   # UConn (2) over Duke (1)
}

# ── Final Four ─────────────────────────────────────────────────────────────────
FF_MATCHUPS = [
    ("Final Four", "UConn",    1163, 2, "Illinois", 1228, 3),
    ("Final Four", "Michigan", 1276, 1, "Arizona",  1112, 1),
]
FF_ACTUAL_WINNERS = {
    ("Final Four", 1163): 1163,  # UConn over Illinois
    ("Final Four", 1276): 1276,  # Michigan over Arizona
}
FF_UPSETS = set()  # no upsets — results matched seeding expectations

# ── Championship ───────────────────────────────────────────────────────────────
CHAMP_MATCHUPS = [
    ("Championship", "Michigan", 1276, 1, "UConn", 1163, 2),
]
CHAMP_ACTUAL_WINNERS = {
    ("Championship", 1276): 1276,  # Michigan over UConn — CHAMPIONS
}
CHAMP_UPSETS = set()

# ── Model definition ───────────────────────────────────────────────────────────
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


# ── Build 2026 features from scratch ──────────────────────────────────────────
def build_2026_features() -> pd.DataFrame:
    """
    Re-derive all feature columns for Season=2026 teams directly from
    MRegularSeasonDetailedResults.csv and MMasseyOrdinals.csv.
    This mirrors the logic in 02_build_features.py but targets only 2026.
    """
    print("Building 2026 season features from raw data...")
    reg    = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
    massey = pd.read_csv(os.path.join(DATA_DIR, "MMasseyOrdinals.csv"))
    seeds_hist = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
    tourney    = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv"))

    reg26 = reg[reg["Season"] == 2026].copy()

    # ── Box scores ──
    w = reg26[["WTeamID","WScore","LScore",
               "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA",
               "WOR","WDR","WAst","WTO","WStl","WBlk",
               "LFGM","LFGA"]].copy()
    w.columns = ["TeamID","PtsFor","PtsAgainst",
                 "FGM","FGA","FGM3","FGA3","FTM","FTA",
                 "OR","DR","Ast","TO","Stl","Blk","Opp_FGM","Opp_FGA"]

    l = reg26[["LTeamID","LScore","WScore",
               "LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA",
               "LOR","LDR","LAst","LTO","LStl","LBlk",
               "WFGM","WFGA"]].copy()
    l.columns = ["TeamID","PtsFor","PtsAgainst",
                 "FGM","FGA","FGM3","FGA3","FTM","FTA",
                 "OR","DR","Ast","TO","Stl","Blk","Opp_FGM","Opp_FGA"]

    games = pd.concat([w, l], ignore_index=True)
    games["FGPct"]    = games["FGM"]     / games["FGA"].replace(0, np.nan)
    games["FG3Pct"]   = games["FGM3"]    / games["FGA3"].replace(0, np.nan)
    games["FTPct"]    = games["FTM"]     / games["FTA"].replace(0, np.nan)
    games["OppFGPct"] = games["Opp_FGM"] / games["Opp_FGA"].replace(0, np.nan)
    games["RebTotal"] = games["OR"] + games["DR"]

    box = games.groupby("TeamID").agg(
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
    box["AvgPointDiff"] = box["AvgPtsFor"] - box["AvgPtsAgainst"]
    box["Season"] = 2026

    # ── Win/loss record ──
    wins   = reg26.groupby("WTeamID").size().reset_index(name="Wins").rename(columns={"WTeamID":"TeamID"})
    losses = reg26.groupby("LTeamID").size().reset_index(name="Losses").rename(columns={"LTeamID":"TeamID"})
    record = pd.merge(wins, losses, on="TeamID", how="outer").fillna(0)
    record["WinPct"] = record["Wins"] / (record["Wins"] + record["Losses"])
    record["Season"] = 2026

    # ── Recent form (last 10 games) ──
    wg = reg26[["WTeamID","DayNum","WScore","LScore"]].copy()
    wg["TeamID"] = wg["WTeamID"]; wg["Win"] = 1; wg["PtDiff"] = wg["WScore"] - wg["LScore"]
    lg = reg26[["LTeamID","DayNum","WScore","LScore"]].copy()
    lg["TeamID"] = lg["LTeamID"]; lg["Win"] = 0; lg["PtDiff"] = lg["LScore"] - lg["WScore"]
    all_g = pd.concat([wg[["TeamID","DayNum","Win","PtDiff"]],
                       lg[["TeamID","DayNum","Win","PtDiff"]]], ignore_index=True)
    all_g = all_g.sort_values(["TeamID","DayNum"])
    recent = (all_g.groupby("TeamID").tail(10)
              .groupby("TeamID")
              .agg(RecentWinPct=("Win","mean"), RecentAvgPtDiff=("PtDiff","mean"))
              .reset_index())
    recent["Season"] = 2026

    # ── Massey / Pomeroy rank ──
    m26 = massey[(massey["Season"]==2026) & (massey["RankingDayNum"]==133)]
    pom = (m26[m26["SystemName"]=="POM"][["TeamID","OrdinalRank"]]
           .rename(columns={"OrdinalRank":"PomRank"}))
    med = (m26.groupby("TeamID")["OrdinalRank"].median().reset_index()
           .rename(columns={"OrdinalRank":"MedianRank"}))
    ranks = pd.merge(med, pom, on="TeamID", how="left")
    ranks["PomRank"] = ranks["PomRank"].fillna(ranks["MedianRank"])
    ranks["Season"]  = 2026

    # ── SOS ──
    winpct_map = record.set_index("TeamID")["WinPct"].to_dict()
    pom_map    = pom.set_index("TeamID")["PomRank"].to_dict()
    wo = reg26[["WTeamID","LTeamID"]].rename(columns={"WTeamID":"TeamID","LTeamID":"OppID"})
    lo = reg26[["LTeamID","WTeamID"]].rename(columns={"LTeamID":"TeamID","WTeamID":"OppID"})
    mo = pd.concat([wo, lo], ignore_index=True)
    mo["OppWinPct"]  = mo["OppID"].map(winpct_map)
    mo["OppPomRank"] = mo["OppID"].map(pom_map)
    sos = mo.groupby("TeamID").agg(
        SOS_AvgOppWinPct  = ("OppWinPct",  "mean"),
        SOS_AvgOppPomRank = ("OppPomRank", "mean"),
    ).reset_index()
    sos["Season"] = 2026

    # ── Tournament experience (from historical data) ──
    t_wins = (tourney.groupby(["Season","WTeamID"]).size()
              .reset_index(name="RoundsWon").rename(columns={"WTeamID":"TeamID"}))
    appeared = seeds_hist[["Season","TeamID"]].copy()
    appeared = pd.merge(appeared, t_wins, on=["Season","TeamID"], how="left")
    appeared["RoundsWon"] = appeared["RoundsWon"].fillna(0).astype(int)

    all_team_ids = box["TeamID"].unique()
    exp_rows = []
    for tid in all_team_ids:
        prior = appeared[appeared["TeamID"] == tid]
        exp_rows.append({
            "TeamID": tid, "Season": 2026,
            "TourneyAppearances": len(prior),
            "AvgPastRounds": prior["RoundsWon"].mean() if len(prior) > 0 else 0.0,
        })
    exp = pd.DataFrame(exp_rows)

    # ── Merge all ──
    feat = box.copy()
    for df in [record[["Season","TeamID","Wins","Losses","WinPct"]],
               recent[["Season","TeamID","RecentWinPct","RecentAvgPtDiff"]],
               ranks[["Season","TeamID","PomRank","MedianRank"]],
               exp[["Season","TeamID","TourneyAppearances","AvgPastRounds"]],
               sos[["Season","TeamID","SOS_AvgOppWinPct","SOS_AvgOppPomRank"]]]:
        feat = pd.merge(feat, df, on=["Season","TeamID"], how="left")

    # SeedNum will be filled per-matchup from the bracket dict
    feat["SeedNum"] = np.nan
    print(f"  Built features for {len(feat)} teams in 2026")
    return feat


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model  = MarchMadnessNN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "nn_model.pt"), weights_only=True))
    model.eval()
    return scaler, model


# ── Predict one matchup ────────────────────────────────────────────────────────
def predict_matchup(tid_a, seed_a, tid_b, seed_b, feat26, scaler, model):
    """Returns P(team_a wins) or None if features missing."""
    fm = feat26.set_index("TeamID")
    if tid_a not in fm.index or tid_b not in fm.index:
        return None

    a = fm.loc[tid_a, RAW_COLS].copy()
    b = fm.loc[tid_b, RAW_COLS].copy()

    # Inject seeds
    a["SeedNum"] = seed_a
    b["SeedNum"] = seed_b

    diff = (a.values - b.values).reshape(1, -1).astype(float)
    if np.any(np.isnan(diff)):
        diff = np.nan_to_num(diff, nan=0.0)

    diff_scaled = scaler.transform(diff)
    with torch.no_grad():
        prob = model(torch.tensor(diff_scaled, dtype=torch.float32)).item()
    return prob


# ── Build prediction table ─────────────────────────────────────────────────────
def build_predictions(feat26, scaler, model):
    rows = []
    for region, teams in BRACKET_2026.items():
        seed_map = {s: (n, tid) for s, n, tid in teams}
        for s_fav, s_dog in R1_PAIRS:
            fav_name, fav_id = seed_map[s_fav]
            dog_name, dog_id = seed_map[s_dog]
            prob = predict_matchup(fav_id, s_fav, dog_id, s_dog,
                                   feat26, scaler, model)
            is_upset = (region, s_dog) in ACTUAL_UPSETS_R1
            rows.append({
                "Region": region, "FavSeed": s_fav, "DogSeed": s_dog,
                "Favorite": fav_name, "Underdog": dog_name,
                "FavProb": prob, "ActualUpset": is_upset,
            })
    return pd.DataFrame(rows)


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_predictions(df):
    regions = ["East", "West", "Midwest", "South"]
    cmap    = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)
    fig.suptitle(
        "2026 NCAA Tournament — Round 1 Predicted Win Probabilities\n"
        "Green = high confidence favorite  |  Yellow = toss-up  |  ★ = actual upset",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, region in zip(axes, regions):
        rdf = df[df["Region"] == region].reset_index(drop=True)

        for i, row in rdf.iterrows():
            y        = len(rdf) - 1 - (i % len(rdf))
            prob     = row["FavProb"]
            fav_won  = R1_FAV_WON.get((region, row["FavSeed"]), True)

            # Show winner on left — flip prob and labels if underdog won
            if fav_won:
                left_name, left_seed   = row["Favorite"], row["FavSeed"]
                right_name, right_seed = row["Underdog"],  row["DogSeed"]
                bar_val = prob if prob is not None else 0.5
            else:
                left_name, left_seed   = row["Underdog"],  row["DogSeed"]
                right_name, right_seed = row["Favorite"], row["FavSeed"]
                bar_val = (1 - prob) if prob is not None else 0.5

            color    = cmap(bar_val) if prob is not None else "#cccccc"
            prob_str = f"{bar_val:.0%}" if prob is not None else "N/A"

            ax.barh(y, bar_val, color=color, height=0.72,
                    edgecolor="white", linewidth=0.5)

            ax.text(0.02, y, f"#{left_seed} {left_name}",
                    va="center", ha="left", fontsize=8,
                    fontweight="bold", color="black")

            ax.text(0.98, y, f"#{right_seed} {right_name}",
                    va="center", ha="right", fontsize=7.5, color="#333333",
                    transform=ax.get_yaxis_transform())

            text_x = bar_val - 0.02 if bar_val > 0.15 else bar_val + 0.02
            ax.text(text_x, y, prob_str, va="center",
                    ha="right" if bar_val > 0.15 else "left",
                    fontsize=8, color="white", fontweight="bold")

            if row["ActualUpset"]:
                ax.text(1.02, y, "★", va="center", ha="left",
                        fontsize=10, color="#DC2626", fontweight="bold",
                        transform=ax.get_yaxis_transform())

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(rdf) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["50%", "75%", "100%"], fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{region} Region", fontweight="bold", fontsize=11, pad=8)
        ax.set_xlabel("P(Winner wins — as predicted by model)", fontsize=8)

    patches = [
        mpatches.Patch(color=cmap(0.88), label="High confidence (>85%)"),
        mpatches.Patch(color=cmap(0.70), label="Moderate (65–85%)"),
        mpatches.Patch(color=cmap(0.52), label="Toss-up (<65%)"),
        mpatches.Patch(color="#DC2626",  label="★ Actual upset"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "2026_tournament_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.show()


# ── Print table ────────────────────────────────────────────────────────────────
def print_summary(df):
    print("\n2026 Round 1 Predictions")
    print("="*68)
    print(f"{'Region':<10} {'Matchup':<35} {'Model Prob':>10} {'Result':>10}")
    print("-"*68)
    for _, r in df.iterrows():
        prob_str = f"{r['FavProb']:.1%}" if r["FavProb"] is not None else "N/A"
        result   = "UPSET ★" if r["ActualUpset"] else "Fav won"
        matchup  = f"#{r['FavSeed']} {r['Favorite']:12s} vs #{r['DogSeed']} {r['Underdog']}"
        print(f"{r['Region']:<10} {matchup:<35} {prob_str:>10} {result:>10}")

    upsets = df[df["ActualUpset"]]
    print(f"\nModel predicted upset correctly: "
          f"{(upsets['FavProb'] < 0.5).sum()}/{len(upsets)} actual upsets")
    correct = ((df["FavProb"] >= 0.5) == ~df["ActualUpset"]).sum()
    print(f"Overall R1 accuracy: {correct}/{len(df)} ({correct/len(df):.1%})")


def build_r2_predictions(feat26, scaler, model):
    rows = []
    for entry in R2_MATCHUPS:
        region, name_a, id_a, seed_a, name_b, id_b, seed_b = entry
        prob = predict_matchup(id_a, seed_a, id_b, seed_b, feat26, scaler, model)
        actual_winner = R2_ACTUAL_WINNERS.get((region, id_a))
        actual_upset  = (region, id_a) in R2_UPSETS
        # prob is P(team_a wins); if team_a is the favorite (lower seed)
        fav_won = (actual_winner == id_a)
        rows.append({
            "Region":      region,
            "FavName":     name_a,
            "FavID":       id_a,
            "FavSeed":     seed_a,
            "DogName":     name_b,
            "DogID":       id_b,
            "DogSeed":     seed_b,
            "FavProb":     prob,
            "ActualUpset": actual_upset,
            "FavWon":      fav_won,
        })
    return pd.DataFrame(rows)


def plot_r2_predictions(df):
    regions = ["East", "West", "Midwest", "South"]
    cmap    = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(
        "2026 NCAA Tournament — Round of 32 Predicted Win Probabilities\n"
        "Green = high confidence favorite  |  Yellow = toss-up  |  ★ = actual upset",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, region in zip(axes, regions):
        rdf = df[df["Region"] == region].reset_index(drop=True)

        for i, row in rdf.iterrows():
            y        = len(rdf) - 1 - (i % len(rdf))
            prob     = row["FavProb"]
            bar_val  = prob if prob is not None else 0.5
            color    = cmap(bar_val) if prob is not None else "#cccccc"
            prob_str = f"{prob:.0%}" if prob is not None else "N/A"

            ax.barh(y, bar_val, color=color, height=0.72,
                    edgecolor="white", linewidth=0.5)

            ax.text(0.02, y, f"#{row['FavSeed']} {row['FavName']}",
                    va="center", ha="left", fontsize=8,
                    fontweight="bold", color="black")

            ax.text(0.98, y, f"#{row['DogSeed']} {row['DogName']}",
                    va="center", ha="right", fontsize=7.5, color="#333333",
                    transform=ax.get_yaxis_transform())

            text_x = bar_val - 0.02 if bar_val > 0.15 else bar_val + 0.02
            ax.text(text_x, y, prob_str, va="center",
                    ha="right" if bar_val > 0.15 else "left",
                    fontsize=8, color="white", fontweight="bold")

            if row["ActualUpset"]:
                ax.text(1.02, y, "★", va="center", ha="left",
                        fontsize=10, color="#DC2626", fontweight="bold",
                        transform=ax.get_yaxis_transform())

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(rdf) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["50%", "75%", "100%"], fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{region} Region", fontweight="bold", fontsize=11, pad=8)
        ax.set_xlabel("P(Top team wins)", fontsize=8)

    patches = [
        mpatches.Patch(color=cmap(0.88), label="High confidence (>85%)"),
        mpatches.Patch(color=cmap(0.70), label="Moderate (65–85%)"),
        mpatches.Patch(color=cmap(0.52), label="Toss-up (<65%)"),
        mpatches.Patch(color="#DC2626",  label="★ Actual upset"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "2026_r2_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.show()


def print_r2_summary(df):
    print("\n2026 Round of 32 Predictions")
    print("="*70)
    print(f"{'Region':<10} {'Matchup':<38} {'Model Prob':>10} {'Result':>10}")
    print("-"*70)
    correct = 0
    for _, r in df.iterrows():
        prob_str = f"{r['FavProb']:.1%}" if r["FavProb"] is not None else "N/A"
        result   = "UPSET ★" if r["ActualUpset"] else "Fav won"
        matchup  = f"#{r['FavSeed']} {r['FavName']:12s} vs #{r['DogSeed']} {r['DogName']}"
        print(f"{r['Region']:<10} {matchup:<38} {prob_str:>10} {result:>10}")
        if r["FavProb"] is not None:
            model_correct = (r["FavProb"] >= 0.5) == r["FavWon"]
            if model_correct:
                correct += 1
    total = len(df[df["FavProb"].notna()])
    print(f"\nR2 model accuracy: {correct}/{total} ({correct/total:.1%})")



def build_s16_predictions(feat26, scaler, model):
    rows = []
    for entry in S16_MATCHUPS:
        region, name_a, id_a, seed_a, name_b, id_b, seed_b = entry
        prob       = predict_matchup(id_a, seed_a, id_b, seed_b, feat26, scaler, model)
        actual_win = S16_ACTUAL_WINNERS.get((region, id_a))
        fav_won    = (actual_win == id_a)
        is_upset   = (region, id_a) in S16_UPSETS
        rows.append({
            "Region":      region,
            "FavName":     name_a,  "FavID":   id_a,  "FavSeed": seed_a,
            "DogName":     name_b,  "DogID":   id_b,  "DogSeed": seed_b,
            "FavProb":     prob,
            "FavWon":      fav_won,
            "ActualUpset": is_upset,
        })
    return pd.DataFrame(rows)


def plot_s16_predictions(df):
    regions = ["East", "West", "Midwest", "South"]
    cmap    = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    fig.suptitle(
        "2026 NCAA Tournament — Sweet 16 Predicted Win Probabilities\n"
        "Green = high confidence favorite  |  Yellow = toss-up  |  ★ = actual upset",
        fontsize=12, fontweight="bold", y=1.05
    )

    for ax, region in zip(axes, regions):
        rdf = df[df["Region"] == region].reset_index(drop=True)

        for i, row in rdf.iterrows():
            y       = len(rdf) - 1 - i
            prob    = row["FavProb"]
            fav_won = row["FavWon"]

            if fav_won:
                left_name, left_seed   = row["FavName"], row["FavSeed"]
                right_name, right_seed = row["DogName"], row["DogSeed"]
                bar_val = prob if prob is not None else 0.5
            else:
                left_name, left_seed   = row["DogName"], row["DogSeed"]
                right_name, right_seed = row["FavName"], row["FavSeed"]
                bar_val = (1 - prob) if prob is not None else 0.5

            color    = cmap(bar_val) if prob is not None else "#cccccc"
            prob_str = f"{bar_val:.0%}" if prob is not None else "N/A"

            ax.barh(y, bar_val, color=color, height=0.72,
                    edgecolor="white", linewidth=0.5)

            ax.text(0.02, y, f"#{left_seed} {left_name}",
                    va="center", ha="left", fontsize=8,
                    fontweight="bold", color="black")

            ax.text(0.98, y, f"#{right_seed} {right_name}",
                    va="center", ha="right", fontsize=7.5, color="#333333",
                    transform=ax.get_yaxis_transform())

            text_x = bar_val - 0.02 if bar_val > 0.15 else bar_val + 0.02
            ax.text(text_x, y, prob_str, va="center",
                    ha="right" if bar_val > 0.15 else "left",
                    fontsize=8, color="white", fontweight="bold")

            if row["ActualUpset"]:
                ax.text(1.02, y, "★", va="center", ha="left",
                        fontsize=10, color="#DC2626", fontweight="bold",
                        transform=ax.get_yaxis_transform())

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(rdf) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["50%", "75%", "100%"], fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{region} Region", fontweight="bold", fontsize=11, pad=8)
        ax.set_xlabel("P(Winner wins — as predicted by model)", fontsize=8)

    patches = [
        mpatches.Patch(color=cmap(0.88), label="High confidence (>85%)"),
        mpatches.Patch(color=cmap(0.70), label="Moderate (65–85%)"),
        mpatches.Patch(color=cmap(0.52), label="Toss-up (<65%)"),
        mpatches.Patch(color="#DC2626",  label="★ Actual upset"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "2026_s16_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {path}")
    plt.show()


def print_s16_summary(df):
    print("\n2026 Sweet 16 Predictions")
    print("="*70)
    print(f"{'Region':<10} {'Matchup':<38} {'Model Prob':>10} {'Result':>10}")
    print("-"*70)
    correct = 0
    for _, r in df.iterrows():
        prob_str = f"{r['FavProb']:.1%}" if r["FavProb"] is not None else "N/A"
        result   = "UPSET ★" if r["ActualUpset"] else "Fav won"
        matchup  = f"#{r['FavSeed']} {r['FavName']:12s} vs #{r['DogSeed']} {r['DogName']}"
        print(f"{r['Region']:<10} {matchup:<38} {prob_str:>10} {result:>10}")
        if r["FavProb"] is not None:
            if (r["FavProb"] >= 0.5) == r["FavWon"]:
                correct += 1
    total = len(df[df["FavProb"].notna()])
    print(f"\nS16 model accuracy: {correct}/{total} ({correct/total:.1%})")



def build_late_round(matchups, actual_winners, upsets, feat26, scaler, model):
    """Generic builder for E8, FF, and Championship."""
    rows = []
    for entry in matchups:
        region, name_a, id_a, seed_a, name_b, id_b, seed_b = entry
        prob     = predict_matchup(id_a, seed_a, id_b, seed_b, feat26, scaler, model)
        fav_won  = actual_winners.get((region, id_a)) == id_a
        is_upset = (region, id_a) in upsets
        rows.append({
            "Region":      region,
            "FavName":     name_a, "FavID":   id_a, "FavSeed": seed_a,
            "DogName":     name_b, "DogID":   id_b, "DogSeed": seed_b,
            "FavProb":     prob,   "FavWon":  fav_won, "ActualUpset": is_upset,
        })
    return pd.DataFrame(rows)


def plot_late_round(df, title, filename, ncols=None):
    """
    Generic plot for E8 (4 games), FF (2 games), Championship (1 game).
    Uses a single row of subplots, one per game.
    """
    cmap   = plt.cm.RdYlGn
    n      = len(df)
    ncols  = ncols or n
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)

    for ax, (_, row) in zip(axes, df.iterrows()):
        prob    = row["FavProb"]
        fav_won = row["FavWon"]

        if fav_won:
            left_name, left_seed   = row["FavName"], row["FavSeed"]
            right_name, right_seed = row["DogName"], row["DogSeed"]
            bar_val = prob if prob is not None else 0.5
        else:
            left_name, left_seed   = row["DogName"], row["DogSeed"]
            right_name, right_seed = row["FavName"], row["FavSeed"]
            bar_val = (1 - prob) if prob is not None else 0.5

        color    = cmap(bar_val) if prob is not None else "#cccccc"
        prob_str = f"{bar_val:.0%}" if prob is not None else "N/A"

        ax.barh(0, bar_val, color=color, height=0.5,
                edgecolor="white", linewidth=0.5)

        # Winner (left, bold)
        ax.text(0.02, 0.32, f"#{left_seed} {left_name}",
                va="center", ha="left", fontsize=10,
                fontweight="bold", color="black")

        # Loser (left, below)
        ax.text(0.02, -0.32, f"#{right_seed} {right_name}",
                va="center", ha="left", fontsize=9, color="#555555")

        # Probability on bar
        text_x = bar_val - 0.03 if bar_val > 0.2 else bar_val + 0.03
        ax.text(text_x, 0, prob_str, va="center",
                ha="right" if bar_val > 0.2 else "left",
                fontsize=10, color="white", fontweight="bold")

        # Upset star
        if row["ActualUpset"]:
            ax.text(0.98, 0.32, "★ UPSET", va="center", ha="right",
                    fontsize=9, color="#DC2626", fontweight="bold",
                    transform=ax.get_yaxis_transform())

        # Champion crown for championship game
        if "Championship" in row["Region"]:
            ax.text(0.98, -0.32, "🏆 CHAMPION", va="center", ha="right",
                    fontsize=9, color="#D97706", fontweight="bold",
                    transform=ax.get_yaxis_transform())

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["50%", "75%", "100%"], fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_title(row["Region"], fontsize=9, pad=6)
        ax.set_xlabel("P(Winner wins)", fontsize=8)

    patches = [
        mpatches.Patch(color=cmap(0.88), label="High confidence"),
        mpatches.Patch(color=cmap(0.60), label="Toss-up"),
        mpatches.Patch(color="#DC2626",  label="★ Actual upset"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


def print_late_round(df, label):
    print(f"\n{label}")
    print("="*65)
    correct = 0
    for _, r in df.iterrows():
        prob_str = f"{r['FavProb']:.1%}" if r["FavProb"] is not None else "N/A"
        result   = "UPSET ★" if r["ActualUpset"] else "Fav won"
        matchup  = f"#{r['FavSeed']} {r['FavName']} vs #{r['DogSeed']} {r['DogName']}"
        print(f"  {r['Region']:<20} {matchup:<35} {prob_str:>7}  {result}")
        if r["FavProb"] is not None and (r["FavProb"] >= 0.5) == r["FavWon"]:
            correct += 1
    total = len(df[df["FavProb"].notna()])
    if total:
        print(f"  Accuracy: {correct}/{total} ({correct/total:.1%})")


if __name__ == "__main__":
    feat26        = build_2026_features()
    scaler, model = load_model()

    print("\n--- ROUND 1 ---")
    df_r1 = build_predictions(feat26, scaler, model)
    print_summary(df_r1)
    plot_predictions(df_r1)

    print("\n--- ROUND OF 32 ---")
    df_r2 = build_r2_predictions(feat26, scaler, model)
    print_r2_summary(df_r2)
    plot_r2_predictions(df_r2)

    print("\n--- SWEET 16 ---")
    df_s16 = build_s16_predictions(feat26, scaler, model)
    print_s16_summary(df_s16)
    plot_s16_predictions(df_s16)

    print("\n--- ELITE EIGHT ---")
    df_e8 = build_late_round(E8_MATCHUPS, E8_ACTUAL_WINNERS, E8_UPSETS, feat26, scaler, model)
    print_late_round(df_e8, "2026 Elite Eight Predictions")
    plot_late_round(df_e8,
                    "2026 NCAA Tournament — Elite Eight Predicted Win Probabilities",
                    "2026_e8_predictions.png")

    print("\n--- FINAL FOUR ---")
    df_ff = build_late_round(FF_MATCHUPS, FF_ACTUAL_WINNERS, FF_UPSETS, feat26, scaler, model)
    print_late_round(df_ff, "2026 Final Four Predictions")
    plot_late_round(df_ff,
                    "2026 NCAA Tournament — Final Four Predicted Win Probabilities",
                    "2026_ff_predictions.png")

    print("\n--- CHAMPIONSHIP ---")
    df_ch = build_late_round(CHAMP_MATCHUPS, CHAMP_ACTUAL_WINNERS, CHAMP_UPSETS, feat26, scaler, model)
    print_late_round(df_ch, "2026 Championship Prediction")
    plot_late_round(df_ch,
                    "2026 NCAA Championship — Michigan vs UConn",
                    "2026_championship_prediction.png", ncols=1)

    print("\n✓ 2026 predictions complete.")