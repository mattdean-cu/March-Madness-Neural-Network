"""
08_simulate_tournament.py
=========================
Simulates the full 2026 NCAA Tournament bracket using the trained
neural network to estimate each team's probability of winning
the national championship.

Method: Monte Carlo simulation
  - Run N simulations of the full bracket
  - In each simulation, every game is decided probabilistically
    (a random draw against the model's predicted win probability)
  - Count how often each team wins the championship
  - Report championship win % for all 64 teams

This is more realistic than multiplying probabilities along a fixed
path because it accounts for all possible opponents a team might face.

Output:
  - Console table ranked by championship probability
  - figures/championship_odds.png  — bar chart of top 20 teams
  - figures/win_probability_heatmap.png — probability of reaching each round

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

N_SIMULATIONS = 10000   # increase for more accuracy, decrease for speed
np.random.seed(42)

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

# ── Full 2026 bracket in bracket order ────────────────────────────────────────
# Each region is listed as seeds 1-16 in bracket order
# Bracket order pairs: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
# This order matters for correctly simulating who plays who in R2+
BRACKET_ORDER = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

REGIONS = {
    "East": [
        (1,"Duke",1181),(16,"Siena",1373),(8,"Ohio St",1326),(9,"TCU",1395),
        (5,"St. John's",1385),(12,"N Iowa",1320),(4,"Kansas",1242),(13,"Cal Baptist",1465),
        (6,"Louisville",1257),(11,"S Florida",1378),(3,"Michigan St",1277),(14,"N Dakota St",1295),
        (7,"UCLA",1417),(10,"UCF",1416),(2,"UConn",1163),(15,"Furman",1202),
    ],
    "West": [
        (1,"Arizona",1112),(16,"LIU Brooklyn",1254),(8,"Villanova",1437),(9,"Utah St",1429),
        (5,"Wisconsin",1458),(12,"High Point",1219),(4,"Arkansas",1116),(13,"Hawaii",1218),
        (6,"BYU",1140),(11,"Texas",1400),(3,"Gonzaga",1211),(14,"Kennesaw St",1244),
        (7,"Miami FL",1274),(10,"Missouri",1281),(2,"Purdue",1345),(15,"Queens NC",1474),
    ],
    "Midwest": [
        (1,"Michigan",1276),(16,"UMBC",1420),(8,"Georgia",1208),(9,"St Louis",1387),
        (5,"Texas Tech",1403),(12,"Akron",1103),(4,"Alabama",1104),(13,"Hofstra",1220),
        (6,"Tennessee",1397),(11,"Miami OH",1275),(3,"Virginia",1438),(14,"Wright St",1460),
        (7,"Kentucky",1246),(10,"Santa Clara",1365),(2,"Iowa St",1235),(15,"Tennessee St",1398),
    ],
    "South": [
        (1,"Florida",1196),(16,"Prairie View",1341),(8,"Clemson",1155),(9,"Iowa",1234),
        (5,"Vanderbilt",1435),(12,"McNeese St",1270),(4,"Nebraska",1304),(13,"Troy",1407),
        (6,"N Carolina",1314),(11,"VCU",1433),(3,"Illinois",1228),(14,"Penn",1335),
        (7,"St Mary's CA",1388),(10,"Texas A&M",1401),(2,"Houston",1222),(15,"Idaho",1225),
    ],
}

# Actual champion and final four for annotation
ACTUAL_CHAMPION  = 1276   # Michigan
ACTUAL_FINAL_FOUR = {1276, 1163, 1228, 1112}  # Michigan, UConn, Illinois, Arizona


# ── Model ──────────────────────────────────────────────────────────────────────
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


def load_model():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model  = MarchMadnessNN(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "nn_model.pt"), weights_only=True))
    model.eval()
    return scaler, model


# ── Build 2026 features ────────────────────────────────────────────────────────
def build_2026_features():
    reg        = pd.read_csv(os.path.join(DATA_DIR, "MRegularSeasonDetailedResults.csv"))
    massey     = pd.read_csv(os.path.join(DATA_DIR, "MMasseyOrdinals.csv"))
    seeds_hist = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))
    tourney    = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv"))
    reg26 = reg[reg["Season"]==2026].copy()

    w = reg26[["WTeamID","WScore","LScore","WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA",
               "WOR","WDR","WAst","WTO","WStl","WBlk","LFGM","LFGA"]].copy()
    w.columns = ["TeamID","PtsFor","PtsAgainst","FGM","FGA","FGM3","FGA3","FTM","FTA",
                 "OR","DR","Ast","TO","Stl","Blk","Opp_FGM","Opp_FGA"]
    l = reg26[["LTeamID","LScore","WScore","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA",
               "LOR","LDR","LAst","LTO","LStl","LBlk","WFGM","WFGA"]].copy()
    l.columns = ["TeamID","PtsFor","PtsAgainst","FGM","FGA","FGM3","FGA3","FTM","FTA",
                 "OR","DR","Ast","TO","Stl","Blk","Opp_FGM","Opp_FGA"]
    games = pd.concat([w,l],ignore_index=True)
    games["FGPct"]    = games["FGM"]/games["FGA"].replace(0,np.nan)
    games["FG3Pct"]   = games["FGM3"]/games["FGA3"].replace(0,np.nan)
    games["FTPct"]    = games["FTM"]/games["FTA"].replace(0,np.nan)
    games["OppFGPct"] = games["Opp_FGM"]/games["Opp_FGA"].replace(0,np.nan)
    games["RebTotal"] = games["OR"]+games["DR"]
    box = games.groupby("TeamID").agg(
        Games=("PtsFor","count"), AvgPtsFor=("PtsFor","mean"),
        AvgPtsAgainst=("PtsAgainst","mean"), AvgFGPct=("FGPct","mean"),
        AvgFG3Pct=("FG3Pct","mean"), AvgFTPct=("FTPct","mean"),
        AvgOppFGPct=("OppFGPct","mean"), AvgReb=("RebTotal","mean"),
        AvgOR=("OR","mean"), AvgTO=("TO","mean"), AvgStl=("Stl","mean"),
        AvgBlk=("Blk","mean"), AvgAst=("Ast","mean"),
    ).reset_index()
    box["AvgPointDiff"] = box["AvgPtsFor"]-box["AvgPtsAgainst"]
    box["Season"] = 2026

    wins   = reg26.groupby("WTeamID").size().reset_index(name="Wins").rename(columns={"WTeamID":"TeamID"})
    losses = reg26.groupby("LTeamID").size().reset_index(name="Losses").rename(columns={"LTeamID":"TeamID"})
    record = pd.merge(wins,losses,on="TeamID",how="outer").fillna(0)
    record["WinPct"] = record["Wins"]/(record["Wins"]+record["Losses"])
    record["Season"] = 2026

    wg = reg26[["WTeamID","DayNum","WScore","LScore"]].copy()
    wg["TeamID"]=wg["WTeamID"]; wg["Win"]=1; wg["PtDiff"]=wg["WScore"]-wg["LScore"]
    lg = reg26[["LTeamID","DayNum","WScore","LScore"]].copy()
    lg["TeamID"]=lg["LTeamID"]; lg["Win"]=0; lg["PtDiff"]=lg["LScore"]-lg["WScore"]
    all_g = pd.concat([wg[["TeamID","DayNum","Win","PtDiff"]],
                       lg[["TeamID","DayNum","Win","PtDiff"]]],ignore_index=True)
    all_g = all_g.sort_values(["TeamID","DayNum"])
    recent = (all_g.groupby("TeamID").tail(10).groupby("TeamID")
              .agg(RecentWinPct=("Win","mean"),RecentAvgPtDiff=("PtDiff","mean"))
              .reset_index())
    recent["Season"]=2026

    m26 = massey[(massey["Season"]==2026)&(massey["RankingDayNum"]==133)]
    pom = m26[m26["SystemName"]=="POM"][["TeamID","OrdinalRank"]].rename(columns={"OrdinalRank":"PomRank"})
    med = m26.groupby("TeamID")["OrdinalRank"].median().reset_index().rename(columns={"OrdinalRank":"MedianRank"})
    ranks = pd.merge(med,pom,on="TeamID",how="left")
    ranks["PomRank"] = ranks["PomRank"].fillna(ranks["MedianRank"])
    ranks["Season"]=2026

    winpct_map = record.set_index("TeamID")["WinPct"].to_dict()
    pom_map    = pom.set_index("TeamID")["PomRank"].to_dict()
    wo = reg26[["WTeamID","LTeamID"]].rename(columns={"WTeamID":"TeamID","LTeamID":"OppID"})
    lo = reg26[["LTeamID","WTeamID"]].rename(columns={"LTeamID":"TeamID","WTeamID":"OppID"})
    mo = pd.concat([wo,lo],ignore_index=True)
    mo["OppWinPct"]  = mo["OppID"].map(winpct_map)
    mo["OppPomRank"] = mo["OppID"].map(pom_map)
    sos = mo.groupby("TeamID").agg(
        SOS_AvgOppWinPct=("OppWinPct","mean"),
        SOS_AvgOppPomRank=("OppPomRank","mean")
    ).reset_index()
    sos["Season"]=2026

    t_wins = (tourney.groupby(["Season","WTeamID"]).size()
              .reset_index(name="RoundsWon").rename(columns={"WTeamID":"TeamID"}))
    appeared = seeds_hist[["Season","TeamID"]].copy()
    appeared = pd.merge(appeared,t_wins,on=["Season","TeamID"],how="left")
    appeared["RoundsWon"] = appeared["RoundsWon"].fillna(0).astype(int)
    exp_rows = []
    for tid in box["TeamID"].unique():
        prior = appeared[appeared["TeamID"]==tid]
        exp_rows.append({"TeamID":tid,"Season":2026,
                         "TourneyAppearances":len(prior),
                         "AvgPastRounds":prior["RoundsWon"].mean() if len(prior)>0 else 0.0})
    exp = pd.DataFrame(exp_rows)

    feat = box.copy()
    for df in [record[["Season","TeamID","Wins","Losses","WinPct"]],
               recent[["Season","TeamID","RecentWinPct","RecentAvgPtDiff"]],
               ranks[["Season","TeamID","PomRank","MedianRank"]],
               exp[["Season","TeamID","TourneyAppearances","AvgPastRounds"]],
               sos[["Season","TeamID","SOS_AvgOppWinPct","SOS_AvgOppPomRank"]]]:
        feat = pd.merge(feat,df,on=["Season","TeamID"],how="left")
    feat["SeedNum"] = np.nan
    return feat


# ── Win probability between two teams ─────────────────────────────────────────
def get_win_prob(id_a, seed_a, id_b, seed_b, feat_map, scaler, model):
    """Returns P(team_a beats team_b). Falls back to seed-based if features missing."""
    if id_a not in feat_map.index or id_b not in feat_map.index:
        # Fallback: lower seed wins with 65% probability
        return 0.65 if seed_a < seed_b else 0.35

    a = feat_map.loc[id_a, RAW_COLS].copy(); a["SeedNum"] = seed_a
    b = feat_map.loc[id_b, RAW_COLS].copy(); b["SeedNum"] = seed_b
    diff = np.nan_to_num((a.values - b.values).reshape(1,-1).astype(float), nan=0.0)
    with torch.no_grad():
        prob = model(torch.tensor(scaler.transform(diff), dtype=torch.float32)).item()
    return prob


# ── Simulate one full tournament ───────────────────────────────────────────────
def simulate_once(regions, feat_map, scaler, model, rng):
    """
    Simulates the entire tournament bracket once.
    Returns the TeamID of the champion.

    Bracket structure:
      - 4 regions of 16 teams each
      - Within each region: R1 (16->8), R2 (8->4), S16 (4->2), E8 (2->1)
      - Final Four: winner of East vs West, winner of Midwest vs South
      - Championship: Final Four winners
    """
    region_winners = {}

    for region_name, teams in regions.items():
        # teams is a list of (seed, name, id) in bracket order
        # Bracket order: slots pair as (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
        bracket = [(seed, tid) for seed, name, tid in teams]

        for _ in range(4):  # 4 rounds within region
            next_round = []
            for i in range(0, len(bracket), 2):
                seed_a, id_a = bracket[i]
                seed_b, id_b = bracket[i+1]
                prob_a = get_win_prob(id_a, seed_a, id_b, seed_b, feat_map, scaler, model)
                winner = (seed_a, id_a) if rng.random() < prob_a else (seed_b, id_b)
                next_round.append(winner)
            bracket = next_round

        region_winners[region_name] = bracket[0]  # (seed, id)

    # Final Four: East vs West, Midwest vs South
    ff_pairs = [
        (region_winners["East"],    region_winners["West"]),
        (region_winners["Midwest"], region_winners["South"]),
    ]
    ff_winners = []
    for (seed_a, id_a), (seed_b, id_b) in ff_pairs:
        prob_a = get_win_prob(id_a, seed_a, id_b, seed_b, feat_map, scaler, model)
        winner = (seed_a, id_a) if rng.random() < prob_a else (seed_b, id_b)
        ff_winners.append(winner)

    # Championship
    (seed_a, id_a), (seed_b, id_b) = ff_winners
    prob_a = get_win_prob(id_a, seed_a, id_b, seed_b, feat_map, scaler, model)
    champion = id_a if rng.random() < prob_a else id_b
    return champion


# ── Run full simulation ────────────────────────────────────────────────────────
def run_simulation(feat26, scaler, model, n=N_SIMULATIONS):
    feat_map = feat26.set_index("TeamID")
    rng      = np.random.default_rng(42)

    champ_counts = {}
    print(f"Running {n:,} simulations...")
    for i in range(n):
        champ = simulate_once(REGIONS, feat_map, scaler, model, rng)
        champ_counts[champ] = champ_counts.get(champ, 0) + 1
        if (i+1) % 2000 == 0:
            print(f"  {i+1:,} / {n:,} done")

    # Build results table
    # Collect all teams
    all_teams = {}
    for region, teams in REGIONS.items():
        for seed, name, tid in teams:
            all_teams[tid] = (name, seed, region)

    rows = []
    for tid, (name, seed, region) in all_teams.items():
        count = champ_counts.get(tid, 0)
        rows.append({
            "TeamID": tid,
            "Team":   name,
            "Seed":   seed,
            "Region": region,
            "ChampCount": count,
            "ChampPct": count / n * 100,
        })

    results = pd.DataFrame(rows).sort_values("ChampPct", ascending=False).reset_index(drop=True)
    results["Rank"] = results.index + 1
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_championship_odds(results):
    top20 = results.head(20).copy()

    colors = []
    for _, r in top20.iterrows():
        if r["TeamID"] == ACTUAL_CHAMPION:
            colors.append("#D97706")   # gold for actual champion
        elif r["TeamID"] in ACTUAL_FINAL_FOUR:
            colors.append("#2563EB")   # blue for final four
        else:
            colors.append("#6B7280")   # gray for others

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(top20)), top20["ChampPct"].values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    labels = [f"#{r['Seed']} {r['Team']} ({r['Region']})"
              for _, r in top20.iterrows()]
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels[::-1], fontsize=9)

    for i, (val, tid) in enumerate(zip(top20["ChampPct"].values[::-1],
                                        top20["TeamID"].values[::-1])):
        ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=8,
                fontweight="bold" if tid == ACTUAL_CHAMPION else "normal")

    ax.set_xlabel("Championship Win Probability (%)", fontsize=10)
    ax.set_title("2026 NCAA Tournament — Predicted Championship Win Probabilities\n"
                 f"Based on {N_SIMULATIONS:,} Monte Carlo simulations",
                 fontweight="bold", fontsize=12)

    patches = [
        mpatches.Patch(color="#D97706", label="Actual Champion (Michigan)"),
        mpatches.Patch(color="#2563EB", label="Actual Final Four"),
        mpatches.Patch(color="#6B7280", label="Other teams"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.set_xlim(0, top20["ChampPct"].max() * 1.15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "championship_odds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.show()


def print_results_table(results):
    print(f"\n{'Rank':<5} {'Team':<16} {'Seed':<5} {'Region':<12} {'Champ %':>8}  {'Actual'}")
    print("─"*60)
    for _, r in results.head(30).iterrows():
        champion_flag = " ← CHAMPION" if r["TeamID"] == ACTUAL_CHAMPION else ""
        ff_flag       = " ← Final Four" if r["TeamID"] in ACTUAL_FINAL_FOUR and r["TeamID"] != ACTUAL_CHAMPION else ""
        flag = champion_flag or ff_flag
        print(f"{r['Rank']:<5} {r['Team']:<16} #{r['Seed']:<4} {r['Region']:<12} {r['ChampPct']:>7.1f}%{flag}")

    # Summary stats
    champ_rank = results[results["TeamID"]==ACTUAL_CHAMPION]["Rank"].values[0]
    champ_pct  = results[results["TeamID"]==ACTUAL_CHAMPION]["ChampPct"].values[0]
    print(f"\n  Actual champion (Michigan) ranked #{champ_rank} with {champ_pct:.1f}% odds")
    ff_in_top8 = sum(1 for tid in ACTUAL_FINAL_FOUR
                     if results[results["TeamID"]==tid]["Rank"].values[0] <= 8)
    print(f"  Final Four teams in model's top 8: {ff_in_top8}/4")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model and building 2026 features...")
    scaler, model = load_model()
    feat26        = build_2026_features()

    results = run_simulation(feat26, scaler, model, n=N_SIMULATIONS)
    print_results_table(results)
    plot_championship_odds(results)

    # Save results to CSV for use in report
    out = os.path.join(DATA_DIR, "2026_championship_odds.csv")
    results.to_csv(out, index=False)
    print(f"\n[Saved] {out}")
    print("\n✓ Simulation complete.")