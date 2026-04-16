# March Madness Winner Prediction using Neural Networks - Matt Dean, Shona Doyle

## Overview
This project uses machine learning to predict winners of NCAA March Madness tournament games. The model is trained on historical NCAA basketball data and learns patterns from regular season performance, tournament seeds, and other team-level statistics in order to estimate the probability that one team will beat another.

The main goal of the project is to build a matchup-based prediction system. Instead of trying to predict an entire bracket at once, the model evaluates individual games and outputs a win probability for a given team matchup. These game-level predictions can then be used to simulate a full tournament bracket.

---

## Project Goals
- Build a machine learning pipeline for NCAA tournament prediction
- Use historical regular season and tournament data from Kaggle
- Engineer team-level and matchup-level features
- Train a neural network to predict game outcomes
- Compare neural network performance to simpler baseline models
- Simulate a tournament bracket using predicted game probabilities

---

## Dataset
https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data

This project uses the NCAA March Madness dataset from Kaggle. The dataset includes historical information for both men's and women's Division I college basketball teams, such as:

- Team names and IDs
- Tournament seeds
- Regular season game results
- Tournament game results
- Detailed box score statistics
- Rankings data
- Conference information
- Bracket structure data

For the first version of this project, the focus is on the men's dataset.

### Main files used
- `MTeams.csv`
- `MRegularSeasonDetailedResults.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneyDetailedResults.csv`
- `MNCAATourneySeeds.csv`
- `MMasseyOrdinals.csv`
- `MNCAATourneySlots.csv`

---

## Problem Formulation
This project treats tournament prediction as a **binary classification problem**.

Given two teams:
- Team A
- Team B

the model predicts:

- the probability that Team A beats Team B

This allows the project to predict any possible tournament matchup and later use those predictions to advance teams through a bracket.

---

## Approach

### 1. Team-level feature engineering
For each team in each season, regular season statistics are aggregated to create pre-tournament team profiles.

Examples of team-level features include:
- Win percentage
- Average points scored
- Average points allowed
- Average scoring margin
- Field goal percentage
- Three-point percentage
- Free throw percentage
- Rebounds per game
- Assists per game
- Turnovers per game
- Recent form
- Tournament seed
- Ranking information

### 2. Matchup-level feature engineering
For each possible tournament game, team-level features are converted into matchup features by taking differences between teams.

Examples:
- `seed_diff`
- `win_pct_diff`
- `scoring_margin_diff`
- `fg_pct_diff`
- `rebound_diff`
- `ranking_diff`

This helps the model learn relative team strength rather than memorizing team identities.

### 3. Model training
A neural network is trained on historical tournament matchups using pre-tournament team features. The output is a probability between 0 and 1 representing the chance that one team wins the matchup.

### 4. Bracket simulation
After training, the model can be used to predict tournament games round by round and simulate a complete March Madness bracket.

---

## Models
The primary model for this project is a neural network built with PyTorch.

Planned model comparisons include:
- Logistic Regression
- Random Forest
- Neural Network

This comparison helps evaluate whether the neural network provides meaningful improvement over simpler models.
