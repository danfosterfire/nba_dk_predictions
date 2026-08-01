# NBA Best Ball Points Prediction

Predict a player's performance in a popular NBA best-ball fantasy contest (`dk_pts`) for 
**each game** of an upcoming season, using only information available before the
season starts.

## The problem

### Target

Scoring is a linear function of the box score, plus a nonlinear bonus:

```
dk_pts = 1.0·PTS + 0.5·FG3M + 1.25·REB + 1.5·AST + 2.0·STL + 2.0·BLK − 0.5·TOV + bonus
bonus  = 1.5 for a double-double, 4.5 for a triple-double
```

### The prediction-time constraint

This is the defining constraint of the project. At prediction time — before the season
starts — we know:

1. The **season schedule** — who plays whom, when, home or away.
2. **Season-start rosters** — definitively which team each player is on, and each team's
   identity. Offseason moves are known.
3. The **previous-season stats** of every team and player.

We do **not** know: **within-season** roster changes (mid-season trades), current-season
minutes, injuries, or in-season form.

### This project

This is the AI-assisted version 2 of my 
[hand-coded effort using the R stack last year](https://github.com/danfosterfire/nba_stats). 

This version:

1. Replaces an R/tidyverse stack with a Python stack, for fun and learning
2. Refines and improves the per-boxscore-component Bayesian models (written in Stan)
3. Adds a major new feature: Simulated drafts and seasons in order to leverage 
the full posteriors available from the models and test out more complex drafting 
strategies
4. Communicates process and findings via a streamlit dashboard