# NBA Game Predictor

A machine learning system that forecasts every NBA game of the season (scores and win probabilities) and re-forecasts the whole calendar twice a day via GitHub Actions.

Live predictions: [harisbyhaqi.github.io/nba-predictor](https://harisbyhaqi.github.io/nba-predictor)

---

## How It Works

**Every morning (10 AM ET)** the workflow:

1. **Refreshes the season schedule** and downloads box scores for newly finished games from ESPN
2. **Preprocesses** the data: rolling team form, opponent defense, rest days, home/away records
3. **Retrains** the Random Forest model, measuring accuracy on the most recent held-out games
4. **Snapshots rosters and the injury report**, logging any signings, trades, releases and status changes
5. **Re-forecasts every remaining game** of the season from each team's latest form, with injury adjustments
6. **Commits** the results, which redeploys the GitHub Pages site

**Every evening (6 PM ET)** it re-runs steps 2–6, so late injury news is reflected before tip-off.

The site has four views:
- **Calendar:** every game of the season, filterable by team
- **Standings:** current record plus projected final W-L
- **Accuracy:** each game is scored against the prediction frozen before tip-off
- **Updates:** a roster/injury change feed and the current injury report

---

## Model

**Algorithm:** Random Forest Regressor (predicts points scored per team per game)

**Features used per team:**

| Feature | Description |
|---|---|
| `pts_roll5`, `fgpct_roll5`, `fg3pct_roll5`, `reb_roll5`, `ast_roll5`, `tov_roll5` | Exponentially weighted recent averages (span 5) |
| `efg_roll5`, `tov_rate_roll5` | Effective FG% and turnover rate |
| `rest_days` | Days since last game (1–10), taken from the schedule, so back-to-backs are known ahead of time |
| `home` | 1 if home team, 0 if away |
| `home_win_pct`, `away_win_pct` | This season's home/away win %, shrunk toward .500 early on |
| `win_pct_last10` | Win % over the last 10 games |
| `opp_pts_allowed_roll5`, `opp_efg_allowed_roll5` | The opponent's recent defense |

**Season weighting:** Current season games count 5x more than older games.

**Win probabilities:** The actual margin is treated as normally distributed around the predicted margin. The spread (σ) is measured on held-out games each time the model retrains.

**Season start:** Recent-form stats carry over from the end of last season, so opening-night predictions use each team's latest form. They adapt as new games come in.

---

## Injury Adjustments

Players averaging **15+ PPG** who have been playing recently lower their team's predicted score when listed on the injury report:

| PPG | Out / Doubtful | Questionable | Day-to-day |
|---|---|---|---|
| 30+ | −8.0 | −3.0 | −1.5 |
| 25–30 | −6.0 | −3.0 | −1.5 |
| 20–25 | −4.5 | −2.0 | −1.0 |
| 15–20 | −3.0 | −1.5 | −1.0 |

Out/Doubtful players count against every game until ESPN's estimated return date (or the next 3 games if none is given). Questionable/Day-to-day players count against the next game only. Players who have already missed 10+ games are skipped, because their absence is already reflected in team form.

---

## Data Sources

| Source | Used for | Auth |
|---|---|---|
| ESPN public API (`site.api.espn.com`) | Schedule, box scores, rosters, injury reports | None |

> Earlier versions used BallDontLie (its free tier no longer serves box scores) and `cdn.nba.com` / `stats.nba.com` (which block CI runners). Data before 2026-03-30 came from `nba_api`.

---

## Setup

```bash
git clone https://github.com/harisbyhaqi/nba-predictor.git
cd nba-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/data_download.py   # schedule + new box scores
python src/preprocess.py      # features
python src/train.py           # model + models/metrics.json
python src/rosters.py         # roster & injury snapshot
python src/update_daily.py    # full-season forecast → docs/season.json
python src/evaluate.py        # hold-out accuracy

python -m http.server -d docs # preview at http://localhost:8000
pytest tests/
```

---

## Project Structure

```
nba-predictor/
├── data/
│   ├── raw/                     # games_raw.csv (team-games), player_games.csv (player box scores)
│   ├── processed/               # games_preprocessed.csv (with features)
│   ├── schedule.csv             # every game, past and future
│   ├── rosters.json             # latest roster snapshot
│   ├── injuries.json            # latest injury report
│   ├── changes.json             # roster/injury change log
│   └── frozen_predictions.csv   # pre-tip predictions used for accuracy
├── docs/                        # GitHub Pages site (index.html, app.js, style.css, season.json)
├── models/                      # metrics.json (points_rf.joblib is rebuilt each run, not committed)
├── src/
│   ├── espn.py                  # ESPN API client
│   ├── data_download.py         # schedule + incremental box scores
│   ├── preprocess.py            # feature engineering
│   ├── train.py                 # model training + hold-out metrics
│   ├── predict.py               # matchup scoring
│   ├── simulate.py              # win probability
│   ├── rosters.py               # roster/injury snapshots + change log
│   ├── update_daily.py          # full-season forecaster
│   ├── evaluate.py              # hold-out evaluation
│   └── utils.py                 # features and team state helpers
├── tests/
└── .github/workflows/update_predictions.yml
```

---

## GitHub Actions Setup

No secrets are needed. On a fork:

1. Enable GitHub Pages from the `docs/` folder on the `main` branch
2. Enable the workflow under the Actions tab. GitHub disables scheduled workflows after 60 days without repo activity; re-enable it with `gh workflow enable update_predictions.yml`
