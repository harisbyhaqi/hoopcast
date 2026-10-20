# Hoopcast

Hoopcast is a machine learning system that forecasts every NBA game of the season (scores and win probabilities) and re-forecasts the whole calendar via GitHub Actions (twice a day, plus hourly after games go final).

Live predictions: [harisbyhaqi.github.io/hoopcast](https://harisbyhaqi.github.io/hoopcast)

---

## How It Works

**Every morning (10 AM ET)** the workflow:

1. **Refreshes the season schedule** and downloads box scores for newly finished games from ESPN
2. **Preprocesses** the data: rolling team form, opponent defense, rest days, home/away records
3. **Retrains** the Random Forest model, measuring accuracy on the most recent held-out games
4. **Snapshots rosters and the injury report**, logging any signings, trades, releases and status changes
5. **Rates every roster** from player DARKO DPM (offense + defense)
6. **Re-forecasts every remaining game** of the season, blending roster strength with current form and adjusting for injuries
7. **Commits** the results, which redeploys the GitHub Pages site

**Every evening (6 PM ET)** it re-runs the full pipeline, so late injury news is reflected before tip-off.

**Every hour from 3 PM to 3 AM ET** it checks for games that just went final. When one has, it records the result, retrains, scores the frozen pick (how far off the score was) and updates the standings; otherwise it stops after the check.

The site has four views:
- **Games:** every game of the season, filterable by team
- **Standings:** current record plus projected final W-L
- **Accuracy:** each game is scored against the prediction frozen before tip-off
- **Updates:** a roster/injury change feed and the current injury report

---

## Model

Every game blends two views of each team:

**1. Who's on the roster (team rating)**
- Each player on today's ESPN roster is rated with **[DARKO](https://www.darko.app) DPM** (Daily Plus-Minus, by Kostya Medvedovsky): points per 100 possessions vs. an average player. It's built from box scores, on/off-court impact and tracking data, it splits into offense (O-DPM) and defense (D-DPM), and it's updated after every game. So defense that never shows up as a steal or block still counts, and a player in good or bad form moves daily.
- **Fallback:** for players DARKO doesn't list, and for everyone if darko.app can't be reached, the model uses Box Plus/Minus mapped onto the DPM scale. That's Basketball-Reference BPM for finished seasons, and this season an estimate from ESPN box scores (a regression of per-36 stats onto Basketball-Reference BPM, fitted on last season). Rookies and overseas signings get −1.5.
- **Depth chart:** each player's minutes are half his usual minutes over recent seasons and half the slot his rating earns on the roster (best players play most). A star coming back from injury still starts, and late-season garbage time doesn't make a bench player a starter. The rotation fills 240 minutes. Team rating = minutes-weighted DPM ≈ net rating per 100 possessions.
- **Preseason**, it's blended 50/50 with **sportsbook win totals** (`data/preseason_win_totals.csv`, the average of Fanatics and Covers lines, Oct 2026). Win totals see what ratings can't: rookies, new roles, and players returning from injury.

**2. How they're playing (Random Forest)**

Predicts each team's points from **this season's results only**. Form resets every season, so last year's team doesn't decide this year's games.

| Feature | Description |
|---|---|
| `pts_roll5`, `fgpct_roll5`, `fg3pct_roll5`, `reb_roll5`, `ast_roll5`, `tov_roll5` | Exponentially weighted recent averages (span 5) |
| `efg_roll5`, `tov_rate_roll5` | Effective FG% and turnover rate |
| `rest_days` | Days since last game (1–10), taken from the schedule, so back-to-backs are known ahead of time |
| `home` | 1 if home team, 0 if away |
| `home_win_pct`, `away_win_pct` | This season's home/away win %, shrunk toward .500 early on |
| `win_pct_last10` | Win % over the last 10 games |
| `opp_pts_allowed_roll5`, `opp_efg_allowed_roll5` | The opponent's recent defense |

**Blend:** form weight = games played / (games played + 20). That's 0% on opening night, 50% after 20 games, and about 80% by the All-Star break. The roster side adds home-court advantage, measured from the last two seasons (about +1.7 points).

**Win probabilities:** The actual margin is treated as normally distributed around the predicted margin. The spread (σ) is measured on held-out games each time the model retrains.

---

## Injury Adjustments

An injured player drops out of the rotation, everyone below him moves up a slot (or a replacement-level player fills in if the roster runs out), and the team rating is recomputed. Losing Jokić costs Denver far more than losing a bench guard. The minutes each status removes:

| Status | Minutes removed |
|---|---|
| Out / Doubtful | 100% |
| Questionable | 50% |
| Day-to-day | 20% |

Out/Doubtful players count against every game until ESPN's estimated return date (or the next 3 games if none is given). Questionable/Day-to-day players count against the next game only.

---

## Data Sources

| Source | Used for | Auth |
|---|---|---|
| ESPN public API (`site.api.espn.com`) | Schedule, box scores, rosters, injury reports | None |
| [DARKO](https://www.darko.app) | Player DPM (offense + defense) and projected minutes, daily (last pull saved in `data/darko.csv`) | None |
| [Basketball-Reference](https://www.basketball-reference.com) advanced stats | Fallback player Box Plus/Minus for finished seasons (cached in `data/bpm/`) | None |
| Fanatics / Covers win totals | Preseason team expectations (one-time, `data/preseason_win_totals.csv`) | None |

> Earlier versions used BallDontLie (its free tier no longer serves box scores) and `cdn.nba.com` / `stats.nba.com` (which block CI runners). Data before 2026-03-30 came from `nba_api`.

---

## Setup

```bash
git clone https://github.com/harisbyhaqi/hoopcast.git
cd hoopcast
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/data_download.py   # schedule + new box scores
python src/preprocess.py      # features
python src/train.py           # model + models/metrics.json
python src/rosters.py         # roster & injury snapshot
python src/ratings.py         # player DARKO DPM → team ratings
python src/update_daily.py    # full-season forecast → docs/season.json
python src/evaluate.py        # hold-out accuracy

python -m http.server -d docs # preview at http://localhost:8000
pytest tests/
```

---

## Project Structure

```
hoopcast/
├── data/
│   ├── raw/                     # games_raw.csv (team-games), player_games.csv (full player box-score lines)
│   ├── processed/               # games_preprocessed.csv (with features)
│   ├── schedule.csv             # every game, past and future
│   ├── rosters.json             # latest roster snapshot
│   ├── injuries.json            # latest injury report
│   ├── changes.json             # roster/injury change log
│   ├── frozen_predictions.csv   # pre-tip predictions used for accuracy
│   ├── darko.csv                # last DARKO DPM pull
│   ├── bpm/                     # cached Basketball-Reference advanced stats per season (fallback)
│   ├── preseason_win_totals.csv # sportsbook win totals (preseason prior)
│   └── team_ratings.json        # roster rating, market rating, blended prior, rotation
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
│   ├── ratings.py               # player DPM → roster-based team ratings
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
