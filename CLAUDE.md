# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Overview

ML system that forecasts every NBA game of the season (scores + win probabilities) and re-forecasts the whole calendar twice a day via GitHub Actions as results, rosters and injuries change. Live at harisbyhaqi.github.io/nba-predictor (static site in `docs/`, served by GitHub Pages).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys. All data comes from ESPN's public JSON API (`site.api.espn.com`).

## Pipeline (run in order)

```bash
python src/data_download.py   # ESPN schedule → data/schedule.csv; new box scores → data/raw/games_raw.csv + player_games.csv
python src/preprocess.py      # EWM form, opponent defense, rest, home/away win % → data/processed/games_preprocessed.csv
python src/train.py           # time-based hold-out → models/metrics.json; refit on all → models/points_rf.joblib
python src/rosters.py         # snapshot rosters + injuries → data/rosters.json, injuries.json; diffs → data/changes.json
python src/update_daily.py    # re-forecast the season → docs/season.json (optional arg: YYYY-MM-DD as "today")
python src/evaluate.py        # print hold-out MAE/RMSE/win accuracy/margin σ
```

Preview the site: `python -m http.server -d docs` → http://localhost:8000

## Architecture

- `src/espn.py` — ESPN client: schedule, box scores, injuries, rosters; ESPN→NBA abbreviation map (NY→NYK, SA→SAS, GS→GSW, NO→NOP, UTAH→UTA, WSH→WAS); team names/IDs/conferences. Seasons are labelled by end year (2026-27 = 2027).
- `src/data_download.py` — refreshes `data/schedule.csv` (current + previous season; past season cached once nothing is left to play) and appends finished games. Rows before 2026-03-30 came from nba_api (numeric `GAME_ID`); ESPN rows use `espn_<event_id>`. ESPN only fills dates after the last nba_api row.
- `src/utils.py` — `FEATURE_COLS`, `rolling_team_stats` (all features use prior games only), `team_states` (each team's state *after* its latest game, via a placeholder row), `nba_games_only` (drops preseason/All-Star/intl exhibitions).
- `src/preprocess.py` — `clean_games()` (shared with `update_daily.py`) + feature table.
- `src/train.py` — Random Forest regressor predicting one team's points; current season weighted 5x; holds out the most recent 15% by date, writes `models/metrics.json` (incl. `margin_sigma`), then refits on everything.
- `src/predict.py` — `predict_games()` batch scoring; `matchup_row()` fills `opp_*` features from the opponent's state.
- `src/simulate.py` — closed-form win probability `Φ(margin / margin_sigma)` (replaced the old 10k Monte Carlo).
- `src/rosters.py` — daily roster/injury snapshots + change log (signed / released / traded / injury_new / status_change / cleared). Fetch failures keep the previous snapshot.
- `src/update_daily.py` — season forecaster: predicts every upcoming game from current team states, applies injury + H2H offsets, freezes today's pre-tip predictions in `data/frozen_predictions.csv`, computes projected standings and accuracy, writes `docs/season.json`.
- `docs/` — vanilla JS site: Calendar (month grid + team filter + game cards), Standings (projected W-L), Accuracy (frozen picks vs results), Updates (change feed + injury report).

## Features

`pts_roll5`, `fgpct_roll5`, `fg3pct_roll5`, `reb_roll5`, `ast_roll5`, `tov_roll5`, `efg_roll5`, `tov_rate_roll5` (EWM span 5, carry across seasons), `rest_days` (from the schedule at prediction time, clipped 1–10), `home`, `home_win_pct` / `away_win_pct` (reset each season, shrunk toward .500), `win_pct_last10`, `opp_pts_allowed_roll5`, `opp_efg_allowed_roll5`.

## Injury adjustments

Only players averaging 15+ PPG (from `player_games.csv`: current season with ≥3 games, else prior season) who appeared in one of the team's last 10 games — longer absences are already reflected in team form. Points removed from the team's predicted score:

| PPG | Out / Doubtful | Questionable | Day-to-day |
|---|---|---|---|
| 30+ | 8.0 | 3.0 | 1.5 |
| 25–30 | 6.0 | 3.0 | 1.5 |
| 20–25 | 4.5 | 2.0 | 1.0 |
| 15–20 | 3.0 | 1.5 | 1.0 |

Out/Doubtful apply to every game before ESPN's return date, else the team's next 3 games. Questionable/Day-to-day apply to the next game only.

## GitHub Actions

`.github/workflows/update_predictions.yml`:
- 14:00 UTC (10 AM EDT / 9 AM EST): tests → download → preprocess → retrain → rosters → forecast → commit.
- 22:00 UTC (6 PM EDT / 5 PM EST): same minus the download (evening injury refresh).
- `models/points_rf.joblib` is gitignored and rebuilt every run; `models/metrics.json` is committed.
- Each commit redeploys GitHub Pages. Daily commits also stop GitHub from auto-disabling the schedule for inactivity (that happened in mid-2026: re-enable with `gh workflow enable update_predictions.yml`).

## Tests

```bash
pytest tests/
pytest tests/test_features.py      # single test file
```

Fixtures in `tests/fixtures/` are trimmed real ESPN responses.
