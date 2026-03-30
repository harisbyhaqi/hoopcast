# NBA Game Predictor

A machine learning system that forecasts NBA game scores and win probabilities, updated automatically every day via GitHub Actions.

Live predictions: [harisbyhaqi.github.io/nba-predictor](https://harisbyhaqi.github.io/nba-predictor)

---

## How It Works

Each day at 10 AM ET the workflow:

1. **Fetches new game results** from the [BallDontLie API](https://www.balldontlie.io) and appends them to the historical dataset
2. **Preprocesses** the data — computing rolling team stats, rest days, home/away flags, momentum signals
3. **Retrains** the Random Forest model on the updated dataset (current season weighted 8x over prior season)
4. **Generates predictions** for today's games using current-season-only team features, with injury adjustments applied
5. **Commits** the updated predictions back to the repo, which redeploys the GitHub Pages site

---

## Model

**Algorithm:** Random Forest Regressor (predicts points scored per team per game)

**Features used per team:**

| Feature | Description |
|---|---|
| `pts_roll5` | Rolling 5-game scoring average |
| `fgpct_roll5` | Rolling 5-game FG% |
| `fg3pct_roll5` | Rolling 5-game 3PT% |
| `reb_roll5` | Rolling 5-game rebounds |
| `ast_roll5` | Rolling 5-game assists |
| `tov_roll5` | Rolling 5-game turnovers |
| `rest_days` | Days since last game (clipped 1–10) |
| `home` | 1 if home team, 0 if away |
| `win_pct_last10` | Win % over last 10 games (momentum signal) |
| `pts_diff_roll5` | Rolling 5-game point differential (net rating proxy) |

**Season weighting:** Current season games count 8x more than prior season during training, so the model reflects how teams are playing *now* rather than who they were last year.

**Win probabilities** are generated via Monte Carlo simulation (10,000 runs) with Gaussian noise (σ = 8 pts) applied to the predicted scores.

---

## Injury Adjustments

Before finalising predictions, the system fetches the active injury report from ESPN and looks up each injured player's season PPG via BallDontLie.

For any player averaging **15+ PPG** who is listed as **Out or Doubtful**, a partial score adjustment is applied:

- **Out / Doubtful:** −30% of their season PPG subtracted from the team's predicted score
- **Questionable:** −12%
- **Day-to-day:** −8%

The other ~70% is left in place since teammates absorb usage when a player sits.

---

## Data Sources

| Source | Used for | Auth |
|---|---|---|
| [BallDontLie API](https://www.balldontlie.io) | Historical & daily game results | Free API key (`BALLDONTLIE_API_KEY` secret) |
| NBA CDN (`cdn.nba.com`) | Today's game schedule | None |
| ESPN unofficial API | Injury reports | None |

> **Why not `nba_api`?** `stats.nba.com` blocks requests from GitHub Actions runner IPs. BallDontLie serves the same data and works fine from CI.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/harisbyhaqi/nba-predictor.git
cd nba-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Get a BallDontLie API key

Sign up for free at [balldontlie.io](https://www.balldontlie.io), then set your key:

```bash
export BALLDONTLIE_API_KEY=your_key_here
```

### 3. Download game data

Fetches all games since the last date already in `games_raw.csv` (or full season history if starting fresh):

```bash
python src/data_download.py
```

### 4. Preprocess

```bash
python src/preprocess.py
```

### 5. Train

```bash
python src/train.py
```

### 6. Run predictions

```bash
python src/update_daily.py
```

Writes predictions to `docs/predictions.json`.

### 7. Evaluate

```bash
python src/evaluate.py
```

---

## Project Structure

```
nba-predictor/
├── data/
│   ├── raw/                  # games_raw.csv (one row per team per game)
│   └── processed/            # games_preprocessed.csv (with rolling features)
├── docs/
│   └── predictions.json      # today's predictions (served by GitHub Pages)
├── models/
│   └── points_rf.joblib      # trained Random Forest model
├── src/
│   ├── data_download.py      # BallDontLie API — incremental game data fetch
│   ├── preprocess.py         # feature engineering
│   ├── train.py              # model training
│   ├── predict.py            # single-game score prediction
│   ├── simulate.py           # Monte Carlo win probability simulation
│   ├── update_daily.py       # daily pipeline: schedule + injuries + predictions
│   ├── evaluate.py           # hold-out set evaluation
│   └── utils.py              # shared feature columns and rolling stat helpers
├── .github/
│   └── workflows/
│       └── update_predictions.yml
├── requirements.txt
└── README.md
```

---

## GitHub Actions Setup

The workflow runs daily at 14:00 UTC (10 AM ET). To enable it on your own fork:

1. Add a repository secret named `BALLDONTLIE_API_KEY` with your BallDontLie free API key
   - Settings → Secrets and variables → Actions → New repository secret
2. Enable GitHub Pages from the `docs/` folder on the `main` branch
3. The workflow will handle everything else automatically
