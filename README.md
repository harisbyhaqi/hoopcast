# 🏀 NBA Game Predictor

A machine learning project that forecasts **NBA game scores** using the past two seasons of data.  
The system predicts **team points scored** and simulates outcomes to estimate win probabilities.

---

## 📌 Features
- Download historical NBA game data using `nba_api`
- Preprocess data: rolling averages, home/away flag, rest days
- Train a regression model (baseline: Random Forest)
- Predict final scores for upcoming games
- Simulate game outcomes with Monte Carlo sampling
- Evaluate model accuracy (MAE, RMSE, win prediction %)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/nba-score-predictor.git
cd nba-score-predictor
```

### 2. Set up environment
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


### 3. Download data
Fetch the last two seasons:
```bash
python src/data_download.py
```
This saves raw data in data/raw/games_raw.csv.


### 4. Preprocess
```bash
python src/preprocess.py
```
Creates data/processed/games_preprocessed.csv.


### 5. Train the model
```bash
python src/train.py
```
Trains a regression model and saves it in models/points_rf.joblib.


### 6. Predict a game
```
Example usage:
from src.predict import predict_game

Example features (replace with real preprocessed features)
teamA_feat = {"pts_last5": 110, "rest_days": 2, "home": 1}
teamB_feat = {"pts_last5": 107, "rest_days": 1, "home": 0}

scoreA = predict_game(teamA_feat, teamB_feat)
scoreB = predict_game(teamB_feat, teamA_feat)

print(f"Predicted: Team A {scoreA:.1f} - Team B {scoreB:.1f}")
```

### 📊 Project Structure
```
nba-score-predictor/
├─ data/               # raw & processed game data
│  ├─ raw/
│  └─ processed/
├─ models/             # saved ML models
├─ notebooks/          # experiments
├─ src/                # main source code
│  ├─ data_download.py
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ predict.py
│  ├─ simulate.py
│  ├─ evaluate.py
│  └─ utils.py
├─ tests/              # pytest unit tests
├─ requirements.txt
└─  README.md
```
