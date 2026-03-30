import os
from datetime import date
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def current_seasons() -> list[str]:
    """Return [current_season, previous_season] in 'YYYY-YY' format based on today's date."""
    today = date.today()
    # NBA season starts in October; before October we're still in the prior season
    season_start = today.year if today.month >= 10 else today.year - 1
    current  = f"{season_start}-{str(season_start + 1)[-2:]}"
    previous = f"{season_start - 1}-{str(season_start)[-2:]}"
    return [current, previous]


def download_seasons(seasons):
    frames = []
    for season in seasons:
        print(f"Fetching season {season}")
        lg = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = lg.get_data_frames()[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    seasons = current_seasons()
    print(f"Downloading seasons: {seasons}")
    df = download_seasons(seasons)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "games_raw.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
