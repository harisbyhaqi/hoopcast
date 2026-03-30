import os
import time
from datetime import date
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}


def current_seasons() -> list[str]:
    """Return [current_season, previous_season] in 'YYYY-YY' format based on today's date."""
    today = date.today()
    # NBA season starts in October; before October we're still in the prior season
    season_start = today.year if today.month >= 10 else today.year - 1
    current  = f"{season_start}-{str(season_start + 1)[-2:]}"
    previous = f"{season_start - 1}-{str(season_start)[-2:]}"
    return [current, previous]


def download_seasons(seasons, retries=3, timeout=60):
    frames = []
    for season in seasons:
        print(f"Fetching season {season}")
        for attempt in range(1, retries + 1):
            try:
                lg = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    timeout=timeout,
                    headers=HEADERS,
                )
                df = lg.get_data_frames()[0]
                frames.append(df)
                break
            except Exception as e:
                print(f"Attempt {attempt}/{retries} failed for {season}: {e}")
                if attempt == retries:
                    raise
                time.sleep(5 * attempt)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    seasons = current_seasons()
    print(f"Downloading seasons: {seasons}")
    df = download_seasons(seasons)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "games_raw.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
