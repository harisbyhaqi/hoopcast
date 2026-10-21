"""
Decide whether this tick of the 5-minute workflow cron should run the pipeline.

Slots (Eastern time, so daylight saving is handled here rather than in cron):
  morning     10 AM   overnight results, today's schedule and tip times, rosters/injuries
  post-east   11 PM   East-coast games are final
  post-west    2 AM   West-coast games are final
  tip HH:MM   15 min before each distinct tip time today, for game-time decisions;
              games tipping at the same time share one run

GitHub fires scheduled workflows hours apart under load, so ticks can't be relied on to land
in a slot. Instead a missed fixed slot is made up on the next tick, and a tick with nothing
due sleeps until the next slot (if it starts within MAX_WAIT) and runs then.

Each slot runs once: slots already run are recorded in data/run_slots.json, which is
committed with the predictions, so a failed run is retried on the next tick.
Stdlib only: it runs before dependencies are installed.

Usage: python src/run_gate.py   → prints due slots, records them, writes run=true to $GITHUB_OUTPUT
"""

import csv
import json
import os
import time
from datetime import datetime, time as clock, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEDULE_PATH = os.path.join(ROOT, "data", "schedule.csv")
SLOTS_PATH = os.path.join(ROOT, "data", "run_slots.json")

ET = ZoneInfo("America/New_York")
FIXED_SLOTS = {"morning": 10, "post-east": 23, "post-west": 2}   # ET hour
TIP_LEAD = timedelta(minutes=15)   # earliest run before tip-off
TIP_CUTOFF = timedelta(minutes=3)  # the pipeline takes ~2 min; later than this misses the freeze
LOOKBACK = timedelta(hours=24)    # how late a missed fixed slot is still made up
MAX_WAIT = timedelta(hours=5)     # longest a tick sleeps for the next slot (jobs may run 6 h)
KEEP_DAYS = 3


def tip_times(schedule: list[dict], day: str) -> list[datetime]:
    """Distinct UTC tip times of the not-yet-final games on an ET date."""
    tips = {r["tip_utc"] for r in schedule if r["date"] == day and r["status"] != "final" and r["tip_utc"]}
    return sorted(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in tips)


def slot_windows(now: datetime, schedule: list[dict]) -> dict[str, tuple[datetime, datetime]]:
    """Every slot from yesterday to tomorrow (ET) → the UTC window in which it may run."""
    today = now.astimezone(ET).date()
    windows = {}
    for day in (today - timedelta(days=1), today, today + timedelta(days=1)):
        d = day.isoformat()
        for name, hour in FIXED_SLOTS.items():
            start = datetime.combine(day, clock(hour), ET).astimezone(timezone.utc)
            windows[f"{d} {name}"] = (start, start + LOOKBACK)
        for tip in tip_times(schedule, d):
            windows[f"{d} tip {tip.astimezone(ET):%H:%M}"] = (tip - TIP_LEAD, tip - TIP_CUTOFF)
    return windows


def due_slots(now: datetime, schedule: list[dict], done: set[str]) -> list[str]:
    return [k for k, (start, end) in slot_windows(now, schedule).items()
            if start <= now < end and k not in done]


def next_start(now: datetime, schedule: list[dict], done: set[str]) -> datetime | None:
    """When the next slot not yet run opens."""
    return min((start for k, (start, _) in slot_windows(now, schedule).items()
                if start > now and k not in done), default=None)


def load_done() -> set[str]:
    if not os.path.exists(SLOTS_PATH):
        return set()
    with open(SLOTS_PATH) as f:
        return set(json.load(f))


def record(done: set[str], new: list[str], now: datetime) -> None:
    cutoff = (now.astimezone(ET).date() - timedelta(days=KEEP_DAYS)).isoformat()
    kept = sorted(s for s in done | set(new) if s[:10] >= cutoff)
    with open(SLOTS_PATH, "w") as f:
        json.dump(kept, f, indent=1)


def main():
    now = datetime.now(timezone.utc)
    with open(SCHEDULE_PATH, newline="") as f:
        schedule = list(csv.DictReader(f))
    done = load_done()
    due = due_slots(now, schedule, done)
    if not due:
        start = next_start(now, schedule, done)
        if start and start - now <= MAX_WAIT:
            print(f"{now.astimezone(ET):%Y-%m-%d %H:%M} ET — sleeping until {start.astimezone(ET):%H:%M} ET")
            time.sleep((start - now).total_seconds())
            now = datetime.now(timezone.utc)
            due = due_slots(now, schedule, done)
    print(f"{now.astimezone(ET):%Y-%m-%d %H:%M} ET — due: {', '.join(due) or 'nothing'}")
    if not due:
        return
    record(done, due, now)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"run=true\nslots={'; '.join(due)}\n")


if __name__ == "__main__":
    main()
