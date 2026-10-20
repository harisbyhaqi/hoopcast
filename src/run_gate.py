"""
Decide whether this tick of the 5-minute workflow cron should run the pipeline.

Slots (Eastern time, so daylight saving is handled here rather than in cron):
  morning     10 AM   overnight results, today's schedule and tip times, rosters/injuries
  post-east   11 PM   East-coast games are final
  post-west    2 AM   West-coast games are final
  tip HH:MM   15 min before each distinct tip time today, for game-time decisions;
              games tipping at the same time share one run

Each slot runs once: slots already run are recorded in data/run_slots.json, which is
committed with the predictions, so a failed run is retried on the next tick.
Stdlib only: it runs before dependencies are installed.

Usage: python src/run_gate.py   → prints due slots, records them, writes run=true to $GITHUB_OUTPUT
"""

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEDULE_PATH = os.path.join(ROOT, "data", "schedule.csv")
SLOTS_PATH = os.path.join(ROOT, "data", "run_slots.json")

ET = ZoneInfo("America/New_York")
FIXED_SLOTS = {"morning": 10, "post-east": 23, "post-west": 2}   # ET hour → slot window is that hour
TIP_LEAD = timedelta(minutes=15)   # earliest run before tip-off
TIP_CUTOFF = timedelta(minutes=3)  # the pipeline takes ~2 min; later than this misses the freeze
KEEP_DAYS = 3


def tip_times(schedule: list[dict], day: str) -> list[datetime]:
    """Distinct UTC tip times of the not-yet-final games on an ET date."""
    tips = {r["tip_utc"] for r in schedule if r["date"] == day and r["status"] != "final" and r["tip_utc"]}
    return sorted(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in tips)


def due_slots(now: datetime, schedule: list[dict], done: set[str]) -> list[str]:
    local = now.astimezone(ET)
    day = local.date().isoformat()
    due = [f"{day} {name}" for name, hour in FIXED_SLOTS.items() if local.hour == hour]
    for tip in tip_times(schedule, day):
        if tip - TIP_LEAD <= now < tip - TIP_CUTOFF:
            due.append(f"{day} tip {tip.astimezone(ET):%H:%M}")
    return [s for s in due if s not in done]


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
    print(f"{now.astimezone(ET):%Y-%m-%d %H:%M} ET — due: {', '.join(due) or 'nothing'}")
    if not due:
        return
    record(done, due, now)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"run=true\nslots={'; '.join(due)}\n")


if __name__ == "__main__":
    main()
