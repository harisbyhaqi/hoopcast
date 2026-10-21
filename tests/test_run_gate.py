from datetime import datetime, timezone

from run_gate import due_slots, next_start

SCHEDULE = [  # Oct 20, 2026 (EDT, UTC-4): tips at 7:00 PM (two games) and 10:00 PM ET
    {"date": "2026-10-20", "tip_utc": "2026-10-20T23:00:00Z", "status": "pre"},
    {"date": "2026-10-20", "tip_utc": "2026-10-20T23:00:00Z", "status": "pre"},
    {"date": "2026-10-20", "tip_utc": "2026-10-21T02:00:00Z", "status": "pre"},
    {"date": "2026-10-20", "tip_utc": "2026-10-20T16:00:00Z", "status": "final"},
]
# Fixed slots before Oct 20's morning, already run
EARLIER = {"2026-10-19 morning", "2026-10-19 post-east", "2026-10-20 post-west"}


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_fixed_slots_follow_eastern_time():
    assert due_slots(utc("2026-10-20T14:05"), [], EARLIER) == ["2026-10-20 morning"]       # 10:05 AM EDT
    assert due_slots(utc("2027-01-12T15:05"), [], {"2027-01-11 morning", "2027-01-11 post-east",
                                                   "2027-01-12 post-west"}) == ["2027-01-12 morning"]  # 10:05 AM EST
    done = EARLIER | {"2026-10-20 morning"}
    assert due_slots(utc("2026-10-21T03:10"), [], done) == ["2026-10-20 post-east"]       # 11:10 PM EDT
    assert due_slots(utc("2026-10-21T06:10"), [], done | {"2026-10-20 post-east"}) == ["2026-10-21 post-west"]
    assert due_slots(utc("2026-10-20T18:00"), [], done) == []


def test_missed_fixed_slots_run_on_the_next_tick():
    # GitHub skipped every tick from 8:48 PM to 6:24 AM: both overnight slots are made up at once
    done = EARLIER | {"2026-10-20 morning"}
    assert due_slots(utc("2026-10-21T10:24"), [], done) == ["2026-10-20 post-east", "2026-10-21 post-west"]


def test_next_start_is_the_earliest_open_slot():
    done = EARLIER | {"2026-10-20 morning"}
    assert next_start(utc("2026-10-20T18:00"), SCHEDULE, done) == utc("2026-10-20T22:45")  # 7 PM tip − 15 min
    assert next_start(utc("2026-10-21T10:24"), [], done | {"2026-10-20 post-east", "2026-10-21 post-west"}) \
        == utc("2026-10-21T14:00")                                                            # 10 AM EDT


def test_games_tipping_together_share_one_pre_tip_slot():
    assert due_slots(utc("2026-10-20T22:48"), SCHEDULE, EARLIER | {"2026-10-20 morning"}) == ["2026-10-20 tip 19:00"]


def test_pre_tip_window():
    done = EARLIER | {"2026-10-20 morning"}
    assert due_slots(utc("2026-10-20T22:40"), SCHEDULE, done) == []   # too early
    assert due_slots(utc("2026-10-20T22:58"), SCHEDULE, done) == []   # too close to tip to finish
    assert due_slots(utc("2026-10-20T15:50"), SCHEDULE, done) == []   # final games are skipped


def test_late_tip_uses_its_eastern_date():
    assert "2026-10-20 tip 22:00" in due_slots(utc("2026-10-21T01:50"), SCHEDULE, EARLIER)


def test_slots_already_run_are_skipped():
    done = EARLIER | {"2026-10-20 morning", "2026-10-20 tip 19:00"}
    assert due_slots(utc("2026-10-20T22:48"), SCHEDULE, done) == []
