from datetime import datetime, timezone

from run_gate import due_slots

SCHEDULE = [  # Oct 20, 2026 (EDT, UTC-4): tips at 7:00 PM (two games) and 10:00 PM ET
    {"date": "2026-10-20", "tip_utc": "2026-10-20T23:00:00Z", "status": "pre"},
    {"date": "2026-10-20", "tip_utc": "2026-10-20T23:00:00Z", "status": "pre"},
    {"date": "2026-10-20", "tip_utc": "2026-10-21T02:00:00Z", "status": "pre"},
    {"date": "2026-10-20", "tip_utc": "2026-10-20T16:00:00Z", "status": "final"},
]


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_fixed_slots_follow_eastern_time():
    assert due_slots(utc("2026-10-20T14:05"), [], set()) == ["2026-10-20 morning"]         # 10:05 AM EDT
    assert due_slots(utc("2027-01-12T15:05"), [], set()) == ["2027-01-12 morning"]         # 10:05 AM EST
    assert due_slots(utc("2026-10-21T03:10"), [], set()) == ["2026-10-20 post-east"]       # 11:10 PM EDT
    assert due_slots(utc("2026-10-21T06:10"), [], set()) == ["2026-10-21 post-west"]       # 2:10 AM EDT
    assert due_slots(utc("2026-10-20T18:00"), [], set()) == []


def test_games_tipping_together_share_one_pre_tip_slot():
    assert due_slots(utc("2026-10-20T22:48"), SCHEDULE, set()) == ["2026-10-20 tip 19:00"]


def test_pre_tip_window():
    assert due_slots(utc("2026-10-20T22:40"), SCHEDULE, set()) == []   # too early
    assert due_slots(utc("2026-10-20T22:58"), SCHEDULE, set()) == []   # too close to tip to finish
    assert due_slots(utc("2026-10-20T15:50"), SCHEDULE, set()) == []   # final games are skipped


def test_late_tip_uses_its_eastern_date():
    assert due_slots(utc("2026-10-21T01:50"), SCHEDULE, set()) == ["2026-10-20 tip 22:00"]


def test_slots_already_run_are_skipped():
    assert due_slots(utc("2026-10-20T22:48"), SCHEDULE, {"2026-10-20 tip 19:00"}) == []
