"""Kolkata's traffic is simulated, but its clock is Kolkata's real one."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_engine import api  # noqa: E402
from route_engine.simulator import fmt_clock  # noqa: E402


def minutes_apart(a: str, b: str) -> int:
    """Minutes between two HH:MM clock readings, the short way round."""
    to_min = lambda c: int(c[:2]) * 60 + int(c[3:])
    d = (to_min(a) - to_min(b)) % 1440
    return min(d, 1440 - d)


@pytest.fixture(scope="module")
def following():
    """The website's engine, built as the API builds it: on the real clock."""
    saved = api.ENGINE, api.FOLLOWING, api.SIM_CLOCK
    api.ENGINE, api.FOLLOWING, api.SIM_CLOCK = None, None, ""
    eng = api.engine()
    yield eng
    api.ENGINE, api.FOLLOWING, api.SIM_CLOCK = saved


@pytest.fixture
def fresh(following):
    """Each test starts with the engine following the real clock again."""
    api.ENGINE, api.FOLLOWING = following, following
    return following


def test_kolkata_starts_at_the_real_time(fresh):
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M")
    assert minutes_apart(fresh.clock, now) <= 1


def test_the_clock_catches_up_with_real_time(fresh, monkeypatch):
    start = fresh.now_s
    monkeypatch.setattr(api, "kolkata_now_s", lambda: (start + 600) % 86400)
    api.engine()
    assert fresh.now_s == pytest.approx(start + 600, abs=1)


def test_catching_up_over_midnight_goes_forward(fresh, monkeypatch):
    fresh.set_clock("23:59")
    before = fresh.now_s
    monkeypatch.setattr(api, "kolkata_now_s", lambda: 60.0)      # 00:01
    api.engine()
    assert fresh.clock == "00:01"
    assert fresh.now_s == pytest.approx(before + 120, abs=1)


def test_the_page_is_told_the_clock_is_real(fresh):
    from fastapi.testclient import TestClient
    state = TestClient(api.app).get("/api/state").json()
    assert state["real_time"] is True


def test_choosing_a_time_stops_the_clock_following(fresh, monkeypatch):
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    assert client.post("/api/clock", json={"time": "09:00"}).json()["clock"] == "09:00"
    monkeypatch.setattr(api, "kolkata_now_s", lambda: 20 * 3600.0)
    assert api.engine().clock == "09:00"
    assert client.get("/api/state").json()["real_time"] is False


def test_a_pinned_clock_stays_where_it_was_pinned(monkeypatch):
    """TRIFFY_SIM_CLOCK=18:30 keeps the evening-rush demo available."""
    monkeypatch.setattr(api, "ENGINE", None)
    monkeypatch.setattr(api, "FOLLOWING", None)
    monkeypatch.setattr(api, "SIM_CLOCK", "18:30")
    eng = api.engine()
    assert eng.clock == "18:30"
    monkeypatch.setattr(api, "kolkata_now_s", lambda: 9 * 3600.0)
    assert api.engine().clock == "18:30"
