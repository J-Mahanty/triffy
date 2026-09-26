"""Choosing which cameras to watch, and the launcher that starts them."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from route_engine.coverage import cameras_near_routes, distance_to_routes_m


def test_distance_to_routes_is_in_metres():
    route = np.array([[51.5, -0.10, 51.5, -0.12]])   # one 1.4 km straight segment
    on = SimpleNamespace(id="on", lat=51.5, lon=-0.11)
    north = SimpleNamespace(id="north", lat=51.501, lon=-0.11)    # ~111 m north
    d = distance_to_routes_m([on, north], route)
    assert d["on"] < 1.0
    assert 105 < d["north"] < 117


@pytest.fixture(scope="module")
def london():
    from route_engine.live_engine import LiveEngine
    return LiveEngine(city="lon")


def test_the_chosen_cameras_sit_on_the_demo_routes(london):
    near = cameras_near_routes(30, engine=london)
    assert len(near) == 30
    assert [d for _, d in near] == sorted(d for _, d in near)
    # Every one of them watches a road the demo trips actually drive.
    assert near[-1][1] < 50, near[-1]
    assert {i for i, _ in near} <= set(london.by_id)


def test_launcher_starts_the_right_things():
    def dry(n):
        out = subprocess.run([sys.executable, "start.py", "--cameras", str(n),
                              "--dry-run"], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stderr[-400:]
        return out.stdout
    replay = dry(0)
    assert "--replay" in replay and "collector" not in replay
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return    # without the vision packages the launcher falls back to replay
    live = dry(30)
    assert "--near-routes --cameras 30" in live and "--replay" not in live
