"""A public, read-only site: viewers can plan, but not change or load the host."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_engine import api  # noqa: E402


@pytest.fixture
def viewer(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "READONLY", True)
    return TestClient(api.app)


@pytest.mark.parametrize("path", ["/api/livecam/00001.07450/measure",
                                  "/api/livecam/00001.07450/clip",
                                  "/api/livecam/00001.07450/clip/stream",
                                  "/api/livecam/00001.07450/image"])
def test_viewers_cannot_make_the_host_run_vision(viewer, path):
    r = viewer.get(path)
    assert r.status_code == 403
    assert "collectors" in r.json()["detail"]


def test_viewers_cannot_change_the_clock(viewer):
    assert viewer.post("/api/clock", json={"time": "03:00"}).status_code == 403


def test_the_camera_registry_is_swapped_in_whole(tmp_path, monkeypatch):
    """The website and two collectors fetch it at once on a fresh server; a
    reader must never see a half-written file, or leftovers pile up."""
    from route_engine import livecams

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            prop = lambda k, v: {"key": k, "value": v}
            return [{"id": "JamCams_00001.1", "commonName": "Test Rd", "lat": 51.5,
                     "lon": -0.1, "additionalProperties": [prop("imageUrl", "i.jpg"),
                                                           prop("videoUrl", "v.mp4")]}]

    cache = tmp_path / "livecam_registry.json"
    monkeypatch.setattr(livecams, "REGISTRY_CACHE", cache)
    monkeypatch.setattr(livecams.requests, "get", lambda *a, **k: Reply())
    cams = livecams.fetch_registry(force=True)
    assert [c.id for c in cams] == ["00001.1"]
    assert [c.id for c in livecams.fetch_registry()] == ["00001.1"]     # read back from cache
    assert [p.name for p in tmp_path.iterdir()] == ["livecam_registry.json"]


def test_the_camera_list_is_not_caught_by_the_vision_guard():
    """/api/livecams (a cheap listing) shares a prefix with the guarded paths."""
    assert not "/api/livecams".startswith(api.VISION_ON_DEMAND)
