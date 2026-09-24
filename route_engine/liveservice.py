"""Serve real live cameras to the dashboard, with caching.

Annotating a frame costs a YOLO forward pass, and a public camera only refreshes
every few minutes, so re-running detection on every page poll would burn CPU to
redraw an identical picture. This keeps a small TTL cache: the dashboard can
poll freely, and we hit the real camera roughly as often as it actually updates.

Everything here is clearly labelled as real in the UI. The cameras are in
London, the router serves Kolkata, and conflating the two would be exactly the
kind of thing this module exists to avoid.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import DATA
from .livecams import (DASH_DEVICE, LiveCameraReader, annotate_frame,
                       fetch_registry, pick_cameras)

SHOTS_DIR = DATA / "live" / "shots"
SHOTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_TTL_S = 150.0        # public cameras refresh every few minutes
MEASURE_TTL_S = 300.0


class LiveCamService:
    """Thread-safe, lazily-refreshed view of a handful of real cameras."""

    def __init__(self, n_cameras: int = 6):
        self.n = n_cameras
        self._cams = None
        self._reader = None
        self._lock = threading.Lock()
        self._image_cache: dict = {}      # cam_id -> {path, n, age, fetched}
        self._measure_cache: dict = {}    # cam_id -> {meas, fetched}
        self._cam_locks: dict = {}        # cam_id -> Lock, see annotated_image
        self._registry = None             # cam_id -> LiveCamera, see _find
        self._error = None

    # -- lazy init ----------------------------------------------------------

    @property
    def cameras(self):
        if self._cams is None:
            with self._lock:
                if self._cams is None:
                    try:
                        self._cams = pick_cameras(self.n)
                    except Exception as exc:
                        self._error = "%s: %s" % (type(exc).__name__, exc)
                        self._cams = []
        return self._cams

    @property
    def reader(self):
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    # CPU, like the stills: the GPU belongs to the collector.
                    self._reader = LiveCameraReader(device=DASH_DEVICE)
        return self._reader

    def _find(self, cam_id: str):
        for c in self.cameras:
            if c.id == cam_id:
                return c
        # Clicking any of the London map's cameras asks for its footage, not
        # just the handful on the wall, so fall back to the whole registry
        # (disk-cached, so this is a dictionary lookup after the first call).
        if self._registry is None:
            try:
                self._registry = {c.id: c for c in fetch_registry()}
            except Exception:
                self._registry = {}
        return self._registry.get(cam_id)

    # -- images -------------------------------------------------------------

    def annotated_image(self, cam_id: str):
        """Path to a YOLO-annotated still, refreshed at most every IMAGE_TTL_S.

        Returns (path, n_vehicles, feed_age_s) or (None, 0, None).
        """
        cam = self._find(cam_id)
        if cam is None:
            return None, 0, None
        # The wall asks for /image and /info of the same camera at the same
        # moment; without this both miss the cache and annotate twice.
        with self._lock:
            cam_lock = self._cam_locks.setdefault(cam_id, threading.Lock())
        with cam_lock:
            now = time.time()
            hit = self._image_cache.get(cam_id)
            if hit and now - hit["fetched"] < IMAGE_TTL_S and Path(hit["path"]).exists():
                return Path(hit["path"]), hit["n"], hit["age"]

            out = SHOTS_DIR / ("%s.jpg" % cam_id)
            try:
                path, n, age = annotate_frame(cam, out)
            except Exception:
                return None, 0, None
            if path is None:
                return None, 0, None
            self._image_cache[cam_id] = {"path": str(path), "n": n, "age": age,
                                         "fetched": now}
            return path, n, age

    # -- measurements -------------------------------------------------------

    def measurement(self, cam_id: str):
        """Full tracked measurement (count, moving fraction), TTL-cached."""
        now = time.time()
        hit = self._measure_cache.get(cam_id)
        if hit and now - hit["fetched"] < MEASURE_TTL_S:
            return hit["meas"]
        cam = self._find(cam_id)
        if cam is None:
            return None
        try:
            meas = self.reader.measure(cam)
        except Exception:
            return None
        if meas:
            self._measure_cache[cam_id] = {"meas": meas, "fetched": now}
        return meas

    # -- listing ------------------------------------------------------------

    def listing(self) -> dict:
        cams = self.cameras
        return {
            "source": "Transport for London JamCams (open data, no API key)",
            "note": ("Real roadside cameras, refreshed every few minutes. These are "
                     "in London because Kolkata Police do not publish their feeds; "
                     "the detection pipeline is identical either way."),
            "error": self._error,
            "cameras": [{
                "id": c.id, "name": c.name, "lat": c.lat, "lon": c.lon,
                "cached": c.id in self._image_cache,
            } for c in cams],
        }


SERVICE: LiveCamService | None = None


def service() -> LiveCamService:
    global SERVICE
    if SERVICE is None:
        SERVICE = LiveCamService()
    return SERVICE
