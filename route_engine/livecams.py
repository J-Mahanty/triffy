"""Real live traffic cameras.

This module is what stops Triffie being a simulation with a map on top.

Transport for London publishes ~890 roadside traffic cameras ("JamCams") as
open data: a still image and a short video clip per camera, refreshed every few
minutes, with no API key. We pull those clips, run the same YOLO11 detector and
tracker used everywhere else, and emit the same ``CamObservation``. Nothing
downstream can tell a live London camera from a simulated Kolkata one, which is
the entire point - the sensing layer is real, and it is the *conditions* that
are borrowed.

**Why London when the router serves Kolkata?** Because Kolkata Police do not
publish their camera feeds, and we would rather demonstrate the pipeline on
cameras that genuinely exist than assert it would work on cameras we cannot
reach. The adapter is deliberately thin: pointing it at any city that opens its
cameras is a config change, not a rewrite.

**What we measure, and why it is defensible.** Vehicle *count in frame* needs no
calibration at all - it is counting. Speed needs a per-camera ground-plane
calibration we cannot perform for 890 cameras, so absolute speeds here are
rough. Everything that matters downstream uses ratios against each camera's own
baseline, where a constant scale error cancels. Validation therefore targets
count and occupancy, which nobody can attack by attacking our geometry.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .cams import CamObservation
from .config import DATA

UA = {"User-Agent": "Triffie-DesignThinkingLab/0.1 (academic prototype)"}
TFL_REGISTRY = "https://api.tfl.gov.uk/Place/Type/JamCam"
REGISTRY_CACHE = DATA / "livecam_registry.json"
LIVE_DIR = DATA / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]          # bicycle, car, motorcycle, bus, truck


@dataclass
class LiveCamera:
    """One real camera we can actually fetch from."""
    id: str
    name: str
    lat: float
    lon: float
    image_url: str
    video_url: str
    available: bool = True

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "lat": self.lat, "lon": self.lon,
                "image_url": self.image_url, "video_url": self.video_url}


def _prop(cam: dict, key: str):
    for p in cam.get("additionalProperties", []):
        if p.get("key") == key:
            return p.get("value")
    return None


def fetch_registry(force: bool = False, max_age_s: float = 86400.0) -> list:
    """Camera list, cached on disk so a demo works without hammering the API."""
    if REGISTRY_CACHE.exists() and not force:
        age = time.time() - REGISTRY_CACHE.stat().st_mtime
        if age < max_age_s:
            blob = json.loads(REGISTRY_CACHE.read_text(encoding="utf-8"))
            return [LiveCamera(**c) for c in blob]

    r = requests.get(TFL_REGISTRY, headers=UA, timeout=90)
    r.raise_for_status()
    out = []
    for c in r.json():
        img, vid = _prop(c, "imageUrl"), _prop(c, "videoUrl")
        if not img or not vid:
            continue
        out.append(LiveCamera(
            id=str(c.get("id", "")).replace("JamCams_", ""),
            name=c.get("commonName", "unknown"),
            lat=float(c.get("lat", 0.0)), lon=float(c.get("lon", 0.0)),
            image_url=img, video_url=vid,
        ))
    REGISTRY_CACHE.write_text(json.dumps([c.as_dict() for c in out]), encoding="utf-8")
    return out


def cameras_in_box(south: float, west: float, north: float, east: float) -> list:
    """Every registered camera inside a bounding box."""
    return [c for c in fetch_registry()
            if south <= c.lat <= north and west <= c.lon <= east]


def pick_cameras(n: int = 24, bbox=None, city: str | None = None) -> list:
    """A spatially spread sample, so we are not watching one street repeatedly.

    Greedy farthest-point selection rather than random: with random sampling a
    handful of cameras inevitably cluster on the same junction, which would make
    our 'independent observations' correlated and quietly overstate coverage.

    ``bbox`` or ``city`` restricts the pool, which matters when the cameras are
    meant to feed a specific road network: observations from a camera 20 km
    outside the routed area cannot inform any edge in it.
    """
    if city and bbox is None:
        from .config import CITIES
        box = CITIES.get(city)
        if box:
            bbox = (box.south, box.west, box.north, box.east)

    cams = cameras_in_box(*bbox) if bbox else fetch_registry()
    if not cams:
        return []
    import math
    chosen = [cams[0]]
    pool = cams[1:]
    while len(chosen) < min(n, len(cams)):
        best, best_d = None, -1.0
        for c in pool:
            d = min((c.lat - k.lat) ** 2 + (c.lon - k.lon) ** 2 for k in chosen)
            if d > best_d:
                best, best_d = c, d
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
    return chosen


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _sampled_frames(cap, max_frames: int, stride: int):
    """Yield (n, frame) for every ``stride``-th frame, n counting from 1,
    until ``max_frames`` have been yielded or the clip ends."""
    idx = used = 0
    while used < max_frames:
        ok, frame = cap.read()
        if not ok:
            return
        idx += 1
        if idx % stride:
            continue
        used += 1
        yield used, frame


class LiveCameraReader:
    """Downloads a real camera clip and measures it with YOLO11."""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.25,
                 imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device      # None = ultralytics' choice (the GPU if present)

    def fetch_clip(self, cam: LiveCamera, timeout: float = 60.0):
        """Download the camera's latest clip. Returns (path, age_seconds)."""
        r = requests.get(cam.video_url, headers=UA, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return None, None
        path = LIVE_DIR / ("%s.mp4" % cam.id)
        path.write_bytes(r.content)

        age = None
        lm = r.headers.get("Last-Modified")
        if lm:
            try:
                from email.utils import parsedate_to_datetime
                age = time.time() - parsedate_to_datetime(lm).timestamp()
            except Exception:
                age = None
        return path, age

    def _tally(self, b, used: int, per_class: dict, tracks: dict) -> int:
        """Count one frame's boxes by class and extend each track's path."""
        n = 0 if b is None else len(b)
        if not n:
            return 0
        for c in b.cls.int().tolist():
            name = self.model.names[int(c)]
            per_class[name] = per_class.get(name, 0) + 1
        if b.id is not None:
            xy = b.xywh.cpu().numpy()
            for i, tid in enumerate(b.id.int().tolist()):
                tracks.setdefault(tid, []).append((used, float(xy[i][0]),
                                                   float(xy[i][1])))
        return n

    def measure(self, cam: LiveCamera, max_frames: int = 40, stride: int = 2):
        """Detect and track vehicles in the latest clip from a real camera.

        Returns a dict of raw measurements, or None if the feed was unavailable.
        Counting is per-frame and then averaged: a single frame flickers as
        partly-occluded vehicles drop in and out of detection, and traffic state
        does not meaningfully change across a ten-second clip anyway.
        """
        import cv2
        import numpy as np

        path, age = self.fetch_clip(cam)
        if path is None:
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        counts, per_class, tracks = [], {}, {}
        w = h = 0
        for used, frame in _sampled_frames(cap, max_frames, stride):
            h, w = frame.shape[0], frame.shape[1]
            kw = {"device": self.device} if self.device else {}
            res = self.model.track(frame, persist=True, conf=self.conf,
                                   imgsz=self.imgsz, verbose=False,
                                   classes=VEHICLE_CLASS_IDS, **kw)[0]
            counts.append(self._tally(res.boxes, used, per_class, tracks))
        cap.release()
        if not counts:
            return None

        # Fraction of tracked vehicles that are actually moving. This is
        # calibration-free - it needs no metres-per-pixel, only whether a box
        # shifted - and it is a direct proxy for whether traffic is flowing.
        moving_frac = self._moving_fraction(tracks, max(w, 1))

        frames = len(counts)
        return {
            "camera_id": cam.id,
            "name": cam.name,
            "lat": cam.lat, "lon": cam.lon,
            "t_wall": time.time(),
            "feed_age_s": round(age, 1) if age is not None else None,
            "frames": frames,
            "count_mean": float(np.mean(counts)),
            "count_max": int(np.max(counts)),
            "classes": {k: round(v / frames, 2) for k, v in per_class.items()},
            "tracks": len(tracks),
            "moving_frac": moving_frac,
            "width": w, "height": h,
        }

    @staticmethod
    def _moving_fraction(tracks: dict, frame_w: int) -> float:
        """Share of tracked vehicles whose box moved appreciably across the clip.

        Threshold is a fraction of frame width rather than an absolute pixel
        count, so it behaves the same on a 352px and a 1920px feed.
        """
        if not tracks:
            return 0.0
        thresh = max(2.0, 0.012 * frame_w)
        moving = 0
        counted = 0
        for pts in tracks.values():
            if len(pts) < 3:
                continue
            counted += 1
            dx = pts[-1][1] - pts[0][1]
            dy = pts[-1][2] - pts[0][2]
            if (dx * dx + dy * dy) ** 0.5 > thresh:
                moving += 1
        return round(moving / counted, 3) if counted else 0.0

    # -- adapter into the engine -------------------------------------------

    def to_observation(self, meas: dict, cam_id: str, edge: int, t_s: float,
                       free_kph: float, baseline_count: float | None = None
                       ) -> CamObservation:
        """Convert a real measurement into the engine's camera observation.

        The mapping is deliberately conservative. We do not pretend to know the
        absolute speed on a London street and transplant it onto a Kolkata road.
        Instead we convert what we *did* measure without calibration - how busy
        the frame is relative to that camera's own normal, and what share of
        vehicles are moving - into a congestion level, then express that against
        the target road's own free-flow speed.
        """
        base = baseline_count if baseline_count else max(meas["count_mean"], 1.0)
        load = min(1.5, meas["count_mean"] / max(base, 0.5))
        stalled = 1.0 - meas.get("moving_frac", 1.0)
        congestion = min(1.0, 0.55 * load + 0.65 * stalled)

        kph = max(3.5, free_kph * (1.0 - 0.80 * congestion))
        density = meas["count_mean"] / 0.10          # veh per km, ~100 m of view
        return CamObservation(
            cam_id=cam_id, edge=edge, t_s=t_s,
            vehicle_count=int(round(meas["count_mean"])),
            density_vpkm=float(density),
            speed_kph=float(kph),
            queue_m=float(80.0 * stalled),
            occupancy=float(congestion),
            confidence=0.72 if meas["frames"] >= 10 else 0.4,
            source="live",
            classes={k: int(round(v)) for k, v in meas["classes"].items()
                     if round(v) > 0},
        )


# The dashboard and the collector share one small GPU. The collector runs YOLO +
# tracking on clips back to back; when the dashboard's camera wall then fired
# eight request threads, each building its own YOLO model on CUDA at once, the
# device wedged and *both* processes hung in cuda.synchronize - the collector
# stopped collecting. So the collector owns the GPU, and the dashboard annotates
# single stills on the CPU (tens of ms for yolo11n at this resolution) with one
# shared model and one inference at a time: ultralytics' predict() is not
# thread-safe, and a stuck CUDA call cannot be timed out from Python anyway.
DASH_DEVICE = os.environ.get("TRIFFIE_DASH_DEVICE", "cpu")
_DASH_MODELS: dict = {}
_DASH_MODEL_LOCK = threading.Lock()
_DASH_INFER_LOCK = threading.Lock()


def limit_cpu_threads(n: int = 4) -> None:
    """Keep dashboard inference from taking every core. Torch defaults to all
    16 here, which would starve collector B (also on the CPU) mid-cycle."""
    try:
        import torch
        if torch.get_num_threads() > n:
            torch.set_num_threads(n)
    except Exception:
        pass


def _dashboard_model(model_name: str):
    with _DASH_MODEL_LOCK:
        m = _DASH_MODELS.get(model_name)
        if m is None:
            limit_cpu_threads()
            from ultralytics import YOLO
            m = _DASH_MODELS[model_name] = YOLO(model_name)
        return m


def annotate_frame(cam: LiveCamera, out_path: Path,
                   model_name: str = "yolo11n.pt", conf: float = 0.25,
                   device: str | None = None):
    """Fetch this camera's current still and draw detections on it.

    Used by the dashboard's live camera wall. Showing the boxes on a real image
    that is five minutes old is the fastest way to answer 'is any of this real?'
    """
    import cv2
    import numpy as np

    # The download happens outside the inference lock, so a burst of cameras
    # still fetches in parallel and only the short CPU forward pass queues.
    r = requests.get(cam.image_url, headers=UA, timeout=45)
    if r.status_code != 200:
        return None, 0, None
    age = None
    lm = r.headers.get("Last-Modified")
    if lm:
        try:
            from email.utils import parsedate_to_datetime
            age = time.time() - parsedate_to_datetime(lm).timestamp()
        except Exception:
            age = None

    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0, age

    model = _dashboard_model(model_name)
    with _DASH_INFER_LOCK:
        res = model.predict(img, conf=conf, verbose=False, classes=VEHICLE_CLASS_IDS,
                            device=device or DASH_DEVICE)[0]
    vis = res.plot()
    n = 0 if res.boxes is None else len(res.boxes)
    cv2.imwrite(str(out_path), vis)
    return out_path, n, age
