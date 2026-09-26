"""Live engine: the whole system running on real cameras and a real network.

This is the answer to "isn't it all simulated?". Everything in this module is
measurement:

* a real road network (OpenStreetMap, London Zone 1),
* real roadside cameras (Transport for London, ~172 inside the routed area),
* real detections from our own YOLO11,
* real observations, collected continuously by ``collector.py``,
* the same nowcaster, forecaster and router used everywhere else.

No simulator is imported here. If the cameras go dark, this engine degrades to
its historical prior and says so, exactly as a deployed system would.

**Why London.** It is the only city in this project with public live camera
feeds. Proving the pipeline end-to-end on real data there is what makes the
Kolkata proposal credible: the missing piece for Kolkata is feed access, not
method. The Kolkata engine (``engine.py``) keeps its simulator because without
real feeds there is nothing else to drive it.

**Turning a camera into a road speed, without calibration.** We cannot survey
172 cameras, so each one calibrates against *its own history*: the 20th
percentile of its observed counts is "quiet here", the 85th is "busy here", and
today sits somewhere between. That needs no geometry at all. It is combined with
the fraction of tracked vehicles actually moving, which is the more direct
congestion signal - fifteen vehicles flowing and fifteen vehicles stopped look
identical to a counter, but not to a tracker.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .cams import CamObservation
from .collector import OBS_PATH
from .config import ACTIVE_CITY, ALTERNATIVES, MIN_SPEED_KPH
from .forecast import Forecaster, SpeedProfile
from .livecams import LiveCamera, pick_cameras
from .network import load_network
from .nowcast import Nowcaster
from .personalize import ProfileStore, UserProfile
from .router import Router
from .simulator import fmt_clock, parse_clock

# How stale a camera reading may be before we stop trusting it. TfL refreshes
# every few minutes; beyond ~25 minutes a reading says more about the past than
# the present.
MAX_OBS_AGE_S = 1500.0

# Replay: treat a recorded moment as 'now', e.g. TRIFFY_REPLAY="2026-09-25 17:30"
# (London time). The clock then runs forward from there in real time, so the
# recording plays like a film: readings 'arrive' as they did on the day.
LONDON = ZoneInfo("Europe/London")


def parse_replay(text) -> float | None:
    """'YYYY-MM-DD HH:MM' in London time -> Unix time; empty -> None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        moment = datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("Replay time must look like 2026-09-25 17:30 "
                         "(London time), not %r" % text) from None
    return moment.replace(tzinfo=LONDON).timestamp()


# ---------------------------------------------------------------------------
# Mapping cameras onto road edges
# ---------------------------------------------------------------------------

@dataclass
class MappedCamera:
    """A real camera bound to the road edge it watches."""
    id: str
    name: str
    lat: float
    lon: float
    edge: int
    dist_m: float
    road: str

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "lat": self.lat, "lon": self.lon,
                "edge": int(self.edge), "dist_m": round(self.dist_m, 1),
                "road": self.road}


def map_cameras_to_edges(net, cameras, max_dist_m: float = 120.0) -> list:
    """Bind each camera to the nearest road edge, dropping ones that miss.

    A camera watches one approach to a junction, and we cannot tell which from
    the registry alone. We therefore snap to the edge whose midpoint is nearest,
    and drop any camera further than ``max_dist_m`` from any road we model -
    attributing a reading to the wrong street is worse than not using it.

    Two cameras landing on the same edge would silently shadow one another in the
    nowcaster, which keys observations by edge, so the closer one wins.
    """
    best_for_edge: dict = {}
    for cam in cameras:
        node = net.nearest_node(cam.lat, cam.lon)
        candidates = list(net.out_edge_ids(node)) + list(net.in_edge_ids(node))
        if not candidates:
            continue
        best, best_d, best_score = None, 1e18, 1e18
        for eid in candidates:
            eid = int(eid)
            mlat, mlon = net.edge_midpoint(eid)
            dx = (mlon - cam.lon) * net.mx
            dy = (mlat - cam.lat) * net.my
            d = math.hypot(dx, dy)
            # Prefer the main carriageway over a service alley that happens to
            # be a few metres closer. Traffic authorities mount cameras to watch
            # significant roads, so when two candidates are comparably near, the
            # more important one is almost always the intended subject.
            score = d * (1.0 + 0.30 * int(net.erank[eid]))
            if score < best_score:
                best, best_d, best_score = eid, d, score
        if best is None or best_d > max_dist_m:
            continue
        prev = best_for_edge.get(best)
        if prev is None or best_d < prev.dist_m:
            best_for_edge[best] = MappedCamera(
                id=cam.id, name=cam.name, lat=cam.lat, lon=cam.lon,
                edge=best, dist_m=best_d, road=net.describe_edge(best))
    return list(best_for_edge.values())


class _CamSet:
    """Minimal adapter so Nowcaster can consume mapped live cameras unchanged."""

    def __init__(self, net, mapped):
        self.net = net
        self.cams = mapped
        self.covered = np.zeros(net.n_edges, dtype=bool)
        if mapped:
            self.covered[[m.edge for m in mapped]] = True

    def coverage_stats(self) -> dict:
        net = self.net
        arterial = net.erank <= 3
        return {
            "cameras": len(self.cams),
            "edges_watched": int(self.covered.sum()),
            "edges_total": int(net.n_edges),
            "pct_edges": round(100.0 * self.covered.sum() / max(net.n_edges, 1), 2),
            "pct_arterial_edges": round(
                100.0 * (self.covered & arterial).sum() / max(1, arterial.sum()), 1),
            "km_watched": round(float(net.elen[self.covered].sum() / 1000.0), 1),
        }


# ---------------------------------------------------------------------------
# Self-calibration from collected history
# ---------------------------------------------------------------------------

class CameraBaselines:
    """Per-camera count percentiles, so each camera is its own yardstick.

    A count of 12 means nothing in isolation: it is heavy on a back street and
    empty on Euston Road. What matters is where today sits within *that camera's*
    own range, which is learnable from its history and needs no calibration.
    """

    def __init__(self, path=OBS_PATH, min_samples: int = 6):
        self.quiet: dict = {}
        self.busy: dict = {}
        self.min_samples = min_samples
        self.n_cameras = 0
        self._load(path)

    def _load(self, path) -> None:
        if not path or not path.exists():
            return
        by_cam: dict = {}
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    by_cam.setdefault(r["camera_id"], []).append(
                        float(r.get("count_mean", 0.0)))
        except Exception:
            return
        for cid, vals in by_cam.items():
            if len(vals) < self.min_samples:
                continue
            arr = np.array(vals, dtype=float)
            self.quiet[cid] = float(np.percentile(arr, 20))
            self.busy[cid] = float(np.percentile(arr, 85))
        self.n_cameras = len(self.quiet)

    def congestion_from_count(self, cam_id: str, count: float) -> float | None:
        """0 (as quiet as this camera gets) .. 1 (as busy as it gets)."""
        q = self.quiet.get(cam_id)
        b = self.busy.get(cam_id)
        if q is None or b is None or b - q < 0.5:
            return None                     # not enough history to calibrate
        return float(np.clip((count - q) / (b - q), 0.0, 1.0))


# Below this many tracked vehicles, "what fraction is moving" stops meaning
# anything: there is nothing to move.
MIN_TRACKS_FOR_STALL = 2.0


def observation_to_congestion(row: dict, baselines: CameraBaselines):
    """Combine 'how busy' and 'how stalled' into one congestion figure.

    Returns ``None`` when the frame carries no usable information.

    ``moving_frac`` is weighted more heavily than count because it is the more
    direct evidence: a jam is precisely many vehicles and few of them moving.

    **The zero-detection trap.** ``moving_frac`` is 0 both when every vehicle is
    stopped and when there are no vehicles at all, and those are opposite
    traffic states. Reading it naively as ``1 - moving_frac`` made an empty road
    the most congested thing on the map: cameras reporting 0 vehicles were being
    given occupancy 1.00, and Oxford Street was quoted at 67 minutes.

    Flipping it to "empty means clear" would be just as wrong, because at night,
    in rain, or on a degraded feed a genuine jam also detects as nothing. With no
    detections we simply do not know, so we say so and let the nowcaster fall
    back to its prior rather than inventing a reading in either direction.
    """
    count = float(row.get("count_mean", 0.0))
    tracks = float(row.get("tracks", 0.0))
    count_c = baselines.congestion_from_count(row["camera_id"], count)

    if tracks < MIN_TRACKS_FOR_STALL or count < 0.5:
        # Not enough detections to judge flow. If this camera has enough history
        # for its count to be meaningful, use that alone; otherwise abstain.
        return count_c

    stalled = 1.0 - float(row.get("moving_frac", 1.0))
    if count_c is None:
        return float(np.clip(stalled, 0.0, 1.0))
    return float(np.clip(0.45 * count_c + 0.55 * stalled, 0.0, 1.0))


class LiveHistoricalModel:
    """Expected speed per edge, learned from collected real observations.

    Stands in for the weeks of archive a deployment would have. With a day or two
    it can only support a coarse network-wide time-of-day congestion curve, which
    is applied against each road's free-flow speed. That is thin, and we say so
    rather than dressing it up: the honest consequence is that the live residual
    carries most of the signal, which the forecaster already handles.
    """

    def __init__(self, net, path=OBS_PATH, bucket_min: int = 60):
        self.net = net
        self.bucket_min = bucket_min
        self.by_bucket: dict = {}
        self.global_congestion = 0.35
        self._fit(path)

    def _fit(self, path) -> None:
        if not path or not path.exists():
            return
        baselines = CameraBaselines(path)
        buckets: dict = {}
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    c = observation_to_congestion(r, baselines)
                    if c is None:
                        continue        # uninformative frame; see that function
                    lt = time.localtime(r["t_wall"])
                    b = (lt.tm_hour * 60 + lt.tm_min) // self.bucket_min
                    buckets.setdefault(b, []).append(c)
        except Exception:
            return
        self.by_bucket = {b: float(np.mean(v)) for b, v in buckets.items() if v}
        if self.by_bucket:
            self.global_congestion = float(np.mean(list(self.by_bucket.values())))

    def expected_congestion(self, t_wall: float) -> float:
        lt = time.localtime(t_wall)
        b = (lt.tm_hour * 60 + lt.tm_min) // self.bucket_min
        return self.by_bucket.get(b, self.global_congestion)

    def expected_kph(self, t_wall: float) -> np.ndarray:
        c = self.expected_congestion(t_wall)
        # Minor roads are less affected by the arterial congestion cycle.
        scale = np.where(self.net.erank <= 3, 1.0, 0.6)
        return np.maximum(self.net.ekph * (1.0 - 0.80 * c * scale), MIN_SPEED_KPH)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def _camera_row(m, o, r, now: float) -> dict:
    """One camera for the dashboard: its latest usable observation ``o`` and
    raw collector row ``r`` (either may be missing)."""
    moving = None
    if o and r and r.get("moving_frac") is not None:
        moving = round(100 * float(r["moving_frac"]))
    return {
        "id": m.id, "name": m.name, "lat": m.lat, "lon": m.lon,
        "road": m.road,
        # kph is *derived* from congestion, not measured; the panel labels it
        # as an estimate.
        "kph": round(o.speed_kph, 1) if o else None,
        "count": round(float(r.get("count_mean", 0)), 1) if o and r else None,
        "occ": round(o.occupancy, 2) if o else None,
        "moving_pct": moving,
        "age_s": round(now - r["t_wall"]) if r else None,
        "classes": o.classes if o else {},
        "reporting": o is not None,
    }


class LiveEngine:
    """Routing driven entirely by real measurements."""

    def __init__(self, city: str = "lon", max_cameras: int = 172,
                 obs_path=OBS_PATH, replay_at: float | None = None):
        self.city = city
        # Replay moment: an explicit argument, else TRIFFY_REPLAY, else live.
        self.replay_at = (replay_at if replay_at is not None
                          else parse_replay(os.environ.get("TRIFFY_REPLAY")))
        self._clock_started = time.time()
        self.net = load_network(city)
        self.obs_path = obs_path

        cameras = pick_cameras(max_cameras, city=city)
        self.mapped = map_cameras_to_edges(self.net, cameras)
        self.cams = _CamSet(self.net, self.mapped)
        self.by_id = {m.id: m for m in self.mapped}

        self.nowcaster = Nowcaster(self.net, self.cams)
        self.baselines = CameraBaselines(obs_path)
        self.hist = LiveHistoricalModel(self.net, obs_path)
        self.forecaster = Forecaster(self.net, self.hist)
        self.profiles = ProfileStore()

        self.observations: list = []
        self.state = None
        self.last_refresh = 0.0
        self.data_age_s = None
        self.refresh()

    # -- clock --------------------------------------------------------------

    def now(self) -> float:
        """The moment the engine treats as 'now' (Unix time).

        Everything that asks what time it is goes through here, so the
        engine can be pointed at another moment in one place. In replay it
        starts at the replay moment and advances in real time.
        """
        if self.replay_at is not None:
            return self.replay_at + (time.time() - self._clock_started)
        return time.time()

    def replay_info(self) -> dict | None:
        """What the UI needs to label a replay honestly, or None when live."""
        if self.replay_at is None:
            return None
        fmt = lambda t: datetime.fromtimestamp(t, LONDON).strftime("%Y-%m-%d %H:%M")
        return {"from": fmt(self.replay_at), "now": fmt(self.now()),
                "now_s": round(self.now())}

    # -- ingest -------------------------------------------------------------

    def _latest_rows(self, until: float | None = None) -> dict:
        """Most recent observation per camera, from the collector's log.

        Readings recorded after ``until`` are ignored: in replay, the future
        of the recording has not happened yet.
        """
        latest: dict = {}
        if not self.obs_path.exists():
            return latest
        with self.obs_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cid = r.get("camera_id")
                if cid not in self.by_id:
                    continue
                if until is not None and r["t_wall"] > until:
                    continue
                prev = latest.get(cid)
                if prev is None or r["t_wall"] > prev["t_wall"]:
                    latest[cid] = r
        return latest

    def refresh(self):
        """Rebuild the network belief from the freshest real observations."""
        now = self.now()
        rows = self._latest_rows(until=now)
        self.latest_rows = rows        # raw readings, for the camera panel

        obs, ages = [], []
        for cid, r in rows.items():
            age = now - r["t_wall"]
            if age > MAX_OBS_AGE_S:
                continue                      # too stale to describe 'now'
            m = self.by_id[cid]
            cong = observation_to_congestion(r, self.baselines)
            if cong is None:
                # The camera saw nothing usable. Passing it on as "clear" or
                # "jammed" would both be fabrication; dropping it lets the
                # nowcaster fall back to the prior, which is what not knowing
                # actually means.
                continue
            free = float(self.net.ekph[m.edge])
            kph = max(MIN_SPEED_KPH, free * (1.0 - 0.80 * cong))

            # Confidence falls with staleness and with having few tracks.
            fresh = max(0.0, 1.0 - age / MAX_OBS_AGE_S)
            tracks = float(r.get("tracks", 0))
            conf = float(np.clip(0.35 + 0.45 * fresh + 0.05 * min(tracks, 4), 0.15, 0.95))

            obs.append(CamObservation(
                cam_id=cid, edge=m.edge, t_s=r["t_wall"],
                vehicle_count=int(round(r.get("count_mean", 0))),
                density_vpkm=float(r.get("count_mean", 0)) * 10.0,
                speed_kph=kph,
                queue_m=float(80.0 * (1.0 - r.get("moving_frac", 1.0))),
                occupancy=cong,
                confidence=conf,
                source="live",
                classes={k: int(round(v)) for k, v in
                         (r.get("classes") or {}).items() if round(v) > 0},
            ))
            ages.append(age)

        self.observations = obs
        self.data_age_s = float(np.median(ages)) if ages else None
        hist_kph = self.hist.expected_kph(now)
        self.state = self.nowcaster.update(obs, hist_kph, now)
        self.last_refresh = now
        return self.state

    # -- planning -----------------------------------------------------------

    def resolve(self, text: str):
        return self.net.resolve(text)

    def ensure_fresh(self, max_age_s: float = 120.0) -> None:
        """Re-read the collector's log if the belief is older than ``max_age_s``."""
        if time.time() - self.last_refresh > max_age_s:
            self.refresh()

    def plan(self, origin: str, destination: str, user_id: str = "guest",
             k: int = ALTERNATIVES, max_age_s: float = 120.0):
        """Plan a journey against live measured conditions."""
        self.ensure_fresh(max_age_s)

        o = self.resolve(origin)
        d = self.resolve(destination)
        if o is None:
            raise ValueError("I could not find %r on the map." % origin)
        if d is None:
            raise ValueError("I could not find %r on the map." % destination)
        if o.node == d.node:
            raise ValueError("Origin and destination are the same place.")

        user = self.profiles.get(user_id)
        routes, baseline = self.plan_routes(o.node, d.node, user_id=user_id, k=k)

        route_dicts = []
        for r in routes:
            rd = r.as_dict(self.net)
            rd["live_share"] = self.live_share(r.edges)
            route_dicts.append(rd)

        return {
            "origin": o.name, "destination": d.name,
            "user": {"id": user.user_id, "name": user.name,
                     "vehicle": user.vehicle},
            "data_age_s": round(self.data_age_s) if self.data_age_s else None,
            "cameras_reporting": len(self.observations),
            "replay": self.replay_info(),
            "routes": route_dicts,
            "baseline": baseline.as_dict(self.net) if baseline else None,
        }

    def plan_routes(self, src: int, dst: int, user_id: str = "guest",
                    k: int = ALTERNATIVES, depart: float | None = None,
                    with_baseline: bool = True):
        """(routes, baseline) between two nodes, leaving at wall-clock
        ``depart`` (default now). Raises ValueError when there is no route.

        A later departure is forecast from today's measured anomaly relaxing
        back to the usual pattern for that time, which is what the forecaster
        does for every step of every trip anyway.
        """
        user = self.profiles.get(user_id)
        depart = self.now() if depart is None else float(depart)
        prof = self.forecaster.build_profile(self.state, depart)
        user.adapt_profile(self.net, prof)
        router = Router(self.net, prof, prefs=user)
        routes = router.route(src, dst, depart, k=k)
        if not routes:
            raise ValueError("No route found between those two points.")
        if not with_baseline:
            return routes, None

        # The snapshot baseline, for the same comparison the simulated engine
        # makes - but here both are costed against real measured conditions.
        snap = self.forecaster.snapshot_profile(self.state, depart)
        user.adapt_profile(self.net, snap)
        snap_router = Router(self.net, snap, prefs=user)
        picks = snap_router.route(src, dst, depart, k=1)
        baseline = None
        if picks:
            baseline = router.evaluate_path(picks[0].edges, depart)
            baseline.label = "Snapshot router (baseline)"
            baseline.claimed_s = picks[0].mean_s
        return routes, baseline

    def live_share(self, edges) -> dict:
        """How much of this route is actually informed by live cameras.

        A guardrail against our own marketing. "Planned against live conditions"
        is a claim, and a route can be 98% camera-informed or 2%, depending on
        whether it happens to run along watched corridors. Before corridor
        propagation the median route was 18% informed and one demo route was
        2% - and the UI said "live" for all of them, which was close to false.

        Reporting the number per route means the claim is checkable on screen
        instead of asserted, and it fails loudly if coverage ever regresses.
        """
        e = np.asarray(list(edges), dtype=np.int64)
        if e.size == 0:
            return {"informed_pct": 0.0, "strong_pct": 0.0, "observed_edges": 0}
        lens = self.net.elen[e]
        total = float(lens.sum()) or 1.0
        sup = self.state.support[e]
        return {
            # Any camera influence at all.
            "informed_pct": round(100.0 * float((sup > 0.02) @ lens) / total, 1),
            # Enough influence that the reading, not the prior, is driving it.
            "strong_pct": round(100.0 * float((sup > 0.20) @ lens) / total, 1),
            "observed_edges": int(self.state.observed[e].sum()),
        }

    # -- dashboard feed -----------------------------------------------------

    def stats(self) -> dict:
        net, st = self.net, self.state
        art = net.erank <= 3
        cong = st.congestion(net)
        return {
            "mode": "live",
            "city": net.meta.get("label", self.city),
            "mean_kph": round(float(st.kph.mean()), 1),
            "arterial_kph": round(float(st.kph[art].mean()), 1),
            "congested_pct": round(100.0 * float((cong > 0.55).mean()), 1),
            "cameras_mapped": len(self.mapped),
            "cameras_reporting": len(self.observations),
            "data_age_s": round(self.data_age_s) if self.data_age_s else None,
            "edges": int(net.n_edges),
            "inferred_pct": round(100.0 * float((st.support > 0.02).mean()), 1),
            "baselines_calibrated": self.baselines.n_cameras,
            "history_buckets": len(self.hist.by_bucket),
        }

    def live_state(self, max_edges: int = 5000) -> dict:
        net, st = self.net, self.state
        cong = st.congestion(net)
        keep = np.nonzero(net.erank <= 5)[0]
        if len(keep) > max_edges:
            keep = keep[np.argsort(-cong[keep])[:max_edges]]

        # Keyed by camera, not edge: two cameras on one edge must not overwrite
        # each other's reading.
        obs_by_cam = {o.cam_id: o for o in self.observations}
        rows = getattr(self, "latest_rows", {}) or {}
        now = self.now()
        cams = [_camera_row(m, obs_by_cam.get(m.id), rows.get(m.id), now)
                for m in self.mapped]

        return {
            "city": net.meta.get("label", self.city),
            "center": net.meta.get("center"),
            "bbox": net.meta.get("bbox"),
            "clock": time.strftime("%H:%M", time.localtime(now)),
            "ids": [int(e) for e in keep],
            "cong": [round(float(cong[e]), 3) for e in keep],
            "kph": [round(float(st.kph[e]), 1) for e in keep],
            "obs": [int(bool(st.observed[e])) for e in keep],
            "cams": cams,
            "now_s": round(now),
            "replay": self.replay_info(),
            "stats": self.stats(),
        }

    def network_geometry(self) -> dict:
        net = self.net
        keep = np.nonzero(net.erank <= 5)[0]
        return {
            "city": net.meta.get("label", self.city),
            "center": net.meta.get("center"),
            "bbox": net.meta.get("bbox"),
            "edges": [{"id": int(e), "g": net.egeom[e], "r": int(net.erank[e]),
                       "n": net.ename[e], "f": float(net.ekph[e])} for e in keep],
            "cams": [m.as_dict() for m in self.mapped],
            "stats": {
                "nodes": int(net.n_nodes), "edges": int(net.n_edges),
                "km": round(net.total_km, 1),
                "coverage": self.cams.coverage_stats(),
                "nowcast": self.nowcaster.coverage_report(),
            },
        }
