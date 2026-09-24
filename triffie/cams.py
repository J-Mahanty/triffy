"""Traffic camera network: placement, observation model, and the live feed.

Two ideas drive this module.

**Placement is an optimisation problem.** A city cannot instrument 11,000 road
segments; it can afford a few hundred cameras. Where they go decides how much of
the network you can infer. We place greedily by value (road importance x junction
degree, because a camera at a busy junction constrains many downstream edges)
while enforcing spatial spread so the whole budget does not land on one corridor.

**The observation model is real traffic physics.** A camera does not measure
speed directly; it counts vehicles in a known field of view and tracks them
across frames. Count converts to density, and density relates to flow and speed
through the fundamental diagram of traffic flow:

    flow q [veh/h] = density k [veh/km] * speed v [km/h]

So a camera that sees 34 vehicles over 120 m of a 3-lane road is reporting a
density of ~94 veh/km/lane, which pins down speed on the congested branch of the
curve. That is why camera data is genuinely informative and not just a proxy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import CAM_COVERAGE, CAM_SPEED_NOISE, MIN_SPEED_KPH
from .network import RoadNetwork


@dataclass
class Camera:
    """A pole-mounted camera watching one approach to a junction."""
    id: str
    edge: int
    node: int
    lat: float
    lon: float
    road: str
    fov_m: float = 110.0          # length of carriageway in frame
    health: float = 1.0           # 1.0 = clean feed; degrades at night / in rain

    def as_dict(self) -> dict:
        return {"id": self.id, "edge": int(self.edge), "lat": self.lat,
                "lon": self.lon, "road": self.road, "fov_m": self.fov_m}


@dataclass
class CamObservation:
    """One analysed frame-window from one camera."""
    cam_id: str
    edge: int
    t_s: float
    vehicle_count: int
    density_vpkm: float           # per lane
    speed_kph: float
    queue_m: float
    occupancy: float              # 0..1 fraction of visible road covered by metal
    confidence: float             # 0..1, how much the nowcaster should trust this
    source: str = "sim"           # "sim" or "cv" (real computer vision)
    classes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "cam_id": self.cam_id, "edge": int(self.edge), "t_s": round(self.t_s, 1),
            "count": int(self.vehicle_count), "density": round(self.density_vpkm, 1),
            "kph": round(self.speed_kph, 1), "queue_m": round(self.queue_m, 1),
            "occupancy": round(self.occupancy, 3), "confidence": round(self.confidence, 3),
            "source": self.source, "classes": self.classes,
        }


class CameraNetwork:
    """The deployed fleet, plus the model that turns ground truth into pixels."""

    def __init__(self, net: RoadNetwork, coverage: float = CAM_COVERAGE,
                 seed: int = 7, budget: int | None = None):
        self.net = net
        self.rng = np.random.default_rng(seed)
        target = budget if budget is not None else max(12, int(net.n_edges * coverage))
        self.cams: list[Camera] = self._place(target)
        self.by_edge: dict[int, Camera] = {c.edge: c for c in self.cams}
        self.covered = np.zeros(net.n_edges, dtype=bool)
        self.covered[[c.edge for c in self.cams]] = True

    # -- placement ----------------------------------------------------------

    def _place(self, budget: int) -> list[Camera]:
        net = self.net
        # Value of watching an edge: important road, feeding a complex junction,
        # and long enough that a camera sees a meaningful stretch of it.
        rank_w = np.array([1.00, 0.95, 0.90, 0.78, 0.60, 0.42, 0.30, 0.18, 0.10])
        value = rank_w[np.clip(net.erank, 0, len(rank_w) - 1)]

        deg = np.diff(net.out_start).astype(np.float64)      # out-degree per node
        junction_val = deg[net.ev]                            # complexity downstream
        value = value * (0.55 + 0.45 * np.clip(junction_val / 4.0, 0, 1.6))
        value = value * np.clip(net.elen / 140.0, 0.30, 1.0)

        # Never place two cameras on opposite directions of the same street.
        order = np.argsort(-value)
        chosen: list[int] = []
        taken_twin: set[int] = set()
        min_sep_m = 150.0
        pts: list[tuple[float, float]] = []

        for eid in order:
            eid = int(eid)
            if len(chosen) >= budget:
                break
            if eid in taken_twin:
                continue
            mlat, mlon = net.edge_midpoint(eid)
            x, y = mlon * net.mx, mlat * net.my
            # Spatial spread: reject anything too close to an existing camera.
            too_close = False
            for (px, py) in pts:
                if (px - x) ** 2 + (py - y) ** 2 < min_sep_m ** 2:
                    too_close = True
                    break
            if too_close:
                continue
            chosen.append(eid)
            pts.append((x, y))
            tw = int(net.etwin[eid])
            if tw >= 0:
                taken_twin.add(tw)

        cams = []
        for i, eid in enumerate(chosen):
            mlat, mlon = net.edge_midpoint(eid)
            cams.append(Camera(
                id="CAM%03d" % (i + 1),
                edge=eid, node=int(net.ev[eid]),
                lat=float(mlat), lon=float(mlon),
                road=net.describe_edge(eid),
                fov_m=float(min(150.0, max(60.0, net.elen[eid] * 0.55))),
            ))
        return cams

    # -- observation model --------------------------------------------------

    def _health(self, t_s: float) -> float:
        """Feeds degrade at night: glare, headlights, lower contrast."""
        h = (t_s % 86400.0) / 3600.0
        night = 1.0 if (h < 6.0 or h > 19.0) else 0.0
        return 0.78 if night else 1.0

    def observe(self, sim, t_s: float, jitter: bool = True):
        """Produce one observation per camera from simulator ground truth.

        This is the *synthetic* path used for the live demo and the benchmark.
        ``vision.py`` provides the real computer-vision path that produces the
        identical ``CamObservation`` shape from actual video, so downstream code
        cannot tell the two apart.
        """
        net = self.net
        kph_all = sim.speeds(t_s)
        vc_all = sim.vc_ratio(t_s)
        health = self._health(t_s)
        out = []

        for cam in self.cams:
            e = cam.edge
            v = float(kph_all[e])
            vc = float(vc_all[e])

            # Flow from v/c, then density from the fundamental diagram k = q / v.
            q = vc * float(net.ecap[e])                     # veh/h across all lanes
            lanes = max(1, int(net.elanes[e]))
            k_lane = q / max(v, MIN_SPEED_KPH) / lanes      # veh/km/lane

            # Vehicles actually inside the field of view.
            true_count = k_lane * lanes * (cam.fov_m / 1000.0)

            # Occlusion: in dense traffic a mono camera under-counts, because
            # vehicles hide behind each other. This is a real, well-known bias.
            occ_ratio = np.clip(k_lane / 120.0, 0.0, 1.0)
            detect_rate = 1.0 - 0.28 * occ_ratio ** 1.5
            count = true_count * detect_rate * health

            if jitter:
                count = max(0.0, count + self.rng.normal(0.0, 0.9 + 0.05 * count))

            # Speed from multi-frame tracking; noisier when few vehicles are
            # visible, because the estimate averages over fewer tracks.
            n_eff = max(1.0, count)
            rel_sigma = CAM_SPEED_NOISE * (1.0 + 2.2 / math.sqrt(n_eff)) / health
            v_obs = v * (1.0 + (self.rng.normal(0.0, rel_sigma) if jitter else 0.0))
            v_obs = float(max(MIN_SPEED_KPH * 0.8, v_obs))

            # Queue length: the standing-vehicle tail back from the stop line.
            # This is the measurement that phone-GPS floating-car data misses.
            jam_k = 135.0
            stopped_frac = float(np.clip((k_lane - 28.0) / (jam_k - 28.0), 0.0, 1.0))
            queue_m = stopped_frac * cam.fov_m * (0.6 + 0.8 * min(1.0, vc / 1.4))

            occupancy = float(np.clip(k_lane / jam_k, 0.0, 1.0))

            # Confidence: clean feed, enough vehicles to average over, and not
            # so dense that occlusion dominates.
            conf = health * (1.0 - 0.35 * occ_ratio) * min(1.0, 0.45 + 0.1 * n_eff)
            conf = float(np.clip(conf, 0.12, 0.99))

            out.append(CamObservation(
                cam_id=cam.id, edge=e, t_s=t_s,
                vehicle_count=int(round(count)),
                density_vpkm=float(k_lane),
                speed_kph=v_obs,
                queue_m=float(queue_m),
                occupancy=occupancy,
                confidence=conf,
                source="sim",
                classes=self._class_mix(count),
            ))
        return out

    def _class_mix(self, count: float) -> dict:
        """Indian urban traffic is two-wheeler dominated; the mix matters.

        A stream that is 55% motorcycles behaves very differently from a
        car-dominated one: two-wheelers filter through gaps, so the same density
        produces a higher speed. The nowcaster uses this.
        """
        n = int(round(max(0.0, count)))
        if n == 0:
            return {}
        frac = np.array([0.52, 0.28, 0.07, 0.05, 0.08])   # moto, car, bus, truck, auto
        counts = self.rng.multinomial(n, frac)
        names = ["motorcycle", "car", "bus", "truck", "auto"]
        return {k: int(v) for k, v in zip(names, counts) if v}

    # -- convenience --------------------------------------------------------

    def as_geojson(self) -> dict:
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                "properties": c.as_dict(),
            } for c in self.cams],
        }

    def coverage_stats(self) -> dict:
        net = self.net
        watched_km = float(net.elen[self.covered].sum() / 1000.0)
        arterial = net.erank <= 3
        return {
            "cameras": len(self.cams),
            "edges_watched": int(self.covered.sum()),
            "edges_total": net.n_edges,
            "pct_edges": round(100.0 * self.covered.sum() / net.n_edges, 1),
            "pct_arterial_edges": round(
                100.0 * (self.covered & arterial).sum() / max(1, arterial.sum()), 1),
            "km_watched": round(watched_km, 1),
        }
