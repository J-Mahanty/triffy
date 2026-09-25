"""Ground-truth traffic simulator.

This module plays two distinct roles, and keeping them straight is what makes
the benchmark honest:

1. It is the *world*. It decides the real speed on every edge at every instant.
   Cameras observe it noisily and sparsely; the router never sees it directly.
2. It is the *referee*. When we compare Triffy against a baseline router, both
   routes are driven through this same world, so the comparison measures routing
   quality rather than luck.

Traffic here is deliberately built from two components:

    v/c(edge, t) = base(edge) * diurnal(t) * incident(edge, t)

* ``base * diurnal`` is the **predictable** part. A model trained on weeks of
  history can learn it, which is exactly what ``forecast.py`` assumes.
* ``incident`` is the **unpredictable** part: crashes, a stalled bus, a flooded
  underpass. Nobody can forecast the onset, but once cameras *detect* it, its
  decay is predictable. That is the part live camera data buys you.

A router that only looks at a snapshot of current speeds mishandles both: it
misses the peak ramping up over the next 30 minutes, and it assumes an incident
that is already clearing will still be there when you arrive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import (BPR_ALPHA, BPR_BETA, BASE_SIGNAL_DELAY_S, DEFAULT_SEED,
                     MAX_SIGNAL_DELAY_S, MIN_SPEED_KPH)
from .network import RoadNetwork

SEC_PER_DAY = 86_400.0


# ---------------------------------------------------------------------------
# Time-of-day demand
# ---------------------------------------------------------------------------

def diurnal_components(t_s: float):
    """Break the day into demand components that different roads feel differently.

    Keeping the morning and evening peaks separate is what lets us model the
    **commute tide**: at 09:00 the roads heading into the business district are
    jammed while the outbound carriageway runs free, and at 19:00 it reverses.
    A single network-wide multiplier cannot express that, and without it a
    time-dependent router has no reason to ever pick a different *path* - only a
    different ETA. The tide is the main reason route choice should change with
    departure time.
    """
    h = (t_s % SEC_PER_DAY) / 3600.0

    def bump(centre, width, height):
        return height * math.exp(-0.5 * ((h - centre) / width) ** 2)

    night_relief = 1.0 - 0.55 * math.exp(-0.5 * ((h - 3.2) / 1.6) ** 2)
    return {
        "base": 0.30,                      # background trickle
        "am": bump(9.4, 1.15, 0.86),       # morning commute
        "lunch": bump(13.2, 1.5, 0.22),    # midday bulge
        "pm": bump(19.0, 1.7, 1.05),       # evening peak, the worst of the day
        "late": bump(22.2, 1.0, 0.18),     # dinner / nightlife
        "_relief": night_relief,
    }


def diurnal(t_s: float | np.ndarray):
    """Network-average demand multiplier, for reporting and quick sanity checks."""
    arr = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    out = []
    for t in arr:
        c = diurnal_components(float(t))
        out.append((c["base"] + c["am"] + c["lunch"] + c["pm"] + c["late"])
                   * c["_relief"])
    res = np.array(out)
    return float(res[0]) if np.isscalar(t_s) or res.size == 1 else res


@dataclass
class Incident:
    """A localised, temporary shock to capacity."""
    edges: np.ndarray
    weights: np.ndarray       # severity per affected edge (1.0 at the epicentre)
    start_s: float
    duration_s: float
    severity: float           # peak multiplier on v/c
    kind: str
    label: str
    road: str = ""

    @property
    def pretty(self) -> str:
        """Lower-case kind, properly-cased road: reads well mid-sentence."""
        return "%s on %s" % (self.kind, self.road or "an unnamed road")

    def factor(self, t_s: float) -> float:
        """Severity envelope: fast onset, slow clearance."""
        if t_s < self.start_s or t_s > self.start_s + self.duration_s:
            return 0.0
        u = (t_s - self.start_s) / self.duration_s
        # Rises within the first 12% of its life, then decays roughly linearly.
        rise = min(1.0, u / 0.12)
        fall = 1.0 - max(0.0, (u - 0.12) / 0.88) ** 1.4
        return float(rise * fall)

    def active(self, t_s: float) -> bool:
        return self.start_s <= t_s <= self.start_s + self.duration_s


class TrafficSim:
    """Deterministic-given-seed mesoscopic traffic world."""

    def __init__(self, net: RoadNetwork, seed: int = DEFAULT_SEED,
                 incident_rate_per_hour: float = 6.0):
        self.net = net
        self.rng = np.random.default_rng(seed)
        self.incident_rate = incident_rate_per_hour
        self.incidents: list[Incident] = []
        self._build_base_demand()
        self._build_upstream()

    # -- static structure ---------------------------------------------------

    def _build_base_demand(self) -> None:
        """Per-edge baseline volume/capacity ratio at demand multiplier 1.0.

        Two effects dominate in a real city and both are reproduced here:
        arterials carry proportionally more load than their extra capacity
        provides, and demand concentrates toward the centre of the area.
        """
        net = self.net
        # Arterials (low rank) run closer to capacity than quiet residentials.
        # Calibrated so a core arterial at the evening peak drops to ~8-10 km/h,
        # which is what Bengaluru actually does, while the parallel residential
        # grid stays around 14-16 km/h. That gap is the whole reason an
        # alternative route can win, so it has to be modelled honestly.
        by_rank = np.array([1.00, 1.12, 1.18, 1.08, 0.92, 0.78, 0.66, 0.50, 0.38])
        base = by_rank[np.clip(net.erank, 0, len(by_rank) - 1)]

        # Radial gradient: the middle of the extract is the busy core.
        clat, clon = float(np.mean(net.lat)), float(np.mean(net.lon))
        mids = np.array([net.edge_midpoint(i) for i in range(net.n_edges)])
        dx = (mids[:, 1] - clon) * net.mx
        dy = (mids[:, 0] - clat) * net.my
        dist_km = np.sqrt(dx * dx + dy * dy) / 1000.0
        radial = 0.65 + 0.50 * np.exp(-(dist_km / 1.7) ** 2)

        # Persistent per-road idiosyncrasy: some streets are simply worse.
        quirk = np.exp(self.rng.normal(0.0, 0.26, size=net.n_edges))

        self.base_vc = base * radial * quirk
        # A handful of notorious corridors, for a recognisable demo.
        self.base_vc *= np.where(self.rng.random(net.n_edges) < 0.045, 1.45, 1.0)

        self._build_tide(clat, clon, mids)

    def _build_tide(self, clat: float, clon: float, mids: np.ndarray) -> None:
        """Per-edge morning/evening weighting: the commute tide.

        An edge pointing toward the centre carries the morning inbound rush; one
        pointing away carries the evening exodus. Because our graph stores both
        directions of a two-way street as separate edges, this automatically
        makes one carriageway jam while the opposite one flows - which is what
        real arterials do, and what a router should be able to exploit.
        """
        net = self.net
        # Unit vector along each edge, in metres.
        head = np.column_stack([net.lat[net.ev], net.lon[net.ev]])
        tail = np.column_stack([net.lat[net.eu], net.lon[net.eu]])
        ex = (head[:, 1] - tail[:, 1]) * net.mx
        ey = (head[:, 0] - tail[:, 0]) * net.my
        enorm = np.maximum(np.hypot(ex, ey), 1e-6)

        # Unit vector from the edge toward the city centre.
        cx = (clon - mids[:, 1]) * net.mx
        cy = (clat - mids[:, 0]) * net.my
        cnorm = np.maximum(np.hypot(cx, cy), 1e-6)

        # +1 = heading straight at the centre, -1 = straight away from it.
        inbound = (ex * cx + ey * cy) / (enorm * cnorm)
        # Edges very near the centre have no meaningful direction; damp them.
        inbound = inbound * np.clip(cnorm / 900.0, 0.0, 1.0)

        TIDE = 0.62
        self.am_weight = 1.0 + TIDE * inbound      # inbound roads jam in the morning
        self.pm_weight = 1.0 - TIDE * inbound      # outbound roads jam in the evening

        # Not every road peaks at the network-average minute.
        self.phase_s = self.rng.normal(0.0, 1500.0, size=net.n_edges)

    def _build_upstream(self) -> None:
        """Precompute 1-2 hop upstream edges, so incidents can queue backwards."""
        net = self.net
        self.upstream: list[np.ndarray] = []
        for eid in range(net.n_edges):
            u = int(net.eu[eid])
            ups = [int(x) for x in net.in_edge_ids(u)]
            self.upstream.append(np.array(ups, dtype=np.int32))

    # -- incidents ----------------------------------------------------------

    def seed_incidents(self, t0_s: float, horizon_s: float) -> None:
        """Populate a Poisson stream of incidents across a time window."""
        n = self.rng.poisson(self.incident_rate * horizon_s / 3600.0)
        for _ in range(int(n)):
            start = t0_s + float(self.rng.random()) * horizon_s
            self.add_incident(start)

    def _busy_edge(self) -> int:
        """A random edge, weighted towards busy main roads.

        Incidents cluster on busy roads, where they also hurt most.
        """
        net = self.net
        w = self.base_vc * (net.erank <= 4)
        if w.sum() <= 0:
            w = np.ones(net.n_edges)
        return int(self.rng.choice(net.n_edges, p=w / w.sum()))

    def _spillback(self, core: list, spill_hops: int | None):
        """Edges and weights of a blockage plus its queue, walked upstream.

        A queue from a partial blockage bleeds a long way back. A *closure*
        behaves differently: traffic is turned away at the barricade and
        diverts, so parallel streets get busier rather than gridlocked. Fewer
        spill hops models a closure; the full ramp models a jam.
        """
        edges = list(core)
        weights = [1.0] * len(core)
        seen = set(core)
        frontier = list(core)
        ramp = (0.78, 0.58, 0.40, 0.26, 0.15)
        if spill_hops is not None:
            ramp = ramp[:max(0, spill_hops)]
        for decay in ramp:
            nxt = []
            for e in frontier:
                for up in self.upstream[e]:
                    up = int(up)
                    if up in seen:
                        continue
                    seen.add(up)
                    edges.append(up)
                    weights.append(decay)
                    nxt.append(up)
            frontier = nxt[:28]
        return edges, weights

    def add_incident(self, start_s: float, edge: int | None = None,
                     severity: float | None = None,
                     duration_s: float | None = None,
                     kind: str | None = None,
                     core_edges: list | None = None,
                     spill_hops: int | None = None) -> Incident:
        """Add a disruption. ``core_edges`` blocks a whole corridor, not one segment.

        A single OSM edge is often only 200-400 m. Real closures - a procession,
        a flooded underpass, an accident with lanes coned off - shut a
        continuous stretch of road, frequently a kilometre or more. Supporting an
        explicit corridor makes the model *more* faithful, not less.
        """
        net = self.net
        if core_edges:
            edge = int(core_edges[0]) if edge is None else edge
        if edge is None:
            edge = self._busy_edge()

        kind = kind or str(self.rng.choice(
            ["collision", "breakdown", "waterlogging", "roadworks", "procession"],
            p=[0.34, 0.26, 0.14, 0.16, 0.10]))
        severity = severity if severity is not None else float(self.rng.uniform(1.6, 3.1))
        duration_s = duration_s if duration_s is not None else float(
            self.rng.uniform(600, 2700))

        # Queue spillback. A blocked arterial in a dense city does not inconvenience
        # 200 m of road; the tail backs up for a kilometre or more and bleeds into
        # every street feeding it. We walk several hops upstream with decaying
        # severity, which is what makes an incident worth routing around at all.
        core = [int(e) for e in (core_edges or [edge])]
        edges, weights = self._spillback(core, spill_hops)

        inc = Incident(
            edges=np.array(edges, dtype=np.int32),
            weights=np.array(weights, dtype=np.float64),
            start_s=start_s, duration_s=duration_s, severity=severity,
            kind=kind,
            label="%s on %s" % (kind.title(), net.describe_edge(edge)),
            road=net.describe_edge(edge),
        )
        self.incidents.append(inc)
        return inc

    def active_incidents(self, t_s: float):
        return [i for i in self.incidents if i.active(t_s)]

    def _incident_multiplier(self, t_s: float) -> np.ndarray:
        mult = np.ones(self.net.n_edges)
        for inc in self.incidents:
            f = inc.factor(t_s)
            if f <= 0.0:
                continue
            # Severity above 1.0 scaled by envelope and per-edge spillback weight.
            mult[inc.edges] *= 1.0 + (inc.severity - 1.0) * f * inc.weights
        return mult

    # -- the world ----------------------------------------------------------

    def vc_ratio(self, t_s: float, with_incidents: bool = True) -> np.ndarray:
        """Volume/capacity on every edge at time t, including the commute tide."""
        c = diurnal_components(t_s)
        level = (c["base"]
                 + c["am"] * self.am_weight
                 + c["lunch"]
                 + c["pm"] * self.pm_weight
                 + c["late"]) * c["_relief"]
        vc = self.base_vc * level
        if with_incidents:
            vc = vc * self._incident_multiplier(t_s)
        return vc

    def speeds(self, t_s: float, with_incidents: bool = True) -> np.ndarray:
        """Ground-truth speed (km/h) on every edge at time t, via a BPR curve."""
        vc = self.vc_ratio(t_s, with_incidents)
        kph = self.net.ekph / (1.0 + BPR_ALPHA * np.power(vc, BPR_BETA))
        return np.maximum(kph, MIN_SPEED_KPH)

    def signal_delay(self, t_s: float, with_incidents: bool = True) -> np.ndarray:
        """Per-edge intersection delay (s) at its downstream junction.

        This is the quantity floating-car GPS data estimates poorly and a camera
        watching the stop line measures directly: how many cycles you wait.
        """
        vc = self.vc_ratio(t_s, with_incidents)
        sat = np.clip(vc, 0.0, 2.2)
        delay = BASE_SIGNAL_DELAY_S + (MAX_SIGNAL_DELAY_S - BASE_SIGNAL_DELAY_S) * \
            (sat ** 2.4) / (1.0 + sat ** 2.4)
        # Only real junctions cost anything; see RoadNetwork._build_junctions.
        return delay * self.net.junction_weight

    def edge_cost_s(self, t_s: float) -> np.ndarray:
        """Total traversal cost (s) of each edge if entered at time t."""
        kph = self.speeds(t_s)
        return self.net.elen / (kph / 3.6) + self.signal_delay(t_s)

    # -- driving a route through the world ----------------------------------

    def drive(self, edges, depart_s: float, driver_factor: float = 1.0):
        """Traverse a route against ground truth. Returns (arrival_s, per_edge_log).

        Crucially the clock advances *while* driving, so a later edge is costed
        at the time you actually reach it. This is the mechanism that punishes a
        router which planned against a stale snapshot.
        """
        net = self.net
        t = float(depart_s)
        log = []
        for eid in edges:
            kph = float(self.speeds(t)[eid]) * driver_factor
            travel = net.elen[eid] / (kph / 3.6)
            delay = float(self.signal_delay(t)[eid])
            t += travel + delay
            log.append({"edge": int(eid), "enter_s": t - travel - delay,
                        "kph": kph, "travel_s": travel, "delay_s": delay})
        return t, log

    def drive_fast(self, edges, depart_s: float, driver_factor: float = 1.0) -> float:
        """Same as ``drive`` but only recomputes the world every 45 s of sim time.

        The benchmark drives thousands of routes; recomputing an 11k-edge speed
        vector per edge is wasteful when traffic barely moves in a few seconds.
        """
        net = self.net
        t = float(depart_s)
        cache_t, kph_v, del_v = -1e18, None, None
        for eid in edges:
            if t - cache_t > 45.0:
                kph_v = self.speeds(t)
                del_v = self.signal_delay(t)
                cache_t = t
            kph = float(kph_v[eid]) * driver_factor
            t += net.elen[eid] / (kph / 3.6) + float(del_v[eid])
        return t


def historical_profile(sim: TrafficSim, t_s: float) -> np.ndarray:
    """What a model trained on weeks of clean history would expect right now.

    Deliberately incident-free: history averages incidents out. Triffy's edge
    over a purely historical model is that cameras reveal *today's* anomaly.
    """
    return sim.speeds(t_s, with_incidents=False)


def fmt_clock(t_s: float) -> str:
    t = t_s % SEC_PER_DAY
    return "%02d:%02d" % (int(t // 3600), int((t % 3600) // 60))


def parse_clock(text: str, default_s: float = 9 * 3600.0) -> float:
    """Parse '8:45', '0845', '8.45am', '18:20' into seconds past midnight."""
    import re
    s = (text or "").strip().lower().replace(".", ":")
    m = re.search(r"(\d{1,2})[:h ]?(\d{2})?\s*(am|pm)?", s)
    if not m:
        return default_s
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    ap = m.group(3)
    if ap == "pm" and hh < 12:
        hh += 12
    if ap == "am" and hh == 12:
        hh = 0
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    return hh * 3600.0 + mm * 60.0
