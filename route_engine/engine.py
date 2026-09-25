"""The Triffy engine: one object that owns the whole pipeline.

Every interface - web dashboard, Discord bot, CLI, benchmark - talks to this
class, so they cannot drift apart. It holds the simulated world, the camera
fleet, the nowcaster, the forecaster and the profile store, and exposes three
things a commuter actually wants:

* ``plan(...)``          - routes from A to B leaving now or later
* ``leave_by(...)``      - the latest safe departure for a hard deadline
* ``live_state()``       - what the network looks like right now, for the map
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .cams import CameraNetwork
from .config import ACTIVE_CITY, ALTERNATIVES
from .forecast import Forecaster, HistoricalModel
from .network import load_network
from .nowcast import Nowcaster
from .personalize import ProfileStore, UserProfile
from .router import Route, Router
from .simulator import TrafficSim, fmt_clock, parse_clock


@dataclass
class Plan:
    """The answer to one routing question."""
    origin: str
    destination: str
    depart_s: float
    routes: list
    user: UserProfile
    baseline: Route | None = None      # what a snapshot router would have picked
    advisory: str = ""

    @property
    def best(self):
        return self.routes[0] if self.routes else None

    def as_dict(self, net) -> dict:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "depart_s": round(self.depart_s),
            "depart_clock": fmt_clock(self.depart_s),
            "user": {"id": self.user.user_id, "name": self.user.name,
                     "vehicle": self.user.vehicle,
                     "risk_aversion": round(self.user.risk_aversion, 2)},
            "advisory": self.advisory,
            "routes": [r.as_dict(net) for r in self.routes],
            "baseline": self.baseline.as_dict(net) if self.baseline else None,
        }


NOT_ON_MAP = "I could not find %r on the map."


def _baseline_bits(baseline, r) -> list:
    """What this route does better than the snapshot router's pick, if anything."""
    saved = baseline.median_s - r.median_s
    if saved > 60:
        return ["saves about %d min versus the route a snapshot app "
                "would pick" % max(1, round(saved / 60.0))]
    if baseline.claimed_s:
        # Same road, better promise: the honest ETA is still the win.
        err = baseline.mean_s - baseline.claimed_s
        if err > 90:
            return ["comes with a realistic ETA (a snapshot app would "
                    "have under-promised by about %d min)" % max(1, round(err / 60.0))]
    return []


def _join_bits(bits: list) -> str:
    """Join clauses cleanly: "A, B and C" rather than "A, and B"."""
    if len(bits) == 1:
        body = bits[0]
        for lead in ("and ", "though "):
            if body.startswith(lead):
                body = body[len(lead):]
        return body
    if bits[-1].startswith(("and ", "though ")):
        return ", ".join(bits[:-1]) + " " + bits[-1]
    return ", ".join(bits[:-1]) + " and " + bits[-1]


def latest_departure(planner, origin: str, destination: str, arrive_by: str,
                     user_id: str = "guest", confidence: float = 0.9):
    """The search behind ``leave_by``, for any engine with ``now_s`` and
    ``plan()``. Returns ((depart_s, plan) or None, deadline_s)."""
    deadline_s = parse_clock(arrive_by, planner.now_s)
    if deadline_s < planner.now_s:
        deadline_s += 86400.0

    lo, hi = planner.now_s, deadline_s
    best = None
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        try:
            plan = planner.plan(origin, destination, mid, user_id=user_id,
                                k=1, with_baseline=False)
        except ValueError:
            break
        r = plan.best
        arrival = mid + r.percentile_s(confidence)
        if arrival <= deadline_s:
            best = (mid, plan)
            lo = mid           # we can afford to leave later
        else:
            hi = mid
    return best, deadline_s


class TriffyEngine:
    def __init__(self, city: str = ACTIVE_CITY, seed: int = 20260919,
                 cam_budget: int = 400, start_clock: str = "18:30"):
        self.city = city
        self.net = load_network(city)
        self.sim = TrafficSim(self.net, seed=seed)
        self.cams = CameraNetwork(self.net, budget=cam_budget, seed=seed + 1)
        self.nowcaster = Nowcaster(self.net, self.cams)
        self.hist = HistoricalModel(self.sim, seed=seed + 2)
        self.forecaster = Forecaster(self.net, self.hist)
        self.profiles = ProfileStore()

        self.now_s = parse_clock(start_clock, 18.5 * 3600)
        # Populate a day of incidents so the world is already alive on first tick.
        self.sim.seed_incidents(self.now_s - 3600.0, 8 * 3600.0)
        self.ensure_live_incidents(3)
        self.state = None
        self.observations = []
        self._history: list = []
        self.baseline_lag_s = 300.0   # see state_at_lag(): GPS aggregation latency
        self._wall_anchor = time.time()
        self.tick()

    def ensure_live_incidents(self, n: int = 3) -> list:
        """Guarantee there is something happening right now.

        A Poisson process is the honest way to generate incidents, but it is
        also free to hand you an empty road network at the exact moment you
        start a demo. This tops up the count without changing the model: the
        added incidents are ordinary draws, merely placed so their window
        covers the current clock.
        """
        added = []
        while len(self.sim.active_incidents(self.now_s)) < n:
            start = self.now_s - float(self.sim.rng.uniform(120, 900))
            inc = self.sim.add_incident(start,
                                        duration_s=float(self.sim.rng.uniform(1500, 3000)))
            added.append(inc)
        return added

    def _watched_edge(self, edges: list) -> int:
        """The edge to block in the middle third of a route.

        Middle third: blocking the first edge leaves no alternative, and
        blocking the last is too late to route around. Within it, prefer a
        stretch the cameras can see. With 400 cameras over 13k edges, a
        blockage on an unwatched road changes ground truth without changing
        anyone's *belief* - true of sparse sensing, but it makes a demo where
        nothing visibly happens. A monitored edge keeps the scenario honest
        ("a blockage on a monitored corridor") and legible.
        """
        net = self.net
        lo, hi = len(edges) // 3, max(len(edges) // 3 + 1, 2 * len(edges) // 3)
        window = edges[lo:hi] or edges
        reach = self.nowcaster.reach
        return int(max(window, key=lambda e: (reach[e], net.ecap[e])))

    def _corridor_around(self, edges: list, edge: int) -> list:
        """About a kilometre of route centred on ``edge``.

        Close a corridor, not a single 300 m segment: walk outward along the
        route until roughly a kilometre is shut, as a real closure would be.
        """
        elen = self.net.elen
        i = edges.index(edge)
        span, lo_i, hi_i = float(elen[edge]), i, i
        last = len(edges) - 1
        while span < 1100.0 and (lo_i > 0 or hi_i < last):
            if lo_i > 0:
                lo_i -= 1
                span += float(elen[edges[lo_i]])
            if hi_i < last and span < 1100.0:
                hi_i += 1
                span += float(elen[edges[hi_i]])
        return edges[lo_i:hi_i + 1]

    def inject_incident(self, where: str = "", severity: float = 2.6,
                        minutes: float = 30.0, kind: str = "collision",
                        on_route: list | None = None):
        """Drop an incident on the network: the live demo's 'watch this' moment.

        Two modes, and the difference matters for a demo that has to work.

        Naming a road blocks the busiest edge at that junction, which is
        realistic but may land nowhere near the route on screen - in which case
        nothing visible happens, because correctly, nothing should.

        Passing ``on_route`` blocks a road the current recommendation actually
        uses, chosen from the middle of the trip so there is still room to
        divert. That is the honest "what if this road closed?" question, and it
        reliably shows the router re-thinking.
        """
        net = self.net
        if on_route:
            edges = [int(e) for e in on_route]
            edge = self._watched_edge(edges)
            core = self._corridor_around(edges, edge)
            # Start it far enough back that the onset ramp has completed. The
            # severity envelope rises over the first 12% of an incident's life,
            # so injecting at now-60s would show a blockage at a quarter
            # strength and look like the system had ignored it.
            dur = minutes * 60.0
            inc = self.sim.add_incident(self.now_s - 0.25 * dur, edge=edge,
                                        severity=severity,
                                        duration_s=dur, kind=kind,
                                        core_edges=core, spill_hops=1)
            self.tick()
            return inc

        place = self.resolve(where)
        if place is None:
            raise ValueError(NOT_ON_MAP % where)
        cands = list(net.out_edge_ids(place.node)) or [0]
        edge = int(max(cands, key=lambda e: net.ecap[int(e)]))

        inc = self.sim.add_incident(self.now_s - 60.0, edge=edge, severity=severity,
                                    duration_s=minutes * 60.0, kind=kind)
        self.tick()
        return inc

    # -- clock --------------------------------------------------------------

    def set_clock(self, text: str) -> float:
        self.now_s = parse_clock(text, self.now_s)
        self._wall_anchor = time.time()
        self.tick()
        return self.now_s

    def advance(self, seconds: float) -> float:
        self.now_s += seconds
        self.tick()
        return self.now_s

    @property
    def clock(self) -> str:
        return fmt_clock(self.now_s)

    # -- the live loop ------------------------------------------------------

    def tick(self, observations=None):
        """Pull a frame from every camera and refresh the network belief."""
        self.observations = (observations if observations is not None
                             else self.cams.observe(self.sim, self.now_s))
        self._apply_cv()
        hist_kph = self.hist.expected_kph(self.now_s)
        self.state = self.nowcaster.update(self.observations, hist_kph, self.now_s)
        self._history.append((self.now_s, self.state))
        while len(self._history) > 64:
            self._history.pop(0)
        return self.state

    def _apply_cv(self) -> None:
        """Splice in a reading produced by real computer vision, if one is fresh.

        Nothing downstream knows the difference, which is the whole point: the
        CV path and the simulated path emit the same object, so the nowcaster
        needs no special case for real video.
        """
        try:
            from .cv_bridge import apply_to_observations, read_observation
            blob = read_observation()
            if blob:
                self.cv_active = apply_to_observations(self.observations, blob, self.net)
            else:
                self.cv_active = False
        except Exception:
            self.cv_active = False

    def state_at_lag(self, lag_s: float):
        """The network picture as it looked ``lag_s`` ago.

        This exists to model a real asymmetry rather than to handicap anyone.
        Camera sensing reports a blocked lane within seconds. Crowdsourced
        floating-car data has to wait for enough vehicles to traverse a segment
        before its aggregate speed moves, and incident confirmation is slower
        still. Any comparison that hands the GPS-based baseline camera-latency
        data is quietly overstating that baseline.
        """
        if lag_s <= 0 or not self._history:
            return self.state
        cutoff = self.now_s - lag_s
        older = [s for (t, s) in self._history if t <= cutoff]
        return older[-1] if older else self._history[0][1]

    # -- planning -----------------------------------------------------------

    def _profile_for(self, user: UserProfile, depart_s: float, horizon_s=5400.0):
        prof = self.forecaster.build_profile(self.state, depart_s, horizon_s=horizon_s)
        user.adapt_profile(self.net, prof)
        return prof

    def _snapshot_profile_for(self, user: UserProfile, depart_s: float,
                              horizon_s=5400.0, lag_s: float | None = None):
        lag = self.baseline_lag_s if lag_s is None else lag_s
        prof = self.forecaster.snapshot_profile(self.state_at_lag(lag), depart_s,
                                                horizon_s=horizon_s)
        user.adapt_profile(self.net, prof)
        return prof

    def resolve(self, text: str):
        return self.net.resolve(text)

    def plan(self, origin: str, destination: str, depart: str | float | None = None,
             user_id: str = "guest", k: int = ALTERNATIVES,
             with_baseline: bool = True) -> Plan:
        """Plan a journey. ``depart`` may be a clock string, seconds, or None (now)."""
        o = self.resolve(origin)
        d = self.resolve(destination)
        if o is None:
            raise ValueError(NOT_ON_MAP % origin)
        if d is None:
            raise ValueError(NOT_ON_MAP % destination)
        if o.node == d.node:
            raise ValueError("Origin and destination are the same place.")

        if depart is None:
            depart_s = self.now_s
        elif isinstance(depart, (int, float)):
            depart_s = float(depart)
        else:
            depart_s = parse_clock(depart, self.now_s)
        # A departure earlier in the day than "now" means tomorrow morning.
        if depart_s < self.now_s - 600:
            depart_s += 86400.0

        user = self.profiles.get(user_id)
        prof = self._profile_for(user, depart_s)
        router = Router(self.net, prof, prefs=user)
        routes = router.route(o.node, d.node, depart_s, k=k)
        if not routes:
            raise ValueError("No route found between those two points.")

        baseline = None
        if with_baseline:
            baseline = self._baseline_route(user, o.node, d.node, depart_s, router)

        plan = Plan(origin=o.name, destination=d.name, depart_s=depart_s,
                    routes=routes, user=user, baseline=baseline)
        plan.advisory = self._advisory(plan)
        return plan

    def _baseline_route(self, user: UserProfile, src: int, dst: int,
                        depart_s: float, td_router: Router):
        """What a snapshot router would choose, scored under the real forecast.

        The point of the comparison is that both routes are judged by the same
        forecast. The baseline is penalised only for *choosing* with stale
        information, which is exactly the effect we claim to fix.
        """
        snap_prof = self._snapshot_profile_for(user, depart_s)
        snap_router = Router(self.net, snap_prof, prefs=user)
        picks = snap_router.route(src, dst, depart_s, k=1)
        if not picks:
            return None
        honest = td_router.evaluate_path(picks[0].edges, depart_s)
        honest.label = "Snapshot router (baseline)"
        # What the snapshot router *told its user* the trip would take. The gap
        # between this and the honest figure is the ETA error we are fixing.
        honest.claimed_s = picks[0].mean_s
        return honest

    def leave_by(self, origin: str, destination: str, arrive_by: str,
                 user_id: str = "guest", confidence: float = 0.9):
        """Latest departure that still hits a deadline with ``confidence``.

        Uses the p90 (not the average) arrival, because "usually on time" is not
        what someone with a hard deadline is asking for. Binary search over
        departure time, because travel time is not monotone in departure time -
        leaving later can mean arriving *earlier* once a peak passes.
        """
        return latest_departure(self, origin, destination, arrive_by,
                                user_id=user_id, confidence=confidence)

    # -- narrative ----------------------------------------------------------

    def _advisory(self, plan: Plan) -> str:
        """A one-line 'why this route' explanation, which is what sells the app."""
        r = plan.best
        if r is None:
            return ""
        bits = []
        if plan.baseline is not None:
            bits += _baseline_bits(plan.baseline, r)
        hot = self._incidents_near(r)
        if hot:
            bits.append("steers clear of %s" % hot[0])
        if r.reliability > 0.90:
            bits.append("and is unusually predictable right now")
        elif r.reliability < 0.75:
            bits.append("though conditions ahead are volatile, so leave a buffer")

        if not bits:
            return ""
        return "This route " + _join_bits(bits) + "."



    def _incidents_near(self, route: Route, radius_m: float = 450.0):
        """Name active incidents the chosen route steers clear of."""
        net = self.net
        on_route = set(route.edges)
        out = []
        for inc in self.sim.active_incidents(self.now_s):
            if any(int(e) in on_route for e in inc.edges):
                continue
            # Only mention it if it was plausibly in the way.
            elat, elon = net.edge_midpoint(int(inc.edges[0]))
            for e in route.edges[::4]:
                rlat, rlon = net.edge_midpoint(e)
                dx = (rlon - elon) * net.mx
                dy = (rlat - elat) * net.my
                if dx * dx + dy * dy < radius_m * radius_m:
                    out.append(inc.pretty)
                    break
        return out

    # -- dashboard feed -----------------------------------------------------

    def live_state(self, max_edges: int = 4000) -> dict:
        """Congestion overlay + camera readings for the map."""
        net, st = self.net, self.state
        cong = st.congestion(net)
        # Draw the roads that matter: all arterials, plus the worst side streets.
        keep = np.nonzero(net.erank <= 4)[0]
        if len(keep) > max_edges:
            keep = keep[np.argsort(-cong[keep])[:max_edges]]

        edges = [{
            "id": int(e),
            "g": net.egeom[e],
            "c": round(float(cong[e]), 3),
            "kph": round(float(st.kph[e]), 1),
            "obs": bool(st.observed[e]),
            "sup": round(float(st.support[e]), 2),
            "name": net.ename[e],
        } for e in keep]

        obs_by_edge = {o.edge: o for o in self.observations}
        cams = []
        for c in self.cams.cams:
            o = obs_by_edge.get(c.edge)
            cams.append({
                "id": c.id, "lat": c.lat, "lon": c.lon, "road": c.road,
                "kph": round(o.speed_kph, 1) if o else None,
                "count": o.vehicle_count if o else None,
                "queue_m": round(o.queue_m) if o else None,
                "occupancy": round(o.occupancy, 2) if o else None,
                "classes": o.classes if o else {},
                "source": o.source if o else "offline",
            })

        incidents = [{
            "label": inc.label, "kind": inc.kind,
            "lat": net.edge_midpoint(int(inc.edges[0]))[0],
            "lon": net.edge_midpoint(int(inc.edges[0]))[1],
            "severity": round(inc.severity, 2),
            "started": fmt_clock(inc.start_s),
            "clears": fmt_clock(inc.start_s + inc.duration_s),
            "intensity": round(inc.factor(self.now_s), 2),
        } for inc in self.sim.active_incidents(self.now_s)]

        return {
            "city": self.net.meta.get("label", self.city),
            "center": self.net.meta.get("center", [22.555, 88.355]),
            "clock": self.clock,
            "now_s": round(self.now_s),
            "edges": edges,
            "cams": cams,
            "incidents": incidents,
            "stats": self.network_stats(),
        }

    def network_stats(self) -> dict:
        net, st = self.net, self.state
        art = net.erank <= 3
        cong = st.congestion(net)
        return {
            "mean_kph": round(float(st.kph.mean()), 1),
            "arterial_kph": round(float(st.kph[art].mean()), 1),
            "congested_pct": round(100.0 * float((cong > 0.55).mean()), 1),
            "cameras_online": len(self.observations),
            "cameras_total": len(self.cams.cams),
            "edges": net.n_edges,
            "incidents_active": len(self.sim.active_incidents(self.now_s)),
            "inferred_pct": round(100.0 * float((st.support > 0.02).mean()), 1),
        }
