"""Time-dependent, risk-aware routing.

Two ideas separate this from a textbook shortest-path search.

**Time dependence.** The cost of an edge depends on *when you enter it*. The
search therefore carries a clock: when it relaxes an edge it looks up the
forecast speed at the time the driver would actually arrive there. A snapshot
router costs the whole route at departure time and is systematically wrong about
everything after the first few minutes.

**Risk awareness.** We minimise ``mean + lambda * sigma``, not mean alone. A
commuter with a 09:30 standup does not want the route with the best average; they
want the route that is rarely late. Because the nowcaster reports honest
uncertainty, ``lambda`` lets a user dial between "fastest" and "most
predictable", and that dial is the core of personalisation.

Implementation note: the search state is an **edge**, not a node. Turn penalties
depend on how you arrived at a junction, so a node-keyed search cannot express
"this left turn across four lanes of Chowringhee costs you 40 seconds". With
~13k edges the extra state is cheap.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .config import (ALTERNATIVES, DIVERSITY_PENALTY, MAX_SETTLED_NODES,
                     SHARP_TURN_PENALTY_S, TURN_PENALTY_S)
from .forecast import SpeedProfile
from .network import RoadNetwork

# Travel times on consecutive edges are positively correlated: the same
# congestion wave slows all of them. Summing variances as if independent
# understates route-level spread, so we inflate the total.
CORRELATION_INFLATION = 1.28

# Congestion bands for colouring a route along its length: free flowing,
# slowing, congested, near gridlock. The same cut-offs the dashboard uses for
# the traffic layer, read from speed through the live engine's speed model
# (speed = free * (1 - 0.8 * congestion)), so a route and the roads under it
# are coloured by the same rule.
TRAFFIC_BANDS = (0.30, 0.52, 0.72)


def traffic_level(speed_ratio: float) -> int:
    """0 (free flowing) .. 3 (near gridlock), from speed as a share of free flow."""
    congestion = (1.0 - speed_ratio) / 0.8
    return sum(congestion >= b for b in TRAFFIC_BANDS)


@dataclass
class Step:
    """One human-readable instruction."""
    instruction: str
    road: str
    distance_m: float
    seconds: float
    edges: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"instruction": self.instruction, "road": self.road,
                "distance_m": round(self.distance_m), "seconds": round(self.seconds)}


@dataclass
class Route:
    edges: list
    depart_s: float
    mean_s: float
    sigma_s: float
    distance_m: float
    label: str = ""
    steps: list = field(default_factory=list)
    cost: float = 0.0
    # What the router that *chose* this path believed it would take. For the
    # snapshot baseline this differs from ``mean_s``, and the gap is precisely
    # the ETA error caused by planning against frozen conditions.
    claimed_s: float = 0.0

    # -- ETA distribution ---------------------------------------------------
    # Travel time is right-skewed: you can always be much later than expected,
    # never much earlier. A log-normal fit captures that, whereas a symmetric
    # interval would quote impossible early arrivals.

    @property
    def cv(self) -> float:
        return self.sigma_s / max(self.mean_s, 1.0)

    @property
    def _s(self) -> float:
        return math.sqrt(math.log(1.0 + self.cv ** 2))

    @property
    def median_s(self) -> float:
        return self.mean_s / math.sqrt(1.0 + self.cv ** 2)

    def percentile_s(self, p: float) -> float:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(p)
        return self.median_s * math.exp(z * self._s)

    @property
    def p90_s(self) -> float:
        return self.percentile_s(0.90)

    @property
    def p10_s(self) -> float:
        return self.percentile_s(0.10)

    @property
    def reliability(self) -> float:
        """0..1. 1.0 means the p90 equals the median: utterly predictable."""
        return float(np.clip(self.median_s / max(self.p90_s, 1.0), 0.0, 1.0))

    @property
    def arrival_s(self) -> float:
        return self.depart_s + self.mean_s

    def geometry(self, net: RoadNetwork) -> list:
        pts = []
        for e in self.edges:
            g = net.egeom[e]
            if pts and pts[-1] == g[0]:
                pts.extend(g[1:])
            else:
                pts.extend(g)
        return pts

    def as_dict(self, net: RoadNetwork) -> dict:
        return {
            "label": self.label,
            "depart_s": round(self.depart_s),
            "mean_s": round(self.mean_s),
            "median_s": round(self.median_s),
            "p10_s": round(self.p10_s),
            "p90_s": round(self.p90_s),
            "sigma_s": round(self.sigma_s),
            "reliability": round(self.reliability, 3),
            "distance_m": round(self.distance_m),
            "claimed_s": round(self.claimed_s) if self.claimed_s else None,
            "n_edges": len(self.edges),
            # Needed by the dashboard so it can ask the engine to close a road
            # this route actually depends on.
            "edge_ids": [int(e) for e in self.edges],
            "steps": [s.as_dict() for s in self.steps],
            "geometry": self.geometry(net),
            "roads": _main_roads(net, self.edges),
        }


def _main_roads(net: RoadNetwork, edges, top: int = 4) -> list:
    """The few streets that characterise a route, by distance carried."""
    acc: dict[str, float] = {}
    for e in edges:
        nm = net.ename[e]
        if nm:
            acc[nm] = acc.get(nm, 0.0) + float(net.elen[e])
    return [k for k, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:top]]


class _SearchState:
    """Per-edge bookkeeping for one A* search."""

    def __init__(self, n_edges: int):
        inf = float("inf")
        self.g_cost = np.full(n_edges, inf)   # risk-adjusted cost, drives the search
        self.g_time = np.full(n_edges, inf)   # expected clock time, drives the forecast
        self.g_var = np.zeros(n_edges)
        self.parent = np.full(n_edges, -1, dtype=np.int32)
        self.closed = np.zeros(n_edges, dtype=bool)

    def path_to(self, best_edge: int, depart_s: float):
        """Walk parents back from ``best_edge``: (edges, mean_s, var_s2)."""
        edges = []
        cur = best_edge
        while cur >= 0:
            edges.append(int(cur))
            cur = int(self.parent[cur])
        edges.reverse()
        mean_s = self.g_time[best_edge] - depart_s
        var_s2 = self.g_var[best_edge] * (CORRELATION_INFLATION ** 2)
        return edges, float(mean_s), float(var_s2)


class Router:
    def __init__(self, net: RoadNetwork, profile: SpeedProfile, prefs=None):
        self.net = net
        self.profile = profile
        self.prefs = prefs
        self.vmax_kph = float(max(net.ekph.max(), 1.0))

    # -- cost model ---------------------------------------------------------

    def _turn_cost(self, prev_edge: int, next_edge: int) -> float:
        """Penalty for the manoeuvre between two edges.

        Sharp turns are slow everywhere, but in left-hand-traffic India a right
        turn crosses oncoming traffic and is materially worse than a left.
        Staying on the same named road is free.
        """
        net = self.net
        if net.ename[prev_edge] and net.ename[prev_edge] == net.ename[next_edge]:
            return 0.0
        ang = net.turn_angle(prev_edge, next_edge)
        if ang < 22.0:
            return 0.0
        # Signed turn: positive = right (across traffic in India).
        delta = (float(net.ebear_in[next_edge]) - float(net.ebear_out[prev_edge])
                 + 540.0) % 360.0 - 180.0
        base = SHARP_TURN_PENALTY_S if ang > 70.0 else TURN_PENALTY_S
        if delta > 45.0:
            base *= 1.8            # right turn across oncoming traffic
        if ang > 150.0:
            base += 25.0           # U-turn
        w = getattr(self.prefs, "turn_weight", 1.0) if self.prefs else 1.0
        return base * w

    def _edge_pref_mult(self, eid: int) -> float:
        """Personal distaste for certain road types, as a cost multiplier."""
        if self.prefs is None:
            return 1.0
        return self.prefs.edge_multiplier(self.net, eid)

    # -- the search ---------------------------------------------------------

    def _search(self, src: int, dst: int, depart_s: float,
                penalty: np.ndarray | None = None, lam: float | None = None):
        """Edge-based time-dependent A*. Returns (edges, mean_s, var_s2) or None."""
        net = self.net
        lam = self.prefs.risk_aversion if (lam is None and self.prefs) else (lam or 0.0)

        # Admissible heuristic: straight-line distance at the fastest free-flow
        # speed anywhere in the network can never overestimate true travel time.
        inv_vmax = 3.6 / self.vmax_kph

        def h(edge: int) -> float:
            v = int(net.ev[edge])
            return net.straight_line(v, dst) * inv_vmax

        s = _SearchState(net.n_edges)
        pq = []
        for e in net.out_edge_ids(src):
            e = int(e)
            mean, sd = self._leg_cost(e, depart_s, penalty)
            c = mean + lam * sd
            s.g_cost[e] = c
            s.g_time[e] = depart_s + mean
            s.g_var[e] = sd * sd
            heapq.heappush(pq, (c + h(e), e))

        best_edge = self._settle(pq, s, dst, h, penalty, lam)
        if best_edge < 0:
            return None
        return s.path_to(best_edge, depart_s)

    def _leg_cost(self, e: int, t_s: float, penalty):
        """Mean and sd of entering edge ``e`` at ``t_s``, with this user's
        road preferences and any diversity penalty applied to the mean."""
        mean, sd = self.profile.traverse_s(self.net, e, t_s)
        mean *= self._edge_pref_mult(e)
        if penalty is not None:
            mean *= penalty[e]
        return mean, sd

    def _settle(self, pq, s, dst: int, h, penalty, lam: float) -> int:
        """Pop edges until the destination is settled; return it, or -1."""
        ev = self.net.ev
        settled = 0
        while pq:
            _, e = heapq.heappop(pq)
            if s.closed[e]:
                continue
            s.closed[e] = True
            settled += 1
            if settled > MAX_SETTLED_NODES:
                return -1
            if int(ev[e]) == dst:
                return e
            self._expand(e, pq, s, h, penalty, lam)
        return -1

    def _expand(self, e: int, pq, s, h, penalty, lam: float) -> None:
        """Relax every edge leaving the end of settled edge ``e``.

        One call per settled edge rather than per neighbour keeps the hot loop's
        function-call overhead where it was; the leg cost is inlined for the
        same reason.
        """
        net, prof = self.net, self.profile
        g_cost, g_time, g_var = s.g_cost, s.g_time, s.g_var
        t_here = g_time[e]
        twin = int(net.etwin[e])
        for nxt in net.out_edge_ids(int(net.ev[e])):
            nxt = int(nxt)
            # Never immediately reverse down the same street.
            if s.closed[nxt] or twin == nxt:
                continue
            mean, sd = prof.traverse_s(net, nxt, t_here)
            mean *= self._edge_pref_mult(nxt)
            if penalty is not None:
                mean *= penalty[nxt]
            turn = self._turn_cost(e, nxt)
            step_cost = mean + turn + lam * sd
            cand = g_cost[e] + step_cost
            if cand < g_cost[nxt]:
                g_cost[nxt] = cand
                g_time[nxt] = t_here + mean + turn
                g_var[nxt] = g_var[e] + sd * sd
                s.parent[nxt] = e
                heapq.heappush(pq, (cand + h(nxt), nxt))

    # -- public API ---------------------------------------------------------

    def route(self, src: int, dst: int, depart_s: float,
              k: int = ALTERNATIVES) -> list:
        """Return up to ``k`` meaningfully different routes, best first."""
        net = self.net
        out: list[Route] = []
        penalty = np.ones(net.n_edges)

        for i in range(k):
            res = self._search(src, dst, depart_s, penalty if i else None)
            if res is None:
                break
            edges, mean_s, var_s2 = res

            # Recompute the honest cost of this path with no diversity penalty,
            # so alternatives are reported at their true travel time.
            mean_s, var_s2 = self._evaluate(edges, depart_s)

            r = Route(edges=edges, depart_s=depart_s, mean_s=mean_s,
                      sigma_s=math.sqrt(var_s2),
                      distance_m=float(sum(net.elen[e] for e in edges)))
            if any(_overlap(r.edges, o.edges) > 0.72 for o in out):
                # Too similar to something we already have: push harder and retry.
                penalty[edges] *= DIVERSITY_PENALTY
                continue
            r.steps = self._directions(edges, depart_s)
            out.append(r)
            penalty[edges] *= DIVERSITY_PENALTY

        for i, r in enumerate(out):
            r.label = _label_route(r, out, i)
        return out

    def _evaluate(self, edges, depart_s: float):
        """Cost an explicit path against the forecast, advancing the clock."""
        net, prof = self.net, self.profile
        t = depart_s
        var = 0.0
        prev = None
        for e in edges:
            mean, sd = prof.traverse_s(net, e, t)
            if prev is not None:
                mean += self._turn_cost(prev, e)
            t += mean
            var += sd * sd
            prev = e
        return float(t - depart_s), float(var * CORRELATION_INFLATION ** 2)

    def evaluate_path(self, edges, depart_s: float) -> Route:
        mean_s, var = self._evaluate(edges, depart_s)
        r = Route(edges=list(edges), depart_s=depart_s, mean_s=mean_s,
                  sigma_s=math.sqrt(var),
                  distance_m=float(sum(self.net.elen[e] for e in edges)))
        r.steps = self._directions(edges, depart_s)
        return r

    # -- turn-by-turn -------------------------------------------------------

    def _directions(self, edges, depart_s: float) -> list:
        """Collapse an edge list into instructions a human can follow."""
        net, prof = self.net, self.profile
        steps: list[Step] = []
        t = depart_s
        run_edges: list[int] = []
        run_name = None
        run_dist = 0.0
        run_time = 0.0

        def flush():
            nonlocal run_edges, run_name, run_dist, run_time
            if not run_edges:
                return
            road = run_name or ("%s road" % net.ehw[run_edges[0]].replace("_", " "))
            if not steps:
                verb = "Head out on"
            else:
                verb = _turn_verb(net, steps[-1].edges[-1], run_edges[0])
            steps.append(Step(instruction="%s %s" % (verb, road), road=road,
                              distance_m=run_dist, seconds=run_time,
                              edges=list(run_edges)))
            run_edges, run_name, run_dist, run_time = [], None, 0.0, 0.0

        for e in edges:
            nm = net.ename[e] or ("%s road" % net.ehw[e].replace("_", " "))
            if run_name is not None and nm != run_name:
                flush()
            run_name = nm
            mean, _ = prof.traverse_s(net, e, t)
            t += mean
            run_dist += float(net.elen[e])
            run_time += mean
            run_edges.append(e)
        flush()

        steps = _compact_steps(steps)
        if steps:
            steps.append(Step(instruction="Arrive at your destination",
                              road=steps[-1].road, distance_m=0.0, seconds=0.0,
                              edges=[edges[-1]]))
        return steps


def _compact_steps(steps, min_m: float = 130.0) -> list:
    """Fold away instructions too short to speak aloud.

    OSM splits streets constantly, so a raw instruction list contains things like
    "Turn right onto Jhowtala Road (16 m)". Nobody navigates like that. Short
    hops are absorbed into the preceding instruction, which is what a human
    giving directions would do.
    """
    if not steps:
        return steps
    out = [steps[0]]
    for s in steps[1:]:
        prev = out[-1]
        if s.distance_m < min_m and len(out) > 0:
            prev.distance_m += s.distance_m
            prev.seconds += s.seconds
            prev.edges.extend(s.edges)
            continue
        out.append(s)
    # A tiny opening instruction is equally useless; merge it forward.
    if len(out) > 1 and out[0].distance_m < min_m:
        out[1].distance_m += out[0].distance_m
        out[1].seconds += out[0].seconds
        out[1].edges = out[0].edges + out[1].edges
        out = out[1:]
    if out:
        # Whatever manoeuvre opened the trip, the first step reads "Start on".
        first = out[0].instruction
        for verb in ("Turn left onto", "Turn right onto", "Bear left onto",
                     "Bear right onto", "Continue onto", "Head out on"):
            first = first.replace(verb, "Start on")
        out[0].instruction = first
    return out


def _turn_verb(net: RoadNetwork, prev_edge: int, next_edge: int) -> str:
    ang = net.turn_angle(prev_edge, next_edge)
    delta = (float(net.ebear_in[next_edge]) - float(net.ebear_out[prev_edge])
             + 540.0) % 360.0 - 180.0
    if ang < 20.0:
        return "Continue onto"
    if ang > 150.0:
        return "Make a U-turn onto"
    side = "right" if delta > 0 else "left"
    if ang > 70.0:
        return "Turn %s onto" % side
    return "Bear %s onto" % side


def _overlap(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _label_route(r: Route, all_routes, i: int) -> str:
    """Name an alternative by what actually makes it worth offering.

    A label has to earn itself. Calling a route 60% slower "most predictable"
    because its reliability is 0.01 higher is technically true and completely
    useless, so a claim only sticks if the trade-off is one a person would
    plausibly take.
    """
    if i == 0:
        return "Recommended"
    best = all_routes[0]
    faster = (best.median_s - r.median_s) / max(best.median_s, 1.0)
    shorter = (best.distance_m - r.distance_m) / max(best.distance_m, 1.0)
    steadier = r.reliability - best.reliability

    if steadier > 0.025 and faster > -0.18:
        return "More predictable"
    if faster > 0.03:
        return "Quicker, less certain"
    if shorter > 0.08:
        return "Shorter, slower"
    if r.distance_m > best.distance_m * 1.08:
        return "Wider roads"
    return "Alternative %d" % i
