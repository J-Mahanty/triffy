"""Head-to-head evaluation: does Triffy actually route better?

**What this does and does not claim.** We cannot prove anything about Google
Maps or Waze here; we have no access to their routing, and a fair comparison
would need matched trips and weeks of ground truth. What we *can* do - and what
this module does - is re-implement the approach those apps take and beat it
under identical conditions.

Three routers compete:

* ``triffy``    - time-dependent forecast, live camera nowcast, risk-aware cost
* ``snapshot``   - live speeds frozen at departure, the standard "live traffic"
                   approach: it knows current conditions but has no forward model
* ``historical`` - typical-day profiles, no live data at all

Every router plans its own route. Then **all of their routes are driven through
the same simulated world**, edge by edge, with the clock advancing as the
vehicle moves. Nobody is scored against their own optimistic estimate. That
separation - plan with your beliefs, get scored by reality - is what makes the
numbers mean something.

Two metrics matter, and they are different:

1. **Travel time**: did the route you chose actually get there sooner?
2. **ETA error**: was the time you *promised* the user correct? A snapshot
   router systematically under-promises during a building peak, which is the
   failure commuters actually complain about.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from .config import DATA
from .engine import TriffyEngine
from .router import Router
from .simulator import fmt_clock


@dataclass
class TripResult:
    origin: int
    dest: int
    depart_s: float
    method: str
    claimed_s: float        # what the router promised
    actual_s: float         # what the world delivered
    promised_p90_s: float   # the buffered promise
    distance_m: float
    disrupted: bool = False   # did this trip's corridor touch a live incident?

    @property
    def eta_error_s(self) -> float:
        return self.actual_s - self.claimed_s

    @property
    def on_time(self) -> bool:
        return self.actual_s <= self.promised_p90_s


METHODS = ("triffy", "snapshot", "historical")


class Benchmark:
    def __init__(self, engine: TriffyEngine, seed: int = 99):
        self.eng = engine
        self.rng = np.random.default_rng(seed)
        self.results: list[TripResult] = []

    # -- trip sampling ------------------------------------------------------

    def _anchor_nodes(self) -> list:
        """Distinct landmark nodes, computed once: the activity centres."""
        anchors = getattr(self, "_anchors", None)
        if anchors is None:
            seen, anchors = set(), []
            for p in self.eng.net.landmarks.values():
                if p.node not in seen:
                    seen.add(p.node)
                    anchors.append(p.node)
            self._anchors = anchors
        return anchors

    def _draw_pair(self, anchors: list, use_anchor: bool):
        """One candidate (origin, destination) node pair."""
        if use_anchor:
            a, b = (int(x) for x in self.rng.choice(anchors, size=2, replace=False))
            return a, b
        a = int(self.rng.integers(self.eng.net.n_nodes))
        b = int(self.rng.integers(self.eng.net.n_nodes))
        return a, b

    def sample_od(self, min_km: float = 1.8, max_km: float = 7.0):
        """Pick an origin/destination pair a real commuter might actually make.

        Uniform random node pairs are a poor model of demand: most of them are
        short hops between quiet residential blocks where every router trivially
        agrees, which drowns the comparison in ties. Real commuting flows between
        activity centres along congested corridors, so most trips here are drawn
        between named landmarks, with the rest random to keep coverage broad.
        """
        net = self.eng.net
        anchors = self._anchor_nodes()
        use_anchor = len(anchors) >= 4 and self.rng.random() < 0.65
        for _ in range(240):
            a, b = self._draw_pair(anchors, use_anchor)
            if a == b:
                continue
            d = net.straight_line(a, b) / 1000.0
            if min_km <= d <= max_km:
                return a, b
            if use_anchor and d > 0.4:
                return a, b            # landmark pairs are legitimate at any range
        return 0, net.n_nodes - 1

    # -- one trip -----------------------------------------------------------

    def run_trip(self, src: int, dst: int, depart_s: float, user) -> dict:
        """Plan with each router, then drive all three routes through reality."""
        eng = self.eng
        out = {}

        # The snapshot baseline plans on data that is deliberately ``lag`` old:
        # see TriffyEngine.state_at_lag for why that is the fair setting.
        stale = eng.state_at_lag(eng.baseline_lag_s)
        profiles = {
            "triffy": eng.forecaster.build_profile(eng.state, depart_s),
            "snapshot": eng.forecaster.snapshot_profile(stale, depart_s),
            "historical": eng.forecaster.historical_only_profile(depart_s),
        }

        hot = self._incident_edges()
        paths = {}
        for name, prof in profiles.items():
            user.adapt_profile(eng.net, prof)
            router = Router(eng.net, prof, prefs=user)
            routes = router.route(src, dst, depart_s, k=1)
            if not routes:
                return {}
            r = routes[0]
            paths[name] = r.edges

            # The world decides. Driver skill is applied here too, so the
            # comparison is between route choices, not between drivers.
            actual = eng.sim.drive_fast(r.edges, depart_s,
                                        driver_factor=self._driver_factor(user))
            out[name] = TripResult(
                origin=src, dest=dst, depart_s=depart_s, method=name,
                claimed_s=r.mean_s, actual_s=actual - depart_s,
                promised_p90_s=r.p90_s, distance_m=r.distance_m,
            )

        # A trip counts as disrupted when the *traffic-unaware* route runs into a
        # live incident: the natural path had something in the way. Defining the
        # subgroup from the historical router keeps it independent of the two
        # systems being compared. Using either contender's chosen path would
        # define the subgroup by the outcome we are trying to measure.
        disrupted = bool(hot) and bool(hot & set(paths.get("historical", [])))
        for tr in out.values():
            tr.disrupted = disrupted
        return out

    def _incident_edges(self) -> set:
        eng = self.eng
        hot = set()
        for inc in eng.sim.active_incidents(eng.now_s):
            if inc.factor(eng.now_s) > 0.25:
                hot.update(int(e) for e in inc.edges)
        return hot

    def _corridor_disrupted(self, src: int, dst: int, hot: set,
                            radius_m: float = 700.0) -> bool:
        """Is there a live incident anywhere near the straight line src->dst?

        Deliberately generous. The question the subgroup asks is 'was there
        something to route around on this corridor', not 'did a particular
        router drive into it'.
        """
        net = self.eng.net
        alat, alon = net.node_latlon(src)
        blat, blon = net.node_latlon(dst)
        ax, ay = alon * net.mx, alat * net.my
        bx, by = blon * net.mx, blat * net.my
        vx, vy = bx - ax, by - ay
        vlen2 = max(vx * vx + vy * vy, 1e-6)
        for e in hot:
            mlat, mlon = net.edge_midpoint(e)
            px, py = mlon * net.mx, mlat * net.my
            t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / vlen2))
            cx, cy = ax + t * vx, ay + t * vy
            if (px - cx) ** 2 + (py - cy) ** 2 < radius_m * radius_m:
                return True
        return False

    def _driver_factor(self, user) -> float:
        """Vehicle effects the simulator should honour when driving the route."""
        return float(user.driver_skill)

    # -- the sweep ----------------------------------------------------------

    def run(self, n_trips: int = 180, departures=None, user_id: str = "guest",
            verbose: bool = True):
        """Sweep trips across the day. One nowcast per departure slot.

        Ticking the engine once per time slot (rather than once per trip) is
        both faster and more faithful: every commuter querying at 18:30 shares
        the same live camera picture.
        """
        eng = self.eng
        user = eng.profiles.get(user_id)
        departures = departures or ["08:00", "08:45", "09:15", "09:45",
                                    "12:30", "15:00",
                                    "17:30", "18:15", "18:45", "19:15",
                                    "20:00", "22:00"]
        per_slot = max(1, n_trips // len(departures))
        t_start = time.time()

        for slot in departures:
            eng.set_clock(slot)
            eng.ensure_live_incidents(3)
            eng.tick()
            depart_s = eng.now_s
            got = 0
            for _ in range(per_slot):
                src, dst = self.sample_od()
                res = self.run_trip(src, dst, depart_s, user)
                if not res:
                    continue
                self.results.extend(res.values())
                got += 1
            if verbose:
                print("  %s  %3d trips  (%5.1fs elapsed)"
                      % (slot, got, time.time() - t_start))
        return self.summarise()

    # -- reporting ----------------------------------------------------------

    def summarise(self) -> dict:
        by = {m: [r for r in self.results if r.method == m] for m in METHODS}
        n = len(by["triffy"])
        if n == 0:
            return {}

        def agg(rs):
            actual = [r.actual_s for r in rs]
            err = [abs(r.eta_error_s) for r in rs]
            bias = [r.eta_error_s for r in rs]
            return {
                "n": len(rs),
                "mean_travel_min": statistics.mean(actual) / 60.0,
                "median_travel_min": statistics.median(actual) / 60.0,
                "mean_abs_eta_error_min": statistics.mean(err) / 60.0,
                "median_abs_eta_error_min": statistics.median(err) / 60.0,
                "eta_bias_min": statistics.mean(bias) / 60.0,
                "on_time_pct": 100.0 * sum(r.on_time for r in rs) / len(rs),
                "mean_km": statistics.mean([r.distance_m for r in rs]) / 1000.0,
            }

        stats = {m: agg(by[m]) for m in METHODS}

        # Paired comparison: same trip, same departure, different router.
        pairs = {}
        for rival in ("snapshot", "historical"):
            deltas, wins, ties = [], 0, 0
            for a, b in zip(by["triffy"], by[rival]):
                d = b.actual_s - a.actual_s          # positive = Triffy faster
                deltas.append(d)
                if d > 20:
                    wins += 1
                elif abs(d) <= 20:
                    ties += 1
            tot = max(1, len(deltas))
            pairs[rival] = {
                "mean_saving_min": statistics.mean(deltas) / 60.0,
                "median_saving_min": statistics.median(deltas) / 60.0,
                "pct_saving": 100.0 * statistics.mean(deltas) /
                              max(1.0, statistics.mean([r.actual_s for r in by[rival]])),
                "win_pct": 100.0 * wins / tot,
                "tie_pct": 100.0 * ties / tot,
                "loss_pct": 100.0 * (tot - wins - ties) / tot,
                "eta_error_reduction_pct": 100.0 * (
                    1.0 - stats["triffy"]["mean_abs_eta_error_min"] /
                    max(1e-9, stats[rival]["mean_abs_eta_error_min"])),
            }

        # Subgroup: trips whose corridor actually had something to route around.
        # Averaging over trips where every router correctly agrees dilutes the
        # effect we are measuring, so both figures are reported side by side.
        subgroups = {}
        for tag, want in (("disrupted", True), ("clear", False)):
            sel = {m: [r for r in by[m] if r.disrupted == want] for m in METHODS}
            if len(sel["triffy"]) < 5:
                continue
            g = {"n": len(sel["triffy"]), "stats": {m: agg(sel[m]) for m in METHODS}}
            deltas = [b.actual_s - a.actual_s
                      for a, b in zip(sel["triffy"], sel["snapshot"])]
            tot = max(1, len(deltas))
            g["vs_snapshot"] = {
                "mean_saving_min": statistics.mean(deltas) / 60.0,
                "pct_saving": 100.0 * statistics.mean(deltas) /
                              max(1.0, statistics.mean([r.actual_s for r in sel["snapshot"]])),
                "win_pct": 100.0 * sum(1 for d in deltas if d > 20) / tot,
                "loss_pct": 100.0 * sum(1 for d in deltas if d < -20) / tot,
            }
            subgroups[tag] = g

        return {"stats": stats, "pairs": pairs, "n_trips": n,
                "subgroups": subgroups,
                "baseline_lag_s": self.eng.baseline_lag_s,
                "city": self.eng.net.meta.get("label", ""),
                "cameras": len(self.eng.cams.cams),
                "edges": self.eng.net.n_edges}

    def report(self, summary: dict) -> str:
        if not summary:
            return "No results."
        s, p = summary["stats"], summary["pairs"]
        L = []
        L.append("=" * 74)
        L.append("TRIFFY BENCHMARK  -  %s" % summary["city"])
        L.append("%d trips  |  %d cameras  |  %d directed edges"
                 % (summary["n_trips"], summary["cameras"], summary["edges"]))
        L.append("=" * 74)
        L.append("")
        L.append("%-12s %10s %10s %12s %11s %9s"
                 % ("router", "travel", "vs best", "ETA error", "ETA bias", "on-time"))
        L.append("%-12s %10s %10s %12s %11s %9s"
                 % ("", "(min)", "", "(min, abs)", "(min)", "(%)"))
        L.append("-" * 74)
        best = min(s[m]["mean_travel_min"] for m in METHODS)
        for m in METHODS:
            d = s[m]
            L.append("%-12s %10.1f %10s %12.2f %11.2f %9.1f"
                     % (m, d["mean_travel_min"],
                        "best" if abs(d["mean_travel_min"] - best) < 1e-9
                        else "+%.1f" % (d["mean_travel_min"] - best),
                        d["mean_abs_eta_error_min"], d["eta_bias_min"],
                        d["on_time_pct"]))
        L.append("")
        L.append("PAIRED COMPARISON (same trip, same departure, same world)")
        L.append("-" * 74)
        for rival, d in p.items():
            L.append("  Triffy vs %s:" % rival)
            L.append("    travel time saved   %+6.2f min/trip  (%+.1f%%)"
                     % (d["mean_saving_min"], d["pct_saving"]))
            L.append("    ETA error reduced   %5.1f%%" % d["eta_error_reduction_pct"])
            L.append("    win / tie / loss    %.0f%% / %.0f%% / %.0f%%"
                     % (d["win_pct"], d["tie_pct"], d["loss_pct"]))
        sg = summary.get("subgroups", {})
        if sg:
            L.append("")
            L.append("WHERE THE VALUE IS  (Triffy vs snapshot, by trip type)")
            L.append("-" * 74)
            names = {"disrupted": "corridor had a live incident",
                     "clear": "network clear on this corridor"}
            for tag in ("disrupted", "clear"):
                if tag not in sg:
                    continue
                g = sg[tag]
                v = g["vs_snapshot"]
                L.append("  %-32s n=%-4d  %+5.2f min (%+.1f%%)  win %.0f%% / loss %.0f%%"
                         % (names[tag], g["n"], v["mean_saving_min"],
                            v["pct_saving"], v["win_pct"], v["loss_pct"]))
            L.append("")
            L.append("  Routing only differs when there is something to route around.")
            L.append("  Averaged over a quiet network every sensible router agrees,")
            L.append("  which is why the headline figure is modest and the")
            L.append("  disrupted-corridor figure is the one that matters.")
        L.append("")
        L.append("ASSUMPTIONS")
        L.append("-" * 74)
        L.append("  * 'snapshot' reimplements the live-traffic approach of mainstream")
        L.append("    navigation apps. It is NOT Google Maps or Waze itself, and no")
        L.append("    claim about their real-world performance is made here.")
        L.append("  * Baseline plans on data lagged by %d s, modelling floating-car GPS"
                 % round(summary.get("baseline_lag_s", 0)))
        L.append("    aggregation latency. Triffy uses camera data at zero lag.")
        L.append("  * All routers are scored by driving their chosen route through the")
        L.append("    same simulated world, never against their own estimate.")
        L.append("  * Triffy's historical model carries a per-edge bias, so it does")
        L.append("    not secretly know the simulator's ground truth.")
        L.append("=" * 74)
        return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Triffy against baselines")
    ap.add_argument("--trips", type=int, default=180)
    ap.add_argument("--user", default="guest")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default=str(DATA / "benchmark.json"))
    args = ap.parse_args()

    print("Booting engine...")
    eng = TriffyEngine()
    bm = Benchmark(eng, seed=args.seed)
    print("Running %d trips across the day..." % args.trips)
    summary = bm.run(n_trips=args.trips, user_id=args.user)
    print()
    print(bm.report(summary))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
