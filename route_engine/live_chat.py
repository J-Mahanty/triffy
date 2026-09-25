"""Let the chat brain answer from live London as well as simulated Kolkata.

``ChatBrain`` was written against ``TriffyEngine``: a ``plan()`` that returns a
``Plan``, a clock, a ``leave_by`` search and network statistics. Rather than a
second brain that would drift from the first, this adapter gives the real-data
``LiveEngine`` that same surface. Three things genuinely differ:

* **Time is real.** The simulated engine keeps its own clock; London's is the
  wall clock in London. Times are exposed as seconds since London's local
  midnight, which is exactly what ``fmt_clock`` and ``parse_clock`` expect, so
  "Waterloo to Bank at 18:30" means half past six in London wherever the
  server happens to be.
* **There is no clock to move and no simulated incident list.** Instead of
  incidents, London offers its most congested camera-watched roads, which is
  the honest equivalent: what the cameras are seeing right now.
* **One profile store per process.** The dashboard server holds both engines,
  and each used to load its own ``ProfileStore`` over the same file, so
  whichever saved last silently erased the other's changes. The adapter makes
  the live engine share the simulated engine's store.
"""
from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .engine import NOT_ON_MAP, Plan, _baseline_bits, _join_bits, latest_departure
from .simulator import fmt_clock, parse_clock

LONDON = ZoneInfo("Europe/London")
MAX_AHEAD_S = 12 * 3600.0       # the forecast is anchored on what cameras see now
LEANS_ON_HISTORY_S = 3600.0     # beyond this the usual pattern dominates


class LiveChatEngine:
    """``TriffyEngine``'s chat-facing surface, answered from real cameras."""

    is_live = True

    def __init__(self, live, profiles=None):
        self.live = live
        self.city = live.city
        self.net = live.net
        if profiles is not None:
            live.profiles = profiles

    # -- the surface ChatBrain uses -------------------------------------------

    @property
    def profiles(self):
        return self.live.profiles

    @property
    def state(self):
        return self.live.state

    def _midnight(self) -> float:
        """Epoch seconds of the most recent local midnight in London."""
        now = datetime.now(LONDON)
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    @property
    def now_s(self) -> float:
        return time.time() - self._midnight()

    @property
    def clock(self) -> str:
        return fmt_clock(self.now_s)

    def resolve(self, text: str):
        return self.net.resolve(text)

    def plan(self, origin: str, destination: str, depart=None,
             user_id: str = "guest", k: int = 3, with_baseline: bool = True) -> Plan:
        self.live.ensure_fresh()
        o, d = self.resolve(origin), self.resolve(destination)
        if o is None:
            raise ValueError(NOT_ON_MAP % origin)
        if d is None:
            raise ValueError(NOT_ON_MAP % destination)
        if o.node == d.node:
            raise ValueError("Origin and destination are the same place.")

        now = self.now_s
        depart_s = self._depart_s(depart, now)
        ahead = depart_s - now
        if ahead > MAX_AHEAD_S:
            raise ValueError("Live mode forecasts from what the cameras see now, "
                             "so I can plan up to 12 hours ahead.")
        routes, baseline = self.live.plan_routes(
            o.node, d.node, user_id=user_id, k=k,
            depart=self._midnight() + depart_s, with_baseline=with_baseline)
        plan = Plan(origin=o.name, destination=d.name, depart_s=depart_s,
                    routes=routes, user=self.profiles.get(user_id),
                    baseline=baseline)
        plan.advisory = self._advisory(plan, ahead)
        return plan

    def leave_by(self, origin: str, destination: str, arrive_by: str,
                 user_id: str = "guest", confidence: float = 0.9):
        return latest_departure(self, origin, destination, arrive_by,
                                user_id=user_id, confidence=confidence)

    def network_stats(self) -> dict:
        s = self.live.stats()
        return {
            "arterial_kph": s["arterial_kph"],
            "congested_pct": s["congested_pct"],
            "incidents_active": 0,
            "cameras_online": s["cameras_reporting"],
            "cameras_total": s["cameras_mapped"],
            "inferred_pct": s["inferred_pct"],
            "data_age_s": s["data_age_s"],
        }

    # -- live-only extras ------------------------------------------------------

    def hotspots(self, limit: int = 6) -> list:
        """The most congested camera-watched roads: (label, congestion 0-1, age_s)."""
        now = time.time()
        obs = sorted(self.live.observations, key=lambda o: -o.occupancy)
        out = []
        for o in obs[:limit]:
            cam = self.live.by_id.get(o.cam_id)
            label = cam.name if cam else (self.net.ename[o.edge] or "an unnamed road")
            out.append((label, float(o.occupancy), max(0.0, now - o.t_s)))
        return out

    def cameras_on(self, edges, limit: int = 3) -> list:
        """Cameras watching this route, in the order the route passes them."""
        order = {int(e): i for i, e in enumerate(edges)}
        hits = [m for m in self.live.mapped if int(m.edge) in order]
        hits.sort(key=lambda m: order[int(m.edge)])
        return hits[:limit]

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _depart_s(depart, now: float) -> float:
        if depart is None:
            return now
        if isinstance(depart, (int, float)):
            depart_s = float(depart)
        else:
            depart_s = parse_clock(depart, now)
        # An earlier time of day than now means tomorrow, as in the simulator.
        if depart_s < now - 600:
            depart_s += 86400.0
        return depart_s

    def _advisory(self, plan: Plan, ahead_s: float) -> str:
        r = plan.best
        bits = _baseline_bits(plan.baseline, r) if plan.baseline is not None else []
        if r.reliability > 0.90:
            bits.append("and is unusually predictable right now")
        elif r.reliability < 0.75:
            bits.append("though conditions ahead are volatile, so leave a buffer")
        out = ["This route " + _join_bits(bits) + "."] if bits else []

        share = self.live.live_share(r.edges)
        age = self.live.data_age_s
        out.append("Live cameras inform %d%% of it%s."
                   % (round(share["informed_pct"]),
                      " (readings about %d min old)" % max(1, round(age / 60)) if age else ""))
        if ahead_s > LEANS_ON_HISTORY_S:
            out.append("That far ahead, it leans on the usual pattern for that "
                       "time more than on what the cameras see now.")
        return " ".join(out)
