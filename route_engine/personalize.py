"""Personalisation: the same road network, costed for a specific human.

Most routing apps personalise almost nothing. They may let you avoid tolls or
highways, but the ETA and the chosen path are otherwise identical for a delivery
rider on a scooter and a family in a hatchback. Three things here change that.

**Vehicle physics.** A two-wheeler is not a slow car. In free-flowing traffic it
is no quicker, but in a jam it filters between lanes and can be close to twice as
fast. The adjustment therefore has to scale with congestion:

    v_effective = v * (1 + filter_bonus * congestion)

In a city where a large share of commuters ride two-wheelers, this single term
changes which route wins, not merely by how much.

**Risk appetite.** ``risk_aversion`` (lambda) is the dial between "fastest on
average" and "least likely to make me late". Someone with a 09:30 stand-up
should be given the predictable route even when it is slower on paper.

**Learned preferences.** Thumbs-down on a route nudges the weights of the roads
it used, so the system drifts toward what this particular person actually
accepts - narrow lanes, flyovers, left-turn-heavy paths.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import DATA
from .network import RoadNetwork

PROFILES_PATH = DATA / "profiles.json"


@dataclass
class UserProfile:
    """Everything Triffy knows about one commuter."""
    user_id: str
    name: str = "Commuter"
    vehicle: str = "car"              # car | motorcycle | auto | taxi
    risk_aversion: float = 0.8        # lambda: 0 = pure speed, 2.5 = paranoid
    turn_weight: float = 1.0          # how much this driver hates manoeuvres
    avoid_narrow: float = 0.0         # 0..1 distaste for residential/service lanes
    prefer_arterial: float = 0.0      # 0..1 preference for big roads
    driver_skill: float = 1.0         # >1 drives faster than the ambient stream
    home: str = ""
    work: str = ""
    usual_depart: str = ""
    trips_logged: int = 0
    road_bias: dict = field(default_factory=dict)   # road name -> cost multiplier
    created_s: float = field(default_factory=time.time)
    # The chat city this user last used ("" = the default), so a Telegram user
    # who switched to live London stays there after a server restart.
    city: str = ""
    # Saved places for cities other than the default one, e.g.
    # {"lon": {"home": "Waterloo", "work": "Bank"}}. ``home``/``work`` above
    # stay the default city's, so existing profiles read exactly as before.
    places: dict = field(default_factory=dict)

    # -- vehicle physics ----------------------------------------------------

    @property
    def filter_bonus(self) -> float:
        """How much this vehicle gains from filtering through stopped traffic."""
        return {"motorcycle": 0.95, "auto": 0.42, "car": 0.0, "taxi": 0.08}.get(
            self.vehicle, 0.0)

    @property
    def size_penalty(self) -> float:
        """How badly this vehicle copes with narrow lanes."""
        return {"motorcycle": 0.0, "auto": 0.10, "car": 0.30, "taxi": 0.30}.get(
            self.vehicle, 0.2)

    def adapt_profile(self, net: RoadNetwork, prof) -> None:
        """Rewrite a forecast speed tensor for this driver, in place.

        Personalising the *speeds* rather than only the costs means the quoted
        ETA is itself personalised. A rider is told when they will really arrive,
        not a generic estimate with a preference nudge applied afterwards.
        """
        free = np.maximum(net.ekph, 1.0)[None, :]
        congestion = np.clip(1.0 - prof.kph / free, 0.0, 1.0)

        # Filtering only helps once traffic is actually stopped.
        gain = 1.0 + self.filter_bonus * (congestion ** 1.5)
        # Skill is a mild, uniform effect.
        gain = gain * self.driver_skill
        prof.kph = np.minimum(prof.kph * gain, free * 1.05)

        # A two-wheeler is barely delayed at a crowded junction; a car queues.
        if self.filter_bonus > 0:
            prof.delay_s = prof.delay_s * (1.0 - 0.45 * self.filter_bonus)

    # -- cost preferences ---------------------------------------------------

    def edge_multiplier(self, net: RoadNetwork, eid: int) -> float:
        """Personal cost multiplier on one edge (1.0 = neutral)."""
        m = 1.0
        rank = int(net.erank[eid])

        if self.avoid_narrow > 0.0 and rank >= 6:
            m *= 1.0 + 0.85 * self.avoid_narrow
        if self.prefer_arterial > 0.0 and rank <= 3:
            m *= 1.0 - 0.18 * self.prefer_arterial
        # Vehicle bulk: a car genuinely struggles on a living street.
        if rank >= 7:
            m *= 1.0 + self.size_penalty

        if self.road_bias:
            nm = net.ename[eid]
            if nm and nm in self.road_bias:
                m *= float(self.road_bias[nm])
        return m

    # -- learning -----------------------------------------------------------

    def record_feedback(self, net: RoadNetwork, route, verdict: str,
                        strength: float = 0.08) -> dict:
        """Update preferences from a thumbs up / down on a completed route.

        Deliberately gentle: a single bad trip should nudge, not overturn. The
        roads that carried the most distance absorb most of the adjustment,
        because they are what the user is actually reacting to.
        """
        changed = {}
        total = max(1.0, sum(float(net.elen[e]) for e in route.edges))
        by_road: dict[str, float] = {}
        for e in route.edges:
            nm = net.ename[e]
            if nm:
                by_road[nm] = by_road.get(nm, 0.0) + float(net.elen[e])

        direction = -1.0 if verdict in ("up", "good", "yes") else 1.0
        for nm, dist in by_road.items():
            share = dist / total
            cur = float(self.road_bias.get(nm, 1.0))
            new = float(np.clip(cur + direction * strength * share * 4.0, 0.6, 1.8))
            if abs(new - cur) > 1e-3:
                self.road_bias[nm] = round(new, 3)
                changed[nm] = round(new, 3)

        # A late arrival makes the user more risk-averse next time.
        if direction > 0:
            self.risk_aversion = float(np.clip(self.risk_aversion + 0.12, 0.0, 2.5))
        else:
            self.risk_aversion = float(np.clip(self.risk_aversion - 0.04, 0.0, 2.5))
        self.trips_logged += 1
        return changed

    def summary(self) -> str:
        if self.risk_aversion > 1.2:
            style = "plays it safe"
        elif self.risk_aversion > 0.5:
            style = "balanced"
        else:
            style = "chases the fastest"
        return ("%s on a %s, %s (lambda=%.2f), %d trips logged"
                % (self.name, self.vehicle, style, self.risk_aversion, self.trips_logged))


# ---------------------------------------------------------------------------
# Personas used by the demo, so a live audience can see the effect immediately
# ---------------------------------------------------------------------------

PERSONAS = {
    "rider": dict(name="Ananya (delivery rider)", vehicle="motorcycle",
                  risk_aversion=0.35, turn_weight=0.7, avoid_narrow=0.0,
                  driver_skill=1.05,
                  home="Park Circus", work="BBD Bagh", usual_depart="09:15"),
    "exec": dict(name="Mr Basu (must not be late)", vehicle="car",
                 risk_aversion=2.0, turn_weight=1.3, avoid_narrow=0.75,
                 prefer_arterial=0.8, driver_skill=0.95,
                 home="Alipore", work="Esplanade", usual_depart="09:00"),
    "student": dict(name="Rohit (student, cheapest fastest)", vehicle="auto",
                    risk_aversion=0.5, turn_weight=0.9, avoid_narrow=0.1,
                    driver_skill=1.0,
                    home="College Street", work="Park Street", usual_depart="08:40"),
    "cabbie": dict(name="Shyamal-da (taxi, knows every lane)", vehicle="taxi",
                   risk_aversion=0.45, turn_weight=0.6, avoid_narrow=0.0,
                   prefer_arterial=0.0, driver_skill=1.08,
                   home="Sealdah", work="Howrah Bridge approach", usual_depart="18:30"),
}


class ProfileStore:
    """Tiny JSON-backed store. A real deployment would use a database."""

    def __init__(self, path=PROFILES_PATH):
        self.path = path
        self.profiles: dict[str, UserProfile] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._seed_personas()
            return
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
            for uid, d in blob.items():
                d.pop("_comment", None)
                self.profiles[uid] = UserProfile(**d)
        except Exception:
            self._seed_personas()

    def _seed_personas(self) -> None:
        for key, kw in PERSONAS.items():
            self.profiles[key] = UserProfile(user_id=key, **kw)
        self.save()

    def save(self) -> None:
        blob = {uid: asdict(p) for uid, p in self.profiles.items()}
        self.path.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    def get(self, user_id: str, name: str = "Commuter") -> UserProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id, name=name)
            self.save()
        return self.profiles[user_id]

    def set_field(self, user_id: str, field_name: str, value) -> UserProfile:
        p = self.get(user_id)
        if hasattr(p, field_name):
            cur = getattr(p, field_name)
            if isinstance(cur, float):
                value = float(value)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                value = int(value)
            setattr(p, field_name, value)
            self.save()
        return p
