"""Central configuration for Triffie.

Everything tunable lives here so the demo can be re-pointed at a new city or
re-tuned live in front of an audience without hunting through modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web"
DATA.mkdir(exist_ok=True)


@dataclass(frozen=True)
class CityBox:
    """A named bounding box to import from OpenStreetMap."""
    key: str
    label: str
    south: float
    west: float
    north: float
    east: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)


# Small, dense, high-congestion areas make the best demos: short trips, many
# alternative paths, visible congestion waves.
CITIES: dict[str, CityBox] = {
    "blr": CityBox("blr", "Bengaluru — Indiranagar / Domlur / MG Road",
                   12.9600, 77.5950, 12.9900, 77.6450),
    "kol": CityBox("kol", "Kolkata — Esplanade / Park Street / BBD Bagh",
                   22.5300, 88.3350, 22.5800, 88.3800),
    "del": CityBox("del", "New Delhi — Connaught Place",
                   28.6150, 77.2000, 28.6450, 77.2400),
    # London exists for one reason: it is the only city in this project where we
    # can obtain real, live, public camera feeds. Running the identical pipeline
    # end-to-end on real data there is what makes the Kolkata claim credible -
    # the argument becomes "here it is working on real cameras; the blocker for
    # Kolkata is feed access, not method".
    "lon": CityBox("lon", "London — Zone 1 (LIVE real cameras)",
                   51.4850, -0.1650, 51.5400, -0.0600),
}

ACTIVE_CITY = os.environ.get("TRIFFIE_CITY", "kol")

# Where the dashboard is reachable, for the "see it on the map" link the chat
# puts under a route. The default is the demo laptop's own server.
#
# Set it to the tunnel or host address when the bot is answering people who are
# not sitting at this machine - a Telegram user on a phone cannot open
# 127.0.0.1. Set it to "" to turn the link off entirely, which is the right
# thing when nothing is serving the dashboard.
MAP_BASE = os.environ.get("TRIFFIE_MAP_BASE", "http://127.0.0.1:8000").rstrip("/")


def city() -> CityBox:
    return CITIES[ACTIVE_CITY]


def graph_path(key: str | None = None) -> Path:
    return DATA / f"city_{key or ACTIVE_CITY}.json"


# ---------------------------------------------------------------------------
# Traffic model constants
# ---------------------------------------------------------------------------

# Free-flow speed (km/h) and capacity (vehicles/hour/lane) by OSM highway class.
# Indian urban values: deliberately lower than Western defaults.
ROAD_CLASS = {
    "motorway":       {"kph": 80, "cap": 1800, "rank": 0},
    "motorway_link":  {"kph": 45, "cap": 1200, "rank": 1},
    "trunk":          {"kph": 60, "cap": 1600, "rank": 1},
    "trunk_link":     {"kph": 40, "cap": 1100, "rank": 2},
    "primary":        {"kph": 50, "cap": 1400, "rank": 2},
    "primary_link":   {"kph": 35, "cap": 1000, "rank": 3},
    "secondary":      {"kph": 40, "cap": 1100, "rank": 3},
    "secondary_link": {"kph": 30, "cap": 900,  "rank": 4},
    "tertiary":       {"kph": 35, "cap": 900,  "rank": 4},
    "tertiary_link":  {"kph": 25, "cap": 700,  "rank": 5},
    "residential":    {"kph": 25, "cap": 600,  "rank": 6},
    "unclassified":   {"kph": 25, "cap": 600,  "rank": 6},
    "living_street":  {"kph": 15, "cap": 400,  "rank": 7},
    "service":        {"kph": 15, "cap": 400,  "rank": 8},
}
DEFAULT_CLASS = "residential"

# BPR-style congestion curve: v = v_free / (1 + ALPHA * (flow/capacity) ** BETA)
BPR_ALPHA = 0.85
BPR_BETA = 3.2
MIN_SPEED_KPH = 3.5           # gridlock floor — you still creep forward

# Signal delay: seconds lost at an intersection, scaled by approach congestion.
BASE_SIGNAL_DELAY_S = 12.0
MAX_SIGNAL_DELAY_S = 95.0

# ---------------------------------------------------------------------------
# Nowcast / forecast
# ---------------------------------------------------------------------------
CAM_COVERAGE = float(os.environ.get("TRIFFIE_CAM_COVERAGE", "0.18"))  # frac of edges with a cam
CAM_SPEED_NOISE = 0.09        # relative sigma of a camera-derived speed estimate
PROPAGATION_HOPS = 3          # how far a cam observation is allowed to inform neighbours
# Sharp decay matters more than it looks. With a slow decay an edge adjacent to
# a jammed camera also collects weight from a dozen *normal* cameras two and
# three hops away, and since distant cameras are far more numerous they outvote
# the near one - so the nowcaster under-reacts to unanimous local evidence.
PROPAGATION_DECAY = 0.38      # legacy hop decay, kept for reference
PRIOR_WEIGHT = 0.30           # how strongly 'conditions are normal' resists evidence

# Influence now decays with *effective distance along a corridor* rather than
# with graph hops. OSM splits roads every few hundred metres, so a 3-hop reach
# was only ~500 m and left ~82% of a typical route with no camera support at
# all. Congestion is correlated along a road, so staying on the same named road
# is cheap and turning off it is expensive (see Nowcaster._build_kernel).
CORRIDOR_LEN_M = 1400.0       # e-folding distance of a camera's influence
MAX_REACH_M = 4200.0          # hard cut-off, keeps the kernel bounded
# How fast a reading stops applying across road classes. With a long corridor
# reach this matters more than it used to: without it a trunk-road camera
# bleeds onto residential lanes whose congestion is not correlated with it.
RANK_PENALTY = 0.82
LIVE_RESIDUAL_TAU_S = 1200.0  # live anomaly half-life into the future (20 min)
FORECAST_SIGMA_FLOOR = 0.06
FORECAST_SIGMA_GROWTH = 0.055 # extra relative sigma per 10 min of horizon

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
TURN_PENALTY_S = 4.0          # generic turn
SHARP_TURN_PENALTY_S = 9.0    # >70 degrees — across oncoming traffic in LHT India
ALTERNATIVES = 3
DIVERSITY_PENALTY = 1.7       # edge cost multiplier when re-searching for alternatives
MAX_SETTLED_NODES = 240_000

SIM_TICK_S = 5.0              # simulator step
DEFAULT_SEED = 20260919
