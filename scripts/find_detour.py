"""Find the best live example of Triffy routing around real congestion.

The single most convincing thing this project can show is a route that is
*longer in distance and shorter in time* because real cameras say the direct
road is blocked. The simulated Kolkata mode could never produce one reliably
(§5 of DEMO.md explains why: in a dense grid the alternatives are usually
worse). On live London data they happen naturally.

The catch is that they are live. Whatever pair is congested this afternoon may
be clear by the time you present. So rather than hard-coding an example that
might have evaporated, this scans candidate pairs and ranks them by how much
time the congestion-aware route actually saves against the shortest path, scored
under the same live conditions.

Run it a few minutes before the demo and use whatever it puts at the top.

Usage:
    python scripts/find_detour.py
    python scripts/find_detour.py --min-saving 3
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_engine.live_engine import LiveEngine
from route_engine.router import Router

PLACES = [
    "King's Cross", "Tower Bridge", "Paddington", "Liverpool Street", "Victoria",
    "Shoreditch", "Camden Town", "Waterloo", "Marble Arch", "Bank", "Euston",
    "Trafalgar Square", "Angel Islington", "Westminster", "Knightsbridge",
    "Holborn", "Elephant and Castle", "Oxford Circus", "Whitechapel",
    "Hyde Park Corner", "Vauxhall Cross", "Farringdon", "Borough", "Aldgate",
]


def shortest_path_router(eng, now):
    """A router that ignores traffic entirely: free-flow speeds, no junction delay.

    This is the 'what the map says is shortest' path, and the comparison target.
    """
    prof = eng.forecaster.build_profile(eng.state, now)
    prof.kph = np.repeat(eng.net.ekph[None, :], prof.kph.shape[0], axis=0)
    prof.delay_s = prof.delay_s * 0.0
    return Router(eng.net, prof, prefs=None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Find live congestion detours")
    ap.add_argument("--min-saving", type=float, default=2.0,
                    help="minutes saved before a pair is worth reporting")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=90)
    args = ap.parse_args()

    print("Booting live engine (London Zone 1, real cameras)...")
    eng = LiveEngine(city="lon")
    st = eng.stats()
    print("  %d/%d cameras reporting, data %s s old"
          % (st["cameras_reporting"], st["cameras_mapped"], st["data_age_s"]))
    if st["cameras_reporting"] < 10:
        print("\n  WARNING: very few cameras reporting. Is the collector running?")
    print("  Scanning pairs...\n")

    now = time.time()
    user = eng.profiles.get("guest")
    live_prof = eng.forecaster.build_profile(eng.state, now)
    user.adapt_profile(eng.net, live_prof)
    live = Router(eng.net, live_prof, prefs=user)
    plain = shortest_path_router(eng, now)

    nodes = {}
    for name in PLACES:
        p = eng.resolve(name)
        if p is not None:
            nodes[name] = p.node

    found = []
    pairs = list(itertools.permutations(nodes, 2))[:args.max_pairs]
    for origin, dest in pairs:
        f = _detour(eng, live, plain, now, nodes[origin], nodes[dest], args.min_saving)
        if f:
            f.update(origin=origin, dest=dest)
            found.append(f)

    found.sort(key=lambda f: -f["saved_min"])
    if not found:
        print("No clear detours right now. That is a legitimate answer: it means")
        print("the network is flowing and the direct route is genuinely best.")
        print("Say that rather than hunting for one - it is the honest result.")
        return
    _print_detours(found, args.top)


def _detour(eng, live, plain, now, a: int, b: int, min_saving: float):
    """A live detour between nodes a and b worth reporting, or None."""
    if eng.net.straight_line(a, b) < 1500:
        return None                       # too short to have a real alternative
    try:
        chosen = live.route(a, b, now, k=1)
        short = plain.route(a, b, now, k=1)
    except Exception:
        return None
    if not chosen or not short:
        return None
    chosen = chosen[0]
    # Score the shortest path under the SAME live conditions, so the two
    # numbers are comparable. Anything else is comparing different worlds.
    short_live = live.evaluate_path(short[0].edges, now)

    saved_min = (short_live.median_s - chosen.median_s) / 60.0
    extra_km = (chosen.distance_m - short_live.distance_m) / 1000.0
    if saved_min < min_saving or extra_km <= 0.2:
        return None
    return {
        "chosen_km": chosen.distance_m / 1000.0,
        "chosen_min": chosen.median_s / 60.0,
        "short_km": short_live.distance_m / 1000.0,
        "short_min": short_live.median_s / 60.0,
        "saved_min": saved_min, "extra_km": extra_km,
        "via": chosen.as_dict(eng.net)["roads"][:2],
        "short_via": short_live.as_dict(eng.net)["roads"][:2],
    }


def _print_detours(found: list, top: int) -> None:
    print("=" * 94)
    print("LIVE CONGESTION DETOURS  (longer in distance, shorter in time)")
    print("=" * 94)
    for f in found[:top]:
        print("\n  %s -> %s" % (f["origin"], f["dest"]))
        print("    Triffy  %5.2f km  %5.1f min   via %s"
              % (f["chosen_km"], f["chosen_min"], ", ".join(f["via"])))
        print("    shortest %5.2f km  %5.1f min   via %s"
              % (f["short_km"], f["short_min"], ", ".join(f["short_via"])))
        print("    -> %.1f km further, arrives %.1f min sooner"
              % (f["extra_km"], f["saved_min"]))

    best = found[0]
    print()
    print("=" * 94)
    print("USE THIS ONE:  %s -> %s" % (best["origin"], best["dest"]))
    print('  "Triffy sends me %.1f km further round, and I get there %.0f minutes'
          % (best["extra_km"], best["saved_min"]))
    print('   sooner, because real cameras say %s is blocked right now."'
          % best["short_via"][0])
    print("=" * 94)
    print("\n  Live conditions change. Re-run this shortly before presenting.")


if __name__ == "__main__":
    main()
