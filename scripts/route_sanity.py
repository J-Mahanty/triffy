"""Sanity-check Triffie's live London ETAs against an external reference.

**What this is, and what it is not.** OSRM's public demo server routes on the
same OpenStreetMap data with fixed textbook speeds and *no traffic model at all*.
So it is not ground truth and not a competitor - it is a free-flow yardstick.
The question it answers is narrow but important:

    Are our distances sane, and is our congestion factor plausible?

Two failure modes this is designed to catch:

* **Distance far off OSRM** means the road graph is wrong - missing streets,
  bad one-ways, a hole in the map. Exactly this caught the first London import,
  where a missing city centre inflated a 2.5 km trip to 7.3 km.
* **Time ratio implausible** means the congestion or junction-delay model is
  miscalibrated. Central London genuinely runs well below free-flow, so a ratio
  near 1.0 would be as suspicious as one near 4.0.

Transport for London publishes average traffic speeds for central London in the
region of 13-19 km/h depending on time of day, which is the band our implied
speeds should land in during busy hours. That is a published figure to compare
against, not something we invented.

Usage:
    python scripts/route_sanity.py
    python scripts/route_sanity.py --pairs 12
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triffie.live_engine import LiveEngine

UA = {"User-Agent": "Triffie-DesignThinkingLab/0.1 (academic prototype)"}
OSRM = "https://router.project-osrm.org/route/v1/driving/%f,%f;%f,%f"

# Central-London pairs a Londoner would actually name, spread across the extract
# so one bad corner of the graph cannot hide behind nine good ones.
PAIRS = [
    ("King's Cross", "Tower Bridge"),
    ("Paddington", "Liverpool Street"),
    ("Victoria", "Shoreditch"),
    ("Camden Town", "Waterloo"),
    ("Marble Arch", "Bank"),
    ("Euston", "Trafalgar Square"),
    ("Angel Islington", "Westminster"),
    ("Knightsbridge", "Holborn"),
    ("Elephant and Castle", "Oxford Circus"),
    ("Whitechapel", "Hyde Park Corner"),
    ("Vauxhall Cross", "Farringdon"),
    ("Borough", "Marylebone Road"),
]


def osrm_route(a_lat, a_lon, b_lat, b_lon, retries: int = 2):
    """Free-flow reference distance and duration, or None if unreachable."""
    url = (OSRM % (a_lon, a_lat, b_lon, b_lat)) + "?overview=false&alternatives=false"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200:
                rt = r.json()["routes"][0]
                return rt["distance"] / 1000.0, rt["duration"] / 60.0
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def _compare_pair(eng, origin: str, dest: str):
    """Plan one pair, print its row, and return (distance ratio, time ratio,
    implied km/h) against OSRM - or None when there is nothing to compare."""
    o, d = eng.resolve(origin), eng.resolve(dest)
    if o is None or d is None:
        print("%-22s %-20s | UNRESOLVED" % (origin[:22], dest[:20]))
        return None
    try:
        plan = eng.plan(origin, dest, user_id="guest", k=1)
    except Exception as exc:
        print("%-22s %-20s | PLAN FAILED %s" % (origin[:22], dest[:20], exc))
        return None
    r = plan["routes"][0]
    t_km = r["distance_m"] / 1000.0
    t_min = r["median_s"] / 60.0
    kph = t_km / (t_min / 60.0) if t_min else 0.0

    ref = osrm_route(o.lat, o.lon, d.lat, d.lon)
    if ref is None:
        print("%-22s %-20s | %6.2f %6s | %6.1f %6s | %5.1f %5s"
              % (origin[:22], dest[:20], t_km, "-", t_min, "-", kph, "-"))
        return None
    o_km, o_min = ref
    dr = t_km / o_km if o_km else 0.0
    tr = t_min / o_min if o_min else 0.0
    print("%-22s %-20s | %6.2f %6.2f | %6.1f %6.1f | %5.1f %5.2f"
          % (origin[:22], dest[:20], t_km, o_km, t_min, o_min, kph, tr))
    return dr, tr, kph


def _verdict(dm: float, tm: float, sm: float) -> bool:
    """Print PASS/FAIL for distance ratio, time ratio and speed; True if all pass."""
    ok = True
    if dm > 1.25:
        print("  FAIL  distances %.0f%% longer than OSRM - suspect the road graph "
              "(missing streets, bad one-ways, holes)." % (100 * (dm - 1)))
        ok = False
    else:
        print("  PASS  distances within %.0f%% of OSRM: the graph routes sensibly."
              % (100 * abs(dm - 1)))

    if not (1.1 <= tm <= 2.6):
        print("  FAIL  time ratio %.2f is outside the plausible congestion band "
              "(1.1-2.6x free-flow)." % tm)
        ok = False
    else:
        print("  PASS  %.2fx free-flow is a plausible central-London penalty." % tm)

    if not (8.0 <= sm <= 30.0):
        print("  FAIL  implied %.1f km/h is outside anything London does." % sm)
        ok = False
    else:
        print("  PASS  implied %.1f km/h sits in the range TfL publishes for "
              "central London (~13-19 km/h in traffic, higher off-peak)." % sm)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare live ETAs to a free-flow reference")
    ap.add_argument("--pairs", type=int, default=len(PAIRS))
    args = ap.parse_args()

    print("Booting live engine (London Zone 1, real cameras)...")
    eng = LiveEngine(city="lon")
    st = eng.stats()
    print("  %d/%d cameras reporting, data %s s old\n"
          % (st["cameras_reporting"], st["cameras_mapped"], st["data_age_s"]))

    print("%-22s %-20s | %6s %6s | %6s %6s | %5s %5s"
          % ("from", "to", "T km", "O km", "T min", "O min", "km/h", "×ff"))
    print("-" * 96)

    dist_ratios, time_ratios, speeds = [], [], []
    for origin, dest in PAIRS[:args.pairs]:
        row = _compare_pair(eng, origin, dest)
        if row is None:
            continue
        dist_ratios.append(row[0])
        time_ratios.append(row[1])
        speeds.append(row[2])
        time.sleep(0.4)   # be polite to a free public service

    if not time_ratios:
        print("\nNo comparisons completed.")
        return

    print()
    print("=" * 96)
    print("SUMMARY  (T = Triffie live, O = OSRM free-flow reference)")
    print("=" * 96)
    print("  distance ratio T/O   median %.2f   range %.2f-%.2f"
          % (statistics.median(dist_ratios), min(dist_ratios), max(dist_ratios)))
    print("  time ratio     T/O   median %.2f   range %.2f-%.2f"
          % (statistics.median(time_ratios), min(time_ratios), max(time_ratios)))
    print("  implied speed        median %.1f km/h   range %.1f-%.1f"
          % (statistics.median(speeds), min(speeds), max(speeds)))
    print()

    # Interpretation, stated as explicit pass/fail so the result cannot be
    # read as whatever the reader hoped for.
    ok = _verdict(statistics.median(dist_ratios), statistics.median(time_ratios),
                  statistics.median(speeds))

    print()
    print("  OSRM is a free-flow yardstick, not ground truth: it has no traffic")
    print("  model. This checks that our geometry is sane and our congestion")
    print("  penalty is believable - it does not prove our ETAs are correct.")
    print("=" * 96)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
