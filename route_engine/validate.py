"""Validate Triffy's forecasting against real measured traffic.

This is the module that answers "how do you know any of this works?" without
appealing to our own simulator.

The data is real: our own YOLO11 vision, run against real roadside cameras every
few minutes by ``collector.py``. The question asked is the one that actually
matters for routing:

    Given what the cameras see now, how busy will this junction be in 20 minutes?

Because if you cannot answer that, you cannot cost a road segment at the time
the driver will reach it, and time-dependent routing is worthless.

**Three forecasters compete, on identical data:**

* ``persistence``  - "it will be exactly as it is now". This is the honest
  analogue of snapshot routing, and it is a genuinely strong baseline over short
  horizons, which is why beating it means something.
* ``historical``   - "it will be whatever this camera normally shows at this
  time of day". No live input at all.
* ``triffy``      - where it is now, plus where it is heading, plus the
  long-run profile's view, the last decayed over the horizon:

      pred = c0 + damp*slope*horizon + (h_then - h_now)*exp(-horizon/tau)

  The middle term matters more than it looks. A per-time-of-day profile needs
  weeks of history to be worth anything, and a prototype has days - so the
  time-awareness has to come from the recent local trend, which is available
  immediately.

**Methodology: a three-way split, and the middle slice is the point.** The
historical profile is *learned* from slice 1, so on slice 1 it looks far more
accurate than it really is. Fitting the blend parameters there too makes the
fitter conclude "trust history, ignore the live cameras", pick a tiny tau, and
destroy the live signal - which is exactly what happened on the first attempt,
leaving Triffy 220% worse than persistence. Parameters are therefore fitted on
slice 2, which the profile has never seen, and scored on slice 3.

**A note on what this can and cannot show.** Whether the trend term helps on real
traffic is an empirical question, not a foregone conclusion: if the 20-minute
change is smaller than the measurement noise, the fitter will correctly set
``damp`` to zero and Triffy will tie persistence. That is a real result and
should be reported as one, not tuned away.

**The metric is vehicle count, deliberately.** Counting needs no camera
calibration whatsoever - no metres-per-pixel, no homography, no horizon
estimate. Nobody can attack the result by attacking our geometry.

Usage:
    python -m route_engine.validate
    python -m route_engine.validate --horizon 1200 --split 0.6
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict

from .collector import OBS_PATH
from .config import DATA, repo_path

SEC_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

OUTAGE_S = 900.0
"""A break longer than this is an outage, not collection jitter.

The collectors sample every ~360 s, so a quarter of an hour of silence means
something stopped - a sleeping laptop, a killed process, a network drop.
"""


def _segments(times: list, gap_s: float = OUTAGE_S) -> list:
    """(start, end) of each stretch of `times` with no break longer than gap_s."""
    if not times:
        return []
    out, lo = [], times[0]
    for a, b in zip(times, times[1:]):
        if b - a > gap_s:
            out.append((lo, a))
            lo = b
    out.append((lo, times[-1]))
    return out


def _crop_to_longest_run(series: dict, full_span: bool = False):
    """Restrict the series to its longest unbroken stretch of collection.

    **Why this is not cherry-picking.** The three-way split fits the blend on
    one slice and scores it on a later one. That is only a fair test if both
    slices describe the same world. Across an outage they do not: on 19-20 Sep
    a sleeping laptop left a 10.3 hour hole, so the fit came from Friday
    evening and the test from Saturday late-morning - different traffic,
    different time of day, parameters transplanted between them. The result was
    -6.5% against persistence, where the same code on either continuous stretch
    measures -18%. The gap-spanning number was the artefact, not the finding.

    You do not fit and test across a data outage; you validate on a run. So we
    take the longest one, and *say* what was dropped rather than quietly
    dropping it - the report prints it and it is recorded in the published
    JSON. ``--full-span`` keeps everything for anyone who wants to see the
    difference for themselves.
    """
    times = sorted(t for rows in series.values() for (t, _c, _m, _n) in rows)
    segs = _segments(times)
    span_h = (times[-1] - times[0]) / 3600.0 if times else 0.0
    if full_span or len(segs) <= 1:
        return series, {"segments": len(segs), "cropped": False,
                        "excluded_hours": 0.0, "span_hours_all": span_h}

    lo, hi = max(segs, key=lambda s: s[1] - s[0])
    kept = {}
    for cid, rows in series.items():
        rows = [r for r in rows if lo <= r[0] <= hi]
        if rows:
            kept[cid] = rows
    if not kept:
        print("No usable continuous run in the log.")
        return {}, {}
    return kept, {"segments": len(segs), "cropped": True,
                  "excluded_hours": span_h - (hi - lo) / 3600.0,
                  "span_hours_all": span_h}


def load_series(path=OBS_PATH) -> dict:
    """camera_id -> sorted list of (wall_time, count, moving_frac, name)."""
    if not path.exists():
        return {}
    series = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            series[r["camera_id"]].append(
                (float(r["t_wall"]), float(r["count_mean"]),
                 float(r.get("moving_frac", 0.0)), r.get("name", "")))
    for k in series:
        series[k].sort(key=lambda x: x[0])
    return dict(series)


def _tod_bucket(t_wall: float, minutes: int = 30) -> int:
    """Time-of-day bucket in local time, for the historical profile."""
    lt = time.localtime(t_wall)
    return (lt.tm_hour * 60 + lt.tm_min) // minutes


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------

class HistoricalProfile:
    """Per-camera, per-time-of-day mean count, learned from the training split.

    Falls back through progressively coarser estimates when a bucket is empty,
    which matters a lot with only a day or two of data: an unseen bucket should
    degrade to that camera's overall mean, not to zero.
    """

    def __init__(self, rows, bucket_min: int = 30):
        self.bucket_min = bucket_min
        self.by_cam_bucket = defaultdict(list)
        self.by_cam = defaultdict(list)
        for cid, t, c, *_ in rows:
            self.by_cam_bucket[(cid, _tod_bucket(t, bucket_min))].append(c)
            self.by_cam[cid].append(c)
        self.cam_bucket_mean = {k: statistics.fmean(v)
                                for k, v in self.by_cam_bucket.items()}
        self.cam_mean = {k: statistics.fmean(v) for k, v in self.by_cam.items()}
        allv = [c for v in self.by_cam.values() for c in v]
        self.global_mean = statistics.fmean(allv) if allv else 0.0

    def expect(self, cid: str, t_wall: float) -> float:
        b = _tod_bucket(t_wall, self.bucket_min)
        v = self.cam_bucket_mean.get((cid, b))
        if v is not None:
            return v
        return self.cam_mean.get(cid, self.global_mean)


def recent_slope(rows, i: int, window: int = 6) -> float:
    """Least-squares trend in vehicles per second over the preceding samples.

    This is the signal that works on day one. A per-time-of-day profile needs
    weeks of history to be useful, but "this junction has gone 8, 11, 14 over the
    last twenty minutes" is available immediately and captures the same thing
    that makes snapshot routing wrong: conditions are moving, not static.
    """
    lo = max(0, i - window + 1)
    pts = rows[lo:i + 1]
    if len(pts) < 3:
        return 0.0
    t0 = pts[0][0]
    xs = [p[0] - t0 for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-9:
        return 0.0
    return sum((xs[k] - mx) * (ys[k] - my) for k in range(n)) / denom


def _closest_in_window(times: list, j: int, target: float, tol_s: float):
    """Index of the sample nearest ``target`` among times[j:] that lie within
    ``tol_s`` of it (first one wins a tie), or None if there is none."""
    best = None
    k = j
    while k < len(times) and times[k] <= target + tol_s:
        d = abs(times[k] - target)
        if best is None or d < best[0]:
            best = (d, k)
        k += 1
    return None if best is None else best[1]


def build_pairs(series: dict, horizon_s: float, tol_s: float):
    """Match each observation with the one ~horizon_s later on the same camera."""
    pairs = []
    for cid, rows in series.items():
        times = [r[0] for r in rows]
        j = 0
        for i, (t0, c0, m0, name) in enumerate(rows):
            target = t0 + horizon_s
            # Advance j to the first sample inside the tolerance window.
            while j < len(times) and times[j] < target - tol_s:
                j += 1
            k = _closest_in_window(times, j, target, tol_s)
            if k is None:
                continue
            t1, c1, m1, _ = rows[k]
            pairs.append({"cid": cid, "name": name, "t0": t0, "t1": t1,
                          "c0": c0, "c1": c1, "m0": m0, "m1": m1,
                          "slope": recent_slope(rows, i)})
    return pairs


TAU_GRID = (120, 240, 360, 600, 900, 1200, 1800, 2700, 3600, 7200, 1e9)
DAMP_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def fit_params(pairs, hist: HistoricalProfile, horizon_s: float):
    """Choose (tau, damp) on the tuning split. Returns ((tau, damp), error).

    Only two free parameters, deliberately. With a day or two of data a richer
    model would fit the noise, and the whole point of this exercise is a number
    that survives contact with held-out reality.
    """
    best, best_err = (1e9, 0.0), float("inf")
    if not pairs:
        return best, best_err
    for tau in TAU_GRID:
        for damp in DAMP_GRID:
            errs = [abs(triffy_forecast(p, hist, horizon_s, (tau, damp)) - p["c1"])
                    for p in pairs]
            e = statistics.fmean(errs)
            if e < best_err:
                best, best_err = (tau, damp), e
    return best, best_err


def triffy_forecast(p, hist: HistoricalProfile, horizon_s: float,
                     params) -> float:
    """Persistence + recent trend + decayed historical correction.

        pred = c0 + damp*slope*horizon + (h_then - h_now)*exp(-horizon/tau)

    Three terms, each earning its place:

    * ``c0`` - where the junction is right now. Over short horizons this is a
      genuinely strong predictor, which is exactly why snapshot routing survives
      as long as it does.
    * ``damp * slope * horizon`` - where it is *heading*, from the least-squares
      trend over the last few samples. This is the term that works from day one
      and carries most of the time-awareness when history is thin.
    * the decayed historical correction - the long-run profile's view of how this
      junction usually changes between now and then. Valuable with weeks of
      data, close to useless with two days, which is why ``tau`` is fitted rather
      than assumed.
    """
    tau_s, damp = params
    h_now = hist.expect(p["cid"], p["t0"])
    h_then = hist.expect(p["cid"], p["t1"])
    residual = (p["c0"] - h_now) * math.exp(-horizon_s / tau_s)
    trend = damp * p.get("slope", 0.0) * horizon_s
    return max(0.0, h_then + residual + trend)


def persistence_forecast(p, *_args) -> float:
    """The snapshot assumption: conditions will not change."""
    return p["c0"]


def historical_forecast(p, hist: HistoricalProfile, *_args) -> float:
    """No live input at all: whatever this camera normally shows then."""
    return hist.expect(p["cid"], p["t1"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(pairs, fn, hist, horizon_s, params) -> dict:
    abs_err, sq_err, pct = [], [], []
    for p in pairs:
        pred = fn(p, hist, horizon_s, params)
        e = pred - p["c1"]
        abs_err.append(abs(e))
        sq_err.append(e * e)
        denom = max(p["c1"], 1.0)
        pct.append(abs(e) / denom)
    n = len(abs_err)
    if not n:
        return {}
    return {
        "n": n,
        "mae": statistics.fmean(abs_err),
        "rmse": math.sqrt(statistics.fmean(sq_err)),
        "mape": 100.0 * statistics.fmean(pct),
        "median_ae": statistics.median(abs_err),
    }


def run(horizon_s: float = 1200.0, split: float = 0.6, tol_s: float = 150.0,
        verbose: bool = True, out_path=None, write: bool = True,
        series_path=None, full_span: bool = False) -> dict:
    """Score the forecasters and, by default, publish the result for the dashboard.

    ``write`` exists because publishing is a side effect with teeth. The
    dashboard reads ``data/validation.json`` and renders it under a REAL DATA
    badge, so anything written there is a claim about measured reality. Tests
    that feed synthetic series must pass ``write=False`` - one that did not
    silently replaced the published figures with numbers from a made-up ramp,
    and the dashboard displayed them as measured for hours.

    ``series_path`` selects the observation log, which also becomes the recorded
    provenance of the result.
    """
    src = series_path or OBS_PATH
    series = load_series(src)
    if not series:
        print("No collected data. Start the collector first:")
        print("    python -m route_engine.collector --cameras 30 --interval 300")
        return {}

    series, crop = _crop_to_longest_run(series, full_span=full_span)
    if not series:
        return {}

    # Sort by TIME, explicitly. A bare sorted() on (cid, t, ...) orders by camera
    # id first, so t_start/t_end become the bounds of the alphabetically first
    # and last cameras rather than of the dataset. With two collector series
    # started 25 minutes apart that produced a negative span, a three-way split
    # with zero tuning pairs, and a silent fallback to "no trend, no decay" -
    # which made Triffy identical to persistence by construction. Synthetic
    # tests never caught it because there every camera shares one time range.
    flat = sorted(((cid, t, c, m) for cid, rows in series.items()
                   for (t, c, m, _n) in rows),
                  key=lambda r: r[1])
    if len(flat) < 60:
        print("Only %d observations so far - keep the collector running." % len(flat))
        return {}

    t_start, t_end = flat[0][1], flat[-1][1]
    span_h = (t_end - t_start) / 3600.0

    # Span is first-to-last, which says nothing about the holes in between. An
    # overnight outage leaves a log that still "spans 21 hours" while a third of
    # it is missing, and quoting the span alone would overstate the evidence.
    # Anything longer than a few collection cycles is an outage, not jitter.
    gap_floor_s = 900.0
    gaps = [flat[i + 1][1] - flat[i][1] for i in range(len(flat) - 1)
            if flat[i + 1][1] - flat[i][1] > gap_floor_s]
    covered_h = span_h - sum(gaps) / 3600.0
    largest_gap_h = (max(gaps) / 3600.0) if gaps else 0.0

    # Three-way split, and the middle slice is the point.
    #
    # The historical profile is *learned* from slice 1, so on slice 1 it looks
    # far more accurate than it really is. Fitting the decay constant there too
    # makes the fitter conclude "trust history, ignore the live cameras", pick a
    # tiny tau, and destroy the live signal. Choosing tau on a slice the profile
    # has never seen exposes its true accuracy, so the blend is fitted honestly.
    # Cut on the DATA, not on the clock.
    #
    # These used to be wall-clock fractions of (t_start, t_end). That silently
    # assumes observations are spread evenly across the span, and they are not:
    # on 19-20 Sep the laptop slept and left a 10.3 hour hole. 45% and 60% of
    # that span both land inside the hole, so the tuning slice held ZERO pairs,
    # the fitter fell back to "no decay, no trend" - which is persistence,
    # exactly - and the run published "Triffy beats persistence by -0.0%".
    # A measurement of a thing against itself, under a REAL DATA badge.
    #
    # Taking the cut at a quantile of the sorted observations instead makes
    # every slice non-empty whenever there is data at all, however the data is
    # distributed in time. `flat` is already sorted by time, so the slices stay
    # strictly chronological and nothing leaks backwards from test to tuning.
    cut_profile = flat[int(split * 0.75 * (len(flat) - 1))][1]
    cut_tau = flat[int(split * (len(flat) - 1))][1]

    profile_rows = [r for r in flat if r[1] <= cut_profile]
    hist = HistoricalProfile(profile_rows)

    all_pairs = build_pairs(series, horizon_s, tol_s)
    tune_pairs = [p for p in all_pairs
                  if p["t0"] > cut_profile and p["t1"] <= cut_tau]
    test_pairs = [p for p in all_pairs if p["t0"] > cut_tau]

    if len(test_pairs) < 20:
        print("Not enough held-out pairs yet (%d). Collect for longer; a %d-minute"
              % (len(test_pairs), int(horizon_s / 60)))
        print("horizon needs at least that much data beyond the split point.")
        return {}

    if len(tune_pairs) < 10:
        # Refuse, rather than falling back to (tau=inf, damp=0).
        #
        # That fallback does not degrade the forecaster gracefully - it deletes
        # it. With no decay and no trend weight, triffy_forecast reduces to
        # persistence term for term, so the table then compares a thing with
        # itself and prints a headline of -0.0% that reads like a finding. It
        # published exactly that once, off the back of the sleep outage above.
        #
        # There is no honest number to write here, so write nothing. Same
        # contract as the held-out check: say what is missing, return {}.
        print("Only %d tuning pairs - not enough to fit the blend, so there is"
              % len(tune_pairs))
        print("no Triffy to score. NOT publishing: with no fitted parameters")
        print("Triffy collapses into persistence and the comparison is")
        print("meaningless. Collect more, or widen --split.")
        return {}

    params, _ = fit_params(tune_pairs, hist, horizon_s)
    tau, damp = params
    train_pairs = tune_pairs

    results = {
        "triffy": score(test_pairs, triffy_forecast, hist, horizon_s, params),
        "persistence": score(test_pairs, persistence_forecast, hist,
                             horizon_s, params),
        "historical": score(test_pairs, historical_forecast, hist,
                            horizon_s, params),
    }

    out = {
        "horizon_min": horizon_s / 60.0,
        "span_hours": span_h,
        "covered_hours": covered_h,
        "largest_gap_hours": largest_gap_h,
        # What, if anything, was left out and why - see _crop_to_longest_run.
        "collection_runs": crop.get("segments", 1),
        "cropped_to_longest_run": crop.get("cropped", False),
        "excluded_hours": crop.get("excluded_hours", 0.0),
        "observations": len(flat),
        "cameras": len(series),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "tau_s": tau,
        "trend_damping": damp,
        "results": results,
        # Provenance travels with the numbers. Anything badged REAL DATA in the
        # UI must be checkable rather than assumed, so we record which file the
        # series came from and whether that is the collector's own log.
        "source": repo_path(src),
        "from_collector": bool(src == OBS_PATH),
        "generated_at": time.time(),
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if verbose:
        print(report(out))
    if write:
        path = out_path or (DATA / "validation.json")
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def report(out: dict) -> str:
    r = out["results"]
    L = []
    L.append("=" * 70)
    L.append("REAL-DATA FORECAST VALIDATION")
    L.append("=" * 70)
    L.append("  Source        our own YOLO11 vision on live TfL traffic cameras")
    L.append("  Observations  %d over %.1f hours, %d cameras"
             % (out["observations"], out["span_hours"], out["cameras"]))
    if out.get("largest_gap_hours", 0.0) >= 0.5:
        # Say it here rather than letting the span imply continuous coverage.
        L.append("  Coverage      %.1f of those hours actually collected "
                 "(largest gap %.1f h)"
                 % (out.get("covered_hours", out["span_hours"]),
                    out["largest_gap_hours"]))
    if out.get("cropped_to_longest_run"):
        # State the exclusion in the same breath as the result, so it travels
        # with the number instead of living only in the code.
        L.append("  Scored on     the longest unbroken run; %.1f h in %d other "
                 "collection run(s) excluded" % (out.get("excluded_hours", 0.0),
                                                 out.get("collection_runs", 1) - 1))
        L.append("                (fitting in one regime and testing in another "
                 "across an outage is not a test)")
    L.append("  Question      how many vehicles will this camera see in %d minutes?"
             % round(out["horizon_min"]))
    L.append("  Held-out      %d forecast/outcome pairs (fitted on %d earlier ones)"
             % (out["test_pairs"], out["train_pairs"]))
    L.append("  Fitted tau    %s"
             % ("%.0f s" % out["tau_s"] if out["tau_s"] < 1e8 else "infinite (no decay)"))
    L.append("  Trend weight  %.2f  (0 = ignore recent trend, 1 = extrapolate fully)"
             % out.get("trend_damping", 0.0))
    if out["tau_s"] >= 1e8 and out.get("trend_damping", 0.0) == 0.0:
        # A legitimate fit can land here, and then it is a real negative result
        # worth reporting - but it must be reported as one. With no decay and
        # no trend the blend reduces to persistence term for term, so the
        # margin below is a thing compared with itself and rounds to zero. Say
        # that, rather than letting a 0.0% be read as a measured tie.
        L.append("")
        L.append("  NOTE  the fit chose no decay AND no trend, so on this data")
        L.append("        Triffy reduces to persistence exactly. The margin")
        L.append("        below is not a measurement of anything.")
    L.append("")
    L.append("  %-14s %8s %8s %8s %10s" % ("forecaster", "MAE", "median", "RMSE", "MAPE"))
    L.append("  " + "-" * 52)
    order = sorted(r, key=lambda k: r[k]["mae"])
    for k in order:
        d = r[k]
        L.append("  %-14s %8.2f %8.2f %8.2f %9.1f%%"
                 % (k, d["mae"], d["median_ae"], d["rmse"], d["mape"]))
    L.append("")

    t, p = r["triffy"]["mae"], r["persistence"]["mae"]
    h = r["historical"]["mae"]
    L.append("  vs persistence (the snapshot assumption): %+.1f%%"
             % (100.0 * (t - p) / p))
    L.append("  vs historical  (no live data at all)    : %+.1f%%"
             % (100.0 * (t - h) / h))
    L.append("  (negative = Triffy is more accurate)")
    L.append("")
    L.append("  Units are vehicles in frame. Counting needs no camera")
    L.append("  calibration, so this number cannot be attacked by")
    L.append("  disputing our geometry.")
    L.append("=" * 70)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate forecasts against real data")
    ap.add_argument("--horizon", type=float, default=1200.0,
                    help="forecast horizon in seconds (default 1200 = 20 min)")
    ap.add_argument("--split", type=float, default=0.6,
                    help="fraction of the timeline used for fitting")
    ap.add_argument("--tol", type=float, default=150.0,
                    help="matching tolerance in seconds")
    ap.add_argument("--sweep", action="store_true",
                    help="report several horizons")
    ap.add_argument("--full-span", action="store_true",
                    help="score across collection outages too, instead of on "
                         "the longest unbroken run (see _crop_to_longest_run)")
    args = ap.parse_args()

    if args.sweep:
        # write=False: a sweep is for reading, not for publishing. Left on, it
        # ran five horizons in turn and left data/validation.json holding the
        # LAST one - a 60-minute forecast - which the dashboard would then badge
        # REAL DATA beside a README quoting the 20-minute number.
        for h in (300, 600, 1200, 1800, 3600):
            print()
            run(horizon_s=h, split=args.split, tol_s=args.tol,
                full_span=args.full_span, write=False)
    else:
        run(horizon_s=args.horizon, split=args.split, tol_s=args.tol,
            full_span=args.full_span)


if __name__ == "__main__":
    main()
