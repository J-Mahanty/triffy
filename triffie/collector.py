"""Continuous real-data collector.

Runs in the background for hours or days, sampling real traffic cameras and
recording what our own computer vision measures. The output is the evidence base
that makes Triffie's claims checkable rather than asserted:

* a genuine time series of genuine traffic, gathered by us,
* against which ``validate.py`` scores real forecasts of real conditions.

It appends to JSONL rather than rewriting a file, so a crash, a laptop sleeping,
or a Ctrl-C costs you one cycle rather than the whole dataset. Restarting simply
continues the same file.

Usage:
    python -m triffie.collector --cameras 24 --interval 300
    python -m triffie.collector --status
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from .config import DATA
from .livecams import LiveCameraReader, pick_cameras

OBS_PATH = DATA / "live_observations.jsonl"
_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print("\nStopping after this cycle...")


def collect_once(reader: LiveCameraReader, cams, verbose: bool = True) -> int:
    """One sweep over every camera. Returns how many succeeded."""
    got = 0
    with OBS_PATH.open("a", encoding="utf-8") as fh:
        for cam in cams:
            got += _collect_camera(reader, cam, fh, verbose)
    return got


def _collect_camera(reader: LiveCameraReader, cam, fh, verbose: bool) -> int:
    """Measure one camera and append the reading. 1 if one was written."""
    label = cam.name[:34]
    try:
        meas = reader.measure(cam)
    except Exception as exc:
        if verbose:
            print("    %-34s ERROR %s" % (label, type(exc).__name__))
        return 0
    if not meas:
        if verbose:
            print("    %-34s no feed" % label)
        return 0
    # One flushed write per line: two collectors can share this file (see
    # --ids / --exclude), and a buffered 8 KB flush can split a line so the
    # other process's write lands in the middle of it.
    fh.write(json.dumps(meas) + "\n")
    fh.flush()
    if verbose:
        age = int(meas["feed_age_s"]) if meas["feed_age_s"] else "?"
        print("    %-34s %4.1f veh  moving %3.0f%%  age %4ss"
              % (label, meas["count_mean"], 100 * meas.get("moving_frac", 0), age))
    return 1


def status() -> None:
    """Summarise what has been collected so far."""
    if not OBS_PATH.exists():
        print("No data yet. Start with:  python -m triffie.collector")
        return
    rows = []
    with OBS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        print("File exists but holds no valid rows yet.")
        return

    ts = [r["t_wall"] for r in rows]
    cams = {}
    for r in rows:
        cams.setdefault(r["camera_id"], []).append(r)
    span_h = (max(ts) - min(ts)) / 3600.0

    print("=" * 62)
    print("COLLECTED REAL CAMERA DATA")
    print("=" * 62)
    print("  observations   %d" % len(rows))
    print("  cameras        %d" % len(cams))
    print("  span           %.1f hours" % span_h)
    print("  first          %s" % time.strftime("%Y-%m-%d %H:%M", time.localtime(min(ts))))
    print("  last           %s" % time.strftime("%Y-%m-%d %H:%M", time.localtime(max(ts))))
    per_cam = sorted((len(v), k, v[0]["name"]) for k, v in cams.items())
    print("  samples/camera min %d, median %d, max %d"
          % (per_cam[0][0], per_cam[len(per_cam) // 2][0], per_cam[-1][0]))
    print()
    print("  busiest cameras by mean vehicle count:")
    ranked = sorted(cams.items(),
                    key=lambda kv: -sum(r["count_mean"] for r in kv[1]) / len(kv[1]))
    for cid, rs in ranked[:6]:
        mean = sum(r["count_mean"] for r in rs) / len(rs)
        mv = sum(r.get("moving_frac", 0) for r in rs) / len(rs)
        print("    %-36s %5.1f veh  %3.0f%% moving  n=%d"
              % (rs[0]["name"][:36], mean, 100 * mv, len(rs)))
    print()
    hours_needed = 6.0
    if span_h < hours_needed:
        print("  Keep collecting: validation wants at least %.0f hours "
              "(have %.1f)." % (hours_needed, span_h))
    else:
        print("  Enough for validation. Run:  python -m triffie.validate")
    print("=" * 62)


def _read_ids(path) -> set:
    with open(path, encoding="utf-8") as fh:
        return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}


def _parse_args():
    ap = argparse.ArgumentParser(description="Collect real traffic camera data")
    ap.add_argument("--cameras", type=int, default=24)
    ap.add_argument("--city", default="lon",
                    help="restrict cameras to this city's bbox (see config.CITIES); "
                         "'' samples all of London")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="seconds between sweeps (TfL refreshes ~every 5 min)")
    ap.add_argument("--cycles", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--mapped", action="store_true",
                    help="sample exactly the cameras the live engine can bind to "
                         "a road edge (avoids spending cycles on unusable feeds)")
    ap.add_argument("--ids", default="",
                    help="file of camera ids (one per line) to sample exactly; "
                         "pins the set so a re-imported map cannot change it")
    ap.add_argument("--exclude", default="",
                    help="file of camera ids to leave out, e.g. those another "
                         "collector is already sampling")
    ap.add_argument("--device", default="",
                    help="'cpu' to stay off the GPU. On the demo laptop (GTX 1650) "
                         "two processes using CUDA at once hang each other, so only "
                         "one collector may use the GPU; any second one needs cpu")
    return ap.parse_args()


def _select_cameras(args, where):
    """The cameras to sample: mapped, spread, or pinned by --ids/--exclude."""
    if args.mapped and where:
        # Sample precisely the cameras that the live engine can attach to a
        # road. Anything it cannot map contributes nothing to routing, so
        # spending a YOLO pass on it every cycle is pure waste.
        from .live_engine import map_cameras_to_edges
        from .network import load_network
        net = load_network(where)
        pool = pick_cameras(400, city=where)
        mapped_ids = {m.id for m in map_cameras_to_edges(net, pool)}
        cams = [c for c in pool if c.id in mapped_ids][:args.cameras]
        print("Selecting up to %d cameras mapped to %s roads..."
              % (args.cameras, where))
    else:
        print("Selecting %d spatially-spread cameras%s..."
              % (args.cameras, (" within %s" % where) if where else ""))
        cams = pick_cameras(args.cameras, city=where)
    if args.ids or args.exclude:
        cams = _pin_cameras(args, where, cams)
    return cams


def _pin_cameras(args, where, cams):
    """A validation series is only meaningful if it follows the same cameras
    throughout, so an explicit list overrides the selection."""
    if args.ids:
        keep = _read_ids(args.ids)
        pool = pick_cameras(400, city=where)
        cams = [c for c in pool if c.id in keep][:args.cameras]
        missing = keep - {c.id for c in cams}
        if missing:
            print("  warning: %d listed ids not in the registry now" % len(missing))
    if args.exclude:
        drop = _read_ids(args.exclude)
        cams = [c for c in cams if c.id not in drop]
    print("  overridden by --ids/--exclude: %d cameras pinned" % len(cams))
    return cams


def _run_cycles(args, reader, cams) -> None:
    """Sweep every interval until stopped, or until --cycles have run."""
    cycle = 0
    total = 0
    while not _STOP:
        cycle += 1
        t0 = time.time()
        print("[cycle %d] %s" % (cycle, time.strftime("%H:%M:%S")), flush=True)
        n = collect_once(reader, cams, verbose=not args.quiet)
        total += n
        dt = time.time() - t0
        print("  %d/%d cameras in %.0fs  (total observations: %d)\n"
              % (n, len(cams), dt, total))

        if args.cycles and cycle >= args.cycles:
            break
        # Sleep the remainder of the interval, waking often so Ctrl-C is snappy.
        wait = max(5.0, args.interval - dt)
        slept = 0.0
        while slept < wait and not _STOP:
            time.sleep(min(2.0, wait - slept))
            slept += 2.0

    print("Collected %d observations over %d cycles." % (total, cycle))
    print("Check progress any time:  python -m triffie.collector --status")


def main() -> None:
    args = _parse_args()

    if args.status:
        status()
        return

    # This runs for hours in a terminal someone is watching, so progress must
    # appear as it happens rather than in a block when the buffer flushes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    signal.signal(signal.SIGINT, _handle_stop)

    cams = _select_cameras(args, args.city or None)
    if not cams:
        print("Could not reach the camera registry. Check the network.")
        sys.exit(1)
    print("  %d cameras selected" % len(cams))
    print("Loading YOLO11...")
    reader = LiveCameraReader(device=args.device or None)
    print("Writing to %s" % OBS_PATH)
    print("Sampling every %.0f s. Ctrl-C to stop.\n" % args.interval)
    _run_cycles(args, reader, cams)


if __name__ == "__main__":
    main()
