"""Start Triffy: the website, plus (if you want) a live camera collector.

Double-click start.bat (Windows) or start.command (Mac), or run:

    python start.py                   asks how many cameras to watch
    python start.py --cameras 30      no questions
    python start.py --cameras 0       no cameras: replay recorded traffic

Live cameras need the vision packages (requirements-vision.txt). Without a GPU
each camera costs about 34 CPU-seconds per reading, so the number to watch is a
question of how many cores this computer can spare.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECOMMENDED = 30          # cameras along the demo routes: ~7% from the full answer
MAX_CAMERAS = 172         # every camera the live engine can place on a road
CPU_S_PER_CAMERA = 34.0   # measured on a CPU, per camera per reading
INTERVAL_S = 300          # TfL replaces each camera's clip about every 5 minutes
# The best-covered recorded evening in the repository's data.
REPLAY_AT = "2026-09-19 17:30"


def cores_needed(cameras: int) -> float:
    return cameras * CPU_S_PER_CAMERA / INTERVAL_S


def ask_cameras() -> int:
    cores = os.cpu_count() or 2
    print("How many live London traffic cameras should Triffy watch?\n")
    print("   0   none: replay the traffic recorded on %s" % REPLAY_AT)
    print("  %2d   recommended: the cameras along the demo routes (~%.1f CPU cores)"
          % (RECOMMENDED, cores_needed(RECOMMENDED)))
    print("       or any number up to %d\n" % MAX_CAMERAS)
    print("This computer has %d CPU cores; each camera keeps about %.2f of a core"
          " busy.\n" % (cores, cores_needed(1)))
    while True:
        text = input("Cameras [%d]: " % RECOMMENDED).strip()
        if not text:
            return RECOMMENDED
        if text.isdigit() and int(text) <= MAX_CAMERAS:
            n = int(text)
            if cores_needed(n) > cores * 0.75:
                print("  %d cameras need about %.0f cores, and readings would fall"
                      " behind here. Try %d or fewer." % (n, cores_needed(n),
                                                         int(cores * 0.75 / cores_needed(1))))
                continue
            return n
        print("  Please type a number from 0 to %d." % MAX_CAMERAS)


def free_port(start: int = 8000) -> int:
    """The first free port from ``start``: another program (or an older copy of
    Triffy) may already be using 8000."""
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("No free port between %d and %d." % (start, start + 19))


def wait_until_up(url: str, proc, timeout_s: float = 120.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1.0)
    return False


def commands(cameras: int) -> tuple:
    """(collector command or None, website command)."""
    py = sys.executable
    api = [py, "-m", "route_engine.api"]
    if cameras == 0:
        return None, api + ["--replay", REPLAY_AT]
    collector = [py, "-m", "route_engine.collector", "--near-routes",
                 "--cameras", str(cameras), "--interval", str(INTERVAL_S),
                 "--device", os.environ.get("TRIFFY_DEVICE") or "cpu", "--quiet"]
    return collector, api


def main() -> None:
    ap = argparse.ArgumentParser(description="Start Triffy.")
    ap.add_argument("--cameras", type=int, default=None,
                    help="live cameras to watch (0 = replay recorded traffic)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be started, and start nothing")
    args = ap.parse_args()
    os.chdir(ROOT)

    if importlib.util.find_spec("fastapi") is None:
        raise SystemExit("Triffy's packages are not installed. Run this once:\n\n"
                         "    %s -m pip install -r requirements.txt\n"
                         % Path(sys.executable).name)

    cameras = args.cameras if args.cameras is not None else ask_cameras()
    if cameras and importlib.util.find_spec("ultralytics") is None:
        print("\nLive cameras need the vision packages, which are not installed:\n"
              "    %s -m pip install -r requirements-vision.txt\n"
              "Replaying recorded traffic instead.\n" % Path(sys.executable).name)
        cameras = 0

    collector_cmd, api_cmd = commands(cameras)
    port = free_port()
    url = "http://127.0.0.1:%d" % port
    if args.dry_run:
        if collector_cmd:
            print("collector:", " ".join(collector_cmd[1:]))
        print("website:  ", " ".join(api_cmd[1:]), "on port", port)
        return

    env = dict(os.environ, TRIFFY_PORT=str(port))
    procs = []
    try:
        if collector_cmd:
            log = open(ROOT / "data" / "collector.log", "a", encoding="utf-8")
            procs.append(subprocess.Popen(collector_cmd, stdout=log,
                                          stderr=subprocess.STDOUT, env=env))
            print("\nWatching %d cameras (progress: data/collector.log). The first"
                  " readings\narrive within a few minutes; until then the map shows"
                  " typical traffic." % cameras)
        api = subprocess.Popen(api_cmd, env=env)
        procs.append(api)
        print("\nStarting the website...")
        if wait_until_up(url + "/", api):
            print("\nTriffy is running at %s  (close this window or press Ctrl+C"
                  " to stop)\n" % url)
            if not args.no_browser:
                webbrowser.open(url)
        api.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        print("Triffy stopped.")


if __name__ == "__main__":
    main()
