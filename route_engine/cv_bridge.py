"""Bridge real computer vision into the live engine.

The point of this file is to prove that nothing downstream is special-cased. A
``CamObservation`` produced by YOLO11 on actual video is written to disk here,
and the engine splices it in place of one simulated camera's reading. The
nowcaster, forecaster and router then treat it exactly like any other feed,
because it *is* exactly like any other feed.

Usage during a demo, in a second terminal:

    python -m route_engine.cv_bridge --video data/video/vehicles.mp4 --cam CAM007

The dashboard will then show CAM007 sourced from "YOLO11 on live video".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .cams import CamObservation
from .config import DATA

CV_FILE = DATA / "cv_observation.json"


def write_observation(obs: CamObservation, cam_id: str, path: Path = CV_FILE) -> Path:
    blob = obs.as_dict()
    blob["cam_id"] = cam_id
    blob["written_at"] = time.time()
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


def read_observation(path: Path = CV_FILE, max_age_s: float = 900.0):
    """Load the most recent CV reading, if it is fresh enough to trust."""
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - blob.get("written_at", 0) > max_age_s:
        return None
    return blob


def apply_to_observations(observations, blob, net) -> bool:
    """Replace the matching camera's reading with the CV one. Returns True if applied.

    Speed is transplanted as a *ratio* rather than an absolute. The video is
    stock motorway footage standing in for a Kolkata street: its absolute speeds
    belong to a different road with a different free-flow speed, so copying
    120 km/h onto Chowringhee would be nonsense. What transfers legitimately is
    the *state* the vision measured - how congested this camera's view is
    relative to that road's own capacity - which is exactly the quantity the
    nowcaster propagates.
    """
    cam_id = blob.get("cam_id")
    for i, o in enumerate(observations):
        if o.cam_id != cam_id:
            continue
        free_kph = float(net.ekph[o.edge])
        # Occupancy measured by the camera maps onto this road's speed range.
        occ = float(blob.get("occupancy", 0.0))
        density = float(blob.get("density", 0.0))
        congestion = min(1.0, max(occ, density / 135.0))
        kph = max(3.5, free_kph * (1.0 - 0.82 * congestion))

        observations[i] = CamObservation(
            cam_id=cam_id, edge=o.edge, t_s=o.t_s,
            vehicle_count=int(blob.get("count", 0)),
            density_vpkm=density,
            speed_kph=kph,
            queue_m=float(blob.get("queue_m", 0.0)),
            occupancy=occ,
            confidence=float(blob.get("confidence", 0.6)),
            source="cv",
            classes=blob.get("classes", {}),
        )
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyse real video and publish it as a live camera feed")
    ap.add_argument("--video", default=str(DATA / "video" / "vehicles.mp4"))
    ap.add_argument("--cam", default="CAM007", help="camera id to take over")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--near-width", type=float, default=35.0)
    ap.add_argument("--near-depth", type=float, default=51.0)
    ap.add_argument("--horizon", type=float, default=0.14)
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--loop", action="store_true",
                    help="keep re-analysing so the dashboard stays live")
    args = ap.parse_args()

    from .vision import VehicleCounter

    print("Loading YOLO11...")
    vc = VehicleCounter(imgsz=args.imgsz, conf=0.25, lanes=args.lanes,
                        near_width_m=args.near_width,
                        near_depth_m=args.near_depth,
                        horizon_frac=args.horizon)

    while True:
        t0 = time.time()
        stats = vc.analyse(args.video, max_frames=args.frames, stride=args.stride)
        if not stats:
            print("No frames analysed - check the video path.")
            return
        obs = vc.to_observation(stats, args.cam, 0, time.time() % 86400)
        write_observation(obs, args.cam)
        print("[%s] %s  %d vehicles  %.1f km/h  occupancy %.0f%%  conf %.2f  (%.1fs)"
              % (time.strftime("%H:%M:%S"), args.cam, obs.vehicle_count,
                 obs.speed_kph, obs.occupancy * 100, obs.confidence,
                 time.time() - t0))
        print("      -> %s" % CV_FILE)
        if not args.loop:
            break


if __name__ == "__main__":
    main()
