"""Computer vision: real vehicle detection and tracking on camera video.

This is the module that makes the camera story real rather than asserted. It
runs YOLO11 over actual traffic footage, tracks each vehicle across frames, and
produces exactly the same ``CamObservation`` object that the simulated camera
path emits - so the nowcaster, the forecaster and the router genuinely cannot
tell whether a reading came from video or from the simulator.

**How speed comes out of a video.** A camera measures pixels, not metres. Proper
practice is a homography from four surveyed ground points, which we do not have
for arbitrary footage. Instead we use a scale calibration: the operator states
roughly how many metres of road the frame spans, giving metres-per-pixel along
the traffic axis. Tracked displacement over known frame intervals then yields
speed. The estimate is therefore approximate in absolute terms but consistent
over time, which is what the nowcaster actually needs - it consumes the *ratio*
of observed to historical speed, and a constant scale error largely cancels.

**Occlusion is handled honestly.** In dense traffic a single camera under-counts
because vehicles hide behind each other, so ``confidence`` falls as density
rises. Reporting that honestly is what lets the router widen its uncertainty
instead of trusting a bad reading.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .cams import CamObservation

# COCO classes that are road vehicles, with a rough length in metres used to
# convert a count into an occupied-length estimate.
VEHICLE_CLASSES = {
    1: ("bicycle", 1.8),
    2: ("car", 4.2),
    3: ("motorcycle", 2.0),
    5: ("bus", 11.0),
    7: ("truck", 7.5),
}
JAM_DENSITY_VPKM = 135.0


@dataclass
class FrameStats:
    frame: int
    t_s: float
    counts: dict
    total: int
    mean_speed_kph: float
    occupancy: float
    stopped: int


class VehicleCounter:
    """YOLO11 detection + tracking over a video source."""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.30,
                 fov_m: float = 110.0, lanes: int = 3, imgsz: int = 640,
                 device: str | None = None, near_width_m: float = 35.0,
                 horizon_frac: float = 0.18, calibration: float = 1.0,
                 near_depth_m: float = 50.0):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf = conf
        self.fov_m = fov_m              # metres of road along the viewing axis
        self.lanes = lanes
        self.imgsz = imgsz
        self.device = device
        self.near_width_m = near_width_m    # metres spanned by the frame's bottom edge
        self.near_depth_m = near_depth_m    # camera-to-road distance at frame bottom
        self.horizon_frac = horizon_frac    # where the horizon sits, 0=top 1=bottom
        self.calibration = calibration      # final per-camera trim
        self.tracks: dict = defaultdict(lambda: deque(maxlen=12))
        self.frame_w = None
        self.frame_h = None

    # -- geometry -----------------------------------------------------------

    def _depth_at(self, y: float) -> float:
        """Distance from camera to the road point imaged at row ``y``, in metres.

        Traffic cameras look *along* a road, and that geometry has a subtlety
        that is easy to get wrong: longitudinal and lateral scale differ, and
        the difference is a square.

        For a pinhole camera over a flat plane, depth falls off as the inverse
        of a pixel's distance below the horizon::

            depth(y) = D0 * (H - y_h) / (y - y_h)

        Sideways motion therefore compresses as 1/(y - y_h), but motion *along*
        the road - which appears as vertical motion in the image - compresses as
        1/(y - y_h)**2. Applying one isotropic metres-per-pixel scale to both
        crushes exactly the along-road component that carries the speed, which
        is why a naive implementation reports motorway traffic at walking pace.

        Working in depth rather than in scale removes the problem entirely:
        along-road distance is simply the difference of two depths.
        """
        h = float(self.frame_h or 720)
        y_h = self.horizon_frac * h
        denom = max(y - y_h, 0.04 * h)          # clamp: never divide near zero
        return self.near_depth_m * (h - y_h) / denom

    def _lateral_mpp(self, y: float) -> float:
        """Metres per pixel across the road at row ``y``.

        Lateral scale is proportional to depth, so it follows directly from the
        depth model and the one measured quantity we can eyeball from a frame:
        how wide the road is at the bottom edge.
        """
        w = float(self.frame_w or 1280)
        near = self.near_width_m / max(w, 1.0)
        return near * (self._depth_at(y) / max(self.near_depth_m, 1e-6))

    def ground_displacement(self, ya: float, xa: float,
                            yb: float, xb: float) -> float:
        """Ground distance in metres between two image points on the road."""
        d_long = abs(self._depth_at(yb) - self._depth_at(ya))
        d_lat = abs(xb - xa) * self._lateral_mpp(0.5 * (ya + yb))
        return math.hypot(d_long, d_lat) * self.calibration

    # -- the main loop ------------------------------------------------------

    def analyse(self, source, max_frames: int = 400, stride: int = 2,
                progress: bool = False):
        """Run detection+tracking over a video. Returns a list of FrameStats."""
        import cv2

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError("Could not open video source: %s" % source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stats: list[FrameStats] = []
        idx = 0
        used = 0

        while used < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            used += 1
            if self.frame_w is None:
                self.frame_h, self.frame_w = frame.shape[0], frame.shape[1]

            kw = dict(persist=True, conf=self.conf, imgsz=self.imgsz,
                      verbose=False, classes=list(VEHICLE_CLASSES))
            if self.device:
                kw["device"] = self.device
            res = self.model.track(frame, **kw)[0]

            t_s = idx / fps
            stats.append(self._frame_stats(res, used, t_s))
            if progress and used % 25 == 0:
                print("    frame %4d  vehicles=%3d  %.1f km/h"
                      % (used, stats[-1].total, stats[-1].mean_speed_kph))

        cap.release()
        return stats

    def _track_speed(self, tid, t_s: float, cx: float, cy: float):
        """Extend a track and return its speed in km/h, or None while it is
        still too short (under three points) to measure.

        Displacement is integrated in metres using the local ground scale at
        each step, so a track crossing the frame is measured correctly even
        though the scale changes along its path.
        """
        hist = self.tracks[tid]
        hist.append((t_s, cx, cy))
        if len(hist) < 3:
            return None
        dt_track = max(t_s - hist[0][0], 1e-3)
        metres = 0.0
        for (_, xa, ya), (_, xb, yb) in zip(hist, list(hist)[1:]):
            metres += self.ground_displacement(ya, xa, yb, xb)
        return (metres / dt_track) * 3.6

    def _frame_stats(self, res, frame_i: int, t_s: float) -> FrameStats:
        counts: dict = defaultdict(int)
        speeds: list[float] = []
        occupied_m = 0.0
        stopped = 0

        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.id is None:
            ids = []
            xywh = np.zeros((0, 4))
            cls = []
        else:
            ids = boxes.id.int().tolist() if boxes.id is not None else []
            xywh = boxes.xywh.cpu().numpy()
            cls = boxes.cls.int().tolist()

        for i, tid in enumerate(ids):
            c = int(cls[i]) if i < len(cls) else 2
            name, length_m = VEHICLE_CLASSES.get(c, ("car", 4.2))
            counts[name] += 1
            occupied_m += length_m

            v_kph = self._track_speed(tid, t_s, float(xywh[i][0]), float(xywh[i][1]))
            # Reject absurd jumps: a track id swap produces a huge displacement.
            if v_kph is not None and v_kph < 180.0:
                speeds.append(v_kph)
                if v_kph < 3.0:
                    stopped += 1

        total = sum(counts.values())
        mean_v = float(np.median(speeds)) if speeds else 0.0
        occupancy = min(1.0, occupied_m / max(self.fov_m * self.lanes, 1.0))
        return FrameStats(frame=frame_i, t_s=t_s, counts=dict(counts), total=total,
                          mean_speed_kph=mean_v, occupancy=occupancy,
                          stopped=stopped)

    # -- convert to a nowcaster input --------------------------------------

    def to_observation(self, stats, cam_id: str, edge: int, t_s: float,
                       window: int = 40) -> CamObservation:
        """Aggregate the last ``window`` frames into one camera reading.

        Averaging matters: per-frame counts are noisy because detection flickers
        on partly occluded vehicles. A short window smooths that without hiding
        genuine change, since traffic state does not move much in a second or two.
        """
        recent = stats[-window:] if stats else []
        if not recent:
            return CamObservation(cam_id, edge, t_s, 0, 0.0, 0.0, 0.0, 0.0, 0.1, "cv")

        counts = np.array([s.total for s in recent], dtype=float)
        speeds = np.array([s.mean_speed_kph for s in recent if s.mean_speed_kph > 0])
        occ = float(np.mean([s.occupancy for s in recent]))
        stopped = float(np.mean([s.stopped for s in recent]))

        mean_count = float(np.mean(counts))
        # Density from count over the visible stretch, per lane.
        density = mean_count / max(self.fov_m / 1000.0, 1e-6) / max(self.lanes, 1)
        speed = float(np.median(speeds)) if speeds.size else 5.0

        # Class mix, summed over the window and normalised back to one frame.
        mix: dict = defaultdict(float)
        for s in recent:
            for k, v in s.counts.items():
                mix[k] += v
        # Round first, then drop empties: a class averaging 0.4 per frame rounds
        # to zero and should not appear in the mix at all.
        mix = {k: n for k, n in
               ((k, int(round(v / len(recent)))) for k, v in mix.items()) if n > 0}

        # Queue: stopped vehicles times a typical occupied length per vehicle.
        queue_m = min(self.fov_m, stopped * 6.0)

        # Confidence falls with occlusion and with having few tracks to average.
        occl = float(np.clip(density / JAM_DENSITY_VPKM, 0.0, 1.0))
        conf = (1.0 - 0.35 * occl) * min(1.0, 0.45 + 0.1 * max(mean_count, 1.0))
        conf = float(np.clip(conf, 0.12, 0.97))

        return CamObservation(
            cam_id=cam_id, edge=edge, t_s=t_s,
            vehicle_count=int(round(mean_count)),
            density_vpkm=float(density),
            speed_kph=speed,
            queue_m=float(queue_m),
            occupancy=float(np.clip(occ, 0.0, 1.0)),
            confidence=conf,
            source="cv",
            classes=mix,
        )


def annotate(source, out_path: str, model_name: str = "yolo11n.pt",
             max_frames: int = 220, conf: float = 0.30) -> str:
    """Write an annotated clip with boxes, ids and a live count overlay.

    Purely for the demo: being able to play four seconds of real footage with
    boxes on the vehicles answers the 'is the CV real?' question faster than any
    amount of explanation.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Could not open %s" % source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = model.track(frame, persist=True, conf=conf, verbose=False,
                          classes=list(VEHICLE_CLASSES))[0]
        vis = res.plot()
        count = 0 if res.boxes is None else len(res.boxes)
        cv2.rectangle(vis, (0, 0), (360, 64), (12, 16, 22), -1)
        cv2.putText(vis, "TRIFFY  CAM FEED", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (192, 214, 55), 1, cv2.LINE_AA)
        cv2.putText(vis, "vehicles in frame: %d" % count, (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 237, 245), 1, cv2.LINE_AA)
        writer.write(vis)

    cap.release()
    writer.release()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Triffy's vehicle CV on a video")
    ap.add_argument("source", help="video file, or a webcam index like 0")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fov", type=float, default=110.0,
                    help="metres of road visible in frame")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--annotate", default="", help="write an annotated mp4 here")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    print("Loading YOLO11...")
    vc = VehicleCounter(fov_m=args.fov, lanes=args.lanes)
    t0 = time.time()
    stats = vc.analyse(src, max_frames=args.frames, progress=True)
    dt = time.time() - t0

    if not stats:
        print("No frames analysed.")
        return

    obs = vc.to_observation(stats, "CAM-CV", 0, 0.0)
    print()
    print("=" * 62)
    print("REAL COMPUTER VISION ON CAMERA FEED")
    print("=" * 62)
    print("  frames analysed    %d in %.1fs (%.1f fps)" % (len(stats), dt,
                                                           len(stats) / max(dt, 1e-6)))
    print("  vehicles in frame  %d (window mean)" % obs.vehicle_count)
    print("  class mix          %s" % (obs.classes or "-"))
    print("  density            %.1f veh/km/lane" % obs.density_vpkm)
    print("  measured speed     %.1f km/h" % obs.speed_kph)
    print("  queue at stop line %.0f m" % obs.queue_m)
    print("  road occupancy     %.0f%%" % (obs.occupancy * 100))
    print("  confidence         %.2f" % obs.confidence)
    print()
    print("  This is the exact CamObservation shape the simulated cameras emit,")
    print("  so the nowcaster consumes real video and synthetic feeds identically.")
    print("=" * 62)

    if args.annotate:
        print("\nWriting annotated clip...")
        annotate(src, args.annotate, max_frames=args.frames)
        print("  -> %s" % args.annotate)


if __name__ == "__main__":
    main()
