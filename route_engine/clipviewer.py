"""Play a real camera's latest clip with our detections and tracking drawn on it.

Transport for London does not publish live streams. Each JamCam publishes a
short clip (~11 s, 25 fps, 352x288) that it replaces every few minutes. So what
the dashboard shows is *the latest clip*, labelled with its age, never "a live
stream" - anyone who checked would find out, and the credibility of the whole
real-data half rests on not overclaiming.

How it works: clicking a camera starts one background job that downloads the
clip and runs YOLO11 + ByteTrack over it. Frames go into a shared list as they
are produced, and any number of browser streams (MJPEG) read from that list,
each paced to the clip's own frame rate - so video appears as soon as the first
frame is tracked. Once the clip is done, the same stream loops the cached
frames at the clip's real frame rate, so replays cost nothing. (An earlier
version also wrote a VP8 WebM for replay; encoding cost 55 ms a frame, half as
much again as tracking, for no visible benefit, so it went.)

Box colours use *the collector's own rule* for moving vs stopped (same stride,
same 40-frame window, same threshold as LiveCameraReader._moving_fraction), so
what the viewer shows is the signal the live engine actually consumes, not a
second, subtly different definition.

Everything runs on the CPU: the GPU belongs to the collector, and two
processes sharing a small laptop GPU can hang it. With collector B also on the CPU, a frame costs
~110-160 ms at the collector's imgsz=640: measured in the dashboard, the first
annotated frame appears ~2 s after the click, the first pass of an 11 s clip
takes ~21 s (about 0.5x speed, with a progress readout), and every loop after
that plays at the clip's real frame rate. Reopening a cached clip shows its
first frame in ~0.4 s. Dropping to imgsz=352 would
halve that, but would detect a different set of small/distant vehicles than
the collector does, and the point of the viewer is to show *its* measurement.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import requests

from .config import DATA
from .livecams import (DASH_DEVICE, UA, VEHICLE_CLASS_IDS, LiveCameraReader,
                       limit_cpu_threads)

VIEW_DIR = DATA / "live" / "viewer"   # separate from the collector's clip files
VIEW_DIR.mkdir(parents=True, exist_ok=True)

STRIDE = 2              # same as the collector: every 2nd frame
MOVE_WINDOW = 40        # processed frames, the collector's max_frames
SCALE = 2               # 352x288 is tiny on a projector; draw at 2x
CLIP_TTL_S = 150.0      # TfL replaces clips every few minutes
ERROR_RETRY_S = 20.0
MAX_JOBS = 8            # finished jobs kept in memory (~8 MB of JPEGs each)

GREEN = (128, 222, 74)   # BGR: moving
AMBER = (102, 209, 255)  # BGR: stopped
GREY = (170, 170, 170)   # BGR: tracked too briefly to judge


class ClipJob:
    def __init__(self, cam):
        self.cam = cam
        self.state = "downloading"       # -> processing -> done | error
        self.error = None
        self.frames: list = []           # annotated JPEG bytes, in order
        self.fps_out = 12.5
        self.n_expected = 0
        self.clip_age_s = None
        self.started = time.time()
        self.finished = None
        self.summary: dict = {}
        self.cond = threading.Condition()

    def status(self) -> dict:
        with self.cond:
            done = len(self.frames)
        return {
            "cam_id": self.cam.id, "state": self.state, "error": self.error,
            "frames_done": done, "frames_expected": self.n_expected,
            "fps": self.fps_out,
            "clip_age_s": (round(self.clip_age_s + (time.time() - self.started))
                           if self.clip_age_s is not None else None),
            "summary": self.summary,
        }


class ClipViewer:
    def __init__(self):
        self._jobs: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        # One clip is tracked at a time: the CPU is shared with collector B and
        # the API, and a queue of two is fine for a demo.
        self._run_lock = threading.Lock()

    def job(self, cam) -> ClipJob:
        """The current job for this camera, starting a fresh one if needed."""
        now = time.time()
        with self._lock:
            j = self._jobs.get(cam.id)
            stale = j is not None and (
                (j.state == "done" and now - j.finished > CLIP_TTL_S) or
                (j.state == "error" and now - j.finished > ERROR_RETRY_S))
            if j is None or stale:
                j = ClipJob(cam)
                self._jobs[cam.id] = j
                threading.Thread(target=self._run, args=(j,), daemon=True).start()
            self._jobs.move_to_end(cam.id)
            while len(self._jobs) > MAX_JOBS:
                old_id, old = next(iter(self._jobs.items()))
                if old.state not in ("done", "error"):
                    break
                self._jobs.pop(old_id)
            return j

    # -- the worker ---------------------------------------------------------

    def _fail(self, j: ClipJob, msg: str) -> None:
        with j.cond:
            j.state, j.error, j.finished = "error", msg, time.time()
            j.cond.notify_all()

    def _run(self, j: ClipJob) -> None:
        try:
            self._process(j)
        except Exception as exc:  # a dead camera must never take the API down
            self._fail(j, "%s: %s" % (type(exc).__name__, exc))

    def _process(self, j: ClipJob) -> None:
        src = self._download(j)
        if src is None:
            return None
        with self._run_lock:
            tracked = self._track_clip(j, src)
        if tracked is None:
            return None
        counts, tracks, w = tracked
        if not counts:
            return self._fail(j, "the clip had no frames")
        self._finish(j, counts, tracks, w)
        return None

    def _download(self, j: ClipJob):
        """Fetch the camera's latest clip to disk; its path, or None on failure."""
        cam = j.cam
        r = requests.get(cam.video_url, headers=UA, timeout=60)
        if r.status_code != 200 or not r.content:
            self._fail(j, "the camera is not publishing a clip right now")
            return None
        lm = r.headers.get("Last-Modified")
        if lm:
            try:
                from email.utils import parsedate_to_datetime
                j.clip_age_s = time.time() - parsedate_to_datetime(lm).timestamp()
            except Exception:
                pass
        src = VIEW_DIR / ("%s.mp4" % cam.id)
        src.write_bytes(r.content)
        return src

    def _track_clip(self, j: ClipJob, src):
        """Track every STRIDE-th frame, publishing annotated JPEGs as they are
        made. (counts, tracks, frame width), or None if the clip is unreadable."""
        import cv2
        from ultralytics import YOLO

        limit_cpu_threads()
        model = YOLO("yolo11n.pt")   # fresh: ByteTrack state lives in the model
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            self._fail(j, "could not decode the clip")
            return None
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        with j.cond:
            j.fps_out = fps_in / STRIDE
            j.n_expected = max(1, total // STRIDE)
            j.state = "processing"
            j.cond.notify_all()

        tracks: dict = {}          # tid -> [(proc_idx, cx, cy)] in source px
        counts = []
        w = 0
        for used, frame in _every_stride(cap):
            w = frame.shape[1]
            res = model.track(frame, persist=True, conf=0.25, imgsz=640,
                              verbose=False, classes=VEHICLE_CLASS_IDS,
                              device=DASH_DEVICE)[0]
            boxes = _frame_boxes(res.boxes, tracks, used, model.names)
            counts.append(len(boxes))

            vis = _draw(frame, boxes, tracks, used, w)
            _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with j.cond:
                j.frames.append(jpg.tobytes())
                j.cond.notify_all()
        cap.release()
        return counts, tracks, w

    def _finish(self, j: ClipJob, counts: list, tracks: dict, w: int) -> None:

        # The collector's number for this clip: its exact function, its window.
        early = {t: [p for p in pts if p[0] <= MOVE_WINDOW] for t, pts in tracks.items()}
        with j.cond:
            j.summary = {
                "vehicles_mean": round(sum(counts) / len(counts), 1),
                "vehicles_max": max(counts),
                "tracks": len(tracks),
                "moving_pct_collector_rule": round(
                    100 * LiveCameraReader._moving_fraction(early, max(w, 1))),
                "frames": len(counts),
            }
            j.state, j.finished = "done", time.time()
            j.cond.notify_all()

    # -- streaming ----------------------------------------------------------

    def mjpeg(self, j: ClipJob, loop: bool = True, max_s: float = 600.0):
        """Frames as a multipart JPEG stream, never faster than the clip's rate.

        While tracking is running this plays frames as they arrive; once the
        clip is done it loops the cached frames (with a short hold on the last
        one, so the restart reads as a restart). ``max_s`` bounds a stream a
        forgotten browser tab would otherwise hold open for ever.
        """
        i, last, t_end = 0, 0.0, time.time() + max_s
        while time.time() < t_end:
            with j.cond:
                while i >= len(j.frames) and j.state not in ("done", "error"):
                    j.cond.wait(timeout=1.0)
                if i >= len(j.frames):
                    if j.state != "done" or not loop or not j.frames:
                        return
                    i, hold = 0, 0.8
                else:
                    hold = 0.0
                frame = j.frames[i]
                period = 1.0 / max(j.fps_out, 1.0)
            # Pace from the previous frame, not from a fixed start: if tracking
            # falls behind and then catches up, we must not burst to "make up".
            wait = last + period + hold - time.time()
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            i += 1


def _every_stride(cap):
    """Yield (n, frame) for every STRIDE-th frame of a clip, n from 1."""
    idx = used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        idx += 1
        if idx % STRIDE:
            continue
        used += 1
        yield used, frame


def _frame_boxes(b, tracks: dict, used: int, names) -> list:
    """One frame's detections as (x1, y1, x2, y2, track id, class name),
    extending each track's centre path as a side effect."""
    boxes = []
    if b is None or not len(b):
        return boxes
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.int().tolist()
    ids = b.id.int().tolist() if b.id is not None else [None] * len(cls)
    for k in range(len(cls)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[k])
        tid = ids[k]
        if tid is not None:
            tracks.setdefault(tid, []).append((used, (x1 + x2) / 2, (y1 + y2) / 2))
        boxes.append((x1, y1, x2, y2, tid, names[int(cls[k])]))
    return boxes


def _draw(frame, boxes, tracks, now_idx: int, frame_w: int):
    """Annotate one frame at SCALE x, colouring each vehicle by the collector's
    moving/stopped rule evaluated over the trailing MOVE_WINDOW frames."""
    import cv2

    vis = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    thresh = max(2.0, 0.012 * frame_w)     # LiveCameraReader._moving_fraction
    n_mov = n_stop = 0
    for x1, y1, x2, y2, tid, name in boxes:
        pts = [p for p in tracks.get(tid, []) if p[0] > now_idx - MOVE_WINDOW]
        if tid is None or len(pts) < 3:
            col = GREY
        else:
            d = ((pts[-1][1] - pts[0][1]) ** 2 + (pts[-1][2] - pts[0][2]) ** 2) ** 0.5
            if d > thresh:
                col = GREEN
                n_mov += 1
            else:
                col = AMBER
                n_stop += 1
        p1 = (int(x1 * SCALE), int(y1 * SCALE))
        p2 = (int(x2 * SCALE), int(y2 * SCALE))
        cv2.rectangle(vis, p1, p2, col, 2)
        label = "%s #%s" % (name, tid) if tid is not None else name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_top = max(0, p1[1] - th - 6)
        cv2.rectangle(vis, (p1[0], y_top), (p1[0] + tw + 6, y_top + th + 6), col, -1)
        cv2.putText(vis, label, (p1[0] + 3, y_top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    # Heads-up strip: what is on screen right now.
    strip = vis[0:30].copy()
    vis[0:30] = (strip * 0.35).astype(strip.dtype)
    hud = "YOLO11 + ByteTrack   moving %d   stopped %d" % (n_mov, n_stop)
    cv2.putText(vis, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    return vis


_WARM = threading.Event()


def prewarm() -> None:
    """Import torch/ultralytics in the background when London mode opens, so
    the first camera click does not pay a multi-second cold start."""
    if _WARM.is_set():
        return
    _WARM.set()

    def go():
        try:
            limit_cpu_threads()
            from ultralytics import YOLO
            YOLO("yolo11n.pt")
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


VIEWER: ClipViewer | None = None


def viewer() -> ClipViewer:
    global VIEWER
    if VIEWER is None:
        VIEWER = ClipViewer()
    return VIEWER
