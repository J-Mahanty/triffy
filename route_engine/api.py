"""FastAPI backend for the Triffy dashboard and bots.

A deliberate split in the wire format keeps the live demo smooth: road
*geometry* never changes, so it is fetched once from ``/api/network`` (a few MB),
while ``/api/state`` returns only per-edge numbers as parallel arrays. Sending
polylines on every tick would push megabytes a second and make the map stutter in
front of an audience.
"""
from __future__ import annotations

import json
import mimetypes
import os
import socket
from pathlib import Path

# Windows' MIME registry has no entry for .woff2, so the self-hosted web font
# would be served as application/octet-stream.
mimetypes.add_type("font/woff2", ".woff2")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import DATA, WEB
from .engine import TriffyEngine
from .simulator import fmt_clock

app = FastAPI(title="Triffy", version="0.1")

CAMERA_UNAVAILABLE = "camera feed unavailable"
LIVE_UNAVAILABLE = "live mode unavailable: %s"
ENGINE: TriffyEngine | None = None
LIVE = None


def engine() -> TriffyEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = TriffyEngine()
    return ENGINE


def live_engine():
    """The real-data engine: London Zone 1, real cameras, no simulator.

    Loaded lazily and separately from the simulated engine so a failure here -
    no network, no collected data - can never take down the main dashboard.
    """
    global LIVE
    if LIVE is None:
        from .live_engine import LiveEngine
        LIVE = LiveEngine(city="lon")
    return LIVE


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PlanReq(BaseModel):
    origin: str
    destination: str
    depart: str | None = None
    user: str = "guest"
    k: int = 3


class ClockReq(BaseModel):
    time: str | None = None
    advance_s: float | None = None


class IncidentReq(BaseModel):
    where: str = ""
    severity: float = 2.6
    minutes: float = 30.0
    kind: str = "collision"
    on_route: list[int] | None = None


class LeaveByReq(BaseModel):
    origin: str
    destination: str
    arrive_by: str
    user: str = "guest"
    confidence: float = 0.9


class FeedbackReq(BaseModel):
    user: str
    origin: str
    destination: str
    verdict: str = "down"


# ---------------------------------------------------------------------------
# Static network (sent once)
# ---------------------------------------------------------------------------

@app.get("/api/network")
def api_network():
    """Road geometry and camera positions. Large, cached by the client."""
    eng = engine()
    net = eng.net
    # Only draw roads worth drawing: everything of tertiary rank or better.
    keep = np.nonzero(net.erank <= 5)[0]
    return {
        "city": net.meta.get("label", eng.city),
        # The client hides the demo controls rather than offering buttons that
        # will 403. The middleware is what actually enforces it; this is only
        # so the UI does not lie about what it can do.
        "readonly": READONLY,
        "center": net.meta.get("center", [22.555, 88.355]),
        "bbox": net.meta.get("bbox"),
        "edges": [{"id": int(e), "g": net.egeom[e], "r": int(net.erank[e]),
                   "n": net.ename[e], "f": float(net.ekph[e])} for e in keep],
        "cams": [c.as_dict() for c in eng.cams.cams],
        "stats": {
            "nodes": net.n_nodes, "edges": net.n_edges,
            "km": round(net.total_km, 1),
            "cameras": len(eng.cams.cams),
            "coverage": eng.cams.coverage_stats(),
            "nowcast": eng.nowcaster.coverage_report(),
        },
    }


@app.get("/api/state")
def api_state():
    """Per-tick numbers only: parallel arrays keyed by the edge ids above."""
    eng = engine()
    net, st = eng.net, eng.state
    keep = np.nonzero(net.erank <= 5)[0]
    cong = st.congestion(net)

    obs_by_edge = {o.edge: o for o in eng.observations}
    cams = []
    for c in eng.cams.cams:
        o = obs_by_edge.get(c.edge)
        cams.append({
            "id": c.id,
            "kph": round(o.speed_kph, 1) if o else None,
            "count": o.vehicle_count if o else 0,
            "queue_m": round(o.queue_m) if o else 0,
            "occ": round(o.occupancy, 2) if o else 0,
            "classes": o.classes if o else {},
            "src": o.source if o else "offline",
        })

    incidents = []
    for inc in eng.sim.active_incidents(eng.now_s):
        lat, lon = net.edge_midpoint(int(inc.edges[0]))
        incidents.append({
            "label": inc.label, "kind": inc.kind, "lat": lat, "lon": lon,
            "severity": round(inc.severity, 2),
            "intensity": round(inc.factor(eng.now_s), 2),
            "started": fmt_clock(inc.start_s),
            "clears": fmt_clock(inc.start_s + inc.duration_s),
            "edges": [int(e) for e in inc.edges[:40]],
        })

    return {
        "clock": eng.clock,
        "now_s": round(eng.now_s),
        "ids": [int(e) for e in keep],
        "cong": [round(float(cong[e]), 3) for e in keep],
        "kph": [round(float(st.kph[e]), 1) for e in keep],
        "obs": [int(bool(st.observed[e])) for e in keep],
        "cams": cams,
        "incidents": incidents,
        "stats": eng.network_stats(),
    }


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@app.post("/api/plan")
def api_plan(req: PlanReq):
    eng = engine()
    try:
        plan = eng.plan(req.origin, req.destination, req.depart,
                        user_id=req.user, k=req.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return plan.as_dict(eng.net)


@app.post("/api/leaveby")
def api_leaveby(req: LeaveByReq):
    eng = engine()
    try:
        best, deadline = eng.leave_by(req.origin, req.destination, req.arrive_by,
                                      user_id=req.user, confidence=req.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not best:
        return {"ok": False,
                "detail": "Even leaving right now you are unlikely to make it."}
    depart_s, plan = best
    r = plan.best
    return {
        "ok": True,
        "leave_at": fmt_clock(depart_s),
        "deadline": fmt_clock(deadline),
        "confidence": req.confidence,
        "typical_min": round(r.median_s / 60.0, 1),
        "buffered_min": round(r.percentile_s(req.confidence) / 60.0, 1),
        "route": r.as_dict(eng.net),
    }


@app.post("/api/clock")
def api_clock(req: ClockReq):
    eng = engine()
    if req.time:
        eng.set_clock(req.time)
    elif req.advance_s:
        eng.advance(req.advance_s)
    else:
        eng.tick()
    return {"clock": eng.clock, "stats": eng.network_stats()}


@app.post("/api/incident")
def api_incident(req: IncidentReq):
    eng = engine()
    try:
        inc = eng.inject_incident(req.where, req.severity, req.minutes, req.kind,
                                  on_route=req.on_route)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "label": inc.label, "road": inc.road,
            "clears": fmt_clock(inc.start_s + inc.duration_s)}


@app.get("/api/profiles")
def api_profiles():
    eng = engine()
    return {uid: {"name": p.name, "vehicle": p.vehicle,
                  "risk_aversion": round(p.risk_aversion, 2),
                  "home": p.home, "work": p.work,
                  "trips": p.trips_logged, "summary": p.summary()}
            for uid, p in eng.profiles.profiles.items()}


@app.post("/api/feedback")
def api_feedback(req: FeedbackReq):
    eng = engine()
    try:
        plan = eng.plan(req.origin, req.destination, user_id=req.user, k=1,
                        with_baseline=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    user = eng.profiles.get(req.user)
    changed = user.record_feedback(eng.net, plan.best, req.verdict)
    eng.profiles.save()
    return {"ok": True, "risk_aversion": round(user.risk_aversion, 2),
            "roads_adjusted": changed, "summary": user.summary()}


# ---------------------------------------------------------------------------
# Real live cameras  (the part of the system that is not simulated)
# ---------------------------------------------------------------------------

@app.get("/api/livecams")
def api_livecams():
    from .liveservice import service
    return service().listing()


@app.get("/api/livecam/{cam_id}/image")
def api_livecam_image(cam_id: str):
    """A real camera still with YOLO detections drawn on it."""
    from .liveservice import service
    path, n, age = service().annotated_image(cam_id)
    if path is None:
        raise HTTPException(status_code=503, detail=CAMERA_UNAVAILABLE)
    return FileResponse(str(path), media_type="image/jpeg", headers={
        "X-Vehicle-Count": str(n),
        "X-Feed-Age-Seconds": str(int(age)) if age is not None else "unknown",
        "Cache-Control": "no-cache",
    })


@app.get("/api/livecam/{cam_id}/info")
def api_livecam_info(cam_id: str):
    """Vehicle count and feed age, without re-sending the image bytes.

    A separate endpoint rather than a HEAD on the image route: FastAPI registers
    GET-only handlers, so a HEAD would 405, and the dashboard polls this once a
    minute per camera.
    """
    from .liveservice import service
    path, n, age = service().annotated_image(cam_id)
    if path is None:
        raise HTTPException(status_code=503, detail=CAMERA_UNAVAILABLE)
    return {"cam_id": cam_id, "vehicles": n,
            "feed_age_s": round(age) if age is not None else None}


@app.get("/api/livecam/{cam_id}/measure")
def api_livecam_measure(cam_id: str):
    """Full tracked measurement from the camera's latest video clip."""
    from .liveservice import service
    meas = service().measurement(cam_id)
    if not meas:
        raise HTTPException(status_code=503, detail=CAMERA_UNAVAILABLE)
    return meas


def _clip_job(cam_id: str):
    from .clipviewer import viewer
    from .liveservice import service
    cam = service()._find(cam_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="unknown camera")
    return viewer().job(cam)


@app.get("/api/livecam/{cam_id}/clip")
def api_livecam_clip(cam_id: str):
    """Status of this camera's latest-clip job; starts one if none is fresh.

    The dashboard polls this for progress and the clip summary while the
    stream below plays.
    """
    return _clip_job(cam_id).status()


@app.get("/api/livecam/{cam_id}/clip/stream")
def api_livecam_clip_stream(cam_id: str):
    """The clip as annotated frames (MJPEG): streamed as they are tracked, then
    looped at the clip's own frame rate."""
    from .clipviewer import viewer
    job = _clip_job(cam_id)
    return StreamingResponse(viewer().mjpeg(job),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# LIVE MODE — real network, real cameras, no simulator anywhere
# ---------------------------------------------------------------------------

@app.get("/api/live/network")
def api_live_network():
    try:
        from .clipviewer import prewarm
        prewarm()                       # so the first camera click is quick
        return live_engine().network_geometry()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=LIVE_UNAVAILABLE % exc)


@app.get("/api/live/state")
def api_live_state():
    try:
        eng = live_engine()
        eng.refresh()
        return eng.live_state()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=LIVE_UNAVAILABLE % exc)


@app.post("/api/live/plan")
def api_live_plan(req: PlanReq):
    try:
        return live_engine().plan(req.origin, req.destination,
                                  user_id=req.user, k=req.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=LIVE_UNAVAILABLE % exc)


@app.get("/api/validation")
def api_validation():
    """Forecast accuracy measured on real collected camera data.

    The dashboard renders this under a REAL DATA badge, so the endpoint refuses
    to serve anything whose provenance does not say it came from the collector's
    own observation log. A test run against a synthetic series once overwrote
    this file and the badge kept claiming the numbers were measured; checking
    provenance here means the UI cannot repeat that on its own.
    """
    path = DATA / "validation.json"
    if not path.exists():
        return JSONResponse(
            {"detail": "Run the collector, then: python -m route_engine.validate"},
            status_code=404)
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not blob.get("from_collector"):
        return JSONResponse(
            {"detail": "Stored validation did not come from the collector log "
                       "and will not be shown as real. Re-run: "
                       "python -m route_engine.validate",
             "source": blob.get("source", "unknown")},
            status_code=409)
    return blob


@app.get("/api/benchmark")
def api_benchmark():
    path = DATA / "benchmark.json"
    if not path.exists():
        return JSONResponse({"detail": "Run: python -m route_engine.benchmark"},
                            status_code=404)
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/suggest")
def api_suggest(q: str = ""):
    eng = engine()
    return {"suggestions": eng.net.suggest(q, limit=8)}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    page = WEB / "index.html"
    if not page.exists():
        # The dashboard is a separate piece of the project; the API works
        # without it, so say where things are rather than failing with a 500.
        return JSONResponse({"service": "triffy",
                             "dashboard": "not installed (expected web/index.html)",
                             "docs": "/docs"})
    return FileResponse(str(page))


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


READONLY = os.environ.get("TRIFFY_READONLY", "").strip() in ("1", "true", "yes")
"""Viewer mode: serve the dashboard, refuse anything that changes shared state.

This API was written for loopback, where every caller is the person presenting.
Sharing the URL breaks that assumption: `POST /api/clock` moves the network
clock the dashboard AND the Telegram bot read, and `POST /api/incident` closes
a road. Either one is a single curl away for anyone holding the link, and both
change what the presenter sees mid-demo.

The Telegram bot already guards the clock behind an admin check for exactly
this reason (see its DEMO_CONTROL). Viewer mode is the same guard for the web.

Planning is *not* blocked — routing is the thing people came to look at, and it
only reads.
"""

# Endpoints that change state everyone shares, or write to disk.
WRITES_SHARED_STATE = ("/api/clock", "/api/incident", "/api/feedback")


@app.middleware("http")
async def _guard_readonly(request, call_next):
    """Refuse shared-state writes in viewer mode, with a reason.

    A middleware rather than a check inside each handler: a new mutating route
    added later is covered by default, and the person who adds it has to think
    about this list rather than silently opting out of it.
    """
    if READONLY and request.url.path in WRITES_SHARED_STATE:
        return JSONResponse(
            status_code=403,
            content={"detail": "Viewer mode: the demo clock, incidents and "
                               "feedback are controlled from the presenter's "
                               "machine. Routing and the live data are open."})
    return await call_next(request)


def _lan_addresses() -> list:
    """This machine's addresses on the networks it has joined, loopback aside.

    Found by opening a UDP socket toward a public address and asking which
    local address the routing table chose. Nothing is sent — UDP connect() only
    sets the peer — so this works with no network and no DNS.
    """
    out = []
    for probe in ("8.8.8.8", "1.1.1.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127.") and ip not in out:
                out.append(ip)
        except OSError:
            pass                      # no route right now; not an error here
        finally:
            s.close()
    return out


def _point_map_links_at(ip: str, port: int) -> None:
    """Aim the chat's "see it on the map" links at this machine's LAN address.

    Serving on the network and leaving MAP_BASE at its 127.0.0.1 default is
    never what anyone wants: the dashboard becomes reachable from a phone while
    the chat keeps handing out a link that phone cannot possibly open. Two
    settings that must agree is one setting too many, and the failure is
    silent — so deriving it here removes the mistake rather than documenting it.

    Only when MAP_BASE is still loopback. An operator who set it deliberately
    (a tunnel, a hostname) knows better than we do.
    """
    from . import chat as chat_mod

    if not chat_mod.MAP_BASE or not chat_mod._is_loopback(chat_mod.MAP_BASE):
        return
    chat_mod.MAP_BASE = "http://%s:%d" % (ip, port)
    print("  Chat map links now point at %s" % chat_mod.MAP_BASE, flush=True)


def _loopback_sockets(port: int) -> list:
    """Listening sockets on BOTH loopback stacks: 127.0.0.1 and [::1].

    Binding only "127.0.0.1" is why `http://localhost:8000` was refused during
    setup while `http://127.0.0.1:8000` worked. On Windows `localhost` resolves
    to the IPv6 loopback `::1` first; nothing was listening there, so the
    browser got connection-refused. `curl` hides this by falling back to IPv4,
    browsers frequently do not — and a browser that has cached the failure keeps
    refusing after the server is fine.

    Two explicit loopback sockets rather than binding "::" or "0.0.0.0": those
    would also publish the dashboard, and the live camera feeds, to every
    machine on whatever network the demo laptop is joined to. Set
    TRIFFY_HOST to do that deliberately (see main), not by accident.
    """
    socks = []
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port))):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                # Keep the two sockets independent; without this the IPv6
                # socket may claim IPv4 too and the second bind fails.
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            s.bind(addr)
            s.listen(128)
            socks.append(s)
        except OSError as exc:
            # One stack missing is survivable; both is not, and main() says so.
            print("could not listen on %s (%s)" % (addr[0], type(exc).__name__),
                  flush=True)
    return socks


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Triffy web server")
    ap.add_argument("--replay", default="", metavar="'YYYY-MM-DD HH:MM'",
                    help="replay recorded London traffic from this moment "
                         "(London time) instead of live readings")
    args = ap.parse_args()
    if args.replay:
        from .live_engine import parse_replay
        parse_replay(args.replay)          # fail now, not on the first request
        os.environ["TRIFFY_REPLAY"] = args.replay
        print("London will replay recorded traffic from %s (London time)."
              % args.replay, flush=True)
    import uvicorn
    engine()  # boot before serving so the first request is fast

    port = int(os.environ.get("TRIFFY_PORT", "8000"))
    host = os.environ.get("TRIFFY_HOST", "").strip()
    if host:
        # Deliberate exposure, e.g. TRIFFY_HOST=0.0.0.0 so a phone on the same
        # wi-fi can open the map links the chat sends. Say it out loud, because
        # it also exposes the live camera feeds to that network.
        print("Serving on %s:%d — reachable from this network." % (host, port),
              flush=True)
        # Print the address a PHONE should actually type. "0.0.0.0" is a bind
        # address, not a destination, and nobody can guess the LAN IP from it —
        # which is how a phone ends up being handed 127.0.0.1 and failing.
        lan = _lan_addresses()
        for ip in lan:
            print("  On this network:  http://%s:%d" % (ip, port), flush=True)
        if lan:
            _point_map_links_at(lan[0], port)
        print("  If a phone still cannot reach it, the firewall is blocking "
              "inbound %d." % port, flush=True)
        uvicorn.run(app, host=host, port=port, log_level="warning")
        return

    socks = _loopback_sockets(port)
    if not socks:
        raise SystemExit("Nothing could listen on port %d — is it already in "
                         "use? Try TRIFFY_PORT=8001." % port)
    print("Dashboard: http://localhost:%d  and  http://127.0.0.1:%d"
          % (port, port), flush=True)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=socks)


if __name__ == "__main__":
    main()
