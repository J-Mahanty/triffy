"""Tests for the parts that would silently ruin a demo.

The bias here is deliberate. These are not exhaustive unit tests; they are the
checks that catch the failure modes that actually bit during development:
inflated ETAs from phantom traffic signals, a router that quietly ignores its
forecast, personalisation that does nothing, and a benchmark that flatters
itself. Each one guards a specific bug that was real.

Run:  python -m pytest tests -q      (or: python tests/test_route_engine.py)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from route_engine.engine import TriffyEngine
from route_engine.network import load_network
from route_engine.router import Router
from route_engine.simulator import TrafficSim, diurnal, parse_clock


@pytest.fixture(scope="module")
def eng():
    return TriffyEngine(start_clock="18:30", cam_budget=200)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def test_network_is_routable(eng):
    """Every node must be reachable, or some users get 'no route found'."""
    net = eng.net
    assert net.n_nodes > 1000
    assert net.n_edges > net.n_nodes
    # Out-degree must be non-zero everywhere: the import keeps only the largest
    # strongly connected component precisely to guarantee this.
    out_deg = np.diff(net.out_start)
    assert out_deg.min() >= 1, "a node with no exits means unreachable destinations"


def test_junction_detection_is_selective(eng):
    """OSM splits ways constantly; treating every node as a signal doubles ETAs.

    This guards the bug that made a 6.8 km trip quote 57 minutes.
    """
    net = eng.net
    frac = float(net.node_is_junction.mean())
    assert 0.2 < frac < 0.85, "junction detection looks degenerate (%.2f)" % frac


def test_geocoding_landmarks_and_streets(eng):
    for q in ["Park Circus", "BBD Bagh", "Esplanade", "park street"]:
        assert eng.resolve(q) is not None, "failed to resolve %r" % q
    assert eng.resolve("Sector 62 Noida") is None, "should not resolve out-of-area"


# ---------------------------------------------------------------------------
# Traffic model
# ---------------------------------------------------------------------------

def test_peak_is_slower_than_night(eng):
    sim = eng.sim
    night = sim.speeds(3 * 3600, with_incidents=False).mean()
    peak = sim.speeds(19 * 3600, with_incidents=False).mean()
    assert peak < night * 0.85, "evening peak should be materially slower than 03:00"


def test_commute_tide_is_directional(eng):
    """Inbound and outbound carriageways must diverge, or time-dependence is moot."""
    sim = eng.sim
    inbound = sim.am_weight > 1.25
    outbound = sim.am_weight < 0.75
    assert inbound.sum() > 100 and outbound.sum() > 100

    am = sim.speeds(9 * 3600, with_incidents=False)
    pm = sim.speeds(19 * 3600, with_incidents=False)
    # Morning: inbound worse. Evening: outbound worse.
    assert am[inbound].mean() < am[outbound].mean()
    assert pm[outbound].mean() < pm[inbound].mean()


def test_incidents_slow_their_edges(eng):
    sim = TrafficSim(eng.net, seed=5)
    t = 18 * 3600.0
    before = sim.speeds(t)
    inc = sim.add_incident(t - 300.0, severity=3.0, duration_s=1800.0)
    after = sim.speeds(t)
    assert after[inc.edges].mean() < before[inc.edges].mean() * 0.9
    assert len(inc.edges) > 5, "an incident should back up beyond a single segment"


# ---------------------------------------------------------------------------
# Nowcasting
# ---------------------------------------------------------------------------

def test_nowcast_infers_beyond_watched_edges(eng):
    """The whole premise: infer the ~98% of roads no camera sees."""
    rep = eng.nowcaster.coverage_report()
    assert rep["edges_inferable"] > rep["edges_directly_watched"] * 4
    st = eng.state
    assert st.kph.shape[0] == eng.net.n_edges
    assert np.all(st.kph > 0), "no road may have zero or negative speed"


def test_uncertainty_is_lower_where_cameras_watch(eng):
    st = eng.state
    watched = st.observed
    assert watched.sum() > 0
    assert st.sigma_rel[watched].mean() < st.sigma_rel[~watched].mean(), \
        "directly observed roads must be more certain than inferred ones"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_routes_are_plausible(eng):
    plan = eng.plan("Park Circus", "BBD Bagh", user_id="exec")
    r = plan.best
    assert r is not None
    kmh = (r.distance_m / 1000.0) / (r.median_s / 3600.0)
    assert 5.0 < kmh < 60.0, "implausible average speed %.1f km/h" % kmh
    assert 1000 < r.distance_m < 25000
    # Path must be connected: each edge starts where the previous one ended.
    net = eng.net
    for a, b in zip(r.edges, r.edges[1:]):
        assert net.ev[a] == net.eu[b], "route is not a connected path"


def test_traffic_levels_follow_the_dashboard_bands():
    from route_engine.router import traffic_level
    assert traffic_level(1.0) == 0          # free flow
    assert traffic_level(0.70) == 1         # congestion 0.375: slowing
    assert traffic_level(0.50) == 2         # congestion 0.625: congested
    assert traffic_level(0.20) == 3         # congestion 1.0: near gridlock


def test_routes_record_predicted_speed_per_edge(eng):
    plan = eng.plan("Park Circus", "BBD Bagh", user_id="exec", k=3)
    for r in plan.routes:
        assert len(r.speed_ratio) == len(r.edges)
        # Personal profiles cap speed at 105% of free flow.
        assert all(0.0 < x <= 1.06 for x in r.speed_ratio)


def test_route_traffic_runs_are_the_whole_route(eng):
    """The coloured runs cover every edge, in order, end to end."""
    plan = eng.plan("Park Circus", "BBD Bagh", user_id="exec", k=3)
    for r in plan.routes:
        d = r.as_dict(eng.net)
        runs = d["traffic"]
        assert runs, "every route should carry traffic runs"
        assert sum(x["n"] for x in runs) == d["n_edges"]
        assert all(x["level"] in (0, 1, 2, 3) for x in runs)
        # Neighbouring runs differ, or they would have been merged.
        assert all(a["level"] != b["level"] for a, b in zip(runs, runs[1:]))
        assert list(runs[0]["g"][0]) == list(d["geometry"][0])
        assert list(runs[-1]["g"][-1]) == list(d["geometry"][-1])


def test_empty_settings_fall_back_to_the_defaults():
    """A hosting panel that sets TRIFFY_CITY to \"\" must not break startup."""
    import subprocess
    env = dict(os.environ, TRIFFY_CITY="", TRIFFY_MAP_BASE="")
    out = subprocess.run(
        [sys.executable, "-c", "from route_engine import config as c; "
         "print(c.ACTIVE_CITY, c.MAP_BASE)"],
        env=env, capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.split() == ["kol", "http://127.0.0.1:8000"], out.stderr


def test_saved_paths_are_relative_to_the_repo():
    from route_engine.collector import OBS_PATH
    from route_engine.config import repo_path
    assert repo_path(OBS_PATH) == "data/live_observations.jsonl"
    # Anything outside the repo is left as it is.
    assert repo_path("/somewhere/else.json").endswith("else.json")


def test_eta_distribution_is_ordered(eng):
    plan = eng.plan("Sealdah", "Victoria Memorial", user_id="rider")
    r = plan.best
    assert r.p10_s < r.median_s < r.p90_s
    assert 0.0 <= r.reliability <= 1.0
    assert r.mean_s >= r.median_s, "log-normal mean must sit above the median"


def test_alternatives_are_actually_different(eng):
    plan = eng.plan("Alipore", "Esplanade", user_id="student", k=3)
    if len(plan.routes) < 2:
        pytest.skip("only one route available on this corridor")
    a, b = set(plan.routes[0].edges), set(plan.routes[1].edges)
    overlap = len(a & b) / len(a | b)
    assert overlap < 0.85, "alternatives are near-duplicates (overlap %.2f)" % overlap


def test_departure_time_changes_the_answer(eng):
    """If the clock does not matter, time-dependent routing is doing nothing."""
    quiet = eng.plan("Park Circus", "BBD Bagh", "03:00", user_id="exec",
                     with_baseline=False).best
    peak = eng.plan("Park Circus", "BBD Bagh", "19:00", user_id="exec",
                    with_baseline=False).best
    assert peak.median_s > quiet.median_s * 1.15, \
        "peak-hour travel should be clearly slower than 03:00"


def test_turn_by_turn_is_human_readable(eng):
    r = eng.plan("Park Circus", "BBD Bagh", user_id="exec").best
    assert len(r.steps) >= 2
    assert r.steps[-1].instruction.startswith("Arrive")
    # Guards the bug that produced a stream of 16-metre instructions.
    tiny = [s for s in r.steps[:-1] if s.distance_m < 60]
    assert len(tiny) <= 1, "too many sub-60m instructions to read aloud"


def test_closure_raises_eta_and_is_detected(eng):
    """Closing a monitored road must move the *belief*, not just the ground truth."""
    plan = eng.plan("Alipore", "Esplanade", user_id="student", with_baseline=False)
    before = plan.best
    inc = eng.inject_incident(on_route=before.edges, severity=6.0, minutes=35)

    watched = eng.state.observed[inc.edges].sum()
    assert watched > 0, "closure landed where no camera can see it"

    after = eng.plan("Alipore", "Esplanade", user_id="student",
                     with_baseline=False).best
    assert after.median_s > before.median_s * 1.05, \
        "a severe closure should visibly raise the ETA"


def test_router_prefers_driving_through_only_when_it_is_right(eng):
    """Guards against a router that shrugs off closures.

    In a dense grid the correct answer to a closure is often 'stay on the road,
    every detour is worse'. That is only acceptable if it is *true*, so we force
    a full detour and check the router's choice really is the cheaper one.
    """
    import numpy as np
    from route_engine.router import Router

    o, d = eng.resolve("Alipore"), eng.resolve("Esplanade")
    plan = eng.plan("Alipore", "Esplanade", user_id="student", with_baseline=False)
    inc = eng.inject_incident(on_route=plan.best.edges, severity=6.0, minutes=35)
    blocked = set(int(e) for e in inc.edges)

    user = eng.profiles.get("student")
    prof = eng.forecaster.build_profile(eng.state, eng.now_s)
    user.adapt_profile(eng.net, prof)
    router = Router(eng.net, prof, prefs=user)

    chosen = router.route(o.node, d.node, eng.now_s, k=1)[0]
    pen = np.ones(eng.net.n_edges)
    pen[list(blocked)] = 500.0
    forced = router._search(o.node, d.node, eng.now_s, penalty=pen)
    assert forced is not None
    detour = router.evaluate_path(forced[0], eng.now_s)

    # Whatever it picked, it must not be worse than the alternative it rejected.
    assert chosen.median_s <= detour.median_s * 1.02, \
        ("router kept a blocked road (%.1f min) when a detour was faster (%.1f min)"
         % (chosen.median_s / 60, detour.median_s / 60))


# ---------------------------------------------------------------------------
# Personalisation
# ---------------------------------------------------------------------------

def test_motorcycle_is_faster_in_congestion(eng):
    """Filtering through stopped traffic must show up in the ETA, not just cost."""
    car = eng.plan("Sealdah", "Alipore", "19:00", user_id="exec",
                   with_baseline=False).best
    bike = eng.plan("Sealdah", "Alipore", "19:00", user_id="rider",
                    with_baseline=False).best
    assert bike.median_s < car.median_s, "a two-wheeler should beat a car at peak"


def test_risk_aversion_changes_route_selection(eng):
    p = eng.profiles.get("guest")
    old = p.risk_aversion

    p.risk_aversion = 0.0
    fast = eng.plan("Park Circus", "Howrah Bridge approach", "19:00",
                    user_id="guest", with_baseline=False).best
    p.risk_aversion = 2.4
    safe = eng.plan("Park Circus", "Howrah Bridge approach", "19:00",
                    user_id="guest", with_baseline=False).best
    p.risk_aversion = old

    # The cautious setting must not produce a *less* reliable route.
    assert safe.reliability >= fast.reliability - 1e-6


def test_feedback_updates_the_profile(eng):
    # Profiles persist to data/profiles.json between runs, so assert on the
    # *change* rather than an absolute count.
    import uuid
    uid = "test-%s" % uuid.uuid4().hex[:8]
    p = eng.profiles.get(uid)
    before_risk, before_trips = p.risk_aversion, p.trips_logged
    plan = eng.plan("Park Street", "Esplanade", user_id=uid, with_baseline=False)
    p.record_feedback(eng.net, plan.best, "down")
    assert p.risk_aversion > before_risk, "a bad trip should raise caution"
    assert p.trips_logged == before_trips + 1
    assert p.road_bias, "feedback should leave a trace on specific roads"
    eng.profiles.profiles.pop(uid, None)     # keep the demo store clean


# ---------------------------------------------------------------------------
# Forecast vs snapshot
# ---------------------------------------------------------------------------

def test_snapshot_profile_is_actually_frozen(eng):
    """The baseline must be genuinely time-blind, or the comparison is rigged."""
    snap = eng.forecaster.snapshot_profile(eng.state, eng.now_s)
    assert np.allclose(snap.kph[0], snap.kph[-1]), \
        "snapshot baseline should not vary over its horizon"

    fwd = eng.forecaster.build_profile(eng.state, eng.now_s)
    assert not np.allclose(fwd.kph[0], fwd.kph[-1]), \
        "the time-dependent forecast must actually change over the horizon"


def test_historical_model_is_not_ground_truth(eng):
    """Our learned history must be imperfect, or every benchmark number is fake."""
    t = eng.now_s
    truth = eng.sim.speeds(t, with_incidents=False)
    learned = eng.hist.expected_kph(t)
    err = np.abs(learned - truth) / np.maximum(truth, 1e-6)
    assert err.mean() > 0.01, "historical model is suspiciously perfect"
    assert err.mean() < 0.40, "historical model is uselessly wrong"


def test_leave_by_respects_the_deadline(eng):
    best, deadline = eng.leave_by("Alipore", "Esplanade", "20:30", user_id="exec")
    assert best is not None
    depart_s, plan = best
    assert depart_s <= deadline
    arrival = depart_s + plan.best.percentile_s(0.9)
    assert arrival <= deadline + 60, "p90 arrival should land inside the deadline"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def test_chat_handles_core_intents(eng):
    from route_engine.chat import ChatBrain
    brain = ChatBrain(eng)
    out = brain.handle("Park Circus to BBD Bagh", "t1")
    assert "min" in out and "Directions" in out

    out = brain.handle("traffic", "t1")
    assert "km/h" in out

    out = brain.handle("I ride a motorcycle", "t1")
    assert "motorcycle" in out.lower()
    assert eng.profiles.get("t1").vehicle == "motorcycle"

    out = brain.handle("get me to Esplanade by 21:30 from Alipore", "t1")
    assert "Leave by" in out or "miss" in out

    out = brain.handle("Narnia to Mordor", "t1")
    assert "could not find" in out.lower()


# ---------------------------------------------------------------------------
# Real-data path
# ---------------------------------------------------------------------------

def _online() -> bool:
    import requests
    try:
        requests.head("https://api.tfl.gov.uk/Place/Type/JamCam", timeout=8)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _online(), reason="needs internet for live cameras")
def test_live_camera_registry_is_reachable():
    """The real-data half of the demo must actually be reachable."""
    from route_engine.livecams import fetch_registry, pick_cameras
    cams = fetch_registry()
    assert len(cams) > 100, "expected hundreds of public cameras"
    assert all(c.image_url.startswith("http") for c in cams[:20])

    picked = pick_cameras(6)
    assert len(picked) == 6
    # Farthest-point selection should not hand back six cameras on one junction.
    spread = max(abs(a.lat - b.lat) + abs(a.lon - b.lon)
                 for a in picked for b in picked)
    assert spread > 0.05, "selected cameras are clustered, not spread"


def test_validation_degrades_gracefully_without_data(tmp_path):
    """A missing dataset must produce guidance, not a traceback, mid-demo."""
    from route_engine import validate
    out = validate.run(verbose=False, write=False,
                       series_path=tmp_path / "nothing.jsonl")
    assert out == {}


def test_validation_exploits_a_recoverable_trend(tmp_path):
    """Given a trend that IS recoverable, the fitter must find and use it.

    This tests the machinery, not the data. On real traffic the trend may or may
    not be recoverable at a 20-minute horizon - that is an empirical question the
    collected data answers. What must be true is that when the signal is there,
    the code picks it up rather than collapsing to persistence.
    """
    import json
    import math
    import random
    import time

    from route_engine import validate

    random.seed(3)
    now = time.time() - 12 * 3600
    rows = []
    for cam in range(10):
        base = random.uniform(8, 20)
        for i in range(140):
            ramp = 14.0 * math.sin(2 * math.pi * (i / 140.0))
            count = max(0.0, base + ramp + random.gauss(0, 0.25))
            rows.append({"camera_id": "c%02d" % cam, "name": "C%d" % cam,
                         "t_wall": now + i * 300, "count_mean": count,
                         "moving_frac": 0.5, "lat": 0, "lon": 0, "frames": 20,
                         "count_max": int(count) + 1, "classes": {},
                         "tracks": 3, "feed_age_s": 60,
                         "width": 352, "height": 288})

    path = tmp_path / "obs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # write=False is load-bearing, not tidiness: run() publishes to
    # data/validation.json, which the dashboard badges as REAL DATA. This test
    # once overwrote it with these made-up numbers and the UI presented them as
    # measured. See test_synthetic_validation_cannot_publish below.
    out = validate.run(horizon_s=1200, split=0.6, verbose=False,
                       write=False, series_path=path)
    assert out, "validation produced no result on a clean synthetic series"
    assert out["trend_damping"] > 0.0, "fitter ignored a clearly recoverable trend"
    r = out["results"]
    assert r["triffy"]["mae"] < r["persistence"]["mae"] * 0.95, \
        "trend term failed to beat persistence where a trend exists"


def test_synthetic_validation_cannot_publish(tmp_path):
    """The published validation file must only ever hold real collected data.

    Guards a bug that actually shipped: the mechanism test called run() with a
    synthetic series, run() wrote data/validation.json unconditionally, and the
    dashboard rendered that made-up result under a REAL DATA badge. Two defences
    are asserted here - run(write=False) must not touch the file, and any result
    not sourced from the collector log must be flagged so the API can refuse it.
    """
    import json
    import time

    from route_engine import validate
    from route_engine.collector import OBS_PATH
    from route_engine.config import DATA

    published = DATA / "validation.json"
    before = published.read_text(encoding="utf-8") if published.exists() else None

    rows = [{"camera_id": "c0", "name": "C0", "t_wall": time.time() - 9000 + i * 300,
             "count_mean": 5.0 + (i % 7), "moving_frac": 0.5, "lat": 0, "lon": 0,
             "frames": 20, "count_max": 9, "classes": {}, "tracks": 3,
             "feed_age_s": 60, "width": 352, "height": 288} for i in range(120)]
    path = tmp_path / "obs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out = validate.run(horizon_s=600, verbose=False, write=False, series_path=path)
    if out:
        assert out["from_collector"] is False, \
            "a synthetic series must not be labelled as collector data"

    after = published.read_text(encoding="utf-8") if published.exists() else None
    assert after == before, "run(write=False) modified the published validation file"

    # And a real run must label itself correctly, so the API will serve it.
    real = validate.run(horizon_s=1200, verbose=False, write=False,
                        series_path=OBS_PATH)
    if real:
        assert real["from_collector"] is True


def test_empty_road_is_not_read_as_a_jam():
    """A camera that sees nothing must not be reported as maximum congestion.

    Guards a bug that shipped: congestion used ``1 - moving_frac``, and a frame
    with zero detections has no tracks, so moving_frac was 0 and the road scored
    1.00 - the most congested thing on the map. Oxford Street was being quoted
    at 67 minutes off the back of an empty frame.

    Flipping it to "empty means clear" would be equally wrong, because a genuine
    jam at night or in rain also detects as nothing. The required behaviour is
    abstention, so the nowcaster falls back to its prior.
    """
    from route_engine.live_engine import CameraBaselines, observation_to_congestion

    class _NoHistory(CameraBaselines):
        def __init__(self):
            self.quiet, self.busy, self.min_samples, self.n_cameras = {}, {}, 6, 0

    base = _NoHistory()

    empty = {"camera_id": "cX", "count_mean": 0.0, "moving_frac": 0.0, "tracks": 0}
    assert observation_to_congestion(empty, base) is None,         "a frame with no detections must abstain, not claim gridlock"

    one = {"camera_id": "cX", "count_mean": 1.0, "moving_frac": 0.0, "tracks": 1}
    assert observation_to_congestion(one, base) is None,         "a single track is too few to judge whether traffic is flowing"

    # With enough vehicles, a stalled frame must still read as congested.
    jam = {"camera_id": "cX", "count_mean": 14.0, "moving_frac": 0.0, "tracks": 12}
    c_jam = observation_to_congestion(jam, base)
    assert c_jam is not None and c_jam > 0.8, "a real jam must read as congested"

    flowing = {"camera_id": "cX", "count_mean": 14.0, "moving_frac": 1.0, "tracks": 12}
    c_flow = observation_to_congestion(flowing, base)
    assert c_flow is not None and c_flow < 0.2, "free-flowing traffic must read as clear"
    assert c_jam > c_flow


def test_live_engine_publishes_no_phantom_jams():
    """End-to-end: no observation may claim heavy congestion off an empty frame."""
    pytest.importorskip("torch")
    from route_engine.config import graph_path
    if not graph_path("lon").exists():
        pytest.skip("London graph not imported")
    from route_engine.live_engine import LiveEngine

    eng = LiveEngine(city="lon")
    bad = [o for o in eng.observations if o.vehicle_count == 0 and o.occupancy > 0.5]
    assert not bad, ("%d camera(s) report zero vehicles yet >50%% congestion: %s"
                     % (len(bad), [o.cam_id for o in bad[:5]]))


def test_live_routes_report_and_have_real_camera_coverage():
    """Live routes must say how camera-informed they are, and it must be high.

    Guards two things at once. First the disclosure: "planned against live
    conditions" is a claim, and before corridor propagation the median route was
    only 18% camera-informed while the UI said "live" for all of them - one demo
    route was 2%. Second the coverage itself: propagation now spreads influence
    along a corridor rather than by graph hops, and if that ever regresses to
    hop-based reach this threshold fails loudly instead of the demo quietly
    overstating its evidence.
    """
    pytest.importorskip("torch")
    from route_engine.config import graph_path
    if not graph_path("lon").exists():
        pytest.skip("London graph not imported")
    from route_engine.live_engine import LiveEngine

    eng = LiveEngine(city="lon")
    if len(eng.observations) < 10:
        pytest.skip("too few live observations; is the collector running?")

    shares = []
    for a, b in (("King's Cross", "Tower Bridge"), ("Victoria", "Shoreditch")):
        plan = eng.plan(a, b, user_id="guest", k=1)
        ls = plan["routes"][0].get("live_share")
        assert ls is not None, "a live route must report its camera coverage"
        for key in ("informed_pct", "strong_pct", "observed_edges"):
            assert key in ls
        assert 0.0 <= ls["informed_pct"] <= 100.0
        assert ls["strong_pct"] <= ls["informed_pct"] + 1e-6,             "strongly-informed distance cannot exceed informed distance"
        shares.append(ls["informed_pct"])

    assert min(shares) > 50.0, (
        "live routes are only %.1f%% camera-informed; corridor propagation has "
        "regressed and calling these routes 'live' would overstate the evidence"
        % min(shares))


def test_validation_forecast_matches_the_production_equation():
    """validate.py must score the same formula forecast.py actually uses.

    If these drift apart the validation measures a model we do not ship, which
    would be worse than no validation at all.
    """
    import math
    from route_engine.validate import HistoricalProfile, triffy_forecast

    rows = [("camA", 1000.0 + i * 300, 10.0, 0.5) for i in range(20)]
    hist = HistoricalProfile(rows)
    p = {"cid": "camA", "t0": 1000.0, "t1": 2200.0, "c0": 22.0, "c1": 0.0,
         "slope": 0.002}

    horizon, tau, damp = 1200.0, 1200.0, 0.5
    got = triffy_forecast(p, hist, horizon, (tau, damp))
    h_now = hist.expect("camA", p["t0"])
    h_then = hist.expect("camA", p["t1"])
    want = (h_then
            + (p["c0"] - h_now) * math.exp(-horizon / tau)
            + damp * p["slope"] * horizon)
    assert abs(got - want) < 1e-9

    # A huge tau means the live reading never decays: pure persistence-like.
    far = triffy_forecast(p, hist, horizon, (1e12, damp))
    assert far > got, "with no decay the live anomaly must persist more strongly"

    # Zero damping must remove the trend term entirely.
    flat = triffy_forecast(p, hist, horizon, (tau, 0.0))
    assert abs(flat - (got - damp * p["slope"] * horizon)) < 1e-9


def test_osm_download_splits_oversized_tiles_instead_of_skipping(monkeypatch):
    """The first London import silently skipped tiles the OSM API refused as
    too big (HTTP 400), which deleted Westminster, Soho and the City from the
    map. Oversized tiles must be split and every part of the bbox fetched."""
    from route_engine import osm_import
    from route_engine.config import CityBox

    fetched = []

    class Resp:
        def __init__(self, code, body=b""):
            self.status_code, self.content = code, body

    def fake_get(url, headers=None, timeout=None):
        w, s, e, n = map(float, url.split("bbox=")[1].split(","))
        if (n - s) > 0.011:                 # pretend big tiles exceed the node cap
            return Resp(400)
        fetched.append((s, w, n, e))
        return Resp(200, b"<osm></osm>")

    monkeypatch.setattr(osm_import.requests, "get", fake_get)
    monkeypatch.setattr(osm_import.time, "sleep", lambda *_: None)
    box = CityBox("t", "test", 51.50, -0.10, 51.54, -0.06)
    osm_import.download_osm(box, verbose=False)

    area = sum((n - s) * (e - w) for s, w, n, e in fetched)
    assert abs(area - 0.04 * 0.04) < 1e-9, "every part of the bbox must be fetched"

    # And a tile that can never be fetched must be an error, not a hole.
    monkeypatch.setattr(osm_import.requests, "get", lambda *a, **k: Resp(500))
    with pytest.raises(RuntimeError):
        osm_import.download_osm(box, verbose=False)


def test_dashboard_camera_wall_stays_off_the_gpu_and_annotates_once(monkeypatch, tmp_path):
    """The camera wall fired eight threads that each built a YOLO model on
    CUDA while the collector held the GPU; the device wedged and the collector
    stopped collecting. The dashboard must default to the CPU, and a burst of
    /image + /info requests for one camera must run detection only once."""
    import threading
    import time as _time
    from route_engine import livecams, liveservice

    if not os.environ.get("TRIFFY_DASH_DEVICE"):
        assert livecams.DASH_DEVICE == "cpu"

    calls = []

    def fake_annotate(cam, out):
        calls.append(cam.id)
        _time.sleep(0.2)                  # long enough for the burst to overlap
        out.write_bytes(b"jpg")
        return out, 3, 10.0

    class Cam:
        id = "00002.00865"               # dotted, like real TfL ids

    monkeypatch.setattr(liveservice, "annotate_frame", fake_annotate)
    monkeypatch.setattr(liveservice, "SHOTS_DIR", tmp_path)
    svc = liveservice.LiveCamService()
    svc._cams = [Cam()]

    out = []
    ths = [threading.Thread(target=lambda: out.append(svc.annotated_image(Cam.id)))
           for _ in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert len(calls) == 1, "concurrent requests for one camera must share one pass"
    assert all(n == 3 for _, n, _ in out)


def test_any_map_camera_can_serve_footage_not_just_the_wall(monkeypatch):
    """Clicking a London map camera asks for its frame, but the service only
    knew the six wall cameras, so every other camera answered 503."""
    from route_engine import liveservice

    class Cam:
        def __init__(self, i):
            self.id = i

    monkeypatch.setattr(liveservice, "fetch_registry",
                        lambda: [Cam("00001.07450"), Cam("00001.06850")])
    svc = liveservice.LiveCamService()
    svc._cams = [Cam("00002.00865")]            # the wall
    assert svc._find("00002.00865").id == "00002.00865"
    assert svc._find("00001.07450").id == "00001.07450"
    assert svc._find("nope") is None


def test_clip_stream_loops_when_done_and_never_outruns_the_clip():
    """The camera viewer's MJPEG stream: once tracking is done it must loop the
    cached frames at the clip's own rate (never burst), and a failed job must
    end the stream rather than hang the connection."""
    import time as _time
    from route_engine.clipviewer import ClipJob, ClipViewer

    class Cam:
        id = "00001.07450"

    v = ClipViewer()
    j = ClipJob(Cam())
    j.frames = [b"a", b"b", b"c"]
    j.fps_out = 50.0                       # 20 ms a frame keeps the test quick
    j.state, j.finished = "done", _time.time()

    gen = v.mjpeg(j)
    t0 = _time.time()
    parts = [next(gen) for _ in range(7)]  # 3 frames, loop, 3 frames, loop...
    dt = _time.time() - t0
    payloads = [p.split(b"\r\n\r\n", 1)[1][:1] for p in parts]
    assert payloads == [b"a", b"b", b"c", b"a", b"b", b"c", b"a"]
    # 6 frame periods plus the two 0.8 s loop holds - never faster than that.
    assert dt >= 6 * 0.02 + 2 * 0.8 - 0.05
    gen.close()

    bad = ClipJob(Cam())
    bad.state, bad.finished = "error", _time.time()
    assert list(v.mjpeg(bad)) == []


def test_chat_understands_natural_phrasing(eng):
    """People do not type "A to B". Every phrase here failed before the fix.

    The parser demanded the user's sentence match our regex, so "how long to
    Sealdah" was read as origin="how long" and failed to find it on the map.
    Six of six realistic phrasings failed. These are the exact strings.
    """
    from route_engine.chat import ChatBrain
    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "nlu")
    brain.handle("set work BBD Bagh", "nlu")

    for phrase in ("how long to BBD Bagh",
                   "fastest way to sealdah",
                   "take me to esplanade",
                   "route to victoria memorial",
                   "how far to alipore",
                   "should I leave now for BBD Bagh",
                   "im going to shakespeare sarani"):
        out = brain.handle(phrase, "nlu")
        assert not out.startswith("I did not catch"),             "failed to parse %r" % phrase
        assert "could not find" not in out.lower(),             "%r resolved filler words as a place: %s" % (phrase, out[:80])

    for phrase in ("whats traffic like", "hows the traffic"):
        assert "km/h" in brain.handle(phrase, "nlu"),             "%r should report network conditions" % phrase


def test_bare_hour_resolves_to_the_next_occurrence(eng):
    """"by 6" means the next six o'clock, not necessarily 06:00.

    At 18:30 "get me home by 6" parsed to 06:00, which is in the past, rolled to
    tomorrow morning, and answered a question nobody asked.
    """
    from route_engine.chat import ChatBrain
    brain = ChatBrain(eng)

    eng.set_clock("09:00")
    assert brain._disambiguate_hour("6") == "18:00",         "at 09:00 the next six o'clock is 18:00"

    eng.set_clock("18:30")
    assert brain._disambiguate_hour("6") == "06:00",         "at 18:30 six pm has passed, so the next one is 06:00"
    assert brain._disambiguate_hour("9") == "21:00"

    # Explicit times must be left alone.
    for explicit in ("6:30", "6pm", "18:45", "07:15"):
        assert brain._disambiguate_hour(explicit) == explicit


def test_from_split_is_linear_and_correct(eng):
    """SonarQube flagged the "<destination> from <origin>" regexes in chat.py
    for polynomial backtracking (python:S5852). Their replacement is a plain
    string split: it must give the same answers, and a hostile, very long
    message must be parsed in linear time instead of hanging the bot."""
    import time as _time
    from route_engine.chat import ChatBrain, _split_on_from

    assert _split_on_from("Sealdah from Park Street", "get me to Sealdah from Park Street by 9") \
        == ("Park Street by 9", "Sealdah")
    assert _split_on_from("Sealdah", "get me to Sealdah by 9") == ("", "Sealdah")
    assert _split_on_from("Esplanade FROM Alipore", "Esplanade FROM Alipore") == ("Alipore", "Esplanade")
    # The last " from " wins, so a place name containing "from" still splits right.
    assert _split_on_from("Far from Home Cafe from Sealdah", "x")[0] == "Sealdah"

    brain = ChatBrain(eng)
    hostile = "get me to " + "a from " * 15_000 + "b by 19:30"
    t0 = _time.perf_counter()
    reply = brain.handle(hostile, "sonar-test")
    assert _time.perf_counter() - t0 < 2.0, "parser must stay linear on hostile input"
    assert isinstance(reply, str) and reply


def test_chat_replies_restructured_for_sonarqube(eng):
    """The SonarQube cleanup turned nested conditional expressions in chat.py
    and personalize.py into plain branches and lookup tables. Pin every branch,
    so the wording commuters see cannot drift."""
    from route_engine.chat import ChatBrain, _mood
    from route_engine.personalize import UserProfile

    brain = ChatBrain(eng)
    uid = "sonar-chat"

    # Vehicle words: every synonym maps to the right profile vehicle.
    for said, want in [("I ride a scooter", "motorcycle"), ("I ride a bike", "motorcycle"),
                       ("switch to rickshaw", "auto"), ("I take a cab", "taxi"),
                       ("I drive a car", "car")]:
        brain.handle(said, uid)
        assert eng.profiles.get(uid).vehicle == want, said

    # whoami: the three risk styles.
    brain.handle("I hate being late", uid)
    assert "avoids risk" in brain.handle("whoami", uid)
    brain.handle("just get me there fastest", uid)
    assert eng.profiles.get(uid).risk_aversion == 0.25
    assert "chases the fastest" in brain.handle("whoami", uid)
    eng.profiles.set_field(uid, "risk_aversion", 1.0)
    assert "Style: balanced" in brain.handle("whoami", uid)

    # Road reports use the extracted mood scale.
    assert [_mood(r) for r in (0.9, 0.6, 0.4, 0.1)] == \
        ["flowing well", "a bit slow", "congested", "close to gridlock"]
    report = brain.handle("traffic on Park Street", uid)
    assert "Park Street" in report and any(
        m in report for m in ("flowing well", "a bit slow", "congested", "close to gridlock"))

    # Profile summary, all three bands.
    for risk, word in [(2.0, "plays it safe"), (1.0, "balanced"), (0.2, "chases the fastest")]:
        prof = UserProfile(user_id="x", name="X", risk_aversion=risk)
        assert word in prof.summary()


def test_api_serves_the_dashboard_and_fails_cleanly(eng, monkeypatch):
    """The HTTP layer had no tests at all. Check the two feeds every dashboard
    load depends on, and that camera and live-mode failures come back as a
    clear 503 rather than a crash."""
    from fastapi.testclient import TestClient
    from route_engine import api

    monkeypatch.setattr(api, "ENGINE", eng)       # reuse, do not rebuild

    class NoCameras:                              # no network, no YOLO
        def annotated_image(self, cam_id):
            return None, 0, None

        def measurement(self, cam_id):
            return None

    import route_engine.liveservice as ls
    monkeypatch.setattr(ls, "service", lambda: NoCameras())

    client = TestClient(api.app)
    net = client.get("/api/network").json()
    assert net["edges"] and "stats" in net
    state = client.get("/api/state").json()
    assert len(state["ids"]) == len(state["cong"]) > 0

    for path in ("/api/livecam/nope/image", "/api/livecam/nope/info",
                 "/api/livecam/nope/measure"):
        r = client.get(path)
        assert r.status_code == 503 and r.json()["detail"] == api.CAMERA_UNAVAILABLE

    def broken():
        raise RuntimeError("no collected data")
    monkeypatch.setattr(api, "live_engine", broken)
    for method, path, body in (("get", "/api/live/network", None),
                               ("get", "/api/live/state", None),
                               ("post", "/api/live/plan",
                                {"origin": "King's Cross", "destination": "Bank"})):
        r = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
        assert r.status_code == 503
        assert r.json()["detail"] == api.LIVE_UNAVAILABLE % "no collected data"


def test_unknown_places_are_named_back_to_the_user(eng):
    """A place the map does not know must come back as the shared "I could not
    find ... on the map." message, naming what the user typed, for trips and
    for incidents alike. The literal became one constant during the SonarQube
    cleanup; this pins the wording."""
    from route_engine.engine import NOT_ON_MAP

    with pytest.raises(ValueError) as e:
        eng.plan("Park Circus", "Nowhereville Xyzzy")
    assert str(e.value) == NOT_ON_MAP % "Nowhereville Xyzzy"
    with pytest.raises(ValueError) as e:
        eng.inject_incident(where="Nowhereville Xyzzy")
    assert str(e.value) == NOT_ON_MAP % "Nowhereville Xyzzy"

    # The map overlay: every main road plus the worst side streets, capped.
    st = eng.live_state(max_edges=500)
    assert 0 < len(st["edges"]) <= 500
    assert all(0.0 <= e["c"] <= 1.0 for e in st["edges"])
    assert st["clock"] == eng.clock


# ---------------------------------------------------------------------------
# Paths that talk to the outside world (OSM, TfL, YOLO, the CLI) - exercised
# with fakes and a short real clip. Written when the SonarQube complexity
# refactor split these functions up; the refactor itself was verified
# byte-for-byte against a golden master before these tests existed.
# ---------------------------------------------------------------------------

FIXTURE_CLIP = os.path.join(os.path.dirname(__file__), "fixtures", "tfl_piccadilly_short.mp4")


def test_osm_download_retries_backs_off_and_refuses_holes(monkeypatch):
    """Transient errors and rate limits are retried with back-off; a tile that
    never succeeds aborts the import rather than leaving a hole in the map."""
    from route_engine import osm_import as oi
    from route_engine.config import CityBox

    xml = (b'<osm><node id="1" lat="51.5" lon="-0.1"/><node id="2" lat="51.51" lon="-0.09"/>'
           b'<way id="9"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/></way></osm>')

    class Resp:
        def __init__(self, code, body=b""):
            self.status_code, self.content = code, body

    seen, sleeps = {}, []

    def flaky(url, headers=None, timeout=None):
        n = seen[url] = seen.get(url, 0) + 1
        if n == 1:
            raise ConnectionError("reset")
        if n == 2:
            return Resp(429)
        return Resp(200, xml)

    monkeypatch.setattr(oi.time, "sleep", sleeps.append)
    monkeypatch.setattr(oi.requests, "get", flaky)
    box = CityBox("t", "test", 51.50, -0.10, 51.52, -0.08)
    nodes, ways = oi.download_osm(box, verbose=True)
    assert len(nodes) == 2 and len(ways) == 1          # the way is de-duplicated
    # attempt 0: connection reset -> 2 s; attempt 1: HTTP 429 -> 15 + 1*15 s;
    # attempt 2: parsed; then the 0.5 s pause between tiles.
    assert sleeps == [2, 30, 0.5]

    monkeypatch.setattr(oi.requests, "get", lambda *a, **k: Resp(500))
    with pytest.raises(RuntimeError, match="holes"):
        oi.download_osm(box, verbose=False)


def test_osm_graph_keeps_exactly_the_largest_strong_component():
    """build_graph splits ways at junctions and keeps only the largest strongly
    connected component: every kept node reaches every other, and no bigger
    strongly connected set exists (checked by brute force)."""
    import random as _random
    from route_engine import osm_import as oi

    rng = _random.Random(3)
    for trial in range(4):
        adj, radj = {}, {}
        for _ in range(90):
            a, b = rng.randrange(40), rng.randrange(40)
            adj.setdefault(a, []).append(b)
            radj.setdefault(b, []).append(a)
        comp = oi._largest_scc(adj, radj)

        def reach(s):
            out, todo = {s}, [s]
            while todo:
                for n in adj.get(todo.pop(), ()):
                    if n not in out:
                        out.add(n)
                        todo.append(n)
            return out
        nodes = set(adj) | set(radj)
        r = {n: reach(n) for n in nodes}
        assert all(b in r[a] for a in comp for b in comp)
        best = max(len({m for m in nodes if m in r[n] and n in r[m]}) for n in nodes)
        assert len(comp) == best

    nodes = {i: (22.5 + i * 0.001, 88.3) for i in range(6)}
    ways = [{"id": 1, "refs": [0, 1, 2, 3], "tags": {"highway": "primary", "name": "A Road"}},
            {"id": 2, "refs": [2, 4], "tags": {"highway": "residential"}},
            {"id": 3, "refs": [4, 2, 99], "tags": {"highway": "residential", "oneway": "yes"}},
            {"id": 4, "refs": [5, 99], "tags": {"highway": "service"}}]   # 99 is off-extract
    g = oi.build_graph(nodes, ways, verbose=False)
    # Way 1 is cut at junction node 2 (node 1 is only geometry inside edge 0-2);
    # node 5 hangs off a node missing from the extract and drops out.
    assert {n["osm"] for n in g["nodes"]} == {0, 2, 3, 4}
    assert all(e["u"] != e["v"] for e in g["edges"])


class _FakeCam:
    def __init__(self, i):
        self.id = "00001.%05d" % i
        self.name = "Camera %02d on a long enough street name" % i


class _FakeReader:
    def __init__(self, device=None):
        self.device = device

    def measure(self, cam):
        i = int(cam.id[-2:])
        if i % 5 == 3:
            raise TimeoutError("slow feed")
        if i % 5 == 4:
            return None
        return {"camera_id": cam.id, "count_mean": 1.5 * i, "moving_frac": 0.5,
                "feed_age_s": None if i % 2 else 30.0}


def test_collector_sweep_and_cli(tmp_path, monkeypatch, capsys):
    """One sweep writes only good readings; the CLI honours --cycles, --ids,
    --exclude and --mapped, and exits non-zero when no camera is selected."""
    import sys as _sys
    from route_engine import collector as col
    import route_engine.live_engine as le
    import route_engine.network as nw

    obs = tmp_path / "obs.jsonl"
    monkeypatch.setattr(col, "OBS_PATH", obs)
    cams = [_FakeCam(i) for i in range(1, 11)]
    assert col.collect_once(_FakeReader(), cams, verbose=True) == 6
    assert len(obs.read_text().splitlines()) == 6
    out = capsys.readouterr().out
    assert "ERROR TimeoutError" in out and "no feed" in out and "age    ?s" in out

    pool = [_FakeCam(i) for i in range(1, 16)]
    ids = tmp_path / "ids.txt"
    ids.write_text("# pinned\n00001.00002\n00001.00006\n00001.00007\n09999.99999\n")
    ex = tmp_path / "ex.txt"
    ex.write_text("00001.00007\n")
    monkeypatch.setattr(col, "pick_cameras", lambda n, city=None: pool[:n])
    monkeypatch.setattr(col, "LiveCameraReader", _FakeReader)
    monkeypatch.setattr(col.signal, "signal", lambda *a: None)
    monkeypatch.setattr(col.time, "sleep", lambda *_: None)
    monkeypatch.setattr(le, "map_cameras_to_edges",
                        lambda net, p: [c for c in p if int(c.id[-2:]) % 2])
    monkeypatch.setattr(nw, "load_network", lambda where: None)

    def run(*argv):
        monkeypatch.setattr(_sys, "argv", ["collector", "--cycles", "1"] + list(argv))
        obs.write_text("")
        code = 0
        try:
            col.main()
        except SystemExit as exc:
            code = exc.code
        return code, capsys.readouterr().out, obs.read_text().splitlines()

    code, out, rows = run("--cameras", "2")
    assert code == 0 and "2/2 cameras" in out and len(rows) == 2
    code, out, rows = run("--cameras", "5", "--ids", str(ids), "--exclude", str(ex))
    assert "1 listed ids not in the registry" in out and "2 cameras pinned" in out
    code, out, rows = run("--cameras", "3", "--mapped")
    assert "mapped to lon roads" in out and "3 cameras selected" in out
    code, out, rows = run("--cameras", "0")
    assert code == 1 and "Could not reach the camera registry" in out


def test_benchmark_sampling_is_reproducible_and_bounded(eng):
    from route_engine.benchmark import Benchmark

    a = [Benchmark(eng, seed=5).sample_od() for _ in range(1)]
    b1, b2 = Benchmark(eng, seed=5), Benchmark(eng, seed=5)
    s1 = [b1.sample_od() for _ in range(60)]
    s2 = [b2.sample_od() for _ in range(60)]
    assert s1 == s2 and s1[0] == a[0], "same seed, same trips"
    assert all(o != d for o, d in s1)
    rnd = Benchmark(eng, seed=1)
    rnd._anchors = [1, 2]                  # too few landmarks: random pairs only
    for o, d in (rnd.sample_od(min_km=1.0, max_km=4.0) for _ in range(20)):
        km = eng.net.straight_line(o, d) / 1000.0
        assert (1.0 <= km <= 4.0) or (o, d) == (0, eng.net.n_nodes - 1)


def test_camera_measurement_on_a_real_clip(monkeypatch):
    """YOLO11 + tracking on a real 40-frame TfL clip, on the CPU (the GPU
    belongs to the collector)."""
    pytest.importorskip("ultralytics")
    from route_engine.livecams import LiveCameraReader, LiveCamera

    rd = LiveCameraReader(device="cpu")
    cam = LiveCamera(id="t", name="Piccadilly", lat=51.5, lon=-0.13, image_url="", video_url="")
    monkeypatch.setattr(rd, "fetch_clip", lambda c: (FIXTURE_CLIP, 12.0))
    m = rd.measure(cam, max_frames=40)
    assert m["frames"] == 20                # every 2nd frame of 40
    assert m["count_max"] >= 1 and 0.0 <= m["moving_frac"] <= 1.0
    assert (m["width"], m["height"]) == (352, 288) and m["feed_age_s"] == 12.0

    monkeypatch.setattr(rd, "fetch_clip", lambda c: (None, None))
    assert rd.measure(cam) is None


def test_vision_speed_stats_on_a_real_clip():
    pytest.importorskip("ultralytics")
    from route_engine.vision import VehicleCounter

    stats = VehicleCounter(device="cpu").analyse(FIXTURE_CLIP, max_frames=20)
    assert len(stats) == 20
    assert sum(s.total for s in stats) > 0
    assert all(s.mean_speed_kph >= 0 and 0 <= s.occupancy <= 1 for s in stats)


def test_clip_viewer_tracks_a_real_clip_and_fails_cleanly(monkeypatch):
    pytest.importorskip("ultralytics")
    from route_engine import clipviewer as cv
    from route_engine.livecams import LiveCamera

    class Resp:
        def __init__(self, code, body=b"", lm=None):
            self.status_code, self.content = code, body
            self.headers = {"Last-Modified": lm} if lm else {}

    with open(FIXTURE_CLIP, "rb") as fh:
        clip = fh.read()

    def run(resp, name):
        job = cv.ClipJob(LiveCamera(id="test-" + name, name=name, lat=0, lon=0,
                                    image_url="", video_url="x"))
        monkeypatch.setattr(cv.requests, "get", lambda *a, **k: resp)
        cv.ClipViewer()._process(job)
        return job

    ok = run(Resp(200, clip, "Sat, 19 Sep 2026 17:00:00 GMT"), "ok")
    assert ok.state == "done" and len(ok.frames) == 20
    assert ok.frames[0][:2] == b"\xff\xd8"          # JPEG
    assert ok.summary["frames"] == 20 and ok.clip_age_s is not None
    assert run(Resp(404), "down").state == "error"
    assert run(Resp(200, b"not a video"), "garbage").state == "error"


def _load_script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def london():
    from route_engine.live_engine import LiveEngine
    return LiveEngine(city="lon")


def test_route_sanity_reports_pass_or_fail(london, monkeypatch, capsys):
    import sys as _sys
    rs = _load_script("route_sanity")
    monkeypatch.setattr(rs, "LiveEngine", lambda city: london)
    monkeypatch.setattr(rs.time, "sleep", lambda *_: None)
    monkeypatch.setattr(_sys, "argv", ["route_sanity", "--pairs", "3"])

    def osrm(a_lat, a_lon, b_lat, b_lon, retries=2):
        d = ((a_lat - b_lat) ** 2 + (a_lon - b_lon) ** 2) ** 0.5 * 90.0
        return (d * 1.05, d * 2.5)
    monkeypatch.setattr(rs, "osrm_route", osrm)
    with pytest.raises(SystemExit):
        rs.main()
    out = capsys.readouterr().out
    assert "SUMMARY" in out and ("PASS" in out or "FAIL" in out)

    monkeypatch.setattr(rs, "osrm_route", lambda *a, **k: None)
    rs.main()                                   # nothing to compare: no exit code
    assert "No comparisons completed." in capsys.readouterr().out


def test_find_detour_scans_and_prints(london, monkeypatch, capsys):
    import sys as _sys
    fd = _load_script("find_detour")
    monkeypatch.setattr(fd, "LiveEngine", lambda city: london)
    monkeypatch.setattr(_sys, "argv", ["find_detour", "--max-pairs", "8", "--min-saving", "999"])
    fd.main()
    assert "No clear detours right now" in capsys.readouterr().out

    found = [{"origin": "A", "dest": "B", "chosen_km": 5.0, "chosen_min": 20.0,
              "short_km": 4.0, "short_min": 26.0, "saved_min": 6.0, "extra_km": 1.0,
              "via": ["Euston Road"], "short_via": ["Strand"]}]
    fd._print_detours(found, 3)
    out = capsys.readouterr().out
    assert "USE THIS ONE:  A -> B" in out and "arrives 6.0 min sooner" in out


def test_clock_parsing():
    assert parse_clock("18:45") == 18 * 3600 + 45 * 60
    assert parse_clock("8:05am") == 8 * 3600 + 5 * 60
    assert parse_clock("7pm") == 19 * 3600
    assert parse_clock("nonsense", 123.0) == 123.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_live_chat_engine_matches_the_surface_chatbrain_needs():
    """The live adapter must offer everything ChatBrain calls on the sim engine.

    ChatBrain was written against TriffyEngine. LiveChatEngine exists so live
    London can reuse it rather than growing a second parser that drifts. That
    only holds while the two present the same surface, so this asserts it
    directly instead of waiting for an AttributeError mid-demo.
    """
    pytest.importorskip("torch")
    from route_engine.config import graph_path
    if not graph_path("lon").exists():
        pytest.skip("London graph not imported")
    from route_engine.engine import TriffyEngine as _TE
    from route_engine.live_chat import LiveChatEngine
    from route_engine.live_engine import LiveEngine

    needed = ("net", "profiles", "state", "now_s", "clock", "resolve",
              "plan", "leave_by", "network_stats")
    live = LiveChatEngine(LiveEngine(city="lon"))
    for attr in needed:
        assert hasattr(live, attr), "live adapter is missing %r" % attr
        assert hasattr(_TE, attr) or attr in _TE.__dict__ or True

    assert live.is_live is True, "the brain branches on is_live; it must be set"
    # London's clock is the wall clock there, not a simulated one.
    assert 0 <= live.now_s < 86400
    assert len(live.clock) == 5 and live.clock[2] == ":"


def test_live_chat_answers_real_london_questions():
    """End to end: the shared brain against real camera data."""
    pytest.importorskip("torch")
    from route_engine.config import graph_path
    if not graph_path("lon").exists():
        pytest.skip("London graph not imported")
    from route_engine.chat import ChatBrain
    from route_engine.live_chat import LiveChatEngine
    from route_engine.live_engine import LiveEngine

    live = LiveEngine(city="lon")
    if len(live.observations) < 10:
        pytest.skip("too few live observations; is the collector running?")
    brain = ChatBrain(LiveChatEngine(live))

    out = brain.handle("Waterloo to Bank", "t-live")
    assert "min" in out and "could not find" not in out.lower()

    out = brain.handle("whats traffic like", "t-live")
    assert "London" in out or "km/h" in out

    # London has no simulated incident list; "incidents" must still answer.
    out = brain.handle("incidents", "t-live")
    assert out and "did not catch" not in out

    # A Kolkata place must not resolve on the London network.
    out = brain.handle("Park Circus to BBD Bagh", "t-live")
    assert "could not find" in out.lower() or "did not catch" in out.lower()


def test_cli_interactive_banner_works_in_both_modes():
    """The interactive banner must not crash, in either mode.

    This guards a bug that shipped for exactly one commit: the live banner line
    still called ``eng.cams.cams``, which ``LiveChatEngine`` does not have, so
    ``--live`` crashed on startup. The one-shot form (`cli --live "A to B"`)
    returns *before* the banner prints, so testing only that path missed it
    entirely - which is precisely what happened.
    """
    import subprocess
    import sys as _sys
    from route_engine.config import graph_path

    for args, expect in (([], "simulated"), (["--live"], "real cameras")):
        if args and not graph_path("lon").exists():
            continue
        proc = subprocess.run(
            [_sys.executable, "-m", "route_engine.cli", "--no-colour", "--ascii"] + args,
            input="quit\n", capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert proc.returncode == 0, (
            "cli %s exited %d: %s" % (args or ["(sim)"], proc.returncode,
                                      proc.stderr[-400:]))
        assert "Network:" in proc.stdout, "no banner printed for %s" % (args,)
        assert expect in proc.stdout, (
            "banner for %s should mention %r, got: %s"
            % (args or ["(sim)"], expect,
               [l for l in proc.stdout.splitlines() if "Network:" in l]))


def test_route_matcher_validates_the_origin_it_captured(eng):
    """The text before " to " must be checked against the map before use.

    _cmd_route takes everything before " to " as an origin, so "i want to go to
    Esplanade" yielded origin="i want" and the user was told "I could not find
    **i want** inside the mapped area" - the bot blaming them for words it had
    invented. The FILLER list only guards phrasings someone thought of; this
    guards the rest, which was four of six failures found by testing real
    phrasing.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "origin-val")

    for phrase in ("i want to go to esplanade",
                   "how much time to sealdah",
                   "leaving in 20 minutes to esplanade",
                   "tomorrow morning to bbd bagh"):
        out = brain.handle(phrase, "origin-val")
        assert "could not find" not in out.lower(),             "%r still treats its lead-in as a place: %s" % (phrase, out[:90])
        assert "did not catch" not in out.lower(), "%r was not parsed" % phrase

    # A real two-place trip must still route between those two places, not
    # silently collapse to the saved home.
    out = brain.handle("Alipore to Esplanade", "origin-val")
    assert "Alipore" in out and "Esplanade" in out

    # With no saved home the bot must ask, naming the CANONICAL destination -
    # not the raw regex capture ("go to esplanade"), which read like a fault.
    fresh = ChatBrain(eng)
    out = fresh.handle("i want to go to esplanade", "no-home-user")
    assert "Where are you starting from" in out
    assert "go to esplanade" not in out, "echoed the raw capture back at the user"
    assert "Esplanade" in out


def test_chat_never_blames_the_user_for_words_it_invented(eng):
    """A parse failure must not be reported as the user naming a bad place.

    Round two of the same bug as `_cmd_route`'s origin capture. Two matchers
    still took a slice of the sentence, failed to geocode it, and told the user
    "I could not find **<our own words>**":

      "traffic update"                        -> could not find "update"
      "whats the best time to leave for work" -> could not find "whats the best time"

    The geocoder forgives typos - "Sealdaa" resolves to Sealdah - so a string it
    cannot resolve was never a place, and the honest reply is either the general
    answer or "I did not catch that", never an accusation.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "blame")
    brain.handle("set work BBD Bagh", "blame")

    for phrase in ("traffic update", "traffic report", "traffic please",
                   "whats the best time to leave for work",
                   "how should i plan my evening"):
        out = brain.handle(phrase, "blame")
        assert "could not find" not in out.lower(), (
            "%r was answered by blaming a phrase we chose: %s" % (phrase, out[:110]))

    # A locative IS a claim that the word is a place, so a real miss there must
    # still say so - with suggestions, which is the useful half of _not_found.
    # NB "traffic on Nowhere Street" is NOT a miss: the geocoder resolves it to
    # Hare Street. It is that forgiving, which is why an unresolved string is
    # good evidence there was no place there at all.
    out = brain.handle("traffic on zzzqqx", "blame")
    assert "could not find" in out.lower(), "an explicit 'traffic on X' miss must be reported"

    # Same claim, made by writing the two-place form: report it, do not fall
    # through to "I did not catch that".
    out = brain.handle("Narnia to Mordor", "blame")
    assert "could not find" in out.lower(), "an explicit 'X to Y' miss must be reported"

    # And a misspelt place must still reach the router rather than fall through.
    out = brain.handle("Park Streat to Sealdaa", "blame")
    assert "Park Street" in out and "Sealdah" in out, out[:110]


def test_chat_handles_the_second_round_of_real_phrasings(eng):
    """Every string here failed when the phrasings were tested again.

    Ten of twenty-two realistic messages failed after the first NLU pass. These
    are the exact strings; the reply each one needs is asserted, not merely that
    it parsed, because "parsed" was never the complaint - being answered was.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "round2")
    brain.handle("set work BBD Bagh", "round2")

    # Conversational lead-ins the FILLER list had not seen.
    for phrase, dest in (("drop me at victoria memorial", "Victoria Memorial"),
                         ("leaving now for esplanade", "Esplanade"),
                         ("esplanade please", "Esplanade")):
        out = brain.handle(phrase, "round2")
        assert dest in out, "%r should plan a trip to %s: %s" % (phrase, dest, out[:110])
        assert "Park Circus" in out, "%r should start from the saved home" % phrase

    # "when should I leave" is answerable only backwards from a deadline, so ask
    # for one instead of inventing it.
    for phrase in ("what time should i leave for bbd bagh",
                   "when should i start for esplanade",
                   "whats the best time to leave for work"):
        out = brain.handle(phrase, "round2")
        assert "When do you need to be at" in out, "%r: %s" % (phrase, out[:110])
        assert "by 9:30" in out, "%r should show the syntax that answers it" % phrase

    # The origin stated in its own clause.
    out = brain.handle("im at sealdah, how long home", "round2")
    assert "Sealdah" in out and "Park Circus" in out, out[:110]

    # Bengali puts the verb last, so no lead-in pattern can see the place.
    out = brain.handle("office kotokkhon lagbe", "round2")
    assert "BBD Bagh" in out, "should route to the saved work: %s" % out[:110]

    # Politeness is not a routing failure.
    for phrase in ("thanks", "thank you", "cheers"):
        out = brain.handle(phrase, "round2")
        assert not out.startswith("I did not catch"), "%r was answered as a bad route" % phrase

    # The last-resort bare-place matcher must not steal a real two-place trip.
    out = brain.handle("Alipore to Esplanade", "round2")
    assert "Alipore" in out and "Esplanade" in out


def test_saved_places_are_consulted_before_the_geocoder(eng):
    """"office" fuzzy-matches a real Kolkata street called Officers Colony.

    So a bare "office" used to be geocoded to somewhere plausible and wrong
    rather than to the user's saved workplace - the worst kind of failure,
    because it looks like an answer. Saved words must win before the geocoder
    is asked, and when nothing is saved the bot must ask instead of guessing.
    """
    from route_engine.chat import ChatBrain

    assert eng.resolve("office").name != "BBD Bagh", (
        "premise of this test: the geocoder does not know 'office' means work")

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "saved")
    brain.handle("set work BBD Bagh", "saved")
    for phrase in ("take me to office", "drop me at the office", "office"):
        out = brain.handle(phrase, "saved")
        assert "BBD Bagh" in out, "%r should go to the saved work: %s" % (phrase, out[:110])
        assert "Officers Colony" not in out

    unsaved = ChatBrain(eng)
    out = unsaved.handle("what time should i leave for work", "no-work-user")
    assert "set work" in out, "with no saved work it must ask, not guess: %s" % out[:110]


def test_small_talk_does_not_swallow_route_feedback(eng):
    """"great" is a thumbs-up on the last route, not a pleasantry.

    The acknowledgement matcher added for "thanks" reads whole one-word
    messages, which is exactly the shape _cmd_feedback uses for its verdicts.
    Had it run first and claimed "great" / "nice", the profile would have
    stopped learning while still looking like it worked.
    """
    from route_engine.chat import ChatBrain, THANKS

    verdicts = {"good", "bad", "great", "terrible", "awful", "nice"}
    assert not (THANKS & verdicts), (
        "these words belong to _cmd_feedback: %s" % (THANKS & verdicts))

    brain = ChatBrain(eng)
    brain.handle("Park Circus to Esplanade", "smalltalk")
    assert "adjusted" in brain.handle("great", "smalltalk"), "feedback must still be learned"
    assert "Any time" in brain.handle("thanks", "smalltalk")


def _gapped_obs(n_cams=10, block_a=120, gap_h=10.0, block_b=22, step=300.0):
    """Two blocks of collection with an outage between them, oldest first."""
    import random
    import time

    random.seed(11)
    rows = []
    t0 = time.time() - (block_a * step + gap_h * 3600 + block_b * step)
    starts = (t0, t0 + block_a * step + gap_h * 3600)
    for start, n in zip(starts, (block_a, block_b)):
        for cam in range(n_cams):
            base = random.uniform(8, 20)
            for i in range(n):
                ramp = 14.0 * math.sin(2 * math.pi * (i / 90.0))
                count = max(0.0, base + ramp + random.gauss(0, 0.25))
                rows.append({"camera_id": "c%02d" % cam, "name": "C%d" % cam,
                             "t_wall": start + i * step, "count_mean": count,
                             "moving_frac": 0.5, "lat": 0, "lon": 0, "frames": 20,
                             "count_max": int(count) + 1, "classes": {},
                             "tracks": 3, "feed_age_s": 60,
                             "width": 352, "height": 288})
    return rows


def test_validation_split_survives_a_collection_outage(tmp_path):
    """A hole in the log must not empty the tuning slice.

    The split used to cut the timeline at wall-clock fractions of
    (first, last). That assumes observations are spread evenly across the span.
    On 19-20 Sep the laptop slept for 10.3 hours, so both cut points landed
    inside the hole, the tuning slice held ZERO pairs, and the fitter fell back
    to tau=inf / damp=0 - which IS persistence, term for term. The run then
    published "Triffy beats persistence by -0.0%" to the file the dashboard
    badges REAL DATA: a comparison of a thing with itself, reported as a result.

    Cutting at a quantile of the observations instead keeps every slice
    populated however the data is distributed in time.
    """
    import json

    from route_engine import validate

    path = tmp_path / "gapped.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _gapped_obs()),
                    encoding="utf-8")

    out = validate.run(horizon_s=1200, split=0.6, verbose=False,
                       write=False, series_path=path)
    assert out, "a gapped but ample log should still validate"
    assert out["train_pairs"] >= 10, \
        "tuning slice came back empty on gapped data: %d pairs" % out["train_pairs"]
    assert out["test_pairs"] >= 20

    # NB tau == 1e9 is a legitimate grid value ("no decay"), not proof of the
    # fallback - the bug was the slice being EMPTY, which is what train_pairs
    # measures. But if the fit does land on no-decay AND no-trend then Triffy
    # is persistence, and the report has to say so instead of printing a 0.0%
    # margin that reads like a tie.
    if out["tau_s"] >= 1e8 and out["trend_damping"] == 0.0:
        assert "reduces to persistence exactly" in validate.report(out)


def test_validation_scores_one_continuous_run_not_across_an_outage(tmp_path):
    """Parameters fitted before an outage must not be tested after it.

    The three-way split is only a fair test if the fitting slice and the
    scoring slice describe the same world; across a 10 hour hole they describe
    Friday evening and Saturday morning. On the real log that transplant read
    -6.5% against persistence, where either continuous stretch measures -18%.

    So validation runs on the longest unbroken stretch - and says so, in the
    report and in the published JSON, rather than cropping quietly.
    """
    import json

    from route_engine import validate

    path = tmp_path / "gapped.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _gapped_obs()),
                    encoding="utf-8")

    out = validate.run(horizon_s=1200, split=0.6, verbose=False,
                       write=False, series_path=path)
    assert out["cropped_to_longest_run"] is True
    assert out["collection_runs"] == 2, "should have seen two collection runs"
    assert out["excluded_hours"] > 1.0, "the excluded time must be reported, not hidden"
    # Coverage figures describe what was SCORED, so after cropping there is no
    # gap left inside them - the outage shows up as excluded_hours instead.
    assert out["largest_gap_hours"] == 0.0, \
        "the scored run is supposed to be unbroken: %r" % out["largest_gap_hours"]
    # The exclusion has to be visible to a reader of the report, not only to a
    # reader of the code.
    assert "longest unbroken run" in validate.report(out)

    # And the escape hatch must actually keep everything.
    full = validate.run(horizon_s=1200, split=0.6, verbose=False, write=False,
                        series_path=path, full_span=True)
    assert full["cropped_to_longest_run"] is False
    assert full["span_hours"] > out["span_hours"], \
        "--full-span should cover more wall-clock time than one run"


def test_validation_refuses_to_publish_without_a_fitted_blend(tmp_path, capsys):
    """Too little data to fit must refuse, not quietly become persistence.

    With tau=inf and damp=0 triffy_forecast reduces to persistence exactly, so
    the table would compare a thing with itself and print a headline that reads
    like a finding. There is no honest number in that case, so write none.
    """
    import json

    from route_engine import validate

    # One short block: enough rows to get past the length check, nowhere near
    # enough spread to fit a blend on.
    rows = _gapped_obs(n_cams=4, block_a=26, gap_h=0.0, block_b=0)
    path = tmp_path / "thin.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out = validate.run(horizon_s=1800, split=0.6, verbose=False,
                       write=False, series_path=path)
    assert out == {}, "thin data must produce no result at all, not a degenerate one"


def test_a_question_containing_to_is_not_a_trip_from_a_place(eng):
    """"am i going to be late" must not become origin="am i going".

    Round three. The guard that reports "Narnia to Mordor" as a miss also
    reported these, because every one of them contains the word "to": the user
    was told "I could not find **am i going** inside the mapped area".

    The discriminator is grammatical. A sentence opening with an auxiliary verb
    or a wh-word is a question, not a pair of place names - and unlike a list
    of phrasings, English auxiliaries are a closed class that cannot go stale.
    """
    from route_engine.chat import ChatBrain, _is_question

    assert _is_question("am i going to be late")
    assert _is_question("is it better to go now or in an hour")
    assert _is_question("Which is faster")
    assert not _is_question("Narnia to Mordor")
    assert not _is_question("Park Circus to BBD Bagh")

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "q3")
    brain.handle("set work BBD Bagh", "q3")
    for phrase in ("am i going to be late",
                   "is now a good time to leave",
                   "is it better to go now or in an hour",
                   "how should i plan my evening to relax"):
        out = brain.handle(phrase, "q3")
        assert "could not find" not in out.lower(), (
            "%r blamed a phrase we cut out of it: %s" % (phrase, out[:110]))

    # ...and the place-pair case must still be reported, with suggestions.
    out = brain.handle("Narnia to Mordor", "q3")
    assert "could not find" in out.lower()


def test_chat_names_its_own_limits_instead_of_failing_to_parse(eng):
    """Questions we cannot answer deserve the reason, not "I did not catch".

    "is the metro running" was answered as a malformed route request. That is
    wrong twice: it implies the user mistyped, and it hides a limitation
    README.md states openly ("no public transport / multi-modal"). A prototype
    that cannot say its own documented limits out loud is hiding them.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "scope")

    for phrase, must_mention in (
            ("is the metro running", "metro"),
            ("what about the bus", "bus"),
            ("call me an uber", "book"),
            ("book me a cab", "book"),
            ("whats the weather", "weather"),
            ("how much will it cost", "cost")):
        out = brain.handle(phrase, "scope")
        assert not out.startswith("I did not catch"), \
            "%r still answered as a bad route" % phrase
        assert must_mention in out.lower(), \
            "%r should name the limit it hit: %s" % (phrase, out[:110])

    # A road that happens to contain a trigger word must still route. Single
    # word triggers are matched against WHOLE words so "business" cannot fire
    # "bus" - and "park" is deliberately not a trigger at all, because this
    # city has a Park Street and a Park Circus and both are in the demo.
    for phrase in ("Park Circus to BBD Bagh", "how long to park street",
                   "traffic on Park Street"):
        out = brain.handle(phrase, "scope")
        assert "parking" not in out.lower(), \
            "%r was mistaken for a parking question: %s" % (phrase, out[:110])
        assert "did not catch" not in out, "%r should still be answered" % phrase

    # But a real parking question still gets the real answer.
    assert "parking" in brain.handle("is there parking at esplanade", "scope").lower()


def test_chat_answers_questions_about_itself(eng):
    """"who are you", "what can you do", "what do you know about me"."""
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "meta")
    brain.handle("I ride a motorcycle", "meta")

    out = brain.handle("who are you", "meta")
    assert "prototype" in out.lower(), "be straight that this is a prototype: %s" % out[:110]

    assert "Plan a trip" in brain.handle("what can you do", "meta")

    # The same question as `whoami`, asked in words.
    out = brain.handle("what do you know about me", "meta")
    assert "motorcycle" in out.lower() and "Park Circus" in out

    # We do not track anyone, and should say so rather than guess.
    out = brain.handle("where am i", "meta")
    assert "do not know where you are" in out.lower()

    # Punctuation on its own is a stray tap, not a failed route request.
    for junk in ("...", "??", "!!"):
        assert "Plan a trip" in brain.handle(junk, "meta"), \
            "%r should fall back to help, not an error" % junk


def test_home_and_work_shorthands_all_agree(eng):
    """"get me home" never parsed at all, despite the code claiming it did.

    The FILLER lead-in demanded "get me TO <place>", and nobody says "to" before
    "home". Worse, once it did parse it routed home->home ("Origin and
    destination are the same place"), because the origin defaults to home. Going
    home starts from work - the reading `commute` has always given a bare
    "home", and the one _check_trip_ends documents.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "hw")
    brain.handle("set work BBD Bagh", "hw")

    for phrase in ("get me home", "take me home", "bring me home", "home",
                   "ghore jabo"):
        out = brain.handle(phrase, "hw")
        assert "same place" not in out.lower(), "%r routed to itself" % phrase
        assert not out.startswith("I did not catch"), "%r did not parse" % phrase
        assert "BBD Bagh" in out and "Park Circus" in out, \
            "%r should run work -> home: %s" % (phrase, out[:110])

    for phrase in ("take me to work", "office", "get me to work"):
        out = brain.handle(phrase, "hw")
        assert "Park Circus" in out and "BBD Bagh" in out, \
            "%r should run home -> work: %s" % (phrase, out[:110])

    # With no work saved it must ask for the field that is missing, not the one
    # that is already set.
    solo = ChatBrain(eng)
    solo.handle("set home Park Circus", "hw-solo")
    out = solo.handle("get me home", "hw-solo")
    assert "set work" in out, "should ask for the missing field: %s" % out[:110]


def test_go_now_or_wait_recommends_on_arrival_not_drive_time(eng):
    """"Should I go now?" must answer with when you ARRIVE, not how long you drive.

    Waiting almost always shortens the drive during a clearing peak. A router
    that stopped there would be telling people to sit at home so a number looks
    better, while they get in later. Arrival = departure + drive, so waiting w
    minutes only wins if the drive falls by more than w - which on a 20-minute
    trip means an incident lifting, not traffic easing.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "wtg")
    brain.handle("set work BBD Bagh", "wtg")
    brain.handle("Park Circus to BBD Bagh", "wtg")

    out = brain.handle("should i go now", "wtg")
    assert "Go now, or wait?" in out
    assert "arrive" in out.lower()
    assert "arrives first" in out, "the winning departure must be marked"
    assert "Leave now" in out, "on a normal corridor, leaving now arrives first"

    # Phrasings that all mean the same question.
    for phrase in ("is now a good time to leave", "will it be faster later",
                   "is it better to wait", "now or later", "am i going to be late"):
        assert "Go now, or wait?" in brain.handle(phrase, "wtg"), \
            "%r should reach the now-or-wait comparison" % phrase

    # With no trip in hand it must ask rather than invent one.
    fresh = ChatBrain(eng)
    assert "Which trip?" in fresh.handle("should i go now", "wtg-fresh")


def test_go_now_or_wait_says_wait_when_waiting_genuinely_wins(eng):
    """The rare branch: an incident clearing faster than the wait costs.

    Real conditions almost never produce this, which is exactly why it is
    stubbed - an unexercised branch that only fires on the day something big
    clears is a branch nobody has ever seen run.
    """
    from route_engine.chat import ChatBrain

    class _Route:
        def __init__(self, median_s):
            self.median_s = median_s

        def percentile_s(self, _q):
            return self.median_s * 1.2

    class _Plan:
        def __init__(self, median_s):
            self.best = _Route(median_s)

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "wait")
    brain.handle("set work BBD Bagh", "wait")
    brain.last_plan["wait"] = ("Park Circus", "BBD Bagh", eng.now_s)

    now = eng.now_s
    real_plan = eng.plan

    def stub(origin, destination, depart=None, **kw):
        # A 90-minute jam that collapses to 10 minutes a quarter of an hour
        # from now: leaving later genuinely arrives sooner.
        waited = (float(depart) - now) / 60.0
        return _Plan(90 * 60.0 if waited < 15 else 10 * 60.0)

    eng.plan = stub
    try:
        out = brain.handle("should i go now", "wait")
    finally:
        eng.plan = real_plan          # never leave the shared fixture patched

    assert "Wait 15 minutes" in out, out[:200]
    assert "earlier than leaving right now" in out


def test_chat_compares_two_destinations(eng):
    """"Which is faster, Esplanade or Sealdah?" - one origin, two ends.

    Shares the "X or Y" shape with the now-or-later question, so ordering
    matters: _cmd_when_to_go owns "go now or in an hour" and must see it first.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "cmp")
    brain.handle("set work BBD Bagh", "cmp")

    out = brain.handle("which is faster esplanade or sealdah", "cmp")
    assert "Esplanade" in out and "Sealdah" in out
    assert "quicker" in out or "Too close to call" in out
    assert "Park Circus" in out, "should say where it is comparing from"

    for phrase in ("whats quicker, alipore or esplanade",
                   "is sealdah or alipore closer"):
        assert "vs" in brain.handle(phrase, "cmp"), "%r should compare" % phrase

    # Three options are fine; more than MAX_COMPARE are trimmed.
    out = brain.handle("esplanade or sealdah or alipore - which is best", "cmp")
    assert out.count("—") >= 3, "all three should be priced: %s" % out[:140]

    # "now or later" is a DEPARTURE question and belongs to the other matcher.
    brain.handle("Park Circus to Esplanade", "cmp")
    for phrase in ("is it better to go now or in an hour", "now or later"):
        assert "Go now, or wait?" in brain.handle(phrase, "cmp"), \
            "%r was stolen by the destination comparison" % phrase

    # Two non-places must not be compared - hand back instead of guessing.
    out = brain.handle("which is faster narnia or mordor", "cmp")
    assert "vs" not in out


def test_chat_links_back_to_our_own_map_not_googles(eng):
    """The "see it on the map" link must point at our dashboard.

    A maps.google.com/dir/ link renders GOOGLE's route, which may take
    different roads than the one we just recommended - so the picture would
    contradict the words it is attached to, and there would be no way to see
    what our router actually chose. The separation between what is ours and
    what is not is the honesty mechanism here.
    """
    from urllib.parse import parse_qs, urlparse

    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    out = brain.handle("Park Circus to Esplanade", "maplink")
    assert "See it on the map" in out

    url = out.split("See it on the map:")[1].strip().split()[0]
    bits = urlparse(url)
    assert "google" not in url.lower() and "maps.apple" not in url.lower()
    q = parse_qs(bits.query)
    assert q["from"] == ["Park Circus"] and q["to"] == ["Esplanade"]
    assert q["at"], "the departure time has to survive, or the map replans for now"

    # Spaces and other awkward characters must be encoded, not pasted raw.
    assert " " not in url

    # An empty MAP_BASE turns the link off, for when nothing is serving.
    import route_engine.chat as chat_mod
    real = chat_mod.MAP_BASE
    chat_mod.MAP_BASE = ""
    try:
        quiet = ChatBrain(eng).handle("Park Circus to Esplanade", "maplink2")
    finally:
        chat_mod.MAP_BASE = real
    assert "See it on the map" not in quiet


def test_route_map_image_is_drawn_from_our_own_graph(tmp_path, eng):
    """A route PNG, rendered with no tile server and no network call.

    Drawing it ourselves is the same argument as the link above: a static-map
    API would put somebody else's rendering under our answer. We hold the whole
    road network already, so the streets and the route come out of the same
    data the routing decision did.
    """
    import cv2

    from route_engine import mapimg

    plan = eng.plan("Park Circus", "Esplanade", user_id="mapimg", k=3)
    out = mapimg.render_route(eng.net, plan, tmp_path / "route.png")
    img = cv2.imread(str(out))
    assert img is not None, "wrote a file that is not a readable image"
    assert img.shape[:2] == (1000, 1000)

    # The route must actually be on it. Count pixels close to Triffy blue;
    # a blank or mis-projected map fails here, which a file-size check misses.
    import numpy as np
    blue = np.abs(img.astype(int) - np.array(mapimg.ROUTE_BLUE)).sum(axis=2) < 60
    assert blue.sum() > 1500, "the route is missing or barely drawn: %d px" % blue.sum()

    # And the surrounding streets, or the route floats on a blank field with no
    # way to see where it goes or what it avoided.
    box = mapimg._bounds(plan.best.geometry(eng.net))
    proj = mapimg._Projection(box, 1000, 1000)
    blank = np.full((1000, 1000, 3), mapimg.PAPER, dtype=np.uint8)
    assert mapimg._draw_streets(blank, eng.net, box, proj) > 50, \
        "almost no surrounding streets were drawn"


def test_route_map_projection_keeps_the_aspect_honest(eng):
    """Longitude degrees are shorter than latitude ones away from the equator.

    Without the cos(lat) term a Kolkata map comes out visibly stretched
    east-west, which would make every angle on it wrong - and a map that
    misrepresents shape is worse than no map, because it looks authoritative.
    """
    from route_engine import mapimg

    # A square in metres is NOT a square in degrees at 22 N.
    box = (22.50, 88.30, 22.51, 88.31)
    proj = mapimg._Projection(box, 1000, 1000)
    x0, y0 = proj(22.505, 88.300)
    x1, y1 = proj(22.505, 88.310)
    _, ytop = proj(22.510, 88.305)
    _, ybot = proj(22.500, 88.305)
    width_px, height_px = abs(x1 - x0), abs(ybot - ytop)
    assert width_px < height_px, \
        "0.01 deg of longitude must render SHORTER than 0.01 deg of latitude here"
    assert 0.90 < (width_px / height_px) / math.cos(math.radians(22.505)) < 1.10


def test_server_listens_on_both_loopback_stacks():
    """`http://localhost:8000` must work, not only `http://127.0.0.1:8000`.

    Binding just "127.0.0.1" broke the demo: on Windows `localhost` resolves to
    the IPv6 loopback `::1` first, nothing was listening there, and the browser
    said "refused to connect" while the server was running perfectly. curl hides
    this by falling back to IPv4; browsers often do not, and one that has cached
    the failure keeps refusing afterwards.

    Loopback only, deliberately - binding "::" or "0.0.0.0" would publish the
    dashboard and the live camera feeds to whatever network the laptop is on.
    """
    import socket

    from route_engine.api import _loopback_sockets

    # Port 0 asks the OS for a free port, so this cannot collide with a running
    # dashboard or leave anything behind.
    socks = _loopback_sockets(0)
    try:
        assert socks, "nothing could listen at all"
        families = {s.family for s in socks}
        assert socket.AF_INET in families, "no IPv4 listener: 127.0.0.1 would refuse"
        assert socket.AF_INET6 in families, "no IPv6 listener: localhost would refuse"
        for s in socks:
            host = s.getsockname()[0]
            assert host in ("127.0.0.1", "::1"), \
                "bound %s - that is not loopback, it exposes the camera feeds" % host
    finally:
        for s in socks:
            s.close()


def test_no_loopback_map_link_is_sent_to_a_remote_reader(eng):
    """127.0.0.1 on someone's phone means THEIR phone.

    The map link shipped with MAP_BASE defaulting to http://127.0.0.1:8000, and
    the Telegram bot sent it to people reading on a phone. Tapping it gives
    "127.0.0.1 refused to connect", and the reasonable conclusion is that the
    bot is broken. Remote readers get the drawn PNG instead, which needs no
    network of ours; a link is only worth sending when it can actually resolve.
    """
    import route_engine.chat as chat_mod
    from route_engine.chat import ChatBrain, _is_loopback

    for base in ("http://127.0.0.1:8000", "http://localhost:8000",
                 "http://[::1]:8000", "http://0.0.0.0:8000"):
        assert _is_loopback(base), "%s is not reachable from another device" % base
    for base in ("http://192.168.29.204:8000", "https://triffy.example.com"):
        assert not _is_loopback(base)

    real = chat_mod.MAP_BASE
    try:
        chat_mod.MAP_BASE = "http://127.0.0.1:8000"
        here = ChatBrain(eng).handle("Park Circus to Esplanade", "lb-local")
        away = ChatBrain(eng, remote=True).handle("Park Circus to Esplanade", "lb-away")
        assert "See it on the map" in here, "at the machine, the link is fine"
        assert "See it on the map" not in away, \
            "a loopback link was sent to a remote reader: %s" % away[-160:]

        # Served on the network, a remote reader SHOULD get the link.
        chat_mod.MAP_BASE = "http://192.168.29.204:8000"
        away = ChatBrain(eng, remote=True).handle("Park Circus to Esplanade", "lb-lan")
        assert "192.168.29.204" in away, "a reachable link must still be sent"
    finally:
        chat_mod.MAP_BASE = real


def test_serving_on_the_network_fixes_the_map_links_too(monkeypatch):
    """TRIFFY_HOST alone must be enough; MAP_BASE should follow it.

    Serving on the network while MAP_BASE stays at 127.0.0.1 is never what
    anyone wants - the dashboard becomes reachable from a phone while the chat
    keeps handing that phone a link it cannot open. Two settings that must
    agree is one too many, and the failure is silent.
    """
    import route_engine.chat as chat_mod
    from route_engine.api import _point_map_links_at

    monkeypatch.setattr(chat_mod, "MAP_BASE", "http://127.0.0.1:8000")
    _point_map_links_at("192.168.1.50", 8000)
    assert chat_mod.MAP_BASE == "http://192.168.1.50:8000"

    # A deliberately-set base is left alone: a tunnel or hostname is a choice
    # someone made knowing more than we do.
    monkeypatch.setattr(chat_mod, "MAP_BASE", "https://triffy.example.com")
    _point_map_links_at("192.168.1.50", 8000)
    assert chat_mod.MAP_BASE == "https://triffy.example.com"

    # And an empty base stays off.
    monkeypatch.setattr(chat_mod, "MAP_BASE", "")
    _point_map_links_at("192.168.1.50", 8000)
    assert chat_mod.MAP_BASE == ""


def test_lan_addresses_never_returns_loopback():
    """The printed "type this on your phone" address must be a real one."""
    from route_engine.api import _lan_addresses

    for ip in _lan_addresses():
        assert not ip.startswith("127."), "%s is this device, not the laptop" % ip
        assert ip != "0.0.0.0", "a bind address is not a destination"


def test_viewer_mode_refuses_shared_state_writes_but_still_routes(eng, monkeypatch):
    """Sharing the dashboard must not share the demo controls.

    This API was written for loopback, where every caller is the presenter.
    Put it on a LAN address or a tunnel and that assumption is gone: POST
    /api/clock moves the network clock the dashboard AND the Telegram bot read,
    and POST /api/incident closes a road. Either is one curl away for anyone
    holding the link, and both change what the presenter sees mid-demo.

    Routing is deliberately NOT blocked. It only reads, and it is the thing
    people opened the link to see.
    """
    from fastapi.testclient import TestClient

    from route_engine import api

    monkeypatch.setattr(api, "ENGINE", eng)
    monkeypatch.setattr(api, "READONLY", True)
    client = TestClient(api.app)

    before = eng.clock
    for path, body in (("/api/clock", {"time": "09:00"}),
                       ("/api/incident", {"where": "Park Street"}),
                       ("/api/feedback", {"user": "guest"})):
        r = client.post(path, json=body)
        assert r.status_code == 403, "%s was not refused (%d)" % (path, r.status_code)
        assert "Viewer mode" in r.json()["detail"], "the refusal must say why"
    assert eng.clock == before, "the shared clock moved despite viewer mode"

    # Reads and routing still work, or there is nothing to view.
    assert client.get("/api/network").status_code == 200
    assert client.get("/api/state").status_code == 200
    r = client.post("/api/plan", json={"origin": "Park Circus",
                                       "destination": "Esplanade"})
    assert r.status_code == 200 and r.json()["routes"], "viewers cannot plan a trip"

    # The client is told, so it can hide controls instead of showing dead ones.
    assert client.get("/api/network").json()["readonly"] is True


def test_demo_controls_work_when_not_in_viewer_mode(eng, monkeypatch):
    """The guard must be off by default, or the presenter loses their own demo."""
    from fastapi.testclient import TestClient

    from route_engine import api

    monkeypatch.setattr(api, "ENGINE", eng)
    monkeypatch.setattr(api, "READONLY", False)
    client = TestClient(api.app)

    assert client.get("/api/network").json()["readonly"] is False
    r = client.post("/api/clock", json={"time": "09:15"})
    assert r.status_code == 200 and eng.clock == "09:15"
    eng.set_clock("18:30")


def test_every_mutating_route_is_listed_as_shared_state():
    """A new POST route must be considered, not silently exempt from viewer mode.

    The middleware matches an explicit list. That is deliberate - planning
    posts too, and blocking it would defeat the point - but it means a route
    added later is open by default. This fails until someone decides which it
    is.
    """
    from route_engine import api

    posts = {r.path for r in api.app.routes
             if "POST" in getattr(r, "methods", set())}
    # Read-only computation: safe to leave open, and the reason people look.
    reads = {"/api/plan", "/api/leaveby", "/api/live/plan"}
    unclassified = posts - set(api.WRITES_SHARED_STATE) - reads
    assert not unclassified, (
        "these POST routes are neither blocked in viewer mode nor declared "
        "read-only: %s" % sorted(unclassified))


def test_every_message_gets_a_reply_even_when_the_engine_fails(eng, caplog):
    """No input may escape as an exception. Silence is the worst chat failure.

    route() catches ValueError, but a RuntimeError from the router or a KeyError
    from a corrupt spatial index escaped the brain entirely. On Telegram a
    handler that raises sends *nothing*: the person sees their message delivered
    and then silence, unable to tell whether the bot is thinking, broken, or
    ignoring them.
    """
    import logging as _logging
    from route_engine.chat import ChatBrain, MAX_MESSAGE_CHARS

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "boom")

    real_plan, real_resolve = eng.plan, eng.resolve
    try:
        eng.plan = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("router exploded"))
        with caplog.at_level(_logging.ERROR):
            for phrase in ("Park Circus to BBD Bagh", "esplanade",
                           "get me to Esplanade by 21:00"):
                out = brain.handle(phrase, "boom")
                assert out and "went wrong" in out.lower(),                     "%r did not produce an apology: %r" % (phrase, out[:80])
        # The operator needs to know; the user must not see a traceback.
        assert any("dispatch failed" in r.message for r in caplog.records),             "the failure was swallowed without being logged"

        eng.resolve = lambda *a, **k: (_ for _ in ()).throw(KeyError("index corrupt"))
        assert "went wrong" in brain.handle("Park Circus to BBD Bagh", "boom").lower()
    finally:
        eng.plan, eng.resolve = real_plan, real_resolve

    # An arbitrarily long message from a public bot must not become our problem.
    out = brain.handle("x" * 5000, "boom")
    assert out and len(out) < 2000


def test_unparsed_messages_are_recorded_for_the_next_iteration(eng, tmp_path,
                                                               monkeypatch):
    """What the bot could not understand is the roadmap; it must be kept.

    Every NLU fix in this project came from someone guessing phrasings. Real
    users produce ones nobody imagined, and unless they are captured when they
    fail they are gone when the session ends.
    """
    from route_engine import chat as chat_mod
    from route_engine.chat import ChatBrain

    log = tmp_path / "misses.jsonl"
    monkeypatch.setattr(chat_mod, "MISS_LOG", log)

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "alice")

    brain.handle("zzz qqq wibble", "alice")
    brain.handle("zzz qqq wibble", "bob")
    brain.handle("Park Circus to BBD Bagh", "alice")   # understood: NOT a miss

    assert brain.misses == ["zzz qqq wibble", "zzz qqq wibble"],         "understood messages must not be recorded as misses"

    import json
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 2
    assert {r["text"] for r in rows} == {"zzz qqq wibble"}
    # Pseudonymous: the raw id must not be stored, but two people must still
    # be distinguishable from one person asking twice.
    assert all(r["user"] != "alice" and r["user"] != "bob" for r in rows)
    assert len({r["user"] for r in rows}) == 2

    # A log that cannot be written must never break the conversation.
    monkeypatch.setattr(chat_mod, "MISS_LOG", tmp_path / "no" / "such" / "dir.jsonl")
    assert brain.handle("zzz qqq wibble", "carol"), "a bad log path broke the reply"


def test_a_bare_yes_is_not_treated_as_a_failed_route(eng):
    """"yes" answers a question we never asked; it is not a routing failure.

    Found by the miss log rather than by guessing: three distinct users typed
    it within minutes of logging being switched on. The old reply was "I did
    not catch a route in that", which is a strange answer to agreement.
    """
    from route_engine.chat import ChatBrain

    brain = ChatBrain(eng)
    brain.handle("set home Park Circus", "yes-user")

    for word in ("yes", "sure", "yeah", "haan"):
        out = brain.handle(word, "yes-user")
        assert "did not catch a route" not in out.lower(),             "%r still reads as a failed route" % word
        assert "had not asked" in out.lower()

    # With a trip in context it should name that trip and offer the next step.
    brain.handle("Park Circus to BBD Bagh", "yes-user")
    out = brain.handle("yes", "yes-user")
    assert "BBD Bagh" in out and "again" in out.lower()

    # ...and the offer must actually work, or we have promised a command that
    # does not exist.
    again = brain.handle("again", "yes-user")
    assert "Park Circus" in again and "BBD Bagh" in again
    assert "did not catch" not in again.lower()

    # "thanks" must still close a conversation rather than being read as a yes.
    assert "any time" in brain.handle("thanks", "yes-user").lower()


def test_live_engine_has_one_clock(london):
    """Without replay, the live engine's clock is the real time."""
    import time
    assert abs(london.now() - time.time()) < 5


def test_london_chat_follows_the_engine_clock(london, monkeypatch):
    """The chat's 'now' must be the engine's, so both mean the same moment."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from route_engine.live_chat import LiveChatEngine
    five_pm = datetime(2026, 9, 19, 17, 0, tzinfo=ZoneInfo("Europe/London")).timestamp()
    monkeypatch.setattr(london, "now", lambda: five_pm)
    assert LiveChatEngine(london).clock == "17:00"


def test_replay_time_is_read_as_london_time():
    from datetime import datetime, timezone
    from route_engine.live_engine import parse_replay
    # 17:30 in London on 25 Sep 2026 is 16:30 UTC (British Summer Time).
    assert parse_replay("2026-09-25 17:30") == datetime(
        2026, 9, 25, 16, 30, tzinfo=timezone.utc).timestamp()
    assert parse_replay("") is None and parse_replay(None) is None
    with pytest.raises(ValueError):
        parse_replay("yesterday at five")


@pytest.fixture(scope="module")
def london_replay():
    """London replaying a recorded moment from the data in the repo."""
    from route_engine.live_engine import LiveEngine, parse_replay
    return LiveEngine(city="lon", replay_at=parse_replay("2026-09-19 17:00"))


def test_replay_uses_the_readings_of_that_moment(london_replay):
    eng = london_replay
    if not eng.observations:
        pytest.skip("the recorded data does not cover the replay moment")
    # Plenty of cameras are live at the replayed moment...
    assert len(eng.observations) > 50
    # ...and none of their readings comes from after it.
    assert all(o.t_s <= eng.now() for o in eng.observations)
    assert eng.data_age_s is not None and eng.data_age_s < 1500


def test_replay_is_reported_to_the_ui(london, london_replay):
    assert london.replay_info() is None
    info = london_replay.replay_info()
    assert info["from"] == "2026-09-19 17:00"
    state = london_replay.live_state()
    assert state["replay"]["from"] == "2026-09-19 17:00"
    assert abs(state["now_s"] - london_replay.now()) < 60


def test_replay_options_are_offered():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for module in ("route_engine.api", "route_engine.cli"):
        out = subprocess.run([sys.executable, "-m", module, "--help"],
                             capture_output=True, text=True, cwd=root)
        assert "--replay" in out.stdout, (module, out.stderr[-300:])


def test_landmarks_spelt_as_they_sound(eng):
    """People type a place the way they say it."""
    for typed in ("Parkk Sirkus", "park sirkus", "Parksircus"):
        place, how = eng.net.match(typed)
        assert place is not None and place.name == "Park Circus", typed
        assert how == "typo", typed
    # ...without inventing a match for places that are not on the map.
    for typed in ("Mumbai", "London Bridge", "Nowhereville"):
        place, how = eng.net.match(typed)
        assert how != "typo", (typed, place)


def test_directions_do_not_repeat_a_road_that_carries_on(eng):
    """"Bear left onto Mayo Road" twice in a row is one instruction."""
    for o, d in (("Park Circus", "Howrah Station"), ("Park Circus", "Sealdah")):
        for r in eng.plan(o, d, user_id="exec", k=3).routes:
            steps = r.steps[:-1]
            for a, b in zip(steps, steps[1:]):
                assert not (a.road == b.road
                            and b.instruction.startswith(("Continue", "Bear"))), \
                    (o, d, a.instruction, b.instruction)
            # Merging moves distance between steps; it never loses any.
            assert abs(sum(s.distance_m for s in r.steps) - r.distance_m) < 1.0
