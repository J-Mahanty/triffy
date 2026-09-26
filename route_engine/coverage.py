"""Which cameras a small collector should watch.

Spread evenly across London, 30 cameras mostly watch roads that none of the
trips we plan ever uses. The camera-budget experiment found that 30 cameras
chosen along those trips' routes keep journey times within about 7% of the
answer from all 172, so that is how a collector on an ordinary computer (with no
GPU, where each camera costs about 34 CPU-seconds a reading) should choose.
"""
from __future__ import annotations

import numpy as np

# The trips the demo plans. The cameras nearest their routes, alternatives
# included, are the ones whose readings change the answers people see.
DEMO_TRIPS = {
    "lon": [("King's Cross", "Tower Bridge"), ("Paddington", "Liverpool Street"),
            ("Victoria", "Waterloo"), ("Euston", "Trafalgar Square"),
            ("Marble Arch", "Bank"), ("Angel", "Elephant & Castle"),
            ("Knightsbridge", "Piccadilly Circus"), ("Camden Town", "Holborn")],
}


def route_segments(engine, trips) -> np.ndarray:
    """Every straight piece of every route offered for ``trips``, as rows of
    (lat1, lon1, lat2, lon2)."""
    segs: list = []
    for origin, destination in trips:
        a, b = engine.resolve(origin), engine.resolve(destination)
        if a is None or b is None:
            continue
        routes, _ = engine.plan_routes(a.node, b.node, k=3, with_baseline=False)
        for r in routes:
            for e in r.edges:
                g = engine.net.egeom[e]
                segs.extend((p[0], p[1], q[0], q[1]) for p, q in zip(g, g[1:]))
    return np.asarray(segs, dtype=float).reshape(-1, 4)


def distance_to_routes_m(cameras, segs: np.ndarray) -> dict:
    """Each camera's distance in metres to the nearest route segment.

    To the segment, not its end points: a straight road can be drawn as two
    points hundreds of metres apart, with the camera halfway between them.
    """
    if not len(segs):
        return {}
    squash = np.cos(np.radians(segs[:, 0].mean()))   # a degree of longitude is shorter
    ay, ax = segs[:, 0] * 111320.0, segs[:, 1] * 111320.0 * squash
    by, bx = segs[:, 2] * 111320.0, segs[:, 3] * 111320.0 * squash
    dy, dx = by - ay, bx - ax
    length2 = np.maximum(dx * dx + dy * dy, 1e-9)
    out = {}
    for c in cameras:
        py, px = c.lat * 111320.0, c.lon * 111320.0 * squash
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
        out[c.id] = float(np.hypot(ax + t * dx - px, ay + t * dy - py).min())
    return out


def cameras_near_routes(n: int, city: str = "lon", engine=None) -> list:
    """(id, metres from the nearest demo route) for the ``n`` mapped cameras
    closest to the demo routes, nearest first."""
    trips = DEMO_TRIPS.get(city)
    if not trips:
        raise ValueError("no demo trips are defined for %r" % city)
    if engine is None:
        from .live_engine import LiveEngine
        engine = LiveEngine(city=city)
    d = distance_to_routes_m(engine.mapped, route_segments(engine, trips))
    return sorted(d.items(), key=lambda kv: kv[1])[:n]
