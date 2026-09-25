"""Draw a route as a picture, from our own road graph.

**Why we draw this ourselves.** The obvious shortcut is a static-map API or a
raster tile server. Both would mean the picture under a Triffy answer is
somebody else's rendering of somebody else's map, which is the same problem as
linking to Google Maps directions: the image stops being evidence of what our
router did. We already hold the entire road network in memory - every edge with
its geometry - so we can draw the streets and the route from the same data the
routing decision came out of.

The practical benefits fall out of that: no API key, no tile-usage terms, no
network call on the demo laptop, and nothing that can rate-limit us in front of
an audience. ``cv2`` and ``numpy`` are already required by the vision half of
the project, so this adds no dependency either.

Used by the Telegram bot, which can send a photo but cannot render a web page.
The dashboard link (``chat.ChatBrain.map_link``) is the better answer wherever
a browser is available - it is interactive and always current. This is for
where it is not.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Matched to the dashboard (web/app.js) so the picture and the live map read as
# one product rather than two different tools that happen to share data.
ROUTE_BLUE = (230, 90, 31)        # BGR of #1f5ae6
BASELINE_PURPLE = (209, 63, 109)  # BGR of #6d3fd1
ALT_GREY = (174, 164, 154)        # BGR of #9aa4ae
CASING = (255, 255, 255)
PAPER = (247, 245, 242)
STREET = (219, 214, 208)
BIG_STREET = (198, 192, 184)
INK = (40, 36, 32)
MUTED = (120, 112, 104)

PAD_FRAC = 0.10        # blank margin around the route, as a fraction of its span
MAJOR_RANK = 3         # net.erank <= this is drawn heavier (trunk/primary/secondary)
BAR_H = 86             # title bar height in pixels
FONT = cv2.FONT_HERSHEY_DUPLEX


def _bounds(points: list, pad_frac: float = PAD_FRAC):
    """(lat0, lon0, lat1, lon1) around ``points`` with a margin.

    The margin is a fraction of the larger span, not of each axis separately:
    padding a short axis proportionally would squash a north-south trip into a
    letterbox while a similar east-west one filled the frame.
    """
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat0, lat1 = min(lats), max(lats)
    lon0, lon1 = min(lons), max(lons)
    # A degree of longitude is shorter than a degree of latitude away from the
    # equator; compare spans in the same units or the aspect is wrong.
    scale = math.cos(math.radians((lat0 + lat1) / 2.0)) or 1.0
    span = max(lat1 - lat0, (lon1 - lon0) * scale, 1e-4)
    pad = span * pad_frac
    return lat0 - pad, lon0 - pad / scale, lat1 + pad, lon1 + pad / scale


class _Projection:
    """Equirectangular lat/lon -> pixel, fitted to a box and a canvas.

    Good enough at city scale, and it keeps the aspect honest: without the
    cos(lat) term a Kolkata map comes out visibly stretched east-west.
    """

    def __init__(self, box, width: int, height: int, top: int = 0):
        self.lat0, self.lon0, self.lat1, self.lon1 = box
        self.w, self.h, self.top = width, height, top
        self.k = math.cos(math.radians((self.lat0 + self.lat1) / 2.0)) or 1.0
        dlat = max(self.lat1 - self.lat0, 1e-9)
        dlon = max((self.lon1 - self.lon0) * self.k, 1e-9)
        # One scale for both axes, chosen so the whole box fits: separate
        # scales would fill the frame but distort every angle on the map.
        self.s = min(height / dlat, width / dlon)
        self.ox = (width - dlon * self.s) / 2.0
        self.oy = (height - dlat * self.s) / 2.0

    def __call__(self, lat: float, lon: float):
        x = self.ox + (lon - self.lon0) * self.k * self.s
        y = self.oy + (self.lat1 - lat) * self.s       # screen y grows downward
        return int(round(x)), int(round(y)) + self.top

    def pts(self, geometry: list):
        return np.array([self(p[0], p[1]) for p in geometry], dtype=np.int32)


def _in_box(geom: list, box) -> bool:
    lat0, lon0, lat1, lon1 = box
    return any(lat0 <= p[0] <= lat1 and lon0 <= p[1] <= lon1 for p in geom)


def _draw_streets(img, net, box, proj) -> int:
    """The surrounding road network, so the route has a city to sit in.

    A bare polyline on a blank field tells you nothing about where it goes or
    what it avoided. Returns how many edges were drawn, which the tests use to
    check we did not silently render an empty map.
    """
    drawn = 0
    # net.erank, NOT net.rank - and LOW is big: motorway is 0, trunk and
    # primary 1-2, residential 6. Getting either of those wrong draws a
    # perfectly plausible map with the hierarchy inverted or absent, which is
    # the sort of thing nobody notices until someone asks why the main road is
    # the faint one.
    ranks = getattr(net, "erank", None)
    for eid, geom in enumerate(net.egeom):
        if not geom or not _in_box(geom, box):
            continue
        major = bool(ranks is not None and eid < len(ranks)
                     and ranks[eid] <= MAJOR_RANK)
        cv2.polylines(img, [proj.pts(geom)], False,
                      BIG_STREET if major else STREET,
                      2 if major else 1, cv2.LINE_AA)
        drawn += 1
    return drawn


def _text(img, s: str, org, scale=0.5, colour=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _title_bar(img, plan, width: int) -> None:
    from .simulator import fmt_clock

    cv2.rectangle(img, (0, 0), (width, BAR_H), (255, 255, 255), -1)
    cv2.line(img, (0, BAR_H), (width, BAR_H), (226, 221, 215), 1, cv2.LINE_AA)
    r = plan.best
    _text(img, "%s  to  %s" % (plan.origin, plan.destination), (22, 34),
          0.62, INK, 1)
    _text(img, "leaving %s   %d min typical   budget %d min   %.1f km"
          % (fmt_clock(plan.depart_s), round(r.median_s / 60),
             round(r.percentile_s(0.9) / 60), r.distance_m / 1000.0),
          (22, 64), 0.48, MUTED, 1)
    _text(img, "Triffy", (width - 96, 34), 0.55, ROUTE_BLUE, 1)


def _markers(img, proj, geom) -> None:
    sx, sy = proj(geom[0][0], geom[0][1])
    cv2.circle(img, (sx, sy), 9, CASING, -1, cv2.LINE_AA)
    cv2.circle(img, (sx, sy), 6, INK, -1, cv2.LINE_AA)
    ex, ey = proj(geom[-1][0], geom[-1][1])
    cv2.circle(img, (ex, ey), 12, CASING, -1, cv2.LINE_AA)
    cv2.circle(img, (ex, ey), 9, ROUTE_BLUE, -1, cv2.LINE_AA)
    cv2.circle(img, (ex, ey), 4, CASING, -1, cv2.LINE_AA)


def _scale_bar(img, proj, width: int, height: int) -> None:
    """A distance scale, because a map without one invites the wrong question.

    Picks a round number of metres that lands near 160 px, so the bar is a
    useful length whatever the trip.
    """
    m_per_px = 111320.0 / proj.s
    target_m = 160 * m_per_px
    nice = min((1000, 500, 2000, 200, 5000, 100),
               key=lambda v: abs(math.log(v / target_m)) if target_m > 0 else 0)
    px = int(nice / m_per_px)
    x0, y0 = 22, height - 26
    cv2.line(img, (x0, y0), (x0 + px, y0), INK, 2, cv2.LINE_AA)
    cv2.line(img, (x0, y0 - 5), (x0, y0 + 5), INK, 2, cv2.LINE_AA)
    cv2.line(img, (x0 + px, y0 - 5), (x0 + px, y0 + 5), INK, 2, cv2.LINE_AA)
    label = "%d m" % nice if nice < 1000 else "%g km" % (nice / 1000.0)
    _text(img, label, (x0 + px + 10, y0 + 5), 0.45, INK, 1)


def _legend(img, width: int, height: int, has_alts: bool, has_baseline: bool) -> None:
    """Name the lines, because this picture travels without its UI.

    On the dashboard you can click a route to find out what it is. Sent to a
    phone it is three coloured lines and no way to ask, and the purple one in
    particular means nothing unless we say so.
    """
    rows = [(ROUTE_BLUE, "Triffy's route")]
    if has_alts:
        rows.append((ALT_GREY, "alternative"))
    if has_baseline:
        # "where it differs" is not hedging - it is what you are looking at.
        # The recommended route draws last and covers the baseline wherever the
        # two agree, so the purple that survives is exactly the disagreement.
        # Labelling it as the whole snapshot route would be a lie about a
        # fragment.
        rows.append((BASELINE_PURPLE, "snapshot route, where it differs"))

    pad, line_h = 12, 22
    box_w = 272
    box_h = pad * 2 + line_h * len(rows)
    x1, y1 = width - 18, height - 18
    x0, y0 = x1 - box_w, y1 - box_h
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (226, 221, 215), 1, cv2.LINE_AA)
    for i, (colour, label) in enumerate(rows):
        y = y0 + pad + line_h * i + 14
        cv2.line(img, (x0 + pad, y - 4), (x0 + pad + 26, y - 4), colour, 5,
                 cv2.LINE_AA)
        _text(img, label, (x0 + pad + 36, y), 0.44, INK, 1)


def render_route(net, plan, out_path, width: int = 1000, height: int = 1000,
                 with_alternatives: bool = True) -> str:
    """Write a PNG of ``plan`` over its surrounding streets. Returns the path.

    Draw order is the dashboard's: the snapshot baseline underneath (it is the
    comparison, not the answer), then alternatives, then the recommended route
    on top, each with a white casing so it reads above the street mat.
    """
    best = plan.best
    geom = best.geometry(net)
    if not geom:
        raise ValueError("route has no geometry to draw")

    all_pts = list(geom)
    others = [r for r in getattr(plan, "routes", []) if r is not best]
    if with_alternatives:
        for r in others:
            all_pts.extend(r.geometry(net))
    box = _bounds(all_pts)

    img = np.full((height, width, 3), PAPER, dtype=np.uint8)
    proj = _Projection(box, width, height - BAR_H, top=BAR_H)
    _draw_streets(img, net, box, proj)

    if with_alternatives:
        for r in others:
            pts = proj.pts(r.geometry(net))
            cv2.polylines(img, [pts], False, CASING, 8, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, ALT_GREY, 5, cv2.LINE_AA)

    # The snapshot baseline goes ON TOP of the alternatives, not under them.
    # Drawn first it was simply invisible wherever the two overlapped - and it
    # is the single most interesting line on the map, because it is what a
    # router without a forward model would have sent you down.
    baseline = getattr(plan, "baseline", None)
    base_geom = baseline.geometry(net) if baseline is not None else None
    # When the snapshot router would have picked the same roads there is
    # nothing to show, and a legend row pointing at an invisible line is worse
    # than no row at all. Agreeing is a perfectly good outcome; it just is not
    # a drawing.
    if base_geom == geom:
        base_geom = None
    if base_geom:
        cv2.polylines(img, [proj.pts(base_geom)], False, CASING, 7, cv2.LINE_AA)
        cv2.polylines(img, [proj.pts(base_geom)], False,
                      BASELINE_PURPLE, 4, cv2.LINE_AA)

    pts = proj.pts(geom)
    cv2.polylines(img, [pts], False, CASING, 11, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, ROUTE_BLUE, 7, cv2.LINE_AA)

    _markers(img, proj, geom)
    _scale_bar(img, proj, width, height)
    _legend(img, width, height, bool(others) and with_alternatives,
            bool(base_geom))
    _title_bar(img, plan, width)

    out_path = str(out_path)
    if not cv2.imwrite(out_path, img):
        raise IOError("could not write %s" % out_path)
    return out_path
