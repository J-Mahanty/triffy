"""Draw a planned route as a picture, for chat replies.

A route card lists roads; a commuter reads a map. This renders the road graph we
already hold (OpenStreetMap data, imported by ``osm_import``) rather than
fetching map tiles, for three reasons:

* **It works offline and on stage.** No tile server, no API key, no rate limit,
  no request leaving the laptop at the moment someone is watching.
* **It shows what only Triffy knows.** The recommended route is coloured by the
  congestion forecast along it, edge by edge, in the dashboard's own traffic
  colours; a tile map would only show where the road is.
* **It costs nothing to add.** Matplotlib is already installed as a dependency
  of the vision stack.

Rendering uses matplotlib's object API (``Figure`` + Agg canvas), never
``pyplot``, whose global state is not safe from the bot's thread; a lock
serialises the rest.
"""
from __future__ import annotations

import io
import threading

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib import patheffects

# The dashboard's palette (web/style.css), so a map in chat looks like Triffy.
TRAFFIC = ("#1e9e57", "#e7a300", "#e0600e", "#b42318")   # --t0 .. --t3
TRAFFIC_NAMES = ("Flowing", "Slowing", "Congested", "Near gridlock")
BLUE = "#1f5ae6"
INK = "#111820"
INK_2 = "#45515e"
GROUND = "#f3f5f7"
ROAD = "#dde2e7"
ROAD_MAJOR = "#c3cbd4"
ALT = "#7d8a99"
LIVE = "#0b8a4d"
WARN = "#e0600e"

SIZE_IN, DPI = 8.0, 120                 # 960 px square
MIN_HALF_SPAN_M = 450.0                 # never zoom in tighter than ~1 km across
PAD = 0.14

_LOCK = threading.Lock()
_SEGMENTS: dict = {}                    # id(net) -> (net, segs, edge_of_seg, width)


def _xy(net, pts):
    """[lat, lon] pairs to local metres (equirectangular, like the KD-tree)."""
    a = np.asarray(pts, dtype=np.float64)
    return np.column_stack([a[:, 1] * net.mx, a[:, 0] * net.my])


def _segments(net):
    """Every road segment of the network in metres, built once per network."""
    hit = _SEGMENTS.get(id(net))
    if hit is not None and hit[0] is net:
        return hit[1:]
    segs, owner = [], []
    for e, g in enumerate(net.egeom):
        if len(g) < 2:
            continue
        xy = _xy(net, g)
        segs.append(np.stack([xy[:-1], xy[1:]], axis=1))
        owner.append(np.full(len(g) - 1, e, dtype=np.int32))
    segs = np.concatenate(segs)
    owner = np.concatenate(owner)
    rank = net.erank[owner]
    width = np.where(rank <= 2, 2.4, np.where(rank <= 4, 1.5, 0.8))
    _SEGMENTS[id(net)] = (net, segs, owner, width)
    return segs, owner, width


def congestion_colour(ratio: float) -> str:
    """Traffic colour for a road running at ``ratio`` of its free-flow speed.
    The same bands as the chat's words (see ``chat._mood``)."""
    if ratio > 0.75:
        return TRAFFIC[0]
    if ratio > 0.55:
        return TRAFFIC[1]
    if ratio > 0.35:
        return TRAFFIC[2]
    return TRAFFIC[3]


def _bounds(net, routes, extra_pts):
    pts = [_xy(net, r.geometry(net)) for r in routes]
    if extra_pts:
        pts.append(_xy(net, extra_pts))
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    centre = (lo + hi) / 2.0
    half = max(float((hi - lo).max()) * (0.5 + PAD), MIN_HALF_SPAN_M)
    return centre - half, centre + half


def _halo(text_artist, width: float = 3.0):
    text_artist.set_path_effects([patheffects.withStroke(linewidth=width,
                                                         foreground="white")])
    return text_artist


def render(net, routes, highlight: int = 0, speeds=None, title: str = "",
           subtitle: str = "", footer: str = "", incidents=(), cameras=(),
           route_cameras=()) -> bytes:
    """PNG bytes of ``routes`` on the road network, ``routes[highlight]`` on top.

    ``speeds`` (km/h per edge) colours the highlighted route by congestion;
    without it the route is drawn in Triffy blue. ``incidents`` and
    ``cameras`` are (lat, lon) points; ``route_cameras`` are cameras that watch
    this route and are drawn larger.
    """
    with _LOCK:
        return _render(net, routes, highlight, speeds, title, subtitle, footer,
                       list(incidents), list(cameras), list(route_cameras))


def _render(net, routes, highlight, speeds, title, subtitle, footer,
            incidents, cameras, route_cameras) -> bytes:
    fig = Figure(figsize=(SIZE_IN, SIZE_IN), dpi=DPI, facecolor=GROUND)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(GROUND)
    ax.set_axis_off()

    lo, hi = _bounds(net, routes, route_cameras)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")

    _draw_roads(ax, net, lo, hi)
    for i, r in enumerate(routes):
        if i != highlight:
            _draw_alternative(ax, net, r, i + 1)
    _draw_route(ax, net, routes[highlight], speeds)
    _draw_points(ax, net, incidents, cameras, route_cameras)
    _draw_ends(ax, net, routes[highlight])
    _draw_text(fig, title, subtitle, footer, speeds is not None)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=GROUND)
    return buf.getvalue()


def _draw_roads(ax, net, lo, hi):
    segs, _, width = _segments(net)
    mid = segs.mean(axis=1)
    inside = np.all((mid > lo) & (mid < hi), axis=1)
    major = width > 1.4
    for mask, colour in ((inside & ~major, ROAD), (inside & major, ROAD_MAJOR)):
        ax.add_collection(LineCollection(segs[mask], colors=colour,
                                         linewidths=width[mask], capstyle="round",
                                         zorder=1))


def _route_segments(net, route):
    """(segments, edge of each segment) along a route, in order."""
    segs, owner = [], []
    for e in route.edges:
        g = net.egeom[e]
        if len(g) < 2:
            continue
        xy = _xy(net, g)
        segs.append(np.stack([xy[:-1], xy[1:]], axis=1))
        owner.append(np.full(len(g) - 1, e, dtype=np.int32))
    return np.concatenate(segs), np.concatenate(owner)


def _draw_alternative(ax, net, route, number: int):
    segs, _ = _route_segments(net, route)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=7.0,
                                     capstyle="round", zorder=2))
    ax.add_collection(LineCollection(segs, colors=ALT, linewidths=3.6,
                                     capstyle="round", zorder=3, alpha=0.9))
    # Number the alternative where it is furthest from the others' shared ends.
    x, y = segs[len(segs) * 2 // 3].mean(axis=0)
    ax.scatter([x], [y], s=260, c="white", edgecolors=ALT, linewidths=2.0, zorder=4)
    ax.text(x, y, str(number), ha="center", va="center", fontsize=10,
            fontweight="bold", color=INK_2, zorder=5)


def _draw_route(ax, net, route, speeds):
    segs, owner = _route_segments(net, route)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=11.0,
                                     capstyle="round", zorder=6))
    if speeds is None:
        colours = [BLUE] * len(segs)
    else:
        ratio = np.asarray(speeds)[owner] / np.maximum(net.ekph[owner], 1.0)
        colours = [congestion_colour(float(r)) for r in ratio]
    ax.add_collection(LineCollection(segs, colors=colours, linewidths=6.5,
                                     capstyle="round", zorder=7))


def _draw_points(ax, net, incidents, cameras, route_cameras):
    if cameras:
        xy = _xy(net, cameras)
        ax.scatter(xy[:, 0], xy[:, 1], s=16, marker="s", c=LIVE, alpha=0.55,
                   linewidths=0, zorder=8)
    if route_cameras:
        xy = _xy(net, route_cameras)
        ax.scatter(xy[:, 0], xy[:, 1], s=90, marker="s", c=LIVE,
                   edgecolors="white", linewidths=1.6, zorder=9)
    if incidents:
        xy = _xy(net, incidents)
        ax.scatter(xy[:, 0], xy[:, 1], s=190, marker="^", c=WARN,
                   edgecolors="white", linewidths=1.8, zorder=9)


def _draw_ends(ax, net, route):
    g = route.geometry(net)
    start, end = _xy(net, [g[0], g[-1]])
    ax.scatter([start[0]], [start[1]], s=210, c="white", edgecolors=INK,
               linewidths=3.0, zorder=10)
    ax.scatter([end[0]], [end[1]], s=260, c=INK, edgecolors="white",
               linewidths=3.0, zorder=10)
    for (x, y), word in ((start, "Start"), (end, "Finish")):
        _halo(ax.annotate(word, (x, y), xytext=(0, 14), textcoords="offset points",
                          ha="center", fontsize=10, fontweight="bold", color=INK,
                          zorder=11))


def _draw_text(fig, title, subtitle, footer, coloured: bool):
    if title:
        _halo(fig.text(0.03, 0.965, title, fontsize=17, fontweight="bold",
                       color=INK, va="top"), 4.0)
    if subtitle:
        _halo(fig.text(0.03, 0.915, subtitle, fontsize=12, color=INK_2, va="top"), 4.0)
    if coloured:
        x = 0.03
        for colour, name in zip(TRAFFIC, TRAFFIC_NAMES):
            fig.patches.append(_chip(fig, x, colour))
            _halo(fig.text(x + 0.03, 0.058, name, fontsize=9.5, color=INK_2,
                           va="center"))
            x += 0.045 + 0.0105 * len(name)
    if footer:
        _halo(fig.text(0.03, 0.022, footer, fontsize=8.5, color=INK_2, va="center"))


def _chip(fig, x, colour):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch((x, 0.051), 0.022, 0.014, boxstyle="round,pad=0.002",
                          transform=fig.transFigure, facecolor=colour,
                          edgecolor="white", linewidth=1.0, figure=fig)
