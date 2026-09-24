"""Import a routable road graph from OpenStreetMap.

Why this module exists: a raw OSM extract is a soup of *ways* (polylines that
may run for kilometres through a dozen junctions). A router needs *edges*
between decision points. The core job here is splitting ways at shared nodes to
recover the real intersection topology, then keeping only the largest
strongly-connected component so every origin can actually reach every
destination.
"""
from __future__ import annotations

import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import requests

from .config import CITIES, DEFAULT_CLASS, ROAD_CLASS, CityBox, graph_path

UA = {"User-Agent": "Triffie-DesignThinkingLab/0.1 (academic prototype)"}
OSM_API = "https://api.openstreetmap.org/api/0.6/map"
WANTED = set(ROAD_CLASS)

EARTH_R = 6_371_008.8


def haversine(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def bearing(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Compass bearing in degrees from A to B."""
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dl = math.radians(b_lon - a_lon)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _tiles(box: CityBox, step: float = 0.02):
    """Split a bbox into tiles small enough for the OSM API node limit."""
    lat = box.south
    while lat < box.north - 1e-9:
        lon = box.west
        lat2 = min(lat + step, box.north)
        while lon < box.east - 1e-9:
            lon2 = min(lon + step, box.east)
            yield (lat, lon, lat2, lon2)
            lon = lon2
        lat = lat2


MIN_TILE_DEG = 0.0025   # ~250 m; below this a 400 means something else is wrong


def _fetch_tile(tile, queue, nodes, ways, seen_ways, progress, verbose) -> bool:
    """Fetch one tile, retrying transient failures. True once it is handled:
    parsed, or split into quadrants and queued. False if it never succeeded."""
    s, w, n, e = tile
    url = "%s?bbox=%.5f,%.5f,%.5f,%.5f" % (OSM_API, w, s, e, n)
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=180)
        except Exception as exc:  # transient network issue -> retry
            if attempt == 3:
                print("  tile (%.4f,%.4f): giving up (%s)" % (s, w, type(exc).__name__))
            time.sleep(2 + attempt * 3)
            continue
        if _take_tile(r, tile, queue, nodes, ways, seen_ways, progress, verbose):
            return True
        if r.status_code in (429, 509):
            time.sleep(15 + attempt * 15)   # rate limited: back off, retry
        else:
            time.sleep(2 + attempt * 3)
    return False


def _take_tile(r, tile, queue, nodes, ways, seen_ways, progress, verbose) -> bool:
    """Act on one response: parse it, or split an oversized tile. True if done."""
    s, w, n, e = tile
    if r.status_code == 200:
        _parse_osm_xml(r.content, nodes, ways, seen_ways)
        progress["done"] += 1
        if verbose:
            print("  tile %3d (%.4f,%.4f) %6d KB  nodes=%7d ways=%6d  queued=%d"
                  % (progress["done"], s, w, len(r.content) // 1024, len(nodes),
                     len(ways), len(queue)))
        return True
    if r.status_code == 400 and min(n - s, e - w) > MIN_TILE_DEG:
        # Too many nodes: split into quadrants and try again.
        ml, mo = (s + n) / 2.0, (w + e) / 2.0
        queue.extend([(s, w, ml, mo), (s, mo, ml, e),
                      (ml, w, n, mo), (ml, mo, n, e)])
        if verbose:
            print("  tile (%.4f,%.4f) too big, split into 4" % (s, w))
        return True
    return False


def download_osm(box: CityBox, verbose: bool = True):
    """Fetch nodes and highway ways for a bbox. Returns (nodes, ways).

    The /map endpoint returns *every* element in the box - buildings, shops,
    footpaths - and answers HTTP 400 once that exceeds 50,000 nodes. A 0.02 deg
    tile is fine for Kolkata but not for central London, and the first London
    import silently skipped those tiles: Westminster, Soho and the City were
    simply absent, routes detoured around the hole (Euston -> Trafalgar Square
    came out at 7.3 km for a 2.5 km crow-fly), and 80 of 172 cameras had no
    road to map to. So a too-big tile is split into quadrants and retried, and
    any tile that still cannot be fetched is a hard error: a partial map looks
    healthy after the SCC filter and poisons everything downstream.
    """
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict] = []
    seen_ways: set[int] = set()
    failed: list = []

    queue = deque(_tiles(box))
    progress = {"done": 0}
    while queue:
        tile = queue.popleft()
        if not _fetch_tile(tile, queue, nodes, ways, seen_ways, progress, verbose):
            failed.append(tile)
        time.sleep(0.5)  # be polite to a free public API

    if failed:
        raise RuntimeError("could not fetch %d tile(s), refusing to write a map "
                           "with holes in it: %s" % (len(failed), failed))
    return nodes, ways


def _parse_osm_xml(payload: bytes, nodes: dict, ways: list, seen: set) -> None:
    root = ET.fromstring(payload)
    for nd in root.iter("node"):
        nid = int(nd.get("id"))
        if nid not in nodes:
            nodes[nid] = (float(nd.get("lat")), float(nd.get("lon")))
    for wy in root.iter("way"):
        wid = int(wy.get("id"))
        if wid in seen:
            continue
        tags = {t.get("k"): t.get("v") for t in wy.findall("tag")}
        if tags.get("highway") not in WANTED:
            continue
        if tags.get("access") in ("private", "no"):
            continue
        refs = [int(nd.get("ref")) for nd in wy.findall("nd")]
        if len(refs) < 2:
            continue
        seen.add(wid)
        ways.append({"id": wid, "refs": refs, "tags": tags})


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _oneway(tags: dict):
    """Return (is_oneway, is_reversed)."""
    ow = (tags.get("oneway") or "").lower()
    if ow in ("yes", "true", "1"):
        return True, False
    if ow in ("-1", "reverse"):
        return True, True
    if tags.get("junction") in ("roundabout", "circular") and ow != "no":
        return True, False
    if tags.get("highway") in ("motorway", "motorway_link") and ow != "no":
        return True, False
    return False, False


def _lanes(tags: dict, oneway: bool) -> int:
    raw = tags.get("lanes")
    try:
        n = int(float(str(raw).split(";")[0]))
        if n > 0:
            return n if oneway else max(1, n // 2)
    except (TypeError, ValueError):
        pass
    spec = ROAD_CLASS.get(tags.get("highway", DEFAULT_CLASS), ROAD_CLASS[DEFAULT_CLASS])
    rank = spec["rank"]
    if rank <= 1:
        return 3
    if rank <= 3:
        return 2
    return 1


def _junctions(ways: list) -> set:
    """Nodes where two or more highway ways touch, plus every way's endpoints."""
    use_count: dict[int, int] = defaultdict(int)
    for w in ways:
        for r in w["refs"]:
            use_count[r] += 1
        use_count[w["refs"][0]] += 1   # force way endpoints to count as junctions
        use_count[w["refs"][-1]] += 1
    return set(n for n, c in use_count.items() if c >= 2)


def _split_way(w: dict, nodes: dict, junction: set) -> list:
    """Cut one OSM way into edges, one per stretch between junctions."""
    tags = w["tags"]
    hw = tags.get("highway", DEFAULT_CLASS)
    spec = ROAD_CLASS.get(hw, ROAD_CLASS[DEFAULT_CLASS])
    ow, rev = _oneway(tags)
    lanes = _lanes(tags, ow)
    name = tags.get("name") or tags.get("ref") or ""

    refs = [r for r in w["refs"] if r in nodes]
    if len(refs) < 2:
        return []

    edges = []
    chunk = [refs[0]]
    for r in refs[1:]:
        chunk.append(r)
        if r not in junction and r != refs[-1]:
            continue
        if len(chunk) >= 2 and chunk[0] != chunk[-1]:
            edges.append(_make_edge(chunk, nodes, hw, spec, name,
                                    lanes, ow, rev, w["id"]))
        chunk = [r]
    return edges


def build_graph(nodes: dict, ways: list, verbose: bool = True) -> dict:
    """Split ways at shared nodes to produce an intersection-to-intersection graph."""
    junction = _junctions(ways)
    edges: list[dict] = []
    for w in ways:
        edges.extend(_split_way(w, nodes, junction))

    adj: dict[int, list] = defaultdict(list)
    radj: dict[int, list] = defaultdict(list)
    for e in edges:
        adj[e["u"]].append(e["v"])
        radj[e["v"]].append(e["u"])
        if not e["oneway"]:
            adj[e["v"]].append(e["u"])
            radj[e["u"]].append(e["v"])

    keep = _largest_scc(adj, radj)
    if verbose:
        print("  raw edges=%d nodes=%d -> largest SCC nodes=%d"
              % (len(edges), len(adj), len(keep)))

    edges = [e for e in edges if e["u"] in keep and e["v"] in keep]

    # Re-index to compact integer ids: smaller JSON, array-friendly at runtime.
    used_nodes = sorted(set(n for e in edges for n in (e["u"], e["v"])))
    remap = {osm_id: i for i, osm_id in enumerate(used_nodes)}
    out_nodes = [{"id": remap[o], "osm": o, "lat": nodes[o][0], "lon": nodes[o][1]}
                 for o in used_nodes]

    out_edges = []
    for i, e in enumerate(edges):
        e = dict(e)
        e["id"] = i
        e["u"] = remap[e["u"]]
        e["v"] = remap[e["v"]]
        out_edges.append(e)

    return {"nodes": out_nodes, "edges": out_edges}


def _make_edge(chunk, nodes, hw, spec, name, lanes, oneway, reversed_, way_id) -> dict:
    pts = [nodes[r] for r in chunk]
    length = sum(haversine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                 for i in range(len(pts) - 1))
    if reversed_:
        chunk = list(reversed(chunk))
        pts = list(reversed(pts))
    return {
        "u": chunk[0], "v": chunk[-1],
        "way": way_id, "name": name, "hw": hw,
        "len": round(max(length, 1.0), 2),
        "lanes": lanes,
        "oneway": oneway,
        "kph": spec["kph"],
        "cap": spec["cap"] * lanes,
        "geom": [[round(p[0], 6), round(p[1], 6)] for p in pts],
        "bear_in": round(bearing(pts[0][0], pts[0][1], pts[1][0], pts[1][1]), 1),
        "bear_out": round(bearing(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1]), 1),
    }


def _largest_scc(adj: dict, radj: dict) -> set:
    """Iterative Kosaraju, so deep urban graphs cannot blow the Python stack."""
    allnodes = set(adj) | set(radj)
    order, seen = [], set()
    for start in allnodes:
        if start not in seen:
            _finish_order(start, adj, seen, order)

    assigned, best = set(), set()
    for node in reversed(order):
        if node in assigned:
            continue
        comp = _reverse_reach(node, radj, assigned)
        if len(comp) > len(best):
            best = comp
    return best


def _finish_order(start, adj: dict, seen: set, order: list) -> None:
    """Iterative depth-first search from ``start``, appending nodes to
    ``order`` as they finish (Kosaraju's first pass)."""
    stack = [(start, iter(adj.get(start, ())))]
    seen.add(start)
    while stack:
        _, it = stack[-1]
        nxt = next((x for x in it if x not in seen), None)
        if nxt is None:
            order.append(stack.pop()[0])
        else:
            seen.add(nxt)
            stack.append((nxt, iter(adj.get(nxt, ()))))


def _reverse_reach(node, radj: dict, assigned: set) -> set:
    """Everything that reaches ``node`` and is not yet in a component
    (Kosaraju's second pass, breadth-first over the reversed graph)."""
    comp, dq = set(), deque([node])
    assigned.add(node)
    while dq:
        cur = dq.popleft()
        comp.add(cur)
        for prv in radj.get(cur, ()):
            if prv not in assigned:
                assigned.add(prv)
                dq.append(prv)
    return comp


def _bidirectional(graph: dict) -> dict:
    """Materialise the reverse direction of every two-way edge."""
    extra = []
    nid = len(graph["edges"])
    for e in graph["edges"]:
        if e["oneway"]:
            continue
        r = dict(e)
        r["id"] = nid
        nid += 1
        r["u"], r["v"] = e["v"], e["u"]
        r["geom"] = list(reversed(e["geom"]))
        g = r["geom"]
        r["bear_in"] = round(bearing(g[0][0], g[0][1], g[1][0], g[1][1]), 1)
        r["bear_out"] = round(bearing(g[-2][0], g[-2][1], g[-1][0], g[-1][1]), 1)
        r["twin"] = e["id"]
        e["twin"] = r["id"]
        extra.append(r)
    graph["edges"].extend(extra)
    return graph


def import_city(key: str, verbose: bool = True) -> dict:
    box = CITIES[key]
    if verbose:
        print("Importing %s" % box.label)
        print("  bbox=(%s,%s)-(%s,%s)" % (box.south, box.west, box.north, box.east))
    nodes, ways = download_osm(box, verbose)
    if not ways:
        raise RuntimeError("no road ways downloaded - check network")
    graph = build_graph(nodes, ways, verbose)
    graph = _bidirectional(graph)
    graph["meta"] = {
        "city": key,
        "label": box.label,
        "bbox": [box.south, box.west, box.north, box.east],
        "center": list(box.center),
        "imported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "OpenStreetMap contributors (ODbL)",
    }
    path = graph_path(key)
    path.write_text(json.dumps(graph), encoding="utf-8")
    if verbose:
        km = sum(e["len"] for e in graph["edges"]) / 1000.0
        print("  -> %s: %d nodes, %d directed edges, %.1f km (%.1f MB)"
              % (path.name, len(graph["nodes"]), len(graph["edges"]), km,
                 path.stat().st_size / 1e6))
    return graph


if __name__ == "__main__":
    import_city(sys.argv[1] if len(sys.argv) > 1 else "blr")
