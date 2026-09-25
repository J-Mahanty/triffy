"""Runtime road network: arrays, adjacency, spatial lookup, geocoding.

The imported JSON is convenient but slow to route over. This module converts it
once into flat NumPy arrays plus a CSR-style adjacency list, which is what makes
a time-dependent A* over ~30k edges finish in milliseconds instead of seconds.

Design note: we store *out-edge ids* per node rather than neighbour node ids.
Time-dependent routing costs an **edge at a time**, and turn penalties depend on
which edge you arrived on, so the search state is naturally edge-centric.
"""
from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from .config import ROAD_CLASS, DEFAULT_CLASS, graph_path
from .osm_import import haversine

# How close a misspelling must be to count (difflib ratio). 0.8 accepts
# "park sircus" and "sialdah" but not an unrelated word of similar length.
TYPO_CUTOFF = 0.8


def _norm(s: str) -> str:
    folded = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    key = re.sub(r"[^a-z0-9 ]+", " ", folded.lower()).strip()
    if key or not (s or "").strip():
        return key
    # Nothing Latin survived: the name is written in another script, such as
    # Bengali or Devanagari. Folding to ASCII would erase it entirely, so keep
    # its letters and marks (a Bengali vowel sign is a mark, not a letter).
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " "
                   for ch in unicodedata.normalize("NFC", s).lower())
    return " ".join(kept.split())


@dataclass
class Place:
    """A resolvable destination the user can name in chat."""
    name: str
    lat: float
    lon: float
    node: int
    kind: str = "street"


class RoadNetwork:
    def __init__(self, data: dict):
        self.meta = data.get("meta", {})
        nodes = data["nodes"]
        edges = data["edges"]

        self.n_nodes = len(nodes)
        self.n_edges = len(edges)

        self.lat = np.array([n["lat"] for n in nodes], dtype=np.float64)
        self.lon = np.array([n["lon"] for n in nodes], dtype=np.float64)

        self.eu = np.array([e["u"] for e in edges], dtype=np.int32)
        self.ev = np.array([e["v"] for e in edges], dtype=np.int32)
        self.elen = np.array([e["len"] for e in edges], dtype=np.float64)
        self.ekph = np.array([e["kph"] for e in edges], dtype=np.float64)
        self.ecap = np.array([e["cap"] for e in edges], dtype=np.float64)
        self.elanes = np.array([e["lanes"] for e in edges], dtype=np.int16)
        self.ebear_in = np.array([e["bear_in"] for e in edges], dtype=np.float32)
        self.ebear_out = np.array([e["bear_out"] for e in edges], dtype=np.float32)
        self.erank = np.array(
            [ROAD_CLASS.get(e["hw"], ROAD_CLASS[DEFAULT_CLASS])["rank"] for e in edges],
            dtype=np.int8)
        self.ename = [e.get("name", "") for e in edges]
        self.ehw = [e["hw"] for e in edges]
        self.egeom = [e["geom"] for e in edges]
        self.etwin = np.array([e.get("twin", -1) for e in edges], dtype=np.int32)

        # Free-flow traversal time, the floor every ETA is measured against.
        self.efree_s = self.elen / (self.ekph / 3.6)

        self._build_adjacency()
        self._build_junctions()
        self._build_spatial_index()
        self._build_places()

    # -- construction -------------------------------------------------------

    def _build_adjacency(self) -> None:
        """CSR out-edge lists: out_edges[out_start[n]:out_start[n+1]]."""
        order = np.argsort(self.eu, kind="stable")
        self.out_edges = order.astype(np.int32)
        counts = np.bincount(self.eu, minlength=self.n_nodes)
        self.out_start = np.zeros(self.n_nodes + 1, dtype=np.int32)
        np.cumsum(counts, out=self.out_start[1:])

        rorder = np.argsort(self.ev, kind="stable")
        self.in_edges = rorder.astype(np.int32)
        rcounts = np.bincount(self.ev, minlength=self.n_nodes)
        self.in_start = np.zeros(self.n_nodes + 1, dtype=np.int32)
        np.cumsum(rcounts, out=self.in_start[1:])

    def _build_junctions(self) -> None:
        """Identify which nodes are real intersections.

        This matters more than it looks. OSM splits a way at every geometry or
        attribute change, so a 7 km route can contain 120 graph edges while
        crossing only ~25 actual junctions. Charging a signal cycle at every
        graph node would inflate every ETA roughly twofold. A node is treated as
        a junction only when traffic genuinely has to negotiate it: out-degree
        of three or more (a fork or a crossing), rather than a mere continuation.
        """
        out_deg = np.diff(self.out_start)
        in_deg = np.diff(self.in_start)
        # A two-way street continuing straight shows out-degree 2 (onward + back),
        # so the threshold is 3 for a real fork.
        self.node_is_junction = (out_deg >= 3)
        self.node_degree = np.maximum(out_deg, in_deg).astype(np.int16)

        # Delay is incurred entering an edge, i.e. at that edge's tail node.
        tail_is_junction = self.node_is_junction[self.eu]
        # Signalled arterials queue; minor roads are give-way and much cheaper.
        class_w = np.where(self.erank <= 2, 1.0,
                  np.where(self.erank <= 4, 0.78, 0.30))
        # A bigger junction costs more to clear.
        size_w = np.clip(self.node_degree[self.eu] / 3.0, 0.7, 1.8)
        self.junction_weight = (tail_is_junction * class_w * size_w).astype(np.float64)

    def _build_spatial_index(self) -> None:
        """KD-tree over node positions, in a locally-equirectangular metric."""
        self.lat0 = float(np.mean(self.lat))
        self.mx = 111_320.0 * math.cos(math.radians(self.lat0))
        self.my = 110_540.0
        pts = np.column_stack([self.lon * self.mx, self.lat * self.my])
        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(pts)
        except Exception:
            self._tree = None
        self._pts = pts

    def _build_places(self) -> None:
        """Build a name -> location index so users can type places in chat.

        Two tiers, and the order matters. People say "Park Circus", not "Acharya
        Jagadish Chandra Bose Road segment 4412", so a curated landmark
        gazetteer is consulted first; street names from OSM are the fallback.
        """
        by_name: dict[str, list[int]] = {}
        for eid, nm in enumerate(self.ename):
            if not nm:
                continue
            by_name.setdefault(_norm(nm), []).append(eid)

        self.places: dict[str, Place] = {}
        for key, eids in by_name.items():
            if not key:
                continue
            # Anchor the name at the midpoint of its longest segment.
            best = max(eids, key=lambda e: self.elen[e])
            g = self.egeom[best]
            mid = g[len(g) // 2]
            self.places[key] = Place(self.ename[best], mid[0], mid[1],
                                     int(self.eu[best]), "street")

        self.landmarks: dict[str, Place] = {}
        self._load_landmarks()
        # Landmarks win ties against same-named streets.
        self.places.update(self.landmarks)
        self.place_keys = list(self.places)

    def _load_landmarks(self) -> None:
        from .config import DATA
        key = (self.meta or {}).get("city", "")
        path = DATA / ("landmarks_%s.json" % key)
        if not path.exists():
            return
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        for lm in blob.get("landmarks", []):
            node = self.nearest_node(lm["lat"], lm["lon"])
            plat, plon = self.node_latlon(node)
            place = Place(lm["name"], plat, plon, node, "landmark")
            for alias in [lm["name"]] + list(lm.get("aliases", [])):
                self.landmarks[_norm(alias)] = place

    # -- lookup -------------------------------------------------------------

    def nearest_node(self, lat: float, lon: float) -> int:
        q = np.array([lon * self.mx, lat * self.my])
        if self._tree is not None:
            return int(self._tree.query(q)[1])
        d = np.sum((self._pts - q) ** 2, axis=1)
        return int(np.argmin(d))

    def node_latlon(self, n: int):
        return float(self.lat[n]), float(self.lon[n])

    def resolve(self, text: str):
        """Resolve free text to a Place. Accepts 'lat,lon' or a fuzzy street name."""
        return self.match(text)[0]

    def match(self, text: str):
        """(Place or None, how it matched).

        ``how`` is "exact" when the text names the place outright (including
        one of its aliases), and otherwise says how far we had to reach:
        "phrase" (a landmark inside a longer phrase), "partial" (part of a
        street name), or "typo" (a close misspelling). A chat reply uses it to
        say "I read X as Y" instead of silently answering a different question.
        """
        text = (text or "").strip()
        if not text:
            return None, ""

        m = re.fullmatch(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*", text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            return Place("%.5f, %.5f" % (lat, lon), lat, lon,
                         self.nearest_node(lat, lon), "coord"), "exact"

        key = _norm(text)
        if not key:
            return None, ""
        if key in self.landmarks:
            return self.landmarks[key], "exact"
        if key in self.places:
            return self.places[key], "exact"

        # A landmark mentioned inside a longer phrase still wins over a street.
        lm_hits = [k for k in self.landmarks if key in k or k in key]
        if lm_hits:
            return self.landmarks[min(lm_hits, key=len)], "phrase"

        # A misspelt landmark ("park sircus", "sialdah") is what people most
        # often mean, so it is tried before loose street matching could pick
        # an unrelated street that happens to share a word.
        near = difflib.get_close_matches(key, list(self.landmarks), n=1,
                                         cutoff=TYPO_CUTOFF)
        if near:
            return self.landmarks[near[0]], "typo"

        # Substring, then token-overlap scoring: forgiving enough for chat.
        cands = ([k for k in self.place_keys if key in k or k in key]
                 or self._overlap_candidates(key))
        if cands:
            return self.places[min(cands, key=len)], "partial"

        near = difflib.get_close_matches(key, self.place_keys, n=1,
                                         cutoff=TYPO_CUTOFF)
        if near:
            return self.places[near[0]], "typo"
        return None, ""

    def _overlap_candidates(self, key: str) -> list:
        """Street names sharing enough words with ``key`` (Jaccard >= 0.34)."""
        toks = set(key.split())
        scored = []
        for k in self.place_keys:
            kt = set(k.split())
            ov = len(toks & kt) if kt else 0
            if ov:
                scored.append((ov / max(len(toks | kt), 1), k))
        scored.sort(reverse=True)
        return [k for s, k in scored[:5] if s >= 0.34]

    def suggest(self, text: str, limit: int = 6):
        key = _norm(text)
        if not key:
            return []
        hits = [k for k in self.place_keys if key in k]
        hits.sort(key=len)
        # Near-misses too, so a typo still gets a "did you mean".
        hits += difflib.get_close_matches(key, self.place_keys, n=limit, cutoff=0.7)
        names: list = []
        for k in hits:
            name = self.places[k].name
            if name not in names:
                names.append(name)
        return names[:limit]

    # -- geometry helpers ---------------------------------------------------

    def turn_angle(self, prev_edge: int, next_edge: int) -> float:
        """Absolute heading change in degrees when moving prev -> next."""
        a = float(self.ebear_out[prev_edge])
        b = float(self.ebear_in[next_edge])
        d = abs(b - a) % 360.0
        return 360.0 - d if d > 180.0 else d

    def straight_line(self, a: int, b: int) -> float:
        return haversine(self.lat[a], self.lon[a], self.lat[b], self.lon[b])

    def out_edge_ids(self, node: int):
        return self.out_edges[self.out_start[node]:self.out_start[node + 1]]

    def in_edge_ids(self, node: int):
        return self.in_edges[self.in_start[node]:self.in_start[node + 1]]

    def edge_midpoint(self, eid: int):
        g = self.egeom[eid]
        return g[len(g) // 2]

    def describe_edge(self, eid: int) -> str:
        return self.ename[eid] or "%s road" % self.ehw[eid].replace("_", " ")

    @property
    def total_km(self) -> float:
        return float(self.elen.sum() / 1000.0)


_CACHE: dict[str, RoadNetwork] = {}


def load_network(city: str | None = None) -> RoadNetwork:
    from .config import ACTIVE_CITY
    key = city or ACTIVE_CITY
    if key in _CACHE:
        return _CACHE[key]
    path = graph_path(key)
    if not path.exists():
        raise FileNotFoundError(
            "No road graph for '%s'. Run:  python -m route_engine.osm_import %s" % (key, key))
    data = json.loads(path.read_text(encoding="utf-8"))
    net = RoadNetwork(data)
    _CACHE[key] = net
    return net
