"""Nowcasting: sparse camera observations -> a speed estimate on every edge.

The central problem of this project in one sentence: we have ~180 cameras and
~13,000 road segments, so 98.6% of the network is never directly observed. The
nowcaster's job is to infer the unobserved 98.6% and, just as importantly, to
report *how confident* it is about each inference.

Three design decisions carry most of the weight.

1. **Propagate anomalies, not speeds.** A camera reading 11 km/h on Chowringhee
   says little about the absolute speed of a back lane. But "Chowringhee is at
   0.45x its normal speed for 18:30 on a Thursday" generalises well, because
   congestion *shocks* are spatially correlated even where baseline speeds are
   not. So we propagate the ratio r = v_observed / v_historical.

2. **Propagation is asymmetric.** Queues grow backwards. Roads feeding *into* a
   jam are affected far more than roads leading away from it, so upstream
   neighbours receive roughly double the weight of downstream ones. This is
   traffic physics, not a tuning hack.

3. **Precompute the propagation kernel.** The graph never changes, only the
   readings do. We expand each camera's influence region once at startup and
   store it as flat arrays, so a live update is two vectorised scatter-adds
   rather than 180 breadth-first searches per tick.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import (CORRIDOR_LEN_M, FORECAST_SIGMA_FLOOR, MAX_REACH_M,
                     MIN_SPEED_KPH, PRIOR_WEIGHT, PROPAGATION_DECAY,
                     PROPAGATION_HOPS, RANK_PENALTY)
from .network import RoadNetwork


@dataclass
class TrafficState:
    """The nowcaster's belief about the network at one instant."""
    t_s: float
    kph: np.ndarray            # estimated speed per edge
    sigma_rel: np.ndarray      # relative 1-sigma uncertainty per edge
    anomaly: np.ndarray        # estimated v_now / v_historical per edge
    observed: np.ndarray       # bool: directly seen by a camera this tick
    support: np.ndarray        # 0..1 how well-informed each edge is
    queue_m: np.ndarray        # estimated standing queue at the downstream stop line
    n_obs: int = 0

    def congestion(self, net: RoadNetwork) -> np.ndarray:
        """0 (free) .. 1 (gridlock), for map colouring."""
        ratio = self.kph / np.maximum(net.ekph, 1.0)
        return np.clip(1.0 - ratio, 0.0, 1.0)


class Nowcaster:
    def __init__(self, net: RoadNetwork, cams, hops: int = PROPAGATION_HOPS,
                 decay: float = PROPAGATION_DECAY,
                 corridor_len_m: float = CORRIDOR_LEN_M,
                 max_reach_m: float = MAX_REACH_M,
                 rank_penalty: float = RANK_PENALTY):
        self.net = net
        self.cams = cams
        self.hops = hops              # retained for reporting; see _build_kernel
        self.decay = decay
        self.corridor_len_m = corridor_len_m
        self.max_reach_m = max_reach_m
        self.rank_penalty = rank_penalty
        self._build_kernel()
        # Exponential memory of recent anomalies, so a camera that briefly loses
        # its feed does not make a road snap back to "normal" for one tick.
        self._anom_memory = np.ones(net.n_edges)
        self._memory_age = np.full(net.n_edges, 1e9)

    # -- propagation kernel -------------------------------------------------

    # Turning off the corridor costs this much extra effective distance. Large
    # enough that a side street is clearly weaker than staying on the main road,
    # small enough that adjacent streets still get some signal.
    TURN_OFF_PENALTY_M = 550.0
    # Going against the flow is cheaper (queues back up), with the flow dearer.
    UP_EXTRA_M, DOWN_EXTRA_M = 0.0, 260.0

    def _corridor_costs(self, start: int) -> dict:
        """Dijkstra over effective corridor distance from a camera's edge.

        Effective distance, not graph hops: staying on the same named road
        costs the edge's length, turning off it costs TURN_OFF_PENALTY_M more.
        """
        net = self.net
        best: dict = {start: 0.0}
        frontier = [(0.0, start)]
        while frontier:
            cost, eid = heapq.heappop(frontier)
            if cost > best.get(eid, 1e18) + 1e-9 or cost > self.max_reach_m:
                continue
            for nb, extra in self._neighbours(eid, self.UP_EXTRA_M, self.DOWN_EXTRA_M):
                nb = int(nb)
                step = float(net.elen[nb]) + extra
                # Staying on the same named road is the cheap move.
                if not (net.ename[nb] and net.ename[nb] == net.ename[eid]):
                    step += self.TURN_OFF_PENALTY_M
                nc = cost + step
                if nc < best.get(nb, 1e18) and nc <= self.max_reach_m:
                    best[nb] = nc
                    heapq.heappush(frontier, (nc, nb))
        return best

    def _camera_weights(self, cam, costs: dict):
        """(edge, weight) pairs a camera informs, from its corridor costs."""
        net = self.net
        cam_name = net.ename[cam.edge]
        for eid, cost in costs.items():
            w = math.exp(-cost / self.corridor_len_m)
            # A camera on a trunk road says little about a service alley.
            rank_gap = abs(int(net.erank[eid]) - int(net.erank[cam.edge]))
            w *= (self.rank_penalty ** rank_gap)
            # A reading generalises further along its own named road.
            if cam_name and net.ename[eid] == cam_name:
                w = min(1.0, w * 1.35)
            if w >= 0.04:
                yield eid, w

    def _build_kernel(self) -> None:
        """Expand each camera's influence region once; store as flat arrays.

        **Influence decays with distance along a corridor, not with graph hops.**

        The first version spread a reading three graph-hops outward. But OSM
        splits a road every few hundred metres, so three hops is roughly 500 m:
        a camera on Euston Road informed a fraction of Euston Road and nothing
        else. Measured on live London, only ~18% of a typical route's distance
        had any camera support at all, and one demo route had 2%.

        Hops are the wrong unit because congestion is correlated *along a
        corridor*, not within a radius. A camera showing Euston Road stopped
        says a great deal about the rest of Euston Road and much less about a
        side street that happens to be two hops away. So the search accumulates
        an **effective distance**: staying on the same named road costs the
        edge's true length, while turning off it costs an extra penalty. Weight
        is then exp(-effective_distance / CORRIDOR_L).

        Upstream still counts for more than downstream, because queues grow
        backwards - that part was right.
        """
        net = self.net
        src_list, dst_list, w_list = [], [], []

        for ci, cam in enumerate(self.cams.cams):
            for eid, w in self._camera_weights(cam, self._corridor_costs(cam.edge)):
                src_list.append(ci)
                dst_list.append(eid)
                w_list.append(w)

        self.k_src = np.array(src_list, dtype=np.int32)
        self.k_dst = np.array(dst_list, dtype=np.int32)
        self.k_w = np.array(w_list, dtype=np.float64)

        reach = np.zeros(net.n_edges)
        np.add.at(reach, self.k_dst, self.k_w)
        self.reach = reach

    def _neighbours(self, eid: int, up_extra: float, down_extra: float):
        """Yield (neighbour_edge, extra_effective_metres).

        Upstream neighbours - the edges feeding into this one - are reached more
        cheaply, because a queue grows backwards from its cause. Downstream
        neighbours pay a surcharge: traffic ahead of a jam is usually flowing.
        """
        net = self.net
        for nb in net.in_edge_ids(int(net.eu[eid])):
            yield int(nb), up_extra
        for nb in net.out_edge_ids(int(net.ev[eid])):
            yield int(nb), down_extra

    # -- the update ---------------------------------------------------------

    def update(self, observations, historical_kph: np.ndarray,
               t_s: float, prior_sigma: float = 0.16) -> TrafficState:
        """Fuse camera observations with the historical prior.

        ``historical_kph`` is what a model trained on weeks of clean history
        expects for this time of day. It is the prior; cameras supply today's
        correction.
        """
        net = self.net
        n = net.n_edges
        hist = np.maximum(historical_kph, MIN_SPEED_KPH)

        obs_edges = np.array([o.edge for o in observations], dtype=np.int32)
        obs_kph = np.array([o.speed_kph for o in observations], dtype=np.float64)
        obs_conf = np.array([o.confidence for o in observations], dtype=np.float64)
        obs_queue = np.array([o.queue_m for o in observations], dtype=np.float64)

        observed = np.zeros(n, dtype=bool)
        if len(obs_edges):
            observed[obs_edges] = True

        # 1. Anomaly at each observed edge: how far from normal is it right now?
        if len(obs_edges):
            anom_obs = obs_kph / hist[obs_edges]
            anom_obs = np.clip(anom_obs, 0.12, 2.2)
        else:
            anom_obs = np.array([])

        # 2. Scatter camera anomalies across their influence regions, weighted
        #    by kernel weight x camera confidence. Work in log space so that
        #    "half speed" and "double speed" are symmetric.
        num = np.zeros(n)
        den = np.zeros(n)
        if len(obs_edges):
            cam_index = {c.edge: i for i, c in enumerate(self.cams.cams)}
            obs_by_cam = np.full(len(self.cams.cams), np.nan)
            conf_by_cam = np.zeros(len(self.cams.cams))
            for e, a, c in zip(obs_edges, anom_obs, obs_conf):
                ci = cam_index.get(int(e))
                if ci is not None:
                    obs_by_cam[ci] = np.log(a)
                    conf_by_cam[ci] = c

            valid = ~np.isnan(obs_by_cam[self.k_src])
            src = self.k_src[valid]
            dst = self.k_dst[valid]
            w = self.k_w[valid] * conf_by_cam[src]
            np.add.at(num, dst, w * obs_by_cam[src])
            np.add.at(den, dst, w)

        # 3. Blend with the prior (anomaly 1.0 = "exactly as history predicts").
        #    An edge with strong camera support trusts the cameras; an edge with
        #    no support falls back to pure history.
        support = den / (den + PRIOR_WEIGHT)    # 0..1, saturating
        log_anom = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        anomaly = np.exp(log_anom * support)

        # 4. Temporal memory: fade last tick's belief in rather than replacing.
        fresh = support > 0.02
        self._anom_memory = np.where(fresh, anomaly,
                                     1.0 + (self._anom_memory - 1.0) * 0.86)
        anomaly = np.where(fresh, anomaly, self._anom_memory)

        kph = np.maximum(hist * anomaly, MIN_SPEED_KPH)
        # A directly observed edge is pinned to what its camera actually saw.
        if len(obs_edges):
            trust = np.clip(obs_conf, 0.0, 1.0)
            kph[obs_edges] = obs_kph * trust + kph[obs_edges] * (1.0 - trust)
            kph = np.maximum(kph, MIN_SPEED_KPH)

        # Keep the state self-consistent: kph == hist * anomaly, always.
        #
        # This is not bookkeeping. The forecaster rebuilds future speeds from
        # the *anomaly*, so if pinning changes kph without updating anomaly, the
        # single most reliable number in the system - a camera looking straight
        # at the jam - gets averaged away before it ever reaches the router.
        anomaly = np.clip(kph / hist, 0.08, 2.4)

        # 5. Uncertainty. Directly observed edges are tight; inferred edges widen
        #    with distance from any camera; unsupported edges carry the full
        #    prior spread. Being honest here is what lets the router trade off
        #    speed against reliability instead of chasing a fragile shortcut.
        sigma = prior_sigma * (1.0 - 0.72 * support) + FORECAST_SIGMA_FLOOR
        if len(obs_edges):
            sigma[obs_edges] = FORECAST_SIGMA_FLOOR + 0.05 * (1.0 - obs_conf)

        # 6. Queues: only meaningful where a camera can see the stop line.
        queue = np.zeros(n)
        if len(obs_edges):
            queue[obs_edges] = obs_queue

        return TrafficState(
            t_s=t_s, kph=kph, sigma_rel=sigma, anomaly=anomaly,
            observed=observed, support=support, queue_m=queue,
            n_obs=len(obs_edges),
        )

    # -- introspection ------------------------------------------------------

    def coverage_report(self) -> dict:
        """How much of the network any camera can speak to at all."""
        informed = self.reach > 0.04
        net = self.net
        arterial = net.erank <= 3
        return {
            "edges_total": int(net.n_edges),
            "edges_directly_watched": int(len(self.cams.cams)),
            "edges_inferable": int(informed.sum()),
            "pct_inferable": round(100.0 * informed.sum() / net.n_edges, 1),
            "pct_arterial_inferable": round(
                100.0 * (informed & arterial).sum() / max(1, arterial.sum()), 1),
            "mean_kernel_entries": int(len(self.k_dst)),
        }
