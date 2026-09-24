r"""Forecasting: what will each road look like when the driver actually gets there?

This is the module that makes Triffie different from a snapshot router.

A consumer app typically chooses a route using current speeds, then quotes an
ETA. But you reach the eighth segment of your route 25 minutes from now, and by
then the evening peak has thickened and the stalled bus has been towed. Costing
every edge at *the time you will arrive at it* is called time-dependent routing,
and it needs a speed forecast, not a speed reading.

The forecast has two terms:

    v_hat(e, t) = H(e, t)  x  [ 1 + (r(e, now) - 1) * exp(-(t - now) / tau) ]
                  \______/     \_________________________________________/
                  learned                   live correction,
                  history               decaying toward normal

``H`` is the predictable part - the shape of an average Thursday, learned from
weeks of data. ``r`` is today's anomaly measured by the cameras. The exponential
says: an incident detected now still matters in five minutes, matters less in
twenty, and should not be extrapolated to an hour out, because incidents clear.

A fairness note that matters for the benchmark: ``HistoricalModel`` is
deliberately given a per-edge bias, so our learned history is imperfect the way
a real trained model would be. Handing the forecaster the simulator's exact
diurnal truth would leak the answer and inflate every number we report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (FORECAST_SIGMA_FLOOR, FORECAST_SIGMA_GROWTH,
                     LIVE_RESIDUAL_TAU_S, MIN_SPEED_KPH)
from .network import RoadNetwork


class HistoricalModel:
    """A stand-in for a model trained on weeks of archived camera history.

    In production this would be a learned per-edge, per-time-of-day, per-weekday
    speed profile. Here it wraps the simulator's incident-free behaviour and then
    corrupts it with a fixed per-edge bias plus mild shape error, because a real
    model is never exactly right and our benchmark should not pretend otherwise.
    """

    def __init__(self, sim, seed: int = 4242, bias_sigma: float = 0.085):
        self.sim = sim
        rng = np.random.default_rng(seed)
        n = sim.net.n_edges
        # Persistent per-edge bias: the model systematically over- or
        # under-estimates some roads, exactly as a real trained model does.
        self.bias = np.exp(rng.normal(0.0, bias_sigma, size=n))
        # Mild error in the *shape* of each edge's daily curve.
        self.phase = rng.normal(0.0, 900.0, size=n)      # seconds of phase error
        self.amp = np.exp(rng.normal(0.0, 0.05, size=n))

    def expected_kph(self, t_s: float) -> np.ndarray:
        """Speed this model expects on every edge at time ``t_s``."""
        # Phase error means the model thinks the peak arrives slightly early or
        # late on each road. Sample the clean profile at the shifted time.
        base = self.sim.speeds(t_s, with_incidents=False)
        shifted = self.sim.speeds(t_s + float(np.mean(self.phase)), with_incidents=False)
        blend = 0.75 * base + 0.25 * shifted
        return np.maximum(blend * self.bias * self.amp, MIN_SPEED_KPH)


@dataclass
class SpeedProfile:
    """A precomputed [timestep x edge] forecast tensor.

    Building this once per query and then doing array lookups is what keeps a
    time-dependent A* interactive. Predicting inside the search loop would be
    two orders of magnitude slower.
    """
    t0_s: float
    step_s: float
    kph: np.ndarray            # shape (T, E)
    sigma_rel: np.ndarray      # shape (T, E)
    delay_s: np.ndarray        # shape (T, E) intersection delay

    @property
    def n_steps(self) -> int:
        return self.kph.shape[0]

    @property
    def horizon_s(self) -> float:
        return self.step_s * (self.n_steps - 1)

    def index(self, t_s: float) -> int:
        i = int((t_s - self.t0_s) / self.step_s)
        return max(0, min(self.n_steps - 1, i))

    def speed(self, edge: int, t_s: float) -> float:
        return float(self.kph[self.index(t_s), edge])

    def traverse_s(self, net: RoadNetwork, edge: int, t_s: float):
        """Return (mean_seconds, sigma_seconds) to traverse ``edge`` entering at t."""
        i = self.index(t_s)
        v = self.kph[i, edge]
        mean = net.elen[edge] / (v / 3.6) + self.delay_s[i, edge]
        return float(mean), float(mean * self.sigma_rel[i, edge])


class Forecaster:
    """Turns a nowcast plus a historical model into a forward speed profile."""

    def __init__(self, net: RoadNetwork, hist: HistoricalModel,
                 tau_s: float = LIVE_RESIDUAL_TAU_S):
        self.net = net
        self.hist = hist
        self.tau_s = tau_s

    def build_profile(self, state, t0_s: float, horizon_s: float = 5400.0,
                      step_s: float = 120.0) -> SpeedProfile:
        """Project the current nowcast forward over a time grid."""
        net = self.net
        n_steps = int(horizon_s / step_s) + 1
        kph = np.empty((n_steps, net.n_edges))
        sig = np.empty((n_steps, net.n_edges))
        dly = np.empty((n_steps, net.n_edges))

        # Today's measured anomaly, which will relax back toward normal.
        anom_now = np.clip(state.anomaly, 0.12, 2.2)

        for i in range(n_steps):
            t = t0_s + i * step_s
            dt = t - state.t_s
            decay = float(np.exp(-max(0.0, dt) / self.tau_s))
            anom_t = 1.0 + (anom_now - 1.0) * decay

            h = self.hist.expected_kph(t)
            v = np.maximum(h * anom_t, MIN_SPEED_KPH)
            kph[i] = v

            # Uncertainty grows with horizon, and is larger where the nowcast had
            # little camera support to begin with.
            grow = FORECAST_SIGMA_GROWTH * (max(0.0, dt) / 600.0)
            sig[i] = np.clip(state.sigma_rel + grow, FORECAST_SIGMA_FLOOR, 0.55)

            # Intersection delay implied by how far below free-flow we are.
            ratio = np.clip(v / np.maximum(net.ekph, 1.0), 0.05, 1.0)
            sat = np.clip((1.0 / ratio) - 1.0, 0.0, 3.0)
            base = 12.0 + 83.0 * (sat ** 2.0) / (1.0 + sat ** 2.0)
            dly[i] = base * net.junction_weight

        return SpeedProfile(t0_s=t0_s, step_s=step_s, kph=kph,
                            sigma_rel=sig, delay_s=dly)

    def snapshot_profile(self, state, t0_s: float, horizon_s: float = 5400.0,
                         step_s: float = 120.0) -> SpeedProfile:
        """The baseline competitor: freeze current conditions for the whole trip.

        This is what a router does when it plans against 'live traffic' but has
        no forward model. Every timestep in the tensor is identical. Building it
        through the same machinery guarantees the comparison differs only in the
        forecast, not in the graph, the costs, or the search.
        """
        n_steps = int(horizon_s / step_s) + 1
        h_now = self.hist.expected_kph(t0_s)
        v_now = np.maximum(h_now * np.clip(state.anomaly, 0.12, 2.2), MIN_SPEED_KPH)

        net = self.net
        ratio = np.clip(v_now / np.maximum(net.ekph, 1.0), 0.05, 1.0)
        sat = np.clip((1.0 / ratio) - 1.0, 0.0, 3.0)
        delay_now = (12.0 + 83.0 * (sat ** 2.0) / (1.0 + sat ** 2.0)) * \
            net.junction_weight

        kph = np.repeat(v_now[None, :], n_steps, axis=0)
        sig = np.repeat(np.clip(state.sigma_rel, FORECAST_SIGMA_FLOOR, 0.55)[None, :],
                        n_steps, axis=0)
        dly = np.repeat(delay_now[None, :], n_steps, axis=0)
        return SpeedProfile(t0_s=t0_s, step_s=step_s, kph=kph,
                            sigma_rel=sig, delay_s=dly)

    def historical_only_profile(self, t0_s: float, horizon_s: float = 5400.0,
                                step_s: float = 120.0) -> SpeedProfile:
        """A second baseline: typical-day routing with no live data at all.

        Useful for isolating how much the cameras actually contribute, versus
        how much comes from simply being time-aware.
        """
        net = self.net
        n_steps = int(horizon_s / step_s) + 1
        kph = np.empty((n_steps, net.n_edges))
        dly = np.empty((n_steps, net.n_edges))
        for i in range(n_steps):
            v = self.hist.expected_kph(t0_s + i * step_s)
            kph[i] = v
            ratio = np.clip(v / np.maximum(net.ekph, 1.0), 0.05, 1.0)
            sat = np.clip((1.0 / ratio) - 1.0, 0.0, 3.0)
            dly[i] = (12.0 + 83.0 * (sat ** 2.0) / (1.0 + sat ** 2.0)) * \
                net.junction_weight
        sig = np.full((n_steps, net.n_edges), 0.22)
        return SpeedProfile(t0_s=t0_s, step_s=step_s, kph=kph,
                            sigma_rel=sig, delay_s=dly)
