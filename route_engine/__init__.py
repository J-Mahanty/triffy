"""Triffy — a commuter routing assistant that optimises for not being late.

Design Thinking lab prototype. Real OpenStreetMap road network, real computer
vision on camera footage, simulated traffic conditions.

Entry points::

    python -m route_engine.osm_import kol     # build the road network
    python -m route_engine.api                # dashboard on http://localhost:8000
    python -m route_engine.cli                # terminal chat assistant
    python -m route_engine.benchmark          # head-to-head evaluation
    python -m route_engine.vision <video>     # vehicle detection on real footage

See README.md for how to run it.
"""

__version__ = "0.1.0"
