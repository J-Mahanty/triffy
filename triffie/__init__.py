"""Triffie — a commuter routing assistant that optimises for not being late.

Design Thinking lab prototype. Real OpenStreetMap road network, real computer
vision on camera footage, simulated traffic conditions.

Entry points::

    python -m triffie.osm_import kol     # build the road network
    python -m triffie.api                # dashboard on http://localhost:8000
    python -m triffie.cli                # terminal chat assistant
    python -m triffie.benchmark          # head-to-head evaluation
    python -m triffie.vision <video>     # vehicle detection on real footage

See README.md for how to run it.
"""

__version__ = "0.1.0"
