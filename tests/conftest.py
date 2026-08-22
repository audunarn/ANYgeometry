"""Reusable neutral geometry fixtures."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from anygeometry import GeometryModel


def pytest_configure(config):
    if getattr(config.option, "basetemp", None) is None:
        root = Path(__file__).resolve().parents[1]
        config.option.basetemp = str(root / f".pytest_tmp_{uuid4().hex}")


@pytest.fixture
def rectangle() -> tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]]:
    geometry = GeometryModel()
    vertices = tuple(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0))
        )
    )
    edges = tuple(geometry.add_polyline(vertices, close=True))
    face = geometry.add_face(edges)
    return geometry, face, vertices, edges


@pytest.fixture
def quarter_cylinder() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    radius = 2.0
    start = geometry.add_point(radius, 0.0, 0.0)
    via = geometry.add_point(radius / np.sqrt(2.0), radius / np.sqrt(2.0), 0.0)
    end = geometry.add_point(0.0, radius, 0.0)
    arc = geometry.add_arc(start, via, end)
    face = geometry.extrude((arc,), (0.0, 0.0, 3.0))[0]
    return geometry, arc, face
