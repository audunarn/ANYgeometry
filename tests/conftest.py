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


def _ci_shards():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ci_shards.py"
    spec = importlib.util.spec_from_file_location("anygeometry_ci_shards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """CI tiers C2: the tests the duration table records as slow get ``@pytest.mark.slow`` before
    ``-m`` deselection, so ``-m slow`` and ``-m "not slow"`` partition the suite."""
    shards = _ci_shards()
    slow = shards.slow_ids(shards.load_table())
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    """CI tiers C3: with ``ANYGEOMETRY_TEST_SHARD=k/N`` keep shard ``k`` of the selected tests
    (duration-balanced, ``tools/ci_shards.py``); ``ANYGEOMETRY_SHARD_REPORT`` records the shard: the ids
    it runs, every selected id and the selected slow ids."""
    import json
    import os
    shard = os.environ.get("ANYGEOMETRY_TEST_SHARD")
    report = os.environ.get("ANYGEOMETRY_SHARD_REPORT")
    if not shard and not report:
        return
    shards = _ci_shards()
    table = shards.load_table()
    selected = [item.nodeid for item in session.items]
    slow = sorted(item.nodeid for item in session.items if item.get_closest_marker("slow"))
    index, count = shards.parse_shard(shard) if shard else (1, 1)
    owner = shards.assign(selected, count, table)
    keep = [item for item in session.items if owner[item.nodeid] == index]
    dropped = [item for item in session.items if owner[item.nodeid] != index]
    if dropped:
        session.config.hook.pytest_deselected(items=dropped)
        session.items[:] = keep
    if report:
        Path(report).write_text(json.dumps({"index": index, "count": count, "selected": sorted(selected),
                                            "ran": [item.nodeid for item in keep], "slow": slow}, indent=0),
                                encoding="utf-8")


_B1_MODELS: dict = {}


def b1_half_hull_shared(**options):
    """The B1 half-hull for these options, built once per test session (CI tiers C1: each build costs
    5-10 s). Shared: callers only read it; a test that changes its model takes ``model.clone()``."""
    from anygeometry.b1_example import build_b1_half_hull
    key = ("half", tuple(sorted(options.items())))
    if key not in _B1_MODELS:
        _B1_MODELS[key] = build_b1_half_hull(**options)
    return _B1_MODELS[key]


def b1_full_hull_shared():
    """The full B1 hull, built once per test session (read-only, as ``b1_half_hull_shared``)."""
    from anygeometry.b1_example import build_b1_hull
    if "full" not in _B1_MODELS:
        _B1_MODELS["full"] = build_b1_hull()
    return _B1_MODELS["full"]


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
