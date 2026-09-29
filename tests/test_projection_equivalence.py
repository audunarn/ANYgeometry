"""Faster surface evaluation and inversion must not change any result.

The original scalar sampling, evaluation and ``closest_uv`` are kept verbatim
below as oracles.  Production scalar and batched paths must agree with them
bit for bit, not within a tolerance.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from anygeometry import (
    CoonsSurface,
    Cone,
    Cylinder,
    GeometryModel,
    OrientedEdge,
    RuledSurface,
    closest_uv,
)
from anygeometry.surfaces import _closest_uv_many, _vector3


def _reference_sample(points, parameter):
    scaled = np.clip(float(parameter), 0.0, 1.0) * (len(points)-1)
    index = min(int(np.floor(scaled)), len(points)-2)
    local = scaled-index
    return (1.0-local)*points[index] + local*points[index+1]


class _ReferenceSurface:
    """The 0.4.4 ``evaluate`` of a ruled or Coons surface."""

    def __init__(self, surface):
        self.surface = surface

    def evaluate(self, u, v):
        s = self.surface
        if isinstance(s, RuledSurface):
            first = _reference_sample(s.first_boundary, u)
            second = _reference_sample(s.second_boundary, u)
            return (1.0 - float(v)) * first + float(v) * second
        bottom = _reference_sample(s.bottom, u)
        top = _reference_sample(s.top, u)
        left = _reference_sample(s.left, v)
        right = _reference_sample(s.right, v)
        corner_00, corner_10 = s.bottom[0], s.bottom[-1]
        corner_01, corner_11 = s.top[0], s.top[-1]
        return (
            (1.0-v)*bottom + v*top + (1.0-u)*left + u*right
            - ((1.0-u)*(1.0-v)*corner_00 + u*(1.0-v)*corner_10 + u*v*corner_11 + (1.0-u)*v*corner_01)
        )


def _reference_closest_uv(surface, point, *, initial=(0.5, 0.5), iterations=30):
    target = _vector3(point, "point")
    uv = np.asarray(initial, dtype=float).copy()
    step = 1.0e-6
    for _ in range(int(iterations)):
        current = surface.evaluate(float(uv[0]), float(uv[1]))
        derivatives = []
        for axis in range(2):
            low = uv.copy()
            high = uv.copy()
            low[axis] = max(0.0, uv[axis] - step)
            high[axis] = min(1.0, uv[axis] + step)
            span = float(high[axis] - low[axis])
            derivatives.append(
                (surface.evaluate(float(high[0]), float(high[1]))
                 - surface.evaluate(float(low[0]), float(low[1]))) / span
            )
        du, dv = derivatives
        jacobian = np.column_stack((du, dv))
        delta, *_ = np.linalg.lstsq(jacobian, target - current, rcond=None)
        uv += delta
        uv[:] = np.clip(uv, 0.0, 1.0)
        if float(np.linalg.norm(delta)) <= 1.0e-12:
            break
    return float(uv[0]), float(uv[1])


def _curve(rng, start, end, count, bulge):
    t = np.linspace(0.0, 1.0, count)
    base = np.outer(1.0 - t, start) + np.outer(t, end)
    normal = np.asarray((rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0))
    return base + np.outer(np.sin(np.pi * t) * bulge, normal)


def _surfaces():
    rng = random.Random(4045)
    for case in range(10):
        scale = rng.choice((1.0e-3, 1.0, 250.0))
        offset = np.asarray((rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))) * scale
        corners = [np.asarray(c, dtype=float) * scale + offset for c in
                   ((0, 0, 0), (2, rng.uniform(-.3, .3), 0), (2.2, 1.3, rng.uniform(-.4, .4)), (0, 1, .2))]
        counts = [rng.choice((2, 3, 9, 17)) for _ in range(4)]
        bulge = rng.uniform(0.0, 0.4) * scale
        bottom = _curve(rng, corners[0], corners[1], counts[0], bulge)
        right = _curve(rng, corners[1], corners[2], counts[1], bulge)
        top = _curve(rng, corners[3], corners[2], counts[2], bulge)
        left = _curve(rng, corners[0], corners[3], counts[3], bulge)
        if case % 2:
            yield RuledSurface(bottom, _curve(rng, corners[3], corners[2], counts[0], bulge))
        else:
            yield CoonsSurface(bottom, right, top, left)


def _queries(surface, rng, count=60):
    points = []
    for _ in range(count):
        u, v = rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2)
        base = surface.evaluate(min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))
        points.append(base + np.asarray([rng.gauss(0, 0.3) for _ in range(3)]) * rng.choice((0.0, 1e-9, 1.0)))
    for u in (0.0, 1.0, 0.5):
        for v in (0.0, 1.0, 0.5):
            points.append(surface.evaluate(u, v))
    return np.asarray(points, dtype=float)


@pytest.mark.parametrize("index", range(10))
def test_scalar_evaluation_and_inversion_match_reference(index: int) -> None:
    surface = list(_surfaces())[index]
    reference = _ReferenceSurface(surface)
    rng = random.Random(index)
    for u in (-0.5, -0.0, 0.0, 1e-300, 0.3, 0.5, 1.0 - 1e-16, 1.0, 1.5):
        for v in (-0.5, 0.0, 0.7, 1.0, 2.0):
            assert np.array_equal(surface.evaluate(u, v), reference.evaluate(u, v))
    for point in _queries(surface, rng):
        assert closest_uv(surface, point) == _reference_closest_uv(reference, point)


@pytest.mark.parametrize("index", range(10))
def test_batched_inversion_matches_scalar_rows(index: int) -> None:
    surface = list(_surfaces())[index]
    reference = _ReferenceSurface(surface)
    points = _queries(surface, random.Random(100 + index))
    expected = np.asarray([_reference_closest_uv(reference, point) for point in points])
    assert np.array_equal(_closest_uv_many(surface, points), expected)
    assert _closest_uv_many(surface, points[:0]).shape == (0, 2)


def _coons_face(geometry: GeometryModel, surface: CoonsSurface) -> int:
    """Face bounded by straight segments along the surface's sampled sides."""
    ring = [*surface.bottom[:-1], *surface.right[:-1], *surface.top[::-1][:-1], *surface.left[::-1][:-1]]
    ids = [geometry.add_point(*point) for point in ring]
    edges = [geometry.add_line(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
    sides = (len(surface.bottom), len(surface.right), len(surface.top), len(surface.left))
    corners = (0, sides[0] - 1, sides[0] + sides[1] - 2, sides[0] + sides[1] + sides[2] - 3)
    return geometry.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in edges), corners, surface=surface,
    )


def test_model_batched_face_local_uv_and_trim_loops_match_scalar() -> None:
    surface = CoonsSurface(
        np.asarray(((0.0, 0.0, 0.0), (1.0, -0.2, 0.1), (2.0, 0.0, 0.0))),
        np.asarray(((2.0, 0.0, 0.0), (2.3, 0.5, 0.2), (2.0, 1.0, 0.5))),
        np.asarray(((0.0, 1.0, 0.5), (1.0, 1.3, 0.4), (2.0, 1.0, 0.5))),
        np.asarray(((0.0, 0.0, 0.0), (-0.2, 0.5, 0.3), (0.0, 1.0, 0.5))),
    )
    geometry = GeometryModel()
    face = _coons_face(geometry, surface)
    reference = _ReferenceSurface(surface)
    points = _queries(surface, random.Random(9))
    expected = np.clip(
        np.asarray([_reference_closest_uv(reference, point) for point in points]), 0.0, 1.0
    )
    assert np.array_equal(geometry.face_local_uv_many(face, points), expected)
    loops = geometry.face_trim_loops_uv(face)
    scalar = np.asarray([geometry.face_local_uv(face, point) for point in _trim_points(geometry, face)])
    assert len(loops) == 1 and np.array_equal(loops[0], scalar)


def _trim_points(geometry: GeometryModel, face: int) -> list[np.ndarray]:
    points = []
    for item in geometry.faces[face].loop:
        samples = geometry.sample_edge(item.edge, np.linspace(0.0, 1.0, 2))
        if not item.forward:
            samples = samples[::-1]
        points.extend(samples[:-1])
    return points


def test_cached_circumferential_direction_is_a_fresh_identical_array() -> None:
    for surface in (
        Cylinder(np.zeros(3), (0.1, 0.2, 1.0), (1.0, 0.3, 0.0), 2.0, 3.0, sweep_angle=1.2),
        Cone(np.ones(3), (0.0, 0.3, 1.0), (1.0, 0.0, 0.2), 2.0, 1.0, 3.0),
    ):
        direction = surface.circumferential_direction
        assert np.array_equal(direction, np.cross(surface.axis, surface.radial_direction))
        direction[:] = 0.0  # callers still own the returned array
        assert np.array_equal(
            surface.circumferential_direction, np.cross(surface.axis, surface.radial_direction)
        )
        assert "_circumferential" not in repr(surface)


def _reference_points_in_polygon(points, polygon, *, include_boundary=True):
    """The 0.4.4 edge loop, verbatim."""
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    on_boundary = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        segment_x, segment_y = x2 - x1, y2 - y1
        offset_x, offset_y = x - x1, y - y1
        cross = np.abs(segment_x * offset_y - segment_y * offset_x)
        on_boundary |= (
            (cross <= 1.0e-10)
            & (x >= min(x1, x2) - 1.0e-10)
            & (x <= max(x1, x2) + 1.0e-10)
            & (y >= min(y1, y2) - 1.0e-10)
            & (y <= max(y1, y2) + 1.0e-10)
        )
        if segment_y != 0.0:
            crosses = (y1 > y) != (y2 > y)
            crossing_x = x1 + (y - y1) * segment_x / segment_y
            inside ^= crosses & (x < crossing_x)
        previous = current
    return np.where(on_boundary, include_boundary, inside)


@pytest.mark.parametrize("include_boundary", [True, False])
def test_block_point_in_polygon_matches_edge_loop(include_boundary: bool) -> None:
    rng = random.Random(77)
    for case in range(40):
        count = rng.randint(3, 40)
        if case % 3 == 0:
            polygon = [(i / count, 0.0) for i in range(count)] + [(1.0, 1.0), (0.0, 1.0)]
        else:
            polygon = [(0.5 + (0.3 + 0.2 * rng.random()) * np.cos(2 * np.pi * i / count),
                        0.5 + (0.3 + 0.2 * rng.random()) * np.sin(2 * np.pi * i / count))
                       for i in range(count)]
        polygon = np.asarray(polygon, dtype=float)
        queries = [(rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2)) for _ in range(50)]
        queries += [tuple(p) for p in polygon] + [tuple((a + b) / 2) for a, b in zip(polygon, np.roll(polygon, 1, axis=0))]
        queries += [(x, float(p[1])) for p in polygon for x in (0.0, 0.5, 1.0)]
        for size in (1, len(queries)):
            points = np.asarray(queries[:size], dtype=float)
            expected = _reference_points_in_polygon(points, polygon, include_boundary=include_boundary)
            made = GeometryModel._points_in_polygon(points, polygon, include_boundary=include_boundary)
            assert np.array_equal(made, expected)


# --- cylinder atlas: exact interval fast paths and replayed memoization -----

from fractions import Fraction  # noqa: E402

from anygeometry import CylinderAtlasPolicy, query_cylinder_atlas  # noqa: E402
from anygeometry import cylinder_charts as charts  # noqa: E402


def _atlas_models():
    from test_cylinder_atlas_contract import sector_hole_model, selection

    for options in ({}, {"hole": False}, {"sweep_sign": -1}, {"scale": 1.0e6}, {"equivalent_turns": 1}):
        model, _, sheet, _, _ = sector_hole_model(**options)
        yield model, selection(model, sheet)


def _run(model, selected, policy=None):
    phases = []
    atlas = query_cylinder_atlas(
        model, selected, reference_face_use=selected[0], expected_revision=model.revision,
        policy=policy, cancellation_check=phases.append,
    )
    return atlas, phases


def test_memoized_proof_matches_recomputation_including_work_and_callbacks(monkeypatch) -> None:
    for model, selected in _atlas_models():
        memoized = _run(model, selected)
        with monkeypatch.context() as patch:
            patch.setattr(charts._Proof, "_memo", lambda self, name, key, compute: compute())
            recomputed = _run(model, selected)
        assert memoized[0] == recomputed[0]
        assert memoized[0].certificate.work_counts == recomputed[0].certificate.work_counts
        assert memoized[1] == recomputed[1]


@pytest.mark.parametrize("budget", [1, 500, 5_000, 40_000])
def test_memoized_proof_refuses_at_the_same_budget(monkeypatch, budget: int) -> None:
    model, selected = next(_atlas_models())
    policy = CylinderAtlasPolicy(max_interval_operations=budget)
    memoized = _run(model, selected, policy)
    monkeypatch.setattr(charts._Proof, "_memo", lambda self, name, key, compute: compute())
    recomputed = _run(model, selected, policy)
    assert memoized[0] == recomputed[0] and memoized[1] == recomputed[1]


def _reference_add(proof, a, b):
    proof.charge()
    a, b = proof.i(a), proof.i(b)
    return proof.rounded(a.lo + b.lo, a.hi + b.hi)


def _reference_mul(proof, a, b):
    proof.charge()
    a, b = proof.i(a), proof.i(b)
    values = (a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
    return proof.rounded(min(values), max(values))


def test_integer_add_and_mul_match_fraction_arithmetic() -> None:
    rng = random.Random(11)
    proof = charts._Proof(CylinderAtlasPolicy(max_interval_operations=200000), None)
    reference = charts._Proof(CylinderAtlasPolicy(max_interval_operations=200000), None)

    def value():
        kind = rng.randrange(4)
        if kind == 0:
            return Fraction(rng.randint(-10**30, 10**30), rng.randint(1, 10**30))
        if kind == 1:
            return Fraction(rng.uniform(-1e6, 1e6))
        if kind == 2:
            return Fraction(rng.randint(-5, 5), 1 << rng.randint(0, 90))
        return Fraction(0)

    for _ in range(2500):  # two charges per round, well under the policy cap
        a = sorted((value(), value()))
        b = sorted((value(), value()))
        x, y = charts._I(*a), charts._I(*b)
        assert proof.add(x, y) == _reference_add(reference, x, y)
        assert proof.mul(x, y) == _reference_mul(reference, x, y)
    assert proof.counts == reference.counts


# --- topology-backed Coons evaluation -------------------------------------


def _reference_chain(model, chain, fraction):
    """The 0.4.4 ``_chain_point_and_derivative``, verbatim."""
    lengths = np.asarray([model.edge_length(item.edge) for item in chain])
    total = float(lengths.sum())
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(chain) - 1)
    local = (target - cumulative[index]) / lengths[index]
    item = chain[index]
    parameter = local if item.forward else 1.0 - local
    point = model.sample_edge(item.edge, np.asarray([parameter]))[0]
    derivative = model._edge_parameter_derivative(item.edge, parameter)
    derivative *= total / lengths[index]
    if not item.forward:
        derivative *= -1.0
    return point, derivative


def _reference_topology(model, face_id, u, v):
    """The 0.4.4 ``_topology_face_point_and_derivatives``, verbatim."""
    sides = model.faces[face_id].sides()
    point_a, derivative_a = _reference_chain(model, sides[0], u)
    point_b, derivative_b = _reference_chain(model, sides[1], v)
    point_c, derivative_c_raw = _reference_chain(model, sides[2], 1.0 - u)
    point_d, derivative_d_raw = _reference_chain(model, sides[3], 1.0 - v)
    derivative_c = -derivative_c_raw
    derivative_d = -derivative_d_raw
    corner_00, _ = _reference_chain(model, sides[0], 0.0)
    corner_10, _ = _reference_chain(model, sides[0], 1.0)
    corner_11, _ = _reference_chain(model, sides[2], 0.0)
    corner_01, _ = _reference_chain(model, sides[2], 1.0)
    anchor = corner_00
    point_a, point_b, point_c, point_d = (x - anchor for x in (point_a, point_b, point_c, point_d))
    corner_00, corner_10, corner_11, corner_01 = (
        x - anchor for x in (corner_00, corner_10, corner_11, corner_01))
    blend = ((1.0 - u) * (1.0 - v) * corner_00 + u * (1.0 - v) * corner_10
             + u * v * corner_11 + (1.0 - u) * v * corner_01)
    blend_du = (-(1.0 - v) * corner_00 + (1.0 - v) * corner_10 + v * corner_11 - v * corner_01)
    blend_dv = (-(1.0 - u) * corner_00 - u * corner_10 + u * corner_11 + (1.0 - u) * corner_01)
    point = anchor + ((1.0 - v) * point_a + v * point_c + (1.0 - u) * point_d + u * point_b - blend)
    du = (1.0 - v) * derivative_a + v * derivative_c - point_d + point_b - blend_du
    dv = -point_a + point_c + (1.0 - u) * derivative_d + u * derivative_b - blend_dv
    return point, du, dv


def _topology_face_model():
    model = GeometryModel()
    p0, m, p1, p2, via, p3 = (model.add_point(*xyz) for xyz in (
        (0.0, 0.0, 0.0), (0.45, -0.05, 0.0), (1.0, 0.0, 0.0), (1.1, 1.0, 0.0),
        (0.55, 1.25, 0.0), (0.0, 1.0, 0.0)))
    edges = (model.add_line(m, p0), model.add_line(m, p1), model.add_line(p1, p2),
             model.add_arc(p2, via, p3), model.add_line(p3, p0))
    forward = (False, True, True, True, True)  # one reversed use on a split side
    face = model.add_face_from_loop(
        tuple(OrientedEdge(edge, flag) for edge, flag in zip(edges, forward)), (0, 2, 3, 4),
        surface=CoonsSurface(),
    )
    return model, face, via


def _assert_topology_matches(model, face):
    rng = random.Random(5)
    grid = [(u, v) for u in (-0.2, 0.0, 0.1, 0.45, 0.5, 0.9, 1.0, 1.3)
            for v in (-0.1, 0.0, 0.3, 0.75, 1.0, 1.1)]
    grid += [(rng.random(), rng.random()) for _ in range(400)]
    for u, v in grid:
        made = model._topology_face_point_and_derivatives(face, u, v)
        expected = _reference_topology(model, face, u, v)
        for a, b in zip(made, expected):
            assert np.array_equal(a, b)


def test_topology_coons_evaluation_matches_reference_through_edits() -> None:
    model, face, via = _topology_face_model()
    _assert_topology_matches(model, face)
    _assert_topology_matches(model, face)  # warm caches
    model.move_point(via, 0.5, 1.35, 0.0)  # committed edit: new revision
    _assert_topology_matches(model, face)
    try:
        with model.transaction():
            model.move_point(via, 0.6, 1.5, 0.0)
            _assert_topology_matches(model, face)  # inside a transaction
            raise RuntimeError("roll back")
    except RuntimeError:
        pass
    _assert_topology_matches(model, face)  # rolled back to the committed map
    clone = model.clone()
    _assert_topology_matches(clone, face)
