"""Shared edge samples for display (plan S1): every emitted interval meets the angular step.

The checks are independent of the sampler: dense exact tangents (65 per interval), the support's
own normal direction at dense points, and the chord distance at dense points.
"""
import math

import numpy as np
import pytest

import display_tessellation_corpus as corpus

import importlib.util  # noqa: E402

if importlib.util.find_spec("anygeometry.display_tessellation") is None:   # absent before S1, not broken
    pytest.skip("anygeometry.display_tessellation is not implemented yet", allow_module_level=True)

from anygeometry import display_tessellation as dt  # noqa: E402
from anygeometry.curves import Straight  # noqa: E402
from anygeometry.errors import GeometryError  # noqa: E402
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface  # noqa: E402

DENSE = 65
SLACK = 1e-6


def _angles(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("...i,...i->...", a, b))


def _support_normal_turning(support, points):
    """Turning of the support normal; for a cylinder or cone the azimuth swept (exact for a
    cylinder, a conservative upper bound for a cone, whose normal turns less than its azimuth)."""
    if isinstance(support, (Cylinder, Cone)):
        offset = points - support.origin
        radial = offset - np.outer(offset @ support.axis, support.axis)
        keep = np.linalg.norm(radial, axis=1) > 1e-9
        azimuth = np.arctan2(radial @ support.circumferential_direction, radial @ support.radial_direction)
        step = np.abs((np.diff(azimuth) + np.pi) % (2 * np.pi) - np.pi)
        return float(np.where(keep[1:] & keep[:-1], step, 0.).sum())
    s = support.extrusion_coordinate(points)
    t = support.directrix.invert(points - s[:, None] * np.asarray(support.vector))
    tangent = support.directrix.derivative(t)
    return float(_angles(tangent[1:], tangent[:-1]).sum())


def _distance_to_segment(points, a, b):
    """Largest distance of ``points`` from the drawn segment ``a``-``b``."""
    d = b - a
    t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
    return float(np.linalg.norm(points - a - np.outer(t, d), axis=1).max())


def _curved_edges(case):
    g = case.model
    edges = set()
    for face_id in case.faces.values():
        face = g.faces[face_id]
        edges.update(use.edge for loop in (face.loop, *face.holes) for use in loop)
    if "edge" in case.info:
        edges.add(case.info["edge"])
    return sorted(e for e in edges if not isinstance(g.edges[e].curve, Straight))


CASES = sorted(corpus.ALL)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("step", corpus.STEPS, ids=("fine", "interaction"))
def test_every_interval_turns_at_most_one_step(name, step):
    case = corpus.ALL[name]()
    g = case.model
    tol = dt.DisplayTolerance(step)
    for edge_id in _curved_edges(case):
        samples = dt.edge_display_samples(g, edge_id, tol)
        t = samples.parameters
        assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0.0)
        # the drawn points are the curve at those parameters (ends: the vertices, within tolerance)
        exact = g.evaluate_edge_many(edge_id, t)
        assert np.array_equal(samples.points[1:-1], exact[1:-1])
        assert np.abs(samples.points[[0, -1]] - exact[[0, -1]]).max() <= 1e-9 * max(1.0, np.abs(exact).max())
        supports = [g.faces[f].surface for f in g.faces_using_edge(edge_id)]
        curved = [s for s in supports if isinstance(s, (Cylinder, Cone, ExtrudedSurface))]
        radii = [s.radius if isinstance(s, Cylinder) else max(s.radius_start, s.radius_end)
                 for s in curved if isinstance(s, (Cylinder, Cone))]
        allowance = min(radii) * (1 - math.cos(step / 2)) if radii else None
        for k, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
            dense = np.linspace(t0, t1, DENSE)
            tangent = g.edge_tangent_many(edge_id, dense)
            assert _angles(tangent[1:], tangent[:-1]).sum() <= step * (1 + SLACK), (name, edge_id, t0, t1)
            points = g.evaluate_edge_many(edge_id, dense)
            for support in curved:
                assert _support_normal_turning(support, points) <= step * (1 + SLACK), (name, edge_id, t0, t1)
            if allowance is not None:
                assert _distance_to_segment(points, samples.points[k], samples.points[k + 1])                     <= allowance * (1 + 1e-9) + 1e-12, (name, edge_id, t0, t1)


@pytest.mark.parametrize("name", CASES)
def test_samples_end_on_the_vertices_and_repeat_bit_for_bit(name):
    case = corpus.ALL[name]()
    g = case.model
    tol = dt.DisplayTolerance(corpus.FINE_STEP)
    for edge_id in sorted(g.edges):
        first = dt.edge_display_samples(g, edge_id, tol)
        again = dt.edge_display_samples(g, edge_id, tol)
        assert first.points.tobytes() == again.points.tobytes()
        assert first.parameters.tobytes() == again.parameters.tobytes()
        edge = g.edges[edge_id]
        assert np.array_equal(first.points[0], g.vertex_position(edge.start))
        assert np.array_equal(first.points[-1], g.vertex_position(edge.end))
        assert not first.points.flags.writeable and not first.parameters.flags.writeable
        if isinstance(edge.curve, Straight):
            assert first.parameters.tolist() == [0.0, 1.0]


def test_the_20_to_1_half_ellipse_meets_the_step_on_every_emitted_chord():
    """ANYfem round 2: chord turning measured over 64 chords per emitted interval."""
    case = corpus.ellipse_edge()
    g, edge = case.model, case.info["edge"]
    for step in corpus.STEPS:
        values = dt.edge_display_samples(g, edge, dt.DisplayTolerance(step)).parameters
        for t0, t1 in zip(values[:-1], values[1:]):
            chords = np.diff(g.evaluate_edge_many(edge, np.linspace(t0, t1, 65)), axis=0)
            assert _angles(chords[:-1], chords[1:]).sum() * 64 / 63 <= step * 1.02


def test_the_1000_to_1_half_ellipse_is_not_one_chord():
    """ANYfem round 3: its curved boundary got one interval with 180 degrees of tangent turning."""
    case = corpus.half_ellipse_extrusion()
    g = case.model
    for step in corpus.STEPS:
        samples = dt.edge_display_samples(g, case.info["lower"], dt.DisplayTolerance(step))
        assert len(samples.parameters) - 1 >= math.ceil(math.pi / step)


@pytest.mark.parametrize("builder", [corpus.lens, lambda: corpus.lens(.001), lambda: corpus.lens(with_neighbour=True)],
                         ids=("lens_01", "lens_0002", "lens_neighbour"))
def test_curved_sides_of_a_thin_lens_get_at_least_two_intervals(builder):
    case = builder()
    g = case.model
    for step in corpus.STEPS:
        for edge_id in _curved_edges(case):
            samples = dt.edge_display_samples(g, edge_id, dt.DisplayTolerance(step))
            assert len(samples.parameters) >= 3, (edge_id, step)


def test_every_face_using_an_edge_gets_the_same_samples():
    """The samples are a function of the edge (and its faces' supports), not of who asks."""
    case = corpus.shared_circle()
    g = case.model
    tol = dt.DisplayTolerance(corpus.INTERACTION_STEP)
    shared = case.info["shared"]
    assert sorted(g.faces_using_edge(shared)) == sorted(case.faces.values())
    first = dt.edge_display_samples(g, shared, tol)
    clone = g.clone()
    again = dt.edge_display_samples(clone, shared, tol)
    assert first.points.tobytes() == again.points.tobytes()
    assert len(first.parameters) - 1 >= math.ceil(2 * case.info["half_angle"] / corpus.INTERACTION_STEP)


def test_long_steep_cuts_follow_the_cylinder_not_their_own_turning():
    """ANYfem round 2: z = 100 sin(theta) barely turns, but crosses 90 degrees of the cylinder."""
    case = corpus.steep_cuts()
    g = case.model
    for step in corpus.STEPS:
        for edge_id in _curved_edges(case):
            samples = dt.edge_display_samples(g, edge_id, dt.DisplayTolerance(step))
            assert len(samples.parameters) - 1 >= math.ceil(math.pi / 2 / step - 1e-9)


@pytest.mark.parametrize("angle", [0.0, -0.1, math.pi, float("nan"), float("inf")])
def test_tolerance_rejects_unusable_angles(angle):
    with pytest.raises(GeometryError):
        dt.DisplayTolerance(angle)


@pytest.mark.parametrize("sag", [0.0, -1.0, float("nan")])
def test_tolerance_rejects_unusable_sag(sag):
    with pytest.raises(GeometryError):
        dt.DisplayTolerance(.1, sag)


def test_refusals_are_typed():
    with pytest.raises(ValueError):
        dt.DisplayTrimRefused("NOT_A_CODE", "x")
    refusal = dt.DisplayTrimRefused("TRIM_INVALID", "x")
    assert isinstance(refusal, GeometryError) and refusal.code == "TRIM_INVALID"


@pytest.mark.parametrize("name", ["shared_circle", "steep_cuts", "elliptic_extrusion_band", "ellipse_edge_20"])
def test_an_explicit_sag_is_honoured(name):
    """A sag tighter than the angular default bounds every drawn segment's distance from its curve."""
    case = corpus.ALL[name]()
    g = case.model
    sag = 1e-4
    for edge_id in _curved_edges(case):
        samples = dt.edge_display_samples(g, edge_id, dt.DisplayTolerance(corpus.INTERACTION_STEP, sag))
        t = samples.parameters
        for k, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
            points = g.evaluate_edge_many(edge_id, np.linspace(t0, t1, DENSE))
            assert _distance_to_segment(points, samples.points[k], samples.points[k + 1]) <= sag * (1 + 1e-9)
