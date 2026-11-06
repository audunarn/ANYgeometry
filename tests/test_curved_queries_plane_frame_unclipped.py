"""Curved edge-face, member-face and face-face queries on Plane faces whose stored frame does not cover the face.

A ``Plane`` support is an unbounded affine chart; its stored frame is free.
The curved query engine used to clip Plane support uv to ``[0, 1]^2`` (and to
sample face-face pairs only on that square), so real crossings outside the
frame's unit square were certified as disjoint (S0g; probe
``redesign/s0f_curved_probe.py``).  Supports other than an exact Plane, and
Planes whose frame covers the face, must keep their old results bit for bit:
``tests/data/s0g_curved_query_bit_identity.json`` holds digests captured on
the unfixed integration tip e66cbee.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.curved_intersections import qualified_curve_face, qualified_face_face
from anygeometry.curves import Straight
from anygeometry.generators import cylinder
from anygeometry.intersections import IntersectionKind, query_intersection
from anygeometry.surfaces import Cylinder, Plane

DATA = pathlib.Path(__file__).with_name("data") / "s0g_curved_query_bit_identity.json"

PLATE = ((-2.0, 0.5, -1.0), (2.0, 0.5, -1.0), (2.0, 0.5, 2.0), (-2.0, 0.5, 2.0))
FRAMES = {
    # Unit square equals the face (the probe's covering frame).
    "covering": ((-2.0, 0.5, -1.0), (4.0, 0.0, 0.0), (0.0, 0.0, 3.0)),
    # Same plane; the unit square covers 1/48 of the face (raw u up to 8).
    "partial": ((-2.0, 0.5, -1.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.5)),
    # Same plane; the frame covers only x, z in [0, 1] of the face.
    "part": ((0.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    # Same plane; frame far from the face with a tiny u vector (cap-like).
    "remote": ((50.0, 0.5, 40.0), (0.009, 0.0, 0.0), (0.0, 0.0, -8.0)),
}
# Covering frames for the bit-identity sample (margin and in-plane rotation).
COVERING_SAMPLE = {
    "covering": FRAMES["covering"],
    "margin": ((-3.0, 0.5, -2.0), (6.0, 0.0, 0.0), (0.0, 0.0, 5.0)),
    # Rotated 30 degrees in the plane; raw uv of the face in about [0.045, 0.95].
    "rotated": (
        (-1.091, 0.5, -3.073),
        (5.5 * np.cos(np.pi / 6.0), 0.0, 5.5 * np.sin(np.pi / 6.0)),
        (-5.5 * np.sin(np.pi / 6.0), 0.0, 5.5 * np.cos(np.pi / 6.0)),
    ),
}


def _probe_model(frame):
    model = cylinder(1.0, 1.0, circumferential_segments=4)
    face = model.add_plate(model.add_points(PLATE))
    model.set_face_surface(face, Plane(*frame))
    assert model.validate_topology() == ()
    return model, face


def _curved_edges(model):
    return [eid for eid, edge in sorted(model.edges.items()) if not isinstance(edge.curve, Straight)]


def _cylinder_faces(model, plate):
    return [fid for fid, face in sorted(model.faces.items()) if fid != plate and isinstance(face.surface, Cylinder)]


def _points(result):
    return [np.asarray(witness, dtype=float) for component in result.components for witness in component.witnesses]


def _assert_same_points(actual, expected, tolerance=1.0e-9):
    assert len(actual) == len(expected)
    for point in expected:
        assert min(float(np.linalg.norm(point - other)) for other in actual) <= tolerance


@pytest.mark.parametrize("frame", ["partial", "part", "remote"])
def test_curved_edge_face_crossings_do_not_depend_on_plane_frame(frame):
    reference, reference_face = _probe_model(FRAMES["covering"])
    model, face = _probe_model(FRAMES[frame])
    crossings = 0
    for eid in _curved_edges(model):
        expected = qualified_curve_face(reference, reference.handle("edge", eid), reference.handle("face", reference_face))
        for result in (
            qualified_curve_face(model, model.handle("edge", eid), model.handle("face", face)),
            query_intersection(model, model.handle("edge", eid), model.handle("face", face)),
        ):
            assert result.kind is expected.kind, (eid, result.diagnostics)
            _assert_same_points(_points(result), _points(expected))
            for component in result.components:
                # Reported face uv are support uv: they evaluate to the witness.
                uv = component.second_parameter
                assert uv is not None
                assert np.linalg.norm(model.face_support_point(face, *uv) - np.asarray(component.witnesses[0])) <= 1.0e-9
        crossings += expected.kind is IntersectionKind.CROSS
    assert crossings == 4


@pytest.mark.parametrize("frame", ["covering", "partial", "part"])
def test_curved_member_face_crossings_do_not_depend_on_plane_frame(frame):
    model, face = _probe_model(FRAMES[frame])
    vertices = model.add_points(((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (-1.0, 0.0, 0.5)))
    member = model.add_member((model.add_arc(*vertices),))
    result = query_intersection(model, model.handle("member", member), model.handle("face", face))
    assert result.kind is IntersectionKind.CROSS, result.diagnostics
    x = np.sqrt(0.75)
    _assert_same_points(_points(result), [np.array((x, 0.5, 0.5)), np.array((-x, 0.5, 0.5))])


@pytest.mark.parametrize("frame", ["partial", "part", "remote"])
def test_curved_face_face_traces_do_not_depend_on_plane_frame(frame):
    reference, reference_face = _probe_model(FRAMES["covering"])
    model, face = _probe_model(FRAMES[frame])
    found = 0
    for fid in _cylinder_faces(model, face):
        expected = qualified_face_face(reference, reference.handle("face", fid), reference.handle("face", reference_face))
        for result in (
            qualified_face_face(model, model.handle("face", fid), model.handle("face", face)),
            query_intersection(model, model.handle("face", fid), model.handle("face", face)),
        ):
            assert result.kind is expected.kind, (fid, result.diagnostics)
            assert len(result.components) == len(expected.components)
            for component, other in zip(result.components, expected.components):
                ends = [np.asarray(component.witnesses[0]), np.asarray(component.witnesses[-1])]
                _assert_same_points(ends, [np.asarray(other.witnesses[0]), np.asarray(other.witnesses[-1])], 1.0e-7)
                for uv, witness in zip(component.second_parameter_path, component.witnesses):
                    assert np.linalg.norm(model.face_support_point(face, *uv) - np.asarray(witness)) <= 1.0e-7
        found += expected.kind is IntersectionKind.CROSS
    assert found == 2


def _codex_plates(frame):
    model = GeometryModel()
    first = model.add_plate(model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    second = model.add_plate(model.add_points(((0.8, 0, -1), (0.8, 1, -1), (0.8, 1, 1), (0.8, 0, 1))))
    model.set_face_surface(first, Plane(*frame))
    return model, first, second


@pytest.mark.parametrize("order", ["plane_first", "plane_second"])
def test_face_face_fallback_finds_crossing_outside_plane_frame(order):
    """Codex S0f-R1 finding 2: the fallback sampled and refined only on [0, 1]^2."""

    for frame in (((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 0, 0), (0.5, 0, 0), (0, 0.5, 0))):
        model, first, second = _codex_plates(frame)
        handles = (model.handle("face", first), model.handle("face", second))
        if order == "plane_second":
            handles = handles[::-1]
        result = qualified_face_face(model, *handles)
        assert result.kind is IntersectionKind.CROSS, (frame, result.diagnostics)
        points = _points(result)
        assert min(abs(float(point[0]) - 0.8) for point in points) <= 1.0e-9
        assert max(abs(float(point[2])) for point in points) <= 1.0e-9
        ys = sorted(float(point[1]) for point in points)
        assert ys[0] <= 1.0e-9 and ys[-1] >= 1.0 - 1.0e-9


def test_arc_trimmed_plane_face_uv_box_includes_the_arc_bulge():
    """The covering decision bounds an arc by its extrema, not by its vertices."""

    from anygeometry.curved_intersections import _plane_chart

    results = {}
    for name, frame in (
        ("vertices_only", ((-2.0, 0.5, -1.0), (4.0, 0.0, 0.0), (0.0, 0.0, 4.0))),
        ("covering", ((-2.5, 0.5, -1.0), (5.0, 0.0, 0.0), (0.0, 0.0, 4.0))),
    ):
        model = cylinder(1.0, 1.0, circumferential_segments=4)
        cylinder_edges = _curved_edges(model)
        # Circle centre (0, 0.5, 0.5), radius 2.5; the vias sit at 45 and 135
        # degrees, so the x extrema (+-2.5 at 0 and 180 degrees) are interior.
        offset = 2.5 * np.sqrt(0.5)
        start, right, top, left, end = model.add_points(
            ((2.0, 0.5, -1.0), (offset, 0.5, 0.5 + offset), (0.0, 0.5, 3.0), (-offset, 0.5, 0.5 + offset), (-2.0, 0.5, -1.0))
        )
        face = model.add_face(
            (model.add_arc(start, right, top), model.add_arc(top, left, end), model.add_line(end, start))
        )
        model.set_face_surface(face, Plane(*frame))
        assert model.validate_topology() == ()
        chart = _plane_chart(model, face)
        if name == "vertices_only":
            # The arc reaches x = +-2.5 (circle centre z = 0.5, radius 2.5).
            assert chart.unbounded
            np.testing.assert_allclose(chart.box, ((-0.125, 0.0), (1.125, 1.0)), atol=1.0e-12)
        else:
            assert not chart.unbounded
        results[name] = [
            qualified_curve_face(model, model.handle("edge", eid), model.handle("face", face))
            for eid in cylinder_edges
        ]
    for result, expected in zip(results["vertices_only"], results["covering"]):
        assert result.kind is expected.kind
        _assert_same_points(_points(result), _points(expected))
    assert sum(item.kind is IntersectionKind.CROSS for item in results["covering"]) == 4


def test_coplanar_plates_with_partial_frames_overlap_in_a_region():
    """The coincidence check samples each unbounded Plane over its face, unclipped."""

    model = GeometryModel()
    first = model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))))
    second = model.add_plate(model.add_points(((1, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0))))
    model.set_face_surface(first, Plane((0, 0, 0), (0.25, 0, 0), (0, 0.25, 0)))
    model.set_face_surface(second, Plane((5, 3, 0), (0.5, 0, 0), (0, 0.5, 0)))
    result = qualified_face_face(model, model.handle("face", first), model.handle("face", second))
    assert result.kind is IntersectionKind.OVERLAP_REGION, result.diagnostics
    assert len(result.components) == 1
    _assert_same_points(
        [np.asarray(point) for point in result.components[0].boundary_paths[0]],
        [np.array(point, dtype=float) for point in ((1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0))],
    )


def test_mixed_covering_and_partial_coplanar_frames_overlap_in_a_region():
    """Codex S0g-R1 finding 5: the covering target was still projected clipped."""

    for order in (0, 1):
        model = GeometryModel()
        first = model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))))
        second = model.add_plate(model.add_points(((1, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0))))
        model.set_face_surface(first, Plane((0, 0, 0), (2, 0, 0), (0, 1, 0)))
        model.set_face_surface(second, Plane((5, 3, 0), (0.5, 0, 0), (0, 0.5, 0)))
        handles = [model.handle("face", first), model.handle("face", second)]
        result = qualified_face_face(model, *(handles[::-1] if order else handles))
        assert result.kind is IntersectionKind.OVERLAP_REGION, (order, result.diagnostics)
        _assert_same_points(
            [np.asarray(point) for point in result.components[0].boundary_paths[0]],
            [np.array(point, dtype=float) for point in ((1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0))],
        )


def _small_cylinder_large_plate(frame):
    model = cylinder(0.1, 1.0, circumferential_segments=4, origin=(0.7, 0.7, 0.0))
    plate = model.add_plate(model.add_points(((-10, -10, 0.5), (10, -10, 0.5), (10, 10, 0.5), (-10, 10, 0.5))))
    model.set_face_surface(plate, Plane(*frame))
    return model, plate


def test_face_face_fallback_on_unbounded_plane_does_not_depend_on_frame_scale():
    """Codex S0g-R1 finding 1: refinement ran in the stored frame's scale.

    The unbounded Plane is refined in its face-box chart, so a remote tiny
    frame gives the result of a frame spanning the face (the sampled
    fallback's completeness itself is a separate, pre-existing limit).
    """

    reference, reference_plate = _small_cylinder_large_plate(((-10, -10, 0.5), (20, 0, 0), (0, 20, 0)))
    model, plate = _small_cylinder_large_plate(((30, 30, 0.5), (0.1, 0, 0), (0, 0.1, 0)))
    for fid in _cylinder_faces(model, plate):
        for swap in (False, True):
            pair = (plate, fid)[::-1] if swap else (plate, fid)
            reference_pair = (reference_plate, fid)[::-1] if swap else (reference_plate, fid)
            expected = query_intersection(reference, *(reference.handle("face", item) for item in reference_pair))
            result = query_intersection(model, *(model.handle("face", item) for item in pair))
            assert expected.kind is IntersectionKind.CROSS
            assert result.kind is expected.kind, (fid, swap, result.diagnostics)


def _spline_topped_face(model):
    a, b, c, control, d = model.add_points(((0, 0, 0), (1, 0, 0), (1, 0.2, 0), (0.5, 1.7, 0), (0, 0.2, 0)))
    edges = (model.add_line(a, b), model.add_line(b, c), model.add_spline(c, (control,), d), model.add_line(d, a))
    face = model.add_face(edges)
    model.set_face_surface(face, Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    return face


def test_spline_trimmed_face_inside_its_frame_keeps_the_bounded_chart():
    """Codex S0g-R1 finding 4: the control polygon (y up to 1.7) is not the curve (y up to 0.95)."""

    from anygeometry.curved_intersections import _plane_chart

    model = GeometryModel()
    face = _spline_topped_face(model)
    assert model.validate_topology() == ()
    assert not _plane_chart(model, face).unbounded


def _rotated_line_range_face(model):
    """Codex S0g-R2: Plane P(u, v) = (u - v, u + v, 0) whose unit square is the face, LineRange trims."""

    from anygeometry.line_ranges import LineRange

    corners = ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 2.0, 0.0), (-1.0, 1.0, 0.0))
    vertices = model.add_points(corners)
    edges = tuple(
        model.add_curve(vertices[index], vertices[(index + 1) % 4], LineRange(corners[index], corners[(index + 1) % 4]))
        for index in range(4)
    )
    face = model.add_face(edges)
    model.set_face_surface(face, Plane((0, 0, 0), (1, 1, 0), (-1, 1, 0)))
    return face


def test_exact_trims_on_a_rotated_covering_frame_keep_the_bounded_chart():
    """Codex S0g-R2: world boxes mapped to uv lose the coordinate correlation."""

    from anygeometry.curved_intersections import _plane_chart

    model = GeometryModel()
    face = _rotated_line_range_face(model)
    assert model.validate_topology() == ()
    assert not _plane_chart(model, face).unbounded


def _far_line_range_face(origin_x):
    """Codex S0g-R3 P1: a LineRange rectangle one ulp left of a Plane frame's origin at large x.

    The face is ``[x, O + 0.2] x [0, 1]`` (z = 0) with ``x = nextafter(O, -inf)``;
    the frame is ``(O, 0, 0) / (0.3, 0, 0) / (0, 1, 0)``, so the exact support u
    of the left edge is ``(x - O) / 0.3 < 0``.
    """

    from anygeometry.line_ranges import LineRange

    left = float(np.nextafter(origin_x, -np.inf))
    corners = ((left, 0.0, 0.0), (origin_x + 0.2, 0.0, 0.0), (origin_x + 0.2, 1.0, 0.0), (left, 1.0, 0.0))
    model = GeometryModel()
    vertices = model.add_points(corners)
    edges = tuple(
        model.add_curve(vertices[index], vertices[(index + 1) % 4], LineRange(corners[index], corners[(index + 1) % 4]))
        for index in range(4)
    )
    face = model.add_face(edges)
    model.set_face_surface(face, Plane((origin_x, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 1.0, 0.0)))
    # The query arc lies in the plane x = left and crosses z = 0 on the face's left edge.
    edge = model.add_arc(*model.add_points(((left, 0.4, -0.1), (left, 0.5, 0.0), (left, 0.4, 0.1))))
    return model, face, edge, left


def _exact_left_u(origin_x, left):
    from fractions import Fraction

    return (Fraction(left) - Fraction(origin_x)) / Fraction(0.3)


def test_rounding_does_not_prove_coverage_of_a_far_face():
    """Codex S0g-R3 P1: at about 1e8 m, rounded transformed bounds proved coverage and the touch came back DISJOINT."""

    from anygeometry.curved_intersections import _plane_chart

    origin_x = 100000000.2
    model, face, edge, left = _far_line_range_face(origin_x)
    assert model.validate_topology() == ()
    slack = 1.0e-3 * model.tolerance.length / 0.3
    assert _exact_left_u(origin_x, left) < -slack  # the frame does not cover the face
    assert _plane_chart(model, face).unbounded
    result = query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    assert result.kind is not IntersectionKind.DISJOINT, result.diagnostics


@pytest.mark.parametrize("exponent", range(2, 13))
def test_coverage_decision_is_fail_closed_across_coordinate_scales(exponent):
    """Scale sweep 1e2 .. 1e12 m: never a covering claim the exact values refute, never a DISJOINT touch."""

    from anygeometry.curved_intersections import _plane_chart
    from anygeometry.errors import GeometryError

    origin_x = 10.0 ** exponent + 0.2
    model, face, edge, left = _far_line_range_face(origin_x)
    slack = 1.0e-3 * model.tolerance.length / 0.3
    chart = _plane_chart(model, face)
    if _exact_left_u(origin_x, left) < -slack:
        assert chart.unbounded
    try:
        result = query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    except GeometryError:
        return  # a typed refusal is fail-closed
    assert result.kind is not IntersectionKind.DISJOINT, (exponent, result.diagnostics)


def _codex_r4_arc_face():
    """Codex S0g-R4: nearly collinear arc points, exact circle radius about 327 km, frame about 650 km."""

    model = GeometryModel()
    a, b, c, d = model.add_points(((0.1, 0.3, 0.0), (2.2, 2.40001, 0.0), (1.1, 1.3, 0.0), (0.6, 0.8, 0.0)))
    face = model.add_face((model.add_arc(a, b, c), model.add_line(c, d), model.add_line(d, a)))
    plane = Plane(
        (-557686.5956591003, -95683.19565212693, 0.0), (653371.1913112272, 0.0, 0.0), (0.0, 653371.1913112272, 0.0)
    )
    model.set_face_surface(face, plane)
    return model, face, plane


def test_arc_construction_error_does_not_prove_coverage():
    """Codex S0g-R4 P2: the exact circle through the stored arc points leaves the frame by 2.75e-12."""

    from fractions import Fraction as Q

    from anygeometry.curved_intersections import _plane_chart, _plane_face_uv_extent

    model, face, plane = _codex_r4_arc_face()
    assert model.validate_topology() == ()
    slack = 1.0e-3 * model.tolerance.length / np.array((np.linalg.norm(plane.u_vector), np.linalg.norm(plane.v_vector)))
    # Exact circumcentre (h, k) of the stored arc points: the circle crosses u = 1 + slack.
    x, y = Q(0.1), Q(0.3)
    bx, by = Q(2.2) - x, Q(2.40001) - y
    cx, cy = Q(1.1) - x, Q(1.3) - y
    big_b, big_c, big_d = bx * bx + by * by, cx * cx + cy * cy, 2 * (bx * cy - by * cx)
    h, k = x + (big_b * cy - big_c * by) / big_d, y + (bx * big_c - cx * big_b) / big_d
    right = Q(float(plane.origin[0])) + Q(float(plane.u_vector[0])) * (1 + Q(float(slack[0])))
    assert h < right and (h - x) ** 2 + (k - y) ** 2 > (right - h) ** 2
    assert not _plane_face_uv_extent(model, face, plane, slack)[1]
    assert _plane_chart(model, face).unbounded


def _quarter_arc_face(frame):
    """Line (0,0)-(2,0), arc (2,0) via (1+sqrt2, 1) to (2,2) about (1,1), line back to the origin."""

    model = GeometryModel()
    a, b, via, c = model.add_points(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0 + np.sqrt(2.0), 1.0, 0.0), (2.0, 2.0, 0.0)))
    face = model.add_face((model.add_line(a, b), model.add_arc(b, via, c), model.add_line(c, a)))
    model.set_face_surface(face, Plane(*frame))
    return model, face


def test_arc_on_a_huge_covering_frame_declines_to_the_unbounded_path():
    """A well-covered arc on a frame above 1e5 m is declined (fail closed); the crossing is still found."""

    from anygeometry.curved_intersections import _plane_chart

    results = {}
    for name, frame in (
        ("huge", ((-1.0e5, -1.0e5, 0.0), (3.0e5, 0.0, 0.0), (0.0, 3.0e5, 0.0))),
        ("covering", ((-1.0, -1.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0))),
    ):
        model, face = _quarter_arc_face(frame)
        assert model.validate_topology() == ()
        assert _plane_chart(model, face).unbounded is (name == "huge")
        edge = model.add_arc(*model.add_points(((1.2, 0.4, -0.5), (1.2, 0.6, 0.0), (1.2, 0.4, 0.5))))
        results[name] = query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    assert results["covering"].kind is IntersectionKind.CROSS, results["covering"].diagnostics
    assert results["huge"].kind is results["covering"].kind
    _assert_same_points(_points(results["huge"]), _points(results["covering"]))


def test_arc_exactly_touching_its_frame_at_tripod_scale_keeps_the_bounded_chart():
    """A semicircle (radius 10 m, centre 20/30 m) whose frame is exactly its bounding box still proves coverage."""

    from anygeometry.curved_intersections import _plane_chart

    model = GeometryModel()
    a, top, c, middle = model.add_points(((30.0, 30.0, 0.0), (20.0, 40.0, 0.0), (10.0, 30.0, 0.0), (20.0, 30.0, 0.0)))
    face = model.add_face((model.add_arc(a, top, c), model.add_line(c, middle), model.add_line(middle, a)))
    model.set_face_surface(face, Plane((10.0, 30.0, 0.0), (20.0, 0.0, 0.0), (0.0, 10.0, 0.0)))
    assert model.validate_topology() == ()
    assert not _plane_chart(model, face).unbounded


# ---------------------------------------------------------------- bit identity

_UUID = re.compile(r"UUID\('[0-9a-f-]+'\)")


def _digest(result) -> str:
    return hashlib.sha256(_UUID.sub("UUID", repr(result)).encode()).hexdigest()[:16]


def sample_digests() -> dict[str, str]:
    """Results that must not change: Cylinder supports and covering Plane frames."""

    digests: dict[str, str] = {}
    # Pure cylinder model: every edge against every face, every face pair.
    model = cylinder(1.0, 1.0, circumferential_segments=4)
    faces = sorted(model.faces)
    for eid in sorted(model.edges):
        for fid in faces:
            digests[f"cyl/e{eid}/f{fid}"] = _digest(
                qualified_curve_face(model, model.handle("edge", eid), model.handle("face", fid))
            )
    for index, fid in enumerate(faces):
        for other in faces[index + 1:]:
            digests[f"cyl/f{fid}/f{other}"] = _digest(
                qualified_face_face(model, model.handle("face", fid), model.handle("face", other))
            )
    # Probe model with covering Plane frames.
    for name, frame in COVERING_SAMPLE.items():
        model, face = _probe_model(frame)
        for eid in sorted(model.edges):
            for fid in sorted(model.faces):
                digests[f"{name}/e{eid}/f{fid}"] = _digest(
                    qualified_curve_face(model, model.handle("edge", eid), model.handle("face", fid))
                )
            digests[f"{name}/q/e{eid}"] = _digest(
                query_intersection(model, model.handle("edge", eid), model.handle("face", face))
            )
        for fid in _cylinder_faces(model, face):
            for pair in ((fid, face), (face, fid)):
                digests[f"{name}/f{pair[0]}/f{pair[1]}"] = _digest(
                    qualified_face_face(model, model.handle("face", pair[0]), model.handle("face", pair[1]))
                )
            digests[f"{name}/q/f{fid}"] = _digest(
                query_intersection(model, model.handle("face", fid), model.handle("face", face))
            )
    model, first, second = _codex_plates(((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    for pair in ((first, second), (second, first)):
        digests[f"plates/f{pair[0]}/f{pair[1]}"] = _digest(
            qualified_face_face(model, model.handle("face", pair[0]), model.handle("face", pair[1]))
        )
    # Codex S0g-R1 finding 4: spline-trimmed face inside its frame, curved queries.
    model = GeometryModel()
    face = _spline_topped_face(model)
    for index, (start, via, end) in enumerate((
        ((0.2, 1.5, 0.0), (0.5, 1.5, 0.3), (0.8, 1.5, 0.0)),
        ((0.2, 0.5, -0.2), (0.5, 0.5, 0.1), (0.8, 0.5, -0.2)),
        ((0.3, 0.9, -0.2), (0.5, 0.9, 0.1), (0.7, 0.9, -0.2)),
    )):
        edge = model.add_arc(*model.add_points((start, via, end)))
        digests[f"spline/e{index}"] = _digest(
            query_intersection(model, model.handle("edge", edge), model.handle("face", face))
        )
    # Codex S0g-R2: LineRange trims on a rotated covering frame.
    model = GeometryModel()
    face = _rotated_line_range_face(model)
    for index, (start, via, end) in enumerate((
        ((1.0, 1.4, 0.0), (0.7, 1.7, 0.3), (0.4, 2.0, 0.0)),
        ((0.3, 0.9, -0.2), (0.0, 1.0, 0.1), (-0.3, 1.1, -0.2)),
    )):
        edge = model.add_arc(*model.add_points((start, via, end)))
        digests[f"linerange/e{index}"] = _digest(
            query_intersection(model, model.handle("edge", edge), model.handle("face", face))
        )
    # Small cylinder against a large plate whose frame covers it.
    model, plate = _small_cylinder_large_plate(((-10, -10, 0.5), (20, 0, 0), (0, 20, 0)))
    for fid in _cylinder_faces(model, plate):
        for pair in ((plate, fid), (fid, plate)):
            digests[f"large/f{pair[0]}/f{pair[1]}"] = _digest(
                query_intersection(model, model.handle("face", pair[0]), model.handle("face", pair[1]))
            )
    return digests


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason=(
        "the stored e66cbee digests were captured on one Windows CPU; bit-exact floating results of the "
        "curved-query path depend on the CPU's BLAS kernel (measured 2026-10-30: 41 of 393 keys differ on "
        "Linux; changing OPENBLAS_CORETYPE alone changes 41-43 keys; one key flips CONTAINED/DISJOINT "
        "between kernels), so the test is valid only on the reference platform"
    ),
)
def test_cylinder_and_covering_plane_results_are_bit_identical_to_e66cbee():
    expected = json.loads(DATA.read_text(encoding="utf-8"))
    actual = sample_digests()
    assert sorted(actual) == sorted(expected)
    changed = sorted(key for key in expected if actual[key] != expected[key])
    assert changed == []
