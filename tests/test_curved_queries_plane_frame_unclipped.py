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
    return digests


def test_cylinder_and_covering_plane_results_are_bit_identical_to_e66cbee():
    expected = json.loads(DATA.read_text(encoding="utf-8"))
    actual = sample_digests()
    assert sorted(actual) == sorted(expected)
    changed = sorted(key for key in expected if actual[key] != expected[key])
    assert changed == []
