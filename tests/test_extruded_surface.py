"""A planar directrix (Bezier or ellipse) swept along a vector, as an explicit surface.

The profile is planar, so the extrusion coordinate is linear in position and the directrix parameter of a
surface point is that of its projection along the vector: evaluation and inversion are exact to rounding,
patches cut from one surface share one directrix and one vector, and the surface refuses what it cannot
represent (a non-planar or degenerate directrix, a vector lying in the profile plane).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import BezierDirectrix, EllipseDirectrix, ExtrudedSurface, GeometryError
from anygeometry.surfaces import _evaluate_surface_many, _surface_derivatives_many, surface_normal

CUBIC = ((0., 0., 0.), (1., .8, 0.), (2., -.4, 0.), (3., 0., 0.))


def _tilted(points, normal=(0., .6, .8)):
    """Rotate the z = 0 profile so that its plane has the given unit normal."""
    n = np.asarray(normal, float)
    e1 = np.cross(n, (1., 0., 0.))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    matrix = np.column_stack((e1, e2, n))
    return tuple(tuple(matrix @ np.asarray(p, float) + (.5, -1., 2.)) for p in points)


SURFACES = {
    "cubic, vertical": ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.)),
    "cubic, oblique vector": ExtrudedSurface(BezierDirectrix(CUBIC), (.3, .2, 1.)),
    "cubic, tilted profile plane": ExtrudedSurface(BezierDirectrix(_tilted(CUBIC)), (.4, -.5, 1.5)),
    "quartic": ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1., 0.), (2., -1., 0.), (3., 1., 0.), (4., 0., 0.))), (0., 0., 1.)),
    "parabola": ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1., 0.), (2., 0., 0.))), (0., 0., 3.)),
    "elliptic arc": ExtrudedSurface(EllipseDirectrix((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), .2, 3.), (0., 0., 2.)),
    "elliptic arc, oblique": ExtrudedSurface(EllipseDirectrix((0., 0., 0.), (2., 0., 0.), (0., 1., 0.), 0., 4.), (.5, .2, 1.)),
    "full ellipse": ExtrudedSurface(EllipseDirectrix((1., 2., 3.), (1., .5, 0.), (-.5, 1., 0.), 0., math.tau), (0., 0., 1.)),
}


@pytest.mark.parametrize("name", SURFACES)
def test_points_invert_to_their_parameters_exactly(name):
    surface = SURFACES[name]
    rng = np.random.default_rng(4)
    uv = rng.uniform(.02, .98, (120, 2))
    points = np.asarray([surface.evaluate(u, v) for u, v in uv])
    assert np.allclose(_evaluate_surface_many(surface, uv), points, atol=1e-14)
    assert np.abs(surface.local_uv_many(points) - uv).max() < 1e-12
    single = np.asarray([surface.local_uv(p) for p in points[:15]])
    assert np.array_equal(single, surface.local_uv_many(points[:15]))


@pytest.mark.parametrize("name", SURFACES)
def test_derivatives_match_finite_differences_and_the_normal_is_unit(name):
    surface = SURFACES[name]
    uv = np.random.default_rng(2).uniform(.1, .9, (12, 2))
    du, dv = _surface_derivatives_many(surface, uv)
    h = 1e-6
    numeric = (_evaluate_surface_many(surface, uv + [h, 0.]) - _evaluate_surface_many(surface, uv - [h, 0.])) / (2 * h)
    assert np.abs(du - numeric).max() < 1e-7 * max(1., np.abs(du).max())
    assert np.allclose(dv, np.asarray(surface.vector))
    assert np.linalg.norm(surface_normal(surface, .4, .6)) == pytest.approx(1.)


def test_the_extrusion_coordinate_is_linear_in_position():
    """Level lines of ``v`` are planes parallel to the profile plane, like a Cylinder's rings."""
    surface = SURFACES["cubic, tilted profile plane"]
    rng = np.random.default_rng(8)
    for u in rng.uniform(0, 1, 6):
        for v in (0., .3, 1.):
            point = surface.evaluate(u, v)
            height = float((point - surface.profile_origin) @ surface.profile_normal)
            assert height == pytest.approx(v * surface.profile_rate, abs=1e-12)
            assert surface.local_uv(point)[1] == pytest.approx(v, abs=1e-12)


def test_a_patch_shares_its_surface_and_composes_exactly():
    surface = SURFACES["cubic, oblique vector"]
    sub = surface.subpatch((.25, .75), (.5, 1.))
    again = sub.subpatch((.5, 1.), (0., .5))
    assert sub.support_key() == surface.support_key() == again.support_key()    # one surface, many patches
    assert sub != surface and sub.evaluate(.3, .4) == pytest.approx(surface.evaluate(.25 + .3 * .5, .5 + .4 * .5), abs=1e-15)
    assert again.u_range == pytest.approx((.5, .75)) and again.v_range == pytest.approx((.5, .75))
    assert np.allclose(sub.local_uv(sub.evaluate(.3, .4)), (.3, .4), atol=1e-12)


@pytest.mark.parametrize("name", ("cubic, oblique vector", "elliptic arc, oblique"))
def test_an_affine_image_is_the_same_surface_moved(name):
    surface = SURFACES[name]
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0., -1., .2], [1., 0., 0.], [0., .3, 2.]])
    matrix[:3, 3] = (1., 2., 3.)
    moved = surface.transformed(matrix)
    for u, v in ((.1, .2), (.5, .5), (.9, .3)):
        assert np.allclose(moved.evaluate(u, v), matrix[:3, :3] @ surface.evaluate(u, v) + matrix[:3, 3], atol=1e-13)
    assert np.allclose(moved.local_uv(moved.evaluate(.35, .65)), (.35, .65), atol=1e-11)


def test_bounds_enclose_every_point_of_the_patch():
    for surface in SURFACES.values():
        lower, upper = surface.bounds()
        uv = np.random.default_rng(6).uniform(0, 1, (200, 2))
        points = _evaluate_surface_many(surface, uv)
        assert np.all(points >= lower - 1e-12) and np.all(points <= upper + 1e-12)


def test_equal_surfaces_hash_alike_and_different_ones_do_not_compare_equal():
    first = ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.))
    assert first == ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.)) and hash(first) == hash(
        ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.)))
    assert first != ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.5))
    assert first != ExtrudedSurface(BezierDirectrix(CUBIC[::-1]), (0., 0., 2.))
    assert first != "surface"


@pytest.mark.parametrize("make,message", [
    (lambda: BezierDirectrix(((0., 0., 0.), (1., 1., 1.), (2., 0., 0.), (3., 1., 2.))), "planar"),
    (lambda: BezierDirectrix(((0., 0., 0.), (1., 0., 0.), (2., 0., 0.))), "line"),
    (lambda: BezierDirectrix(((0., 0., 0.), (1., 0., 0.))), "three"),
    (lambda: EllipseDirectrix((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)), "independent"),
    (lambda: EllipseDirectrix((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 0.), "sweep"),
    (lambda: ExtrudedSurface(BezierDirectrix(CUBIC), (1., 0., 0.)), "leave the profile plane"),
    (lambda: ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 0.)), "leave the profile plane"),
    (lambda: ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 1.), u_range=(.5, .5)), "u_range"),
    (lambda: ExtrudedSurface("not a directrix", (0., 0., 1.)), "BezierDirectrix"),
])
def test_what_cannot_be_represented_is_refused(make, message):
    with pytest.raises(GeometryError, match=message):
        make()


# ---------------------------------------------------------------- inside a model

import json

from anygeometry import GeometryModel, from_dict, to_dict
from anygeometry.editing import copy_entities, reverse_face
from anygeometry.operations import transform


def _extruded_face(vector=(0., 0., 2.), extra_surface=True):
    model = GeometryModel()
    points = model.add_points(CUBIC)
    edge = model.add_spline(points[0], (points[1], points[2]), points[3])
    (face,) = model.extrude([edge], vector)
    surface = ExtrudedSurface(BezierDirectrix(CUBIC), vector)
    if extra_surface:
        model.set_face_surface(face, surface)
    return model, face, surface


def test_a_face_can_carry_an_extruded_support_and_validates():
    model, face, surface = _extruded_face()
    assert model.validate_topology() == ()
    assert model.faces[face].surface == surface
    probe = model.face_point(face, .3, .6)
    assert np.allclose(probe, surface.evaluate(.3, .6), atol=1e-12)


def test_the_document_is_schema_six_and_older_schemas_refuse_the_surface():
    import hashlib
    model, face, surface = _extruded_face()
    document = to_dict(model)
    assert document["version"] == 6
    restored = from_dict(json.loads(json.dumps(document)))
    assert restored.faces[face].surface == surface and to_dict(restored) == document and restored.validate_topology() == ()
    plain, _face, _surface = _extruded_face(extra_surface=False)
    assert to_dict(plain)["version"] == 5                               # stock Coons extrusions keep schema 5
    stale = json.loads(json.dumps(document))
    stale["version"] = 5
    body = {k: v for k, v in stale.items() if k != "checksum"}
    stale["checksum"] = {"algorithm": "sha256", "value": hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
    with pytest.raises(GeometryError, match="require schema 6"):
        from_dict(stale)


@pytest.mark.parametrize("damage", [
    lambda s: s.pop("vector"),
    lambda s: s.update(extra=1),
    lambda s: s["directrix"].update(kind="spiral"),
    lambda s: s["directrix"].pop("controls"),
    lambda s: s.update(u_range=[.5, .5]),
    lambda s: s.update(vector=[1., 0., 0.]),
])
def test_malformed_extruded_records_are_refused_whole(damage):
    import hashlib
    model, face, _surface = _extruded_face()
    document = json.loads(json.dumps(to_dict(model)))
    record = next(f for f in document["faces"] if f["id"] == face)["surface"]
    damage(record)
    body = {k: v for k, v in document.items() if k != "checksum"}
    document["checksum"] = {"algorithm": "sha256", "value": hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
    with pytest.raises(GeometryError):
        from_dict(document)


def test_transforms_reversal_and_copies_keep_the_support_exact():
    model, face, surface = _extruded_face((.3, .2, 1.))
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 2.]])
    matrix[:3, 3] = (1., 2., 3.)
    expected = matrix[:3, :3] @ surface.evaluate(.4, .6) + matrix[:3, 3]
    reverse_face(model, face)
    assert np.allclose(model.faces[face].surface.evaluate(.6, .6), surface.evaluate(.4, .6), atol=1e-13)   # u runs backwards
    reverse_face(model, face)
    transform(model, matrix)
    assert model.validate_topology() == ()
    assert np.allclose(model.face_point(face, .4, .6), expected, atol=1e-12)
    other = GeometryModel()
    other.insert_model(model)
    assert other.validate_topology() == () and len(other.faces) == 1
    assert np.allclose(next(iter(other.faces.values())).surface.evaluate(.4, .6), expected, atol=1e-12)


def test_batch_inversion_through_the_model_matches_the_surface():
    model, face, surface = _extruded_face()
    uv = np.random.default_rng(3).uniform(.05, .95, (30, 2))
    points = np.asarray([surface.evaluate(u, v) for u, v in uv])
    assert np.abs(model.face_local_uv_many(face, points) - uv).max() < 1e-12


def test_the_face_bounds_cover_the_whole_support_patch():
    model, face, surface = _extruded_face()
    lower, upper = surface.bounds()
    box = model._entity_bounds(("face", face))
    assert np.all(np.asarray(box[:3]) <= lower + 1e-12) and np.all(np.asarray(box[3:]) >= upper - 1e-12)
