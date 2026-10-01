"""Cone x Cylinder and Cone x Plane through ``plan_intersections`` / ``apply_intersections``.

A cone meeting a cylinder at an angle (here 10 degrees off the perpendicular, and others) has an
intersection curve that is not a conic: it is an exact ``QuadricIntersectionCurve`` chart. These tests drive
whole shells through the public workflow and check what a user can see: the topology validates, every
joint edge lies on both supports, material area is conserved exactly (an independent integral of the
split faces agrees with the intact ones), the cone's area is exact, and the document round-trips under the
schema-6 policy (version 6 only when a quadric curve is stored, so ordinary documents keep version 5).
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from anygeometry import (ConnectionIntent, apply_intersections, from_dict, plan_intersections,
                         query_trimmed_surface_charts, to_dict)
from anygeometry.generators import cone, cylinder
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.surfaces import Cone, Cylinder, Plane


def _total_area(model):
    return sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)


def _plate(model, center, normal, size):
    n = np.asarray(normal, float)
    u = np.array([1., 0., 0.])
    v = np.cross(n, u)
    corners = [tuple(np.asarray(center, float) + size * (a * u + b * v)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return model.add_plate(model.add_points(corners))


def _cone_and_plate(tilt_degrees, offset=1.5, *, shift=0., r0=.5, r1=1.5, height=3.):
    model = cone(r0, r1, height, circumferential_segments=6)
    cone_faces = sorted(model.faces)
    t = math.radians(tilt_degrees)
    normal = np.array([0., math.sin(t), math.cos(t)])
    plate = _plate(model, np.array([0., 0., offset]) + shift * normal, normal, 3.0)
    return model, cone_faces + [plate]


def _cone_and_cylinder(off_degrees, *, r0=.5, r1=1., radius=2., axis_offset=0., segments=(8, 6)):
    model = cylinder(radius, 6., origin=(0., 0., -3.), circumferential_segments=segments[0])
    cylinder_faces = set(model.faces)
    a = math.radians(off_degrees)
    model.insert_model(cone(r0, r1, 5., origin=(0., axis_offset, 0.), axis=(math.cos(a), 0., math.sin(a)),
                            radial_direction=(0., 1., 0.), circumferential_segments=segments[1]))
    return model, sorted(cylinder_faces) + sorted(set(model.faces) - cylinder_faces)


def _run(model, faces):
    handles = [model.handle("face", f) for f in faces]
    plan = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    return model


def _joint_residual(model):
    """Largest distance of any sampled exact joint-edge point from the supports of the faces it bounds."""
    owners = collections.defaultdict(set)
    for face_id, face in model.faces.items():
        for oriented in (*face.loop, *(e for hole in face.holes for e in hole)):
            owners[oriented.edge].add(face_id)
    worst, joints = 0., 0
    for edge_id, faces in owners.items():
        curve = model.edges[edge_id].curve
        if type(curve).__name__ not in ("QuadricIntersectionCurve", "EllipticArc", "CylinderIntersectionCurve"):
            continue
        points = curve.evaluate(np.linspace(0., 1., 25))
        joints += 1
        for face_id in faces:
            surface = model.faces[face_id].surface
            if isinstance(surface, (Cone, Cylinder, Plane)):
                quadric = QuadricSupport.from_surface(surface)
                worst = max(worst, float(np.max(np.abs(quadric.value(points)) / quadric.gradient_norm(points))))
    return worst, joints


def _edge_kinds(model):
    return collections.Counter(type(edge.curve).__name__ for edge in model.edges.values())


def _check_model(model, before, *, area_tolerance=1e-12):
    assert model.validate_topology() == ()
    assert abs(_total_area(model) - before) <= area_tolerance * before
    residual, joints = _joint_residual(model)
    assert residual < 1e-9
    return joints


def test_the_cone_area_is_exact():
    r0, r1, h = .5, 1.5, 3.
    model = cone(r0, r1, h, circumferential_segments=8)
    assert _total_area(model) == pytest.approx(math.pi * (r0 + r1) * math.hypot(h, r1 - r0), rel=1e-13)


@pytest.mark.parametrize("tilt,offset,shift,expected", [
    (0., 1.5, 0., {"Arc", "Straight"}),                  # circle: arcs the engine already had
    (10., 1.5, 0., {"EllipticArc"}),                     # ellipse
    (70., 1.5, 0., {"EllipticArc"}),                     # still an ellipse (parallel to a generator at 71.6 degrees)
    (80., 2.0, 0., {"QuadricIntersectionCurve"}),        # hyperbola
    (90., 1.5, .4, {"QuadricIntersectionCurve"}),        # vertical plate beside the axis: hyperbola
])
def test_a_plate_cutting_a_cone_splits_both_faces_and_conserves_material(tilt, offset, shift, expected):
    model, faces = _cone_and_plate(tilt, offset, shift=shift)
    before = _total_area(model)
    _run(model, faces)
    joints = _check_model(model, before)
    kinds = set(_edge_kinds(model))
    assert expected <= kinds
    if "QuadricIntersectionCurve" in expected:
        assert to_dict(model)["version"] == 6
    else:
        assert to_dict(model)["version"] == 5            # no quadric curve stored: the established schema
    if tilt != 0.:
        assert joints >= 2                                # at least both branches / the closed section was imprinted
    back = from_dict(to_dict(model))
    assert back.validate_topology() == ()
    assert _total_area(back) == pytest.approx(_total_area(model), rel=1e-13)


def test_a_ring_cut_rebases_cone_children_into_exact_smaller_patches():
    """A circle splits each facet into two rectangular patches: each becomes its own exact Cone, like a Cylinder."""
    model, faces = _cone_and_plate(0., 1.5)
    before = _total_area(model)
    _run(model, faces)
    _check_model(model, before)
    patches = [face.surface for face in model.faces.values() if isinstance(face.surface, Cone)]
    assert len(patches) == 12
    lower = [c for c in patches if c.origin[2] == 0.]
    upper = [c for c in patches if c.origin[2] == 1.5]
    assert len(lower) == len(upper) == 6
    for c in lower:
        assert (c.radius_start, c.radius_end, c.height) == pytest.approx((.5, 1., 1.5))
    for c in upper:
        assert (c.radius_start, c.radius_end, c.height) == pytest.approx((1., 1.5, 1.5))
    assert sum(c.sweep_angle for c in lower) == pytest.approx(math.tau)
    # the same physical surface: every patch evaluates onto the original cone
    original = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 3., 0., math.tau)
    quadric = QuadricSupport.from_surface(original)
    for c in patches:
        points = np.asarray([c.evaluate(u, v) for u in (0., .3, 1.) for v in (0., .5, 1.)])
        assert np.max(np.abs(quadric.value(points)) / quadric.gradient_norm(points)) < 1e-12


def test_a_beam_crossing_a_cone_meets_it_at_the_exact_points():
    from anygeometry.generators import cone as make_cone
    model = make_cone(.5, 1.5, 3., circumferential_segments=8)
    faces = sorted(model.faces)
    beam = model.add_member((model.add_line(*model.add_points(((-3., .1, 1.5), (3., .1, 1.5)))),))
    handles = [*(model.handle("face", f) for f in faces), model.handle("member", beam)]
    plan = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert model.validate_topology() == ()
    contacts = sorted(tuple(round(float(x), 9) for x in contact.position) for contact in plan.contacts)
    radius = math.sqrt(1. - .1 ** 2)                            # the cone has radius 1 at height 1.5
    assert contacts == [(-round(radius, 9), .1, 1.5), (round(radius, 9), .1, 1.5)]


def test_a_plate_through_the_axis_meets_the_cone_in_its_generators():
    model, faces = _cone_and_plate(90., 1.5)
    before = _total_area(model)
    _run(model, faces)
    _check_model(model, before)
    assert to_dict(model)["version"] == 5


def test_a_cone_hitting_a_cylinder_ten_degrees_off_the_perpendicular():
    """The case that motivated the curve family: not a conic, still exact and sharing its supports."""
    model, faces = _cone_and_cylinder(10.)
    before = _total_area(model)
    _run(model, faces)
    joints = _check_model(model, before)
    kinds = _edge_kinds(model)
    assert joints >= 8 and kinds["QuadricIntersectionCurve"] >= 8
    doc = to_dict(model)
    assert doc["version"] == 6
    back = from_dict(doc)
    assert back.validate_topology() == ()
    assert to_dict(back)["checksum"] == doc["checksum"]


@pytest.mark.parametrize("off_degrees,axis_offset,segments", [
    (0., 0., (8, 6)), (30., 0., (8, 6)), (10., 1.2, (8, 6)),
    (10., 1.5, (12, 8)),            # the cone's rim touches the wall at a facet corner: a tangential contact
    (10., 1.5001, (12, 8)),         # a hair past it: very short arcs next to the corner
    (10., 1.8, (12, 8)),            # the loop folds next to facet seams
    (10., 2.2, (12, 8)),
])
def test_cone_cylinder_variants_including_tangent_and_folding_contacts(off_degrees, axis_offset, segments):
    """Perpendicular, oblique and offset cones, including contacts that tangle the facet arrangement."""
    model, faces = _cone_and_cylinder(off_degrees, axis_offset=axis_offset, segments=segments)
    before = _total_area(model)
    _run(model, faces)
    joints = _check_model(model, before)
    assert joints >= 8
    assert from_dict(to_dict(model)).validate_topology() == ()


@pytest.mark.parametrize("tilt_degrees", (60., 30.))
def test_two_cones_crossing_at_an_angle(tilt_degrees):
    """Both supports are cones: every trace is a branch of a quadric against a quadric."""
    model = cone(.5, 1.5, 3., circumferential_segments=8)
    first = set(model.faces)
    t = math.radians(tilt_degrees)
    model.insert_model(cone(.6, 1.4, 4., origin=(-.2, 0., -.8), axis=(math.sin(t), 0., math.cos(t)),
                            radial_direction=(0., 1., 0.), circumferential_segments=6))
    faces = sorted(first) + sorted(set(model.faces) - first)
    before = _total_area(model)
    _run(model, faces)
    joints = _check_model(model, before)
    assert joints >= 6 and _edge_kinds(model)["QuadricIntersectionCurve"] >= 6
    assert to_dict(model)["version"] == 6
    assert from_dict(to_dict(model)).validate_topology() == ()


def test_a_cone_entering_a_cylinder_sideways_and_coaxial_rings():
    model = cylinder(2., 6., origin=(0., 0., -3.), circumferential_segments=8)
    cylinder_faces = set(model.faces)
    model.insert_model(cone(1., 3., 3., origin=(0., 0., -1.5), circumferential_segments=6))       # coaxial, crosses the wall
    faces = sorted(cylinder_faces) + sorted(set(model.faces) - cylinder_faces)
    before = _total_area(model)
    _run(model, faces)
    _check_model(model, before)
    assert to_dict(model)["version"] == 5                 # rings only: no quadric curve
