"""The B1 driving case: a closed curved-bow half-hull between two frames (generality slice B1-P5).

``build_b1_half_hull`` makes six faces (the bow piece bounded by four plane x NURBS section curves,
deck, bottom, two frames, centreline) sharing twelve edges. The model must audit clean; a plane face
cutting through it without shared topology is reported.
"""
from __future__ import annotations

import pytest

from anygeometry import Plane
from anygeometry.b1_example import build_b1_half_hull


@pytest.fixture(scope="module")
def b1():
    return build_b1_half_hull()


def test_every_edge_bounds_exactly_two_faces(b1):
    model, faces = b1
    assert len(faces) == 6
    uses: dict = {}
    for face in model.faces.values():
        for loop in (face.loop,) + tuple(face.holes):
            for item in loop:
                uses.setdefault(item.edge, []).append(face.id)
    assert len(uses) == 12 and all(len(owners) == 2 for owners in uses.values())
    assert len(model.vertices) == 8                              # V - E + F = 8 - 12 + 6 = 2


def test_the_b1_half_hull_audits_clean(b1):
    model, _faces = b1
    report = model.strict_audit()
    assert report.clean, [(issue.code, issue.details) for issue in report.issues]


def test_a_plane_cutting_through_the_hull_is_reported():
    model, _faces = build_b1_half_hull()
    level = Plane((0., 0., 2.), (1., 0., 0.), (0., 1., 0.))
    corners = [model.add_point(x, y, 2.) for x, y in ((2., 1.), (3., 1.), (3., 9.), (2., 9.))]
    model.add_face([model.add_line(corners[i], corners[(i + 1) % 4]) for i in range(4)], surface=level)
    assert not model.strict_audit().clean


# ------------------------------------------------------------------ B1-P5c: closed shell, volume, centroid


def reference_volume_and_centroid(frames=(1.5, 3.5), bottom=.5, deck=3., order=80):
    """Independent: z = 4 v; for each Gauss v, the frame limits u_a, u_b by Newton on x(u, v), then
    int y dx over [u_a, u_b] by Gauss in u (dx = x_u du); moments likewise."""
    import numpy as np
    from anygeometry.b1_example import b1_bow_surface
    surface = b1_bow_surface()
    nodes, weights = np.polynomial.legendre.leggauss(order)

    def solve_u(target, v, u):
        for _ in range(60):
            point = surface.evaluate_many(np.array([[u, v]]))[0]
            s_u, _s_v = surface.derivatives_many(np.array([[u, v]]))
            step = (point[0] - target) / s_u[0][0]
            u -= step
            if abs(step) < 1e-15:
                break
        return u
    v0, v1 = bottom / 4, deck / 4
    totals = np.zeros(4)
    for v, wv in zip(.5 * (v1 - v0) * (nodes + 1) + v0, .5 * (v1 - v0) * weights):
        ua, ub = solve_u(frames[0], v, .3), solve_u(frames[1], v, .5)
        u = .5 * (ub - ua) * (nodes + 1) + ua
        uv = np.stack((u, np.full_like(u, v)), axis=1)
        points = surface.evaluate_many(uv)
        s_u, _s_v = surface.derivatives_many(uv)
        w = .5 * (ub - ua) * weights * s_u[:, 0] * 4 * wv
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        totals += np.array([w @ y, w @ (x * y), w @ (.5 * y * y), w @ (z * y)])
    return totals[0], totals[1:] / totals[0]


def test_the_b1_shell_is_closed_and_consistently_oriented(b1):
    from anygeometry.shell_volume import closed_shell
    model, faces = b1
    assert closed_shell(model, faces.values()) == []


def test_volume_and_centroid_match_an_independent_reference(b1):
    import numpy as np
    from anygeometry.shell_volume import enclosed_volume
    model, faces = b1
    result = enclosed_volume(model, list(faces.values()))
    volume, centroid = reference_volume_and_centroid()
    assert abs(result["volume"] - volume) <= 1e-9 * volume
    assert np.all(np.abs(result["centroid"] - centroid) <= 1e-9 * np.abs(centroid))
    assert result["estimate"] <= 1e-9 * volume


def test_a_shell_missing_a_face_is_refused(b1):
    from anygeometry import GeometryError
    from anygeometry.shell_volume import closed_shell, enclosed_volume
    model, faces = b1
    partial = [face for name, face in faces.items() if name != "deck"]
    assert any("bounds 1 faces" in problem for problem in closed_shell(model, partial))
    with pytest.raises(GeometryError, match="closed shell"):
        enclosed_volume(model, partial)


def test_a_flipped_face_is_inconsistent():
    from dataclasses import replace
    from anygeometry.entities import OrientedEdge
    from anygeometry.shell_volume import closed_shell
    model, faces = build_b1_half_hull()
    deck = model.faces[faces["deck"]]
    flipped = tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(deck.loop))
    with model.transaction():
        model._put_entity("face", replace(deck, loop=flipped, corners=()))
    assert any("inconsistent orientation" in problem for problem in closed_shell(model, faces.values()))


# ------------------------------------------------------------------ review B1-P5 round 1


def test_plane_faces_crossing_between_shared_vertices_are_reported():
    from anygeometry import GeometryModel
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    model = GeometryModel()
    a, b = model.add_point(-1., 0., 0.), model.add_point(1., 0., 0.)
    top, bottom = model.add_point(0., 1., 0.), model.add_point(0., -1., 0.)
    strip = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                         (((-1., 0., -.5), (-1., 0., .5)), ((0., 1., -.5), (0., 1., .5))))
    flat = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    side = PlaneNurbsCurve.from_plane(flat, strip, 0)
    if side.evaluate(0.)[1] > .5:
        side = side.subcurve(1., 0.)
    model.add_face([model.add_curve(a, top, side), model.add_line(top, b), model.add_line(b, bottom),
                    model.add_line(bottom, a)], surface=flat)
    up, down = model.add_point(0., 0., 1.), model.add_point(0., 0., -1.)
    model.add_face([model.add_line(a, up), model.add_line(up, b), model.add_line(b, down), model.add_line(down, a)],
                   surface=Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.)))
    assert not model.strict_audit().clean


def test_a_crossing_beyond_the_joint_ball_is_not_a_joint():
    from anygeometry.branch_predicates import certify_line_section
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    # C(t) = (t, 0.01 t - 5e-5, 0) crosses the segment (0,0,0) -> (1,0,0) at x = 0.005 (50 tolerances out)
    strip = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                         (((0., -5e-5, -1.), (0., -5e-5, 1.)), ((1., .01 - 5e-5, -1.), (1., .01 - 5e-5, 1.))))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), strip, 0)
    if curve.evaluate(0.)[0] > .5:
        curve = curve.subcurve(1., 0.)
    origin = (0., 0., 0.)
    assert certify_line_section(curve, origin, (1., 0., 0.), [origin], 1e-4) is not True


def test_volume_quadrature_splits_at_knots():
    from anygeometry import GeometryModel
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.shell_volume import _nurbs_face_moments
    # a roof z = 1 + b(u) with a triangular bump of height 1 on [0.499, 0.501]: int z dx dy = 1.001
    knots_u = (0., 0., .499, .5, .501, 1., 1.)
    heights = (1., 1., 2., 1., 1.)
    xs = (0., .499, .5, .501, 1.)
    roof = NurbsSurface(1, 1, knots_u, (0., 0., 1., 1.),
                        tuple(((x, 0., z), (x, 1., z)) for x, z in zip(xs, heights)))
    model = GeometryModel()
    corners = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    vertices = [model.add_point(*map(float, roof.exact_point(*c))) for c in corners]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4],
                             CurveOnSurface(roof, NurbsCurve(1, (0., 0., 1., 1.), (corners[i], corners[(i + 1) % 4]))))
             for i in range(4)]
    face = model.add_face(edges, surface=roof)
    value = abs(_nurbs_face_moments(model, model.faces[face], 24)[0])
    assert abs(value - 1.001) <= 1e-12


def test_disconnected_shells_are_refused():
    from anygeometry import GeometryModel
    from anygeometry.shell_volume import closed_shell
    model = GeometryModel()
    faces = []
    for offset in (0., 4.):
        p = {(i, j, k): model.add_point(offset + i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        lines = {}

        def line(u, v):
            key = frozenset((u, v))
            if key not in lines:
                lines[key] = model.add_line(p[u], p[v])
            return lines[key]
        quads = [((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
                 ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
                 ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))]
        for quad in quads:
            faces.append(model.add_face([line(quad[i], quad[(i + 1) % 4]) for i in range(4)]))
    assert any("not connected" in problem for problem in closed_shell(model, faces))


def test_face_iterators_are_accepted(b1):
    from anygeometry.shell_volume import enclosed_volume
    model, faces = b1
    assert enclosed_volume(model, iter(faces.values()))["volume"] > 0
