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



def _section_side(plane, normal, start, end):
    """The section of ``plane`` through a bilinear strip extruding ``start -> end`` by +-1 along ``normal``."""
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    rows = tuple(tuple(tuple(x + sign * n for x, n in zip(point, normal)) for sign in (-1., 1.))
                 for point in (start, end))
    curve = PlaneNurbsCurve.from_plane(plane, NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.), rows), 0)
    first = curve.evaluate(0.)
    if sum((a - b) ** 2 for a, b in zip(first, start)) > sum((a - b) ** 2 for a, b in zip(first, end)):
        curve = curve.subcurve(1., 0.)
    return curve


def test_faces_meeting_beyond_their_only_shared_vertex_are_reported():
    from anygeometry import GeometryModel
    # review B1-P5 round 2: triangles in z = 0 and y = 0 whose geometric apexes (0,5e-10,0) and (0,0,5e-10)
    # share the vertex (0,0,0); their interiors cross along the x-axis over about +-5e-4, with no shared edge
    model = GeometryModel()
    apex = model.add_point(0., 0., 0.)
    flat = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    left, right = model.add_point(-1., -1e-6, 0.), model.add_point(1., -1e-6, 0.)
    model.add_face([model.add_curve(apex, left, _section_side(flat, (0., 0., 1.), (0., 5e-10, 0.), (-1., -1e-6, 0.))),
                    model.add_line(left, right),
                    model.add_curve(right, apex, _section_side(flat, (0., 0., 1.), (1., -1e-6, 0.), (0., 5e-10, 0.)))],
                   surface=flat)
    upright = Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.))
    left, right = model.add_point(-2., 0., -1e-6), model.add_point(2., 0., -1e-6)
    model.add_face([model.add_curve(apex, left, _section_side(upright, (0., 1., 0.), (0., 0., 5e-10), (-2., 0., -1e-6))),
                    model.add_line(left, right),
                    model.add_curve(right, apex, _section_side(upright, (0., 1., 0.), (2., 0., -1e-6), (0., 0., 5e-10)))],
                   surface=upright)
    assert not model.strict_audit().clean


def test_a_tolerance_coplanar_segment_does_not_make_a_far_contact_a_joint():
    from anygeometry.branch_predicates import certify_line_section
    # review B1-P5 round 2: C(t) = (t, 1e-6 t - 5e-9, 0) against the segment (0,0,5e-5) -> (1,0,9e-5), tol 1e-4:
    # projected contacts about 0.005 (50 tolerances) from the joint
    flat = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    curve = _section_side(flat, (0., 0., 1.), (0., -5e-9, 0.), (1., 1e-6 - 5e-9, 0.))
    p = (0., 0., 5e-5)
    assert certify_line_section(curve, p, (1., 0., 9e-5), [p], 1e-4) is not True


def test_a_flat_crossing_far_from_the_shared_vertex_is_reported_by_the_audit():
    from anygeometry import GeometryModel
    # review B1-P5 round 3: C(t) = (t, 0.01 t - 5e-10, 0) starts at the shared vertex (0,0,0) and crosses the
    # line (0,0,0) -> (1,0,0) at x = 5e-8, 50 default tolerances out (the joint radius scaled by 1/sin(angular))
    model = GeometryModel()
    origin = model.add_point(0., 0., 0.)
    flat = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    model.add_curve(origin, model.add_point(1., .01 - 5e-10, 0.),
                    _section_side(flat, (0., 0., 1.), (0., -5e-10, 0.), (1., .01 - 5e-10, 0.)))
    model.add_line(origin, model.add_point(1., 0., 0.))
    assert not model.strict_audit().clean


def _overlap_past_shared_end(shift=0., low_end=(0., -1., 0.), down_end=(0., 0., -1.)):
    """Triangles in z = 0 and y = 0 sharing A -> B = (-1,0,0) -> (0,0,0) (shifted along x); their sections
    from B start at (9e-10,1e-10,0) and (9e-10,0,1e-10), so the interiors overlap past B along the x-axis."""
    from anygeometry import GeometryModel
    model = GeometryModel()
    a, b = model.add_point(shift - 1., 0., 0.), model.add_point(shift, 0., 0.)
    shared = model.add_line(a, b)
    flat = Plane((shift, 0., 0.), (1., 0., 0.), (0., 1., 0.))
    low = model.add_point(shift + low_end[0], *low_end[1:])
    model.add_face([shared, model.add_curve(b, low, _section_side(flat, (0., 0., 1.), (shift + 9e-10, 1e-10, 0.),
                                                                  (shift + low_end[0], *low_end[1:]))),
                    model.add_line(low, a)], surface=flat)
    upright = Plane((shift, 0., 0.), (1., 0., 0.), (0., 0., 1.))
    down = model.add_point(shift + down_end[0], *down_end[1:])
    model.add_face([shared, model.add_curve(b, down, _section_side(upright, (0., 1., 0.), (shift + 9e-10, 0., 1e-10),
                                                                   (shift + down_end[0], *down_end[1:]))),
                    model.add_line(down, a)], surface=upright)
    return model


@pytest.mark.parametrize("shift", [0., 1024.])
def test_faces_overlapping_within_the_joint_band_past_a_shared_end_are_clean(shift):
    # review B1-P5 rounds 3-4: the overlap (about 9e-10, within the tolerance) past the shared end B lies in
    # B's joint band; under the owner's contract (a) for face pairs (P4e, 2026-10-15) this is a joint contact
    assert _overlap_past_shared_end(shift).strict_audit().clean


def test_faces_overlapping_beyond_the_joint_band_are_reported():
    # both sections leave B at a shallow angle: the faces overlap along the x-axis to x ~ 1e-7 (100
    # tolerances), beyond the joint band
    model = _overlap_past_shared_end(0., low_end=(1., -1e-3, 0.), down_end=(1., 0., -1e-3))
    assert not model.strict_audit().clean


def test_volume_is_translation_invariant():
    from anygeometry import GeometryModel
    from anygeometry.shell_volume import closed_shell, enclosed_volume
    from anygeometry.entities import OrientedEdge
    # review B1-P5 round 3: the unit cube at 1e6 lost ~1e-4 of its volume to cancellation
    model = GeometryModel()
    offset = 1e6
    p = {(i, j, k): model.add_point(offset + i, offset + j, offset + k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    lines = {}

    def line(u, v):
        key = frozenset((u, v))
        if key not in lines:
            lines[key] = model.add_line(p[u], p[v])
        return lines[key]
    quads = [((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),   # outward
             ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
             ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))]
    # explicit oriented loops (add_face orders a loop from its first edge's stored direction)
    faces = []
    for q in quads:
        loop = []
        for i in range(4):
            u, v = q[i], q[(i + 1) % 4]
            edge = line(u, v)
            loop.append(OrientedEdge(edge, model.edges[edge].start == p[u]))
        corner = [offset + c for c in q[0]]
        u, v = ([float(b - a) for a, b in zip(q[0], q[k])] for k in (1, 3))
        faces.append(model.add_face_from_loop(loop, surface=Plane(corner, u, v)))
    assert closed_shell(model, faces) == []
    result = enclosed_volume(model, faces)
    assert result["volume"] == pytest.approx(1., abs=1e-12)
    assert result["centroid"] == pytest.approx([offset + .5] * 3, abs=1e-9)

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
    value = abs(_nurbs_face_moments(model, model.faces[face], 24, (0., 0., 0.))[0])
    assert abs(value - 1.001) <= 1e-12


def test_an_unresolved_volume_quadrature_is_refused():
    from anygeometry import GeometryModel
    from anygeometry.entities import OrientedEdge
    from anygeometry.errors import GeometryError
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.shell_volume import closed_shell, enclosed_volume
    # review B1-P5 round 4: the cube [0, L]^3 with a flat rational top S(u, v) = (L u / (e + (1 - e) u), L v, L),
    # e = 2**-19: both Gauss orders underresolve it (volume 5.04e17 for 1e18) and one moment scale accepted it
    size, e = 1e6, 2.0 ** -19
    top = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                       (((0., 0., size), (0., size, size)), ((size, 0., size), (size, size, size))),
                       weights=((e, e), (1., 1.)))
    model = GeometryModel()
    p = {(i, j, k): model.add_point(size * i, size * j, size * k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    edges = {}
    chart = {(0, 0): (0., 0.), (1, 0): (1., 0.), (1, 1): (1., 1.), (0, 1): (0., 1.)}
    for a, b in (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))):
        edges[frozenset(((*a, 1), (*b, 1)))] = model.add_curve(
            p[(*a, 1)], p[(*b, 1)], CurveOnSurface(top, NurbsCurve(1, (0., 0., 1., 1.), (chart[a], chart[b]))))

    def loop_of(corners):
        loop = []
        for i in range(len(corners)):
            u, v = corners[i], corners[(i + 1) % len(corners)]
            key = frozenset((u, v))
            if key not in edges:
                edges[key] = model.add_line(p[u], p[v])
            loop.append(OrientedEdge(edges[key], model.edges[edges[key]].start == p[u]))
        return loop
    faces = [model.add_face_from_loop(loop_of(((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))), surface=top)]
    for q in (((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
              ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
              ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))):
        u, v = ([size * float(b - a) for a, b in zip(q[0], q[k])] for k in (1, 3))
        faces.append(model.add_face_from_loop(loop_of(q), surface=Plane([size * c for c in q[0]], u, v)))
    assert closed_shell(model, faces) == []
    try:
        result = enclosed_volume(model, faces)
    except GeometryError:
        return                                                  # refused: acceptable
    assert result["volume"] == pytest.approx(size ** 3, rel=1e-8)


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


def _corner_entry(c_point=(.5, -1., 0.)):
    """A deck face in z = 0 and a vertical NURBS face y = -a (a = 5e-10, chart z in [-1, 0]) sharing the
    section A -> B; the NURBS face's other section B -> D also lies in z = 0 and enters the deck where the
    deck edge B -> C crosses y = -a (review B1-P5 round 4 finding 2)."""
    from anygeometry import GeometryModel
    from anygeometry.b1_example import _parameter
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    a = 5e-10
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((-1., -a, -1.), (-1., -a, 1.)), ((1., -a, -1.), (1., -a, 1.))))
    deck = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    section = PlaneNurbsCurve.from_plane(deck, surface, 0)
    if section.evaluate(0.)[0] > 0:
        section = section.subcurve(1., 0.)
    t_b = _parameter(section, (1, 0, 0), 0.)
    model = GeometryModel()
    va, vd = model.add_point(*section.evaluate(0.)), model.add_point(*section.evaluate(1.))
    vb = model.add_point(0., 0., 0.)
    shared = model.add_curve(va, vb, section.subcurve(0., t_b))
    onward = model.add_curve(vb, vd, section.subcurve(t_b, 1.))
    vd2, va2 = model.add_point(1., -a, -1.), model.add_point(-1., -a, -1.)

    def segment(start, end, p, q):
        return model.add_curve(start, end, CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (p, q))))
    model.add_face([shared, onward, segment(vd, vd2, (1., .5), (1., 0.)), segment(vd2, va2, (1., 0.), (0., 0.)),
                    segment(va2, va, (0., 0.), (0., .5))], surface=surface)
    vc, ve, vf = model.add_point(*c_point), model.add_point(1., -2., 0.), model.add_point(-1., -2., 0.)
    model.add_face([shared, model.add_line(vb, vc), model.add_line(vc, vd), model.add_line(vd, ve),
                    model.add_line(ve, vf), model.add_line(vf, va)], surface=deck)
    return model


def test_a_section_entering_a_plane_face_within_the_corner_band_is_clean():
    # the unshared section enters the deck for 0 < x < a / 2 = 2.5e-10, next to the shared corner B: a joint
    # contact under contract (a) for face pairs (P4e, owner 2026-10-15)
    assert _corner_entry().strict_audit().clean


def test_a_section_entering_a_plane_face_beyond_the_corner_band_is_reported():
    # the deck edge B -> C nearly parallel to the section (slope 1e-3): the section runs inside the deck
    # to x = a / 1e-3 = 5e-7, about 500 tolerances from B
    assert not _corner_entry((.5, -5e-4, 0.)).strict_audit().clean


def _deck_and_wall(heights):
    """A deck face in z = 0 (y below the section) and a NURBS wall z = h(y) over x in [-1, 1], y in [0, 1]
    (``heights``: Bernstein coefficients of h), sharing the wall's section with z = 0."""
    from anygeometry import GeometryModel
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    q = len(heights) - 1
    surface = NurbsSurface(1, q, (0., 0., 1., 1.), (0.,) * (q + 1) + (1.,) * (q + 1),
                           tuple(tuple((x, j / q, heights[j]) for j in range(q + 1)) for x in (-1., 1.)))
    deck = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    section = PlaneNurbsCurve.from_plane(deck, surface, 0)
    if section.evaluate(0.)[0] > 0:
        section = section.subcurve(1., 0.)
    model = GeometryModel()
    va, vb = model.add_point(*section.evaluate(0.)), model.add_point(*section.evaluate(1.))
    shared = model.add_curve(va, vb, section)
    (ua, wa), (ub, wb) = section.chart(0.), section.chart(1.)
    corners = [model.add_point(*surface.evaluate(ub, 1.)), model.add_point(*surface.evaluate(ua, 1.))]

    def segment(start, end, p, q_):
        return model.add_curve(start, end, CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (p, q_))))
    model.add_face([shared, segment(vb, corners[0], (ub, wb), (ub, 1.)),
                    segment(corners[0], corners[1], (ub, 1.), (ua, 1.)), segment(corners[1], va, (ua, 1.), (ua, wa))],
                   surface=surface)
    low = [model.add_point(1., -1., 0.), model.add_point(-1., -1., 0.)]
    model.add_face([shared, model.add_line(vb, low[0]), model.add_line(low[0], low[1]), model.add_line(low[1], va)],
                   surface=deck)
    return model


def _bernstein(power):
    from math import comb
    n = len(power) - 1
    return [sum(comb(j, i) / comb(n, i) * power[i] for i in range(j + 1)) for j in range(n + 1)]


def test_a_steep_wall_on_a_deck_edge_is_clean():
    assert _deck_and_wall(_bernstein([-.1, 1.])).strict_audit().clean            # z = y - 0.1 (45 degrees)


def _deck_and_surface(surface, deck_tail):
    """A NURBS face on ``surface`` above its section with z = 0 (chart v from the section to 1) and a deck
    polygon in z = 0 sharing that section A -> B, continuing B -> deck_tail... -> A."""
    from anygeometry import GeometryModel
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve
    from anygeometry.support_curves import PlaneNurbsCurve
    deck = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    section = PlaneNurbsCurve.from_plane(deck, surface, 0)
    if section.evaluate(0.)[0] > section.evaluate(1.)[0]:
        section = section.subcurve(1., 0.)
    model = GeometryModel()
    va, vb = model.add_point(*section.evaluate(0.)), model.add_point(*section.evaluate(1.))
    shared = model.add_curve(va, vb, section)
    (ua, wa), (ub, wb) = section.chart(0.), section.chart(1.)
    corners = [model.add_point(*surface.evaluate(ub, 1.)), model.add_point(*surface.evaluate(ua, 1.))]

    def segment(start, end, p, q_):
        return model.add_curve(start, end, CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (p, q_))))
    model.add_face([shared, segment(vb, corners[0], (ub, wb), (ub, 1.)),
                    segment(corners[0], corners[1], (ub, 1.), (ua, 1.)), segment(corners[1], va, (ua, 1.), (ua, wa))],
                   surface=surface)
    points = [vb] + [model.add_point(*p) for p in deck_tail] + [va]
    model.add_face([shared] + [model.add_line(p, q) for p, q in zip(points, points[1:])], surface=deck)
    return model


def _ruled(heights, y=(0., 1.)):
    """z = h(y) over x in [-1, 1] (Bernstein coefficients ``heights`` on y in ``y``)."""
    from anygeometry.nurbs import NurbsSurface
    q = len(heights) - 1
    return NurbsSurface(1, q, (0., 0., 1., 1.), (0.,) * (q + 1) + (1.,) * (q + 1),
                        tuple(tuple((x, y[0] + (y[1] - y[0]) * j / q, heights[j]) for j in range(q + 1))
                              for x in (-1., 1.)))


def test_a_steep_wall_over_the_deck_is_clean():
    # z = y - 0.1 over the deck y in [0.1, 0.9] (the faces form a 45-degree wedge): band ~1.4 tolerances
    assert _deck_and_surface(_ruled([-.1, .9]), [(1., .9, 0.), (-1., .9, 0.)]).strict_audit().clean


def test_a_shallow_wall_over_the_deck_is_reported():
    # z = 0.05 (y - 0.1) over the deck y in [0.1, 0.9]: the faces stay within the tolerance of each other
    # up to ~20 tolerances from the shared edge, beyond the joint band (review P4e round 1: the
    # opposite-side version did not show this)
    assert not _deck_and_surface(_ruled([-.005, .045]), [(1., .9, 0.), (-1., .9, 0.)]).strict_audit().clean


def test_a_crease_valley_touching_the_deck_is_reported():
    from anygeometry.nurbs import NurbsSurface
    # review P4e round 1: a degree-(1,1) surface steep on every patch but with a valley at a knot crease
    # (y = 0.6) coming within 5e-10 of the deck, 0.5 from the shared edge
    xs, ys = (-1., 0., 1.), (0., .2, .4, .6, 1.)
    surface = NurbsSurface(1, 1, (0., 0., .5, 1., 1.), (0., 0., .2, .4, .6, 1., 1.),
                           tuple(tuple((x, y, h) for y, h in zip(ys, (-.1, .1, .3 + abs(x), 5e-10 + abs(x), .4 + abs(x))))
                                 for x in xs))
    model = _deck_and_surface(surface, [(1., .9, 0.), (-1., .9, 0.)])
    assert not model.strict_audit().clean


def test_a_descent_leaving_the_surface_is_not_a_band():
    from anygeometry import GeometryModel
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.section_audit import face_face
    from anygeometry.support_curves import PlaneNurbsCurve
    # review P4e round 1: S(u, v) = (u, v, v + 0.1 (u - 0.5)); the section leaves the chart at v = 0, and
    # near u = 0.5 + 9 tol, v ~ 0 the face is within the tolerance of the deck polygon, 9 tol from the section
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., -.05), (0., 1., .95)), ((1., 0., .05), (1., 1., 1.05))))
    deck = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    section = PlaneNurbsCurve.from_plane(deck, surface, 0)
    if section.evaluate(0.)[0] > section.evaluate(1.)[0]:
        section = section.subcurve(1., 0.)
    model = GeometryModel()
    va, vb = model.add_point(*section.evaluate(0.)), model.add_point(*section.evaluate(1.))
    shared = model.add_curve(va, vb, section)
    (ua, wa), (ub, wb) = section.chart(0.), section.chart(1.)

    def segment(start, end, p, q):
        return model.add_curve(start, end, CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (p, q))))
    c1, c2, c3 = (model.add_point(*surface.evaluate(*uv)) for uv in ((1., 0.), (1., 1.), (0., 1.)))
    nurbs = model.add_face([shared, segment(vb, c1, (ub, wb), (1., 0.)), segment(c1, c2, (1., 0.), (1., 1.)),
                            segment(c2, c3, (1., 1.), (0., 1.)), segment(c3, va, (0., 1.), (ua, wa))], surface=surface)
    tail = [model.add_point(1., 0., 0.), model.add_point(1., 1., 0.), model.add_point(0., 1., 0.)]
    points = [vb] + tail + [va]
    plane = model.add_face([shared] + [model.add_line(p, q) for p, q in zip(points, points[1:])], surface=deck)
    assert face_face(model, plane, nurbs, 1e-9) is None


def test_an_uncovered_section_near_the_plane_face_is_not_cleared():
    from anygeometry.section_audit import face_face
    # review P4e round 1: the deck edge B -> C at slope 0.1 runs within ~0.4 tol of the surface 8.5 tol from
    # B; an intersection certificate accepted the uncovered section, a distance certificate must not
    assert face_face(_corner_entry((.5, -.05, 0.)), 1, 2, 1e-9) is None


def test_a_short_intersection_far_from_the_origin_is_reported():
    from anygeometry import GeometryModel
    # review P4e round 1: two faces at z = 65536 sharing vertices A and B = A + (4e-8, 0, 0) whose
    # interiors meet along AB (20 tolerances from either vertex); a 2**-40 coordinate slack hid it
    big, gap = 65536., 4e-8
    model = GeometryModel()
    a, b = model.add_point(0., 0., big), model.add_point(gap, 0., big)
    flat = Plane((0., 0., big), (1., 0., 0.), (0., 1., 0.))
    first = [(-1., 1.), (1., 1.)]
    second = [(1., -1.), (-1., -1.)]
    p = [model.add_point(x, y, big) for x, y in first + second]
    model.add_face([model.add_curve(a, p[0], _section_side(flat, (0., 0., 1.), (0., 0., big), (-1., 1., big))),
                    model.add_line(p[0], p[1]), model.add_line(p[1], b), model.add_line(b, p[2]),
                    model.add_line(p[2], p[3]), model.add_line(p[3], a)], surface=flat)
    upright = Plane((0., 0., big), (1., 0., 0.), (0., 0., 1.))
    q = [model.add_point(x, 0., big + y) for x, y in first + second]
    model.add_face([model.add_curve(a, q[0], _section_side(upright, (0., 1., 0.), (0., 0., big), (-1., 0., big + 1.))),
                    model.add_line(q[0], q[1]), model.add_line(q[1], b), model.add_line(b, q[2]),
                    model.add_line(q[2], q[3]), model.add_line(q[3], a)], surface=upright)
    assert not model.strict_audit().clean


def test_crossings_that_floats_cannot_separate_are_refused():
    from anygeometry import GeometryModel
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry.tolerance import TolerancePolicy
    # review P4e round 2: at y ~ 2**40 the crossings A and B = A + (4e-8, 0, 0) round to one s value; the
    # faces meet 20 tolerances from both shared vertices. Floats there cannot resolve the tolerance: refused
    big, k, gap = 2.0 ** 40, 1 / 64, 4e-8
    model = GeometryModel(tolerance=TolerancePolicy(relative_length=1e-30))
    a, b = model.add_point(0., big, 0.), model.add_point(gap, big, 0.)
    flat = Plane((0., big, 0.), (1., 0., 0.), (0., 1., 0.))
    ring = [(-1., big + 1.), (1., big + 1.), None, (1., big - 1.), (1., 1.), (1., 0.), (-1., 0.), (-1., 1.), (-1., big - 1.)]
    ids = [a] + [b if xy is None else model.add_point(xy[0], xy[1], 0.) for xy in ring]
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((1., 0., -1.), (1., 0., 1.)), ((-1., 0., -1.), (-1., 0., 1.))))
    section = PlaneNurbsCurve.from_plane(flat, surface, 0)
    if section.evaluate(0.)[0] < 0:
        section = section.subcurve(1., 0.)
    edges = []
    for p0, p1 in zip(ids, ids[1:] + ids[:1]):
        q0, q1 = model.vertices[p0].position, model.vertices[p1].position
        if q0[0] == 1. and q0[1] == 0. and q1[0] == -1. and q1[1] == 0.:
            edges.append(model.add_curve(p0, p1, section))
        else:
            edges.append(model.add_line(p0, p1))
    model.add_face(edges, surface=flat)
    tilted = Plane((0., big, 0.), (1., k, 0.), (0., 0., 1.))
    others = [model.add_point(-1., big - k, 1.), model.add_point(1., big + k, 1.), b,
              model.add_point(1., big + k, -1.), model.add_point(-1., big - k, -1.)]
    ids = [a] + others
    model.add_face([model.add_line(p0, p1) for p0, p1 in zip(ids, ids[1:] + ids[:1])], surface=tilted)
    assert not model.strict_audit().clean


def test_a_steep_diagonal_section_through_rectangle_corners_is_clean():
    from anygeometry import GeometryModel
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    from anygeometry.section_audit import face_face
    from anygeometry.support_curves import PlaneNurbsCurve
    # review P4e round 2: S(u, v) = (u, v, u + v - 0.5) over the deck polygon (0,.5) (.5,0) (1,0) (1,1) (0,1)
    # sharing the diagonal section, which passes through dyadic rectangle corners
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., -.5), (0., 1., .5)), ((1., 0., .5), (1., 1., 1.5))))
    deck = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    section = PlaneNurbsCurve.from_plane(deck, surface, 0)
    if section.evaluate(0.)[0] > section.evaluate(1.)[0]:
        section = section.subcurve(1., 0.)
    model = GeometryModel()
    va, vb = model.add_point(*section.evaluate(0.)), model.add_point(*section.evaluate(1.))   # (0,.5), (.5,0)
    shared = model.add_curve(va, vb, section)

    def segment(start, end, p, q):
        return model.add_curve(start, end, CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (p, q))))
    (ua, wa), (ub, wb) = section.chart(0.), section.chart(1.)
    c1, c2, c3 = (model.add_point(*surface.evaluate(*uv)) for uv in ((1., 0.), (1., 1.), (0., 1.)))
    nurbs = model.add_face([shared, segment(vb, c1, (ub, wb), (1., 0.)), segment(c1, c2, (1., 0.), (1., 1.)),
                            segment(c2, c3, (1., 1.), (0., 1.)), segment(c3, va, (0., 1.), (ua, wa))], surface=surface)
    tail = [model.add_point(1., 0., 0.), model.add_point(1., 1., 0.), model.add_point(0., 1., 0.)]
    points = [vb] + tail + [va]
    plane = model.add_face([shared] + [model.add_line(p, q) for p, q in zip(points, points[1:])], surface=deck)
    assert face_face(model, plane, nurbs, 1e-9) is True
