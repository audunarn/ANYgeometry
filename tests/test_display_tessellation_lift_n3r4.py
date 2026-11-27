"""N3 Codex round 4 (codex-n3r4.md, codex-n3r4-lead-overlap.txt; SPEC-N3-R4-FIXES.md): every
reproduction as a regression test.

H1  class "chart non-overlap + per-triangle orientation does not imply that the LIFTED triangles do
    not overlap in 3D" (r2 H2 via winding, r4 via a notch narrower than the chord deviation): the
    lift certifies global injectivity of a projection P of the lifted mesh back to the chart
    (local: every lifted triangle's P-image positively oriented; boundary: the P-image of the
    lifted boundary is a set of disjoint simple closed curves, nested as the trim).  The Codex
    notch face is refused; the lead's topology-invalid self-crossing loop is refused (or accepted
    only if not overlapping); the cylinder/cone oracles check P-image overlap exactly, so the Codex
    mesh fails the oracle.
"""
from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

import test_display_tessellation_lift as T
import test_display_tessellation_lift_cylinder as C
import test_display_tessellation_lift_n3r1 as R1
from anygeometry import _display_lift as lift
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone, Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy

_mesh, _status = T._mesh, R1._status


# ----------------------------------------------------------------------------- exact planar overlap
def _orient2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p, q, r, s):
    """Closed segments ``pq`` and ``rs`` (2D rationals) properly cross (interiors, not collinear)."""
    o1, o2, o3, o4 = _orient2(p, q, r), _orient2(p, q, s), _orient2(r, s, p), _orient2(r, s, q)
    return o1 * o2 < 0 and o3 * o4 < 0


def _strictly_inside(x, tri):
    a, b, c = tri
    s = _orient2(a, b, c)
    return all(_orient2(u, v, x) * s > 0 for u, v in ((a, b), (b, c), (c, a)))


def _planar_overlaps(mesh, coordinates):
    """Exact overlap test of a mesh whose points all lie on one plane: ``coordinates`` maps a 3D
    point to exact 2D coordinates on that plane.  Two published triangles overlap if an edge of one
    properly crosses an edge of the other, or a vertex of one lies strictly inside the other."""
    pts = [coordinates([F(float(x)) for x in p]) for p in mesh.points]
    tris = mesh.triangles.tolist()
    hits = []
    for i, t in enumerate(tris):
        for j in range(i + 1, len(tris)):
            u = tris[j]
            a, b = [pts[k] for k in t], [pts[k] for k in u]
            crossing = any(_segments_cross(a[x], a[(x + 1) % 3], b[y], b[(y + 1) % 3])
                           for x in range(3) for y in range(3)
                           if len({t[x], t[(x + 1) % 3], u[y], u[(y + 1) % 3]}) == 4)
            inside = (any(_strictly_inside(pts[k], b) for k in t if k not in u)
                      or any(_strictly_inside(pts[k], a) for k in u if k not in t))
            if crossing or inside:
                hits.append((t, u))
    return hits


# ----------------------------------------------------------------------------- H1: the Codex notch
def _codex_notch():
    """Codex N3 r4 H1: a topology-valid cylinder face (its points on the plane 128 x = z) with a
    notch narrower than the chord deviation: the chart is certified, every lifted triangle is
    oriented and within the deviation bound, but triangles [10, 11, 14] and [9, 2, 3] cover the
    same open patch of the plane."""
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.01))
    s = Cylinder((0, 0, 128), (0, 0, 1), (1, 0, 0), 1., 1., start_angle=-1.)
    a, b, c = map(math.atan, (.02, .08, .1))
    chart = [(-.02, -.05), (b, -.05), (b, b), (a, a), (0., 1e-5), (c, c + 1e-5), (c, .15), (-.02, .15)]

    def point(y, h):
        z = 128 + h
        return g.add_point(z / 128, y, z)
    p = [point((1 + h / 128) * math.tan(t), h) for t, h in chart]
    via = point((1 + (c + 1e-5) / 128) * math.tan(c) / 2, c / 2 + 1e-5 + .0003)
    edges = [g.add_arc(x, via, y) if i == 4 else g.add_line(x, y) for i, (x, y) in enumerate(zip(p, p[1:] + p[:1]))]
    f = g.add_face(edges, surface=s)
    assert not g.validate_topology()
    return g, f, dt.DisplayTolerance(.25, sag=.01)


def _notch_plane(p):
    """Exact coordinates on the plane 128 x = z (every point of the Codex mesh lies on it)."""
    return p[1], p[2]


def test_h1_a_notch_whose_lifted_sides_cross_is_refused():
    """Codex N3 r4 H1: the lifted notch sides cross (the chord deviation exceeds the notch width).
    The global certificate: the projection of the lifted boundary back to the chart is not simple
    -> TRIM_INVALID.  If it were accepted, no two published triangles may overlap (exact)."""
    g, f, tol = _codex_notch()
    code, mesh = _status(lambda: dt.face_display_mesh(g, f, tol))
    if mesh is not None:
        assert all(128 * F(x) == F(z) for x, y, z in mesh.points)
        assert not _planar_overlaps(mesh, _notch_plane)
    assert code == "TRIM_INVALID"


def test_h1_the_cylinder_oracle_rejects_the_overlapping_notch_mesh(monkeypatch):
    """The cylinder oracle checks P-image (3D) overlap exactly: the Codex mesh, built with the new
    certificate switched off, overlaps (exactly, on its plane) and fails the oracle."""
    monkeypatch.setattr(lift, "_check_projection", lambda *args, **kwargs: None, raising=False)
    g, f, tol = _codex_notch()
    mesh = dt.face_display_mesh(g, f, tol)
    hits = _planar_overlaps(mesh, _notch_plane)
    assert ([10, 11, 14], [9, 2, 3]) in hits or ([9, 2, 3], [10, 11, 14]) in hits, hits
    with pytest.raises(AssertionError, match="overlap"):
        C._check_cylinder(g, f, tol, mesh)


# ----------------------------------------------------------------------------- lead: self-crossing loop
def _lead_face():
    """The aborted r4 run's lead: a loop that self-intersects in 3D (on the plane x = 1) but not in
    the chart; the model refuses it, so it is forced in (topology-invalid)."""
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.5))
    ring = [(.07, 0), (.133, 0), (.383, .5), (.633, 1), (.603, 1), (.353, .5), (.118, .03), (.115, .03), (.6, 1),
            (.57, 1)]
    v = [g.add_point(1., math.tan(t), z) for t, z in ring]
    e = [g.add_line(a, b) for a, b in zip(v, v[1:] + v[:1])]
    s = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.)
    loop = tuple(OrientedEdge(k, True) for k in e)
    f = g.add_face_from_loop(loop[:3] + (OrientedEdge(g.add_line(v[3], v[0]), True),), surface=s)
    g._put_entity("face", replace(g.faces[f], loop=loop))
    assert g.validate_topology()
    return g, f, dt.DisplayTolerance(.5, sag=.5)


def test_lead_a_self_crossing_loop_is_refused_or_not_overlapping():
    """Lead (codex-n3r4-lead-overlap.txt): accepted on 5c4d38c with overlapping triangles.  Refused
    (TRIM_INVALID), or accepted only if no two published triangles overlap (exact, on x = 1)."""
    g, f, tol = _lead_face()
    code, mesh = _status(lambda: dt.face_display_mesh(g, f, tol))
    if mesh is None:
        assert code == "TRIM_INVALID"
    else:
        assert all(F(float(p[0])) == 1 for p in mesh.points)
        assert not _planar_overlaps(mesh, _notch_plane)
        C._check_cylinder(g, f, tol, mesh)


# ----------------------------------------------------------------------------- the boundary certificate
def _square_projection(points):
    plane = lift._PlaneChart(7, Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    return lift._PlaneProjection(plane, np.array([(x, y, 0.) for x, y in points], dtype=float))


def _chain(*loops):
    tails = np.array([v for loop in loops for v in loop], dtype=np.int64)
    heads = np.array([v for loop in loops for v in loop[1:] + loop[:1]], dtype=np.int64)
    return tails, heads


SQUARE_WITH_HOLE = [(0, 0), (4, 0), (4, 4), (0, 4), (1, 1), (2, 1), (2, 2), (1, 2)]


def test_h1_the_boundary_certificate_accepts_a_square_with_a_hole():
    proj = _square_projection(SQUARE_WITH_HOLE)
    assert lift._check_boundary(7, proj, *_chain([0, 1, 2, 3], [4, 7, 6, 5])) is None


@pytest.mark.parametrize("name, points, loops", [
    ("bow tie", [(0, 0), (1, 0), (1, 1), (0, 1)], ([0, 1, 3, 2],)),
    ("spike back along an edge", [(0, 0), (2, 0), (1, 0), (1, 1)], ([0, 1, 2, 3],)),
    ("a vertex on a non-adjacent edge", [(0, 0), (2, 0), (2, 2), (1, 0), (0, 2)], ([0, 1, 2, 3, 4],)),
    ("a hole crossing the outer loop", SQUARE_WITH_HOLE[:4] + [(3, 1), (5, 1), (5, 2), (3, 2)],
     ([0, 1, 2, 3], [4, 7, 6, 5])),
    ("two touching loops", SQUARE_WITH_HOLE[:4] + [(4, 1), (3, 1), (3, 2)], ([0, 1, 2, 3], [4, 6, 5])),
])
def test_h1_a_lifted_boundary_that_is_not_simple_is_refused(name, points, loops):
    """(ii): non-adjacent boundary edges must be disjoint, adjacent ones meet only at their shared
    vertex (exact predicates after a box sweep)."""
    proj = _square_projection(points)
    with pytest.raises(dt.DisplayTrimRefused, match="not simple") as info:
        lift._check_boundary(7, proj, *_chain(*loops))
    assert info.value.code == "TRIM_INVALID"


def test_h1_a_hole_oriented_like_the_outer_loop_is_refused():
    """(iii) nesting: with every lifted triangle positive, the projection's degree is at most 1
    only if exactly one boundary loop is positively oriented (the outer); a hole turned the other
    way (degree 2 inside it) is refused."""
    proj = _square_projection(SQUARE_WITH_HOLE)
    with pytest.raises(dt.DisplayTrimRefused, match="nested") as info:
        lift._check_boundary(7, proj, *_chain([0, 1, 2, 3], [4, 5, 6, 7]))
    assert info.value.code == "TRIM_INVALID"


def test_h1_curved_boundary_edges_are_compared_as_curves_not_chords():
    """On a cylinder the projection of a lifted chord is a curve (azimuth, axial): two chords whose
    chart chords are disjoint but whose lifted projections cross are refused.  The chord from
    (theta, h) = (0, 0) to (.2, .2) lifts to a secant whose azimuth at h = .05 is about .049875
    (``.1 - atan(tan(.1) / 2)``), not the chart chord's .05; the short chord at h = .05 over theta
    in [.0498, .04995] misses the chart chord but crosses the lifted one."""
    support = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.)
    chart = lift._CylinderChart(support)
    o, e1, e2 = chart.o, chart.e1, chart.e2

    def at(t, h):
        return o + math.cos(t) * e1 + math.sin(t) * e2 + h * chart.a
    ring = [(0., 0.), (.2, .2), (.2, .3), (.04995, .05), (.0498, .05), (0., .3)]
    points = np.array([at(t, h) for t, h in ring])
    theta = np.array([t for t, _ in ring])
    proj = lift._CurvedProjection(7, chart, points, theta)
    with pytest.raises(dt.DisplayTrimRefused, match="not simple"):
        lift._check_boundary(7, proj, *_chain([0, 1, 2, 3, 4, 5]))


# ----------------------------------------------------------------------------- (i) the cone's projection Jacobian
def test_h1_the_cone_projection_jacobian_is_the_radial_one_not_the_support_normal():
    """(i) on a cone: P = (azimuth, axial), whose Jacobian sign on a lifted triangle is n . c(theta)
    (radial, perpendicular to the axis), not the support normal n . (c - k a) of Q7.  A triangle
    whose normal is (-0.1, 0, -1) in the frame is positive for Q7 at k = 1 but folds under P:
    refused."""
    support = Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), .5, 1.5, 1.)
    chart = lift._ConeChart(support)
    e1, e2, a = chart.e1, chart.e2, chart.a
    A = chart.o + 1.0 * e1
    points = np.array([A, A + .01 * e2, A + .1 * e1 - .01 * a])
    rows = np.array([[0, 1, 2]])
    zero, folded = lift._thin_curved(points, rows, [0.], [.0101], e1, e2, a, chart.k, 1)
    assert not folded[0] and not zero[0]                      # Q7 (support normal): positive
    theta = np.array([0., .01, 0.])
    proj = lift._CurvedProjection(7, chart, points, theta)
    with pytest.raises(dt.DisplayTrimRefused, match="projection") as info:
        lift._projection_triangles(7, proj, points, rows, 1)
    assert info.value.code == "TRIM_INVALID"
    lift._projection_triangles(7, proj, points, rows[:, [0, 2, 1]], 1)      # reversed: P-positive
