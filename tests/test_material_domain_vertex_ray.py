"""T2-0: the float ray test of MaterialDomain.contains must not lose a crossing at a boundary vertex.

Reproduces the three-way tripod node defect (faces 961/601, centre_r8.5): the query height lies within
128 eps (in the line parameter) of a generator/cap-arc vertex. The generator reports its endpoint root
t = 1 (snapped; skipped as "the next curve owns it"), the following cap EllipticArc starts 1e-13 above
the ray and reports no root, so the vertex crossing was lost: a point 6 deg outside the strip was
inside, and a point inside at the same height was outside.
"""
import math

import numpy as np
import pytest

from anygeometry.arrangement_geometry import LinePath
from anygeometry.exact_curves import EllipticArc
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain
from anygeometry.surfaces import Cylinder

R = 6.5
SIX = math.radians(6.)


def _strip():
    """Strip 0 <= theta <= 6 deg on a full-turn cylinder whose chart starts at -0.25 rad (as the tripod arms).

    Oblique caps: z(theta) = z0 - sin(theta), so the top cap rises from the 6 deg vertex to the 0 deg vertex.
    """
    support = Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius=R,
                       height=52., start_angle=-.25)
    bottom = EllipticArc((0., 0., 4.), (R, 0., 0.), (0., R, -1.), 0., SIX)
    top = EllipticArc((0., 0., 45.), (R, 0., 0.), (0., R, -1.), SIX, -SIX)
    up = LinePath(tuple(map(float, bottom.evaluate(1.))), tuple(map(float, top.evaluate(0.))))
    down = LinePath(tuple(map(float, top.evaluate(1.))), tuple(map(float, bottom.evaluate(0.))))
    loop = tuple(ArrangementPath(c) for c in (bottom, up, top, down))
    return MaterialDomain(0, support, (loop,)), up


def _contains(domain, theta, z):
    point = (R*math.cos(theta), R*math.sin(theta), z)
    return domain.contains(LinePath(point, point), 0., 1e-9)


@pytest.mark.parametrize("delta", (1e-13, 3e-13, 5e-13))
def test_point_outside_strip_at_vertex_height_is_outside(delta):
    domain, up = _strip()
    z = float(up.end[2])-delta            # within 128 eps of the 6 deg vertex in the generator's parameter
    assert not _contains(domain, -SIX, z)
    assert not _contains(domain, -3*SIX, z)


@pytest.mark.parametrize("delta", (1e-13, 3e-13, 5e-13))
def test_point_inside_strip_at_vertex_height_is_inside(delta):
    domain, up = _strip()
    z = float(up.end[2])-delta
    assert _contains(domain, .5*SIX, z)


def _plane_face():
    """Plane face: long right edge up to vertex V at 5 deg on a unit circle, arc 5 -> 15 deg, then back.

    The edge is 100 long, so a query 1e-12 below V snaps its root to t = 1 (128 eps), while the
    arc's start root lies 6e-12 of its sweep outside the arc and is not reported.
    """
    from anygeometry.surfaces import Plane
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), math.radians(5.), math.radians(10.))
    v_point, w_point = (tuple(map(float, arc.evaluate(t))) for t in (0., 1.))
    corners = ((v_point[0], -99.5, 0.), v_point, None, w_point, (0., w_point[1], 0.), (0., -99.5, 0.))
    curves = (LinePath(corners[0], corners[1]), arc, LinePath(corners[3], corners[4]),
              LinePath(corners[4], corners[5]), LinePath(corners[5], corners[0]))
    support = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    return MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in curves),)), v_point


@pytest.mark.parametrize("route", ("exact", "float"))
@pytest.mark.parametrize("delta", (1e-12, 2e-12))
def test_plane_vertex_height_line_then_arc(route, delta):
    domain, v_point = _plane_face()
    y = v_point[1]-delta

    def contains(x):
        if route == "exact":           # a line station: exact chart point, contains_loop_exact_curved
            point = (x, y, 0.)
            return domain.contains(LinePath(point, point), 0., 1e-9)
        # an arc station: float chart point, the float MaterialDomain.contains_loop on the Plane
        query = EllipticArc((x-1e-3, y, 0.), (1e-3, 0., 0.), (0., 1e-3, 0.), 0., 1.)
        return domain.contains(query, 0., 1e-9)

    assert not contains(-1.)
    assert contains(.5)
    assert not contains(2.)


def test_ordinary_heights_unchanged():
    domain, up = _strip()
    z = .5*(float(up.start[2])+float(up.end[2]))
    assert _contains(domain, .5*SIX, z)
    assert not _contains(domain, -SIX, z)
    assert not _contains(domain, 2*SIX, z)
    assert _contains(domain, SIX, z)                  # on the boundary generator
    assert _contains(domain, SIX, float(up.end[2]))  # the vertex itself


# --- T2-0 fix round 1: horizontal runs, several junctions, short lines, loop start (review P1) -----------

def _plane():
    from anygeometry.surfaces import Plane
    return Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))


def _point(curve, t):
    return tuple(map(float, curve.evaluate(t)))


def _reversed(curve):
    if isinstance(curve, LinePath):
        return LinePath(curve.end, curve.start)
    return EllipticArc(tuple(curve.center), tuple(curve.u_vector), tuple(curve.v_vector),
                       curve.start_angle+curve.sweep_angle, -curve.sweep_angle)


def _variants(curves):
    """The loop in both orientations and from every first curve."""
    for loop in (list(curves), [_reversed(c) for c in reversed(curves)]):
        for shift in range(len(loop)):
            yield loop[shift:]+loop[:shift]


def _plane_query(route, x, y):
    if route == "exact":           # a line station: exact chart point (exact planar routes)
        point = (x, y, 0.)
        return LinePath(point, point)
    return EllipticArc((x-1e-3, y, 0.), (1e-3, 0., 0.), (0., 1e-3, 0.), 0., 1.)   # float route


def _assert_plane_cases(curves, cases, route):
    support = _plane()
    for loop in _variants(curves):
        domain = MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in loop),))
        got = [domain.contains(_plane_query(route, x, y), 0., 1e-9) for x, y, _ in cases]
        assert got == [want for _, _, want in cases], (loop[0], route)


def _horizontal_run_face():
    """The review repro: _plane_face with its right edge P -> V replaced by P -> H -> V, H = V - (0.1, 0)."""
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), math.radians(5.), math.radians(10.))
    v_point, w_point = (_point(arc, t) for t in (0., 1.))
    h_point = (v_point[0]-.1, v_point[1], 0.)
    p_point = (v_point[0], -99.5, 0.)
    curves = [LinePath(p_point, h_point), LinePath(h_point, v_point), arc, LinePath(w_point, (0., w_point[1], 0.)),
              LinePath((0., w_point[1], 0.), (0., -99.5, 0.)), LinePath((0., -99.5, 0.), p_point)]
    return curves, v_point


@pytest.mark.parametrize("route", ("exact", "float"))
def test_horizontal_run_before_arc_keeps_the_vertex_crossing(route):
    curves, v_point = _horizontal_run_face()
    y = v_point[1]-1e-12
    _assert_plane_cases(curves, ((-1., y, False), (.5, y, True), (2., y, False)), route)


def _two_junction_face(delta=1e-12):
    """One upward line -> arc junction 1e-12 above the ray and one downward line -> horizontal -> arc
    junction 1e-12 below it (review: the parent lost both, T2-0 restored one; x = -4 must stay outside)."""
    up_arc = EllipticArc((1., delta, 0.), (1., 0., 0.), (0., 1., 0.), math.pi, -math.pi/2)
    down_arc = EllipticArc((2.9, -delta, 0.), (1., 0., 0.), (0., 1., 0.), 0., -math.pi/2)
    v1, w1, v2, w2 = _point(up_arc, 0.), _point(up_arc, 1.), _point(down_arc, 0.), _point(down_arc, 1.)
    g2 = (4., v2[1], 0.)
    return [LinePath((0., -99., 0.), v1), up_arc, LinePath(w1, (4., 99., 0.)), LinePath((4., 99., 0.), g2),
            LinePath(g2, v2), down_arc, LinePath(w2, (2., -99., 0.)), LinePath((2., -99., 0.), (0., -99., 0.))]


@pytest.mark.parametrize("route", ("exact", "float"))
def test_two_vertex_junctions_count_once_each(route):
    _assert_plane_cases(_two_junction_face(), ((-4., 0., False), (2., 0., True), (3.5, 0., True),
                                               (6., 0., False)), route)


def _short_line_face():
    """Straight edges only: a long edge ends 1e-13 above the ray (root snapped to t = 1) and a 1e-5 long
    edge continues upward; its root lies 1e-8 before its start, outside the line root window."""
    points = [(0., -99., 0.), (0., 1e-13, 0.), (3e-6, 1e-13+1e-5, 0.), (5., 99., 0.), (5., -99., 0.)]
    return [LinePath(points[i], points[(i+1) % len(points)]) for i in range(len(points))]


@pytest.mark.parametrize("route", ("exact", "float"))
def test_short_line_after_vertex_keeps_the_crossing(route):
    # exact: the all-straight exact route (_contains_loop_exact); float: MaterialDomain.contains_loop
    _assert_plane_cases(_short_line_face(), ((-1., 0., False), (2., 0., True), (6., 0., False)), route)


def test_cylinder_horizontal_cap_run_keeps_the_vertex_crossing():
    """Tripod-like strip: generator up to V, a constant-height arc (on the ray), then the rising cap."""
    support = Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius=R,
                       height=52., start_angle=-.25)
    four = math.radians(4.)
    bottom = EllipticArc((0., 0., 4.), (R, 0., 0.), (0., R, -1.), 0., SIX)
    z_vertex = 44.
    flat = EllipticArc((0., 0., z_vertex), (R, 0., 0.), (0., R, 0.), SIX, -(SIX-four))
    cap = EllipticArc((0., 0., z_vertex+math.sin(four)), (R, 0., 0.), (0., R, -1.), four, -four)
    up = LinePath(_point(bottom, 1.), _point(flat, 0.))
    down = LinePath(_point(cap, 1.), _point(bottom, 0.))
    curves = [bottom, up, flat, cap, down]
    z = float(_point(flat, 0.)[2])-1e-12
    for loop in _variants(curves):
        domain = MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in loop),))
        assert not _contains(domain, -SIX, z)
        assert _contains(domain, math.radians(2.), z)
        assert not _contains(domain, 2*SIX, z)


# --- brute-force oracle: random line/arc loops with vertices within 1e-13 of the ray ----------------------

def _arc_between(p, q, phi):
    mx, my = .5*(p[0]+q[0]), .5*(p[1]+q[1])
    dx, dy = q[0]-p[0], q[1]-p[1]
    d = math.hypot(dx, dy)
    h = .5*d/math.tan(.5*phi)
    cx, cy = mx-dy/d*h, my+dx/d*h
    r = math.hypot(p[0]-cx, p[1]-cy)
    return EllipticArc((cx, cy, 0.), (r, 0., 0.), (0., r, 0.), math.atan2(p[1]-cy, p[0]-cx), phi)


def _random_loop(rng, y0):
    """A star-shaped loop of long lines and arcs; 1-3 vertices moved within 1e-13 of y0, some followed
    by one or two horizontal edges on the ray or by a 1e-5 kink edge."""
    k = rng.randint(4, 8)
    angles = sorted(rng.uniform(0, math.tau) for _ in range(k))
    vertices = [[rng.uniform(20, 100)*math.cos(a), rng.uniform(20, 100)*math.sin(a)+y0] for a in angles]
    near = rng.sample(range(k), rng.randint(1, min(3, k)))
    spec = []
    for i, vertex in enumerate(vertices):
        if i in near:
            vertex[1] = y0+rng.choice((-1, 1))*rng.choice((rng.uniform(0, 1e-13), 1e-13, 0.))
        spec.append((tuple(vertex), False))
        if i in near:
            draw = rng.random()
            if draw < .35:
                for _ in range(rng.randint(1, 2)):
                    last = spec[-1][0]
                    spec.append(((last[0]+rng.choice((-1, 1))*rng.uniform(.05, 2.), last[1]), True))
            elif draw < .55:
                last, angle = spec[-1][0], rng.uniform(0, math.tau)
                spec.append(((last[0]+1e-5*math.cos(angle), last[1]+1e-5*math.sin(angle)), False))
    curves, end = [], None
    for i in range(len(spec)):
        a = spec[i][0] if end is None else end
        b = spec[i+1][0] if i+1 < len(spec) else _point(curves[0], 0.)[:2]
        line = spec[(i+1) % len(spec)][1] or math.hypot(b[0]-a[0], b[1]-a[1]) < 1e-3 or rng.random() < .5
        if line:
            curves.append(LinePath((a[0], a[1], 0.), (b[0], b[1], 0.)))
            end = b
        else:
            curves.append(_arc_between(a, b, rng.choice((-1, 1))*rng.uniform(.2, 2.)))
            end = _point(curves[-1], 1.)[:2]
    return curves


def _winding(curves, q):
    """Winding number by summed subtended angles (the query is >= 1e-6 from the boundary)."""
    def wrap(x):
        return (x+math.pi) % math.tau-math.pi
    total = 0.
    for curve in curves:
        s, e = _point(curve, 0.), _point(curve, 1.)
        turn = math.atan2(e[1]-q[1], e[0]-q[0])-math.atan2(s[1]-q[1], s[0]-q[0])
        if isinstance(curve, EllipticArc) and math.hypot(q[0]-curve.center[0], q[1]-curve.center[1]) < curve.u_vector[0]:
            turn %= math.tau                       # inside the circle: the angle is monotone along the arc
            total += turn if curve.sweep_angle > 0 else (turn-math.tau if turn else 0.)
        else:
            total += wrap(turn)                    # a segment, or an arc seen from outside its circle
    return round(total/math.tau)


def _distance(curves, q):
    best = math.inf
    for curve in curves:
        if isinstance(curve, LinePath):
            s, e = curve.start, curve.end
            dx, dy = e[0]-s[0], e[1]-s[1]
            t = max(0., min(1., ((q[0]-s[0])*dx+(q[1]-s[1])*dy)/(dx*dx+dy*dy)))
            best = min(best, math.hypot(s[0]+t*dx-q[0], s[1]+t*dy-q[1]))
            continue
        cx, cy, r = float(curve.center[0]), float(curve.center[1]), float(curve.u_vector[0])
        angle = math.atan2(q[1]-cy, q[0]-cx)
        if any(0 <= (angle+k*math.tau-curve.start_angle)/curve.sweep_angle <= 1 for k in range(-2, 3)):
            best = min(best, abs(math.hypot(q[0]-cx, q[1]-cy)-r))
        best = min(best, *(math.hypot(p[0]-q[0], p[1]-q[1]) for p in (_point(curve, 0.), _point(curve, 1.))))
    return best


@pytest.mark.parametrize("seed", (2, 5))
def test_vertex_ray_containment_matches_winding_oracle(seed):
    import random
    rng = random.Random(seed)
    support = _plane()
    wrong = []
    for trial in range(6):
        y0 = rng.choice((0., .37, -12.5))
        curves = _random_loop(rng, y0)
        shift = rng.randrange(len(curves))
        xs = [rng.uniform(-110, 110) for _ in range(4)]
        for loop in (curves, [_reversed(c) for c in reversed(curves)], curves[shift:]+curves[:shift]):
            domain = MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in loop),))
            for x in xs:
                if _distance(loop, (x, y0)) < 1e-6:
                    continue
                want = _winding(loop, (x, y0)) != 0
                for route in ("exact", "float"):
                    if domain.contains(_plane_query(route, x, y0), 0., 1e-9) != want:
                        wrong.append((trial, route, x, y0, want))
    assert not wrong
