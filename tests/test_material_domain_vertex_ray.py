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


# --- T2-0 fix round 2: an off-band end root keeps its own crossing location (review R2 P2) ---------------
#
# A long oblique edge ends at a vertex V k eps (of its height) away from the ray, outside the tolerance
# band, so its root t ~ 1 - k eps is not snapped. Fix 1 skipped that root and counted the crossing at V,
# i.e. moved it horizontally by ~k eps * L (31x the tolerance for L = 1e8, k = 4): a query between the
# true crossing and V was inside. The truth below is an exact rational winding number.

_EPS = 2.**-52
_TOL = 1e-9


def _exact_winding(vertices, q):
    """Winding number of a closed polygon around q (exact rationals; q is not on the polygon)."""
    from fractions import Fraction
    qx, qy = map(Fraction, q)
    winding = 0
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:]+vertices[:1]):
        ax, ay, bx, by = map(Fraction, (ax, ay, bx, by))
        side = (bx-ax)*(qy-ay)-(qx-ax)*(by-ay)
        if ay <= qy < by and side > 0:
            winding += 1
        elif by <= qy < ay and side < 0:
            winding -= 1
    return winding


def _exact_distance2(vertices, q):
    from fractions import Fraction
    qx, qy = map(Fraction, q)
    best = None
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:]+vertices[:1]):
        ax, ay, bx, by = map(Fraction, (ax, ay, bx, by))
        dx, dy = bx-ax, by-ay
        length2 = dx*dx+dy*dy
        t = 0 if length2 == 0 else max(Fraction(0), min(Fraction(1), ((qx-ax)*dx+(qy-ay)*dy)/length2))
        ex, ey = ax+t*dx-qx, ay+t*dy-qy
        best = ex*ex+ey*ey if best is None else min(best, ex*ex+ey*ey)
    return best


def _bulge(p, q):
    """A half circle from p to q bulging to the right of p -> q (start = centre + u: p up to rounding)."""
    centre = (.5*(p[0]+q[0]), .5*(p[1]+q[1]), 0.)
    u = (p[0]-centre[0], p[1]-centre[1], 0.)
    v = (-u[1], u[0], 0.)                       # u turned counterclockwise: theta = pi/2 is right of p -> q
    return EllipticArc(centre, u, v, 0., math.pi)


def _lens_winding(arc, q):
    """Winding number around q of the arc closed by its chord (float; q is far from both)."""
    a, b = _point(arc, 0.), _point(arc, 1.)
    radius = math.hypot(arc.u_vector[0], arc.u_vector[1])
    centre = arc.center
    if math.hypot(q[0]-centre[0], q[1]-centre[1]) >= radius:
        return 0
    chord_side = (b[0]-a[0])*(q[1]-a[1])-(q[0]-a[0])*(b[1]-a[1])      # > 0: q left of the chord a -> b
    mid = _point(arc, .5)
    bulge_side = (b[0]-a[0])*(mid[1]-a[1])-(mid[0]-a[0])*(b[1]-a[1])
    if chord_side*bulge_side <= 0:
        return 0
    return 1 if bulge_side < 0 else -1          # arc a -> b then chord b -> a: counterclockwise if bulging right


def _arc_distance(arc, q):
    centre = arc.center
    radius = math.hypot(arc.u_vector[0], arc.u_vector[1])
    mid = _point(arc, .5)
    best = min(math.hypot(q[0]-p[0], q[1]-p[1]) for p in (_point(arc, 0.), _point(arc, 1.)))
    # the half circle is the half on the side of its midpoint
    if (q[0]-centre[0])*(mid[0]-centre[0])+(q[1]-centre[1])*(mid[1]-centre[1]) >= 0:
        best = min(best, abs(math.hypot(q[0]-centre[0], q[1]-centre[1])-radius))
    return best


def _off_band_loop(vertices, arc_after=None):
    """Lines through ``vertices``; with ``arc_after = i`` the edge from vertex i is a right-bulging half circle."""
    curves = []
    for i, a in enumerate(vertices):
        b = vertices[(i+1) % len(vertices)]
        if i == arc_after:
            curves.append(_bulge(a, b))
        else:
            curves.append(LinePath((a[0], a[1], 0.), (b[0], b[1], 0.)))
    return curves


def _truth(vertices, curves, q):
    """(inside, distance) by the exact polygon winding plus the float lens of any arc (q far from it)."""
    winding = _exact_winding(vertices, q)
    distance = math.sqrt(float(_exact_distance2(vertices, q)))
    for curve in curves:
        if isinstance(curve, EllipticArc):
            winding += _lens_winding(curve, q)
            distance = min(distance, _arc_distance(curve, q))
    return winding != 0, distance




def _answer(domain, loop, uv, boundary):
    return domain.contains_loop(loop, uv, _TOL, boundary=boundary)


def _applies(domain, loop):
    """Whether the T2-0 junction rule applies to this loop (its certified domain)."""
    return domain._junction_rule_applies(loop, _TOL)


def _route_answers(curves, q, boundary, *, with_trust=False, support=None):
    """{route: contains_loop answer} over the exact and float routes for the loop ``curves``."""
    from fractions import Fraction
    loop = tuple(ArrangementPath(c) for c in curves)
    domain = MaterialDomain(0, support or _plane(), (loop,))
    exact = "exact-curved" if any(isinstance(c, EllipticArc) for c in curves) else "exact-straight"
    answers = {exact: _answer(domain, loop, tuple(map(Fraction, q)), boundary),
               "float": _answer(domain, loop, tuple(map(float, q)), boundary)}
    return (answers, _applies(domain, loop)) if with_trust else answers


def _check_off_band(vertices, arc_after, queries, wrong, label, margin=2*_TOL, trust=None):
    """Both orientations, every first curve, both boundary settings; queries closer than margin are skipped.

    ``trust`` (a list) collects whether the junction rule applied to each loop."""
    curves = _off_band_loop(vertices, arc_after)
    checked = 0
    for q in queries:
        want, distance = _truth(vertices, curves, q)
        if distance <= margin:
            continue
        for loop in _variants(curves):
            for boundary in (False, True):
                answers, trusted = _route_answers(loop, q, boundary, with_trust=True)
                if trust is not None:
                    trust.append(trusted)
                for route, got in answers.items():
                    checked += 1
                    if got != want:
                        wrong.append((label, route, q, want, got, type(loop[0]).__name__, boundary))
    return checked


def _codex_queries(scale, k):
    """The review repro family: incoming edge (-L, -L) -> V = (1, 0), the ray at y = -k eps L."""
    from fractions import Fraction
    delta = k*_EPS*scale
    crossing = 1-Fraction(delta)*(1+1/Fraction(scale))      # exact x of the incoming edge at y = -delta
    gap = max(4*delta, 1e-7)
    xs = (float(crossing)-gap, float((crossing+1)/2), 1.-delta/2, 1.+gap, 1.75, -scale/2., 2.)
    return [(x, -delta) for x in xs]


def _off_band_family(scale, arc, trust=None):
    vertices = [(-scale, -scale), (1., 0.), (1., 1.), (-scale-1., 1.), (-scale-1., -scale)]
    wrong, checked = [], 0
    for k in range(1, 10):
        checked += _check_off_band(vertices, 1 if arc else None, _codex_queries(scale, k), wrong, (scale, k),
                                   trust=trust)
    return checked, wrong


@pytest.mark.parametrize("scale", (1., 100., 1000.))
@pytest.mark.parametrize("arc", (False, True), ids=("lines", "arc"))
def test_off_band_end_root_keeps_its_crossing_location(scale, arc):
    """Review R2 P2 repro family inside the junction rule's domain (both orientations, every first curve,
    three routes, offsets 1..9 eps): every answer is the exact oracle's."""
    trust = []
    checked, wrong = _off_band_family(scale, arc, trust)
    assert checked and all(trust)
    assert not wrong, (len(wrong), checked, wrong[:6])


def _spike_wrong(arc, tip, scale=1e8, trust=None):
    """A spike: two long edges both cross the ray within 8 eps of a common apex k eps L above it."""
    wrong, checked = [], 0
    for k in range(1, 10):
        apex = (0., k*_EPS*scale)
        vertices = [(-scale, -scale), apex, (scale, -scale), (scale, -2*scale), (-scale, -2*scale)]
        curves = _off_band_loop(vertices, 3 if arc else None)
        xs = (0.,) if tip else (-1e-7-2*k*_EPS*scale, 1e-7+2*k*_EPS*scale, -scale/2, 2*scale)
        for x in xs:
            q = (x, 0.)
            want, distance = _truth(vertices, curves, q)
            if distance <= 2*_TOL:
                continue
            for loop in _variants(curves):
                for boundary in (False, True):
                    answers, trusted = _route_answers(loop, q, boundary, with_trust=True)
                    if trust is not None:
                        trust.append(trusted)
                    for route, got in answers.items():
                        checked += 1
                        if got != want:
                            wrong.append((k, route, x, want, got, type(loop[0]).__name__, boundary))
    return checked, wrong


@pytest.mark.parametrize("scale", (1., 60., 300.))
@pytest.mark.parametrize("arc", (False, True), ids=("lines", "arc"))
def test_spike_away_from_the_tip_matches_the_oracle(scale, arc):
    """A spike inside the junction rule's domain (the arc variant's half circle of diameter 2 L has speed
    pi 2 L, so L <= 300): queries left of, right of and far from the tip."""
    trust = []
    checked, wrong = _spike_wrong(arc, tip=False, scale=scale, trust=trust)
    assert checked and all(trust)
    assert not wrong, (len(wrong), wrong[:6])


@pytest.mark.parametrize("scale", (1., 60., 500.))
def test_spike_tip_is_a_boundary_point_inside_the_scale_contract(scale):
    """Inside the domain the spike mechanism is unreachable: an apex k eps L (k <= 9) off the ray leaves a
    tip narrower than the tolerance, so every tip query is a boundary point (answer = the boundary flag)."""
    for k in range(1, 10):
        apex = (0., k*_EPS*scale)
        vertices = [(-scale, -scale), apex, (scale, -scale), (scale, -2*scale), (-scale, -2*scale)]
        curves = _off_band_loop(vertices)
        q = (0., 0.)
        assert _truth(vertices, curves, q)[1] <= _TOL
        for loop in _variants(curves):
            for boundary in (False, True):
                answers, trusted = _route_answers(loop, q, boundary, with_trust=True)
                assert trusted
                assert set(answers.values()) == {boundary}, (scale, k, answers)


# --- T2-0 contract: the junction rule applies only inside a narrow certified domain --------------------------
#
# Four review rounds each found a failure of the new junction/ray rules outside the regime the tripod needs
# (huge scales, tuned curves, ill-conditioned charts, several roots in one window). The final contract
# (2026-10-29): the junction rule applies only to a loop of LinePath/EllipticArc curves on a well-conditioned
# Plane (chart condition number <= 16) or a Cylinder, at a scale (support origin, curve coordinates and
# parametric speeds) <= tol / (2048 eps) (2199 for tol = 1e-9), and not to a query where a curve has two
# roots in one 8-eps end window or two adjacent curves both have a root strictly inside the 8-eps windows of
# their common junction. Everywhere else contains_loop is the parent's (2608332) rule, bit for bit.

_R3_SCALE = 1.1e8
_R3_QUERY = (0.09999989601589046, -9.769962616701378e-08)


def _r3_vertices(scale=_R3_SCALE, x0=.1):
    return [(-scale, -scale), (x0, 0.), (x0, 1.), (-scale-1., 1.), (-scale-1., -scale)]


def _sweep_queries(scale, k, x0):
    """Queries around the junction of the R2/R3 family at ray height -k eps L: inside and outside the
    incoming edge's true crossing at 3..100 tol, between the crossing and V, past V, and far."""
    from fractions import Fraction
    delta = k*_EPS*scale
    crossing = Fraction(x0)-Fraction(delta)*(Fraction(x0)+Fraction(scale))/Fraction(scale)
    xs = [float(crossing)+m*_TOL for m in (-100, -10, -5, -3, 3, 5, 10, 100)]
    xs += [float((crossing+Fraction(x0))/2), x0-delta/2, x0+3*_TOL, x0+max(4*delta, 1e-7), -scale/2., 2.*abs(x0)+2.]
    return [(x, -delta) for x in xs]


def _sweep_family(scales, trust=None):
    wrong, checked = [], 0
    for scale in scales:
        for x0 in (1., .1):
            vertices = _r3_vertices(scale, x0)
            for k in range(1, 10):
                checked += _check_off_band(vertices, None, _sweep_queries(scale, k, x0), wrong, (scale, x0, k),
                                           trust=trust)
    return checked, wrong


# The largest L of the R2/R3 family the guard admits: its scale is the incoming edge length ~ sqrt(2) L.
_SWEEP_TOP = 1550.


@pytest.mark.parametrize("scales", ((1.,), (10.,), (100.,), (1000.,), (_SWEEP_TOP,)))
def test_scale_sweep_against_exact_oracle(scales):
    """Scales 1..1550 (measured scale up to ~2193, the guard's limit 2199) x ray offsets 1..9 eps x two
    vertex abscissae, both orientations, every first curve, both boundary flags, float and exact queries:
    the junction rule applies to every loop and no answer is wrong."""
    trust = []
    checked, wrong = _sweep_family(scales, trust)
    assert checked and all(trust)
    assert not wrong, (len(wrong), checked, wrong[:6])


def test_scale_guard_constants():
    """K = 128 (the 128-eps endpoint snapping band, which contains the 8-eps junction windows), M = 16."""
    from anygeometry import material_arrangement as ma
    assert (ma._FLOAT_RAY_BAND, ma._FLOAT_RAY_MARGIN, ma._CHART_CONDITION) == (128, 16, 16)
    limit = 1e-9/(128*16*_EPS)
    assert ma.float_ray_scale_trusted(limit*(1-1e-12), 1e-9) and not ma.float_ray_scale_trusted(limit*(1+1e-12), 1e-9)
    assert ma.float_ray_scale_trusted(60., 1e-9)             # tripod coordinates, 37x inside the limit
    assert not ma.float_ray_scale_trusted(float("nan"), 1e-9)
    loop = tuple(ArrangementPath(c) for c in _off_band_loop(_r3_vertices(_SWEEP_TOP)))
    assert _applies(MaterialDomain(0, _plane(), (loop,)), loop)
    loop = tuple(ArrangementPath(c) for c in _off_band_loop(_r3_vertices(1600.)))
    assert not _applies(MaterialDomain(0, _plane(), (loop,)), loop)


def _square_loop(size=1.):
    corners = ((0., 0., 0.), (size, 0., 0.), (size, size, 0.), (0., size, 0.))
    return tuple(ArrangementPath(LinePath(corners[i], corners[(i+1) % 4])) for i in range(4))


@pytest.mark.parametrize("u, v, admitted", (
    ((1., 0., 0.), (0., 1., 0.), True),
    ((0., 0., 2.), (0., -3., 0.), True),                       # orthogonal, ratio 1.5: condition 1.5
    ((1., 0., 0.), (0., 15.9, 0.), True),                      # ratio 15.9: condition 15.9
    ((1., 0., 0.), (0., 16.1, 0.), False),                     # ratio 16.1
    ((1., 0., 0.), (1., .12, 0.), False),                      # sin ~ 0.119: condition ~ 16.7
    ((1., 0., 0.), (1., .13, 0.), True),                       # sin ~ 0.129: condition ~ 15.4
    ((1., 0., 0.), (1., 1e-12, 0.), False),                    # review R4 P2-3
    ((1e-6, 0., 0.), (0., 1e-6, 0.), True),                    # a uniform scale keeps the condition
))
def test_junction_rule_needs_a_well_conditioned_plane_chart(u, v, admitted):
    """The chart condition number sigma_max / sigma_min of [u v] must be <= 16, i.e.
    |u|^2 + |v|^2 <= (16 + 1/16) |u x v|; this implies sin(u, v) >= 1/8.03 and |u|/|v| in [1/16, 16]."""
    from anygeometry.surfaces import Plane
    loop = _square_loop()
    domain = MaterialDomain(0, Plane((0., 0., 0.), u, v), (loop,))
    assert _applies(domain, loop) is admitted


def test_junction_rule_domain_supports_and_curves():
    """Seam-free Cylinder loops are admitted (review R5 closure); other supports and other curve types are not."""
    from anygeometry.arrangement_geometry import BezierPath
    from anygeometry.surfaces import Cone
    domain, _up = _strip()
    assert _applies(domain, domain.boundaries[0])
    cone = Cone(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius_start=R,
                radius_end=R/2, height=52.)
    cone_domain = MaterialDomain(0, cone, domain.boundaries)
    assert not _applies(cone_domain, domain.boundaries[0])
    bezier = (ArrangementPath(BezierPath(((0., 0., 0.), (.5, 1., 0.), (1., 0., 0.)))),
              ArrangementPath(LinePath((1., 0., 0.), (0., 0., 0.))))
    assert not _applies(MaterialDomain(0, _plane(), (bezier,)), bezier)


def _random_off_band_polygon(rng, top_exponent=2.3):
    """A long edge P0 -> V with V k eps (1..16) of the edge's height off the ray y = 0, then W either on V's
    side (the next edge has no root) or far across the ray (its root is a few eps after its start), then
    1-3 far vertices. Returns (vertices, queries near the junction and one far query)."""
    from fractions import Fraction
    scale = 10.**rng.uniform(0., top_exponent)
    k = rng.randint(1, 16)
    side = rng.choice((-1., 1.))
    vx = rng.uniform(-1., 1.)
    v = (vx, side*k*_EPS*scale)
    p0 = (vx-rng.uniform(.2, 2.)*scale, -side*scale)
    if rng.random() < .5:
        w = (vx+rng.choice((-1., 1.))*rng.uniform(.5, 2.), side*rng.uniform(.5, 2.))
    else:
        w = (vx+rng.uniform(.2, 2.)*scale*rng.choice((-1., 1.)), -side*scale*rng.uniform(.5, 1.5))
    vertices = [p0, v, w]
    for _ in range(rng.randint(1, 3)):
        vertices.append((rng.uniform(-3., 3.)*scale, rng.uniform(-3., 3.)*scale))
    (ax, ay), (bx, by) = (tuple(map(Fraction, p)) for p in (p0, v))
    crossing = ax+(bx-ax)*(0-ay)/(by-ay)                     # exact x of the incoming edge at y = 0
    gap = max(abs(float(crossing)-vx), 1e-7)
    xs = [float(crossing)-2*gap, float((crossing+Fraction(vx))/2), vx+(float(crossing)-vx)/4,
          vx+2*gap, vx-2*gap, rng.uniform(-3., 3.)*scale]
    return vertices, [(x, 0.) for x in xs]


@pytest.mark.parametrize("seed", (11, 12, 13))
def test_off_band_junctions_match_exact_winding_oracle(seed):
    """Random long-edge junctions (L 1..200, 1..16 eps) inside the domain: lines, and a following half circle."""
    import random
    rng = random.Random(seed)
    wrong, checked, trust = [], 0, []
    for trial in range(6):
        vertices, queries = _random_off_band_polygon(rng)
        checked += _check_off_band(vertices, None, queries, wrong, (seed, trial, "lines"), trust=trust)
        if math.hypot(vertices[2][0]-vertices[1][0], vertices[2][1]-vertices[1][1]) < 10.:
            checked += _check_off_band(vertices, 1, queries, wrong, (seed, trial, "arc"), trust=trust)
    assert checked and all(trust)
    assert not wrong, (len(wrong), checked, wrong[:6])


def test_window_root_pairs_keep_the_parent_rule():
    """Clause (c) of the contract, with the junction rule forced on in the exact replicas: a curve with two
    roots in one end window (review R4 P2-1) or two crossings in the windows of one junction (the spike)
    keep the parent's answer for that query rather than the junction rule's."""
    from fractions import Fraction
    from anygeometry.planar_arcs import contains_loop_exact_curved
    from anygeometry.material_arrangement import _contains_loop_exact
    loop, query = _r4_circle()
    chart = MaterialDomain(0, _plane(), (loop,))._exact_chart()
    exact = tuple(map(Fraction, query))
    for boundary in (False, True):
        assert (contains_loop_exact_curved(chart, loop, exact, _TOL, boundary=boundary, junctions=True)
                == contains_loop_exact_curved(chart, loop, exact, _TOL, boundary=boundary)
                == _PARENT["r4_circle"][f"exact-{boundary}"])
    scale = 1e8
    for k in range(1, 10):
        vertices = [(-scale, -scale), (0., k*_EPS*scale), (scale, -scale), (scale, -2*scale), (-scale, -2*scale)]
        for curves in _variants(_off_band_loop(vertices)):
            spike = tuple(ArrangementPath(c) for c in curves)
            for boundary in (False, True):
                old = _contains_loop_exact(chart, spike, (Fraction(0), Fraction(0)), _TOL, boundary=boundary)
                assert _contains_loop_exact(chart, spike, (Fraction(0), Fraction(0)), _TOL, boundary=boundary,
                                            junctions=True) == old


# --- KNOWN LIMITS: outside the domain the parent's (2608332) answers stand ---------------------------------
#
# The supported range of the junction rule is stated above. Outside it contains_loop keeps the parent's
# float ray rule (and its exact replicas) bit for bit, including these pre-existing wrong answers (they are
# not refused). The expected values below were recorded on a 2608332 source export; any future change to
# them is a behaviour change outside the supported domain and must be deliberate.

_PARENT = {          # recorded on a 2608332 source export; answer lists as '1'/'0' strings
    'r4_circle': {'float-False': False, 'float-True': False, 'exact-False': True, 'exact-True': True},
    'r4_bezier': '11000011',
    'r4_plane': {'float-False': False, 'float-True': True, 'exact-False': True, 'exact-True': True},
    'r3': '0000000000000000000011111111111111111111',
    'r3_clip': {'intervals': [], 'contacts': 0},
    'r3_residual': '0'*40,
    'r2': '0'*40,
    'spike_tip_False': '0'*320+'1'*40,
    'spike_tip_True': '0'*320+'1'*40,
    'family_counts': {
        'off_band_10000_False': (1800, 0),
        'off_band_10000_True': (1440, 360),
        'off_band_1e+08_False': (2520, 320),
        'off_band_1e+08_True': (2520, 320),
        'spike_away_False': (1440, 0),
        'spike_away_True': (1440, 0),
        'sweep_1e4': (8640, 0),
        'sweep_1e5': (8640, 0),
        'sweep_1e6': (8640, 800),
        'sweep_1e7': (9760, 1500),
        'sweep_1e8': (20160, 3250),
    },
    'cylinder_1e7': True,
}


def _r4_circle():
    """Review R4 P2-1: one full circle R = 1e24 starting 1e-15 rad before its top; the query lies 490 tol
    outside it, and the ray crosses the circle twice within the start window."""
    radius = 1e24
    arc = EllipticArc((0., -radius, 0.), (0., radius, 0.), (radius, 0., 0.), -1e-15, math.tau)
    return (ArrangementPath(arc),), (-1e9, -1e-8)


def test_known_limit_r4_circle_two_roots_in_one_window():
    from fractions import Fraction
    loop, query = _r4_circle()
    domain = MaterialDomain(0, _plane(), (loop,))
    assert not _applies(domain, loop)
    got = {f"{route}-{boundary}": _answer(domain, loop, uv, boundary)
           for route, uv in (("float", query), ("exact", tuple(map(Fraction, query)))) for boundary in (False, True)}
    assert got == _PARENT["r4_circle"]


def _r4_bezier():
    """Review R4 P2-2: degree-10 Bezier x = t, y = 1e12 (t (t-1/4) (t-1/2) (t-3/4) (t-1))^2, closed by a line;
    y and y' vanish at the five parameters d0a9dbd sampled (its measured scale was ~2, the true ~4.5e7)."""
    from fractions import Fraction
    from math import comb
    from anygeometry.arrangement_geometry import BezierPath
    roots = [Fraction(i, 4) for i in range(5)]
    quintic = [Fraction(1)]
    for r in roots:                                          # power coefficients of prod (t - r)
        quintic = [(quintic[i-1] if i else 0)-r*(quintic[i] if i < len(quintic) else 0) for i in range(len(quintic)+1)]
    power = [sum(quintic[i]*quintic[j-i] for i in range(len(quintic)) if 0 <= j-i < len(quintic))*10**12
             for j in range(11)]
    n = 10
    ys = [sum(Fraction(comb(k, i), comb(n, i))*power[i] for i in range(k+1)) for k in range(n+1)]
    controls = tuple((k/n, float(y), 0.) for k, y in enumerate(ys))
    return (ArrangementPath(BezierPath(controls)), ArrangementPath(LinePath((1., 0., 0.), (0., 0., 0.))))


_R4_BEZIER_QUERIES = ((.1, 1e6), (.1, 2e7), (.5, -1.), (.9, 5e6))


def test_known_limit_r4_bezier_is_outside_the_domain():
    loop = _r4_bezier()
    domain = MaterialDomain(0, _plane(), (loop,))
    assert not _applies(domain, loop)
    got = _bits(_answer(domain, loop, q, boundary) for q in _R4_BEZIER_QUERIES for boundary in (False, True))
    assert got == _PARENT["r4_bezier"]


def test_known_limit_r4_ill_conditioned_plane_chart():
    """Review R4 P2-3: a unit square on Plane(u = (1, 0, 0), v = (1, 1e-12, 0)); the query is inside (world
    ~(6.1e-5, 0.5)), the float route (pre-existing) says outside, the exact route inside."""
    from fractions import Fraction
    from anygeometry.surfaces import Plane
    loop = _square_loop()
    domain = MaterialDomain(0, Plane((0., 0., 0.), (1., 0., 0.), (1., 1e-12, 0.)), (loop,))
    assert not _applies(domain, loop)
    uv = (-499999999999.99994, 500000000000.)
    got = {f"{route}-{boundary}": _answer(domain, loop, q, boundary)
           for route, q in (("float", uv), ("exact", tuple(map(Fraction, uv)))) for boundary in (False, True)}
    assert got == _PARENT["r4_plane"]


def _bits(answers):
    return "".join("1" if answer else "0" for answer in answers)


def _variant_answers(vertices, query, arc_after=None):
    from fractions import Fraction
    got = []
    for loop in _variants(_off_band_loop(vertices, arc_after)):
        paths = tuple(ArrangementPath(c) for c in loop)
        domain = MaterialDomain(0, _plane(), (paths,))
        assert not _applies(domain, paths)
        for boundary in (False, True):
            got.append(_answer(domain, paths, tuple(map(float, query)), boundary))
            got.append(_answer(domain, paths, tuple(map(Fraction, query)), boundary))
    return _bits(got)


def test_known_limit_r3_float_root_repro():
    """Review R3 P2 (L = 1.1e8): an interior point 4.44e-9 from the boundary."""
    want, distance = _truth(_r3_vertices(), _off_band_loop(_r3_vertices()), _R3_QUERY)
    assert want and 4e-9 < distance < 5e-9
    assert _variant_answers(_r3_vertices(), _R3_QUERY) == _PARENT["r3"]


def test_known_limit_r3_clip_intervals():
    """Review R3 P2 through _clip_intervals: a semicircle of radius 2e-9 around the R3 query."""
    from anygeometry.material_arrangement import _clip_intervals
    loop = tuple(ArrangementPath(c) for c in _off_band_loop(_r3_vertices()))
    domain = MaterialDomain(0, _plane(), (loop,))
    arc = EllipticArc((*_R3_QUERY, 0.), (2e-9, 0., 0.), (0., 2e-9, 0.), math.pi/2, math.pi)
    contacts = []
    intervals = _clip_intervals(domain, arc, _TOL, lambda: None, isolated_contacts=contacts)
    assert [list(map(float, i)) for i in intervals] == _PARENT["r3_clip"]["intervals"]
    assert len(contacts) == _PARENT["r3_clip"]["contacts"]


def test_known_limit_r3_residual_exterior_at_1e8():
    vertices = _r3_vertices(1e8, 1.)
    query = (0.9999999807955393, -2.220446049250313e-08)
    want, distance = _truth(vertices, _off_band_loop(vertices), query)
    assert not want and distance > 2*_TOL
    assert _variant_answers(vertices, query) == _PARENT["r3_residual"]


def test_known_limit_review_r2_repro():
    scale = 1e8
    vertices = [(-scale, -scale), (1., 0.), (1., 1.), (-scale-1., 1.), (-scale-1., -scale)]
    assert _variant_answers(vertices, (1.-2*_EPS*scale, -4*_EPS*scale)) == _PARENT["r2"]


@pytest.mark.parametrize("arc", (False, True), ids=("lines", "arc"))
def test_known_limit_spike_tip_at_1e8(arc):
    got = ""
    for k in range(1, 10):
        vertices = [(-1e8, -1e8), (0., k*_EPS*1e8), (1e8, -1e8), (1e8, -2e8), (-1e8, -2e8)]
        got += _variant_answers(vertices, (0., 0.), 3 if arc else None)
    assert got == _PARENT[f"spike_tip_{arc}"]


def test_known_limit_family_counts():
    """Wrong answers (vs the exact oracle) of the parent rule on the out-of-domain families, unchanged."""
    got = {}
    for scale in (1e4, 1e8):
        for arc in (False, True):
            checked, wrong = _off_band_family(scale, arc)
            got[f"off_band_{scale:g}_{arc}"] = (checked, len(wrong))
    for arc in (False, True):
        checked, wrong = _spike_wrong(arc, tip=False)
        got[f"spike_away_{arc}"] = (checked, len(wrong))
    for exponent in range(4, 9):
        checked, wrong = _sweep_family((1e8, _R3_SCALE) if exponent == 8 else (10.**exponent,))
        got[f"sweep_1e{exponent}"] = (checked, len(wrong))
    assert got == _PARENT["family_counts"]


def test_known_limit_cylinder_strip_at_1e7():
    """A cylinder strip at radius 1e7 is outside the scale guard: the parent's answer, no refusal."""
    from anygeometry.surfaces import Cylinder

    def strip(radius):
        support = Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
                           radius=radius, height=52., start_angle=-.25)
        bottom = EllipticArc((0., 0., 4.), (radius, 0., 0.), (0., radius, 0.), 0., SIX)
        top = EllipticArc((0., 0., 45.), (radius, 0., 0.), (0., radius, 0.), SIX, -SIX)
        up = LinePath(tuple(map(float, bottom.evaluate(1.))), tuple(map(float, top.evaluate(0.))))
        down = LinePath(tuple(map(float, top.evaluate(1.))), tuple(map(float, bottom.evaluate(0.))))
        return MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in (bottom, up, top, down)),))

    def query(radius):
        point = (radius*math.cos(.5*SIX), radius*math.sin(.5*SIX), 20.)
        return LinePath(point, point)

    assert _applies(strip(R), strip(R).boundaries[0])
    assert strip(R).contains(query(R), 0., _TOL)
    big = strip(1e7)
    assert not _applies(big, big.boundaries[0])
    assert big.contains(query(1e7), 0., _TOL) is _PARENT["cylinder_1e7"]


# --- review R5: domain closure (seam period, arc phase, chart-test overflow, cache mutation) -----------------
#
# Each repro is outside the closed domain and keeps the parent's answer (recorded on the 2608332 export), and
# its in-domain counterpart gives the oracle's answer. See redesign/T20-DOMAIN-AUDIT.md (T2-0 work folder).

_PARENT_R5 = {
    "seam": [False, True, True],                 # 3, 9, -6 deg at z = 20; truth: True, False, False
    "seam_clip": [[], [[0., 1.]]],               # generator z 19..21 at 3 and 9 deg
    "phase": [True, True, False, False],         # float (boundary False, True), exact (False, True)
    "plane_overflow": [False, False, True, True],
}


def _seam_strip():
    """Review R5 P2-1: the tripod strip (0..6 deg) on a full-turn chart that starts at 6 deg, i.e. the strip's
    6-deg edge lies on the chart's cut; the period choice of MaterialDomain.uv mixes the loop's periods."""
    from dataclasses import replace
    domain, _up = _strip()
    return replace(domain, support=replace(domain.support, start_angle=SIX))


def test_r5_cylinder_loop_on_the_chart_cut_is_outside_the_domain():
    from anygeometry.material_arrangement import _clip_intervals
    domain = _seam_strip()
    assert not _applies(domain, domain.boundaries[0])
    angles = [math.radians(a) for a in (3., 9., -6.)]
    assert [_contains(domain, a, 20.) for a in angles] == _PARENT_R5["seam"]
    clips = []
    for a in angles[:2]:
        point = (R*math.cos(a), R*math.sin(a))
        clips.append([list(map(float, i)) for i in _clip_intervals(
            domain, LinePath((*point, 19.), (*point, 21.)), _TOL, lambda: None, isolated_contacts=[])])
    assert clips == _PARENT_R5["seam_clip"]
    # In-domain counterpart (the tripod chart, starting at -0.25 rad): the oracle's answers.
    inside, _up = _strip()
    assert _applies(inside, inside.boundaries[0])
    assert [_contains(inside, a, 20.) for a in angles] == [True, False, False]


@pytest.mark.parametrize("start, sweep, admitted", (
    (0., math.tau, False),                     # full turn: the cut is the start, 0 deg, inside the strip
    (-.25, math.tau, True),                    # the tripod arms' chart
    (math.pi, -math.tau, True),
    (-.5, 1., True),                           # partial: cut at start + sweep/2 + pi = pi
    (.1, -1., True),                           # cut at 0.1 - 0.5 + pi
    (-math.pi-.5, 1., False),                  # partial, cut at 0 deg
))
def test_r5_cylinder_seam_rule(start, sweep, admitted):
    """A strip from -3 to 3 deg is admitted exactly when the chart's cut (start + sweep/2 + pi; the start for
    a full turn) is away from it; then its answers are the oracle's. Queries outside the cut-free window
    (and non-finite ones) keep the parent rule."""
    from anygeometry.surfaces import Cylinder
    three = math.radians(3.)
    support = Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius=R,
                       height=52., start_angle=start, sweep_angle=sweep)
    bottom = EllipticArc((0., 0., 4.), (R, 0., 0.), (0., R, -1.), -three, 2*three)
    top = EllipticArc((0., 0., 45.), (R, 0., 0.), (0., R, -1.), three, -2*three)
    up = LinePath(_point(bottom, 1.), _point(top, 0.))
    down = LinePath(_point(top, 1.), _point(bottom, 0.))
    for loop in _variants([bottom, up, top, down]):
        domain = MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in loop),))
        assert _applies(domain, domain.boundaries[0]) is admitted
        if admitted:
            assert [_contains(domain, math.radians(a), 20.) for a in (0., 2., 5., -9.)] == [True, True, False, False]
    if admitted:
        scale = domain._float_ray_scale(domain.boundaries[0])
        assert domain._junction_query_admitted(domain.uv(up, .5), scale)
        assert not domain._junction_query_admitted((float("nan"), .5), scale)
        if abs(sweep) == math.tau:
            assert not domain._junction_query_admitted((0., .5), scale)     # on the cut
            assert not domain._junction_query_admitted((1.02, .5), scale)   # shifted by a period


def _phase_loop(start):
    """Review R5 P2-2: a unit arc of sweep 2 starting at ``start``, closed by its chord."""
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), start, 2.)
    return (ArrangementPath(arc), ArrangementPath(LinePath(_point(arc, 1.), _point(arc, 0.))))


def test_r5_arc_phase_is_part_of_the_scale():
    from fractions import Fraction
    query = (math.sqrt(.75)+1e-6, .5)                         # 866 tolerances outside the unit circle
    loop = _phase_loop(1e12)
    domain = MaterialDomain(0, _plane(), (loop,))
    assert not _applies(domain, loop)
    got = [_answer(domain, loop, q, b) for q in (query, tuple(map(Fraction, query))) for b in (False, True)]
    assert got == _PARENT_R5["phase"]
    # In-domain counterpart: the same arc with its start reduced to [0, 2 pi).
    loop = _phase_loop(math.fmod(1e12, math.tau))
    domain = MaterialDomain(0, _plane(), (loop,))
    assert _applies(domain, loop)
    assert [_answer(domain, loop, q, b) for q in (query, tuple(map(Fraction, query))) for b in (False, True)] == [False]*4


def test_r5_plane_chart_test_does_not_overflow():
    from fractions import Fraction
    from anygeometry.material_arrangement import _plane_chart_conditioned
    from anygeometry.surfaces import Plane
    s = 1e100                                                  # condition ~2e12; |u x v|^2 overflowed to inf
    loop = _square_loop()
    domain = MaterialDomain(0, Plane((0., 0., 0.), (s, 0., 0.), (s, s*1e-12, 0.)), (loop,))
    assert not _applies(domain, loop)
    uv = (-499999999999.99994/s, 500000000000./s)
    got = [_answer(domain, loop, q, b) for q in (uv, tuple(map(Fraction, uv))) for b in (False, True)]
    assert got == _PARENT_R5["plane_overflow"]
    # A uniformly scaled orthonormal chart stays admitted inside [2^-128, 2^128], is refused outside it.
    domain = MaterialDomain(0, Plane((0., 0., 0.), (1e30, 0., 0.), (0., 1e30, 0.)), (loop,))
    assert _applies(domain, loop)
    assert not _plane_chart_conditioned((1e39, 0., 0.), (0., 1e39, 0.))
    assert not _plane_chart_conditioned((float("inf"), 0., 0.), (0., 1., 0.))


def test_r5_scale_is_not_reused_after_its_coordinates_change():
    """Review R5 P2-4: LinePaths built on lists; the lists change after the first query."""
    from fractions import Fraction
    points = [[x, y, 0.] for x, y in _r3_vertices(100.)]
    loop = tuple(ArrangementPath(LinePath(points[i], points[(i+1) % len(points)])) for i in range(len(points)))
    domain = MaterialDomain(0, _plane(), (loop,))
    assert _applies(domain, loop)
    domain.contains_loop(loop, (0., .5), _TOL)
    for point, (x, y) in zip(points, _r3_vertices()):
        point[:] = [x, y, 0.]
    fresh = MaterialDomain(0, _plane(), (loop,))
    assert not _applies(fresh, loop) and not _applies(domain, loop)
    for uv in (tuple(map(Fraction, _R3_QUERY)), _R3_QUERY):
        for boundary in (False, True):
            assert _answer(domain, loop, uv, boundary) is _answer(fresh, loop, uv, boundary) is False
    # Immutable payloads are still cached.
    loop = tuple(ArrangementPath(c) for c in _off_band_loop(_r3_vertices(100.)))
    domain = MaterialDomain(0, _plane(), (loop,))
    assert _applies(domain, loop) and domain._float_scale_cache[2][0] is not None
