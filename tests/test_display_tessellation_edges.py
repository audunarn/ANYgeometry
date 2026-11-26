"""Shared edge samples for display (N1, owner decision 2026-11-25): lines, circles and ellipses on
planes, cylinders and cones, each divided in closed form; everything else is a typed refusal.

The checks are independent of the divider: dense exact tangents (65 per interval), the support's
own azimuth at dense points, and the distance of dense curve points from the drawn segment.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

import display_tessellation_corpus as corpus

from anygeometry import display_tessellation as dt
from anygeometry.curves import Straight
from anygeometry.errors import GeometryError
from anygeometry.exact_curves import EllipticArc
from anygeometry.line_ranges import LineRange
from anygeometry.surfaces import Cone, Cylinder, Plane

DENSE = 65
EPS = np.finfo(float).eps
_PUBLIC = {name: getattr(dt, name) for name in ("edge_display_samples", "display_rounding_allowance")}


@pytest.fixture(autouse=True)
def _public_calls_under_a_raising_numpy_error_state(monkeypatch):
    """N2 round-2 audit (owner decision 3): every public call in these tests runs under
    ``np.errstate(all="raise")``, so no raw ``FloatingPointError`` can depend on the caller's
    ``np.seterr`` (fixtures are built under the default state; only the public calls are strict)."""
    for name, function in _PUBLIC.items():
        def strict(*args, _function=function, **kwargs):
            with np.errstate(all="raise"):
                return _function(*args, **kwargs)
        monkeypatch.setattr(dt, name, strict)


def test_the_public_calls_are_strict_in_these_tests():
    assert dt.edge_display_samples is not _PUBLIC["edge_display_samples"]
    with pytest.raises(FloatingPointError):
        with np.errstate(all="raise"):
            np.float64(1e-300) * np.float64(1e-300)


def _angles(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("...i,...i->...", a, b))


def _azimuth(support, points):
    offset = np.atleast_2d(points) - support.origin
    radial = offset - np.outer(offset @ support.axis, support.axis)
    return np.arctan2(radial @ support.circumferential_direction, radial @ support.radial_direction)


def _distance_to_segment(points, a, b):
    """Largest distance of ``points`` from the drawn segment ``a``-``b``."""
    d = b - a
    t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
    return float(np.linalg.norm(points - a - np.outer(t, d), axis=1).max())


def _case_edges(case):
    g = case.model
    edges = set()
    for face_id in case.faces.values():
        face = g.faces[face_id]
        edges.update(use.edge for loop in (face.loop, *face.holes) for use in loop)
    if "edge" in case.info:
        edges.add(case.info["edge"])
    return sorted(edges)


def _curved_edges(case):
    g = case.model
    return [e for e in _case_edges(case) if not isinstance(g.edges[e].curve, (Straight, LineRange))]


def _round_supports(g, edge_id):
    found = []
    for face_id in g.faces_using_edge(edge_id):
        support = g.faces[face_id].surface
        if isinstance(support, (Cylinder, Cone)) and not any(support is s for s in found):
            found.append(support)
    return found


def _default_allowance(supports, step):
    radii = [s.radius if isinstance(s, Cylinder) else max(s.radius_start, s.radius_end) for s in supports]
    return min(radii) * (1 - math.cos(step / 2)) if radii else None


def _exact_line_points(g, edge_id, t):
    """The line's points at parameters ``t`` in exact rational arithmetic (``a + t (b - a)``, ``a``,
    ``b`` its evaluated ends): the oracle measures the line itself, not the evaluator's rounding."""
    a, b = (np.asarray(p, dtype=float) for p in g.evaluate_edge_many(edge_id, [0., 1.]))
    start = [Fraction(float(x)) for x in a]
    step = [Fraction(float(y)) - Fraction(float(x)) for x, y in zip(a, b)]
    return [[start[i] + Fraction(float(s)) * step[i] for i in range(3)] for s in t]


def _exact_distance_to_segment(points, a, b):
    """Largest distance of exact rational ``points`` from the drawn segment ``a``-``b`` (exact
    until the final square root)."""
    a = [Fraction(float(x)) for x in a]
    d = [Fraction(float(y)) - x for x, y in zip(a, b)]
    dd = sum(x * x for x in d)
    worst = 0.0
    for p in points:
        q = [x - y for x, y in zip(p, a)]
        t = min(max(sum(x * y for x, y in zip(q, d)) / dd, Fraction(0)), Fraction(1)) if dd else Fraction(0)
        worst = max(worst, math.sqrt(float(sum((x - t * y) ** 2 for x, y in zip(q, d)))))
    return worst


def _radial(support, points, exact):
    """(radial, circumferential) coordinates of ``points`` about ``support``'s axis."""
    if not exact:
        offset = np.atleast_2d(points) - support.origin
        return offset @ support.radial_direction, offset @ support.circumferential_direction
    origin = [Fraction(float(x)) for x in support.origin]
    frame = [[Fraction(float(x)) for x in v] for v in (support.radial_direction, support.circumferential_direction)]
    xy = [[float(sum((p[i] - origin[i]) * f[i] for i in range(3))) for f in frame] for p in points]
    return np.array([q[0] for q in xy]), np.array([q[1] for q in xy])


def _azimuth_change(support, points, floor, exact=False):
    """``(range, backtrack)`` of the azimuth about ``support``'s axis over ``points``: the range of
    its continuous lift and its largest step against the net direction.  Only points at least
    ``floor`` from the axis count (the declared domain defines an azimuth there; a cone's apex
    has none)."""
    x, y = _radial(support, points, exact)
    keep = np.hypot(x, y) >= floor
    if keep.sum() < 2:
        return 0.0, 0.0
    lift = np.unwrap(np.arctan2(y[keep], x[keep]))
    steps = np.diff(lift) * (1.0 if lift[-1] >= lift[0] else -1.0)
    return float(lift.max() - lift.min()), float(max(0.0, -steps.min()))


def _dense_ok(g, edge_id, samples, angle, sag=None, supports=None, count=DENSE, label=None):
    """Every drawn segment, against the module's ONE documented rounding allowance
    (``display_rounding_allowance``, no other slack), which is itself at most 2 % of the angle and
    sag:

    * tangent turning: the largest angle of any dense tangent from the interval's first or last
      (a conic's tangent turns monotonically, so this is its turning; the noise of two samples,
      not of every one);
    * support azimuth: the range of the dense azimuth's lift, and no step against its net
      direction above the allowance (monotone);
    * sag: the largest distance of dense curve points from the drawn segment.

    A line is measured exactly (rational points); a circle or an ellipse at the evaluator's points."""
    supports = _round_supports(g, edge_id) if supports is None else supports
    allowance = dt.display_rounding_allowance(g, edge_id, dt.DisplayTolerance(angle, sag))
    # each support's OWN floor (r5: a floor shared across supports skipped a small one entirely)
    floors = {id(s): f for s, f in zip(_round_supports(g, edge_id), allowance.axis_floors)}
    assert allowance.angle <= .02 * angle, (label, edge_id, allowance)
    assert sag is None or allowance.sag <= .02 * sag, (label, edge_id, allowance)
    line = isinstance(g.edges[edge_id].curve, (Straight, LineRange))
    t = samples.parameters
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0.0)
    limit = angle + allowance.angle
    for k, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
        dense = np.linspace(t0, t1, count)
        tangent = g.edge_tangent_many(edge_id, dense)
        turning = max(_angles(tangent[:1], tangent).max(), _angles(tangent[-1:], tangent).max())
        assert turning <= limit, (label, edge_id, t0, t1, turning, limit)
        points = _exact_line_points(g, edge_id, dense) if line else g.evaluate_edge_many(edge_id, dense)
        for support in supports:
            turned, back = _azimuth_change(support, points, floors[id(support)], exact=line)
            assert turned <= limit and back <= allowance.angle, (label, edge_id, t0, t1, turned, back, limit)
        if sag is not None:
            if line:
                distance = _exact_distance_to_segment(points, samples.points[k], samples.points[k + 1])
            else:
                distance = _distance_to_segment(points, samples.points[k], samples.points[k + 1])
            assert distance <= sag + allowance.sag, (label, edge_id, t0, t1, distance, sag, allowance.sag)


def _refused(call, code):
    with pytest.raises(dt.DisplayTrimRefused) as caught:
        call()
    assert caught.value.code == code, caught.value


CASES = sorted(corpus.ALL)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("step", corpus.STEPS, ids=("fine", "interaction"))
def test_every_interval_turns_at_most_one_step(name, step):
    case = corpus.ALL[name]()
    g = case.model
    tol = dt.DisplayTolerance(step)
    for edge_id in _case_edges(case):
        samples = dt.edge_display_samples(g, edge_id, tol)
        t = samples.parameters
        # the drawn points are the curve at those parameters (ends: the vertices, within tolerance)
        exact = g.evaluate_edge_many(edge_id, t)
        assert np.array_equal(samples.points[1:-1], exact[1:-1])
        assert np.abs(samples.points[[0, -1]] - exact[[0, -1]]).max() <= 1e-9 * max(1.0, np.abs(exact).max())
        supports = _round_supports(g, edge_id)
        _dense_ok(g, edge_id, samples, step, _default_allowance(supports, step), supports, label=name)


@pytest.mark.parametrize("name,sag", [("shared_circle", 1e-4), ("steep_cuts", 1e-4), ("ellipse_edge_20", 1e-4),
                                      ("skew_ellipse_edge", 1e-4), ("cone_section", 1e-4),
                                      ("cone_section_steep", 1e-4), ("wavy_band", 1e-4),
                                      ("tiny_offset_cylinder", 1e-12)])
def test_an_explicit_sag_is_honoured(name, sag):
    """A sag tighter than the angular default bounds every drawn segment's distance from its curve
    (lines included: their only sag is a vertex offset)."""
    case = corpus.ALL[name]()
    g = case.model
    for edge_id in _case_edges(case):
        samples = dt.edge_display_samples(g, edge_id, dt.DisplayTolerance(corpus.INTERACTION_STEP, sag))
        assert samples.points.shape == (len(samples.parameters), 3)
        _dense_ok(g, edge_id, samples, corpus.INTERACTION_STEP, sag, label=name)


@pytest.mark.parametrize("name", CASES)
def test_samples_end_on_the_vertices_and_repeat_bit_for_bit(name):
    case = corpus.ALL[name]()
    g = case.model
    tol = dt.DisplayTolerance(corpus.FINE_STEP)
    for edge_id in sorted(g.edges):
        first = dt.edge_display_samples(g, edge_id, tol)
        points, parameters = first.points.tobytes(), first.parameters.tobytes()   # before the second call
        again = dt.edge_display_samples(g, edge_id, tol)
        assert points == again.points.tobytes() == first.points.tobytes()
        assert parameters == again.parameters.tobytes() == first.parameters.tobytes()
        edge = g.edges[edge_id]
        assert first.edge_id == edge_id
        assert np.array_equal(first.points[0], g.vertex_position(edge.start))
        assert np.array_equal(first.points[-1], g.vertex_position(edge.end))
        assert not first.points.flags.writeable and not first.parameters.flags.writeable
        if isinstance(edge.curve, (Straight, LineRange)) and not _round_supports(g, edge_id):
            assert first.parameters.tolist() == [0.0, 1.0]


def test_the_20_to_1_half_ellipse_meets_a_tight_sag_by_exact_bisection():
    """The exact ellipse sag (parametric midpoint) refines only where needed: sag 1e-4 on the 20:1
    half ellipse is met in fewer than 1024 intervals (the former rho_max bound needed more than
    4096); every interval's dense distance is at most the sag."""
    case = corpus.ellipse_edge()
    g, edge = case.model, case.info["edge"]
    for sag in (1e-3, 1e-4):
        samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(corpus.INTERACTION_STEP, sag))
        assert len(samples.parameters) - 1 < 1024
        _dense_ok(g, edge, samples, corpus.INTERACTION_STEP, sag, count=257)


def test_the_20_to_1_half_ellipse_meets_the_step_on_every_emitted_chord():
    """ANYfem round 2: chord turning measured over 64 chords per emitted interval."""
    case = corpus.ellipse_edge()
    g, edge = case.model, case.info["edge"]
    for step in corpus.STEPS:
        values = dt.edge_display_samples(g, edge, dt.DisplayTolerance(step)).parameters
        for t0, t1 in zip(values[:-1], values[1:]):
            chords = np.diff(g.evaluate_edge_many(edge, np.linspace(t0, t1, 65)), axis=0)
            assert _angles(chords[:-1], chords[1:]).sum() * 64 / 63 <= step * 1.02


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
    again = dt.edge_display_samples(g.clone(), shared, tol)
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


def test_a_later_incident_support_still_bounds_the_edge():
    """The lower cut is used by a plane face first and by the cylinder second."""
    case = corpus.steep_cuts(with_plane=True)
    g = case.model
    lower = case.info["lower"]
    assert [type(g.faces[f].surface).__name__ for f in g.faces_using_edge(lower)] == ["Plane", "Cylinder"]
    for step in corpus.STEPS:
        samples = dt.edge_display_samples(g, lower, dt.DisplayTolerance(step))
        assert len(samples.parameters) - 1 >= math.ceil(math.pi / 2 / step - 1e-9)


def test_the_stricter_of_two_curved_supports_bounds_the_edge():
    """The lower arc lies on a coaxial cylinder (first face) and on the 8e-10 off-axis cylinder
    (second face), whose azimuth turns 0.08 % faster: at .1 rad the second needs three steps."""
    case = corpus.tiny_offset_cylinder(coaxial_below=True)
    g, low, support = case.model, case.info["low"], case.info["support"]
    supports = _round_supports(g, low)
    assert len(supports) == 2 and supports[1] is support
    ends = _azimuth(support, g.evaluate_edge_many(low, [0., 1.]))
    span = (ends[1] - ends[0]) % (2 * np.pi)
    assert .2 < span < .2003
    samples = dt.edge_display_samples(g, low, dt.DisplayTolerance(.1))
    azimuth = np.unwrap(_azimuth(support, samples.points))
    assert np.abs(np.diff(azimuth)).max() <= .1 + dt.display_rounding_allowance(g, low, dt.DisplayTolerance(.1)).angle
    _dense_ok(g, low, samples, .1, supports=[support], count=257)


@pytest.mark.parametrize("angle", [0.0, -0.1, math.pi, float("nan"), float("inf"), 1e-320])
def test_tolerance_rejects_unusable_angles(angle):
    with pytest.raises(GeometryError):
        dt.DisplayTolerance(angle)


@pytest.mark.parametrize("sag", [0.0, -1.0, float("nan")])
def test_tolerance_rejects_unusable_sag(sag):
    with pytest.raises(GeometryError):
        dt.DisplayTolerance(.1, sag)


def test_refusals_are_typed():
    assert dt.REFUSAL_CODES == {"SUPPORT_UNSUPPORTED", "TRIM_INVALID", "TRIM_WINDS_SUPPORT", "EDGE_UNSUPPORTED",
                                "OUT_OF_DOMAIN"}
    with pytest.raises(ValueError):
        dt.DisplayTrimRefused("NOT_A_CODE", "x")
    refusal = dt.DisplayTrimRefused("TRIM_INVALID", "x")
    assert isinstance(refusal, GeometryError) and refusal.code == "TRIM_INVALID"
    domain = dt.DisplayDomainRefused("x")
    assert isinstance(domain, dt.DisplayTrimRefused) and domain.code == "OUT_OF_DOMAIN"


def test_extreme_tolerances_are_refused_typed():
    case = corpus.ALL["shared_circle"]()
    _refused(lambda: dt.edge_display_samples(case.model, case.info["shared"], dt.DisplayTolerance(.1, 1e-18)),
             "EDGE_UNSUPPORTED")
    with pytest.raises(GeometryError):
        dt.edge_display_samples(case.model, case.info["shared"], .1)


def test_more_than_the_interval_cap_is_refused():
    g, edge = _arc_edge((0., 0., 0.), 0., 2 * math.pi - 1e-3)
    assert len(dt.edge_display_samples(g, edge, dt.DisplayTolerance(2 * math.pi / 4000)).parameters) - 1 <= 4096
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(1e-5)), "EDGE_UNSUPPORTED")


# ---------------------------------------------------------------------------------------------
# exact divisions
# ---------------------------------------------------------------------------------------------
def _arc_edge(centre, start, sweep, u=(1., 0., 0.), v=(0., 1., 0.)):
    g = corpus.GeometryModel()
    arc = EllipticArc(centre, u, v, start, sweep)
    ends = arc.evaluate(np.array([0., 1.]))
    return g, g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), arc)


def _tangent_angle_lift(g, edge_id, t):
    """Tangent direction angle in the curve's plane, unwrapped along increasing ``t``."""
    curve = g.edges[edge_id].curve
    u, v = np.asarray(curve.u_vector), np.asarray(curve.v_vector)
    e1 = u / np.linalg.norm(u)
    w = v - (v @ e1) * e1
    e2 = w / np.linalg.norm(w)
    tangent = g.edge_tangent_many(edge_id, t)
    return np.unwrap(np.arctan2(tangent @ e2, tangent @ e1))


@pytest.mark.parametrize("angle", [corpus.FINE_STEP, corpus.INTERACTION_STEP, .05])
def test_a_skew_ellipse_is_divided_uniformly_in_tangent_angle(angle):
    """u and v neither orthogonal nor of equal length: the division is the exact psi-uniform one,
    n = ceil(total turning / angle) equal steps of tangent angle (no support, no sag)."""
    case = corpus.skew_ellipse_edge()
    g, edge = case.model, case.info["edge"]
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(angle))
    lift = _tangent_angle_lift(g, edge, np.linspace(0., 1., 200001))
    total = abs(lift[-1] - lift[0])
    n = math.ceil(total / angle - 1e-9)
    assert len(samples.parameters) - 1 == n
    psi = _tangent_angle_lift(g, edge, samples.parameters)
    assert np.abs(np.abs(np.diff(psi)) - total / n).max() <= 1e-9
    _dense_ok(g, edge, samples, angle, count=257)


@pytest.mark.parametrize("name", ["skew_ellipse_edge", "ellipse_edge_20", "steep_cuts", "cone_section_steep"])
def test_the_ellipse_sag_is_exact_at_the_parametric_midpoint(name):
    """An ellipse is an affine image of a circle: over every emitted interval the farthest point
    from the drawn segment is the curve at the theta midpoint (dense check to rounding), and the
    refinement is not grossly over-fine (some interval sags more than 1/8 of the allowance)."""
    case = corpus.ALL[name]()
    g = case.model
    sag = 1e-4
    for edge in _curved_edges(case):
        curve = g.edges[edge].curve
        samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, sag))
        t = samples.parameters
        _dense_ok(g, edge, samples, .2, sag)
        distances = []
        for k, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
            a, b = g.evaluate_edge_many(edge, [t0, t1])
            dense = g.evaluate_edge_many(edge, np.linspace(t0, t1, 1025))
            middle = g.evaluate_edge_many(edge, [0.5 * (t0 + t1)])
            far = _distance_to_segment(dense, a, b)
            at_middle = _distance_to_segment(middle, a, b)
            assert at_middle >= far * (1 - 1e-6) - 1e-13, (name, edge, t0, t1)
            distances.append(far)
        if not np.allclose(np.linalg.norm(curve.u_vector), np.linalg.norm(curve.v_vector), rtol=1e-12):
            assert max(distances) > sag / 8


def test_a_circle_off_its_cylinder_axis_gets_the_exact_azimuth_division():
    """The tiny_offset_cylinder arc (centre 8e-10 off a 1e-6 cylinder's axis): its parameters are
    the union of the uniform tangent division and the exact uniform division of the support
    azimuth (equal azimuth steps to rounding), not a measured one."""
    case = corpus.tiny_offset_cylinder()
    g, edge, support = case.model, case.info["low"], case.info["support"]
    angle = .1
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(angle))
    t = samples.parameters
    ends = _azimuth(support, g.evaluate_edge_many(edge, [0., 1.]))
    span = (ends[1] - ends[0]) % (2 * np.pi)
    n_azimuth = math.ceil(span / angle - 1e-9)
    n_tangent = math.ceil(.2 / angle - 1e-9)
    assert (n_azimuth, n_tangent) == (3, 2)
    assert len(t) - 1 == n_azimuth + n_tangent - 1           # the two divisions share only the ends
    assert np.any(t == .5)                                   # the circle's own uniform step
    azimuth = (_azimuth(support, g.evaluate_edge_many(edge, t)) - ends[0]) % (2 * np.pi)
    targets = span * np.arange(1, n_azimuth) / n_azimuth
    assert all(np.abs(azimuth - target).min() <= 1e-12 for target in targets)
    _dense_ok(g, edge, samples, angle, supports=[support], count=257)


@pytest.mark.parametrize("name", ["cone_section", "cone_section_steep", "cone_section_far"])
def test_ellipses_on_cones_are_divided_by_their_azimuth_about_the_axis(name):
    """A plane section of a cone has the axis at a focus of its projection: the azimuth is
    monotone but not uniform in the arc parameter."""
    case = corpus.ALL[name]()
    g, arc, support = case.model, case.info["arc"], case.info["support"]
    for angle in (corpus.FINE_STEP, corpus.INTERACTION_STEP, .05):
        samples = dt.edge_display_samples(g, arc, dt.DisplayTolerance(angle))
        dense = np.unwrap(_azimuth(support, g.evaluate_edge_many(arc, np.linspace(0., 1., 100001))))
        total = abs(dense[-1] - dense[0])
        assert len(samples.parameters) - 1 >= math.ceil(total / angle - 1e-9)
        _dense_ok(g, arc, samples, angle, _default_allowance([support], angle), [support], count=129, label=name)


@pytest.mark.parametrize("name", ["cone_apex_30", "cone_apex_220", "cone_section", "cone_section_far", "far_apex_cone"])
def test_generators_through_the_apex_have_constant_azimuth(name):
    """Lines from a cone's exact apex (radius exactly 0) are generators: no azimuth division."""
    case = corpus.ALL[name]()
    g = case.model
    lines = [e for e in _case_edges(case) if isinstance(g.edges[e].curve, (Straight, LineRange))]
    assert lines
    for step in corpus.STEPS:
        for edge in lines:
            assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(step)).parameters.tolist() == [0., 1.]


def test_the_apex_of_a_translated_cone_is_recognised_exactly():
    """Apex at z = 1e4 (origin there, radius_start exactly 0): the apex vertex is recognised (its
    generators are drawn whole, not refused as reaching the axis)."""
    case = corpus.far_apex_cone()
    g = case.model
    rising, falling = case.info["generators"]
    assert np.array_equal(g.vertex_position(g.edges[rising].start), (0., 0., 1e4))
    for edge in (rising, falling):
        assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)).parameters.tolist() == [0., 1.]
    arc = case.info["arc"]
    _dense_ok(g, arc, dt.edge_display_samples(g, arc, dt.DisplayTolerance(.1)), .1, count=257)


def test_a_line_range_on_a_tiny_cylinder_follows_its_azimuth_exactly():
    """Lines between azimuths -.15 and .15 on a 1e-7 cylinder (within its tolerance): three equal
    azimuth steps of .1."""
    g = corpus.GeometryModel()
    r = 1e-7
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), r, 1.0)
    pts = {(a, z): (r * math.cos(a), r * math.sin(a), z) for a in (-.15, .15) for z in (.2, .8)}
    ids = {k: g.add_point(*v) for k, v in pts.items()}
    loop = [((-.15, .2), (.15, .2)), ((.15, .2), (.15, .8)), ((.15, .8), (-.15, .8)), ((-.15, .8), (-.15, .2))]
    edges = [g.add_curve(ids[a], ids[b], LineRange(pts[a], pts[b])) for a, b in loop]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    for edge in (edges[0], edges[2]):
        samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))
        assert len(samples.parameters) - 1 == 3
        azimuth = _azimuth(support, samples.points)
        assert np.abs(np.abs(np.diff(azimuth)) - .1).max() <= 1e-12
        _dense_ok(g, edge, samples, .1, supports=[support], count=257)
    for edge in (edges[1], edges[3]):                      # rulings: constant azimuth
        assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)).parameters.tolist() == [0., 1.]


def _frustum(small, big=5e8, sweep=1.0):
    """A frustum with radii ``small`` at z = 0 and ``big`` at z = 1e9 (no apex): a band from its
    narrow circle up its generators to z = 1."""
    g = corpus.GeometryModel()
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), small, big, 1e9, 0.0, sweep)
    top_radius = (1 - 1e-9) * small + 1e-9 * big
    p0, p1 = (g.add_point(small * math.cos(a), small * math.sin(a), 0.) for a in (0., sweep))
    q0, q1 = (g.add_point(top_radius * math.cos(a), top_radius * math.sin(a), 1.) for a in (0., sweep))
    bottom = g.add_curve(p0, p1, EllipticArc((0., 0., 0.), (small, 0., 0.), (0., small, 0.), 0., sweep))
    top = g.add_curve(q0, q1, EllipticArc((0., 0., 1.), (top_radius, 0., 0.), (0., top_radius, 0.), 0., sweep))
    rising, falling = g.add_line(p1, q1), g.add_line(q0, p0)
    g.add_face((bottom, rising, top, falling), corners=(0, 1, 2, 3), surface=support)
    return g, support, bottom, (rising, falling)


def test_a_narrow_frustum_end_is_not_a_false_apex():
    """Radii 3e-6 and 5e8 near the origin: the narrow end keeps its azimuth (no apex: neither
    radius is 0), its generators are drawn whole."""
    g, support, bottom, generators = _frustum(3e-6)
    samples = dt.edge_display_samples(g, bottom, dt.DisplayTolerance(.1))
    assert len(samples.parameters) - 1 == 10
    _dense_ok(g, bottom, samples, .1, supports=[support], count=257)
    for edge in generators:
        assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)).parameters.tolist() == [0., 1.]


def test_the_circle_division_honours_the_coordinates_resolution():
    """A translated arc whose sag is below its coordinates' resolution: refused; a resolvable sag
    at the same place is met."""
    g, edge = _arc_edge((1e9 - 2., 0., 0.), 0., .03)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, 4 * EPS * 1e9 * .9)),
             "EDGE_UNSUPPORTED")
    _dense_ok(g, edge, dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, 1e-3)), .0153, 1e-3, count=257)


def test_an_exact_circle_division_uses_the_full_tolerance():
    """The closed-form circle division is exact: a .4 rad arc at .2 is two exact .2 rad steps."""
    g, edge = _arc_edge((0., 0., 0.), 0., .4)
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, .00505))
    assert samples.parameters.tolist() == [0., .5, 1.]
    _dense_ok(g, edge, samples, .2, .00505, count=257)


# ---------------------------------------------------------------------------------------------
# refusals: one per excluded kind
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(corpus.REFUSED))
def test_refused_corpus_cases_are_refused_with_their_code(name):
    """Every edge of a refused case is drawn or refused with the case's code; at least one is
    refused, and every edge used by a face on another support than a plane, cylinder or cone is."""
    builder, code = corpus.REFUSED[name]
    case = builder()
    g = case.model
    refused = 0
    for edge_id in _case_edges(case):
        kinds = {type(g.faces[f].surface) for f in g.faces_using_edge(edge_id)}
        try:
            dt.edge_display_samples(g, edge_id, dt.DisplayTolerance(corpus.FINE_STEP))
        except dt.DisplayTrimRefused as error:
            assert error.code == code, (name, edge_id, error)
            refused += 1
        else:
            assert kinds <= {Plane, Cylinder, Cone}, (name, edge_id, kinds)
    assert refused >= 1


def test_a_spline_edge_is_refused():
    case = corpus.spline_edge()
    _refused(lambda: dt.edge_display_samples(case.model, case.info["edge"], dt.DisplayTolerance(.2)),
             "EDGE_UNSUPPORTED")


def test_a_nurbs_edge_is_refused():
    case = corpus.nurbs_edge()
    _refused(lambda: dt.edge_display_samples(case.model, case.info["edge"], dt.DisplayTolerance(.2)),
             "EDGE_UNSUPPORTED")


def test_a_curve_on_a_surface_is_refused():
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    sheet = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                         (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., .2))))
    curve = CurveOnSurface(sheet, NurbsCurve(1, (0., 0., 1., 1.), ((.1, .2), (.9, .7))))
    g = corpus.GeometryModel()
    ends = curve.evaluate(np.array([0., 1.]))
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), curve)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_an_edge_used_by_an_extrusion_face_is_refused():
    """The lower cut of the steep band is shared with an extrusion face: the cylinder alone does
    not make it drawable here (the caller keeps its own route for that edge)."""
    case = corpus.steep_cuts(with_extrusion=True)
    _refused(lambda: dt.edge_display_samples(case.model, case.info["lower"], dt.DisplayTolerance(.2)),
             "SUPPORT_UNSUPPORTED")


def test_an_edge_of_a_coons_face_is_refused():
    case = corpus.unsupported_support()
    g = case.model
    for edge_id in _case_edges(case):
        _refused(lambda: dt.edge_display_samples(g, edge_id, dt.DisplayTolerance(.2)), "SUPPORT_UNSUPPORTED")


def test_an_edge_of_a_face_with_an_explicit_parameterization_is_refused():
    g = corpus.GeometryModel()
    p = [g.add_point(*xyz) for xyz in ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))]
    edges = [g.add_line(p[0], p[1]), g.add_line(p[1], p[2]), g.add_line(p[2], p[0])]
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    face = g.add_face(edges, surface=plane)
    assert dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.2)).parameters.tolist() == [0., 1.]
    g.set_face_parameterization(face, plane)
    _refused(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.2)), "SUPPORT_UNSUPPORTED")


# ---------------------------------------------------------------------------------------------
# The declared numeric domain (owner decision 2026-11-25): at each bound the guarantee holds,
# just beyond it the refusal is typed (OUT_OF_DOMAIN).
# ---------------------------------------------------------------------------------------------
def _refused_out_of_domain(call):
    with pytest.raises(dt.DisplayDomainRefused) as caught:
        call()
    assert caught.value.code == "OUT_OF_DOMAIN" and isinstance(caught.value, dt.DisplayTrimRefused)


def test_the_domain_bounds_are_declared():
    assert dt.MAX_DISPLAY_COORDINATE == 1e9 and dt.MAX_DISPLAY_ANGLE == 1e6
    assert dt.MIN_DISPLAY_RADIUS_RATIO == 1e-9 and dt.MAX_DISPLAY_ELLIPSE_ASPECT == 1e4      # 1e6 before N1 r6
    assert "OUT_OF_DOMAIN" in dt.REFUSAL_CODES


@pytest.mark.parametrize("x,inside", [(1e9, True), (np.nextafter(1e9, 2e9), False), (-1e9, True),
                                      (np.nextafter(-1e9, -2e9), False)])
def test_vertices_at_and_beyond_the_coordinate_bound(x, inside):
    g = corpus.GeometryModel()
    edge = g.add_line(g.add_point(float(x), 0., 0.), g.add_point(float(x) / 2, 1., 0.))
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))  # noqa: E731
    if inside:
        assert call().parameters.tolist() == [0., 1.]
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("x,inside", [(-1e9, True), (np.nextafter(-1e9, -2e9), False)])
def test_line_carriers_at_and_beyond_the_coordinate_bound(x, inside):
    """A LineRange carrier starting at or beyond the bound: drawn whole at it (its whole carrier),
    OUT_OF_DOMAIN beyond it, before any other check (a retained half near the origin too).  Inside
    the bound a retained half is EDGE_UNSUPPORTED (owner decision r4: only a whole carrier)."""
    g = corpus.GeometryModel()
    whole = LineRange((float(x), 0., 0.), (1., 0., 0.))
    edge = g.add_curve(g.add_point(*whole.a), g.add_point(*whole.b), whole)
    half = LineRange((float(x), 0., 0.), (1., 0., 0.), (Fraction(1), Fraction(1, 2)))
    ends = half.evaluate(np.array([0., 1.]))
    part = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), half)
    call = lambda e: (lambda: dt.edge_display_samples(g, e, dt.DisplayTolerance(.2)))  # noqa: E731
    if inside:
        assert call(edge)().parameters.tolist() == [0., 1.]
        _refused(call(part), "EDGE_UNSUPPORTED")
    else:
        _refused_out_of_domain(call(edge))
        _refused_out_of_domain(call(part))


def test_plane_supports_are_in_the_domain_check():
    g = corpus.GeometryModel()
    p = [g.add_point(*xyz) for xyz in ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))]
    edges = [g.add_line(p[0], p[1]), g.add_line(p[1], p[2]), g.add_line(p[2], p[0])]
    g.add_face(edges, surface=Plane((1e9 + 1., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.2)))


@pytest.mark.parametrize("shift,inside", [(0.0, True), (1e9 - 1e3, True), (2e12, False), (1e14, False)])
def test_steep_cuts_at_and_beyond_the_coordinate_bound(shift, inside):
    """In the domain a far band keeps its support azimuth steps; beyond it, typed refusal."""
    case = corpus.steep_cuts(shift)
    g, lower = case.model, case.info["lower"]
    call = lambda: dt.edge_display_samples(g, lower, dt.DisplayTolerance(.2, 1.0))  # noqa: E731
    if inside:
        samples = call()
        assert len(samples.parameters) - 1 >= math.ceil(math.pi / 2 / .2 - 1e-9)
        _dense_ok(g, lower, samples, .2, 1.0, [case.info["support"]], count=257)
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("z0,inside", [(2. ** 29, True), (2. ** 48, False)])
def test_a_band_far_along_a_long_cylinder(z0, inside):
    g = corpus.GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 4 * z0, -math.pi / 2, math.pi)
    a = math.pi / 4
    z1 = z0 + 2. ** 20
    p00, p10 = corpus._section_point(g, -a, 0., 100., z0), corpus._section_point(g, a, 0., 100., z0)
    p01, p11 = corpus._section_point(g, -a, 0., 100., z1), corpus._section_point(g, a, 0., 100., z1)
    lower = corpus._section(g, p00, p10, 0., 100., z0, -a, a)
    upper = corpus._section(g, p01, p11, 0., 100., z1, -a, a)
    g.add_face((lower, g.add_line(p10, p11), upper, g.add_line(p01, p00)), corners=(0, 1, 2, 3), surface=support)
    call = lambda: dt.edge_display_samples(g, lower, dt.DisplayTolerance(.2, 1.0))  # noqa: E731
    if inside:
        samples = call()
        assert len(samples.parameters) - 1 >= math.ceil(math.pi / 2 / .2 - 1e-9)
        _dense_ok(g, lower, samples, .2, 1.0, [support], count=257)
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("start,inside", [(1e6 - 7., True), (-1e6 + 7., True), (1e6 + 1., False),
                                          (1e15, False)])
def test_arcs_at_and_beyond_the_angle_bound(start, inside):
    """Stored angles up to 1e6 rad keep uniform steps; beyond, typed refusal (no collapse)."""
    g, edge = _arc_edge((0., 0., 0.), start, 2 * math.pi - 1e-3)
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, .005))  # noqa: E731
    if inside:
        _dense_ok(g, edge, call(), .2, .005, count=257)
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("start,sweep,inside", [(1e6 - 2., 2., True), (-1e6, 2., True),
                                                (np.nextafter(1e6, 2e6) - 2., 2., False),
                                                (np.nextafter(-1e6, -2e6), 2., False)])
def test_stored_angles_exactly_at_the_bound(start, sweep, inside):
    g, edge = _arc_edge((0., 0., 0.), float(start), sweep)
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))  # noqa: E731
    if inside:
        _dense_ok(g, edge, call(), .2, count=257)
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("cx,inside", [(1e9 - 2., True), (2. ** 40, False)])
def test_circles_at_and_beyond_the_coordinate_bound(cx, inside):
    g, edge = _arc_edge((cx, 0., 0.), 0., .03)
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, 3e-4))  # noqa: E731
    if inside:
        _dense_ok(g, edge, call(), .0153, 3e-4, count=257)
    else:
        _refused_out_of_domain(call)


def test_ellipse_definition_points_are_in_the_domain_check():
    """Centre and vertices inside; centre + v beyond the bound."""
    g, edge = _arc_edge((0., 0., 0.), 0., .1, (1., 0., 0.), (0., 1e9 + 1e3, 0.))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)))


def test_a_sampled_point_alone_can_cross_the_coordinate_bound():
    """Vertices and definition points inside, the arc's middle at x = 1e9 + 2 outside."""
    g, edge = _arc_edge((1., 0., 0.), -math.pi / 2, math.pi, (1e9 - 1., 0., 0.), (0., 1e9 - 1., 0.))
    dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))         # its middle is exactly 1e9
    g, edge = _arc_edge((3., 0., 0.), -math.pi / 2, math.pi, (1e9 - 1., 0., 0.), (0., 1e9 - 1., 0.))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)))


def _start_for(end, sweep):
    """A start angle whose stored ``start + sweep`` is exactly ``end``."""
    start = end - sweep
    for _ in range(16):
        if start + sweep == end:
            return float(start)
        start = np.nextafter(start, math.inf if start + sweep < end else -math.inf)
    pytest.fail("no start angle reaches the end exactly")


@pytest.mark.parametrize("end,inside", [(1e6, True), (float(np.nextafter(1e6, 2e6)), False),
                                        (-1e6 + 1., True), (float(np.nextafter(-1e6, -2e6)) + 1., False)])
def test_the_cylinder_angle_bound_applies_to_supports(end, inside):
    """A support whose start + sweep (or start) is exactly at 1e6 rad is inside, one ulp beyond
    is not; the band's own arcs carry the same angles reduced modulo 2 pi."""
    sweep = 1.0
    start = _start_for(end, sweep)
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.0, 1.0, start, sweep)
    assert support.start_angle + support.sweep_angle == end
    g = corpus.GeometryModel()
    ids = {(u, v): g.add_point(*support.evaluate(u, v)) for u in (.1, .9) for v in (.2, .8)}
    a0 = math.fmod(start + .1 * sweep, 2 * math.pi)

    def arc(v):
        p, q = ids[.1, v], ids[.9, v]
        return g.add_curve(p, q, EllipticArc((0., 0., v), (1., 0., 0.), (0., 1., 0.), a0, .8 * sweep))

    low, high = arc(.2), arc(.8)
    g.add_face((low, g.add_line(ids[.9, .2], ids[.9, .8]), high, g.add_line(ids[.1, .8], ids[.1, .2])),
               corners=(0, 1, 2, 3), surface=support)
    call = lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.2))  # noqa: E731
    if inside:
        _dense_ok(g, low, call(), .2, count=257)
    else:
        _refused_out_of_domain(call)


def _tiny_cylinder_band(radius):
    """A band on a cylinder of ``radius`` whose largest coordinate is exactly 1024 (its bottom
    at z = -1024): the radius bound there is 1e-9 * 1024."""
    g = corpus.GeometryModel()
    support = Cylinder((0., 0., -1024.), (0., 0., 1.), (1., 0., 0.), radius, 1.0)
    a = .5
    pts = [(radius, 0., -1024.), (radius * math.cos(a), radius * math.sin(a), -1024.),
           (radius * math.cos(a), radius * math.sin(a), -1023.5), (radius, 0., -1023.5)]
    ids = [g.add_point(*p) for p in pts]
    low = g.add_curve(ids[0], ids[1], EllipticArc((0., 0., -1024.), (radius, 0., 0.), (0., radius, 0.), 0., a))
    high = g.add_curve(ids[3], ids[2], EllipticArc((0., 0., -1023.5), (radius, 0., 0.), (0., radius, 0.), 0., a))
    ruling = g.add_curve(ids[3], ids[0], LineRange(pts[3], pts[0]))
    g.add_face((low, g.add_line(ids[1], ids[2]), high, ruling), corners=(0, 1, 2, 3), surface=support)
    return g, ruling, low


@pytest.mark.parametrize("inside", [True, False])
def test_cylinder_radius_at_and_beyond_the_radius_ratio(inside):
    bound = dt.MIN_DISPLAY_RADIUS_RATIO * 1024.
    radius = bound if inside else float(np.nextafter(bound, 0.))
    g, ruling, low = _tiny_cylinder_band(radius)
    for edge in (ruling, low):
        call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))  # noqa: E731
        if inside:
            call()
        else:
            _refused_out_of_domain(call)


@pytest.mark.parametrize("small,inside", [(2e-6, True), (.5e-6, False)])
def test_cone_points_at_and_beyond_the_radius_ratio(small, inside):
    """A frustum (no apex) at z = 1000 whose narrow end sits ``small`` from its axis: the bound
    there is about 1e-9 x 1001; a generator from the narrow end is inside the domain at 2e-6 and
    refused at .5e-6."""
    g = corpus.GeometryModel()
    sweep = 2.0
    support = Cone((0., 0., 1000.), (0., 0., 1.), (1., 0., 0.), small, 1.0, 1.0, 0.0, sweep)
    p0, p1 = (g.add_point(*support.evaluate(u, 0.)) for u in (0., 1.))
    q0, q1 = (g.add_point(*support.evaluate(u, 1.)) for u in (0., 1.))
    bottom = g.add_curve(p0, p1, EllipticArc((0., 0., 1000.), (small, 0., 0.), (0., small, 0.), 0., sweep))
    top = g.add_curve(q0, q1, EllipticArc((0., 0., 1001.), (1., 0., 0.), (0., 1., 0.), 0., sweep))
    generators = (g.add_line(p1, q1), g.add_line(q0, p0))
    g.add_face((bottom, generators[0], top, generators[1]), corners=(0, 1, 2, 3), surface=support)
    for edge in (*generators, bottom):
        call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))  # noqa: E731
        if inside:
            samples = call()
            if edge != bottom:
                assert samples.parameters.tolist() == [0., 1.]
        else:
            _refused_out_of_domain(call)


# ---------------------------------------------------------------------------------------------
# Codex N1 round 1: each finding's geometry, then its class
# ---------------------------------------------------------------------------------------------
from anygeometry.tolerance import TolerancePolicy  # noqa: E402


@pytest.mark.parametrize("start", [2 * math.pi * 159154 - .01, -.01])
def test_r1_1_a_1e9_aspect_ellipse_is_outside_the_domain(start):
    """r1 #1: u = (1e8, 0, 0), v = (0, .1, 0) (aspect 1e9), sweep .02: at angle .1 one interval
    turned .185 rad; now outside the declared domain at either start."""
    g, edge = _arc_edge((0., 0., 0.), start, .02, (1e8, 0., 0.), (0., .1, 0.))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)))


def _aspect_edge(aspect, start, sweep=.02):
    root = math.sqrt(aspect)
    return _arc_edge((0., 0., 0.), start, sweep, (root, 0., 0.), (0., 1 / root, 0.))


@pytest.mark.parametrize("u,v,inside", [(1e2, 1e-2, True),
                                         (1e2, float(np.nextafter(1e-2 / (1 + 8 * EPS), 0.)), False),
                                         (1e3, 1e-3, False)])
def test_r1_1_the_ellipse_aspect_at_and_beyond_the_bound(u, v, inside):
    """a / b = 1e2 / 1e-2 = 1e4 (stored angles within 1): inside, every interval turns at most
    the angle through both high-curvature ends; beyond the bound's 4 eps slack it is refused, and
    so is the former boundary 1e3 / 1e-3 (owner decision N1 round 6: the bound 1e6 became 1e4)."""
    g, edge = _arc_edge((0., 0., 0.), -.01, .02, (u, 0., 0.), (0., v, 0.))
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))  # noqa: E731
    if inside:
        samples = call()
        _dense_ok(g, edge, samples, .1, count=257)
        assert len(samples.parameters) - 1 >= math.ceil(math.pi / .1)
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("aspect,start,inside", [(10., 2 * math.pi * 159 - .01, True),
                                                 (10., 2 * math.pi * 160 - .01, False),
                                                 (1e3, 2 * math.pi * 159 - .01, False),     # inside before N1 r6
                                                 (1e6, 2 * math.pi * 159154 - .01, False)])
def test_r1_1_the_ellipse_aspect_counts_with_its_stored_angle(aspect, start, inside):
    """The evaluator rounds start + sweep t by eps |theta|, which turns the tangent by up to
    aspect x that: the bound is aspect x max(1, |stored angle|) <= 1e4 (1e6 before N1 round 6, so
    aspect 1e3 at 999 rad is now refused).  Measured: aspect 1e6 at
    999994 rad turns 10 % more than .001 per interval even with exact local divisions."""
    g, edge = _aspect_edge(aspect, start)
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.01))  # noqa: E731
    if inside:
        _dense_ok(g, edge, call(), .01, count=129)
    else:
        _refused_out_of_domain(call)


@pytest.mark.parametrize("start,inside", [(9.9e3 - .3, True), (1e4 - .3, False), (9.9e5 - .3, False)])
def test_r1_1_the_ellipse_inverse_at_the_stored_angle_bound(start, inside):
    """A non-circular ellipse of aspect 1 + 1e-6 near 1e4 rad goes through the ellipse inverse
    (|u| != |v|) in local angles: its psi steps are equal; at 1e4 x (1 + 1e-6) it is refused, and
    so is the former case near 1e6 rad (inside before the N1 round-6 bound)."""
    g, edge = _arc_edge((0., 0., 0.), start, .6, (1., 0., 0.), (0., 1. + 1e-6, 0.))
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.05))  # noqa: E731
    if inside:
        samples = call()
        psi = _tangent_angle_lift(g, edge, samples.parameters)
        assert np.abs(np.abs(np.diff(psi)) - abs(psi[-1] - psi[0]) / (len(psi) - 1)).max() <= 1e-9
        _dense_ok(g, edge, samples, .05, count=257)
    else:
        _refused_out_of_domain(call)


def test_r1_2_a_retained_line_range_is_refused():
    """r1 #2: a retained range 3e-8 long on a 2e9 carrier (rational bounds 1/2 +- 7.5e-18) on a
    tiny cylinder: the float carrier fields collapse it to a point.  Owner decision (r4): a
    LineRange drawn over part of its carrier is EDGE_UNSUPPORTED; the same segment as a whole
    carrier is divided from its ends."""
    g = corpus.GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), math.hypot(1e-7, 1.5e-8), 1.0, -.5, 1.0)
    pts = {(s, z): (1e-7, s * 1.5e-8, z) for s in (-1, 1) for z in (.2, .8)}
    ids = {k: g.add_point(*v) for k, v in pts.items()}
    half = Fraction(1.5e-8) / 2_000_000_000
    low = g.add_curve(ids[-1, .2], ids[1, .2], LineRange((1e-7, -1e9, .2), (1e-7, 1e9, .2),
                                                         (Fraction(1, 2) - half, Fraction(1, 2) + half)))
    edges = [low, g.add_line(ids[1, .2], ids[1, .8]), g.add_line(ids[1, .8], ids[-1, .8]),
             g.add_line(ids[-1, .8], ids[-1, .2])]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    _refused(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1)), "EDGE_UNSUPPORTED")
    g = corpus.GeometryModel()
    ids = {k: g.add_point(*v) for k, v in pts.items()}
    whole = g.add_curve(ids[-1, .2], ids[1, .2], LineRange(pts[-1, .2], pts[1, .2]))
    edges = [whole, g.add_line(ids[1, .2], ids[1, .8]), g.add_line(ids[1, .8], ids[-1, .8]),
             g.add_line(ids[-1, .8], ids[-1, .2])]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    samples = dt.edge_display_samples(g, whole, dt.DisplayTolerance(.1))
    assert len(samples.parameters) - 1 == 3
    _dense_ok(g, whole, samples, .1, supports=[support], count=257)


def _near_apex_sector(offset=5e-9, azimuth=.4):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-8))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 0., 1., 1., 0., 1.)
    tip = g.add_point(offset * math.cos(azimuth), offset * math.sin(azimuth), offset)
    a0, a1 = g.add_point(1., 0., 1.), g.add_point(math.cos(1.), math.sin(1.), 1.)
    side0 = g.add_line(tip, a0)
    arc = g.add_curve(a0, a1, EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., 1.))
    side1 = g.add_line(a1, tip)
    g.add_face((side0, arc, side1), surface=support)
    return g, support, side0, side1


def test_r1_3_a_line_from_near_an_apex_is_divided_by_its_azimuth():
    """r1 #3: a side from 7e-9 off the apex (within the model length 1e-8) at azimuth .4 to the
    unit circle at azimuth 0 turns .4 rad round the axis: it is divided like any line."""
    g, support, side0, side1 = _near_apex_sector()
    for edge in (side0, side1):
        samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))
        assert len(samples.parameters) - 1 >= 4
        _dense_ok(g, edge, samples, .1, supports=[support], count=1025)


def test_r1_3_a_line_crossing_the_axis_is_refused():
    """A line whose projection runs through the axis between its ends (collinear with the axis
    point, opposite sides) has no monotone azimuth: EDGE_UNSUPPORTED; the rulings are drawn whole."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1.0))
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.0, 0., math.pi)
    p = {(x, z): g.add_point(x, 0., z) for x in (.5, -.5) for z in (.2, .8)}
    across = g.add_line(p[.5, .2], p[-.5, .2])
    edges = [across, g.add_line(p[-.5, .2], p[-.5, .8]), g.add_line(p[-.5, .8], p[.5, .8]),
             g.add_line(p[.5, .8], p[.5, .2])]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    _refused(lambda: dt.edge_display_samples(g, across, dt.DisplayTolerance(.1)), "EDGE_UNSUPPORTED")
    for ruling in edges[1::2]:
        assert dt.edge_display_samples(g, ruling, dt.DisplayTolerance(.1)).parameters.tolist() == [0., 1.]


def _offset_arc(offset, u=(1., 0., 0.), v=(0., 1., 0.)):
    g = corpus.GeometryModel()
    arc = EllipticArc((0., 0., 0.), u, v, 0., .1)
    end = arc.evaluate(np.array([1.]))[0]
    return g, g.add_curve(g.add_point(u[0], u[1], offset), g.add_point(*end), arc)


def test_r1_4_a_vertex_offset_above_the_sag_is_refused():
    """r1 #4: the start vertex 8e-9 off the arc's start, sag 2e-9: the first chord cannot meet
    the allowance (it was returned 8e-9 from the curve)."""
    g, edge = _offset_arc(8e-9)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, 2e-9)), "EDGE_UNSUPPORTED")


@pytest.mark.parametrize("v", [(0., 1., 0.), (0., 2., 0.)], ids=("circle", "ellipse"))
def test_r1_4_the_end_chords_carry_the_vertex_offset_in_their_sag(v):
    """Offset 8e-9 below sag 1e-8: the end intervals are bisected until sag + offset fits; every
    drawn segment (from the vertex) is within the sag of the curve."""
    g, edge = _offset_arc(8e-9, v=v)
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, 1e-8))
    _dense_ok(g, edge, samples, .2, 1e-8, count=1025)


@pytest.mark.parametrize("sag,ok", [(1e-8, True), (2e-9, False)])
def test_r1_4_a_line_range_sag_is_its_vertex_offset(sag, ok):
    """A LineRange whose end vertex is 5e-9 off its exact end: the drawn chord is within sag 1e-8
    of the line; sag 2e-9 is refused.  Since N1 round 5 the carrier endpoints must be the edge's
    vertices within the model length, so the model length is 1e-8 here (at the default 1e-9 the
    edge is refused: test_r5_4)."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-8))
    edge = g.add_curve(g.add_point(0., 0., 0.), g.add_point(1., 0., 5e-9), LineRange((0., 0., 0.), (1., 0., 0.)))
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, sag))  # noqa: E731
    if ok:
        samples = call()
        assert samples.parameters.tolist() == [0., 1.]
        _dense_ok(g, edge, samples, .2, sag, count=257)
    else:
        _refused(call, "EDGE_UNSUPPORTED")


def _circle_band_on_frustum(start=-5e-9, sweep=1e-8):
    # N1 round 8: over 1e-8 rad from -5e-9 (the minimum sweep, MIN_DISPLAY_SWEEP; 8e-9 from -4e-9
    # before, now OUT_OF_DOMAIN)
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-13))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2.5e-9, 3e-9, .1, -1., 2.)
    arcs = [EllipticArc((-.5 + 2e-9, 0., z), (.5, 0., 0.), (0., .5, 0.), start, sweep) for z in (0., .1)]
    ends = [[g.add_point(*a.evaluate(np.array([t]))[0]) for t in (0., 1.)] for a in arcs]
    low = g.add_curve(ends[0][0], ends[0][1], arcs[0])
    high = g.add_curve(ends[1][0], ends[1][1], arcs[1])
    g.add_face((low, g.add_line(ends[0][1], ends[1][1]), high, g.add_line(ends[1][0], ends[0][0])),
               corners=(0, 1, 2, 3), surface=support)
    return g, support, low


def test_r1_5_the_closest_approach_does_not_cancel():
    """r1 #5: a .5 circle passing 2e-9 from a frustum's axis: its closest approach (lower bound
    = the real distance, the projection being circular) is in the domain (it cancelled to 0)."""
    g, support, low = _circle_band_on_frustum()
    samples = dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1e-6))
    _dense_ok(g, low, samples, .1, 1e-6, supports=[support], count=257)
    # the original geometry (8e-9 rad from -4e-9) is below the minimum sweep since N1 round 8
    g, _support, low = _circle_band_on_frustum(-4e-9, 8e-9)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1e-6)))


def test_r1_5_a_near_axis_arc_counts_its_stored_angle_rounding():
    """The same arc at a stored angle near 1e6 rad: the evaluator's rounding of start + sweep t
    (about 6e-11 rad) moves the point by 3e-11, a large azimuth error 2e-9 from the axis."""
    g, support, low = _circle_band_on_frustum(2 * math.pi * 159154 - 5e-9)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1e-6)))


def _r1_6_band(start):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-12, surface_residual=1e-3))
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .001, 1.0, 1.5, .5)
    arcs = [EllipticArc((-.001, .001, z), (.001, 0., 0.), (0., .001, 0.), start, .4) for z in (.2, .8)]
    ends = [[g.add_point(*a.evaluate(np.array([t]))[0]) for t in (0., 1.)] for a in arcs]
    low = g.add_curve(ends[0][0], ends[0][1], arcs[0])
    high = g.add_curve(ends[1][0], ends[1][1], arcs[1])
    g.add_face((low, g.add_line(ends[0][1], ends[1][1]), high, g.add_line(ends[1][0], ends[0][0])),
               corners=(0, 1, 2, 3), surface=support)
    return g, support, low


def test_r1_6_a_stationary_azimuth_at_an_end_is_out_of_domain():
    """r1 #6: a circle on a .001 cylinder tangent at its start to the ray from the axis (g = 0
    there, positive inside).  Drawn from round 1 to round 8; since N1 round 9 (owner decision) an
    end whose g is within its rounding bound is near-stationary, its branch not decidable:
    OUT_OF_DOMAIN."""
    g, _support, low = _r1_6_band(0.)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.05, 1e-5)))


def test_r1_6_the_same_arc_with_a_resolvable_g_is_drawn():
    """The otherwise identical arc from .01 (g = 1 - cos x + sin x, about .01 at its start, far
    beyond the bound) is drawn and passes the dense oracle."""
    g, support, low = _r1_6_band(.01)
    samples = dt.edge_display_samples(g, low, dt.DisplayTolerance(.05, 1e-5))
    _dense_ok(g, low, samples, .05, 1e-5, supports=[support], count=257)


def test_r1_7_counts_are_plain_ceilings():
    """r1 #7: sweep .4000000001 at .2 takes three steps (two overshot by 5e-11 rad); an exact .4
    keeps two (the step is verified, not slackened)."""
    g, edge = _arc_edge((0., 0., 0.), 0., .4000000001)
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))
    assert len(samples.parameters) - 1 == 3
    assert (np.diff(samples.parameters) * .4000000001).max() <= .2
    g, edge = _arc_edge((0., 0., 0.), 0., .4)
    assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)).parameters.tolist() == [0., .5, 1.]


def _ACCEPT(a, b):
    """A verification that accepts every interval: the union's merge proposal alone."""
    return True


def test_r1_7_the_union_drops_a_near_duplicate_only_when_its_angle_agrees():
    """A near-duplicate cut is dropped only when its own division's angle there agrees with the
    kept cut's within that division's threshold (its headroom, never below 64 eps max(1, |angle|));
    otherwise it is kept.  Since the round-6 follow-up the threshold is the headroom; since round 7
    this is only the proposal, which the union's verification accepts or undoes (``_ACCEPT``
    here)."""
    first = dt._Division([.5], lambda t: .2 * t)
    agree = dt._Division([.5 + 1e-14], lambda t: .2 * t)               # .2 x 1e-14: rounding
    differ = dt._Division([.5 + 1e-14], lambda t: 1e6 * t)              # 1e6 x 1e-14 = 1e-8 rad
    assert dt._union(0, [first, agree], [0., 0.], _ACCEPT).tolist() == [0., .5, 1.]
    assert dt._union(0, [first, differ], [0., 0.], _ACCEPT).tolist() == [0., .5, .5 + 1e-14, 1.]
    assert dt._union(0, [first, differ], [0., 1e-7], _ACCEPT).tolist() == [0., .5, 1.]          # within its headroom
    assert dt._union(0, [first, differ], [1e-7, 0.], _ACCEPT).tolist() == [0., .5 + 1e-14, 1.]  # the stricter is kept
    many = dt._Division(np.linspace(0., 1., 4097)[1:-1].tolist(), lambda t: t)
    _refused(lambda: dt._union(0, [dt._Division([.5 + 1e-3], lambda t: t), many], [0., 0.], _ACCEPT), "EDGE_UNSUPPORTED")


def test_r1_9_the_union_is_one_sort_at_the_cap():
    """4095 cuts from each of two divisions, all duplicates: one sort and one pass."""
    import time
    first = np.linspace(0., 1., 4097)[1:-1].tolist()
    other = dt._Division(first, lambda t: t)
    began = time.perf_counter()
    assert len(dt._union(0, [dt._Division(first, lambda t: t), other], [0., 0.], _ACCEPT)) == 4097
    assert time.perf_counter() - began < 1.0


def _cone_line_at_radius(radius):
    """A non-generator line from (radius, 0, -1024) on a frustum about x = y = 0 (origin at
    z = -1024): the radius bound there is exactly 1e-9 x 1024."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1e-6))
    support = Cone((0., 0., -1024.), (0., 0., 1.), (1., 0., 0.), radius, 2e-6, 1.0, 0., 1.)
    p = g.add_point(radius, 0., -1024.)
    q0, q1 = g.add_point(2e-6, 0., -1023.), g.add_point(2e-6 * math.cos(.5), 2e-6 * math.sin(.5), -1023.)
    line = g.add_line(p, q1)
    top = g.add_curve(q1, q0, EllipticArc((0., 0., -1023.), (2e-6, 0., 0.), (0., 2e-6, 0.), .5, -.5))
    g.add_face((line, top, g.add_line(q0, p)), surface=support)
    return g, support, line


@pytest.mark.parametrize("inside", [True, False])
def test_r1_8_cone_radius_at_and_one_ulp_below_the_ratio(inside):
    """r1 #8: equality with the radius bound is inside, one ulp below is refused."""
    bound = dt.MIN_DISPLAY_RADIUS_RATIO * 1024.
    g, support, line = _cone_line_at_radius(bound if inside else float(np.nextafter(bound, 0.)))
    call = lambda: dt.edge_display_samples(g, line, dt.DisplayTolerance(.1))  # noqa: E731
    if inside:
        _dense_ok(g, line, call(), .1, supports=[support], count=1025)
    else:
        _refused_out_of_domain(call)


def test_r1_8_an_intersection_curve_is_refused():
    from anygeometry.exact_curves import CylinderIntersectionCurve
    first = Cylinder((0., 0., -2.), (0., 0., 1.), (1., 0., 0.), 1., 4.)
    second = Cylinder((-2., 0., 0.), (1., 0., 0.), (0., 1., 0.), .5, 4.)
    curve = CylinderIntersectionCurve(first, second, -.3, .6, 1)
    g = corpus.GeometryModel()
    ends = curve.evaluate(np.array([0., 1.]))
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), curve)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_r1_8_an_edge_of_a_nurbs_face_is_refused():
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    sheet = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                         (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    g = corpus.GeometryModel()
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    points = [g.add_point(*sheet.evaluate(*q)) for q in uv]
    edges = [g.add_curve(points[i], points[(i + 1) % 4],
                         CurveOnSurface(sheet, NurbsCurve(1, (0., 0., 1., 1.), (uv[i], uv[(i + 1) % 4]))))
             for i in range(4)]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=sheet)
    _refused(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.2)), "SUPPORT_UNSUPPORTED")


def test_r1_8_an_edge_of_a_ruled_face_is_refused():
    from anygeometry.surfaces import RuledSurface
    g = corpus.GeometryModel()
    points = [g.add_point(*xyz) for xyz in ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))]
    edges = [g.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    g.add_face(edges, corners=(0, 1, 2, 3),
               surface=RuledSurface(((0., 0., 0.), (1., 0., 0.)), ((0., 1., 0.), (1., 1., 0.))))
    _refused(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.2)), "SUPPORT_UNSUPPORTED")


def test_an_unknown_edge_is_a_geometry_error():
    g, edge = _arc_edge((0., 0., 0.), 0., .4)
    with pytest.raises(GeometryError) as caught:
        dt.edge_display_samples(g, edge + 999, dt.DisplayTolerance(.2))
    assert not isinstance(caught.value, dt.DisplayTrimRefused)


# ---------------------------------------------------------------------------------------------
# Codex N1 round 2: each finding's geometry, then its class ("a float rounding or cancellation,
# at an admitted scale, breaks a closed-form bound or raises")
# ---------------------------------------------------------------------------------------------
def _band(g, support, low_arc, high_arc):
    ends = [[g.add_point(*a.evaluate(np.array([t]))[0]) for t in (0., 1.)] for a in (low_arc, high_arc)]
    low = g.add_curve(ends[0][0], ends[0][1], low_arc)
    high = g.add_curve(ends[1][0], ends[1][1], high_arc)
    sides = (g.add_line(ends[0][1], ends[1][1]), g.add_line(ends[1][0], ends[0][0]))
    g.add_face((low, sides[0], high, sides[1]), corners=(0, 1, 2, 3), surface=support)
    return low, high, sides


def _thin_projection(v):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1.))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 1.1, 150., .5, 1.)
    low, _high, _sides = _band(g, support, *(EllipticArc((0., 1., z), (1., 0., 0.), v, .1, 1.) for z in (10., 20.)))
    return g, support, low


def test_r2_1_a_thin_projection_is_outside_the_domain():
    """r2 #1: an ellipse whose projection onto the cone's cross-section has aspect 1e8 (the
    curve's own aspect is 100): its first interval turned .266 rad round the axis at angle .1.
    The aspect x stored-angle bound applies to the projected conic too."""
    g, support, low = _thin_projection((0., 1e-8, 100.))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1.)))


def test_r2_1_a_thin_projection_inside_the_domain_is_exact():
    """The same band with a projected aspect 1e3 (inside the bound): every interval meets the
    azimuth bound to the documented rounding allowance.  The former case, projected aspect 1e5,
    is outside the N1 round-6 bound 1e4."""
    g, support, low = _thin_projection((0., 1e-5, 100.))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1.)))
    g, support, low = _thin_projection((0., 1e-3, 100.))
    samples = dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1.))
    _dense_ok(g, low, samples, .1, supports=[support], count=1025)


@pytest.mark.parametrize("thin", [1e-3, 1e-5, 1e-6])
def test_r2_1_the_ray_inverse_does_not_cancel(thin):
    """The azimuth inverse of a thin projected conic (|z| up to 1e6): angle(solve(phi)) = phi to
    aspect x eps over the whole range (the former (z.e)^2 - |z|^2 + 1 cancelled terms of |z|^2)."""
    amap = dt._AngleMap(0, ((1., 0.), (0., thin)), (0., 1.), 0., .1, 1.1)
    first, last = amap.angle(.1), amap.angle(1.1)
    for phi in np.linspace(first, last, 401)[1:-1]:
        assert abs(amap.angle(amap.solve(phi)) - phi) <= 64 * EPS / thin, (thin, phi)


def _near_axis_cylinder_band(start):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-13))
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2e-9, 1., -2., 4.)
    low, _high, _sides = _band(g, support, *(EllipticArc((-.5 + 2e-9, 0., z), (.5, 0., 0.), (0., .5, 0.), start,
                                                         1.6e-8) for z in (.2, .8)))
    return g, support, low


def test_r2_2_a_cylinder_conic_counts_its_stored_angle_rounding():
    """r2 #2: a .5 circle passing 2e-9 from a 2e-9 cylinder's axis at a stored angle near 1e6
    rad turned .115 rad on one interval at angle .1: one closest-approach rule (scale including
    max(1, |stored angle|) sigma_max) now applies to cylinders and cones alike."""
    g, support, low = _near_axis_cylinder_band(math.tau * 159154 - 8e-9)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1e-6)))


def test_r2_2_the_same_arc_near_angle_zero_is_inside():
    g, support, low = _near_axis_cylinder_band(-8e-9)
    samples = dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1e-6))
    _dense_ok(g, low, samples, .1, 1e-6, supports=[support], count=257)


def test_r2_3_a_ruling_on_a_tilted_axis_far_out_is_drawn_whole():
    """r2 #3: a ruling along a (1, 1, 0) axis at coordinates 9e7 raised ZeroDivisionError (its
    ends projected separately gave a nonzero cross product, its exact difference projects to 0).
    Every edge of the band is now drawn or refused typed."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=2.))
    support = Cylinder((0., 0., 0.), (1., 1., 0.), (0., 0., 1.), math.sqrt(13.5), 4e8, -1.6, 3.2)
    p = [g.add_point(*xyz) for xyz in ((9e7, 9e7 + 3, 3.), (9e7 + 1, 9e7 + 4, 3.), (9e7 + 1, 9e7 + 4, 3.75),
                                       (9e7, 9e7 + 3, 3.75))]
    edges = [g.add_line(p[i], p[(i + 1) % 4]) for i in range(4)]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    for edge in edges:
        try:
            samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1, 1e-5))
        except dt.DisplayTrimRefused:
            continue
        _dense_ok(g, edge, samples, .1, 1e-5, supports=[support], count=257)
    assert dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.1, 1e-5)).parameters.tolist() == [0., 1.]


def test_r2_3_a_zero_projected_difference_is_a_ruling():
    """The class: a line whose exact difference projects to zero (parallel to the axis) has
    constant azimuth, whatever its ends' separately rounded projections say."""
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 1.)
    a, b = np.array([1., 0., .2]), np.array([1., 0., .8])
    plan = dt._line_azimuth_division(0, support, a, b, .1, (a, b))
    assert plan.count == 1 and plan.build().values == []


def test_r2_4_the_exact_apex_of_a_tilted_cone_is_on_the_axis():
    """r2 #4: the apex from the cone's own evaluator on a (1, 1, 1) axis projects 3e-16 off the
    axis (the frame is orthogonal to rounding): both generators are drawn whole."""
    g = corpus.GeometryModel()
    support = Cone((0., 0., 0.), (1., 1., 1.), (1., 0., 0.), 1., 0., 1., 0., 1.)
    arc = EllipticArc(tuple(support.origin), tuple(support.radial_direction),
                      tuple(np.cross(support.axis, support.radial_direction)), 0., .6)
    ends = arc.evaluate(np.array([0., 1.]))
    a0, a1, tip = g.add_point(*ends[0]), g.add_point(*ends[1]), g.add_point(*support.evaluate(0., 1.))
    base = g.add_curve(a0, a1, arc)
    sides = (g.add_line(a1, tip), g.add_line(tip, a0))
    g.add_face((base, *sides), surface=support)
    for edge in sides:
        samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))
        assert samples.parameters.tolist() == [0., 1.]
    _dense_ok(g, base, dt.edge_display_samples(g, base, dt.DisplayTolerance(.1)), .1, supports=[support])


def test_r2_4_a_line_on_the_axis_is_outside_the_domain():
    """Reviewer's out-of-scope note: a line with both ends on the axis has no azimuth anywhere
    (every point is below the radius floor): refused, not drawn as a ruling."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-13))
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2e-9, 1., 0., .5)
    r = 2e-9
    p = [g.add_point(*xyz) for xyz in ((0., 0., .2), (r * math.cos(.5), r * math.sin(.5), .2),
                                       (r * math.cos(.5), r * math.sin(.5), .8), (0., 0., .8))]
    edges = [g.add_line(p[i], p[(i + 1) % 4]) for i in range(4)]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edges[3], dt.DisplayTolerance(.1)))


@pytest.mark.parametrize("v", [(0., 1., 0.), (0., 1.0001, 0.)], ids=("circle", "ellipse"))
def test_r2_5_the_two_interval_rule_measures_the_drawn_chord(v):
    """r2 #5: an arc of analytic sag about 8e-9 whose start vertex is 8e-9 inside it: the drawn
    chord leaves the curve by about 1.25e-8 > surface_residual 1e-8, so it gets two intervals."""
    g = corpus.GeometryModel()
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), v, 0., math.sqrt(8 * 8e-9))
    end = arc.evaluate(np.array([1.]))[0]
    edge = g.add_curve(g.add_point(1 - 8e-9, 0., 0.), g.add_point(*end), arc)
    assert len(dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)).parameters) - 1 >= 2


def test_r2_6_a_nearly_circular_frame_takes_the_ellipse_path():
    """r2 #6: u = (1e7, 0, 0), v = (9e-6, 1e7, 0) (u.v = 9e-13 |u||v|) was classed a circle: its
    first interval exceeded the angle by 1.79e-13 rad and the sag by 2.32e-8; the dense oracle
    (now at the documented rounding allowance) checks both."""
    g, edge = _arc_edge((0., 0., 0.), math.pi / 4 - .1, .4, (1e7, 0., 0.), (9e-6, 1e7, 0.))
    sag = 2e7 * math.sin(.05) ** 2
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2, sag))
    _dense_ok(g, edge, samples, .2, sag, count=257)


def test_r2_6_exact_circles_keep_the_circle_path():
    g, edge = _arc_edge((0., 0., 0.), 0., .4, (1e7, 0., 0.), (0., 1e7, 0.))
    assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)).parameters.tolist() == [0., .5, 1.]


def test_r2_7_a_face_without_a_support_is_refused():
    g = corpus.GeometryModel()
    points = [g.add_point(*xyz) for xyz in ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))]
    edges = [g.add_line(points[i], points[(i + 1) % 3]) for i in range(3)]
    face = g.add_face(edges)
    assert g.faces[face].surface is None
    _refused(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.2)), "SUPPORT_UNSUPPORTED")


def test_r2_7_every_count_is_checked_before_any_division_is_built(monkeypatch):
    """An ellipse whose tangent needs one interval but whose azimuth about a frustum's axis (it
    passes 2e-9 from it) needs more than 4096: refused before any division is materialised."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-13))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2.5e-9, 3e-9, .1, -1., 2.)
    # over the minimum sweep 1e-8 since N1 round 8 (8e-9 before)
    low, _high, _sides = _band(g, support, *(EllipticArc((-.5 + 2e-9, 0., z), (.5, 0., 0.), (0., .5000001, 0.),
                                                         -5e-9, 1e-8) for z in (0., .1)))
    built = []
    original = dt._angle_division
    monkeypatch.setattr(dt, "_angle_division", lambda *args: built.append(args) or original(*args))
    _refused(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(1e-4, 1e-6)), "EDGE_UNSUPPORTED")
    assert built == []



# ---------------------------------------------------------------------------------------------
# Codex N1 round 3: each finding's geometry, then its class
# ---------------------------------------------------------------------------------------------
def _quad(g, points, support, curves=None):
    ids = [g.add_point(*p) for p in points]
    curves = curves or {}
    edges = [g.add_curve(ids[i], ids[(i + 1) % 4], curves[i]) if i in curves else g.add_line(ids[i], ids[(i + 1) % 4])
             for i in range(4)]
    g.add_face(edges, corners=(0, 1, 2, 3), surface=support)
    return edges


@pytest.mark.parametrize("reverse", [False, True], ids=("a_to_b", "b_to_a"))
def test_r3_1_a_reconstructed_end_projection_does_not_hide_the_azimuth(reverse):
    """r3 HIGH: on a cone with its apex at the origin, a LineRange from A (2e7 out, azimuth pi/4)
    to B (1.5e-9 from the axis, azimuth pi/2): B' = A' + P(b - a) cancelled B's radius to 0, so
    the line was drawn as one chord (reversed: eight intervals).  Each end's own projection now
    decides its radius, and an evaluated point of a line a + t (b - a) is rounded at the scale of
    its ends: B is below the radius bound of that scale, refused typed in both directions."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-13, surface_residual=1e7))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 0., 3e7, 3e7, .5, 1.1)
    a, b = (2e7, 2e7, math.hypot(2e7, 2e7)), (0., 1.5e-9, 1.5e-9)
    points = [a, b, (0., 1e7, 1e7), (1e7, 2e7, math.hypot(1e7, 2e7))]
    if reverse:
        points = [b, a, (1e7, 2e7, math.hypot(1e7, 2e7)), (0., 1e7, 1e7)]
    edges = _quad(g, points, support, {0: LineRange(points[0], points[1])})
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.1, 1.)))


def test_r3_1_a_near_axis_end_hiding_an_axis_crossing_is_refused():
    """r3 out of scope: on a 2e-9 cylinder, a line from 1.5e-15 behind the axis (on it to the
    frame's rounding) crosses the axis at t = 7.5e-7: it was drawn as one chord by the on-axis
    shortcut.  A cylinder never reaches its axis, so an end on it is outside the domain."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-13, surface_residual=1e-8))
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2e-9, 1., -3.14, 6.28)
    edges = _quad(g, [(-1.5e-15, 0., .2), (2e-9, 0., .8), (2e-9, 1e-9, .8), (1.5e-9, 1e-9, .2)], support)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edges[0], dt.DisplayTolerance(.1, 1.)))


def test_r3_1_an_end_on_the_axis_of_a_frustum_is_refused():
    """The class: only a support that reaches its axis (a cone's apex) has a point on it; a line
    ending on the axis of a frustum (narrow radius 1e-6, within the face tolerance) is refused,
    while the same line on the cone with an apex there is a generator drawn whole."""
    for small, inside in ((1e-6, False), (0., True)):
        g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1e-5))
        support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), small, 1., 1., 0., 1.)
        tip, p0, p1 = g.add_point(0., 0., 0.), g.add_point(1., 0., 1.), g.add_point(math.cos(.5), math.sin(.5), 1.)
        side = g.add_line(tip, p0)
        arc = g.add_curve(p0, p1, EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., .5))
        g.add_face((side, arc, g.add_line(p1, tip)), surface=support)
        call = lambda: dt.edge_display_samples(g, side, dt.DisplayTolerance(.1))  # noqa: E731
        if inside:
            assert call().parameters.tolist() == [0., 1.]
        else:
            _refused_out_of_domain(call)


def _thin_frustum_band(v_y):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1.))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 1.1, 2., .3, 1.5)
    low, _high, _sides = _band(g, support, *(EllipticArc((0., 1., z), (1., 0., 0.), (0., v_y, .02), .2, .78)
                                             for z in (.2, .8)))
    return g, support, low


def test_r3_2_a_thin_projection_inside_the_domain_meets_the_allowance():
    """r3 MEDIUM 2: projected aspect about 990099 (inside the bound then): the last interval turned
    0.02967921180924682 rad round the axis at angle 0.029679211598, above the former documented
    allowance; the allowance now covers the projected inverse's aspect amplification.  That case
    is outside the N1 round-6 bound 1e4; the same band at projected aspect about 9901 is drawn."""
    g, support, low = _thin_frustum_band(1.01e-6)
    tol = dt.DisplayTolerance(.029679211598, 1.)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, tol))
    g, support, low = _thin_frustum_band(1.01e-4)
    samples = dt.edge_display_samples(g, low, tol)
    _dense_ok(g, low, samples, tol.angle, 1., supports=[support], count=1025)


def test_r3_3_the_circle_sag_counts_the_evaluators_stored_angle():
    """r3 MEDIUM 3: a unit arc at 999994 rad, sweep .4, at sag 2 sin^2(.05): its last chord sagged
    0.00499583472546078 against 0.004995834721974234 (the evaluator rounds the interval's span up
    to 0.2000000000698492): within the allowance, which counts that rounding."""
    g, edge = _arc_edge((0., 0., 0.), 999994., .4)
    sag = 2 * math.sin(.05) ** 2
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(1., sag))
    assert samples.parameters.tolist() == [0., .5, 1.]
    _dense_ok(g, edge, samples, 1., sag, count=1025)


def test_r3_3_the_ellipse_sag_counts_the_evaluators_stored_angle():
    """The ellipse of aspect 1.000001 at 999994 rad is outside the N1 round-6 bound (aspect x
    stored angle 1e4); the same ellipse at 9994 rad (inside) meets the sag with its allowance."""
    _refused_out_of_domain(lambda: dt.edge_display_samples(
        *_lone_curve_edge(EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1.000001, 0.), 999994., .4)),
        dt.DisplayTolerance(1., .004995834972010272)))
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1.))
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1.000001, 0.), 9994., .4)
    ends = arc.evaluate(np.array([0., 1.]))
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), arc)
    conic = dt._Conic(edge, arc.center, arc.u_vector, arc.v_vector, arc.start_angle, arc.sweep_angle)
    sag = max(conic.sag(0., .5), conic.sag(.5, 1.)) * (1 + 1e-13)      # just above each half's exact sag
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(1., sag))
    assert samples.parameters.tolist() == [0., .5, 1.]
    _dense_ok(g, edge, samples, 1., sag, count=1025)


def test_r3_the_rounding_allowance_is_public_and_at_most_two_percent():
    """One documented function gives each edge's absolute angle and sag allowances: on every
    drawable edge of these corpus cases at both steps (with and without sag 1e-4) it is at most
    2 % of the angle and of the sag allowance."""
    assert "display_rounding_allowance" in dt.__all__ and "DisplayRoundingAllowance" in dt.__all__
    for name in ("tripod_skins", "cone_section", "steep_cuts", "tiny_offset_cylinder", "far_apex_cone",
                 "ellipse_edge_20", "winding_fixture"):
        case = corpus.ALL[name]()
        g = case.model
        for edge in _case_edges(case):
            for step in corpus.STEPS:
                for sag in (None, 1e-4):
                    tol = dt.DisplayTolerance(step, sag)
                    try:
                        dt.edge_display_samples(g, edge, tol)
                    except dt.DisplayTrimRefused:
                        continue
                    allowance = dt.display_rounding_allowance(g, edge, tol)
                    assert 0. < allowance.angle <= .02 * step, (name, edge, allowance)
                    assert allowance.sag is None or 0. < allowance.sag <= .02 * (sag or 1e300)


def test_r3_a_tolerance_finer_than_the_allowance_is_refused():
    """The domain is tightened so the allowance never exceeds 2 %: a circle passing 2e-9 from a
    frustum's axis is drawn at angles .1 and 1e-3 but refused at 5e-4 (its azimuth rounding there
    is about 1.4e-5 rad); a unit circle near x = 1e9 is drawn at sag 3e-4, refused at sag 3e-5
    (its points round by about 2e-7)."""
    g, support, low = _circle_band_on_frustum()
    assert dt.display_rounding_allowance(g, low, dt.DisplayTolerance(.1, 1e-6)).angle > .02 * 5e-4
    for angle in (.1, 1e-3):
        _dense_ok(g, low, dt.edge_display_samples(g, low, dt.DisplayTolerance(angle, 1e-6)), angle, 1e-6,
                  supports=[support], count=129)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(5e-4, 1e-6)))
    g, edge = _arc_edge((1e9 - 2., 0., 0.), 0., .03)
    _dense_ok(g, edge, dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, 3e-4)), .0153, 3e-4, count=257)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, 3e-5)), "EDGE_UNSUPPORTED")


def _projected_aspect(support, arc):
    frame = np.column_stack((arc.u_vector, arc.v_vector))
    projected = np.vstack((support.radial_direction @ frame, support.circumferential_direction @ frame))
    s = np.linalg.svd(projected, compute_uv=False)
    return s[0] / s[1]


def _thin_band_call(v_y):
    g, _support, low = _thin_frustum_band(v_y)
    return dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1.))


def test_r3_4_the_projected_aspect_boundary_and_its_next_crossing():
    """r3 LOW: the public bound on the projected aspect (aspect x max(1, |stored angle|) <= 1e4
    since N1 round 6, stored angles within 1 here), found through the public call as the last
    admitted float: the independent SVD aspect there is 1e4 (to the bound's 4 eps slack), the
    dense oracle holds on it, and the next float down (a thinner projection) is refused."""
    inside, outside = 1.1e-4, .9e-4
    _thin_band_call(inside)
    _refused_out_of_domain(lambda: _thin_band_call(outside))
    while float(np.nextafter(outside, 1.)) < inside:
        middle = outside + (inside - outside) / 2
        if not outside < middle < inside:
            middle = float(np.nextafter(outside, 1.))
        try:
            _thin_band_call(middle)
            inside = middle
        except dt.DisplayDomainRefused:
            outside = middle
    assert outside == float(np.nextafter(inside, 0.))
    g, support, low = _thin_frustum_band(inside)
    assert abs(_projected_aspect(support, g.edges[low].curve) / dt.MAX_DISPLAY_ELLIPSE_ASPECT - 1) <= 1e-9
    _dense_ok(g, low, dt.edge_display_samples(g, low, dt.DisplayTolerance(.1, 1.)), .1, 1., supports=[support],
              count=1025)
    _refused_out_of_domain(lambda: _thin_band_call(outside))


def _lone_curve_edge(curve):
    g = corpus.GeometryModel()
    ends = np.asarray(curve.evaluate(np.array([0., 1.])), dtype=float)
    return g, g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), curve)


def test_r3_4_a_quadric_intersection_curve_is_refused():
    from anygeometry.quadric_curves import QuadricIntersectionCurve
    from anygeometry.quadric_supports import cone_support
    a = math.radians(10.)
    cone = Cone((0., 0., 0.), (math.cos(a), 0., math.sin(a)), (0., 1., 0.), .5, 1., 5., 0., math.tau)
    other = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau)
    curves = [c for c in cone_support(cone, other, tolerance=1e-9).curves if isinstance(c, QuadricIntersectionCurve)]
    g, edge = _lone_curve_edge(curves[0])
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_r3_4_a_bezier_quadric_curve_is_refused():
    from anygeometry.branch_curves import BezierQuadricCurve
    from anygeometry.branch_supports import bezier_quadric_support
    from anygeometry.extrusions import BezierDirectrix
    from anygeometry.surfaces import ExtrudedSurface
    wall = ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))), (.25, 0., 1.5))
    curves = list(bezier_quadric_support(wall, Cylinder((-2., .4, .6), (1., 0., 0.), (0., 1., 0.), .7, 8.)).curves)
    assert isinstance(curves[0], BezierQuadricCurve)
    g, edge = _lone_curve_edge(curves[0])
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_r3_4_a_plane_nurbs_curve_is_refused():
    from anygeometry.nurbs import cylinder_to_nurbs
    from anygeometry.support_curves import PlaneNurbsCurve
    sheet = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., math.pi))
    g, edge = _lone_curve_edge(PlaneNurbsCurve.from_plane(Plane((0., 0., 1.5), (1., 0., .3), (0., 1., .2)), sheet, 0))
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_r3_4_a_nurbs_surface_intersection_curve_is_refused():
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
    from anygeometry.nurbs_surface_intersection import (IntersectionStatus, NurbsIntersectionCurve,
                                                        query_nurbs_surface_intersection)
    knots = (0., 0., 1., 1.)
    first = NurbsSurface(1, 1, knots, knots, tuple(tuple((x, y, 0.) for y in (-1., 1.)) for x in (0., 1.)))
    second = NurbsSurface(1, 1, knots, knots, tuple(tuple((x, 0., z) for z in (-1., 1.)) for x in (-1., 2.)))
    query = query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy())
    assert query.status is IntersectionStatus.CERTIFIED
    g, edge = _lone_curve_edge(NurbsSurfaceIntersectionCurve(NurbsIntersectionCurve.from_result(query)))
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_r3_5_the_combined_cap_is_enforced_before_any_division_is_built(monkeypatch):
    """Carry-forward: a circle coaxial with its cylinder needs 2100 tangent and 2100 azimuth
    intervals (each admitted alone, 4199 summed): refused from the summed counts before any
    division is built (it was built and merged first)."""
    g = corpus.GeometryModel()
    sweep = 2 * math.pi - 1e-3
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 1., 0., sweep)
    low, _high, _sides = _band(g, support, *(EllipticArc((0., 0., z), (1., 0., 0.), (0., 1., 0.), 0., sweep)
                                             for z in (.2, .8)))
    built = []
    original = dt._angle_division
    monkeypatch.setattr(dt, "_angle_division", lambda *args: built.append(args) or original(*args))
    step = sweep / 2100 * (1 + 1e-9)
    _refused(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(step)), "EDGE_UNSUPPORTED")
    assert built == []


# ---------------------------------------------------------------------------------------------
# Codex N1 round 4: each finding's geometry, then its class
# ---------------------------------------------------------------------------------------------
_T4 = 9e8


def _partial_line_range_face():
    """r4 HIGH geometry: a 1-unit LineRange retained over [0, 2^-26] of a 2^26-long carrier, on a
    cylinder at x = 9e8 (radius 2.1, surface_residual .3)."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=.3))
    support = Cylinder((_T4, 0., 0.), (0., 0., 1.), (1., 0., 0.), 2.1, 1., -.1, 1.)
    points = [(_T4 + 2., 0., .2), (_T4 + 2. - 2. ** -23, 1., .2), (_T4 + 2. - 2. ** -23, 1., .8), (_T4 + 2., 0., .8)]
    carrier = LineRange(points[0], (_T4 - 7., 2. ** 26, .2), (Fraction(0), Fraction(1, 2 ** 26)))
    edges = _quad(g, points, support, {0: carrier})
    return g, support, edges


def test_r4_1_a_line_range_over_part_of_its_carrier_is_refused():
    """r4 HIGH: the exact carrier turned 0.4636476358228977 rad against the limit
    0.4636476328426804 and its exact end lay 1.49e-8 off the drawn segment (sag limit 1e-10),
    hidden from the oracle by the rounded ends.  Owner decision (r4, recommended option): a
    LineRange is supported only when its carrier IS the edge's segment (exact domain [0, 1])."""
    g, _support, edges = _partial_line_range_face()
    tol = dt.DisplayTolerance(math.atan2(1., 2. - 2. ** -23), 1e-10)
    _refused(lambda: dt.edge_display_samples(g, edges[0], tol), "EDGE_UNSUPPORTED")
    _refused(lambda: dt.display_rounding_allowance(g, edges[0], tol), "EDGE_UNSUPPORTED")


@pytest.mark.parametrize("domain", [(Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1)),
                                    (Fraction(1), Fraction(0)), (Fraction(1, 3), Fraction(2, 3))],
                         ids=("first_half", "second_half", "reversed", "middle"))
def test_r4_1_every_line_range_other_than_its_whole_carrier_is_refused(domain):
    g = corpus.GeometryModel()
    carrier = LineRange((0., 0., 0.), (3., 0., 0.), domain)
    ends = carrier.evaluate(np.array([0., 1.]))
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), carrier)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")
    whole = g.add_curve(g.add_point(0., 1., 0.), g.add_point(3., 1., 0.), LineRange((0., 1., 0.), (3., 1., 0.)))
    assert dt.edge_display_samples(g, whole, dt.DisplayTolerance(.2)).parameters.tolist() == [0., 1.]


@pytest.mark.parametrize("kind", ["straight", "line_range"])
def test_r4_1_the_line_allowance_counts_its_ends_coordinate_rounding(kind):
    """The line terms of the allowance include eps x max |coordinate| of the evaluated ends (the
    rounding of a point at 9e8: about 2e-7), so the published bound holds for every line."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=.3))
    support = Cylinder((_T4, 0., 0.), (0., 0., 1.), (1., 0., 0.), 2.1, 1., -.1, 1.)
    points = [(_T4 + 2., 0., .2), (_T4 + 2. - 2. ** -23, 1., .2), (_T4 + 2. - 2. ** -23, 1., .8), (_T4 + 2., 0., .8)]
    curves = {0: LineRange(points[0], points[1])} if kind == "line_range" else None
    edges = _quad(g, points, support, curves)
    tol = dt.DisplayTolerance(.5, 1e-4)
    allowance = dt.display_rounding_allowance(g, edges[0], tol)
    assert allowance.sag >= EPS * (_T4 + 2.)
    assert allowance.angle >= EPS * (_T4 + 2.) / 2.1
    _dense_ok(g, edges[0], dt.edge_display_samples(g, edges[0], tol), .5, 1e-4, supports=[support], count=257)


def _offset_lone_arc(v):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(curve_fit_residual=1e-6, surface_residual=1e-8))
    arc = EllipticArc((_T4, 0., 0.), (1., 0., 0.), v, 0., 1e-4)
    ends = np.asarray(arc.evaluate(np.array([0., 1.])), dtype=float)
    moved = [(float(np.nextafter(p[0], math.inf)), p[1], p[2]) for p in ends]
    return g, g.add_curve(g.add_point(*moved[0]), g.add_point(*moved[1]), arc)


@pytest.mark.parametrize("v", [(0., 1., 0.), (0., 2., 0.)], ids=("circle", "ellipse"))
def test_r4_2_the_minimum_two_rule_measures_the_drawn_chord(v):
    """r4 MEDIUM: both vertices one float (1.19e-7) off a lone arc at x = 9e8, surface_residual
    1e-8: the drawn chord leaves the curve by 1.19e-7, so the edge gets at least two intervals
    (the offsets were zeroed as rounding before the decision)."""
    g, edge = _offset_lone_arc(v)
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))
    assert len(samples.parameters) - 1 >= 2


def test_r4_2_the_minimum_two_rule_counts_the_stored_angle_rounding():
    """r4 MEDIUM: a lone unit circle at 999994 rad, sweep .4, surface_residual = 2 sin^2(.1) (its
    analytic sag): the evaluated sag exceeds it by 2.31e-12, so the edge is not one chord."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=2 * math.sin(.1) ** 2))
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 999994., .4)
    ends = np.asarray(arc.evaluate(np.array([0., 1.])), dtype=float)
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), arc)
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(1.))
    assert len(samples.parameters) - 1 >= 2


def test_r4_2_an_ordinary_arc_far_below_the_residual_stays_one_chord():
    """Ordinary inputs stay as they were: an arc whose drawn chord is far inside the residual."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1e-2))
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., .1)
    ends = np.asarray(arc.evaluate(np.array([0., 1.])), dtype=float)
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), arc)
    assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(1.)).parameters.tolist() == [0., 1.]


def _apex_sector(inner, tip_z):
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=1e-5))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), inner, 1., 1., 0., 1.)
    tip, p0, p1 = g.add_point(0., 0., tip_z), g.add_point(1., 0., 1.), g.add_point(math.cos(.5), math.sin(.5), 1.)
    side = g.add_line(tip, p0)
    arc = g.add_curve(p0, p1, EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., .5))
    g.add_face((side, arc, g.add_line(p1, tip)), surface=support)
    return g, side


@pytest.mark.parametrize("inner,tip_z", [(1e-16, 0.), (0., 1e-6)], ids=("tiny_frustum", "above_the_apex"))
def test_r4_3_an_axis_end_away_from_an_exact_apex_is_refused(inner, tip_z):
    """r4 out of scope: a frustum with inner radius 1e-16 has no apex; on the exact cone, a tip
    1e-6 above the apex (beyond the model length 1e-9) is not at it: both OUT_OF_DOMAIN."""
    g, side = _apex_sector(inner, tip_z)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, side, dt.DisplayTolerance(.1, 1.)))


def test_r4_3_an_end_at_the_exact_apex_is_a_generator():
    g, side = _apex_sector(0., 0.)
    assert dt.edge_display_samples(g, side, dt.DisplayTolerance(.1, 1.)).parameters.tolist() == [0., 1.]


# ---------------------------------------------------------------------------------------------
# Codex N1 round 5: each finding's geometry (owner's standing approval, recommended options)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bound,inside", [(1e4, True), (1e6, False)])
def test_r5_1_a_circular_projection_has_aspect_exactly_one(bound, inside):
    """r5 MEDIUM: a unit circle on a unit Z-axis cylinder whose radial seed is (1, 1, 0), stored
    angle nextafter(1e6, 0), sweep -.2: its projection is exactly circular, but the computed
    aspect x stored angle came out 1000000.0000000001 and the arc was refused OUT_OF_DOMAIN.  Since
    N1 round 6 the bound is 1e4: the same circle at nextafter(1e4, 0) is drawn, at 1e6 refused."""
    g = corpus.GeometryModel()
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 1., 0.), 1., 1., 0., math.tau)
    start = float(np.nextafter(bound, 0.))
    low, _high, _sides = _band(g, support, *(EllipticArc((0., 0., z), (1., 0., 0.), (0., 1., 0.), start, -.2)
                                             for z in (.2, .8)))
    if not inside:
        _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.1)))
        return
    samples = dt.edge_display_samples(g, low, dt.DisplayTolerance(.1))
    _dense_ok(g, low, samples, .1, _default_allowance([support], .1), supports=[support], count=257)


def _offset_frustum_band(radius):
    """A Z-axis frustum with origin (0, 0, -1e8), height 2e8, radii .9 and 1.1 x ``radius``: circular
    arcs of its own radius at z = 0 and 1 over [0, .4], straight sides."""
    g = corpus.GeometryModel()
    support = Cone((0., 0., -1e8), (0., 0., 1.), (1., 0., 0.), .9 * radius, 1.1 * radius, 2e8, -.1, 1.)
    arcs = []
    for z in (0., 1.):
        r = .9 * radius + .2 * radius * (z + 1e8) / 2e8
        arcs.append(EllipticArc((0., 0., z), (r, 0., 0.), (0., r, 0.), 0., .4))
    low, _high, _sides = _band(g, support, *arcs)
    return g, support, low


def test_r5_2_the_conic_radius_scale_is_the_documented_one():
    """r5 MEDIUM: closest approach .001 to a frustum whose origin is at z = -1e8: the documented
    floor is 1e-9 (the conic's definition points) and the subtraction's own rounding 64 eps x 1e8
    = 1.4e-6; the implementation added the support origin to the scale (floor .1) and refused it."""
    g, support, low = _offset_frustum_band(.001)
    tol = dt.DisplayTolerance(.2, 1e-4)
    allowance = dt.display_rounding_allowance(g, low, tol)
    assert allowance.axis_floors[0] <= 64 * EPS * 1e8 * (1 + 1e-12) and allowance.axis_floors[0] < .001
    _dense_ok(g, low, dt.edge_display_samples(g, low, tol), .2, 1e-4, supports=[support], count=257)


def test_r5_2_the_subtractions_rounding_term_binds():
    """The second term of the conic radius rule: a closest approach 1e-6 at 1e8 from the support
    origin is below 64 eps x 1e8 = 1.4e-6 (the definition-point floor is only 1e-9)."""
    g, _support, low = _offset_frustum_band(1e-6)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(.2, 1e-4)))


def _two_cylinder_chord():
    """A straight chord spanning .5 rad on a radius-.01 Z-axis cylinder at the origin, shared with
    a radius-.99 Z-axis cylinder whose origin is (1, 0, -5e8); both faces have vertical rulings."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=.02))
    small = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .01, 1., -.1, .7)
    large = Cylinder((1., 0., -5e8), (0., 0., 1.), (-1., 0., 0.), .99, 1e9, -.1, .2)
    p = [(.01 * math.cos(a), .01 * math.sin(a)) for a in (0., .5)]
    bottom = [g.add_point(x, y, 0.) for x, y in p]
    top = [g.add_point(x, y, 1.) for x, y in p]
    under = [g.add_point(x, y, -1.) for x, y in p]
    chord = g.add_line(bottom[0], bottom[1])
    g.add_face((chord, g.add_line(bottom[1], top[1]), g.add_line(top[1], top[0]), g.add_line(top[0], bottom[0])),
               corners=(0, 1, 2, 3), surface=small)
    g.add_face((chord, g.add_line(bottom[1], under[1]), g.add_line(under[1], under[0]),
                g.add_line(under[0], bottom[0])), corners=(0, 1, 2, 3), surface=large)
    return g, (small, large), chord


def test_r5_3_the_oracle_measures_every_support_at_its_own_floor():
    """r5 MEDIUM: the shared floor .5 (scale 5e8 from the larger cylinder's origin) skipped every
    point on the .01 cylinder, so a one-chord replacement turning .5 rad passed the oracle.  Each
    support's own floor now measures it, and the replacement fails."""
    g, (small, _large), chord = _two_cylinder_chord()
    tol = dt.DisplayTolerance(.2, 1e-4)
    allowance = dt.display_rounding_allowance(g, chord, tol)
    assert allowance.axis_floors[0] <= 1e-9 < allowance.axis_floors[1]
    samples = dt.edge_display_samples(g, chord, tol)
    assert len(samples.parameters) - 1 >= 3
    _dense_ok(g, chord, samples, .2, 1e-4, count=257)
    one = dt.EdgeSamples(chord, np.array([0., 1.]), samples.points[[0, -1]])
    with pytest.raises(AssertionError):
        _dense_ok(g, chord, one, .2, 1e-4, count=257)


def test_r5_4_a_line_range_must_meet_its_topology_vertices():
    """r5 LOW: a [0, 1] carrier (0, 0, 0)-(3, 0, 0) with both vertices 1e-6 off in Y (valid at
    curve_fit_residual 1e-5) was drawn although the model length is 1e-9."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(curve_fit_residual=1e-5))
    edge = g.add_curve(g.add_point(0., 1e-6, 0.), g.add_point(3., 1e-6, 0.), LineRange((0., 0., 0.), (3., 0., 0.)))
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")
    _refused(lambda: dt.display_rounding_allowance(g, edge, dt.DisplayTolerance(.2)), "EDGE_UNSUPPORTED")


def test_r5_5_an_ellipse_beyond_the_aspect_limit_is_out_of_domain():
    """r5 out of scope LOW: u = (1, 0, 0), v = (1, 1e-13, 0) (aspect about 2e13, nondegenerate)
    got EDGE_UNSUPPORTED for parallel axes before the aspect check."""
    g, edge = _lone_curve_edge(EllipticArc((0., 0., 0.), (1., 0., 0.), (1., 1e-13, 0.), 0., .4))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)))


# ---------------------------------------------------------------------------------------------
# Codex N1 round 6 (owner's standing approval, recommended option): the aspect x stored-angle
# bound is 1e4 (was 1e6), inclusive with a 4 eps slack; the in-plane frame's small diagonal is the
# residual's norm
# ---------------------------------------------------------------------------------------------
def test_r6_1_the_skew_reproduction_is_outside_the_new_bound():
    """r6 HIGH: u = (1, 1, 0), v = (1, 1.00001, 0), start .1, sweep 1.6 (aspect x stored angle
    about 680003): at angle 1.047195891 one interval turned 1.047206284359 rad against the bound
    1.047195895832.  Outside the round-6 bound 1e4: OUT_OF_DOMAIN."""
    g, edge = _lone_curve_edge(EllipticArc((0., 0., 0.), (1., 1., 0.), (1., 1.00001, 0.), .1, 1.6))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(1.047195891)))


def _exact_small_diagonal(conic):
    """|u' x v'| / |u'| of the conic's local frame, from its float vectors in exact rationals
    (only the final square root rounded)."""
    lu = [Fraction(float(x)) for x in conic.lu]
    lv = [Fraction(float(x)) for x in conic.lv]
    uu = sum(x * x for x in lu)
    gram = uu * sum(x * x for x in lv) - sum(x * y for x, y in zip(lu, lv)) ** 2
    return math.sqrt(float(gram / uu))


@pytest.mark.parametrize("u,v,start,sweep", [((1., 1., 0.), (1., 1.00001, 0.), .1, 1.6),
                                             ((1., 1., 0.), (1., 1.0004, 0.), .1, 1.6),
                                             ((1., 0., 0.), (0., 1., 100.), -math.pi / 4, math.pi / 2),
                                             ((2., 3., 1.), (2.001, 3., 1.0005), 2., -.8)])
def test_r6_1_the_in_plane_frame_recovers_the_small_diagonal(u, v, start, sweep, monkeypatch):
    """The class of r6 HIGH: the in-plane matrix's small diagonal (the residual of v' against u')
    within 8 eps x aspect relative of its exact value (the stability table), on skew frames inside
    and far outside the bound.  v'.e2 amplified the residual's nonorthogonality to eps x aspect^2
    (relative 1e-5 at the reproduction).  The frame is built with the domain bound lifted (this
    test only): the frame itself does not depend on it."""
    monkeypatch.setattr(dt, "MAX_DISPLAY_ELLIPSE_ASPECT", 1e12)
    conic = dt._Conic(0, (0., 0., 0.), u, v, start, sweep)
    exact = _exact_small_diagonal(conic)
    assert abs(conic.plane[1, 1] - exact) <= 8 * EPS * conic.aspect * exact, (conic.plane[1, 1], exact)


@pytest.mark.parametrize("d", [4.01e-4, 1e-3, 3e-3])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_r6_1_skew_frames_at_the_new_bound_meet_the_tangent_bound(d, n):
    """Skew frames u = (1, 1, 0), v = (1, 1 + d, 0) (aspect about 4 / d, up to 9977; stored angles
    within 1), each at an angle just above its turning / n so every interval is at the limit: the
    dense oracle holds."""
    g, edge = _lone_curve_edge(EllipticArc((0., 0., 0.), (1., 1., 0.), (1., 1. + d, 0.), -.8, 1.6))
    tangent = g.edge_tangent_many(edge, np.array([0., 1.]))
    angle = float(_angles(tangent[:1], tangent[1:])[0]) / n * (1 + 1e-12)
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(angle))
    _dense_ok(g, edge, samples, angle, count=1025)


_AT_LIMIT = 1e4 * (1 + 4 * EPS)       # the last float the inclusive bound admits (4 eps slack)


@pytest.mark.parametrize("scale,inside", [(1e4, True), (_AT_LIMIT, True),
                                          (float(np.nextafter(_AT_LIMIT, 2e4)), False)])
@pytest.mark.parametrize("u,v", [((1., 1., 0.), (-1., 1., 0.)), ((3., 4., 0.), (-4., 3., 0.))],
                         ids=("diagonal", "pythagorean"))
def test_r6_2_a_rotated_frame_exactly_at_the_bound(u, v, scale, inside):
    """r6 MEDIUM (the class): an ellipse with exactly orthogonal, rotated axes, aspect exactly 1e4
    (stored angles within 1) is drawn although its SVD quotient rounds; so is the last float the
    4 eps slack admits; one float beyond it is refused."""
    g, edge = _lone_curve_edge(EllipticArc((0., 0., 0.), tuple(scale * x for x in u), v, 0., .4))
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))  # noqa: E731
    if inside:
        _dense_ok(g, edge, call(), .2, count=257)
    else:
        _refused_out_of_domain(call)


def test_r6_2_the_inclusive_comparison_and_its_slack():
    assert dt._ASPECT_SLACK == 4 * EPS
    assert dt._aspect_admitted(1e4) and dt._aspect_admitted(_AT_LIMIT)
    assert not dt._aspect_admitted(float(np.nextafter(_AT_LIMIT, 2e4))) and not dt._aspect_admitted(math.nan)


def test_r6_2_the_reproduction_at_its_own_bound_is_admitted(monkeypatch):
    """r6 MEDIUM: u = (1e6, 1e6, 0), v = (-1, 1, 0), start 0, sweep .4 (aspect exactly 1e6) was
    refused at the then-bound 1e6 (SVD quotient 1000000.0000000002).  With the bound set back to
    1e6 for this test only, the slacked comparison admits it and the oracle holds; at the real
    bound 1e4 it is OUT_OF_DOMAIN."""
    g, edge = _lone_curve_edge(EllipticArc((0., 0., 0.), (1e6, 1e6, 0.), (-1., 1., 0.), 0., .4))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)))
    monkeypatch.setattr(dt, "MAX_DISPLAY_ELLIPSE_ASPECT", 1e6)
    _dense_ok(g, edge, dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2)), .2, count=257)


def test_r6_3_axis_floor_is_documented_as_a_summary():
    """r6 LOW: callers measuring support turning use axis_floors (one per support); axis_floor is
    a summary only."""
    doc = " ".join(dt.DisplayRoundingAllowance.__doc__.split())
    assert "summary only" in doc and "axis_floors" in doc


@pytest.mark.parametrize("step", corpus.STEPS, ids=("fine", "interaction"))
def test_r6_no_sliver_intervals_over_the_corpus(step):
    """Round-6 follow-up: near-duplicate cuts of different divisions are merged within the
    dropped cut's headroom, so no emitted interval is a sliver: every interval over every corpus
    edge, with and without sag 1e-4, is at least 1024 eps long in parameter (514732a emitted
    1.04e-14 = 47 eps on the steep_cuts ellipses)."""
    for name in CASES:
        case = corpus.ALL[name]()
        g = case.model
        for edge in _case_edges(case):
            for sag in (None, 1e-4):
                try:
                    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(step, sag))
                except dt.DisplayTrimRefused:
                    continue
                assert np.diff(samples.parameters).min() >= 1024 * EPS, (name, edge, sag)


# ---------------------------------------------------------------------------------------------
# N1 round 7: the union's merge fails closed (merge, then verify with exact forward maps;
# restore the dropped cuts of any interval that fails)
# ---------------------------------------------------------------------------------------------
def _far_cylinder_band(origin_x, u_z, heights):
    """A radius-1 Z-axis cylinder with origin (``origin_x``, 0, 0), height 1, and two arcs
    ``(8e8, 0, z) + cos u + sin v`` (``u = (1, 0, u_z)``, ``v = (0, 1, 0)``) over [-1.2, 1.2] at
    ``heights``, joined by lines (surface_residual .001)."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=.001))
    support = Cylinder((origin_x, 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 1., -2., 4.)
    arcs = [EllipticArc((8e8, 0., z), (1., 0., u_z), (0., 1., 0.), -1.2, 2.4) for z in heights]
    low, high, _sides = _band(g, support, *arcs)
    return g, support, (low, high)


def test_r7_1_the_union_does_not_spend_one_headroom_twice():
    """r7 HIGH: at 9efd08e both tangent cuts of the tilted ellipses were replaced by the support
    cuts at 1/3 and 2/3, and the middle interval turned .8000606 rad against the published
    .8000418 (each replacement passed its own check; the two displacements add)."""
    g, support, edges = _far_cylinder_band(8e8, .013, (.2, .8))
    tol = dt.DisplayTolerance(.8000190233461381, .2)
    for edge in edges:
        _dense_ok(g, edge, dt.edge_display_samples(g, edge, tol), tol.angle, tol.sag, supports=[support],
                  count=257)


def test_r7_2_a_merged_circle_interval_meets_the_sag_bound():
    """r7 MEDIUM: unit circles .00025 off the cylinder's axis: at 9efd08e the union moved the
    uniform sag cuts and the circle path skipped the sag check, so the middle chord sagged
    .07894671 against the published .07894327."""
    g, support, edges = _far_cylinder_band(8e8 - .00025, 0., (0., 1.))
    tol = dt.DisplayTolerance(.81, 2 * math.sin(.2) ** 2)
    for edge in edges:
        _dense_ok(g, edge, dt.edge_display_samples(g, edge, tol), tol.angle, tol.sag, supports=[support],
                  count=257)


def test_r7_3_the_mandatory_two_interval_split_is_never_merged():
    """r7 MEDIUM: a lone 1e8 circle over 1e-14 rad with both vertices moved by 1e-4 along x: the
    drawn chord misses the curve by about 1e-4 > surface_residual 1e-5, so the tangent plan
    splits it at .5; at 9efd08e the union merged that cut into the start and returned [0, 1].
    Since N1 round 8 the 1e-14 rad sweep is OUT_OF_DOMAIN; the same split holds at the minimum
    sweep 1e-8 (the drawn chord still misses by about 1e-4)."""
    for sweep in (1e-14, 1e-8):
        g = corpus.GeometryModel(tolerance=TolerancePolicy(curve_fit_residual=1e-3, surface_residual=1e-5))
        arc = EllipticArc((0., 0., 0.), (1e8, 0., 0.), (0., 1e8, 0.), 0., sweep)
        ends = np.asarray(arc.evaluate(np.array([0., 1.])), dtype=float)
        edge = g.add_curve(*(g.add_point(p[0] + 1e-4, p[1], p[2]) for p in ends), arc)
        if sweep < dt.MIN_DISPLAY_SWEEP:
            _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)))
            continue
        samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))
        assert len(samples.parameters) - 1 >= 2, samples.parameters.tolist()
        assert .5 in samples.parameters.tolist()


def test_r7_a_rejected_merge_restores_the_original_cuts():
    """A verification that fails every merged interval leaves the no-merge union: the original
    cuts, none invented (exact duplicates are one cut)."""
    first = dt._Division([.25, .5], lambda t: t)
    other = dt._Division([.25, .5 + 1e-14, .75], lambda t: t)
    calls = []

    def reject(a, b):
        calls.append((a, b))
        return False

    assert dt._union(0, [first, other], [1e-3, 1e-3], reject).tolist() == [0., .25, .5, .5 + 1e-14, .75, 1.]
    assert calls                                     # the merged intervals were checked
    assert dt._union(0, [first, other], [1e-3, 1e-3], _ACCEPT).tolist() == [0., .25, .5, .75, 1.]


def test_r7_another_original_cut_may_stand_for_its_group():
    """An interval that fails first tries another ORIGINAL cut of the group (here the later of
    the two near cuts), before every cut is restored."""
    first = dt._Division([.5], lambda t: t)
    other = dt._Division([.5 + 1e-14], lambda t: t)
    # accept only the intervals whose middle end is .5 + 1e-14
    verify = lambda a, b: .5 not in (a, b)  # noqa: E731
    assert dt._union(0, [first, other], [1e-3, 1e-3], verify).tolist() == [0., .5 + 1e-14, 1.]


def test_r7_the_mandatory_split_survives_when_merged_into_an_end():
    """A protected (minimum-two) division's cut merged into an end is restored: the result is never
    one chord; merged into another interior cut it may go (the edge has two intervals anyway)."""
    split = dt._Division([.5], lambda t: 1e-16 * t, protected=True)
    assert dt._union(0, [split], [1.], _ACCEPT).tolist() == [0., .5, 1.]
    near = dt._Division([.5 + 1e-14], lambda t: t)
    assert dt._union(0, [split, near], [1., 0.], _ACCEPT).tolist() == [0., .5 + 1e-14, 1.]


# ---------------------------------------------------------------------------------------------
# N1 round 8: the minimum sweep (owner decision, the recommended option: |sweep| >= 1e-8 rad,
# MIN_DISPLAY_SWEEP); a forced split never cuts at an end; the azimuth branch keeps its sign
# ---------------------------------------------------------------------------------------------
def _short_ellipse_edge(sweep):
    """r8 MEDIUM 1: a lone 1e8 x 5e7 ellipse from 0 over ``sweep``, both vertices moved by 1e-4
    along x (curve_fit_residual 1e-3, surface_residual 1e-5)."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-12, curve_fit_residual=1e-3, surface_residual=1e-5))
    arc = EllipticArc((0., 0., 0.), (1e8, 0., 0.), (0., 5e7, 0.), 0., sweep)
    ends = np.asarray(arc.evaluate(np.array([0., 1.])), dtype=float)
    edge = g.add_curve(*(g.add_point(p[0] + 1e-4, p[1], p[2]) for p in ends), arc)
    return g, edge


def _r8_turnback_band():
    """r8 MEDIUM 2: a Z-axis cylinder of radius R = 6.3e-5 with a band whose first side is two
    lines and an ellipse over 5e-11 rad whose g = 1 + 1.000000002 cos(theta) is strictly negative
    (-6.28e-16 to -3.79e-15, both ends within the snapping band)."""
    z = 1. + 2e-9
    theta0 = math.acos(-1. / z)
    xx, yy = z + math.cos(theta0), math.sin(theta0)
    radius = math.hypot(xx, yy)
    azimuth = math.atan2(yy, xx)
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-14, curve_fit_residual=1e-10, surface_residual=1e-9))
    support = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), radius, radius, 0., 3.)
    arc = EllipticArc((z, 0., .5 * radius - 10. * yy), (1., 0., 0.), (0., 1., 10.), theta0 + 1e-11, 5e-11)
    p, q = arc.evaluate(np.array([0., 1.]))

    def cap(h):
        c = EllipticArc((0., 0., h * radius), (radius, 0., 0.), (0., radius, 0.), azimuth, .5)
        i, j = [g.add_point(*x) for x in c.evaluate(np.array([0., 1.]))]
        return i, j, g.add_curve(i, j, c)

    i, j, low = cap(.3)
    k, l, high = cap(.8)
    m, n = [g.add_point(*x) for x in (p, q)]
    edge = g.add_curve(m, n, arc)
    g.add_face((low, g.add_line(j, l), high, g.add_line(k, m), edge, g.add_line(n, i)), corners=(0, 1, 2, 3),
               surface=support)
    assert g.validate_topology() == ()
    return g, support, arc, edge


def test_r8_the_minimum_sweep_is_declared():
    assert dt.MIN_DISPLAY_SWEEP == 1e-8


def test_r8_1_a_short_ellipse_below_the_minimum_sweep_is_out_of_domain():
    """r8 MEDIUM 1: at 5f69ccb the forced two-interval tangent plan of the 1e-16 rad ellipse
    rounded its cut to 0 and the edge was drawn as one chord 1e-4 off the curve."""
    g, edge = _short_ellipse_edge(1e-16)
    assert g.validate_topology() == ()
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)))


def test_r8_2_the_turnback_reproduction_is_out_of_domain():
    """r8 MEDIUM 2: at 5f69ccb the strictly monotone 5e-11 rad edge was refused EDGE_UNSUPPORTED
    as a turnback; its sweep is below the minimum, so it is OUT_OF_DOMAIN."""
    g, _support, _arc, edge = _r8_turnback_band()
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1, .001)))


@pytest.mark.parametrize("sign", [1., -1.], ids=("forward", "reverse"))
def test_r8_the_minimum_sweep_boundary_and_the_next_float_below(sign):
    """|sweep| = 1e-8 exactly is admitted (inclusive) and drawn; the next float below is refused
    before any division, for both directions of an ellipse arc."""
    g, edge = _arc_edge((0., 0., 0.), .3, sign * 1e-8, u=(2., 0., 0.), v=(0., 1., 0.))
    samples = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1))
    assert samples.parameters.tolist() == [0., 1.]
    below = sign * float(np.nextafter(1e-8, 0.))
    g, edge = _arc_edge((0., 0., 0.), .3, below, u=(2., 0., 0.), v=(0., 1., 0.))
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)))
    _refused_out_of_domain(lambda: dt.display_rounding_allowance(g, edge, dt.DisplayTolerance(.1)))


def test_r8_a_three_point_arc_cannot_reach_below_the_minimum_sweep():
    """The bound applies to an Arc too (its sweep from the model's arc frame), but a three-point
    arc is refused by the model's own collinearity rule (sin^2 of the chord angle <= 1e-12) long
    before: at a sweep of 4e-9 it cannot be built; the smallest buildable sweep is drawn."""
    from anygeometry.curves import DegenerateArcError
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-14))
    points = [(math.cos(t), math.sin(t), 0.) for t in (0., 2e-9, 4e-9)]
    with pytest.raises(DegenerateArcError):
        g.add_arc(*(g.add_point(*p) for p in points))
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-14))
    points = [(math.cos(t), math.sin(t), 0.) for t in (0., 4e-6, 8e-6)]
    edge = g.add_arc(*(g.add_point(*p) for p in points))
    assert 1e-8 < abs(g.arc_frame(edge).sweep) < 1e-5
    assert dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1)).parameters.tolist() == [0., 1.]


def test_r8_1_a_forced_split_never_cuts_at_an_end():
    """Inside the module (the class, below the public domain): a forced two-interval tangent plan
    whose inverse rounds the cut onto an end uses the parametric midpoint .5 (exactly
    representable and distinct; 5f69ccb returned the cut 0.0)."""
    conic = dt._Conic(0, (0., 0., 0.), (1e8, 0., 0.), (0., 5e7, 0.), 0., 1e-16)
    plan = conic.tangent_division(.1, None, 1e-5, 1e-4)
    assert plan.count == 2
    division = plan.build()
    assert division.protected and division.values == [.5]


def test_r8_2_both_ends_in_the_band_are_near_stationary():
    """Inside the module: both end values of g (computed negative, -6.3e-16 to -3.8e-15) lie
    within g's rounding bound.  5f69ccb refused it as a turnback; round 8 inferred the branch from
    the unsnapped values and drew it; since round 9 (owner decision: no branch from a value
    within its bound) it is near-stationary, OUT_OF_DOMAIN."""
    _g, support, arc, _edge = _r8_turnback_band()
    conic = dt._Conic(0, arc.center, arc.u_vector, arc.v_vector, arc.start_angle, arc.sweep_angle)
    centre = np.asarray(arc.center, dtype=float)
    with pytest.raises(dt.DisplayDomainRefused, match="near-stationary"):
        conic.azimuth_division(support, .1, [centre, centre + arc.u_vector, centre + arc.v_vector])


def test_r8_2_opposite_signs_across_the_band_are_still_refused():
    """The sign rule fails closed: g = 1 + z cos(x) with z = 1 + 1e-15 about x = pi is negative
    inside and positive at both ends: refused as before."""
    z = 1. + 1e-15
    with pytest.raises(dt.DisplayTrimRefused):
        dt._AngleMap(0, np.eye(2), (z, 0.), 0., math.pi - 1e-7, math.pi + 1e-7)


def _last_admitted(predicate, start):
    """The smallest float ``x`` near ``start`` with ``predicate(x)`` true and false at the next
    float below (walked by single floats; the predicate is monotone there)."""
    x = start
    if predicate(x):
        while predicate(float(np.nextafter(x, 0.))):
            x = float(np.nextafter(x, 0.))
    else:
        while not predicate(x):
            x = float(np.nextafter(x, 1e300))
    return x


def test_r8_3_the_angle_share_cutoff_is_inclusive_and_pinned():
    """r8 LOW: the 2 % angle share, pinned at its last admitted display angle (allowance <= .02
    x angle, inclusive) and refused OUT_OF_DOMAIN at the next float below."""
    g, support, low = _circle_band_on_frustum()

    def share(angle):
        return dt.display_rounding_allowance(g, low, dt.DisplayTolerance(angle, 1e-6)).angle <= .02 * angle

    rough = dt.display_rounding_allowance(g, low, dt.DisplayTolerance(1e-3, 1e-6)).angle
    cutoff = _last_admitted(share, rough / (.02 - 64 * EPS))
    below = float(np.nextafter(cutoff, 0.))
    assert share(cutoff) and not share(below)
    _dense_ok(g, low, dt.edge_display_samples(g, low, dt.DisplayTolerance(cutoff, 1e-6)), cutoff, 1e-6,
              supports=[support], count=129)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, low, dt.DisplayTolerance(below, 1e-6)))


def test_r8_3_the_sag_share_cutoff_is_inclusive_and_pinned():
    """r8 LOW: the 2 % sag share, pinned at its last admitted sag (allowance.sag <= .02 x sag,
    inclusive) and refused EDGE_UNSUPPORTED at the next float below."""
    g, edge = _arc_edge((1e9 - 2., 0., 0.), 0., .03)

    def share(sag):
        return dt.display_rounding_allowance(g, edge, dt.DisplayTolerance(.0153, sag)).sag <= .02 * sag

    rough = dt.display_rounding_allowance(g, edge, dt.DisplayTolerance(.0153, 3e-4)).sag
    cutoff = _last_admitted(share, rough / .02)
    below = float(np.nextafter(cutoff, 0.))
    assert share(cutoff) and not share(below)
    _dense_ok(g, edge, dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, cutoff)), .0153, cutoff,
              count=257)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, below)), "EDGE_UNSUPPORTED")


# ---------------------------------------------------------------------------------------------
# N1 round 9 (owner decision, the recommended option): the azimuth branch is decided from g only
# when its sign is decidable with margin; |g| within its rigorous rounding bound at an end, an
# interior extreme or the midpoint is near-stationary: OUT_OF_DOMAIN, never snapped
# ---------------------------------------------------------------------------------------------
def _r9_cone_band(gap):
    """r9 MEDIUM: a Z-axis cone at the origin (radius 0 to 1e-6 over height 1e-6), a band from
    azimuth pi/2 over .5 between heights 5e-9 and 3e-8 whose first side is two lines and an
    ellipse (centre (nextafter(1, inf), 0, 0), u = (1, 0, 0), v = (0, 1, 1)) from pi - gap over
    the minimum sweep 1e-8."""
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-14, curve_fit_residual=1e-12,
                                                       surface_residual=1e-12))
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 0., 1e-6, 1e-6, math.pi / 2, .5)
    arc = EllipticArc((float(np.nextafter(1., 2.)), 0., 0.), (1., 0., 0.), (0., 1., 1.), math.pi - gap, 1e-8)
    p, q = arc.evaluate(np.array([0., 1.]))

    def cap(h):
        c = EllipticArc((0., 0., h), (h, 0., 0.), (0., h, 0.), math.pi / 2, .5)
        i, j = [g.add_point(*x) for x in c.evaluate(np.array([0., 1.]))]
        return i, j, g.add_curve(i, j, c)

    i, j, low = cap(5e-9)
    k, l, high = cap(3e-8)
    m, n = [g.add_point(*x) for x in (p, q)]
    edge = g.add_curve(m, n, arc)
    g.add_face((low, g.add_line(j, l), high, g.add_line(k, m), edge, g.add_line(n, i)), corners=(0, 1, 2, 3),
               surface=support)
    assert g.validate_topology() == ()
    return g, support, arc, edge


@pytest.mark.parametrize("gap", [2.06e-8, 2.40e-8], ids=("monotone", "turnback"))
def test_r9_a_near_stationary_azimuth_is_out_of_domain(gap):
    """r9 MEDIUM: at 5bc1cd8 the strictly monotone variant (g from -9.9e-18 to -1.7e-16, computed
    0) was refused EDGE_UNSUPPORTED as a turnback and the variant with a real 1.8e-10 rad
    turnback (g from +6.6e-17 to -1.2e-16) was drawn as [0, 1].  Both are near-stationary."""
    g, _support, _arc, edge = _r9_cone_band(gap)
    _refused_out_of_domain(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1, .001)))


@pytest.mark.parametrize("gap", [2.06e-8, 2.40e-8], ids=("monotone", "turnback"))
def test_r9_the_branch_decision_refuses_the_variants_directly(gap):
    """The branch decision itself (the support's azimuth plan), not only the dense oracle."""
    _g, support, arc, _edge = _r9_cone_band(gap)
    conic = dt._Conic(0, arc.center, arc.u_vector, arc.v_vector, arc.start_angle, arc.sweep_angle)
    centre = np.asarray(arc.center, dtype=float)
    with pytest.raises(dt.DisplayDomainRefused, match="near-stationary"):
        conic.azimuth_division(support, .1, [centre, centre + arc.u_vector, centre + arc.v_vector])


def _unit_map(xa, xb, z=1.):
    """g = 1 + z cos(x) over [xa, xb]: the unit circle seen from (-z, 0)."""
    return dt._AngleMap(0, np.eye(2), (z, 0.), 0., xa, xb)


def test_r9_the_branch_is_accepted_beyond_the_bound_and_refused_within_it():
    """Direct accept/refuse: g = 1 + cos x, which is d^2 / 2 at x = pi - d (positive, the
    branch +1) or x = pi + d (the branch -1 for z > 1 there, refused as a turnback at z = 1
    across pi).  An end with g = 4 x its bound is accepted with the right branch; with g a
    quarter of the bound it is refused OUT_OF_DOMAIN; an exactly stationary end too."""
    bound = _unit_map(2., 3.).g_bound
    assert 0. < bound < 1e-12
    far = math.sqrt(8. * bound)                       # g = d^2 / 2 = 4 bound at the end
    near = math.sqrt(.5 * bound)                      # g = bound / 4
    assert _unit_map(2., math.pi - far).branch == 1.
    with pytest.raises(dt.DisplayDomainRefused, match="near-stationary"):
        _unit_map(2., math.pi - near)
    assert _unit_map(math.pi + far, 4.).branch == 1.  # the near end first in x order
    with pytest.raises(dt.DisplayDomainRefused, match="near-stationary"):
        _unit_map(math.pi + near, 4.)
    # z = 2: g = 1 + 2 cos x is negative on (2 pi / 3, 4 pi / 3): the branch -1 well inside
    assert _unit_map(2.2, 4., z=2.).branch == -1.
    assert _unit_map(.1, 2.).branch == 1. and _unit_map(.1, 2., z=.5).branch == 1.
    # a sign change with margin is still a turnback (EDGE_UNSUPPORTED)
    with pytest.raises(dt.DisplayTrimRefused) as caught:
        _unit_map(1., 3., z=2.)
    assert caught.value.code == "EDGE_UNSUPPORTED"


def test_r9_the_midpoint_check_refuses_a_near_stationary_interior():
    """g near 0 at the interval's midpoint only (z = 1, the ends at pi -+ d with g = d^2 / 2 far
    beyond the bound, an interior extreme at pi where g = 0): OUT_OF_DOMAIN, not a branch."""
    d = 1e-3
    with pytest.raises(dt.DisplayDomainRefused, match="near-stationary"):
        _unit_map(math.pi - d, math.pi + d)


def test_r9_the_tangent_map_is_never_near_stationary():
    """The tangent division's map (offset exactly 0: g = 1 everywhere) always decides +1."""
    amap = dt._AngleMap(0, ((2., .3), (0., 1.)), (0., 0.), math.pi / 2, 0., 1e-8)
    assert amap.branch == 1. and amap.g_bound < 1e-13


# ---------------------------------------------------------------------------------------------
# N1 round 10 (owner decision, the recommended option): the azimuth is computed in the module's
# OWN orthonormal frame about the support's unit axis, and a cylinder or cone whose stored frame
# is self-inconsistent (orthonormality error above 1e-9) is refused OUT_OF_DOMAIN ("support frame
# not orthonormal"); a stored frame's ordinary rounding is never refused
# ---------------------------------------------------------------------------------------------
_R10_A, _R10_E, _R10_F = np.array([[1., 1., 0.], [1., -1., 0.], [0., 0., -1.]])
_R10_R, _R10_S = 2. ** -10, math.sqrt(2)
_R10_PHI = math.atan2(math.sqrt(3) / 2, 1.5 * _R10_S)
_R10_BAD = (1 + 1e-11, 1 - 1e-11, 0)       # nearly the axis: the stored radial is 2.2e-5 off orthogonal
_R10_GOOD = (1., -1., 0.)                   # exactly orthogonal


def _r10_arc(gap):
    return EllipticArc(_R10_R * (2 * _R10_E + _R10_A), _R10_R * _R10_E, _R10_R * (_R10_F + _R10_A / 2),
                       2 * math.pi / 3 - gap, 1e-5)


def _r10_band(gap, seed):
    """r10 MEDIUM: a cylinder about (1, 1, 0) with the radial seed ``seed``; a band between two
    caps whose target side is an ellipse from 2 pi / 3 - gap over 1e-5 whose projection across
    the axis has g = 1 + 2 cos(theta)."""
    a, e, f, r, s, phi = _R10_A, _R10_E, _R10_F, _R10_R, _R10_S, _R10_PHI
    support = Cylinder((0, 0, 0), a, seed, r * math.sqrt(21) / 2, .004, phi, .5)
    g = corpus.GeometryModel(tolerance=TolerancePolicy(length=1e-14, surface_residual=1e-7))

    def edge(curve):
        i, j = [g.add_point(*p) for p in curve.evaluate(np.array([0., 1.]))]
        return i, j, g.add_curve(i, j, curve)

    def cap(h):
        return edge(EllipticArc(h * a / s, support.radius * e / s, support.radius * f, phi, .5))

    i, j, low = cap(r)
    k, l, high = cap(3 * r)
    m, n, target = edge(_r10_arc(gap))
    lines = (g.add_line(j, l), g.add_line(k, m), g.add_line(n, i))
    g.add_face((low, lines[0], high, lines[1], target, lines[2]), corners=(0, 1, 2, 3), surface=support)
    assert g.validate_topology() == ()
    return g, support, target, lines


def _r10_plan(support, gap):
    arc = _r10_arc(gap)
    conic = dt._Conic(0, arc.center, arc.u_vector, arc.v_vector, arc.start_angle, arc.sweep_angle)
    centre = np.asarray(arc.center, dtype=float)
    return conic.azimuth_division(support, .1, [centre, centre + arc.u_vector, centre + arc.v_vector])


@pytest.mark.parametrize("gap", [3e-6, 1.3e-5], ids=("turnback", "monotone"))
def test_r10_a_support_frame_not_orthonormal_is_out_of_domain(gap):
    """r10 MEDIUM: at 3f066d2 the turnback arc (exact g +5.2e-6 to -1.21e-5) was drawn as [0, 1]
    and the strictly monotone arc (+2.25e-5 to +5.2e-6) was refused EDGE_UNSUPPORTED: the azimuth
    was projected onto a stored radial 2.2e-5 off orthogonal to the axis.  Both are refused, the
    band's lines too (a support domain check)."""
    g, support, edge, lines = _r10_band(gap, _R10_BAD)
    assert abs(float(support.axis @ support.radial_direction)) > 1e-6
    tol = dt.DisplayTolerance(.1, 1e-6)
    for e in (edge, *lines):
        with pytest.raises(dt.DisplayDomainRefused, match="support frame not orthonormal"):
            dt.edge_display_samples(g, e, tol)
        with pytest.raises(dt.DisplayDomainRefused, match="support frame not orthonormal"):
            dt.display_rounding_allowance(g, e, tol)


def test_r10_the_orthogonal_seed_gives_the_correct_results():
    """The same band with the exactly orthogonal radial seed (1, -1, 0): the turnback arc is
    refused as a turnback (EDGE_UNSUPPORTED, not drawn) and the monotone arc is drawn, its
    support azimuth strictly monotone and within the angle on every interval."""
    g, support, edge, _lines = _r10_band(3e-6, _R10_GOOD)
    _refused(lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1, 1e-6)), "EDGE_UNSUPPORTED")
    g, support, edge, _lines = _r10_band(1.3e-5, _R10_GOOD)
    t = dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1, 1e-6)).parameters
    assert t[0] == 0. and t[-1] == 1. and np.all(np.diff(t) > 0)
    turning = np.diff(np.unwrap(_azimuth(support, g.evaluate_edge_many(edge, np.linspace(0., 1., 257)))))
    assert np.all(turning > 0) or np.all(turning < 0)
    for a, b in zip(t[:-1], t[1:]):
        ends = _azimuth(support, g.evaluate_edge_many(edge, np.array([a, b])))
        assert abs(float(np.diff(np.unwrap(ends))[0])) <= .1


def test_r10_the_branch_decision_is_made_in_the_module_frame():
    """The branch decision itself (the support's azimuth plan) on the orthogonal-seed support:
    the monotone arc is accepted, the turnback arc refused as a turnback."""
    _g, support, _edge, _lines = _r10_band(1.3e-5, _R10_GOOD)
    assert _r10_plan(support, 1.3e-5).count >= 1
    with pytest.raises(dt.DisplayTrimRefused) as caught:
        _r10_plan(support, 3e-6)
    assert caught.value.code == "EDGE_UNSUPPORTED"


@pytest.mark.parametrize("kind", ["cylinder", "cone"])
def test_r10_the_support_frame_check_refuses_cylinders_and_cones(kind):
    """The domain check itself: a stored radial off orthogonal to the axis is refused for a
    cylinder and a cone; the exactly orthogonal seed is admitted."""
    def make(seed):
        if kind == "cylinder":
            return Cylinder((0, 0, 0), (1., 1., 0.), seed, 1., 1., 0., 1.)
        return Cone((0, 0, 0), (1., 1., 0.), seed, .5, 1., 1., 0., 1.)

    with pytest.raises(dt.DisplayDomainRefused, match="support frame not orthonormal"):
        dt._check_support_domain(make(_R10_BAD))
    dt._check_support_domain(make(_R10_GOOD))


def test_r10_the_module_frame_is_orthonormal_and_independent_of_the_stored_phase():
    """The module's frame about the unit axis is orthonormal to 4 eps for random axes (the
    stored radial is not used); the same arc on cylinders whose stored radials differ only in
    phase gets bit-identical cuts (turning is independent of the frame's phase)."""
    rng = np.random.default_rng(10)
    worst = 0.
    for _ in range(2000):
        axis = rng.normal(size=3) * 10. ** rng.uniform(-3, 3, size=3)
        support = Cylinder((0, 0, 0), axis, rng.normal(size=3), 1., 1.)
        _o, e1, e2 = dt._axis_frame(support)
        u = np.asarray(support.axis)
        worst = max(worst, abs(e1 @ u), abs(e2 @ u), abs(e1 @ e2), abs(e1 @ e1 - 1.), abs(e2 @ e2 - 1.),
                    abs(float(np.cross(e1, e2) @ u) - 1.))
    assert worst <= 4. * EPS, worst / EPS
    arc = EllipticArc((2., 0., .3), (.5, 0., .1), (0., .7, 0.), .2, 1.2)
    centre = np.asarray(arc.center, dtype=float)
    cuts = []
    for phase in (0., .7, 2.5, -3.):
        support = Cylinder((0, 0, 0), (0., 0., 1.), (math.cos(phase), math.sin(phase), 0.), 2., 1.)
        conic = dt._Conic(0, arc.center, arc.u_vector, arc.v_vector, arc.start_angle, arc.sweep_angle)
        plan = conic.azimuth_division(support, .05, [centre, centre + arc.u_vector, centre + arc.v_vector])
        cuts.append(np.asarray(plan.build().values, dtype=float))
    assert len(cuts[0]) > 3 and all(np.array_equal(c, cuts[0]) for c in cuts)


def test_r10_random_ordinary_supports_are_never_refused_for_their_frame():
    """The frame check is a sanity check for self-inconsistent supports only: 10,000 random
    cylinders and 10,000 random cones (fixed seed; random axes and radial seeds over six orders
    of magnitude per component) are all admitted, though their stored frames are off by
    rounding (the first 8 eps version refused 1756 of them; worst 1.8e-11, 55x below 1e-9)."""
    rng = np.random.default_rng(20261126)
    refused, built, worst = 0, 0, 0.
    for k in range(20000):
        axis = rng.normal(size=3) * 10. ** rng.uniform(-3, 3, size=3)
        seed = rng.normal(size=3) * 10. ** rng.uniform(-3, 3, size=3)
        try:
            support = (Cylinder((0, 0, 0), axis, seed, 1., 1.) if k % 2 == 0
                       else Cone((0, 0, 0), axis, seed, .5, 1., 1.))
        except GeometryError:
            continue
        built += 1
        u, r, c = (np.asarray(v) for v in (support.axis, support.radial_direction, support.circumferential_direction))
        worst = max(worst, abs(u @ r), abs(u @ c), abs(np.linalg.norm(r) - 1.))
        try:
            dt._check_support_domain(support)
        except dt.DisplayDomainRefused:
            refused += 1
    assert refused == 0 and built >= 19990, (refused, built)
    assert worst > 8. * EPS                   # the set includes frames the 8 eps version refused


# ---------------------------------------------------------------------------------------------
# N1 round 11: the stored axis need not be unit (a seed whose squared norm is subnormal leaves it
# at 0.9999999995, or 1.125): _axis_frame normalises it, and the 1e-9 sanity check covers the
# lengths of axis, radial and circumferential and all three pairwise dot products
# ---------------------------------------------------------------------------------------------
_R11_Y, _R11_Z = -math.sin(.1), math.cos(.1)


def _r11_support(kind, multiplier):
    axis = (math.ldexp(1., -537) * multiplier, 0., 0.)
    if kind == "cylinder":
        return Cylinder((0, 0, 0), axis, (0, 1, 1), 1, 1)
    return Cone((0, 0, 0), axis, (0, 1, 1), 1, 2, 1)


def _r11_band(kind, multiplier):
    """r11 MEDIUM: a quad on a support about the x axis whose first edge is a line at x = 0
    turning exactly 0.1 rad about the axis, from (0, 0, 1) to (0, -sin .1, cos .1)."""
    support = _r11_support(kind, multiplier)
    far = 1. if kind == "cylinder" else 2.             # the cone's radius at x = 1
    # the 1.125 frame evaluates a skewed surface (axial and circumferential scaled by 1.125): its
    # face is admitted by the model only at a coarse surface residual
    residual = .01 if multiplier < 1.1 else .5
    g = corpus.GeometryModel(tolerance=TolerancePolicy(surface_residual=residual))
    v = [g.add_point(*p) for p in ((0, 0, 1), (0, _R11_Y, _R11_Z), (1, far * _R11_Y, far * _R11_Z), (1, 0, far))]
    e = [g.add_line(v[i], v[(i + 1) % 4]) for i in range(4)]
    g.add_face(e, corners=(0, 1, 2, 3), surface=support)
    return g, support, e[0]


@pytest.mark.parametrize("kind", ["cylinder", "cone"])
def test_r11_a_subnormal_seed_axis_is_normalised_and_drawn_within_the_bound(kind):
    """r11 MEDIUM, variant 1: the seed (2^-537 (1 - 5e-10), 0, 0) leaves the stored axis at length
    0.9999999995 (within the 1e-9 sanity check: ADMITTED).  At 2787703 the line turning 0.1 rad
    was drawn as one interval at angle 0.1 - 2e-11 (allowance 1.3e-14): 2e-11 rad over the bound.
    With the axis normalised it is DRAWN, every interval within angle + allowance about the
    exact x axis."""
    g, support, edge = _r11_band(kind, 1 - 5e-10)
    assert abs(float(np.linalg.norm(support.axis)) - 1.) > 1e-10
    assert g.validate_topology() == ()
    tol = dt.DisplayTolerance(.1 - 2e-11, .1)
    t = dt.edge_display_samples(g, edge, tol).parameters
    limit = tol.angle + dt.display_rounding_allowance(g, edge, tol).angle
    points = g.evaluate_edge_many(edge, t)
    azimuth = np.unwrap(np.arctan2(points[:, 1], points[:, 2]))        # about the exact x axis
    assert len(t) >= 3 and float(np.abs(np.diff(azimuth)).max()) <= limit


@pytest.mark.parametrize("kind", ["cylinder", "cone"])
def test_r11_axis_and_circumferential_lengths_of_1_125_are_out_of_domain(kind):
    """r11 MEDIUM, variant 2: the seed (2^-537 1.125, 0, 0) leaves the stored axis and
    circumferential direction at length 1.125 (admitted at 2787703): REFUSED OUT_OF_DOMAIN."""
    g, support, edge = _r11_band(kind, 1.125)
    assert float(np.linalg.norm(support.axis)) == 1.125
    with pytest.raises(dt.DisplayDomainRefused, match="support frame not orthonormal"):
        dt._check_support_domain(support)
    with pytest.raises(dt.DisplayDomainRefused, match="support frame not orthonormal"):
        dt.edge_display_samples(g, edge, dt.DisplayTolerance(.1, .1))


def test_r11_the_module_frame_is_built_about_the_normalised_axis():
    """_axis_frame normalises the stored axis: its frame is orthonormal to 4 eps about the unit
    axis direction even when the stored axis is 0.9999999995 or 1.125 long."""
    for multiplier in (1 - 5e-10, 1.125):
        support = _r11_support("cylinder", multiplier)
        _o, e1, e2 = dt._axis_frame(support)
        u = np.array((1., 0., 0.))
        worst = max(abs(e1 @ u), abs(e2 @ u), abs(e1 @ e2), abs(e1 @ e1 - 1.), abs(e2 @ e2 - 1.),
                    abs(float(np.cross(e1, e2) @ u) - 1.))
        assert worst <= 4. * EPS, (multiplier, worst / EPS)
