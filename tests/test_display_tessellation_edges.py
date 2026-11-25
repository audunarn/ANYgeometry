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
SLACK = 1e-6
EPS = np.finfo(float).eps


def _angles(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("...i,...i->...", a, b))


def _azimuth(support, points):
    offset = np.atleast_2d(points) - support.origin
    radial = offset - np.outer(offset @ support.axis, support.axis)
    return np.arctan2(radial @ support.circumferential_direction, radial @ support.radial_direction)


def _support_normal_turning(support, points):
    """The azimuth swept about a cylinder's or cone's axis (exact for a cylinder, a conservative
    upper bound for a cone, whose normal turns less than its azimuth); a point on the axis (a
    cone's apex) has no azimuth and contributes nothing."""
    offset = points - support.origin
    radial = offset - np.outer(offset @ support.axis, support.axis)
    keep = np.linalg.norm(radial, axis=1) > 1e-12 * max(1.0, float(np.abs(points).max()))
    azimuth = _azimuth(support, points)
    step = np.abs((np.diff(azimuth) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.where(keep[1:] & keep[:-1], step, 0.).sum())


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


def _dense_ok(g, edge_id, samples, angle, sag=None, supports=None, count=DENSE, label=None):
    """Every drawn segment: dense tangent turning, support azimuth and distance from the segment."""
    supports = _round_supports(g, edge_id) if supports is None else supports
    t = samples.parameters
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0.0)
    for k, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
        dense = np.linspace(t0, t1, count)
        tangent = g.edge_tangent_many(edge_id, dense)
        assert _angles(tangent[1:], tangent[:-1]).sum() <= angle * (1 + SLACK), (label, edge_id, t0, t1)
        points = g.evaluate_edge_many(edge_id, dense)
        for support in supports:
            assert _support_normal_turning(support, points) <= angle * (1 + SLACK), (label, edge_id, t0, t1)
        if sag is not None:
            distance = _distance_to_segment(points, samples.points[k], samples.points[k + 1])
            assert distance <= sag * (1 + 1e-9) + 1e-12, (label, edge_id, t0, t1, distance, sag)


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
    """A sag tighter than the angular default bounds every drawn segment's distance from its curve."""
    case = corpus.ALL[name]()
    g = case.model
    for edge_id in _curved_edges(case):
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
    assert np.abs(np.diff(azimuth)).max() <= .1 * (1 + SLACK)
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
    assert dt.MIN_DISPLAY_RADIUS_RATIO == 1e-9
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
    """A LineRange whose retained half is near the origin, its carrier start at or beyond the bound."""
    g = corpus.GeometryModel()
    carrier = LineRange((float(x), 0., 0.), (1., 0., 0.), (Fraction(1), Fraction(1, 2)))
    ends = carrier.evaluate(np.array([0., 1.]))
    edge = g.add_curve(g.add_point(*ends[0]), g.add_point(*ends[1]), carrier)
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.2))  # noqa: E731
    if inside:
        assert call().parameters.tolist() == [0., 1.]
    else:
        _refused_out_of_domain(call)


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
    call = lambda: dt.edge_display_samples(g, edge, dt.DisplayTolerance(.0153, 3e-5))  # noqa: E731
    if inside:
        _dense_ok(g, edge, call(), .0153, 3e-5, count=257)
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
