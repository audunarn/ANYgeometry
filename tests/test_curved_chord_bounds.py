"""Exact sufficient COS chord bounds and real owner receipt integration."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
from anygeometry import prepared_material_curvilinear as proof
from anygeometry.nurbs_domain import _exact_segment_distance


def curve(scale=1, weights=None, domain=(0., 1.)):
    s = float(scale)
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., s, 0.)), ((s, 0., 0.), (s, s, 0.))))
    return CurveOnSurface(support, NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        ((0., 0.), (.25, .75), (1., 1.)), weights, domain))


@pytest.mark.parametrize('scale', (Q(1, 4), Q(1), Q(8)))
@pytest.mark.parametrize('interval', ((Q(0), Q(1)), (Q(1, 4), Q(3, 4)), (Q(3, 4), Q(1, 4))))
def test_exact_polynomial_descriptor_taylor_and_endpoint_term(scale, interval):
    c = curve(scale)
    calls = []
    upper = proof._cos_c2_world_chord_bound(c, *interval, Q(1, 1000), lambda: calls.append(1))
    # The projected sufficient bound may now improve the original full L1
    # Taylor reference, but must still contain the exact midpoint deviation.
    assert upper <= scale * (interval[1]-interval[0])**2/4 + Q(1, 1000)
    a, b = sorted(interval)
    midpoint = c.exact_point((a+b)/2)
    assert _exact_segment_distance(midpoint, c.exact_point(a), c.exact_point(b)) <= upper
    assert len(calls) == 2


@pytest.mark.parametrize('domain', ((0., 1.), (.25, .75), (.75, .25)))
def test_rational_cropped_reversed_bound_contains_independent_exact_witnesses(domain):
    c = curve(weights=(1., 2., 1.), domain=domain)
    upper = proof._cos_c2_world_chord_bound(c, Q(1, 4), Q(3, 4), 0, lambda: None)
    assert upper is not None
    first, last = (c.exact_point(t) for t in (Q(1, 4), Q(3, 4)))
    # These point comparisons are regressions, not the source of admission.
    for t in (Q(1, 4), Q(3, 8), Q(1, 2), Q(5, 8), Q(3, 4)):
        assert _exact_segment_distance(c.exact_point(t), first, last) <= upper


def test_direct_chart_hull_avoids_artificial_aabb_corners():
    c = curve(weights=(1., 2., 1.))
    first, last = c.pcurve.exact_point(0), c.pcurve.exact_point(1)
    bound = proof._cos_chart_chord_bound(c.support, c, 0, 1, first, last)
    expected = max(_exact_segment_distance(p, first, last) for p in c.pcurve.exact_hull(0, 1))
    old = max(_exact_segment_distance(p, first, last)
              for p in product(*proof._uv_interval(c.support, c, 0, 1)))
    assert bound == expected and bound < old


def test_pcurve_and_support_knot_crossings_keep_original_route():
    c = curve()
    kink = NurbsCurve(1, (0., 0., .5, 1., 1.), ((0., 0.), (.25, .75), (1., 1.)))
    assert proof._cos_c2_world_chord_bound(replace(c, pcurve=kink), 0, 1, 0, lambda: None) is None
    support = NurbsSurface(1, 1, (0., 0., .5, 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((.5, 0., .1), (.5, 1., .1)), ((1., 0., 0.), (1., 1., 0.))))
    assert proof._cos_c2_world_chord_bound(replace(c, support=support), 0, 1, 0, lambda: None) is None


def test_unknown_method_context_falls_back(monkeypatch):
    c = curve()
    original = NurbsCurve.exact_hull
    monkeypatch.setattr(NurbsCurve, 'exact_hull', lambda self, a, b: original(self, a, b))
    assert proof._cos_c2_world_chord_bound(c, 0, 1, 0, lambda: None) is None
    assert proof._cos_chart_chord_bound(c.support, c, 0, 1, (0, 0), (1, 1)) is None


def test_callback_exception_identity_and_mutation_are_not_fallback():
    c = curve()
    failure = RuntimeError('cancelled at caller frontier')
    def cancel():
        raise failure
    with pytest.raises(RuntimeError) as got:
        proof._cos_c2_world_chord_bound(c, 0, 1, 0, cancel)
    assert got.value is failure
    original = c.pcurve.controls
    def mutate():
        object.__setattr__(c.pcurve, 'controls', ((0., 0.), (.5, .75), (1., 1.)))
    try:
        with pytest.raises(owner.GeometryError, match='changed'):
            proof._cos_c2_world_chord_bound(c, 0, 1, 0, mutate)
    finally:
        object.__setattr__(c.pcurve, 'controls', original)


def test_world_producer_uses_sufficient_bound_but_keeps_true_refusal():
    c = curve()
    endpoints = tuple(tuple(map(float, c.exact_point(t))) for t in (0, 1))
    bound, pieces = proof._compute_world_chord(c, Q(0), Q(1), endpoints, .3, lambda: None)
    assert bound <= .3 and pieces == ((Q(0), Q(1)),)
    with pytest.raises(proof._WorldChordNeedsRefinement):
        proof._compute_world_chord(c, Q(0), Q(1), endpoints, .01, lambda: None)


def test_real_boundary_receipt_uses_true_chart_hull(monkeypatch):
    model = owner.GeometryModel()
    support = curve().support
    points = ((.75, .5), (.5, .75), (.25, .5), (.5, .25))
    corners = ((.75, .75), (.25, .75), (.25, .25), (.75, .25))
    curves = tuple(CurveOnSurface(support, NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        (points[i], corners[i], points[(i+1)%4]))) for i in range(4))
    vertices = tuple(model.add_point(*c.exact_point(0)) for c in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i+1)%4], c) for i, c in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    regions = owner.query_material_surface_regions(model, (face,))
    calls = []
    original = proof._cos_chart_chord_bound
    def observe(*args):
        value = original(*args)
        calls.append(value)
        return value
    monkeypatch.setattr(proof, '_cos_chart_chord_bound', observe)
    result = owner.query_prepared_material_boundary_chords(model, regions,
        ((model.handle('edge', edges[0]), 0., .25),), chord_error=.2)
    owner.validate_prepared_material_boundary_chords_binding(model, regions, result)
    interval = result.intervals[0]
    assert interval.world_error <= .2
    assert calls and all(value is not None for value in calls)
    assert interval.chart_error < .02


def test_work_exhaustion_is_not_an_unsupported_fallback():
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    token = CURRENT_WORK.set(Work(max_boxes=0))
    try:
        with pytest.raises(owner.GeometryError):
            proof._cos_c2_world_chord_bound(curve(), 0, 1, 0, lambda: None)
    finally:
        CURRENT_WORK.reset(token)


def test_callback_method_replacement_is_hard_refusal(monkeypatch):
    c = curve()
    original = NurbsSurface.exact_net
    def change():
        monkeypatch.setattr(NurbsSurface, 'exact_net', lambda self: original(self))
    with pytest.raises(owner.GeometryError, match='evaluator changed'):
        proof._cos_c2_world_chord_bound(c, 0, 1, 0, change)


def test_transformed_or_subclass_curve_does_not_take_specialization():
    c = curve()
    transformed = replace(c, transform=((1., 0., 0., 1.), (0., 1., 0., 0.),
                                       (0., 0., 1., 0.), (0., 0., 0., 1.)))
    assert proof._cos_c2_world_chord_bound(transformed, 0, 1, 0, lambda: None) is None
    class CustomCOS(CurveOnSurface):
        pass
    custom = CustomCOS(c.support, c.pcurve)
    assert proof._cos_c2_world_chord_bound(custom, 0, 1, 0, lambda: None) is None


def projected_curve(controls, *, scale=1, domain=(0., 1.)):
    degree = len(controls)-1
    return CurveOnSurface(curve(scale).support,
        NurbsCurve(degree, (0.,)*(degree+1)+(1.,)*(degree+1), controls, domain=domain))


@pytest.mark.parametrize('scale', (Q(1, 4), Q(1), Q(8)))
@pytest.mark.parametrize('domain', ((0., 1.), (.25, .75), (.75, .25)))
def test_projected_transverse_parabola_matches_exact_maximum(scale, domain):
    # X=(s*t, s*t*(1-t), 0): symmetric cropped endpoints have the same
    # height; their true maximum perpendicular deviation is s*h^2/4.
    c = projected_curve(((0., 0.), (.5, .5), (1., 0.)), scale=scale, domain=domain)
    endpoint_error = Q(1, 128)
    upper = proof._cos_c2_world_chord_bound(c, 1, 0, endpoint_error, lambda: None)
    expected = scale*(Q(domain[1])-Q(domain[0]))**2/4
    assert upper == expected + endpoint_error
    mid = c.exact_point(Q(1, 2))
    assert _exact_segment_distance(mid, c.exact_point(0), c.exact_point(1)) == expected


def test_nonlinear_tangential_curve_has_zero_segment_error():
    # X=(t^2, 1/2, 0): nonuniform descriptor speed changes no segment image.
    c = projected_curve(((0., .5), (0., .5), (1., .5)))
    assert proof._cos_c2_world_chord_bound(c, 0, 1, Q(1, 64), lambda: None) == Q(1, 64)


def test_collinear_overshoot_cannot_use_zero_perpendicular_acceleration():
    # X=(2*t-3*t^2/2, 1/2, 0), all controls inside chart. Endpoint x=1/2;
    # at t=1/2, x=5/8 lies strictly outside the finite endpoint segment.
    c = projected_curve(((0., .5), (1., .5), (.5, .5)))
    assert proof._cos_c2_world_chord_bound(c, 0, 1, 0, lambda: None) == Q(3, 8)
    endpoints = tuple(tuple(map(float, c.exact_point(t))) for t in (0, 1))
    with pytest.raises(proof._WorldChordNeedsRefinement):
        proof._compute_world_chord(c, Q(0), Q(1), endpoints, .01, lambda: None)


def test_interval_uncertain_projection_retains_full_taylor_bound():
    # The component intervals lose correlation: dx in[-3,3], dy in[0,3].
    # Their dot with endpoint d=(1,1) is uncertain even though the exact
    # scalar derivative has nonnegative Bernstein controls (3,0,3).
    c = projected_curve(((0., 0.), (1., 0.), (0., 1.), (1., 1.)))
    # Second derivative hull: x in[-12,12], y in[-6,6], hence L1/8=9/4.
    assert proof._cos_c2_world_chord_bound(c, 0, 1, 0, lambda: None) == Q(9, 4)


def test_degenerate_true_chord_keeps_full_remainder():
    c = projected_curve(((0., .5), (1., .5), (0., .5)))
    assert proof._cos_c2_world_chord_bound(c, 0, 1, 0, lambda: None) == Q(1, 2)


@pytest.mark.parametrize('limit', (1, 2))
def test_each_new_exact_endpoint_read_is_work_charged(limit):
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    work = Work(max_boxes=limit)
    token = CURRENT_WORK.set(work)
    try:
        with pytest.raises(owner.GeometryError):
            proof._cos_c2_world_chord_bound(curve(), 0, 1, 0, lambda: None)
        assert work.boxes == limit+1
    finally:
        CURRENT_WORK.reset(token)


@pytest.mark.parametrize('target', ('definition', 'sqrt'))
def test_final_frontier_rejects_mutation_after_projected_arithmetic(monkeypatch, target):
    c = projected_curve(((0., 0.), (.5, .5), (1., 0.)))
    original = c.pcurve.controls
    calls = []
    def callback():
        calls.append(1)
        if len(calls) == 2:
            if target == 'definition':
                object.__setattr__(c.pcurve, 'controls', ((0., 0.), (.5, .25), (1., 0.)))
            else:
                actual = proof._chord_nurbs._sqrt_up
                monkeypatch.setattr(proof._chord_nurbs, '_sqrt_up', lambda value: actual(value))
    try:
        with pytest.raises(owner.GeometryError):
            proof._cos_c2_world_chord_bound(c, 0, 1, 0, callback)
        assert len(calls) == 2
    finally:
        object.__setattr__(c.pcurve, 'controls', original)


def test_unknown_sqrt_context_uses_original_route(monkeypatch):
    actual = proof._chord_nurbs.math.isqrt
    monkeypatch.setattr(proof._chord_nurbs.math, 'isqrt', lambda value: actual(value))
    assert proof._cos_c2_world_chord_bound(curve(), 0, 1, 0, lambda: None) is None


def test_exact_rotated_transverse_reference_preserves_bound():
    c = projected_curve(((0., 0.), (.5, .5), (1., 0.)))
    # Exact rigid axis permutation/sign plus translation, stored directly in
    # the support so the COS descriptor remains canonical identity-transform.
    controls = tuple(tuple((z+2., x-4., -y+8.) for x, y, z in row)
                     for row in c.support.controls)
    rotated = replace(c, support=replace(c.support, controls=controls))
    assert proof._cos_c2_world_chord_bound(rotated, 0, 1, 0, lambda: None) == Q(1, 4)


@pytest.mark.parametrize('phase', ('entry', 'callback'))
def test_changed_interval_scale_cannot_authorize_projected_bound(monkeypatch, phase):
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    c = curve()
    interval = proof._COS_CHORD_INTERVAL
    def unsound_scale(self, factor):
        return interval(0)  # would falsely erase velocity/acceleration
    work = Work()
    token = CURRENT_WORK.set(work)
    try:
        if phase == 'entry':
            monkeypatch.setattr(interval, 'scale', unsound_scale)
            assert proof._cos_c2_world_chord_bound(c, 0, 1, 0, lambda: None) is None
        else:
            def callback():
                monkeypatch.setattr(interval, 'scale', unsound_scale)
            with pytest.raises(owner.GeometryError, match='evaluator changed'):
                proof._cos_c2_world_chord_bound(c, 0, 1, 0, callback)
        assert work.boxes == 0
    finally:
        CURRENT_WORK.reset(token)


def test_replaced_interval_module_binding_is_unknown_context(monkeypatch):
    class OtherInterval(proof._COS_CHORD_INTERVAL):
        pass
    monkeypatch.setattr(proof._chord_jets, 'I', OtherInterval)
    assert proof._cos_c2_world_chord_bound(curve(), 0, 1, 0, lambda: None) is None
