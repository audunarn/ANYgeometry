"""Focused tests for the conservative invocation-local attachment box pruning.

The pruning must be a pure accelerator: identical attachment fragments with
and without it, whole-curve support certification and cancellation always
before box rejection, exact clipping retained for touching, ambiguous,
unsupported and curved-support cases, and correct behaviour on holes, thin
fragments, repeat cuts, large coordinates and support errors.
"""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import (GeometryModel, plan_intersections, apply_intersections,
                         to_dict)
from anygeometry import attachment_remapping
from anygeometry.arrangement_geometry import LinePath
from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.generators import cylinder as cylinder_generator
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.structural import AttachmentKind, ParameterRange
from anygeometry.surfaces import Plane



def plate_with_axis(axis_end=1.0, cut=2.0, offset=(0., 0., 0.), height=0.):
    model = GeometryModel()
    o = offset
    face = model.add_plate(model.add_points((
        (o[0], o[1], o[2]), (o[0]+4, o[1], o[2]),
        (o[0]+4, o[1]+4, o[2]), (o[0], o[1]+4, o[2]))))
    axis = model.add_member((model.add_line(*model.add_points(
        ((o[0], o[1]+2, o[2]+height), (o[0]+axis_end, o[1]+2, o[2]+height)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(0, axis_end/4), ParameterRange(.5, .5)),
        evidence='exact', tolerance_used=1e-8)
    cut_face = model.add_plate(model.add_points((
        (o[0]+cut, o[1], o[2]-1), (o[0]+cut, o[1]+4, o[2]-1),
        (o[0]+cut, o[1]+4, o[2]+1), (o[0]+cut, o[1], o[2]+1))))
    return model, face, axis, cut_face


def apply_cut(model, face, cut_face):
    apply_intersections(model, plan_intersections(model, (face, cut_face),
                                                  policy='connect'), policy='connect')


def attachment_state(model):
    # Intersection application also creates contact attachments with no
    # member; repr keeps the ordering total over mixed None fields.
    return sorted(((item.member_id, item.target_id, item.member_range.start,
                   item.member_range.end,
                   tuple((r.start, r.end) for r in item.target_parameters),
                   tuple(map(tuple, item.lineage)))
                  for item in model.attachments.values()), key=repr)


def document(model):
    doc = dict(to_dict(model))
    doc.pop('model_id', None)
    doc.pop('checksum', None)
    return doc


def test_pruning_is_a_pure_accelerator_of_the_exact_clipping(monkeypatch):
    model, face, axis, cut_face = plate_with_axis()
    apply_cut(model, face, cut_face)
    pruned_state = attachment_state(model)
    pruned_document = document(model)
    assert model.validate_topology() == ()

    def never_reject(*_args, **_kwargs):
        return False
    monkeypatch.setattr(attachment_remapping, '_box_rejects', never_reject)
    reference_model, reference_face, reference_axis, reference_cut = plate_with_axis()
    apply_cut(reference_model, reference_face, reference_cut)
    assert attachment_state(reference_model) == pruned_state
    assert document(reference_model) == pruned_document
    assert reference_model.validate_topology() == ()


def test_distant_descendant_is_box_pruned_and_fragments_are_exact():
    model, face, axis, cut_face = plate_with_axis()
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_bounds_tests'] == 2
    assert counters['attachment_bounds_pruned'] == 1
    assert counters['attachment_clip_calls'] == 1
    fragments = sorted((item.member_range.start, item.member_range.end)
                        for item in model.attachments.values()
                        if item.member_id == axis)
    assert fragments == [(0., 1.)]
    retained = [item for item in model.attachments.values() if item.member_id == axis]
    assert len(retained) == 1 and retained[0].target_id in model.faces
    assert model.validate_topology() == ()


def test_touching_boxes_keep_the_exact_clipping_route():
    model, face, axis, cut_face = plate_with_axis(axis_end=2.0)
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_bounds_tests'] == 2
    assert counters['attachment_bounds_pruned'] == 0
    assert counters['attachment_clip_calls'] == 2
    fragments = sorted((item.member_range.start, item.member_range.end)
                        for item in model.attachments.values()
                        if item.member_id == axis)
    assert fragments == [(0., 1.)]
    assert model.validate_topology() == ()


def test_curved_support_descendants_keep_the_exact_clipping_route():
    model = GeometryModel()
    model.insert_model(cylinder_generator(1.0, 10.0, origin=(0, 0, 0), axis=(1, 0, 0),
                                          radial_direction=(0, 1, 0),
                                          circumferential_segments=8))
    face = sorted(model.faces)[0]
    axis = model.add_member((model.add_line(*model.add_points(((0, 1, 0), (10, 1, 0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(0, 0), ParameterRange(0, 1)),
        evidence='exact', tolerance_used=1e-8)
    plate = model.add_plate(model.add_points(((0.6, -1.5, -1.5), (0.6, 1.5, -1.5),
                                               (0.6, 1.5, 1.5), (0.6, -1.5, 1.5))))
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, plate)
    # Cylinder-supported descendants have no certified planar box: no box
    # tests, and every pair is clipped exactly.
    assert counters['attachment_bounds_tests'] == 0
    assert counters['attachment_bounds_pruned'] == 0
    assert counters['attachment_clip_calls'] == 2
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert fragments[0][1] == pytest.approx(0.06, abs=1e-9)
    assert fragments == [(0., fragments[0][1]), (fragments[0][1], 1.)]
    assert model.validate_topology() == ()


def test_hole_splits_axis_by_exact_clipping_not_by_box():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    hole = model.add_polyline(model.add_points(((1.5, 1.5, 0), (2.5, 1.5, 0),
                                                (2.5, 2.5, 0), (1.5, 2.5, 0))), close=True)
    loop = tuple(OrientedEdge(edge, True) for edge in hole)
    with model.transaction():
        model._put_entity('face', replace(model.faces[face], holes=(loop,)))
    axis = model.add_member((model.add_line(*model.add_points(((0, 2, 0), (4, 2, 0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(0, 1), ParameterRange(.5, .5)),
        evidence='exact', tolerance_used=1e-8)
    cut = model.add_plate(model.add_points(((2, 0, -1), (2, 4, -1), (2, 4, 1), (2, 0, 1))))
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut)
    # The axis box overlaps both descendant boxes (the hole is interior), so
    # nothing is pruned and the exact clip splits around the hole.
    assert counters['attachment_bounds_pruned'] == 0
    assert counters['attachment_clip_calls'] == 2
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert fragments == [(0., 0.375), (0.625, 1.)]
    assert model.validate_topology() == ()


def test_large_coordinates_prune_and_retain_exactly():
    model, face, axis, cut_face = plate_with_axis(offset=(1e6, 1e6, 0.))
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_bounds_tests'] == 2
    assert counters['attachment_bounds_pruned'] == 1
    assert counters['attachment_clip_calls'] == 1
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert fragments == [(0., 1.)]
    assert model.validate_topology() == ()


def test_repeat_cuts_stay_exact_with_fresh_invocation_caches():
    model, face, axis, cut_face = plate_with_axis()
    with collect_runtime_diagnostics() as first:
        apply_cut(model, face, cut_face)
    assert first['attachment_bounds_pruned'] == 1
    children = sorted({item.target_id for item in model.attachments.values()
                       if item.member_id == axis})
    assert len(children) == 1
    left = children[0]
    second_cut = model.add_plate(model.add_points(((1.5, 0, -1), (1.5, 4, -1),
                                                   (1.5, 4, 1), (1.5, 0, 1))))
    with collect_runtime_diagnostics() as second:
        apply_cut(model, left, second_cut)
    # The axis box now misses the far grandchild: pruned again, with a
    # fresh invocation cache, and the retained fragment is exact.
    assert second['attachment_bounds_tests'] == 2
    assert second['attachment_bounds_pruned'] == 1
    assert second['attachment_clip_calls'] == 1
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert fragments == [(0., 1.)]
    assert model.validate_topology() == ()


def test_support_error_precedes_any_box_rejection(monkeypatch):
    model, face, axis, cut_face = plate_with_axis(height=1.)
    with pytest.raises(GeometryError, match='whole-curve support certificate'):
        apply_cut(model, face, cut_face)
    # Even a deliberately over-eager box rejection cannot swallow the
    # certification error: the certificate runs before the box test.
    def always_reject(*_args, **_kwargs):
        return True
    monkeypatch.setattr(attachment_remapping, '_box_rejects', always_reject)
    model, face, axis, cut_face = plate_with_axis(height=1.)
    with pytest.raises(GeometryError, match='whole-curve support certificate'):
        apply_cut(model, face, cut_face)


def test_cancellation_and_certification_precede_box_rejection(monkeypatch):
    events = []
    real_remap = attachment_remapping.remap_face_attachments
    real_support_roots = attachment_remapping._support_roots
    real_box_rejects = attachment_remapping._box_rejects
    real_clip = attachment_remapping._clip_intervals

    def support_roots(curve, support, tolerance, check_):
        events.append('certify')
        return real_support_roots(curve, support, tolerance, check_)

    def box_rejects(domain_box, curve_box, tolerance):
        events.append('box')
        return real_box_rejects(domain_box, curve_box, tolerance)

    def clip(domain, curve, tolerance, check_):
        events.append('clip')
        return real_clip(domain, curve, tolerance, check_)

    def wrapped_remap(model_, old_face, descendants, snapshots, check_):
        def check():
            events.append('check')
            check_()
        real_remap(model_, old_face, descendants, snapshots, check)

    monkeypatch.setattr(attachment_remapping, '_support_roots', support_roots)
    monkeypatch.setattr(attachment_remapping, '_box_rejects', box_rejects)
    monkeypatch.setattr(attachment_remapping, '_clip_intervals', clip)
    monkeypatch.setattr(attachment_remapping, 'remap_face_attachments', wrapped_remap)
    model, face, axis, cut_face = plate_with_axis()
    apply_cut(model, face, cut_face)
    assert model.validate_topology() == ()

    # Per (curve, descendant) pair the order is check -> certify -> box,
    # and a clip only follows a box test that did not reject.
    assert events.count('box') == 2
    assert events.count('certify') == 2
    assert events[0] == 'check'
    last_box = -1
    for index, event in enumerate(events):
        if event == 'certify':
            # A cancellation poll happened since the previous box test.
            assert 'check' in events[last_box+1:index+1]
        elif event == 'box':
            assert events[index-1] == 'certify'
            last_box = index
    assert events.count('clip') == 1
    assert events[events.index('box')+1] == 'clip'

    # A cancelling check aborts the first pair before any certification
    # or box test.
    events.clear()

    def cancelling_remap(model_, old_face, descendants, snapshots, check_):
        def check():
            events.append('check')
            raise GeometryError('cancelled')
        real_remap(model_, old_face, descendants, snapshots, check)

    monkeypatch.setattr(attachment_remapping, 'remap_face_attachments', cancelling_remap)
    model, face, axis, cut_face = plate_with_axis()
    with pytest.raises(GeometryError, match='cancelled'):
        apply_cut(model, face, cut_face)
    assert events == ['check']


def test_box_rejection_implies_exact_separation_beyond_the_margin():
    # Rejection must imply strict separation beyond the 2*tolerance margin
    # in exact arithmetic: the outward nextafter rounding never lets float
    # rounding move an expanded bound inward. Clearly separated boxes must
    # still be rejected, so the pruning stays effective.
    rng = np.random.default_rng(20261007)
    for _ in range(300):
        domain_lower = float(rng.uniform(-1e6, 1e6))
        tolerance = float(10.0 ** rng.uniform(-12, 6))
        margin = 2.*tolerance
        curve_upper = float(rng.uniform(domain_lower-margin-1.,
                                        domain_lower-margin+1.))
        domain_box = (np.array((domain_lower, -1e9, -1e9)),
                      np.array((domain_lower, 1e9, 1e9)))
        curve_box = (np.array((-1e9, -1e9, -1e9)),
                     np.array((curve_upper, 1e9, 1e9)))
        rejected = attachment_remapping._box_rejects(domain_box, curve_box,
                                                     tolerance)
        exact_bound = Fraction(domain_lower) - Fraction(margin)
        if rejected:
            assert Fraction(curve_upper) < exact_bound
        if Fraction(curve_upper) < exact_bound - 1:
            assert rejected


def test_boxes_touching_the_expanded_bounds_stay_exact():
    for domain_edge, tolerance in [(1.0, 0.5), (0.0, 0.1), (-3.5, 1e-9),
                                  (1e8, 1e-6)]:
        margin = 2.*tolerance
        domain_box = (np.array((domain_edge, -1., -1.)),
                      np.array((domain_edge, 1., 1.)))
        lower_bound = domain_edge - margin
        for curve_upper in (lower_bound, np.nextafter(lower_bound, -np.inf),
                            domain_edge):
            curve_box = (np.array((-1e9, -1., -1.)),
                         np.array((float(curve_upper), 1., 1.)))
            assert not attachment_remapping._box_rejects(domain_box, curve_box,
                                                         tolerance)
        upper_bound = domain_edge + margin
        for curve_lower in (upper_bound, np.nextafter(upper_bound, np.inf),
                            domain_edge):
            curve_box = (np.array((float(curve_lower), -1., -1.)),
                         np.array((1e9, 1., 1.)))
            assert not attachment_remapping._box_rejects(domain_box, curve_box,
                                                         tolerance)


def test_conservative_box_requires_exact_certified_curve_types():
    class _SubLinePath(LinePath):
        pass

    box = attachment_remapping._conservative_box(LinePath((0., 0., 0.),
                                                          (1., 1., 1.)))
    assert box is not None
    assert np.all(box[0] <= np.zeros(3)) and np.all(box[1] >= np.ones(3))
    assert attachment_remapping._conservative_box(
        _SubLinePath((0., 0., 0.), (1., 1., 1.))) is None


def test_conservative_box_rejects_unverified_bounds(monkeypatch):
    curve = LinePath((0., 0., 0.), (1., 1., 1.))
    for bounds in [
        (np.array((np.nan, 0., 0.)), np.array((1., 1., 1.))),
        (np.array((0., 0., 0.)), np.array((1., np.inf, 1.))),
        (np.array((2., 0., 0.)), np.array((1., 1., 1.))),
        (np.array((0., 0.)), np.array((1., 1.))),
        (np.array((0., 0., 0.)),),
        'not a pair of bounds',
    ]:
        monkeypatch.setattr(LinePath, 'bounds', lambda self, _b=bounds: _b)
        assert attachment_remapping._conservative_box(curve) is None


def test_conservative_box_swallows_expected_and_surfaces_unexpected_failures(monkeypatch):
    curve = LinePath((0., 0., 0.), (1., 1., 1.))

    def raise_geometry_error(self):
        raise GeometryError('bounds unavailable')

    def raise_type_error(self):
        raise TypeError('bounds are not a pair')

    monkeypatch.setattr(LinePath, 'bounds', raise_geometry_error)
    assert attachment_remapping._conservative_box(curve) is None
    monkeypatch.setattr(LinePath, 'bounds', raise_type_error)
    assert attachment_remapping._conservative_box(curve) is None

    def raise_memory_error(self):
        raise MemoryError('unexpected probe failure')

    monkeypatch.setattr(LinePath, 'bounds', raise_memory_error)
    with pytest.raises(MemoryError):
        attachment_remapping._conservative_box(curve)


class _StubPath:
    def __init__(self, curve):
        self.curve = curve


class _StubDomain:
    def __init__(self, support, boundaries):
        self.support = support
        self.boundaries = boundaries


def test_domain_box_requires_exact_plane_support_and_certified_boundaries():
    class _SubPlane(Plane):
        pass

    class _SubLinePath(LinePath):
        pass

    plane = Plane(np.zeros(3), np.array((1., 0., 0.)), np.array((0., 1., 0.)))
    line = LinePath((0., 0., 0.), (1., 1., 1.))
    boxes = (line.bounds(),)
    box = attachment_remapping._domain_box(_StubDomain(plane, ((_StubPath(line),),)),
                                           boxes)
    assert box is not None
    assert np.all(box[0] <= np.zeros(3)) and np.all(box[1] >= np.ones(3))
    sub_plane = _SubPlane(np.zeros(3), np.array((1., 0., 0.)),
                          np.array((0., 1., 0.)))
    assert attachment_remapping._domain_box(
        _StubDomain(sub_plane, ((_StubPath(line),),)), boxes) is None
    sub_curve = _SubLinePath((0., 0., 0.), (1., 1., 1.))
    assert attachment_remapping._domain_box(
        _StubDomain(plane, ((_StubPath(sub_curve),),)), boxes) is None
    bad = (np.array((np.nan, 0., 0.)), np.array((1., 1., 1.)))
    assert attachment_remapping._domain_box(
        _StubDomain(plane, ((_StubPath(line),),)), (bad,)) is None
    assert attachment_remapping._domain_box(_StubDomain(plane, ()), ()) is None


def _fail_full_axis_bounds(monkeypatch, error):
    # With a cut at x=0.5 every exact-route subcurve of the axis has
    # endpoints distinct from the full axis, so a bounds() failure for
    # exactly the full-axis curve can only originate in the probe.
    original = LinePath.bounds

    def failing_on_full_axis(self, lower=0., upper=1.):
        if (self.start[0] == 0. and self.start[1] == 2.
                and self.end[0] == 1. and self.end[1] == 2.):
            raise error
        return original(self, lower, upper)

    monkeypatch.setattr(LinePath, 'bounds', failing_on_full_axis)
    return original


def test_probe_failure_falls_back_to_exact_clipping_with_identical_results(monkeypatch):
    reference_model, reference_face, _axis, reference_cut = plate_with_axis(cut=0.5)
    apply_cut(reference_model, reference_face, reference_cut)
    _fail_full_axis_bounds(monkeypatch, GeometryError('axis bounds unavailable'))
    model, face, axis, cut_face = plate_with_axis(cut=0.5)
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    # The probe failure refuses a box, so every pair is clipped exactly.
    assert counters['attachment_bounds_tests'] == 0
    assert counters['attachment_bounds_pruned'] == 0
    assert counters['attachment_clip_calls'] == 2
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert fragments == [(0., 0.5), (0.5, 1.)]
    assert model.validate_topology() == ()
    assert document(model) == document(reference_model)


def test_unexpected_probe_failure_surfaces_and_leaves_the_model_unchanged(monkeypatch):
    original = _fail_full_axis_bounds(monkeypatch, MemoryError('unexpected probe failure'))
    model, face, axis, cut_face = plate_with_axis()
    before = document(model)
    with pytest.raises(MemoryError):
        apply_cut(model, face, cut_face)
    # The failed candidate is discarded: the live model is unchanged.
    assert document(model) == before
    assert model.validate_topology() == ()
    monkeypatch.setattr(LinePath, 'bounds', original)
    apply_cut(model, face, cut_face)
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert fragments == [(0., 1.)]
    assert model.validate_topology() == ()


def test_raw_geometry_change_between_cuts_is_reread_not_cached():
    model, face, axis, cut_face = plate_with_axis()
    apply_cut(model, face, cut_face)
    child = [item.target_id for item in model.attachments.values()
             if item.member_id == axis][0]
    # Raw geometry change with no topology change: move the axis endpoint
    # from x=1 to x=3 between two cuts. A stale invocation box [0, 1]
    # would prune the far grandchild of the second cut; the reread box
    # [0, 3] overlaps both grandchildren.
    endpoint = next(identifier for identifier, vertex in model.vertices.items()
                    if np.allclose(vertex.position, (1., 2., 0.)))
    with model.transaction():
        model._put_entity('vertex', replace(model.vertices[endpoint],
                                            position=np.array((3., 2., 0.))))
    second_cut = model.add_plate(model.add_points(((1.5, 0, -1), (1.5, 4, -1),
                                                   (1.5, 4, 1), (1.5, 0, 1))))
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, child, second_cut)
    assert counters['attachment_bounds_tests'] == 2
    assert counters['attachment_bounds_pruned'] == 0
    assert counters['attachment_clip_calls'] == 2
    fragments = sorted((item.member_range.start, item.member_range.end)
                       for item in model.attachments.values()
                       if item.member_id == axis)
    assert len(fragments) == 2
    assert fragments[0][0] == 0.
    assert fragments[0][1] == pytest.approx(0.5, abs=1e-12)
    assert fragments[1][0] == pytest.approx(0.5, abs=1e-12)
    assert fragments[1][1] == pytest.approx(2./3., abs=1e-12)
    assert model.validate_topology() == ()
