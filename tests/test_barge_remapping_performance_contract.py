"""Performance-contract tests for the barge attachment remapping broad phase.

The measured 60 m baseline profile (evidence/baseline-profile-03) shows
``remap_face_attachments`` spending 10.952 s in 7456 point-attachment
``contains`` probes that bypassed the conservative box broad phase curve
attachments already have, plus 5.58 s rebuilding the immutable
``ExactPlane`` chart per query. These tests pin the two compatible
optimizations to their contracts:

- point-attachment box rejection is a pure accelerator of the exact
  containment route: rejection beyond the 2*tolerance margin certifies the
  exact probe would fail; touching, uncertified, nonfinite, curved-support
  and hole cases keep the exact route, and cancellation still precedes
  every rejection;
- the per-domain exact chart is reused, never shared across different
  supports, and never participates in equality, hashing or repr.
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
from anygeometry.material_arrangement import MaterialDomain, ArrangementPath
from anygeometry.planar_exact import ExactPlane
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.structural import AttachmentKind, ParameterRange
from anygeometry.surfaces import Plane


def plate_with_point(u=0.25, v=0.5, cut=2.0, offset=(0., 0., 0.)):
    model = GeometryModel()
    o = offset
    face = model.add_plate(model.add_points((
        (o[0], o[1], o[2]), (o[0]+4, o[1], o[2]),
        (o[0]+4, o[1]+4, o[2]), (o[0], o[1]+4, o[2]))))
    axis = model.add_member((model.add_line(*model.add_points(
        ((o[0], o[1]+2, o[2]), (o[0]+1, o[1]+2, o[2])))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(u, u), ParameterRange(v, v)),
        evidence='exact', tolerance_used=1e-8)
    cut_face = model.add_plate(model.add_points((
        (o[0]+cut, o[1], o[2]-1), (o[0]+cut, o[1]+4, o[2]-1),
        (o[0]+cut, o[1]+4, o[2]+1), (o[0]+cut, o[1], o[2]+1))))
    return model, face, axis, cut_face


def apply_cut(model, face, cut_face):
    apply_intersections(model, plan_intersections(model, (face, cut_face),
                                                  policy='connect'), policy='connect')


def attachment_state(model):
    return sorted(((item.member_id, item.target_id,
                    item.member_range.start, item.member_range.end,
                    tuple((r.start, r.end) for r in item.target_parameters),
                    tuple(map(tuple, item.lineage)))
                   for item in model.attachments.values()), key=repr)


def document(model):
    doc = dict(to_dict(model))
    doc.pop('model_id', None)
    doc.pop('checksum', None)
    return doc


def point_fragments(model, axis):
    return sorted(((item.target_id,
                     tuple((r.start, r.end) for r in item.target_parameters))
                    for item in model.attachments.values() if item.member_id == axis),
                   key=repr)


# --------------------------------------------------------------- point broad phase

def test_point_pruning_is_a_pure_accelerator_of_the_exact_route(monkeypatch):
    model, face, axis, cut_face = plate_with_point()
    apply_cut(model, face, cut_face)
    pruned_state = attachment_state(model)
    pruned_document = document(model)
    assert model.validate_topology() == ()

    def never_reject(*_args, **_kwargs):
        return False
    monkeypatch.setattr(attachment_remapping, '_box_rejects', never_reject)
    reference_model, reference_face, reference_axis, reference_cut = plate_with_point()
    apply_cut(reference_model, reference_face, reference_cut)
    assert attachment_state(reference_model) == pruned_state
    assert document(reference_model) == pruned_document
    assert reference_model.validate_topology() == ()


def test_distant_descendant_point_is_box_pruned_and_remapped_exactly():
    model, face, axis, cut_face = plate_with_point(u=0.25)
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_point_bounds_tests'] == 2
    assert counters['attachment_point_bounds_pruned'] == 1
    # The existing curve counters are untouched by the point broad phase.
    assert counters['attachment_bounds_tests'] == 0
    assert counters['attachment_bounds_pruned'] == 0
    assert counters['attachment_clip_calls'] == 0
    fragments = point_fragments(model, axis)
    assert len(fragments) == 1
    target_id, parameters = fragments[0]
    assert target_id in model.faces
    # The retained fragment keeps the exact descendant chart point.
    assert parameters == ((0.5, 0.5), (0.5, 0.5))
    assert model.validate_topology() == ()


def test_touching_point_keeps_the_exact_route_on_both_descendants():
    model, face, axis, cut_face = plate_with_point(u=0.5)
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_point_bounds_tests'] == 2
    assert counters.get('attachment_point_bounds_pruned', 0) == 0
    fragments = point_fragments(model, axis)
    # A point exactly on the shared cut edge is retained by both exact
    # descendants: the broad phase must not decide a touching case.
    assert sorted(parameters for _target, parameters in fragments) == \
        [((0.0, 0.0), (0.5, 0.5)), ((1.0, 1.0), (0.5, 0.5))]
    assert model.validate_topology() == ()


def test_large_coordinates_prune_and_retain_exactly():
    model, face, axis, cut_face = plate_with_point(offset=(1e6, 1e6, 0.))
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_point_bounds_tests'] == 2
    assert counters['attachment_point_bounds_pruned'] == 1
    fragments = point_fragments(model, axis)
    assert len(fragments) == 1
    assert fragments[0][1] == ((0.5, 0.5), (0.5, 0.5))
    assert model.validate_topology() == ()


def test_curved_support_point_keeps_the_exact_route():
    model = GeometryModel()
    model.insert_model(cylinder_generator(1.0, 10.0, origin=(0, 0, 0), axis=(1, 0, 0),
                                          radial_direction=(0, 1, 0),
                                          circumferential_segments=8))
    face = sorted(model.faces)[0]
    axis = model.add_member((model.add_line(*model.add_points(((0, 1, 0), (10, 1, 0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(0, 0), ParameterRange(0, 0)),
        evidence='exact', tolerance_used=1e-8)
    plate = model.add_plate(model.add_points(((0.6, -1.5, -1.5), (0.6, 1.5, -1.5),
                                              (0.6, 1.5, 1.5), (0.6, -1.5, 1.5))))
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, plate)
    # Cylinder-supported descendants have no certified planar box: no point
    # box tests, and every pair keeps the exact containment route.
    assert counters.get('attachment_point_bounds_tests', 0) == 0
    assert counters.get('attachment_point_bounds_pruned', 0) == 0
    fragments = point_fragments(model, axis)
    assert len(fragments) == 1
    assert fragments[0][1] == ((0.0, 0.0), (0.0, 0.0))
    assert model.validate_topology() == ()


def test_point_in_hole_keeps_the_exact_route_and_fails_closed():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    hole = model.add_polyline(model.add_points(((1.5, 1.5, 0), (2.5, 1.5, 0),
                                                 (2.5, 2.5, 0), (1.5, 2.5, 0))), close=True)
    loop = tuple(OrientedEdge(edge, True) for edge in hole)
    with model.transaction():
        model._put_entity('face', replace(model.faces[face], holes=(loop,)))
    axis = model.add_member((model.add_line(*model.add_points(((0, 2, 0), (1, 2, 0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(.5, .5), ParameterRange(.5, .5)),
        evidence='exact', tolerance_used=1e-8)
    cut = model.add_plate(model.add_points(((2, 0, -1), (2, 4, -1), (2, 4, 1), (2, 0, 1))))
    # The hole is interior, so both descendant boxes overlap the point box:
    # nothing is pruned and the exact route reports the lost descendant.
    with collect_runtime_diagnostics() as counters:
        with pytest.raises(GeometryError, match='lost all material descendants'):
            apply_cut(model, face, cut)
    assert counters['attachment_point_bounds_tests'] == 2
    assert counters.get('attachment_point_bounds_pruned', 0) == 0


def test_point_rejection_can_only_remove_never_invent(monkeypatch):
    def always_reject(*_args, **_kwargs):
        return True
    monkeypatch.setattr(attachment_remapping, '_box_rejects', always_reject)
    model, face, axis, cut_face = plate_with_point()
    with pytest.raises(GeometryError, match='lost all material descendants'):
        apply_cut(model, face, cut_face)


def test_cancellation_precedes_any_point_box_rejection(monkeypatch):
    events = []
    real_remap = attachment_remapping.remap_face_attachments
    real_box_rejects = attachment_remapping._box_rejects

    def box_rejects(domain_box, curve_box, tolerance):
        events.append('box')
        return real_box_rejects(domain_box, curve_box, tolerance)

    def recording_remap(model_, old_face, descendants, snapshots, check_):
        def check():
            events.append('check')
            check_()
        real_remap(model_, old_face, descendants, snapshots, check)

    monkeypatch.setattr(attachment_remapping, '_box_rejects', box_rejects)
    monkeypatch.setattr(attachment_remapping, 'remap_face_attachments', recording_remap)
    model, face, axis, cut_face = plate_with_point()
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert model.validate_topology() == ()
    # Every point box test follows a cancellation poll of the same pair.
    assert events.count('box') == 2
    assert events[0] == 'check'
    for index, event in enumerate(events):
        if event == 'box':
            assert 'check' in events[:index]

    events.clear()
    monkeypatch.setattr(attachment_remapping, 'remap_face_attachments', real_remap)

    def cancelling_remap(model_, old_face, descendants, snapshots, check_):
        def check():
            events.append('check')
            raise GeometryError('cancelled')
        real_remap(model_, old_face, descendants, snapshots, check)

    monkeypatch.setattr(attachment_remapping, 'remap_face_attachments', cancelling_remap)
    model, face, axis, cut_face = plate_with_point()
    with collect_runtime_diagnostics() as counters:
        with pytest.raises(GeometryError, match='cancelled'):
            apply_cut(model, face, cut_face)
    assert events == ['check']
    assert counters.get('attachment_point_bounds_tests', 0) == 0


def test_nonfinite_or_malformed_point_has_no_certified_box():
    for point in ((float('nan'), 0., 0.), (float('inf'), 0., 0.),
                  (0., float('nan'), 0.), (0., 0.), 'not a point'):
        assert attachment_remapping._certified_box((point, point)) is None


def test_point_box_rejection_follows_the_expanded_margin_contract():
    # A point box is degenerate (lower == upper), so the same _box_rejects
    # margin rules decide it: touching the expanded bound stays exact and
    # clearly separated points are rejected.
    for domain_edge, tolerance in [(1.0, 0.5), (0.0, 0.1), (-3.5, 1e-9),
                                   (1e8, 1e-6)]:
        margin = 2.*tolerance
        domain_box = (np.array((domain_edge, -1., -1.)),
                      np.array((domain_edge, 1., 1.)))
        for point in (domain_edge - margin,
                      np.nextafter(domain_edge - margin, -np.inf),
                      domain_edge - margin - 1.):
            point_box = (np.array((float(point), -1., -1.)),
                         np.array((float(point), 1., 1.)))
            expected = float(point) < domain_edge - margin - .5
            assert attachment_remapping._box_rejects(
                domain_box, point_box, tolerance) is expected
        for point in (domain_edge + margin,
                      np.nextafter(domain_edge + margin, np.inf),
                      domain_edge + margin + 1.):
            point_box = (np.array((float(point), -1., -1.)),
                         np.array((float(point), 1., 1.)))
            expected = float(point) > domain_edge + margin + .5
            assert attachment_remapping._box_rejects(
                domain_box, point_box, tolerance) is expected


# --------------------------------------------------------------- exact chart reuse

def square_domain(origin=(0., 0., 0.), holes=()):
    support = Plane(np.asarray(origin), np.array((1., 0., 0.)), np.array((0., 1., 0.)))
    o = origin
    corners = ((o[0], o[1], o[2]), (o[0]+4, o[1], o[2]),
               (o[0]+4, o[1]+4, o[2]), (o[0], o[1]+4, o[2]))
    outer = tuple(ArrangementPath(LinePath(a, b))
                  for a, b in zip(corners, corners[1:] + corners[:1]))
    boundaries = (outer, *holes)
    return MaterialDomain(0, support, boundaries)


def test_exact_chart_is_built_once_and_reused_per_domain(monkeypatch):
    domain = square_domain()
    calls = []
    real = ExactPlane.from_plane
    monkeypatch.setattr(ExactPlane, 'from_plane',
                        classmethod(lambda cls, plane: (calls.append(plane),
                                                        real(plane))[1]))
    first = domain._exact_chart()
    second = domain._exact_chart()
    assert first is second
    assert len(calls) == 1
    # A domain copied with a different support never reuses the stale chart.
    other = replace(domain, support=Plane(np.array((5., 0., 0.)),
                                          np.array((1., 0., 0.)), np.array((0., 1., 0.))))
    chart = other._exact_chart()
    assert chart is not first
    assert len(calls) == 2
    assert calls[1] is other.support


@pytest.mark.parametrize('u_vector,v_vector', [((4., 0., 0.), (0., 4., 0.)),
                                             ((4., 0., 4.), (0., 4., 0.))])
def test_point_box_preserves_normal_offset_projection(u_vector, v_vector):
    support = Plane((2., -3., 5.), u_vector, v_vector)
    corners = tuple(tuple(support.evaluate(u, v))
                    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)))
    loop = tuple(ArrangementPath(LinePath(a, b))
                 for a, b in zip(corners, corners[1:] + corners[:1]))
    domain = MaterialDomain(1, support, (loop,))
    boxes = tuple(path.curve.bounds() for path in loop)
    domain_box = attachment_remapping._domain_box(domain, boxes)
    for u, expected in ((.25, True), (2., False)):
        point = tuple(support.evaluate(u, .5) + 17. * support.normal)
        probe = LinePath(point, point)
        assert domain.contains(probe, 0., 1e-8) is expected
        projected_box = attachment_remapping._projected_point_box(domain, point)
        assert projected_box is not None
        assert attachment_remapping._box_rejects(domain_box, projected_box, 1e-8) is not expected
        chart = ExactPlane.from_plane(support)
        exact_point = chart.world(*chart.uv(tuple(Fraction(value) for value in point)))
        assert all(Fraction(float(lo)) <= value <= Fraction(float(hi))
                   for lo, value, hi in zip(projected_box[0], exact_point, projected_box[1]))


def test_plane_subclasses_keep_uncached_chart_queries():
    class DerivedPlane(Plane):
        pass
    base = square_domain()
    domain = replace(base, support=DerivedPlane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    assert domain._exact_chart() is not domain._exact_chart()
    assert domain._exact_chart_cache is None
    assert attachment_remapping._projected_point_box(domain, (.5, .5, 0.)) is None


def test_admitted_trim_normal_residual_keeps_point_attachment():
    model = GeometryModel()
    points = model.add_points(((0., 0., 5e-9), (4., 0., 5e-9),
                               (4., 4., 5e-9), (0., 4., 5e-9)))
    edges = model.add_lines(tuple(zip(points, points[1:] + points[:1])))
    face = model.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in edges), (0, 1, 2, 3),
        surface=Plane((0., 0., 0.), (4., 0., 0.), (0., 4., 0.)),
    )
    point = model.add_point(2., 2., 0.)
    identifier = model.add_attachment(
        None, AttachmentKind.VERTEX_ON_FACE, 'face', face,
        ParameterRange.point(0.), (ParameterRange.point(.5), ParameterRange.point(.5)),
        source_kind='vertex', source_id=point, evidence='exact', tolerance_used=1e-8,
    )
    assert not model.validate_topology()
    old_face = model.faces[face]
    snapshots = attachment_remapping.capture_face_attachments(model, face, lambda: None)
    with model.transaction():
        attachment_remapping.remap_face_attachments(
            model, old_face, (face,), snapshots, lambda: None,
        )
    assert model.attachments[identifier].target_id == face
    assert model.attachments[identifier].target_parameters == (
        ParameterRange.point(.5), ParameterRange.point(.5))
    domain = MaterialDomain.from_model(model, face)
    bounds = attachment_remapping._projected_domain_box(domain)
    projected = attachment_remapping._projected_point_box(domain, (2., 2., 0.))
    assert not attachment_remapping._box_rejects(bounds, projected, 1e-9)


def test_chart_cache_stays_out_of_equality_hash_and_repr():
    domain = square_domain()
    assert domain._exact_chart_cache is None
    domain._exact_chart()
    assert domain._exact_chart_cache is not None
    copy = replace(domain)
    assert copy == domain
    assert '_exact_chart_cache' not in repr(domain)
    # Derived cache state is excluded from construction; the copy computes
    # an equal immutable chart in its own bounded cache.
    assert copy._exact_chart() == domain._exact_chart()


def test_exact_containment_is_unchanged_by_chart_reuse():
    corners = ((1.5, 1.5, 0.), (2.5, 1.5, 0.),
               (2.5, 2.5, 0.), (1.5, 2.5, 0.))
    hole = tuple(ArrangementPath(LinePath(a, b))
                 for a, b in zip(corners, corners[1:] + corners[:1]))
    domain = square_domain(holes=(hole,))
    material = LinePath((0.5, 0.5, 0.), (0.5, 0.5, 0.))
    in_hole = LinePath((2., 2., 0.), (2., 2., 0.))
    outside = LinePath((5., 5., 0.), (5., 5., 0.))
    assert domain.contains(material, 0., 1e-8)
    assert not domain.contains(in_hole, 0., 1e-8)
    assert not domain.contains(outside, 0., 1e-8)
    # Repeated queries reuse the chart and keep the exact answers.
    assert domain.contains(material, 0., 1e-8)
    assert not domain.contains(in_hole, 0., 1e-8)
