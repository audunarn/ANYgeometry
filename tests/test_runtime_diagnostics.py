"""Focused tests for the opt-in runtime diagnostics collector contract.

``collect_runtime_diagnostics`` must yield a fresh zeroed counter dict, be
ContextVar-scoped, let nested collectors independently receive the increments
for work within their scope, reset safely on exceptions, stay a no-op when no
collector is active, and never change geometry decisions or serialized output.
"""
import pytest

from anygeometry import GeometryModel, plan_intersections, apply_intersections, to_dict
from anygeometry.runtime_diagnostics import (COUNTER_KEYS,
                                              collect_runtime_diagnostics,
                                              _increment)
from anygeometry.structural import AttachmentKind, ParameterRange


def plate_with_attachments(axis_end=1.0, cut=2.0):
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    axis = model.add_member((model.add_line(*model.add_points(
        ((0, 2, 0), (axis_end, 2, 0)))),))
    model.add_attachment(axis, AttachmentKind.MEMBER_ON_FACE, 'face', face,
        ParameterRange(0, 1), (ParameterRange(0, axis_end/4), ParameterRange(.5, .5)),
        evidence='exact', tolerance_used=1e-8)
    point = model.add_member((model.add_line(*model.add_points(
        ((0.5, 1, -1), (0.5, 1, 1)))),))
    model.add_attachment(point, AttachmentKind.MEMBER_THROUGH_FACE, 'face', face,
        ParameterRange(.5, .5), (ParameterRange(.125, .125), ParameterRange(.25, .25)),
        evidence='exact', tolerance_used=1e-8)
    cut_face = model.add_plate(model.add_points(((cut, 0, -1), (cut, 4, -1),
                                                 (cut, 4, 1), (cut, 0, 1))))
    return model, face, axis, point, cut_face


def apply_cut(model, face, cut_face):
    apply_intersections(model, plan_intersections(model, (face, cut_face),
                                                  policy='connect'), policy='connect')


def test_collector_yields_a_fresh_zeroed_dict():
    with collect_runtime_diagnostics() as first:
        assert first == {key: 0 for key in COUNTER_KEYS}
    with collect_runtime_diagnostics() as second:
        assert second == {key: 0 for key in COUNTER_KEYS}
        assert second is not first


def test_increment_without_collector_is_a_silent_noop():
    _increment('attachment_clip_calls')
    _increment('attachment_bounds_tests', 5)
    with collect_runtime_diagnostics() as counters:
        assert counters == {key: 0 for key in COUNTER_KEYS}


def test_remap_counters_report_clips_tests_and_prunes():
    model, face, axis, point, cut_face = plate_with_attachments()
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    # The axis curve is box-tested against both descendants, pruned against
    # the distant one, and exactly clipped once; the point attachment has no
    # curve and never reaches the box or clip paths.
    assert counters['attachment_bounds_tests'] == 2
    assert counters['attachment_bounds_pruned'] == 1
    assert counters['attachment_clip_calls'] == 1
    assert model.validate_topology() == ()


def test_nested_collectors_independently_receive_scope_increments():
    with collect_runtime_diagnostics() as outer:
        _increment('attachment_clip_calls')
        with collect_runtime_diagnostics() as inner:
            _increment('attachment_clip_calls')
            _increment('attachment_bounds_tests')
        _increment('attachment_bounds_pruned')
    assert inner == {'attachment_clip_calls': 1, 'attachment_bounds_tests': 1,
                     'attachment_bounds_pruned': 0}
    assert outer == {'attachment_clip_calls': 2, 'attachment_bounds_tests': 1,
                    'attachment_bounds_pruned': 1}


def test_exception_resets_the_collector_scope_safely():
    with collect_runtime_diagnostics() as outer:
        with pytest.raises(RuntimeError, match='boom'):
            with collect_runtime_diagnostics() as inner:
                _increment('attachment_clip_calls')
                raise RuntimeError('boom')
        _increment('attachment_bounds_tests')
        with collect_runtime_diagnostics() as after:
            _increment('attachment_bounds_pruned')
    # Work before the failure is retained; the broken scope never leaks.
    assert inner['attachment_clip_calls'] == 1
    # The post-failure collector is nested in the outer scope, so the
    # outer collector also receives its increments.
    assert outer == {'attachment_clip_calls': 1, 'attachment_bounds_tests': 1,
                     'attachment_bounds_pruned': 1}
    assert after == {'attachment_clip_calls': 0, 'attachment_bounds_tests': 0,
                     'attachment_bounds_pruned': 1}


def test_counters_never_change_serialized_geometry():
    reference_model, face, axis, point, cut_face = plate_with_attachments()
    apply_cut(reference_model, face, cut_face)
    reference = to_dict(reference_model)

    model, face, axis, point, cut_face = plate_with_attachments()
    with collect_runtime_diagnostics() as counters:
        apply_cut(model, face, cut_face)
    assert counters['attachment_clip_calls'] > 0
    counted = to_dict(model)
    for document in (reference, counted):
        document.pop('model_id', None)
        document.pop('checksum', None)
    assert counted == reference
