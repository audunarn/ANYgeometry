"""Opt-in eligibility refusal never substitutes for an independence certificate."""
import pytest
from anygeometry import GeometryModel, GeometryError, plan_independent_components, to_dict
from anygeometry import component_partition as module
from anygeometry.structural import AttachmentKind, ParameterRange


def members(*, connected=True, part=False):
    model = GeometryModel()
    part_id = model.add_part() if part else None
    first = model.add_line(*model.add_points(((0.,0.,0.), (1.,0.,0.))))
    second = first if connected else model.add_line(*model.add_points(((10.,0.,0.), (11.,0.,0.))))
    ids = [model.add_member((edge,), part_id=part_id) for edge in (first, second)]
    return model, ids


@pytest.mark.parametrize("connected", [True, False])
def test_default_and_explicit_false_keep_full_certificate_and_callbacks(connected):
    model, _ = members(connected=connected)
    before = to_dict(model)
    calls = [[], []]
    first = plan_independent_components(model, cancellation_check=lambda stage:calls[0].append(stage))
    second = plan_independent_components(model, stop_if_single_component=False,
        cancellation_check=lambda stage:calls[1].append(stage))
    assert first == second and calls[0] == calls[1]
    assert to_dict(model) == before


def test_joined_units_refuse_without_bounds_or_certificate(monkeypatch):
    model, _ = members()
    before = to_dict(model)
    def forbidden(*args):
        raise AssertionError("eligibility refusal must precede bounds")
    monkeypatch.setattr(model, "bounds", forbidden)
    calls = []
    result = plan_independent_components(model, stop_if_single_component=True,
        cancellation_check=lambda stage:calls.append(stage))
    assert not result.certified and result.components == ()
    assert "without full independence certification" in result.refusals[-1].reason
    assert result.merge_reasons and len(calls) == 2
    assert result.model_id == model.model_id and result.revision == model.revision
    assert result.source_checksum == before["checksum"]["value"]
    assert to_dict(model) == before


def test_part_membership_alone_does_not_refuse_independent_units():
    model, _ = members(connected=False, part=True)
    default = plan_independent_components(model)
    probe = plan_independent_components(model, stop_if_single_component=True)
    assert probe == default and probe.certified and len(probe.components) == 2


def test_attachment_join_can_refuse_before_claim_closure(monkeypatch):
    model, ids = members(connected=False)
    model.add_attachment(ids[0], AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER,
        "member", ids[1], ParameterRange.point(1.), (ParameterRange.point(0.),))
    def forbidden(*args):
        raise AssertionError("relation join should precede bounds")
    monkeypatch.setattr(model, "bounds", forbidden)
    probe = plan_independent_components(model, stop_if_single_component=True)
    assert not probe.certified and not probe.components
    assert probe.merge_reasons[-1].reason == "declared attachment (all connection intents)"


def test_unknown_extent_is_declined_not_certified(monkeypatch):
    model = GeometryModel()
    points = model.add_points(((0.,0.,0.),(3.,0.,0.),(2.,1.,0.),(0.,1.,0.)))
    model.add_face(model.add_polyline(points, close=True), corners=(0,1,2,3), surface=None)
    default = plan_independent_components(model)
    assert not default.certified
    monkeypatch.setattr(module, "_certified_face_bound", lambda *args:pytest.fail("no extent audit"))
    probe = plan_independent_components(model, stop_if_single_component=True)
    assert not probe.certified and not probe.components
    assert "without full independence certification" in probe.refusals[-1].reason


@pytest.mark.parametrize("action", ["cancel", "exception", "raw_mutation", "revision"])
def test_early_return_callback_preserves_cancellation_and_mutation_refusal(action):
    model, _ = members()
    marker = RuntimeError("early callback marker")
    calls = []
    def callback(stage):
        calls.append(stage)
        if len(calls) != 2:
            return False
        if action == "cancel":
            return True
        if action == "exception":
            raise marker
        if action == "raw_mutation":
            point = next(iter(model.vertices.values())).position
            point.flags.writeable = True
            point[1] += .25
        else:
            model.add_point(20.,20.,20.)
        return False
    with pytest.raises((RuntimeError, GeometryError)) as caught:
        plan_independent_components(model, stop_if_single_component=True,
                                    cancellation_check=callback)
    if action == "exception":
        assert caught.value is marker
    else:
        assert ("cancelled" if action == "cancel" else "changed during") in str(caught.value)
    assert len(calls) == 2


def test_empty_selection_keeps_full_audit_and_invalid_flag_refuses():
    model, _ = members()
    options = dict(face_ids=(), member_ids=(), edge_ids=())
    assert plan_independent_components(model, **options) == plan_independent_components(
        model, stop_if_single_component=True, **options)
    with pytest.raises(GeometryError, match="must be Boolean"):
        plan_independent_components(model, stop_if_single_component=1)
