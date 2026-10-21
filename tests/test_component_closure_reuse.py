"""Public closure/aggregate contracts, independent of external oracle files."""
from dataclasses import replace

import numpy as np
import pytest
from anygeometry import GeometryModel, GeometryError, plan_independent_components, to_dict
from anygeometry.structural import AttachmentKind, ParameterRange


def linked_plates(count=6):
    model = GeometryModel()
    faces, members, attachments = [], [], []
    for index in range(count):
        x = 10.*index
        face = model.add_plate(model.add_points(((x,0,0),(x+2,0,0),(x+2,1,0),(x,1,0))))
        edge = model.faces[face].loop[0].edge
        member = model.add_member((edge,))
        attachment = model.add_attachment(member, AttachmentKind.MEMBER_ON_BOUNDARY,
            'edge', edge, ParameterRange(0,1), (ParameterRange(0,1),),
            lineage=(('face', face),))
        faces.append(face); members.append(member); attachments.append(attachment)
    for index, attachment in enumerate(attachments):
        old = model.attachments[attachment]
        model._put_structural('attachment', replace(old,
            lineage=old.lineage+(('attachment', attachments[(index+1)%count]),)))
    return model, faces, members


def shared_axis(count=4):
    model = GeometryModel()
    edge = model.add_line(*model.add_points(((0.,0.,0.), (1.,0.,0.))))
    members = [model.add_member((edge,)) for _ in range(count)]
    return model, edge, members


def test_transitive_later_lineage_closes_all_owners_without_mutation():
    model, faces, members = linked_plates()
    before = to_dict(model)
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    assert plan.components[0].face_ids == tuple(faces)
    assert plan.components[0].member_ids == tuple(members)
    assert any('attachment' in reason.reason for reason in plan.merge_reasons)
    assert to_dict(model) == before
    assert plan == plan_independent_components(model)


def test_repeated_closures_retain_selection_and_ordered_refusals():
    model, faces, members = linked_plates()
    plan = plan_independent_components(model, face_ids=(faces[0],), member_ids=())
    assert not plan.certified and not plan.components
    omitted = [item for item in plan.refusals if item.reason == 'selection omits interacting/dependent owners']
    assert len(omitted) == 1
    assert omitted[0].entities == tuple(sorted(
        [model.handle('face', face) for face in faces[1:]] +
        [model.handle('member', member) for member in members]))


@pytest.mark.parametrize('cancel_at', (1, 2, 4))
def test_each_unit_cancellation_remains_observable(cancel_at):
    model, _, _ = shared_axis()
    before = to_dict(model)
    calls = []
    def cancel(stage):
        calls.append(stage)
        return len(calls) == cancel_at
    with pytest.raises(GeometryError, match='cancelled'):
        plan_independent_components(model, cancellation_check=cancel)
    assert calls == ['component partition']*cancel_at
    assert to_dict(model) == before


def test_callback_exception_identity_and_raw_mutation_refusal():
    model, edge, _ = shared_axis()
    marker = RuntimeError('cancel marker')
    def fail(stage):
        raise marker
    with pytest.raises(RuntimeError) as caught:
        plan_independent_components(model, cancellation_check=fail)
    assert caught.value is marker
    calls = 0
    def mutate(stage):
        nonlocal calls
        calls += 1
        if calls == 3:
            point = model.vertices[model.edges[edge].start].position
            point.flags.writeable = True
            point[1] += .1
        return False
    with pytest.raises(GeometryError, match='changed during component partition'):
        plan_independent_components(model, cancellation_check=mutate)


def test_mutable_dependency_bounds_are_not_replaced_by_cached_aggregate(monkeypatch):
    model, _, members = shared_axis()
    bound = [0.,0.,0.,1.,0.,0.]
    monkeypatch.setattr(model, 'bounds', lambda keys: bound)
    calls = 0
    def mutate_bound(stage):
        nonlocal calls
        calls += 1
        if calls == 3:
            bound[0] = 2.  # Invalid only after first unit's aggregate.
        return False
    plan = plan_independent_components(model, cancellation_check=mutate_bound)
    failures = [item for item in plan.refusals if item.reason == 'missing or invalid conservative bound']
    assert [item.entities for item in failures] == [(model.handle('member', member),) for member in members[1:]]
    assert calls == len(members)+2


def test_custom_numeric_bound_conversions_remain_per_unit(monkeypatch):
    model, _, members = shared_axis()
    conversions = []
    class Number(float):
        def __float__(self):
            conversions.append(float.__float__(self))
            return float.__float__(self)
    bound = tuple(Number(v) for v in (0.,0.,0.,1.,0.,0.))
    monkeypatch.setattr(model, 'bounds', lambda keys: bound)
    plan = plan_independent_components(model)
    assert plan.certified
    assert len(conversions) == 6*len(members)


def test_immutable_dependency_boxes_preserve_signed_zero_and_call_count(monkeypatch):
    model, edge, _ = shared_axis(12)
    calls = []
    def bounds(keys):
        calls.append(tuple(keys))
        return (np.float64(-0.), 0., 0., 1., 0., 0.)
    monkeypatch.setattr(model, 'bounds', bounds)
    plan = plan_independent_components(model)
    assert plan.certified and len(plan.components) == 1
    assert calls == [(('edge', edge),)]
