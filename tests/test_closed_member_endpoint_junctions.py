"""Closed source seams retain both exact parent-parameter occurrences."""
import pytest
from anygeometry import GeometryModel
from anygeometry.member_arrangements import MemberPointContact
from anygeometry.member_joints import declare_member_contacts
from anygeometry.structural import AttachmentKind


def fixture(closed=True):
    model = GeometryModel()
    origin, x, y, low, high = model.add_points(((0.,0.,0.), (1.,0.,0.), (0.,1.,0.),
                                               (0.,0.,-1.), (0.,0.,1.)))
    part = model.add_part(name='junction')
    target_edge = model.add_line(low, high)
    target = model.add_member((target_edge,), part_id=part)
    edges = (model.add_line(origin,x), model.add_line(x,y), model.add_line(y,origin)) if closed else (model.add_line(origin,x),)
    source = model.add_member(edges, part_id=part)
    parameters = ((target,.5), (source,0.), (source,1.)) if closed else ((target,.5), (source,0.))
    contact = MemberPointContact(None,(0.,0.,0.),parameters,(),1e-10)
    return model, origin, target, source, contact


@pytest.mark.parametrize('closed', [False, True])
def test_exact_closed_endpoint_occurrences_have_complete_attachment_evidence(closed):
    model, vertex, target, source, contact = fixture(closed)
    with model.transaction():
        declare_member_contacts(model, ((contact,vertex),), lambda: None)
    assert model.validate_topology() == ()
    assert model._validate_structural() == ()
    attachments = tuple(model.attachments.values())
    assert len(attachments) == (2 if closed else 1)
    assert all(a.kind is AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER for a in attachments)
    assert all(a.member_id == source and a.target_id == target for a in attachments)
    assert {(a.member_range.start,a.member_range.end) for a in attachments} == ({(0.,0.),(1.,1.)} if closed else {(0.,0.)})
    assert all([(p.start,p.end) for p in a.target_parameters] == [(.5,.5)] for a in attachments)
    assert all(a.max_residual == 0. and a.tolerance_used == 1e-10 for a in attachments)
    before = repr(model.topology_snapshot())
    with model.transaction():
        declare_member_contacts(model, ((contact,vertex),), lambda: None)
    assert repr(model.topology_snapshot()) == before


def test_endpoint_declaration_cancellation_preserves_atomic_caller_state():
    model, vertex, _target, _source, contact = fixture()
    before = repr(model.topology_snapshot()), model.revision
    marker = RuntimeError('cancel contact declaration')
    def cancel():
        raise marker
    with pytest.raises(RuntimeError) as caught:
        with model.transaction():
            declare_member_contacts(model, ((contact,vertex),), cancel)
    assert caught.value is marker
    assert (repr(model.topology_snapshot()), model.revision) == before


def test_ordinary_end_to_end_retains_one_deterministic_attachment():
    model = GeometryModel()
    origin, x, y = model.add_points(((0.,0.,0.), (1.,0.,0.), (0.,1.,0.)))
    part = model.add_part(name='end-to-end')
    first = model.add_member((model.add_line(origin,x),), part_id=part)
    second = model.add_member((model.add_line(origin,y),), part_id=part)
    contact = MemberPointContact(None,(0.,0.,0.),((first,0.),(second,0.)),(),1e-10)
    with model.transaction():
        declare_member_contacts(model, ((contact,origin),), lambda: None)
    assert model._validate_structural() == ()
    assert len(model.attachments) == 1
    attachment = next(iter(model.attachments.values()))
    assert attachment.member_id == first and attachment.target_id == second
    assert (attachment.member_range.start, attachment.member_range.end) == (0.,0.)
    assert [(p.start,p.end) for p in attachment.target_parameters] == [(0.,0.)]
