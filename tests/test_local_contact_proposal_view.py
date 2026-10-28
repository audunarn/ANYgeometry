"""Real issued source contact proofs; numerical views never issue authority."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
import anygeometry._local_contact_stars as contact
from test_local_contact_stars import source_stars


def test_contact_view_matches_public_both_parents_and_whole_boxes(source_stars):
    source,faces,receipt=source_stars
    view=contact._prepare_local_contact_proposal_query(source,receipt)
    boxes=[((Q(-1),Q(2)),(Q(-1),Q(2)))]
    for star in (receipt.stars[0],receipt.stars[-1]):
        boxes.extend((star.inner_box,tuple((a,(a+b)/2) for a,b in star.inner_box),star.outer_box))
    for face in faces:
        for box in boxes:
            assert view.query(face,box)==contact.query_local_contact_proposal(source,receipt,face,box)


def test_contact_view_rejects_custom_operands_without_conversion(source_stars):
    source,faces,receipt=source_stars
    view=contact._prepare_local_contact_proposal_query(source,receipt)
    class Evil:
        def __iter__(self):raise AssertionError('custom iteration')
        def __int__(self):raise AssertionError('custom conversion')
    class Int(int):pass
    for face,box in ((faces[0],Evil()),(Evil(),receipt.stars[0].inner_box),
                     (Int(faces[0]),receipt.stars[0].inner_box),(faces[0],((0,1),(0,1)))):
        with pytest.raises(GeometryError,match='canonical'):view.query(face,box)


def test_contact_view_issuer_and_extraction_guards(source_stars,monkeypatch):
    source,_,receipt=source_stars
    with pytest.raises(GeometryError):contact._prepare_local_contact_proposal_query(source,replace(receipt))
    original=contact._ContactProposalView;old=receipt.intent
    def mutate(*args):
        result=original(*args)
        object.__setattr__(receipt,'intent','IMPRINT')
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(contact,'_ContactProposalView',mutate)
            with pytest.raises(GeometryError,match='changed'):
                contact._prepare_local_contact_proposal_query(source,receipt)
    finally:object.__setattr__(receipt,'intent',old)


def test_contact_view_only_validates_at_guard_frontiers(source_stars,monkeypatch):
    source,faces,receipt=source_stars;calls=[]
    original=contact.validate_local_contact_stars
    def observe(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(contact,'validate_local_contact_stars',observe)
    registry=contact._ContactProposalRegistry(source);view=registry.prepare(receipt)
    prepared_count=len(calls)
    for _ in range(20):view.query(faces[0],receipt.stars[0].inner_box)
    assert len(calls)==prepared_count
    events=[];sentinel=object()
    def callback(*args,**kwargs):events.append((args,kwargs));return sentinel
    assert registry.wrap_callback(None) is None
    assert registry.wrap_callback(callback)('phase',key=3) is sentinel
    assert events==[(('phase',),{'key':3})] and len(calls)==prepared_count+2
    registry.guard()
    assert len(calls)==prepared_count+3


@pytest.mark.parametrize('operand',('receipt','source','runtime'))
def test_contact_registry_checks_actual_callback_mutation(source_stars,monkeypatch,operand):
    source,faces,receipt=source_stars
    registry=contact._ContactProposalRegistry(source);registry.prepare(receipt)
    target,name=(receipt,'intent') if operand=='receipt' else (source.faces[faces[0]].surface,'controls')
    old=getattr(target,name);calls=[]
    def callback(phase):
        calls.append(phase)
        if operand=='runtime':monkeypatch.setattr(contact.NurbsSurface,'exact_point',lambda *a:None)
        else:object.__setattr__(target,name,'IMPRINT' if operand=='receipt' else
            tuple(tuple((x,y,z+1) for x,y,z in row) for row in old))
    try:
        with pytest.raises(GeometryError,match='changed'):registry.wrap_callback(callback)('same phase')
        assert calls==['same phase']
    finally:
        if operand!='runtime':object.__setattr__(target,name,old)


def test_contact_registry_final_guard_detects_receipt_only_change(source_stars):
    source,faces,receipt=source_stars
    registry=contact._ContactProposalRegistry(source);view=registry.prepare(receipt)
    old=receipt.intent;box=receipt.stars[0].inner_box
    try:
        object.__setattr__(receipt,'intent','IMPRINT')
        # Detached planning is permitted until the adopted next callback/final gate.
        assert view.query(faces[0],box).requires_realization
        with pytest.raises(GeometryError,match='changed'):registry.guard()
        called=[]
        with pytest.raises(GeometryError):registry.wrap_callback(lambda phase:called.append(phase))('late')
        assert not called
    finally:object.__setattr__(receipt,'intent',old)


def test_contact_registry_cancellation_return_and_exception_identity(source_stars):
    source,_,receipt=source_stars
    registry=contact._ContactProposalRegistry(source);registry.prepare(receipt)
    calls=[]
    def cancel(phase):calls.append(phase);return True
    assert registry.wrap_callback(cancel)('cancel') is True
    assert calls==['cancel']
    error=RuntimeError('original exception')
    def fail(phase):raise error
    with pytest.raises(RuntimeError) as caught:registry.wrap_callback(fail)('error')
    assert caught.value is error


@pytest.mark.parametrize('target',('query','guard','guard_code','constructor','class','defaults'))
def test_contact_view_callback_runtime_replacement_refuses(source_stars,monkeypatch,target):
    source,_,receipt=source_stars
    registry=contact._ContactProposalRegistry(source);registry.prepare(receipt)
    def mutate(phase):
        if target=='query':monkeypatch.setattr(contact._ContactProposalView,'query',lambda *args:None)
        elif target=='guard':monkeypatch.setattr(contact._ContactProposalRegistry,'guard',lambda self:None)
        elif target=='guard_code':monkeypatch.setattr(contact._ContactProposalRegistry.guard,'__code__',(lambda self:None).__code__)
        elif target=='constructor':monkeypatch.setattr(contact.ConditionalContactProposal,'__init__',lambda *args:None)
        elif target=='class':monkeypatch.setattr(contact,'_ContactProposalView',object)
        else:monkeypatch.setattr(contact._ContactProposalView.query,'__defaults__',(None,))
    callback=registry.wrap_callback(mutate)
    with pytest.raises(GeometryError,match='runtime changed'):callback('original phase')


def test_contact_view_late_lookup_mutation_refused_by_retained_final_guard(source_stars,monkeypatch):
    source,_,receipt=source_stars
    registry=contact._ContactProposalRegistry(source);registry.prepare(receipt)
    final_guard=registry.guard
    monkeypatch.setattr(contact._ContactProposalView,'query',lambda *args:None)
    monkeypatch.setattr(contact._ContactProposalRegistry,'guard',lambda self:None)
    with pytest.raises(GeometryError,match='runtime changed'):final_guard()
