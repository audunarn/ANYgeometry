"""Real finite common-projection owner proofs; no sampled acceptance stubs."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
from anygeometry._local_contact_stars import query_local_contact_stars, validate_local_contact_stars
from test_nurbs_pair_public import model_pair


@pytest.fixture(scope='module')
def source_stars():
    source,faces=model_pair()
    for f in faces: source.add_sheet((f,))
    stars=query_local_contact_stars(source,faces,connect_intent=True)
    return source,faces,stars


def test_actual_closed_component_has_complete_finite_graph_atlas(source_stars):
    source,faces,result=source_stars
    validate_local_contact_stars(source,result)
    assert result.face_ids==faces and result.intent=='CONNECT'
    assert result.coordinate_system=='shared_native_knots'
    assert result.components and all(c.closed for c in result.components)
    for ci in range(len(result.components)):
        intervals=sorted(s.parameter_interval for s in result.stars if s.component==ci)
        assert intervals[0][0]==0 and intervals[-1][1]==1
        assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    for star in result.stars:
        assert star.derivative_margin>0 and star.side_margin>0 and star.exterior_squared_lower>0
        assert all(0<o[0]<i[0]<=i[1]<o[1]<1 for i,o in zip(star.inner_box,star.outer_box))
        assert all(parts for parts in star.exterior_rectangles)
        # Remote portions of BOTH same features survive in the finite ledger.
        assert all(any(box[0][0]==0 for box in parts) for parts in star.exterior_rectangles)


def test_source_receipt_copy_and_other_source_have_no_authority(source_stars):
    source,_,result=source_stars
    with pytest.raises(GeometryError): validate_local_contact_stars(source,replace(result))
    with pytest.raises(GeometryError): validate_local_contact_stars(source.clone(),result)


@pytest.mark.parametrize('operand',('source','receipt'))
def test_source_and_receipt_same_revision_callback_mutations_refused(source_stars,operand):
    source,faces,result=source_stars
    target,name=(source.faces[faces[0]].surface,'controls') if operand=='source' else (result,'intent')
    old=getattr(target,name)
    bad=tuple(tuple((p[0],p[1],p[2]+1) for p in row) for row in old) if operand=='source' else 'IMPRINT'
    def mutate(phase): object.__setattr__(target,name,bad)
    try:
        with pytest.raises(GeometryError): validate_local_contact_stars(source,result,cancellation_check=mutate)
    finally: object.__setattr__(target,name,old)
    validate_local_contact_stars(source,result)


def test_explicit_intent_budget_and_callback_exception():
    from anygeometry.predicates import IntersectionQualificationPolicy
    source,faces=model_pair()
    with pytest.raises(GeometryError,match='intent'): query_local_contact_stars(source,faces)
    with pytest.raises(GeometryError,match='budget'):
        query_local_contact_stars(source,faces,connect_intent=True,policy=IntersectionQualificationPolicy(max_boxes_per_pair=1))
    class Stop(Exception): pass
    def stop(phase): raise Stop('unchanged callback')
    with pytest.raises(Stop): query_local_contact_stars(source,faces,connect_intent=True,cancellation_check=stop)
    with pytest.raises(GeometryError,match='cancelled'):
        query_local_contact_stars(source,faces,connect_intent=True,cancellation_check=lambda phase: True)


def test_actual_source_hole_refuses_before_exclusion():
    source,faces=model_pair()
    face=source.faces[faces[0]]
    object.__setattr__(face,'holes',(face.loop,))
    with pytest.raises(GeometryError): query_local_contact_stars(source,faces,connect_intent=True)


def test_changed_native_runtime_is_not_a_stale_contact_authority(source_stars,monkeypatch):
    from anygeometry.nurbs import NurbsSurface
    source,_,result=source_stars
    original=NurbsSurface.exact_point
    def replacement(self,*args,**kwargs): return original(self,*args,**kwargs)
    monkeypatch.setattr(NurbsSurface,'exact_point',replacement)
    with pytest.raises(GeometryError): validate_local_contact_stars(source,result)
