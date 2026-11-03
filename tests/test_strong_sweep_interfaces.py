"""Strong support separation is distinct from generic shared-edge witnesses."""
from fractions import Fraction
from dataclasses import replace
import pytest
from anygeometry import GeometryModel,GeometryError
from anygeometry.serialization import to_dict
from anygeometry.surfaces import CoonsSurface
from anygeometry.entities import OrientedEdge
from test_translational_nurbs_pairs import pair


def query(*args,**kwargs):
    from anygeometry import query_translational_profile_interfaces
    return query_translational_profile_interfaces(*args,**kwargs)


def test_public_query_is_fresh_immutable_and_source_bound():
    model,faces=pair(weights=(1.,2.,1.));before=to_dict(model)
    a=query(model,faces,expected_revision=model.revision);b=query(model,faces)
    assert a is not b and a==b
    assert a['schema']=='translational-profile-interfaces-v1'
    assert a['source_model_id']==str(model.model_id) and a['source_revision']==model.revision
    assert a['source_checksum']==before['checksum']['value']
    assert len(a['pairs'])==1 and len(a['pairs'][0]['shared_edges'])==1
    with pytest.raises(TypeError):a['pairs']=()
    assert to_dict(model)==before


def test_query_revision_cancellation_and_callback_mutation():
    model,faces=pair();before=to_dict(model)
    with pytest.raises(GeometryError,match='stale'):query(model,faces,expected_revision=-1)
    with pytest.raises(GeometryError,match='cancel'):query(model,faces,cancellation_check=lambda phase:True)
    assert to_dict(model)==before
    fired=[]
    def mutate(phase):
        if not fired:model.add_point(99,99,99);fired.append(True)
    with pytest.raises(GeometryError,match='changed'):query(model,faces,cancellation_check=mutate)


def test_crossing_profiles_remain_unsupported():
    model,faces=pair(crossing=True)
    assert query(model,faces) is None


def test_shared_edge_with_additional_interior_line_has_no_strong_certificate():
    model=GeometryModel()
    a,b,c,d,e,f=model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0),(1,1,.5),(0,1,-.5)))
    shared=model.add_line(a,b)
    def face(edges):
        return model.add_face_from_loop(tuple(OrientedEdge(edge,True) for edge in edges),(0,1,2,3),surface=CoonsSurface())
    first=face((shared,model.add_line(b,c),model.add_line(c,d),model.add_line(d,a)))
    second=face((shared,model.add_line(b,e),model.add_line(e,f),model.add_line(f,a)))
    assert query(model,(first,second)) is None
    # S1=(u,v,0), S2=(u,v,v*(u-.5)): both share v=0 and also meet at interior u=.5.


def test_nonzero_constant_transport_is_recorded_and_absolute_bound_is_enforced():
    from anygeometry.tolerance import DEFAULT_TOLERANCE_POLICY
    from anygeometry.nurbs import NurbsCurve
    def make(tolerance):
        model=GeometryModel(tolerance=tolerance);edges=[]
        for y in (0.,4.):
            curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,y,0.),(0.,y+.5,1.),(0.,y+1.,0.)),
                             (1.,2.,1.),domain=(.13,.83))
            a,b=model.add_points(curve.evaluate([0.,1.]));edges.append(model.add_curve(a,b,curve))
        return model,model.extrude(edges,(5.,0.,0.))
    model,faces=make(DEFAULT_TOLERANCE_POLICY)
    result=query(model,faces)
    assert result is not None
    assert max(Fraction(*row['upper_L1']) for face in result['faces'] for row in face['connectors'])>0
    model,faces=make(replace(DEFAULT_TOLERANCE_POLICY,length=1e-30))
    before=to_dict(model)
    with pytest.raises(GeometryError,match='original absolute length tolerance'):query(model,faces)
    assert to_dict(model)==before
