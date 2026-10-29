"""Connected finite source field and real participant-aware CONNECT tests."""
from fractions import Fraction as Q
from dataclasses import replace
import pytest
from anygeometry import (GeometryModel,GeometryError,Plane,clone_prepared_geometry,
    plan_intersections,apply_intersections,query_material_surface_regions)
from anygeometry.nurbs import NurbsCurve
from anygeometry.local_approximation_field import (query_local_approximation_field,
    validate_local_approximation_field,query_local_approximation_limit,bind_local_approximation_field)
from anygeometry.local_sweep_plane_contact import SweepPlaneContactStars
from anygeometry.local_field_transport import query_local_field_transport
from anygeometry.local_contact_transport import query_local_contact_transport,validate_local_contact_transport


def closed_source():
    model=GeometryModel()
    yz=((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0))
    profile=NurbsCurve(2,(0,0,0,.25,.25,.5,.5,.75,.75,1,1,1),
        tuple((0,y,z) for y,z in yz),(1,.75,1,.75,1,.75,1,.75,1))
    vertices=model.add_points(tuple((0.,float(y),float(z)) for y,z in yz[::2][:-1]))
    edges=tuple(model.add_curve(vertices[k],vertices[(k+1)%4],replace(profile,domain=(k/4,(k+1)/4))) for k in range(4))
    faces=tuple(model.extrude(edges,(5.,0.,0.)))
    # Independent original sheet identities are intentional, even with shared
    # geometric connector anchors. The field must not weld these participants.
    for face in faces:model.add_sheet((face,))
    corners=model.add_points(((2.5,-2.,-2.),(2.5,2.,-2.),(2.5,2.,2.),(2.5,-2.,2.)))
    boundary=tuple(model.add_line(a,b) for a,b in zip(corners,corners[1:]+corners[:1]))
    plane=model.add_face(boundary,surface=Plane((2.5,0.,0.),(0.,1.,0.),(0.,0.,1.)))
    model.add_sheet((plane,))
    return model,faces,plane


@pytest.fixture(scope='module')
def source_field():
    source,faces,plane=closed_source()
    field=query_local_approximation_field(source,(*faces,plane),relative_accuracy=.001,
        scale_error=.01,policy_floor=1.e-10,connect_intent=True)
    return source,faces,plane,field


def test_complete_source_field_has_finite_participant_ledgers(source_field):
    source,faces,plane,field=source_field
    validate_local_approximation_field(source,field)
    assert len(field.sweep_unions)==1 and field.sweep_unions[0].face_ids==faces
    assert len(field.contacts)==1 and type(field.contacts[0]) is SweepPlaneContactStars
    stars=field.contacts[0]
    assert set(stars.face_ids)=={*faces,plane}
    assert len(stars.law.components)==4
    assert all(source.faces[f].parameterization is None for f in faces)
    assert len(source.sheets)==5
    assert any(kind=='source_sweep_continuation' for face,box,gap,ledger in field.spatial_ledger for other,kind,row in ledger)
    assert stars.partitions
    for row in stars.partitions:
        boxes=[box for box,gap in row.retained]+list(row.excluded)
        assert sum((b-a)*(d-c) for (a,b),(c,d) in boxes)==1
        assert row.excluded and row.exterior_squared_lower>0
        assert all(gap>0 for box,gap in row.retained)
    matrix=next(matrix for face,matrix,tiles in field.face_tiles if face==plane)
    assert matrix==((Q(4),Q(0),Q(-2)),(Q(0),Q(4),Q(-2)))
    with pytest.raises(GeometryError,match='actual prepared'):
        query_local_approximation_limit(source,field,faces[0])


def test_unprepared_clone_cannot_admit_conditional_contact(source_field):
    source,faces,plane,field=source_field
    prepared=clone_prepared_geometry(source)
    regions=query_material_surface_regions(prepared,tuple(prepared.faces))
    transport=query_local_field_transport(source,prepared,regions)
    with pytest.raises(GeometryError):query_local_contact_transport(source,prepared,field.contacts[0],transport)


def test_actual_plane_connect_preserves_all_original_participants(source_field):
    source,faces,plane,field=source_field
    prepared=clone_prepared_geometry(source)
    plan=plan_intersections(prepared,(*faces,plane),policy='connect')
    apply_intersections(prepared,plan,policy='connect')
    regions=query_material_surface_regions(prepared,tuple(prepared.faces))
    transport=query_local_field_transport(source,prepared,regions)
    result=query_local_contact_transport(source,prepared,field.contacts[0],transport)
    validate_local_contact_transport(source,prepared,result)
    assert {row.source_face.id for row in transport.footprints}=={*faces,plane}
    assert {row[0] for row in result.trace_fragments}==set(range(4))
    assert all(len(row[3])==4 for row in result.trace_fragments)
    # Real owner allocation must execute the same realization gate, not accept
    # a previously serialized/caller-created transport stamp.
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    assert allocation.face_limits


@pytest.mark.parametrize('operand',('field','contact','partition','union'))
def test_issued_source_field_rejects_changed_finite_authority(source_field,operand):
    source,faces,plane,field=source_field
    stars=field.contacts[0]
    target,name,bad={'field':(field,'contacts',()),'contact':(stars,'partitions',()),
        'partition':(stars.partitions[0],'excluded',()),'union':(field.sweep_unions[0],'seams',())}[operand]
    old=getattr(target,name)
    try:
        object.__setattr__(target,name,bad)
        with pytest.raises(GeometryError):validate_local_approximation_field(source,field)
    finally:object.__setattr__(target,name,old)
    validate_local_approximation_field(source,field)


def test_callback_mutated_contact_partition_and_external_error_identity(source_field):
    source,faces,plane,field=source_field
    row=field.contacts[0].partitions[0];old=row.retained
    def mutate(phase):object.__setattr__(row,'retained',())
    try:
        with pytest.raises(GeometryError):validate_local_approximation_field(source,field,cancellation_check=mutate)
    finally:object.__setattr__(row,'retained',old)
    token=GeometryError('external field binding token')
    def stop(phase):raise token
    with pytest.raises(GeometryError) as caught:validate_local_approximation_field(source,field,cancellation_check=stop)
    assert caught.value is token


def test_transport_allowance_widens_world_and_directional_distance_bounds():
    from anygeometry._local_spatial_clearance import _inflate_transport,_gap
    from anygeometry.nurbs_surface_intersection._intervals import Interval as I
    hull=(I(1,2),I(0),I(0));direction=(((Q(2),Q(-1),Q(0)),Q(2),Q(4)),)
    widened,projected=_inflate_transport(hull,direction,Q(1,4))
    assert tuple((x.lo,x.hi) for x in widened)==((Q(3,4),Q(9,4)),(Q(-1,4),Q(1,4)),(Q(-1,4),Q(1,4)))
    assert projected[0][1:]==(Q(5,4),Q(19,4))
    # Actual displaced target (3/4,0,0) is closer; the retained lower bound
    # must never exceed its exact squared distance from the source origin.
    assert _gap((I(0),I(0),I(0)),widened,projected)<=Q(9,16)


def test_plane_float64_boundaries_keep_the_actual_field_detached(monkeypatch):
    from anygeometry._detached_geometry_read import NumericalBoundary
    source=GeometryModel()
    points=source.add_points(((0.,0.,0.),(2.,0.,0.),(2.,2.,0.),(0.,2.,0.)))
    edges=tuple(source.add_line(a,b) for a,b in zip(points,points[1:]+points[:1]))
    face=source.add_face(edges,surface=Plane((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)))
    from anygeometry.serialization import to_dict
    before=to_dict(source)
    original=NumericalBoundary.tick;observed=[]
    def observe(self,*args,**kwargs):
        if self.source is source:
            observed.append(self._detached)
            assert self._detached, 'canonical float64 boundary must not force full model qualification per tick'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(NumericalBoundary,'tick',observe)
    field=query_local_approximation_field(source,(face,),relative_accuracy=.001,scale_error=.01,policy_floor=1.e-10)
    validate_local_approximation_field(source,field)
    assert observed and all(observed)
    assert to_dict(source)==before


def test_float64_copy_preserves_live_path_and_signed_zero():
    import math
    import numpy as np
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.local_approximation_field import _canonical_line_operand
    from anygeometry._detached_geometry_read import detach_numeric,DetachedOperandUnavailable
    path=LinePath((np.float64(-0.),np.float64(.5),np.float64(2.)),(np.float64(1.),np.float64(1.),np.float64(2.)))
    original=path.start,path.end
    copied=detach_numeric(_canonical_line_operand(path))
    assert copied is not path and (path.start,path.end)==original
    assert all(type(x) is np.float64 for row in (path.start,path.end) for x in row)
    assert all(type(x) is float for row in (copied.start,copied.end) for x in row)
    assert math.copysign(1.,copied.start[0])==-1.
    class Unknown(float):pass
    unknown=LinePath((Unknown(0),0.,0.),(1.,0.,0.))
    with pytest.raises(DetachedOperandUnavailable):detach_numeric(_canonical_line_operand(unknown))


def test_actual_plane_field_callback_mutation_still_refuses():
    source=GeometryModel()
    points=source.add_points(((0.,0.,0.),(2.,0.,0.),(2.,2.,0.),(0.,2.,0.)))
    edges=tuple(source.add_line(a,b) for a,b in zip(points,points[1:]+points[:1]))
    face=source.add_face(edges,surface=Plane((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)))
    record=source.faces[face];old=record.metadata;touched=[]
    def mutate(phase):
        if phase=='local approximation field':
            touched.append(True);object.__setattr__(record,'metadata',{'same_revision_callback_change':True})
        return False
    try:
        with pytest.raises(GeometryError):query_local_approximation_field(source,(face,),relative_accuracy=.001,
            scale_error=.01,policy_floor=1.e-10,cancellation_check=mutate)
        assert touched
    finally:object.__setattr__(record,'metadata',old)
