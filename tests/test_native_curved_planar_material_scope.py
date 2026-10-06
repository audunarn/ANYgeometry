"""Exact document curved material through public production preparation."""
from dataclasses import replace
from fractions import Fraction as F
from copy import deepcopy
from pathlib import Path
import json,os,sys
from importlib.metadata import version
import pytest,anygeometry
from anygeometry import GeometryModel,Plane,GeometryError,plan_intersections,apply_intersections,to_dict
from anygeometry import query_prepared_native_material_reference_scope as query
from anygeometry import validate_prepared_native_material_reference_scope_binding as validate
from anygeometry.entities import OrientedEdge
from anygeometry.native_material_reference_scope import _material,_compile_document
from anygeometry.native_support_snapshots import capture_native_supports
from anygeometry.cylinder_charts import _Proof
from anygeometry.native_arc_parameter_maps import NativeArcParameterMapPolicy
from anygeometry.curved_planar_material import Curve,Q,positive_chain_census,certify_domain,intersections


def test_runtime():
    origin=Path(anygeometry.__file__).resolve()
    assert origin==Path(__file__).resolve().parents[1]/'src/anygeometry/__init__.py'
    assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    Path(os.environ['CURVED_RUNTIME_EVIDENCE']).write_text(json.dumps(dict(executable=sys.executable,python_version=sys.version,
        source_origin=str(origin),numpy_version=version('numpy'),pytest_version=version('pytest'),threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2))


def prepare(model):
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')


def production(holes=False):
    from anygeometry.generators.structural import cylinder
    model=cylinder(1.,1.,circumferential_segments=4)
    root=model.add_plate(model.add_points(((-2,-2,.5),(2,-2,.5),(2,2,.5),(-2,2,.5))))
    if holes:
        edges=model.add_polyline(model.add_points(((1.3,-.2,.5),(1.6,-.2,.5),(1.6,.2,.5),(1.3,.2,.5))),close=True)
        with model.transaction():model._put_entity('face',replace(model.faces[root],holes=(tuple(OrientedEdge(e,True) for e in edges),)))
    prepare(model)
    return model,root


@pytest.mark.parametrize('holes',(False,True))
def test_public_plane_cylinder_curved_positive_partition(holes):
    model,root=production(holes);before=to_dict(model);receipt=query(model,[root]);row=receipt.material_rows[0]
    assert row.document_material_qualified,row.refusal
    assert len(row.current_face_ids)>1
    evidence=json.loads(row.domain_evidence_json)
    assert evidence['jordan_loops']>=3 and evidence['atomic_spans']>0
    assert 'winding' in evidence['proof']
    assert not receipt.meshing_permitted and not receipt.floating_evaluation_preservation_qualified
    assert not receipt.external_reference_scope_qualified
    validate(model,receipt);assert to_dict(model)==before


def circle_loop(reverse=False):
    points=[(F(5),F(0)),(F(0),F(5)),(F(-5),F(0)),(F(0),F(-5))]
    via=[(F(3),F(4)),(F(-3),F(4)),(F(-3),F(-4)),(F(3),F(-4))]
    loop=tuple(Curve('circle',a,b,(F(0),F(0)),F(25),True,v) for a,b,v in zip(points,points[1:]+points[:1],via))
    return tuple(replace(c,p=c.q,q=c.p,ccw=False) for c in loop[::-1]) if reverse else loop


def test_public_authored_circle_split_at_existing_vertices():
    model=GeometryModel();points=model.add_points(((5,0,0),(0,5,0),(-5,0,0),(0,-5,0)))
    via=model.add_points(((3,4,0),(-3,4,0),(-3,-4,0),(3,-4,0)))
    edges=tuple(model.add_arc(a,v,b) for a,v,b in zip(points,via,points[1:]+points[:1]))
    root=model.add_face(edges,surface=Plane((0,0,0),(1,0,0),(0,1,0)))
    model.add_plate(model.add_points(((0,-6,-1),(0,6,-1),(0,6,1),(0,-6,1))))
    prepare(model);receipt=query(model,[root]);assert receipt.material_rows[0].document_material_qualified,receipt.material_rows[0].refusal
    validate(model,receipt)


@pytest.mark.parametrize('adversary',('missing','duplicate','tampered'))
def test_actual_prepared_child_inventory_defects_refuse(adversary):
    model,root=production();receipt=query(model,[root]);binding=__import__('anygeometry.prepared_face_preimages',fromlist=['query_prepared_face_preimages']).query_prepared_face_preimages(model)
    original=_compile_document(json.loads(binding.authored_document_json));current=_compile_document(to_dict(model))
    children=receipt.material_rows[0].current_face_ids
    if adversary=='missing':children=children[:-1]
    elif adversary=='duplicate':children=children+(children[0],)
    else:
        child=next(c for c in children if current['faces'][c]['holes'])
        current['faces'][child]['holes']=[]
    source={k:(kind,data) for k,kind,data in binding.authored_native_supports}
    now={k:(kind,data) for k,kind,data in capture_native_supports(model)}
    row=_material(root,children,original,current,source,now,_Proof(NativeArcParameterMapPolicy(),None))
    assert not row.document_material_qualified


def test_opposite_duplicate_circles_and_cancellation():
    source=(circle_loop(),)
    with pytest.raises(GeometryError,match='multiplicity'):
        positive_chain_census((source,source,(circle_loop(True),)),_Proof(NativeArcParameterMapPolicy(),None))
    assert positive_chain_census((source,(circle_loop(True),)),_Proof(NativeArcParameterMapPolicy(),None))['atomic_spans']==4
    model,root=production();receipt=query(model,[root])
    with pytest.raises(GeometryError,match='cancelled'):query(model,[root],cancellation_check=lambda _:True)
    with pytest.raises(GeometryError):query(model,[root],policy=NativeArcParameterMapPolicy(max_interval_operations=1))
    epoch=model.revision;model.add_point(9.,9.,9.)
    with pytest.raises(GeometryError):validate(model,receipt)
    assert model.revision>epoch


def test_exact_quadratic_intersections_and_self_crossing():
    a=Curve('line',(F(-2),F(0)),(F(2),F(0)))
    b=Curve('circle',(F(0),F(-1)),(F(0),F(1)),(F(0),F(0)),F(1),True,(F(1),F(0)))
    assert len(intersections(a,b))==1 and intersections(a,b)[0][0]==1
    loop=tuple(Curve('line',a,b) for a,b in zip(((0,0),(2,2),(0,2),(2,0)),((2,2),(0,2),(2,0),(0,0))))
    with pytest.raises(GeometryError,match='Jordan'):certify_domain((loop,),_Proof(NativeArcParameterMapPolicy(),None))


def scaled(loop,scale):
    return tuple(replace(c,p=tuple(scale*x for x in c.p),q=tuple(scale*x for x in c.q),
        via=tuple(scale*x for x in c.via),radius2=c.radius2*scale*scale) for c in loop)


def test_positive_jordan_holes_and_opposite_duplicate_annuli():
    outer=circle_loop();inner=scaled(outer,F(1,2));source=(outer,)
    annulus=(outer,inner)
    reverse=lambda loop:tuple(replace(c,p=c.q,q=c.p,ccw=not c.ccw) for c in loop[::-1])
    assert positive_chain_census((source,annulus,(inner,)),_Proof(NativeArcParameterMapPolicy(),None))['carrier_count']==2
    with pytest.raises(GeometryError,match='multiplicity'):
        positive_chain_census((source,source,annulus,tuple(reverse(loop) for loop in annulus)),_Proof(NativeArcParameterMapPolicy(),None))
    with pytest.raises(GeometryError,match='overlap or nest'):
        certify_domain((outer,inner,scaled(outer,F(1,4))),_Proof(NativeArcParameterMapPolicy(),None))


def test_irrational_line_circle_and_circle_circle_hits_are_exact():
    curve=Curve('circle',(F(1),F(1)),(F(-1),F(1)),(F(0),F(0)),F(2),False,(F(1),F(-1)))
    line=Curve('line',(F(-2),F(0)),(F(2),F(0)))
    hits=intersections(line,curve)
    assert len(hits)==2 and all(hit[0]*hit[0]==2 for hit in hits)
    assert Q(0,1,2)>F(7,5) and Q(0,1,2)<F(3,2)
    # Two intersecting circumcircle spans retain quadratic-field coordinates.
    other=replace(curve,p=(F(2),F(1)),q=(F(0),F(1)),center=(F(1),F(0)),via=(F(2),F(-1)))
    hits=intersections(curve,other)
    assert len(hits)==1 and hits[0][0]==F(1,2)
    assert hits[0][1]*hits[0][1]==F(7,4)


def test_public_callback_mutation_and_exception_identity():
    model,root=production();before=model.revision
    def mutate(phase):
        if phase=='native material/reference scope final check':model.add_point(10.,10.,10.)
        return False
    with pytest.raises(GeometryError):query(model,[root],cancellation_check=mutate)
    assert model.revision>before
    model,root=production();sentinel=RuntimeError('callback sentinel')
    def fail(_):raise sentinel
    with pytest.raises(RuntimeError) as caught:query(model,[root],cancellation_check=fail)
    assert caught.value is sentinel


def test_curved_exterior_refit_is_not_assumed_equal():
    outer=circle_loop();refit=tuple(replace(c,center=(F(1,1000),F(0))) for c in outer)
    # Direct exact material proof cannot cancel unequal carrier definitions.
    with pytest.raises(GeometryError):positive_chain_census(((outer,),(refit,)),_Proof(NativeArcParameterMapPolicy(),None))


def test_tilted_curved_plane_is_explicitly_refused():
    model=GeometryModel()
    lift=lambda p:(p[0],-p[0],p[1])
    points=model.add_points(tuple(lift(p) for p in ((5,0),(0,5),(-5,0),(0,-5))))
    via=model.add_points(tuple(lift(p) for p in ((3,4),(-3,4),(-3,-4),(3,-4))))
    edges=tuple(model.add_arc(a,v,b) for a,v,b in zip(points,via,points[1:]+points[:1]))
    root=model.add_face(edges,surface=Plane((0,0,0),(1,-1,0),(0,0,1)))
    prepare(model);row=query(model,[root]).material_rows[0]
    assert not row.document_material_qualified and 'axis-aligned physical Plane' in row.refusal


def test_y_disjoint_active_scan_is_charged_to_shared_budget():
    from anygeometry.curved_planar_material import candidate_pairs
    from anygeometry.cylinder_charts import _Refusal
    curves=tuple(Curve('line',(F(0),F(i)),(F(100),F(i))) for i in range(20))
    proof=_Proof(NativeArcParameterMapPolicy(max_interval_operations=100),None)
    with pytest.raises(_Refusal,match='qualification_budget_exhausted'):tuple(candidate_pairs(curves,proof))


def test_literal_changed_circle_carriers_refuse_with_valid_domains():
    outer=circle_loop()
    translated=tuple(replace(c,p=(c.p[0]+1,c.p[1]),q=(c.q[0]+1,c.q[1]),
        via=(c.via[0]+1,c.via[1]),center=(F(1),F(0))) for c in outer)
    with pytest.raises(GeometryError,match='multiplicity'):
        positive_chain_census(((outer,),(translated,)),_Proof(NativeArcParameterMapPolicy(),None))


def test_public_same_revision_hole_tamper_refuses_query_and_issued_binding():
    model,root=production();receipt=query(model,[root]);revision=model.revision
    child=next(c for c in receipt.material_rows[0].current_face_ids if model.faces[c].holes)
    object.__setattr__(model.faces[child],'holes',())
    assert model.revision==revision
    with pytest.raises(GeometryError):query(model,[root])
    with pytest.raises(GeometryError):validate(model,receipt)


def test_generic_ray_rejection_scans_consume_shared_budget():
    from anygeometry.curved_planar_material import winding
    from anygeometry.cylinder_charts import _Refusal
    points=tuple((F(k+1),F(k*(k+1))) for k in range(20,-1,-1))+((F(-1),F(-1)),(F(-1),F(500)))
    loop=tuple(Curve('line',p,q) for p,q in zip(points,(*points[1:],points[0])))
    # Actual simple monotone outer loop, not an impossible primitive collection.
    certify_domain((loop,),_Proof(NativeArcParameterMapPolicy(),None))
    assert abs(winding(loop,(F(0),F(0)),_Proof(NativeArcParameterMapPolicy(),None)))==1
    with pytest.raises(_Refusal,match='qualification_budget_exhausted'):
        winding(loop,(F(0),F(0)),_Proof(NativeArcParameterMapPolicy(max_interval_operations=100),None))
