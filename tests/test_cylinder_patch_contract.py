"""Standalone owner proof gates; public authoring, never a hidden full atlas."""
import math
from dataclasses import replace
from decimal import Decimal, localcontext
import numpy as np
import pytest
from anygeometry import GeometryModel, Cylinder, OrientedEdge, trim_face
from anygeometry.cylinder_patch import CylinderPatchPolicy, _PatchProof, _PatchSource, _PatchGeometry, _material
from anygeometry.cylinder_patch import (
    query_cylinder_patch, validate_cylinder_patch_binding,
    evaluate_cylinder_patch_occurrences, CylinderPatchOccurrenceRequest,
    CylinderPatchStatus,
    CylinderPatchError, CylinderPatchErrorCode,
)


def patch(*,start=0.,sweep=math.pi/2,height=2.,radius=1.,hole=False,
          origin=(0.,0.,0.),axis=(0.,0.,1.),radial=(1.,0.,0.),
          reverse=False,native=False,orientation=1,zrange=None,hole_uv=None):
    m=GeometryModel()
    s=Cylinder(origin,axis,radial,radius,height,start,sweep)
    vertices={}
    def point(u,v):
        if (u,v) not in vertices: vertices[u,v]=m.add_point(*s.evaluate(u,v))
        return vertices[u,v]
    def loop(uv):
        made=[]
        for a,b in zip(uv,uv[1:]+uv[:1]):
            x,y=(b,a) if native else (a,b)
            if a[1]==b[1]:
                key=m.add_arc(point(*x),point((a[0]+b[0])/2,a[1]),point(*y))
            else: key=m.add_line(point(*x),point(*y))
            made.append(OrientedEdge(key,not native))
        if reverse: made=[OrientedEdge(u.edge,not u.forward) for u in reversed(made)]
        return tuple(made)
    n=math.ceil(abs(sweep)/(math.pi/2))
    lo,hi=(0.,1.) if zrange is None else tuple(v/height for v in zrange)
    with m.transaction():
        outer=tuple((i/n,lo) for i in range(n+1))+tuple((i/n,hi) for i in reversed(range(n+1)))
        face=m.add_face_from_loop(loop(outer),surface=s)
        if hole or hole_uv is not None:
            trim_face(m,face,(loop(hole_uv if hole_uv is not None else
                ((1/3,.375),(1/3,.625),(2/3,.625),(2/3,.375))),))
        part=m.add_part(name='standalone')
        sheet=m.add_sheet((face,),part_id=part,orientations=(orientation,))
    assert m.validate_topology()==()
    return m,(m.handle('face_use',m.sheets[sheet].face_use_ids[0]),),vertices


CASES=[{},dict(hole=True),dict(start=7*math.pi/4,hole=True),
       dict(reverse=True,hole=True),dict(native=True,hole=True),dict(orientation=-1,hole=True),
       dict(radius=2.,height=3.,axis=(1.,0.,0.),radial=(0.,0.,-1.),origin=(2.,-3.,4.)),
       dict(zrange=(.4,1.6)),dict(sweep=math.pi),dict(sweep=3*math.pi/2),dict(sweep=15*math.pi/8),
       dict(start=3*math.pi/4,hole=True),dict(sweep=-math.pi/2,hole=True),
       dict(height=-2.,hole=True),dict(axis=(0.,0.,-1.),hole=True)]


@pytest.mark.parametrize('settings',CASES)
def test_source_and_vertex_branch_gate(settings):
    m,selected,vertices=patch(**settings)
    before=repr(m.__dict__)
    p=_PatchProof(CylinderPatchPolicy(),None)
    source=_PatchSource(m,selected,p)
    geometry=_PatchGeometry(source,m.tolerance,p)
    for uv,key in vertices.items():
        angle,z,boxes,lift=geometry.vertex(key)
        # Analytic authoring provenance is an independent diagnostic expectation;
        # raw source binary geometry need not equal that input angle exactly.
        assert max(abs(float(b.mid)-v) for b,v in zip(boxes,uv))<1e-12
    assert repr(m.__dict__)==before
    assert len(m.face_uses)==1


def test_public_tampered_result_rejected_without_mutation():
    m,selected,_=patch()
    result=query_cylinder_patch(m,selected,expected_revision=m.revision)
    forged=replace(result,radius=result.radius*2)
    before=repr(m.__dict__)
    with pytest.raises(CylinderPatchError) as caught:
        validate_cylinder_patch_binding(m,forged,selected,expected_revision=m.revision)
    assert caught.value.code is CylinderPatchErrorCode.INVALID_RESULT
    assert repr(m.__dict__)==before


def test_callback_exception_identity_and_query_purity():
    m,selected,_=patch()
    before=repr(m.__dict__)
    sentinel=RuntimeError('caller cancellation')
    def cancel(phase):
        raise sentinel
    with pytest.raises(RuntimeError) as caught:
        query_cylinder_patch(m,selected,expected_revision=m.revision,cancellation_check=cancel)
    assert caught.value is sentinel
    assert repr(m.__dict__)==before


def test_callback_revision_change_detected_immediately():
    m,selected,_=patch()
    revision=m.revision
    calls=[]
    def change(phase):
        calls.append(phase)
        m.add_point(9.,9.,9.)
    with pytest.raises(CylinderPatchError) as caught:
        query_cylinder_patch(m,selected,expected_revision=revision,cancellation_check=change)
    assert caught.value.code is CylinderPatchErrorCode.STALE_REVISION
    assert len(calls)==1


@pytest.mark.parametrize('value',[True,0,200001,1.5,'12'])
def test_policy_rejects_noncanonical_or_excessive_budget(value):
    with pytest.raises(CylinderPatchError):
        CylinderPatchPolicy(max_interval_operations=value)


@pytest.mark.parametrize('settings',CASES)
def test_whole_carrier_gate(settings):
    m,selected,vertices=patch(**settings)
    before=repr(m.__dict__)
    p=_PatchProof(CylinderPatchPolicy(),None)
    source=_PatchSource(m,selected,p)
    geometry=_PatchGeometry(source,m.tolerance,p)
    for edge in source.edges.values():
        result=geometry.curve(edge)
        assert result['residual'].hi<=geometry.residual_limit
    assert repr(m.__dict__)==before


@pytest.mark.parametrize('settings',CASES)
def test_material_gate(settings):
    m,selected,_=patch(**settings)
    before=repr(m.__dict__)
    p=_PatchProof(CylinderPatchPolicy(),None)
    source=_PatchSource(m,selected,p)
    geometry=_PatchGeometry(source,m.tolerance,p)
    curves={edge.id:geometry.curve(edge) for edge in source.edges.values()}
    loops=_material(source,geometry,p,curves)
    assert len(loops)==1+bool(settings.get('hole'))
    assert repr(m.__dict__)==before


@pytest.mark.parametrize('settings',CASES)
def test_public_query_binding_and_native_sampling(settings):
    m,selected,_=patch(**settings)
    before=repr(m.__dict__)
    result=query_cylinder_patch(m,selected,expected_revision=m.revision)
    assert result.status is CylinderPatchStatus.QUALIFIED, result.diagnostics
    assert result.certificate.complete
    validate_cylinder_patch_binding(m,result,selected,expected_revision=m.revision)
    requests=tuple(CylinderPatchOccurrenceRequest(o.coedge,t)
                   for o in result.occurrences for t in (0.,.5,1.))
    sampled=evaluate_cylinder_patch_occurrences(m,result,requests,
        face_uses=selected,expected_revision=m.revision)
    assert len(sampled.samples)==len(requests)
    assert sampled.certificate.complete
    for row in sampled.samples:
        assert row.status is CylinderPatchStatus.QUALIFIED
        assert all(lo<=v<=hi for v,(lo,hi) in zip(row.local_uv,row.local_uv_enclosure))
        assert all(lo<=v<=hi for v,(lo,hi) in zip(row.physical_chart,row.physical_enclosure))
    assert repr(m.__dict__)==before


def test_native_vertex_identity_shared_across_incident_occurrences():
    m,selected,_=patch(hole=True)
    result=query_cylinder_patch(m,selected,expected_revision=m.revision)
    requests=tuple(CylinderPatchOccurrenceRequest(o.coedge,t)
                   for o in result.occurrences for t in (0.,1.))
    evaluated=evaluate_cylinder_patch_occurrences(m,result,requests,
        face_uses=selected,expected_revision=m.revision)
    seen={}
    for row in evaluated.samples:
        assert row.endpoint_vertex is not None
        assert seen.setdefault(row.endpoint_vertex,row.identity_key)==row.identity_key
    assert len(set(seen.values()))==len(seen)


def test_busy_and_stale_requests_do_not_mutate():
    m,selected,_=patch()
    old_revision=m.revision
    result=query_cylinder_patch(m,selected,expected_revision=old_revision)
    m.add_point(8.,8.,8.)
    before=repr(m.__dict__)
    with pytest.raises(CylinderPatchError) as caught:
        validate_cylinder_patch_binding(m,result,selected,expected_revision=m.revision)
    assert caught.value.code is CylinderPatchErrorCode.STALE_RESULT
    assert repr(m.__dict__)==before
    with m.transaction():
        before=repr(m.__dict__)
        with pytest.raises(CylinderPatchError) as caught:
            query_cylinder_patch(m,selected,expected_revision=m.revision)
        assert caught.value.code is CylinderPatchErrorCode.BUSY_MODEL
        assert repr(m.__dict__)==before


def test_insufficient_arithmetic_budget_returns_no_consumable_evidence():
    m,selected,_=patch()
    before=repr(m.__dict__)
    result=query_cylinder_patch(m,selected,expected_revision=m.revision,
                                policy=CylinderPatchPolicy(max_interval_operations=1))
    assert result.status is CylinderPatchStatus.UNRESOLVED
    assert not result.certificate.complete
    assert result.loops==result.occurrences==()
    assert result.radius is None
    assert repr(m.__dict__)==before


def _decimal_sincos(x):
    # Independent 80-decimal-digit Taylor oracle; not production interval code.
    sine=term=x
    cosine=cterm=Decimal(1)
    for n in range(1,160):
        term *= -x*x/Decimal((2*n)*(2*n+1))
        cterm *= -x*x/Decimal((2*n-1)*(2*n))
        sine += term
        cosine += cterm
    return sine,cosine


@pytest.mark.parametrize('settings',[{},dict(start=3*math.pi/4,hole=True),dict(sweep=15*math.pi/8)])
def test_independent_high_precision_returned_world_residual(settings):
    m,selected,_=patch(**settings)
    result=query_cylinder_patch(m,selected,expected_revision=m.revision)
    requests=tuple(CylinderPatchOccurrenceRequest(o.coedge,t)
                   for o in result.occurrences for t in (.125,.5,.875))
    rows=evaluate_cylinder_patch_occurrences(m,result,requests,
        face_uses=selected,expected_revision=m.revision).samples
    with localcontext() as context:
        context.prec=80
        d=Decimal.from_float
        for row in rows:
            theta=d(result.start_angle)+d(result.sweep_angle)*d(row.local_uv[0])
            sine,cosine=_decimal_sincos(theta)
            expected=(d(result.radius)*cosine,d(result.radius)*sine,d(result.height)*d(row.local_uv[1]))
            residual=sum((d(v)-w)**2 for v,w in zip(row.point,expected)).sqrt()
            assert residual<=d(row.residual_bound)


def test_cut_interval_uses_exactly_seventeen_period_candidates():
    m,selected,_=patch(start=3*math.pi/4)
    p=_PatchProof(CylinderPatchPolicy(),None)
    source=_PatchSource(m,selected,p)
    geometry=_PatchGeometry(source,m.tolerance,p)
    from fractions import Fraction
    epsilon=Fraction(1,2**80)
    before=p.counts['lift_candidates']
    angle,_=geometry.lift(p.i(-1),p.i(-epsilon,epsilon))
    assert p.counts['lift_candidates']-before==17
    assert angle.lo<=p.pi_bound().lo<=p.pi_bound().hi<=angle.hi


@pytest.mark.parametrize('bad',[None,True,1,'face_use',(),(None,)])
def test_invalid_selection_rejects_before_mutation(bad):
    m,selected,_=patch()
    before=repr(m.__dict__)
    with pytest.raises((CylinderPatchError,TypeError)):
        query_cylinder_patch(m,bad,expected_revision=m.revision)
    assert repr(m.__dict__)==before


def test_extra_selection_and_wrong_model_rejected():
    m,selected,_=patch()
    other,foreign,_=patch()
    before=repr(m.__dict__)
    with pytest.raises(CylinderPatchError) as caught:
        query_cylinder_patch(m,(*selected,None),expected_revision=m.revision)
    assert caught.value.code is CylinderPatchErrorCode.INVALID_REQUEST
    with pytest.raises(CylinderPatchError) as caught:
        query_cylinder_patch(m,foreign,expected_revision=m.revision)
    assert caught.value.code is CylinderPatchErrorCode.WRONG_MODEL
    assert repr(m.__dict__)==before


def test_sampling_limit_is_atomic_and_query_budget_is_not_reset():
    m,selected,_=patch()
    result=query_cylinder_patch(m,selected,expected_revision=m.revision,
        policy=CylinderPatchPolicy(max_evaluations=1))
    before=repr(m.__dict__)
    request=CylinderPatchOccurrenceRequest(result.occurrences[0].coedge,.5)
    with pytest.raises(CylinderPatchError):
        evaluate_cylinder_patch_occurrences(m,result,(request,request),
            face_uses=selected,expected_revision=m.revision)
    single=evaluate_cylinder_patch_occurrences(m,result,(request,),
        face_uses=selected,expected_revision=m.revision)
    assert dict(single.certificate.work_counts)['interval_operations']>dict(result.certificate.work_counts)['interval_operations']
    assert repr(m.__dict__)==before


def test_final_receipt_counts_match_measured_query_work():
    from anygeometry.cylinder_patch import _run
    m,selected,_=patch(hole=True)
    result,_,_,_,proof=_run(m,selected,m.revision,None,None)
    assert dict(result.certificate.work_counts)==proof.counts
    assert proof.counts['encoding_bytes']>0
    assert proof.counts['encoding_nodes']>0


def test_late_receipt_cancellation_preserves_identity_and_state():
    m,selected,_=patch()
    before=repr(m.__dict__)
    error=RuntimeError('cancel at final receipt')
    def cancel(phase):
        if phase=='patch receipt complete':
            raise error
    with pytest.raises(RuntimeError) as caught:
        query_cylinder_patch(m,selected,expected_revision=m.revision,cancellation_check=cancel)
    assert caught.value is error
    assert repr(m.__dict__)==before


def test_nonrectangular_hole_has_owner_qualified_material_and_void_seeds():
    # Orthogonal L-shaped void uses genuine axial/circular source edges.
    m,selected,_=patch(hole_uv=((.2,.2),(.2,.8),(.5,.8),(.5,.5),(.8,.5),(.8,.2)))
    before=repr(m.__dict__)
    result=query_cylinder_patch(m,selected,expected_revision=m.revision)
    assert result.status is CylinderPatchStatus.QUALIFIED, result.diagnostics
    assert result.certificate.complete
    assert result.certificate.algorithm == 'cylinder-partial-orthogonal-v1'
    assert len(result.loops) == 2 and len(result.occurrences) == 10
    outer, hole = result.loops
    assert outer.winding == -hole.winding
    u, v = hole.seed_uv
    assert .2 < u < .8 and .2 < v < .8 and (u < .5 or v < .5)
    u, v = outer.seed_uv
    assert 0 < u < 1 and 0 < v < 1
    assert not (.2 < u < .8 and .2 < v < .8 and (u < .5 or v < .5))
    validate_cylinder_patch_binding(m,result,selected,expected_revision=m.revision)
    assert repr(m.__dict__)==before


def test_hole_without_qualified_clearance_is_not_admitted():
    # A narrow but valid authored gap must not be silently snapped into material.
    m,selected,_=patch(hole_uv=((1e-7,.375),(1e-7,.625),(.5,.625),(.5,.375)))
    before=repr(m.__dict__)
    result=query_cylinder_patch(m,selected,expected_revision=m.revision)
    if result.status is CylinderPatchStatus.QUALIFIED:
        outer,hole=result.loops
        assert hole.uv_bounds[0][0]>outer.uv_bounds[0][1]
    else:
        assert not result.certificate.complete
        assert result.loops==()
    assert repr(m.__dict__)==before


def test_oversized_single_evidence_string_rejects_before_encoding():
    from anygeometry.cylinder_patch import _EvidenceBudget
    with pytest.raises(CylinderPatchError):
        _EvidenceBudget().visit('\U0001f600'*2049)


def test_hole_count_is_bounded_before_iteration():
    from types import SimpleNamespace
    from anygeometry.cylinder_patch import _Refusal
    m,selected,_=patch()
    use=m.face_uses[selected[0].id]
    face=m.faces[use.face_id]
    class TooManyHoles:
        def __len__(self): return 2
        def __iter__(self): raise AssertionError('unbounded holes copied')
    source=SimpleNamespace(model_id=m.model_id,face_uses=m.face_uses,
        sheets=m.sheets,parts=m.parts,faces={face.id:SimpleNamespace(
            surface=face.surface,parameterization=None,loop=face.loop,holes=TooManyHoles())})
    with pytest.raises(_Refusal,match='patch_loop_family'):
        _PatchSource(source,selected,_PatchProof(CylinderPatchPolicy(),None))
