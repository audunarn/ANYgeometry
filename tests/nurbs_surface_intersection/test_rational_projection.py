"""Exact separable rational admission and public contracts; root executes."""
from dataclasses import replace
from fractions import Fraction as Q
from importlib import import_module
from types import SimpleNamespace
import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import (query_nurbs_surface_intersection,evaluate_arcs,evaluate_arc,
    IntersectionStatus,NurbsSurfaceIntersectionIntegrityError,NurbsSurfaceIntersectionCancelled)
from anygeometry.nurbs_surface_intersection._evaluate import result_digest
from anygeometry.nurbs_surface_intersection._structural import derive_structural
from anygeometry.nurbs_surface_intersection._records import RationalProjectionCertificate
from anygeometry.nurbs_surface_intersection._source import SourceDefinition, Limits, WorkCounters
from anygeometry.nurbs_surface_intersection._topology import run_scalar_section
from test_chart_fallback import nested_rational_pair
from test_fast_path import nested_pair


def test_rational_nested_discriminator_exact_field_and_two_closed_paths():
    a,b = nested_rational_pair()
    certificate,reason = derive_structural(SourceDefinition.capture(a),SourceDefinition.capture(b))
    assert isinstance(certificate,RationalProjectionCertificate),reason
    pa,pb = nested_pair()
    polynomial,_ = derive_structural(SourceDefinition.capture(pa),SourceDefinition.capture(pb))
    assert certificate.field == polynomial.field
    section = run_scalar_section(SourceDefinition.capture(a),certificate,Limits(),lambda:None,WorkCounters())
    assert len(section.paths)==2 and all(path.closed for path in section.paths)


def test_rational_nested_public_batch_all_arcs_and_native_correspondence(monkeypatch):
    a,b = nested_rational_pair()
    before = SourceDefinition.capture(a),SourceDefinition.capture(b)
    result = query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED and result.cover_complete
    assert result.algorithm=="shared-separable-rational-projection-scalar"
    assert len(result.components)==2 and all(c.closed for c in result.components)
    module = import_module("anygeometry.nurbs_surface_intersection.query")
    actual,calls = module.run_scalar_section,[]
    def section(*args,**kwargs):
        calls.append(True)
        return actual(*args,**kwargs)
    monkeypatch.setattr(module,"run_scalar_section",section)
    requests = tuple((ci,ai,t) for ci,c in enumerate(result.components)
                     for ai in range(len(c.arcs)) for t in (Q(0),Q(1,2),Q(1)))
    points = evaluate_arcs(result,requests,first=a,second=b,tolerances=TolerancePolicy())
    assert len(calls)==1 and len(points)==len(requests)
    for request,point in zip(requests,points):
        assert (point.component_index,point.arc_index,point.station)==request
        assert len(point.native_box)==2 and point.first_chart==point.second_chart
        assert point.world_diameter<=1e-9 and not point.witness_is_exact
        (u0,u1),(v0,v1)=point.native_box
        # Exact interval oracle: one of the radii 1/4,3/4 must remain possible.
        def square(lo,hi):
            return (Q(0) if lo<=0<=hi else min(lo*lo,hi*hi),max(lo*lo,hi*hi))
        x,y=square(2*u0-1,2*u1-1),square(2*v0-1,2*v1-1)
        assert any(x[0]+y[0]<=r<=x[1]+y[1] for r in (Q(1,16),Q(9,16)))
    assert before[0].matches(a) and before[1].matches(b)


def test_structural_batch_checks_private_dependency_pins_on_entry(monkeypatch):
    a,b=_cheap_rational()
    result=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())
    dependencies=import_module("anygeometry.nurbs_surface_intersection._dependencies")
    monkeypatch.setattr(dependencies,"PINS",(("nurbs.py","0"*64),))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError,match="source-pinned dependency changed"):
        evaluate_arcs(result,((0,0,Q(1,2)),),first=a,second=b,tolerances=TolerancePolicy())


def test_generic_status_tamper_is_refused_by_single_and_batch():
    from test_generic import seam_planes
    a,b=seam_planes()
    result=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED and result.algorithm=="generic-atlas"
    result=replace(result,status=IntersectionStatus.UNCLASSIFIED)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError,match="CERTIFIED"):
        evaluate_arcs(result,((0,0,Q(1,2)),),first=a,second=b,tolerances=TolerancePolicy())
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError,match="CERTIFIED"):
        evaluate_arc(result,0,0,Q(1,2),first=a,second=b,tolerances=TolerancePolicy())


@pytest.mark.parametrize("kind",("nonproportional","nonrankone","nonseparable","fold"))
def test_rational_admission_rejects_unproved_correspondence(kind):
    a,b = nested_rational_pair()
    if kind in ("nonproportional","nonrankone"):
        weights=[list(row) for row in a.weights]
        weights[1][1]+=1.
        weights=tuple(tuple(row) for row in weights)
        b=replace(b,weights=weights)
        if kind=="nonrankone":
            a=replace(a,weights=weights)
    else:
        controls=[]
        for surface in (a,b):
            grid=[]
            for i,row in enumerate(surface.controls):
                grid.append(tuple(((float((0,1,2,1,0)[i]) if kind=="fold" else p[0]+(Q(1,8) if i==j==1 else 0)),p[1],p[2])
                                  for j,p in enumerate(row)))
            controls.append(tuple(grid))
        a,b=replace(a,controls=controls[0]),replace(b,controls=controls[1])
    certificate,reason=derive_structural(SourceDefinition.capture(a),SourceDefinition.capture(b))
    assert certificate is None and reason


@pytest.mark.parametrize("degree,knots,count,admitted",(
    (1,(0.,0.,.5,1.,1.),3,False),
    (2,(0.,0.,0.,.5,1.,1.,1.),4,True)))
def test_rational_original_supports_require_c1_and_certify_multiple_spans(degree,knots,count,admitted):
    kv=(0.,)*(degree+1)+(1.,)*(degree+1)
    weights=tuple((float(1+i%2),)*(degree+1) for i in range(count))
    surfaces=tuple(NurbsSurface(degree,degree,knots,kv,
        tuple(tuple((float(i),float(j),float(z)) for j in range(degree+1)) for i in range(count)),weights)
        for z in (0,1))
    certificate,reason=derive_structural(*(SourceDefinition.capture(s) for s in surfaces))
    assert isinstance(certificate,RationalProjectionCertificate)==admitted,reason
    if admitted:
        assert len(certificate.derivative_proofs[0])==2
    else:
        assert "C1" in reason


def _cheap_rational():
    knots=(0.,0.,0.,1.,1.,1.)
    weights=((1.,)*3,(2.,)*3,(1.,)*3)
    return tuple(NurbsSurface(2,2,knots,knots,
        tuple(tuple((float(i-1),float(j-1),float(offset*(j-1))) for j in range(3)) for i in range(3)),weights)
        for offset in (0,1))


def test_structural_batch_uses_trusted_records_after_lazy_result_mutation():
    a,b=_cheap_rational()
    result=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED
    def requests():
        object.__setattr__(result,"roots",())
        object.__setattr__(result,"components",())
        object.__setattr__(result,"first_source",None)
        yield 0,0,Q(1,2)
    points=evaluate_arcs(result,requests(),first=a,second=b,tolerances=TolerancePolicy())
    assert len(points)==1 and points[0].world_diameter<=1e-9


@pytest.mark.parametrize("attack",("stale","rehash","duck","iterator_source","cancel"))
def test_structural_batch_authority_and_callback_guards(attack):
    a,b=_cheap_rational()
    result=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())
    if attack=="rehash":
        bad=replace(result.roots[0],lo=result.roots[0].lo-Q(1,100))
        roots=(bad,)+result.roots[1:]
        result=replace(result,roots=roots,
            descriptor_digest=result_digest(result.structural,roots,result.boxes,result.components))
    if attack=="duck":
        result=SimpleNamespace(**{name:getattr(result,name) for name in
            ("status","algorithm","first_source","second_source","structural","roots","boxes","components","descriptor_digest")})
    def mutate():
        controls=list(a.controls)
        row=list(controls[0]);x,y,z=row[0];row[0]=(x,y,z+1.)
        controls[0]=tuple(row)
        object.__setattr__(a,"controls",tuple(controls))
    if attack=="stale":
        mutate()
    def requests():
        if attack=="iterator_source":
            mutate()
        yield 0,0,Q(1,2)
    error=NurbsSurfaceIntersectionCancelled if attack=="cancel" else NurbsSurfaceIntersectionIntegrityError
    with pytest.raises(error):
        evaluate_arcs(result,requests(),first=a,second=b,tolerances=TolerancePolicy(),
                      cancellation_check=(lambda:True) if attack=="cancel" else None)


def test_cancellation_inside_rational_admission_has_no_partial_authority(monkeypatch):
    module=import_module("anygeometry.nurbs_surface_intersection._rational_projection")
    actual=module._product
    armed=[False]
    def product(*args,**kwargs):
        armed[0]=True
        return actual(*args,**kwargs)
    monkeypatch.setattr(module,"_product",product)
    a,b=_cheap_rational()
    before=SourceDefinition.capture(a),SourceDefinition.capture(b)
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),cancellation_check=lambda:armed[0])
    assert before[0].matches(a) and before[1].matches(b)
