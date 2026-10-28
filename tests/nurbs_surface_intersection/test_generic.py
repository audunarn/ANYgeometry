"""Independent rational analytic oracles; definitions only, author has not run."""
from fractions import Fraction as Q
from types import SimpleNamespace
import dataclasses

import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.errors import GeometryError
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import (query_nurbs_surface_intersection,evaluate_arc,evaluate_arcs,
    IntersectionStatus,NurbsSurfaceIntersectionIntegrityError)
from anygeometry.nurbs_surface_intersection._source import SourceDefinition
from anygeometry.nurbs_surface_intersection._structural import derive_structural


def rational_circle():
    """Exact numerator (2u-1)^2+(2v-1)^2-1/4, monotone identical XY charts.

    x=(2u-1)/(1+2u-2u^2) has positive derivative on [0,1]; y=2v-1.
    Thus equality of XY forces u=s,v=t and the complete zero set is one
    interior circle in native parameters. No scalar admission uses this oracle.
    """
    ks=(0.,0.,0.,1.,1.,1.)
    weights=((1.,)*3,(2.,)*3,(1.,)*3)
    square=(Q(1),Q(-1),Q(1))
    controls=[]
    for offset in (0,1):
        grid=[]
        for i in range(3):
            row=[]
            for j in range(3):
                base=square[i]+square[j]
                delta=(base-Q(1,4))/Q(weights[i][j])
                z=base+offset*delta
                assert Q(float(z))==z
                row.append((float(i-1),float(j-1),float(z)))
            grid.append(tuple(row))
        controls.append(tuple(grid))
    return tuple(NurbsSurface(2,2,ks,ks,c,weights) for c in controls)


def test_generic_circle_bypasses_uniform_affine_gate_and_obeys_tiny_budget():
    a,b=rational_circle()
    structural,_=derive_structural(SourceDefinition.capture(a),SourceDefinition.capture(b))
    # Public admission now recognizes this exact rational subset; this module's
    # explicit internal fixture preserves coverage of the general 4D algorithm.
    from anygeometry.nurbs_surface_intersection._records import RationalProjectionCertificate
    assert isinstance(structural,RationalProjectionCertificate)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),
        qualification=SimpleNamespace(max_boxes_per_pair=48,max_subdivision_depth=20))
    assert r.status is IntersectionStatus.UNCLASSIFIED
    assert r.stats.boxes_visited<=48
    assert r.atlas.frontier_remaining+r.atlas.unresolved>0
    assert r.atlas.covered==len(r.atlas.witnesses)


def test_generic_circle_complete_ordered_cycle_and_original_evaluation():
    a,b=rational_circle()
    limits=SimpleNamespace(max_boxes_per_pair=12000,max_subdivision_depth=32)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED,r.reason
    assert r.algorithm=="generic-atlas" and r.cover_complete
    assert len(r.components)==1 and r.components[0].closed
    c=r.components[0]
    assert c.root_ids[0]==c.root_ids[-1]
    assert len({r.atlas.witnesses[a.witness].free_coordinate for a in c.arcs})>=2
    for arc_index,arc in enumerate(c.arcs):
        assert arc.direction in (-1,1)
    left_index=next(i for i,arc in enumerate(c.arcs)
        if r.atlas.witnesses[arc.witness].free_coordinate!=
           r.atlas.witnesses[c.arcs[(i+1)%len(c.arcs)].witness].free_coordinate)
    right_index=(left_index+1)%len(c.arcs)
    assert c.arcs[left_index].second_root==c.arcs[right_index].first_root
    requests=((0,0,Q(0)),(0,0,Q(1,2)),(0,0,Q(1)),
              (0,left_index,Q(1)),(0,right_index,Q(0)))
    points=evaluate_arcs(r,requests,
                        first=a,second=b,tolerances=TolerancePolicy(),qualification=limits)
    left,right=points[-2:]
    assert (left.component_index,left.arc_index,left.station)==(0,left_index,Q(1))
    assert (right.component_index,right.arc_index,right.station)==(0,right_index,Q(0))
    payload=lambda p:(p.native_box,p.first_chart,p.second_chart,p.world_lo,p.world_hi,
                      p.witness,p.witness_is_exact,p.world_diameter,p.target)
    assert payload(left)==payload(right)
    for p in points:
        assert p.world_diameter<=min(TolerancePolicy().length,TolerancePolicy().surface_residual)
        u,v,s,t=(Q(x)+Q(y) for x,y in p.native_box)
        u,v,s,t=(x/2 for x in (u,v,s,t))
        # Centres check the independent analytical oracle, never admission.
        assert abs(float((2*u-1)**2+(2*v-1)**2-Q(1,4)))<1e-7


def test_dependency_source_pin_is_an_enforced_query_gate(monkeypatch):
    from anygeometry.nurbs_surface_intersection import _dependencies
    monkeypatch.setattr(_dependencies,"PINS",(("nurbs.py","0"*64),))
    a,b=rational_circle()
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError,match="dependency changed"):
        query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())


def test_generic_contraction_records_bound_the_exact_owned_k_image():
    a,b=rational_circle()
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),
        qualification=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24))
    assert r.stats.contractions==len(r.atlas.reductions)
    assert r.atlas.reductions
    for reduction in r.atlas.reductions:
        assert all(a<=c<=d<=b for (a,b),(c,d) in zip(reduction.original,reduction.residual))
        f=reduction.free_coordinate
        assert reduction.original[f]==reduction.residual[f]
        columns=[k for k in range(4) if k!=f]
        for k,(lo,hi) in zip(columns,reduction.k_image):
            a,b=reduction.original[k]
            c,d=reduction.residual[k]
            assert c<=max(a,lo)<=min(b,hi)<=d


def test_exact_coefficient_preconditioning_preserves_f_and_all_four_derivatives():
    from anygeometry.nurbs_surface_intersection._certificate import _Pair
    a,b=rational_circle()
    pair=_Pair(a,b,((Q(0),Q(1)),)*4).anchored(((Q(1,8),Q(3,4)),)*4)
    matrix=((Q(1),Q(2),Q(-1)),(Q(0),Q(1),Q(3)),(Q(2),Q(0),Q(1)))
    conditioned=pair.preconditioned(matrix)
    point=((Q(1,4),Q(1,4)),(Q(1,2),Q(1,2)),(Q(1,3),Q(1,3)),(Q(2,3),Q(2,3)))
    f,j=pair.value_and_jacobian(pair.enclose(point))
    cf,cj=conditioned.value_and_jacobian(conditioned.enclose(point))
    for i in range(3):
        expected=sum((matrix[i][k]*f[k].lo for k in range(3)),Q(0))
        assert cf[i].lo==cf[i].hi==expected
        for col in range(4):
            expected=sum((matrix[i][k]*j[k][col].lo for k in range(3)),Q(0))
            assert cj[i][col].lo==cj[i][col].hi==expected


def test_graph_preparation_polls_before_any_witness_work():
    from anygeometry.nurbs_surface_intersection._continuation import Graphs
    from anygeometry.nurbs_surface_intersection._source import Limits,WorkCounters
    class Stop(Exception):
        pass
    def stop():
        raise Stop
    with pytest.raises(Stop):
        Graphs(None,None,(object(),),Limits(),WorkCounters(),stop)


def seam_planes(*,discontinuous=False):
    """Exact analytic x-directed line, only first surface has one u seam."""
    kv=(0.,0.,1.,1.)
    ku=(0.,0.,.5,.5,1.,1.) if discontinuous else (0.,0.,.5,1.,1.)
    xs=(0.,.5,.75,1.) if discontinuous else (0.,.5,1.)
    first=NurbsSurface(1,1,ku,kv,tuple(tuple((x,y,0.) for y in (-1.,1.)) for x in xs))
    second=NurbsSurface(1,1,kv,kv,tuple(tuple((x,0.,z) for z in (-1.,1.)) for x in (-1.,2.)))
    return first,second


@pytest.mark.parametrize("mode",["filtered","exact"])
def test_single_exact_homogeneous_seam_has_one_ordered_open_component(mode):
    a,b=seam_planes()
    limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24,interval_mode=mode)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED,r.reason
    assert r.algorithm=="generic-atlas" and len(r.components)==1 and not r.components[0].closed
    assert len({r.atlas.witnesses[arc.witness].spans[0][0] for arc in r.components[0].arcs})==2
    for left,right in zip(r.components[0].arcs,r.components[0].arcs[1:]):
        assert left.second_root==right.first_root
    points=evaluate_arcs(r,((0,0,Q(0)),(0,len(r.components[0].arcs)-1,Q(1))),
                        first=a,second=b,tolerances=TolerancePolicy(),qualification=limits)
    assert points[0].world_lo[0]<=0<=points[0].world_hi[0]
    assert points[1].world_lo[0]<=1<=points[1].world_hi[0]


def test_discontinuous_seam_is_refused_by_the_source_constructor():
    with pytest.raises(GeometryError):
        seam_planes(discontinuous=True)


def test_invalid_interval_mode_is_rejected():
    a,b=seam_planes()
    with pytest.raises(GeometryError,match="interval_mode"):
        query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),
            qualification=SimpleNamespace(interval_mode="nearest"))


def test_generic_multispan_hull_exclusion_reports_measured_pruning():
    a,b=seam_planes()
    controls=tuple(tuple((x,y,z+10.) for x,y,z in row) for row in b.controls)
    b=NurbsSurface(b.degree_u,b.degree_v,b.knots_u,b.knots_v,controls,b.weights,b.u_range,b.v_range)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy())
    assert r.status is IntersectionStatus.DISJOINT and r.cover_complete
    assert r.algorithm=="generic-cover" and r.structural is None
    assert r.stats.excluded_pairs>0
    assert r.stats.prune_seconds>0
    assert 0<=r.stats.exclusion_seconds<=r.stats.prune_seconds<=r.stats.total_seconds
    assert r.stats.prepare_seconds>=0 and r.stats.certification_seconds>=0


def test_generic_batch_rejects_duck_result_and_sources():
    a,b=rational_circle()
    fake=SimpleNamespace(status=IntersectionStatus.CERTIFIED,algorithm="generic-atlas",
                         first_source=SimpleNamespace(matches=lambda surface:True))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError,match="record types"):
        evaluate_arcs(fake,((0,0,Q(0)),),first=a,second=b,tolerances=TolerancePolicy())


def test_generic_batch_lazy_iterator_cannot_replace_validated_descriptors():
    a,b=seam_planes()
    limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED,r.reason
    def requests():
        # Runs after canonical validation; caller-owned frozen records can be
        # forcibly mutated, but the evaluator must use its own trusted records.
        object.__setattr__(r,"roots",())
        object.__setattr__(r,"components",())
        object.__setattr__(r,"atlas",None)
        yield 0,0,Q(1,2)
    points=evaluate_arcs(r,requests(),first=a,second=b,tolerances=TolerancePolicy(),qualification=limits)
    assert len(points)==1 and points[0].world_diameter<=TolerancePolicy().length


def test_generic_witness_cap_retains_unconsumed_frontier():
    a,b=rational_circle()
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),
        qualification=SimpleNamespace(max_boxes_per_pair=1000,max_trace_segments=1,max_subdivision_depth=28))
    assert len(r.atlas.witnesses)<=1
    assert r.atlas.covered==len(r.atlas.witnesses)
    if "witness limit" in r.reason:
        assert r.atlas.frontier_remaining>=1 and not r.cover_complete


def test_generic_records_rehashed_tamper_cannot_create_authority():
    a,b=seam_planes()
    limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED,r.reason
    from anygeometry.nurbs_surface_intersection._records import digest_of
    root=dataclasses.replace(r.roots[0],value=r.roots[0].value+Q(1,64))
    roots=(root,)+r.roots[1:]
    fake=dataclasses.replace(r,roots=roots,descriptor_digest=digest_of(r.atlas.witnesses,roots,(),r.components))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(fake,0,0,Q(0),first=a,second=b,tolerances=TolerancePolicy(),qualification=limits)
