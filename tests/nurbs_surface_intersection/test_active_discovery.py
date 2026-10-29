"""UNWIRED active discovery experiment: root owns all numerical execution."""
from dataclasses import replace
from fractions import Fraction as Q
from importlib import util,import_module
from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import (query_nurbs_surface_intersection,IntersectionStatus,
    NurbsSurfaceIntersectionCancelled)
from anygeometry.nurbs_surface_intersection._certificate import _Pair,certify_prepared
from anygeometry.nurbs_surface_intersection._intervals import Interval
from anygeometry.nurbs_surface_intersection._source import SourceDefinition,WorkCounters,Limits


@pytest.fixture(scope="module")
def candidate():
    # Package qualification keeps snapshot's relative imports bound to the
    # verified package; this does not replace its live atlas module.
    name="anygeometry.nurbs_surface_intersection._active_discovery_test_candidate"
    spec=util.spec_from_file_location(name,Path(__file__).with_name("_active_discovery_candidate.py"))
    module=util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLIPS=((Q(0),Q(1)),)*4


def _planes():
    knots=(0.,0.,1.,1.)
    a=NurbsSurface(1,1,knots,knots,tuple(tuple((x,y,0.) for y in (0.,1.)) for x in (0.,1.)))
    b=NurbsSurface(1,1,knots,knots,tuple(tuple((x,.5,z) for z in (-.5,.5)) for x in (-1.,2.)))
    return a,b


def _proof(pair,region,free=0):
    return certify_prepared(pair,CLIPS,free,[Interval(*region[k]) for k in range(4) if k!=free],
                            Interval(*region[free]),stats=WorkCounters())


def test_adjacent_shift_has_strict_whole_graph_and_exact_branch_overlap(candidate):
    a,b=_planes()
    region=((Q(1,4),Q(1,2)),(Q(1,4),Q(3,4)),(Q(1,4),Q(3,4)),(Q(1,4),Q(3,4)))
    pair=_Pair(a,b,CLIPS).anchored(region)
    old=_proof(pair,region)
    assert old.certified and old.regular and old.immersed
    proposal=candidate._adjacent_region(pair,region,0,CLIPS,old,1)
    assert proposal is not None and candidate._branch_overlap(region,proposal,0)
    assert proposal[0][0]>region[0][0] and proposal[0][1]>region[0][1]
    assert not all(c<=a<=b<=d for (a,b),(c,d) in zip(region,proposal))  # not old all-box growth
    new=_proof(pair.anchored(proposal),proposal)
    assert new.certified and new.regular and new.immersed
    assert all(c<=a<=b<=d for k,((a,b),(c,d)) in enumerate(zip(region,proposal)) if k!=0)
    # Analytic line: v=t=1/2 and s=(u+1)/3, with same root on positive Phi overlap.
    for u in (max(region[0][0],proposal[0][0]),min(region[0][1],proposal[0][1])):
        assert proposal[2][0]<(u+1)/3<proposal[2][1]


def test_branch_and_boundary_refusals_are_conservative(candidate):
    old=((Q(1,4),Q(1,2)),)*4
    touch=((Q(1,2),Q(3,4)),)+old[1:]
    assert not candidate._branch_overlap(old,touch,0)
    clipped=((Q(3,8),Q(5,8)),(Q(3,8),Q(1,2)),old[2],old[3])
    assert not candidate._branch_overlap(old,clipped,0)
    a,b=_planes()
    pair=_Pair(a,b,CLIPS).anchored(old)
    refused=SimpleNamespace(certified=False)
    assert candidate._adjacent_region(pair,old,0,CLIPS,refused,1) is None
    region=((Q(0),Q(1)),(Q(0),Q(1)),(Q(0),Q(1)),(Q(0),Q(1)))
    proof=_proof(pair,region)
    assert proof.certified
    assert candidate._adjacent_region(pair,region,0,CLIPS,proof,1) is None


def test_pending_success_without_visited_adoption_consumes_no_frontier(candidate,monkeypatch):
    # Control test: an identical, valid proof is used as the pending proposal
    # solely to distinguish cache registration from actual visited-cell coverage.
    monkeypatch.setattr(candidate,"_adjacent_region",lambda pair,region,*a:tuple(region))
    monkeypatch.setattr(candidate,"_free_order",lambda *args:[0,2,1,3])
    a,b=_planes()
    diagnostics={}
    stats=WorkCounters()
    report=candidate.run_atlas(SourceDefinition.capture(a),SourceDefinition.capture(b),TolerancePolicy(),
        replace(Limits(),max_boxes_per_pair=1,max_trace_segments=4),stats,lambda:None,
        active_diagnostics=diagnostics)
    assert report.complete_local_cover and report.covered==len(report.witnesses)==stats.certificates==1
    assert report.covered_cells==1 and not report.coverage_uses
    assert diagnostics["pending_created"]==diagnostics["pending_unused"]==1
    assert diagnostics["pending_adopted"]==0 and diagnostics["attempts"]<=1


def test_cancellation_before_pending_proof_cannot_publish_partial_cache(candidate,monkeypatch):
    armed=[False]
    def proposal(pair,region,*args):
        armed[0]=True
        return tuple(region)
    monkeypatch.setattr(candidate,"_adjacent_region",proposal)
    monkeypatch.setattr(candidate,"_free_order",lambda *args:[0,2,1,3])
    a,b=_planes()
    before=SourceDefinition.capture(a),SourceDefinition.capture(b)
    diagnostics={}
    def tick():
        if armed[0]:
            raise NurbsSurfaceIntersectionCancelled("before speculative proof")
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        candidate.run_atlas(*before,TolerancePolicy(),replace(Limits(),max_boxes_per_pair=1),WorkCounters(),tick,
                            active_diagnostics=diagnostics)
    assert diagnostics["pending_created"]==0
    assert before[0].matches(a) and before[1].matches(b)


def test_callback_cannot_reset_authoritative_attempt_cap(candidate,monkeypatch):
    monkeypatch.setattr(candidate,"_adjacent_region",lambda pair,region,*args:tuple(region))
    monkeypatch.setattr(candidate,"_free_order",lambda *args:[0,2,1,3])
    a,b=_planes()
    observer={}
    def tick():
        observer["attempts"]=0
        observer["pending_created"]="malformed observer value"
        observer["pending_checks"]=-1000000
    stats=WorkCounters()
    report=candidate.run_atlas(SourceDefinition.capture(a),SourceDefinition.capture(b),TolerancePolicy(),
        replace(Limits(),max_boxes_per_pair=1,max_trace_segments=4),stats,tick,active_diagnostics=observer)
    assert report.complete_local_cover and report.covered==1
    assert observer["attempts"]==1 and observer["pending_created"]==1
    assert observer["pending_checks"]>=0


def test_pending_adopts_only_real_visited_right_child(candidate,monkeypatch):
    a,b=_planes()
    before=SourceDefinition.capture(a),SourceDefinition.capture(b)
    monkeypatch.setattr(candidate,"_free_order",lambda *args:[0,2,1,3])
    original=candidate.certify_prepared
    forced=[False]
    def proof(pair,knots,free,unknown,phi,**kwargs):
        result=original(pair,knots,free,unknown,phi,**kwargs)
        if not forced[0] and not kwargs.get("for_root_filter",False) and phi.lo==0 and phi.hi==1:
            forced[0]=True
            # A conservative deliberate refusal forces the ordinary u split.
            return replace(result,certified=False,unknown_enclosure=())
        return result
    monkeypatch.setattr(candidate,"certify_prepared",proof)
    def adjacent(pair,region,free,clips,certificate,direction):
        if direction!=1 or region[free][1]>=1:
            return None
        proposal=list(CLIPS)
        proposal[free]=(Q(1,4),Q(1))
        proposal=tuple(proposal)
        assert candidate._branch_overlap(region,proposal,free)
        return proposal  # still passed to the REAL strict certificate
    monkeypatch.setattr(candidate,"_adjacent_region",adjacent)
    stats,diagnostics=WorkCounters(),{}
    report=candidate.run_atlas(*before,TolerancePolicy(),replace(Limits(),max_boxes_per_pair=32),
        stats,lambda:None,active_diagnostics=diagnostics)
    assert report.complete_local_cover and report.frontier_remaining==report.unresolved==0
    assert report.covered==len(report.witnesses)==stats.certificates==2
    assert report.covered_cells==2 and not report.coverage_uses
    assert diagnostics["pending_created"]==diagnostics["pending_adopted"]==1
    assert report.witnesses[1].cell[0]==(Q(1,2),Q(1))
    assert report.witnesses[1].region[0]==(Q(1,4),Q(1))
    assert before[0].matches(a) and before[1].matches(b)


def test_active_fixed64_public_generic_ab(candidate,monkeypatch,capsys):
    from test_chart_fallback import nested_rational_pair
    from anygeometry.nurbs_surface_intersection._atlas import _Surface
    query=import_module("anygeometry.nurbs_surface_intersection.query")
    original=query.run_atlas
    a,b=nested_rational_pair()
    before=SourceDefinition.capture(a),SourceDefinition.capture(b)
    reports=[]
    for active in (False,True):
        diagnostics={}
        def run(*args):
            return candidate.run_atlas(*args,active_diagnostics=diagnostics)
        with monkeypatch.context() as patch:
            patch.setattr(query,"derive_structural",lambda *args:(None,"explicit active generic experiment"))
            patch.setattr(query,"run_atlas",run if active else original)
            started=time.perf_counter()
            result=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),
                qualification=SimpleNamespace(max_boxes_per_pair=64,max_subdivision_depth=40))
            elapsed=time.perf_counter()-started
        assert result.status is IntersectionStatus.UNCLASSIFIED and not result.cover_complete
        assert result.stats.boxes_visited<=64
        atlas=result.atlas
        assert atlas.frontier_remaining+atlas.unresolved>0
        assert atlas.covered==len(atlas.witnesses)==result.stats.certificates
        assert atlas.covered_cells==atlas.covered+len(atlas.coverage_uses)
        assert before[0].matches(a) and before[1].matches(b)
        if active:
            assert diagnostics["attempts"]<=8
            assert diagnostics["pending_adopted"]<=diagnostics["pending_created"]<=diagnostics["attempts"]
            # Re-prove every ACTUALLY issued witness, never a speculative box.
            sa,sb=_Surface(before[0],WorkCounters()),_Surface(before[1],WorkCounters())
            for witness in atlas.witnesses:
                ia,ib=witness.spans
                knots=(sa.spans[0][ia[0]],sa.spans[1][ia[1]],sb.spans[0][ib[0]],sb.spans[1][ib[1]])
                pair=_Pair.from_pieces(sa.piece(*ia),sb.piece(*ib)).anchored(witness.region)
                proof=certify_prepared(pair,knots,witness.free_coordinate,
                    [Interval(*witness.region[k]) for k in range(4) if k!=witness.free_coordinate],
                    Interval(*witness.region[witness.free_coordinate]),stats=WorkCounters())
                assert proof.certified and proof.immersed and proof.regular
                assert all(c<=lo<=hi<=d for (lo,hi),(c,d) in zip(witness.cell,witness.region))
        reports.append(dict(active=active,seconds=elapsed,boxes=result.stats.boxes_visited,
            attempts=result.stats.certificate_attempts,graphs=atlas.covered,reuse=result.stats.reused_graph_cells,
            covered_cells=atlas.covered_cells,frontier=atlas.frontier_remaining,unresolved=atlas.unresolved,
            active_diagnostics=diagnostics))
    with capsys.disabled():
        print(json.dumps(reports,sort_keys=True))
