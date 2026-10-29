"""CH3d independent verifier trust-boundary regressions.

Fixtures exercise only public provider APIs. No producer algorithms are read or
copied into the verifier or its tests.
"""
import json
from dataclasses import replace
from fractions import Fraction as F
import pytest
from anygeometry.b1_example import build_b1_hull
from anygeometry.certified_hydrostatics import (
    CertifiedHydrostaticsPolicy,CertifiedHydrostaticsRefusal,HalfSpaceQuery,
    WorkBudget,prepare_certified_hydrostatics,query_certified_hydrostatics,
)
from anygeometry.certified_hydrostatics import verify_certified_hydrostatics as verify_curved
from anygeometry.certified_hydrostatics._work import Meter

@pytest.fixture(scope='module')
def honest():
    model,faces=build_b1_hull();faces=tuple(faces.values())
    policy=CertifiedHydrostaticsPolicy(convention_version='2')
    prepared=prepare_certified_hydrostatics(model,faces,policy=policy)
    query=HalfSpaceQuery((0,0,1),2,forms=('binary64','decimal'))
    result=query_certified_hydrostatics(prepared,query)
    expected={'artifact':prepared.implementation,'convention_version':'2','units':'m','policy':policy}
    return model,faces,query,result,expected

def test_left_taylor_rational_oracle():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Rational
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict())
    # Integral 1/(1+u)^2 on [0,1] = 1/2, a rational oracle independent of provider.
    value=ar.integrate(Rational((1,),(1,2,1)),F(0),F(1),F(1,10**12))
    assert value.lo<=F(1,2)<=value.hi
    assert value.width<=F(1,10**12)

def test_sturm_all_real_ordinals():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict())
    roots=ar.roots((-2,0,1),-2,2)
    assert len(roots)==2
    assert [r.identity()['ordinal'] for r in roots]==[0,1]

def test_closed_isolator_excludes_other_endpoint_roots():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,compare
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict())
    roots=ar.roots((0,-2,0,1),F(0),F(2))
    assert len(roots)==2
    assert compare(roots[1],F(0))>0
    assert [r.identity()['ordinal'] for r in roots]==[1,2]

def test_rotated_corner_contact_is_invalid_loop():
    from types import SimpleNamespace as NS
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import Reconstruction
    points=[(F(1,8),F(1,8)),(F(7,8),F(1,8)),(F(5,8),F(5,8)),(F(1,2),F(1,8)),(F(1,4),F(5,8))]
    rec=object.__new__(Reconstruction);rec.ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict());rec.faces={1:NS(loop=tuple(NS(edge=i,forward=True) for i in range(5)))};rec.edges={i:NS(start=i,end=(i+1)%5) for i in range(5)};rec.arcs={}
    for i,(u,v) in enumerate(points):
        x,y=points[(i+1)%5];slope=(y-v)/(x-u);rec.arcs[i]={'graph':((v-slope*u,slope),(F(1),))}
    rec.arc_ends=lambda eid,forward=True:(points[eid][0],points[(eid+1)%5][0])
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:rec.simple_curved_loop(1)
    assert caught.value.category.value=='INVALID_LOOP'

def test_fc_corner_ray_hand_oracle():
    from ch3_solids import solid_fc
    from test_certified_hydrostatics_ch3_admission import build_face,surface_fc,planes_fc
    m,faces=solid_fc(build_face,surface_fc,planes_fc);policy=CertifiedHydrostaticsPolicy(convention_version='2');p=prepare_certified_hydrostatics(m,faces,policy=policy);q=HalfSpaceQuery((0,1,1),F(11,8),forms=('exact','enclosure'));r=query_certified_hydrostatics(p,q);expected={'artifact':p.implementation,'convention_version':'2','units':'m','policy':policy}
    v=verify_curved(m,faces,q,r,expected)
    # Diamond prism: width_x(y)=2(y-1/4),2(3/4-y), clipped height
    # min(1,11/8-y). Integrating these polynomials gives these hand oracles.
    for name,value in {'V':F(167,1536),'Jx':F(167,3072),'Jy':F(1307,24576),'Jz':F(2351,49152)}.items():assert v.enclosures[name][0]<=value<=v.enclosures[name][1]

def test_present_section_cannot_omit_quantities(honest):
    m,f,q,r,e=honest;bad=replace(r,sections=(replace(r.sections[0],quantities=None),));doc=json.loads(r.proof);doc['result']=bad.claim();bad=replace(bad,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal):verify_curved(m,f,q,bad,e)

def test_decimal_exponent_guard_before_conversion(honest,monkeypatch):
    import anygeometry.certified_hydrostatics._ch3_verify as module
    m,f,q,r,e=honest;volume=replace(r.volume,decimal=('1e-999999999999',r.volume.decimal[1],r.volume.decimal[2]));bad=replace(r,volume=volume);doc=json.loads(r.proof);doc['result']=bad.claim();bad=replace(bad,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    original=module.Decimal
    def guarded(value):
        if value=='1e-999999999999':raise AssertionError('unsafe Decimal conversion reached')
        return original(value)
    monkeypatch.setattr(module,'Decimal',guarded)
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,bad,e)
    assert caught.value.phase=='verify'

def test_small_normal_honest_claim_not_producer_failure(honest):
    m,f,_,_,e=honest;scale=F(1,2**20);p=prepare_certified_hydrostatics(m,f,policy=e['policy']);q=HalfSpaceQuery((0,0,scale),2*scale,forms=('enclosure',));r=query_certified_hydrostatics(p,q)
    try:v=verify_curved(m,f,q,r,e)
    except CertifiedHydrostaticsRefusal as exc:
        assert exc.category.value=='BUDGET_EXHAUSTED' and exc.phase=='verify'
    else:assert v.occupancy=='PARTIAL'

def test_honest_b1_partial(honest):
    from anygeometry.certified_hydrostatics._ch3_verify_math import _active
    m,f,q,r,e=honest;prior=_active.get();v=verify_curved(m,f,q,r,e)
    assert _active.get() is prior
    assert v.occupancy=='PARTIAL'
    assert v.volume.enclosure==r.volume.enclosure
    assert set(v.enclosures)>={'V','Jx','Jy','Jz','A','Mu','Mv','Iuu','Ivv','Iuv'}

def test_zero_verify_budget(honest):
    m,f,q,r,e=honest
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,r,e,budget=WorkBudget(0,65536))
    assert caught.value.category.value=='BUDGET_EXHAUSTED'
    assert caught.value.phase=='verify'

def test_cancellation(honest):
    m,f,q,r,e=honest
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,r,e,cancellation_check=lambda:True)
    assert caught.value.category.value=='CANCELLED'
    assert caught.value.phase=='verify'

@pytest.mark.parametrize('attack',('model','snapshot','query','artifact','reversed','accuracy','exact','response'))
def test_tampered_proof(honest,attack):
    m,f,q,r,e=honest;doc=json.loads(r.proof)
    if attack=='model':doc['model']['face_ids']=doc['model']['face_ids'][:3]
    elif attack=='snapshot':doc['model']['snapshot_digest']='0'*64
    elif attack=='query':doc['query']['offset']=[3,1]
    elif attack=='artifact':doc['artifact']['files'][0][1]='0'*64
    elif attack=='reversed':doc['result']['volume']['enclosure']['lo'],doc['result']['volume']['enclosure']['hi']=doc['result']['volume']['enclosure']['hi'],doc['result']['volume']['enclosure']['lo']
    elif attack=='accuracy':doc['result']['volume']['accuracy_met']=True
    elif attack=='exact':doc['result']['volume']['exact']={'numerator':[1,1],'norm_squared':None}
    elif attack=='response':doc['result']['generic']=False
    bad=replace(r,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal):verify_curved(m,f,q,bad,e)

def test_execution_changes_do_not_change_acceptance(honest):
    m,f,q,r,e=honest;doc=json.loads(r.proof)
    doc['result']['execution']={'integrals':123,'taylor_pieces':456,'point_bits':789,'core':{'V':[[0,1],[999,1]]}}
    bad=replace(r,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    assert verify_curved(m,f,q,bad,e).occupancy=='PARTIAL'

@pytest.mark.parametrize('offset,occupancy',((-10,'EMPTY'),(10,'FULL')))
def test_honest_b1_endpoint_states(honest,offset,occupancy):
    m,f,_,_,e=honest;p=prepare_certified_hydrostatics(m,f,policy=e['policy']);q=HalfSpaceQuery((0,0,1),offset,forms=('binary64','decimal'));r=query_certified_hydrostatics(p,q);v=verify_curved(m,f,q,r,e)
    assert v.occupancy==occupancy
    assert v.sections[0].quantities is None
    assert v.sections[0].topology is None
    assert set(v.enclosures)=={'V','Jx','Jy','Jz'}
    assert v.exact['sides']['plane']=={'occupancy':occupancy,'section':'ABSENT','quantities':None,'topology':None}
    assert v.exact['contacts']=={'vertices':(),'edges':(),'faces':()}
    if occupancy=='EMPTY':assert v.exact['V']==0 and v.exact['J']==(0,0,0)
    else:assert v.enclosures['V'][0]>0

def test_coherent_shifted_claim(honest):
    import math
    from decimal import Decimal
    m,f,q,r,e=honest;lo,hi=r.volume.enclosure;width=hi-lo
    lo+=width;hi+=width;flo,fhi=float(lo),float(hi)
    if F(flo)>lo:flo=math.nextafter(flo,-math.inf)
    if F(fhi)<hi:fhi=math.nextafter(fhi,math.inf)
    grid=10**13;low=(lo*grid).__floor__();high=-(-hi*grid).__floor__();dec=(str(Decimal(low)/Decimal(grid)),str(Decimal(high)/Decimal(grid)),r.volume.decimal[2])
    volume=replace(r.volume,enclosure=(lo,hi),binary64=(flo,fhi),decimal=dec)
    bad=replace(r,volume=volume);doc=json.loads(r.proof);doc['result']=bad.claim();bad=replace(bad,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,bad,e)
    assert caught.value.sub_reason=='enclosure'

def test_snapshot_hint_mutation(honest):
    import numpy as np
    m,f,q,r,e=honest;copy=m.clone();vertex=next(iter(copy.vertices));p=copy.vertices[vertex].position
    # Deliberately bypass immutable entity guards to model hostile/stale input;
    # the public point mutator correctly rejects this inconsistent edit itself.
    object.__setattr__(copy.vertices[vertex],'position',np.array([float(p[0])+1e-4,float(p[1]),float(p[2])]))
    with pytest.raises(CertifiedHydrostaticsRefusal):verify_curved(copy,f,q,r,e)

@pytest.mark.parametrize('form',('decimal_width','binary64_endpoint'))
def test_coherent_public_rounding_tamper(honest,form):
    m,f,q,r,e=honest
    if form=='decimal_width':
        lo,hi,_=r.volume.decimal;volume=replace(r.volume,decimal=(lo,hi,F(1,10**30)))
    else:
        lo,hi=r.volume.binary64;volume=replace(r.volume,binary64=(lo+1,hi+1))
    bad=replace(r,volume=volume);doc=json.loads(r.proof);doc['result']=bad.claim();bad=replace(bad,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal):verify_curved(m,f,q,bad,e)

@pytest.mark.parametrize('shift',(.25,2.0))
def test_overlapping_closed_components_refuse_after_coherent_rebinding(honest,shift):
    import hashlib
    from anygeometry.serialization import to_dict
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import Reconstruction
    m,_,q,r,e=honest;overlap=m.clone();overlap.insert_model(m,matrix=((1,0,0,shift),(0,1,0,0),(0,0,1,0),(0,0,0,1)));faces=tuple(sorted(overlap.faces))
    rec=Reconstruction(overlap,faces,Arithmetic(Meter(WorkBudget.unlimited()),e['policy'].as_dict()))
    def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    document=to_dict(overlap);document.pop('checksum',None);proof=json.loads(r.proof)
    proof['model']={'model_id':str(overlap.model_id),'revision':overlap.revision,'document_checksum':sha(document),'face_ids':list(faces),'snapshot_digest':sha(rec.snapshot)}
    bad=replace(r,proof=json.dumps(proof,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(overlap,faces,q,bad,e)
    assert caught.value.category.value in ('EMBEDDING_UNDECIDED','SELF_INTERSECTING')

@pytest.mark.parametrize('pair',('planar_planar','curved_planar'))
def test_independent_pair_predicates_reject_crossing_components(honest,pair):
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import Reconstruction
    from anygeometry.certified_hydrostatics._ch3_verify_embedding import Embedding
    m,_,_,_,e=honest;overlap=m.clone();original=set(overlap.faces);overlap.insert_model(m,matrix=((1,0,0,.25),(0,1,0,0),(0,0,1,1),(0,0,0,1)));faces=tuple(sorted(overlap.faces))
    new_aft=next(fid for fid,f in overlap.faces.items() if fid not in original and hasattr(f.surface,'origin') and float(f.surface.origin[0])==1.75)
    rec=Reconstruction(overlap,faces,Arithmetic(Meter(WorkBudget.unlimited()),e['policy'].as_dict()));embedding=Embedding(rec)
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:
        if pair=='planar_planar':embedding.planar_pair(3,new_aft)
        else:embedding.curved_planar(1,new_aft)
    assert caught.value.category.value in ('EMBEDDING_UNDECIDED','SELF_INTERSECTING')

def test_two_crossings_on_one_planar_arc_use_previous_breakpoint(honest):
    from functools import cmp_to_key
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Rational,compare,dot
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import Reconstruction,middle
    m,faces,_,_,expected=honest;ar=Arithmetic(Meter(WorkBudget.unlimited()),expected['policy'].as_dict());rec=Reconstruction(m,faces,ar)
    deck=next(fid for fid,f in rec.faces.items() if fid not in rec.supports and float(f.surface.origin[2])==3.)
    oe=next(oe for oe in rec.faces[deck].loop if oe.edge in rec.arcs);eid=oe.edge;c=rec.arcs[eid]['curve'];start,end=rec.arc_ends(eid,oe.forward)
    # A supporting functional with its maximum near the traversal END makes
    # two crossings for which midpoint(edge_start,second_root) is on the wrong
    # side. Select only dyadic input numbers; no tolerance is relaxed.
    t=F(round(float((start.lo+start.hi)/2)*4096/10+float((end.lo+end.hi)/2)*4096*9/10),4096)
    k=F(float(-c[1].derivative().at(t)/c[0].derivative().at(t)));function=Rational(k)*c[0]+c[1]
    endpoint=max(ar.range(function,r.lo,r.hi).hi for r in (start,end));peak=function.at(t);assert peak>endpoint
    offset=F(float((peak+endpoint)/2));cut=function-Rational(offset);lo,hi=(start,end) if compare(start,end)<0 else (end,start)
    roots=[r for r in ar.roots(cut.n,lo.lo,hi.hi) if compare(r,lo)>0 and compare(r,hi)<0];assert len(roots)==2
    ordered=sorted(roots,key=cmp_to_key(compare),reverse=compare(start,end)>0)
    old_sign=cut.at(middle(start,ordered[1]))<0;correct_sign=cut.at(middle(ordered[0],ordered[1]))<0;assert old_sign!=correct_sign
    query=HalfSpaceQuery((k,1,0),offset,waterplane_axes=((-1,k,0),(0,0,1)),absolute_accuracy={name:1e-8 for name in ('V','J','A','Mu','Mv','Iuu','Ivv','Iuv')},forms=('enclosure',))
    prepared=prepare_certified_hydrostatics(m,faces,policy=expected['policy']);result=query_certified_hydrostatics(prepared,query)
    assert any(loop['identity'].count(eid)==2 for loop in result.claim()['sections']['plane']['topology']['loops'])
    verified=verify_curved(m,faces,query,result,expected)
    assert [eid,0] in result.claim()['sections']['plane']['topology']['cut_points']
    assert [eid,1] in result.claim()['sections']['plane']['topology']['cut_points']
    assert verified.occupancy=='PARTIAL'

@pytest.mark.parametrize('case',('fa_full','fa_partial','fa_inclined','fa_ruling_outside','fc_vertical','fc_inclined','fc_full'))
def test_honest_curved_fixture_matrix(case):
    from ch3_solids import solid_fa,solid_fc
    from test_certified_hydrostatics_ch3_admission import build_face,surface_fa,planes_fa,surface_fc,planes_fc
    if case.startswith('fa'):m,ids=solid_fa(build_face,surface_fa,planes_fa)
    else:m,ids=solid_fc(build_face,surface_fc,planes_fc)
    ids=tuple(ids.values()) if isinstance(ids,dict) else tuple(ids)
    n,d={'fa_full':((0,0,1),F(1)),'fa_partial':((0,0,1),F(1,2)),'fa_inclined':((0,1,1),F(7,8)),'fa_ruling_outside':((1,0,0),F(15,16)),'fc_vertical':((0,0,1),F(1,2)),'fc_inclined':((1,2,3),F(9,4)),'fc_full':((0,0,1),F(2))}[case]
    axes=((0,1,0),(0,0,1)) if case=='fa_ruling_outside' else ((1,0,0),(0,1,0))
    policy=CertifiedHydrostaticsPolicy(convention_version='2');p=prepare_certified_hydrostatics(m,ids,policy=policy);q=HalfSpaceQuery(n,d,waterplane_axes=axes,absolute_accuracy={'V':1e-9,'A':1e-9},forms=('exact','enclosure'));r=query_certified_hydrostatics(p,q);expected={'artifact':p.implementation,'convention_version':'2','units':'m','policy':policy}
    verified=verify_curved(m,ids,q,r,expected)
    assert verified.occupancy==r.occupancy
    side=verified.exact['sides']['plane'];assert side['section']==r.sections[0].section
    if side['section']=='ABSENT':assert side['quantities'] is None
    else:
        assert isinstance(side['quantities'],dict)
        for name,(numerator,norm) in side['quantities'].items():
            item=next(x for x in r.sections[0].quantities if x.name==name)
            if item.numerator is not None:assert (numerator,norm)==(item.numerator,item.norm_squared)


def test_certified_crossing_order_refines_overlapping_boxes():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Interval
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import certified_crossing_order
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict())
    # Values 1 and 1+2^-200 have misleading initial interval midpoints.
    gap=F(1,2**200)
    def a():return (Interval(1,1+4*gap) if getattr(ar,'point_bits',140)<200 else Interval(1),)*3
    def b():return (Interval(1-gap,1+gap) if getattr(ar,'point_bits',140)<200 else Interval(1+gap),)*3
    entries=[(a(),(1,0),True),(b(),(2,0),False)]
    ordered=certified_crossing_order(entries,0,ar,{(1,0):a,(2,0):b})
    assert [x[1] for x in ordered]==[(1,0),(2,0)]
    assert ordered[0][0][0].hi<ordered[1][0][0].lo


def test_certified_crossing_order_refuses_unresolved_boxes():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Interval
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import certified_crossing_order
    policy=CertifiedHydrostaticsPolicy(convention_version='2').as_dict();policy['max_isolation_bits']=160
    ar=Arithmetic(Meter(WorkBudget.unlimited()),policy);getter=lambda:(Interval(0,1),)*3
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:certified_crossing_order([(getter(),(1,0),True),(getter(),(2,0),False)],0,ar,{(1,0):getter,(2,0):getter})
    assert caught.value.category.value=='BUDGET_EXHAUSTED' and caught.value.phase=='verify'


def test_half_hull_proof_cannot_substitute_full_model(honest):
    from _b1_shared_models import b1_half_hull_shared
    m,f,q,_,e=honest;half,faces=b1_half_hull_shared();ids=tuple(faces.values());p=prepare_certified_hydrostatics(half,ids,policy=e['policy']);r=query_certified_hydrostatics(p,q)
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,r,e)
    assert caught.value.phase=='verify' and caught.value.category.value=='VERIFICATION_FAILED'


def test_mid_integration_verify_budget_refusal(honest,monkeypatch):
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic
    m,f,q,r,e=honest;original=Arithmetic.integrate;entered=[]
    def bounded(self,*args):
        if not entered:
            entered.append(True)
            # Replace the remaining allowance at the actual integration boundary.
            self.meter.budget=WorkBudget(3,65536)
        return original(self,*args)
    monkeypatch.setattr(Arithmetic,'integrate',bounded)
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,r,e)
    assert entered and caught.value.category.value=='BUDGET_EXHAUSTED' and caught.value.phase=='verify'


def test_planar_pairing_uses_certified_crossing_order():
    from types import SimpleNamespace as NS
    from anygeometry import Plane
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Interval,Rational
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import Computation
    policy=CertifiedHydrostaticsPolicy(convention_version='2').as_dict();ar=Arithmetic(Meter(WorkBudget.unlimited()),policy);gap=F(1,2**200)
    # Four crossings on the section line. The middle two safe, overlapping
    # boxes have reversed midpoint order; actual coordinates are exact here.
    xs=[F(1),F(2),2+gap,F(3)];loops=tuple(NS(edge=i,forward=i%2==0) for i in range(4));surface=Plane((0,0,0),(1,0,0),(0,1,0))
    rec=NS(ar=ar,faces={1:NS(surface=surface,loop=loops,holes=())},edges={i:NS(start=i,end=i+1) for i in range(4)},arcs={i:{'curve':(Rational(x),Rational((0,1)),Rational(0))} for i,x in enumerate(xs)},arc_ends=lambda eid,forward=True:((F(-1),F(1)) if forward else (F(1),F(-1))))
    q=HalfSpaceQuery((0,1,0),0,waterplane_axes=((1,0,0),(0,0,-1)));comp=Computation(rec,q,F(1,10**12));comp.plane_values=lambda *args:None;comp.segment_values=lambda *args:None
    comp.crossings={(i,0):(Interval(x),Interval(0),Interval(0)) for i,x in enumerate(xs)}
    comp.crossings[(1,0)]=(Interval(2,2+4*gap),Interval(0),Interval(0));comp.crossings[(2,0)]=(Interval(2-gap,2+gap),Interval(0),Interval(0))
    comp.planar(1,{i:[F(0)] for i in range(4)})
    assert {frozenset(piece[-2:]) for piece in comp.cap}=={frozenset(((0,0),(1,0))),frozenset(((2,0),(3,0)))}
    # Direction distinguishes a reversed pair even when the paired set agrees.
    assert comp.cap[0][-2:]==((1,0),(0,0))


def test_absent_section_cannot_publish_quantities(honest):
    m,f,_,r,e=honest;p=prepare_certified_hydrostatics(m,f,policy=e['policy']);q=HalfSpaceQuery((0,0,1),4,forms=('enclosure',));full=query_certified_hydrostatics(p,q)
    bad=replace(full,sections=(replace(full.sections[0],quantities=r.sections[0].quantities),));doc=json.loads(full.proof);doc['result']=bad.claim();bad=replace(bad,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,bad,e)
    assert caught.value.category.value=='MALFORMED_PROOF'


def test_coherent_claim_width_exceeds_accuracy(honest):
    m,f,q,r,e=honest;accuracy=F(1,10**12);q=HalfSpaceQuery(q.normal,q.offset,absolute_accuracy={'V':accuracy},forms=q.forms)
    lo,hi=r.volume.enclosure;volume=replace(r.volume,accuracy=accuracy,accuracy_met=True,enclosure=(lo,lo+2*accuracy));bad=replace(r,query=q,volume=volume);doc=json.loads(r.proof);doc['query']=q.as_dict();doc['result']=bad.claim();bad=replace(bad,proof=json.dumps(doc,sort_keys=True,separators=(',',':')).encode())
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_curved(m,f,q,bad,e)
    assert caught.value.category.value=='VERIFICATION_FAILED' and caught.value.sub_reason=='accuracy flag'


def test_bernstein_stored_coefficients_obey_bit_cap():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,bernstein
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict())
    endpoint=F(2**4096-1,2**4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:bernstein((F(0),)*17+(F(1),),endpoint,F(1))
    assert caught.value.category.value=='BIT_LENGTH_EXCEEDED'


def test_overlapping_independent_box_is_inconclusive():
    from types import SimpleNamespace as NS
    from anygeometry.certified_hydrostatics._ch3_verify import check_quantity
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Interval
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict());q=HalfSpaceQuery((0,0,1),0,absolute_accuracy={'V':1},forms=('enclosure',))
    claim={'enclosure':{'lo':[0,1],'hi':[1,1]},'accuracy':[1,1],'accuracy_met':True};item=NS(name='V',enclosure=(F(0),F(1)))
    # True value can be0: width<=claim/3 alone cannot prove a false claim.
    with pytest.raises(CertifiedHydrostaticsRefusal) as caught:check_quantity('V',claim,item,Interval(F(-1,10),F(1,10)),q,ar,F(1))
    assert caught.value.category.value=='BUDGET_EXHAUSTED' and caught.value.phase=='verify'


def test_hint_enclosure_cannot_add_tolerance_slack():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Rational
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import point_interval,hint_binds
    ar=Arithmetic(Meter(WorkBudget.unlimited()),CertifiedHydrostaticsPolicy(convention_version='2').as_dict());root=ar.roots((-2,0,1),F(1),F(2))[0];root.refine(140)
    curve=(Rational((-root.lo*2**200,2**200)),Rational(0),Rational(0));image=point_interval(curve,root,ar);hint=(image[0].hi,F(0),F(0));delta=F(1,2**40)*(1+max(map(abs,hint)))
    assert all(x.lo-delta<=v<=x.hi+delta for x,v in zip(image,hint))
    assert not hint_binds(curve,root,hint,ar,delta)


def test_full_volume_sign_uncertainty_is_not_invalid_model():
    from anygeometry.certified_hydrostatics._ch3_verify import require_positive_full
    from anygeometry.certified_hydrostatics._ch3_verify_math import Interval
    require_positive_full(Interval(1,2))
    for bounds,category in (((-1,1),'BUDGET_EXHAUSTED'),((-2,-1),'VERIFICATION_FAILED'),((0,0),'VERIFICATION_FAILED')):
        with pytest.raises(CertifiedHydrostaticsRefusal) as caught:require_positive_full(Interval(*bounds))
        assert caught.value.category.value==category and caught.value.phase=='verify'
