"""CH3e verifier performance-path contract regressions; public fixture only."""
from fractions import Fraction as F
import pytest
from test_certified_hydrostatics_ch3_verify import honest
from anygeometry.certified_hydrostatics import CertifiedHydrostaticsRefusal,verify_certified_hydrostatics


def test_public_qualified_model_does_not_repeat_topology_validation(honest,monkeypatch):
    from anygeometry.serialization import to_dict
    m,f,q,r,e=honest;to_dict(m)
    def tripwire(*args,**kwargs):raise AssertionError('unchanged qualified model redundantly validated')
    monkeypatch.setattr(type(m),'validate_topology',tripwire)
    value=verify_certified_hydrostatics(m,f,q,r,e)
    assert value.occupancy=='PARTIAL' and set(value.enclosures)=={'V','Jx','Jy','Jz','A','Mu','Mv','Iuu','Ivv','Iuv'}


@pytest.mark.parametrize('attack',('persisted','lookup','incidence'))
def test_same_revision_mutation_cannot_reuse_qualification(honest,attack):
    import numpy as np
    from anygeometry.serialization import to_dict
    m,f,q,r,e=honest;to_dict(m);revision=m.revision;keys=sorted(m.vertices);a,b=keys[:2]
    if attack=='persisted':
        record=m.vertices[a];saved=record.position;object.__setattr__(record,'position',np.array([float(saved[0])+0.125,float(saved[1]),float(saved[2])]))
    elif attack=='lookup':m._vertices[a],m._vertices[b]=m._vertices[b],m._vertices[a]
    else:
        key=next(iter(m.edges));saved=m._edge_coedges.get(key);m._edge_coedges[key]={999999999}
    try:
        assert m.revision==revision
        with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_certified_hydrostatics(m,f,q,r,e)
        assert caught.value.phase=='verify'
    finally:
        if attack=='persisted':object.__setattr__(record,'position',saved)
        elif attack=='lookup':m._vertices[a],m._vertices[b]=m._vertices[b],m._vertices[a]
        elif saved is None:m._edge_coedges.pop(key,None)
        else:m._edge_coedges[key]=saved


def test_bernstein_affine_transform_matches_exact_polynomial():
    from math import comb
    from anygeometry.certified_hydrostatics._ch3_verify_math import bernstein,evaluate
    for degree in range(8):
        polynomial=tuple(F((-1)**i*(i+1),i+2) for i in range(degree+1));lo,hi=F(-2,3),F(7,5);coefficients=bernstein(polynomial,lo,hi)
        for t in (F(0),F(1,7),F(1,2),F(1)):
            reconstructed=sum((coefficient*comb(degree,i)*t**i*(1-t)**(degree-i) for i,coefficient in enumerate(coefficients)),F(0))
            assert reconstructed==evaluate(polynomial,lo+(hi-lo)*t)


def test_rational_constant_and_zero_paths_preserve_exact_values():
    from anygeometry.certified_hydrostatics._ch3_verify_math import Rational
    r=Rational((-1,0,1),(-1,1));s=Rational((3,),(-2,0,1))
    for u in (F(-2),F(0),F(1,2),F(2)):
        assert (r*F(-7,3)).at(u)==(u+1)*F(-7,3)
        assert (r/F(-7,3)).at(u)==(u+1)/F(-7,3)
        assert (s+Rational(0)).at(u)==3/(u*u-2)
        assert (Rational(0)*s).at(u)==0


def test_incidence_mutation_during_verification_refuses(honest,monkeypatch):
    from anygeometry.certified_hydrostatics._ch3_verify_geometry import Computation
    m,f,q,r,e=honest;key=next(iter(m.edges));saved=m._edge_coedges.get(key);original=Computation.run
    def mutate(self):
        result=original(self)
        if self.wanted=={'V'}:m._edge_coedges[key]={999999999}
        return result
    monkeypatch.setattr(Computation,'run',mutate)
    try:
        with pytest.raises(CertifiedHydrostaticsRefusal) as caught:verify_certified_hydrostatics(m,f,q,r,e)
        assert caught.value.category.value=='CONCURRENT_MUTATION' and caught.value.phase=='verify'
    finally:
        if saved is None:m._edge_coedges.pop(key,None)
        else:m._edge_coedges[key]=saved


def test_taylor_subdivision_sums_stay_inside_default_bit_cap():
    from anygeometry.certified_hydrostatics import CertifiedHydrostaticsPolicy,WorkBudget
    from anygeometry.certified_hydrostatics._work import Meter
    from anygeometry.certified_hydrostatics._ch3_verify_math import Arithmetic,Rational
    ar=Arithmetic(Meter(WorkBudget(None,65536)),CertifiedHydrostaticsPolicy(convention_version='2').as_dict());q=2**3000;tolerance=F(1,1024*q)
    value=ar.integrate(Rational((1,),(q+1,q)),F(0),F(1),tolerance)
    # Integral is log((2q+1)/(q+1))/q. Independently bound log2 via
    # 2*atanh(1/3) and bound its argument perturbation by1/q.
    terms=12;lower=2*sum((F(1,(2*k+1)*3**(2*k+1)) for k in range(terms)),F(0));tail=F(9,4*(2*terms+1)*3**(2*terms+1))
    oracle_lo=(lower-F(1,q))/q;oracle_hi=(lower+tail)/q
    assert value.width<=tolerance
    assert value.lo<=oracle_hi and oracle_lo<=value.hi
