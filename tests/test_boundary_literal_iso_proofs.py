"""Exact iso-hull oracle and operation-local boundary proof authority."""
from fractions import Fraction as Q
from dataclasses import fields
import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.nurbs import NurbsCurve,NurbsSurface,CurveOnSurface
from anygeometry._boundary_curve_proofs import prepare_iso_hull,IsoHullUnavailable
from anygeometry.nurbs_intersection_curve import intersection_curve_operation
import anygeometry.prepared_material_curvilinear as proof


def iso_curve(rational=False,reversed=False,knotted=False):
    if knotted:
        knots = (0.,0.,0.,.5,1.,1.,1.)
        xs,zs = (-1.,-.5,.5,1.),(1.,0.,0.,1.)
    else:
        knots = (0.,0.,0.,1.,1.,1.)
        xs,zs = (-1.,0.,1.),(1.,-1.,1.)
    weights = tuple((w,w) for w in (1.,2.,1.)) if rational else None
    surface = NurbsSurface(2,1,knots,(0.,0.,1.,1.),
        tuple(tuple((x,y,z) for y in (-1.,1.)) for x,z in zip(xs,zs)),weights)
    controls = ((1.,.5),(0.,.5)) if reversed else ((0.,.5),(1.,.5))
    return CurveOnSurface(surface,NurbsCurve(1,(0.,0.,1.,1.),controls))


def ends(curve,a,b):
    return tuple(tuple(map(float,curve.exact_point(t))) for t in (a,b))


@pytest.mark.parametrize('rational',(False,True))
@pytest.mark.parametrize('reverse',(False,True))
def test_literal_hull_certifies_exact_polynomial_and_rational_reference(rational,reverse):
    curve = iso_curve(rational,reverse)
    # Analytic midpoint z is 0 for the polynomial and -1/3 for the
    # rational curve; both endpoints have z=1. This is an exact lower
    # bound on the true error, independent of the enclosure algorithm.
    true_error = Q(4,3) if rational else Q(1)
    with intersection_curve_operation():
        upper,_ = proof._world_chord(curve,Q(0),Q(1),ends(curve,Q(0),Q(1)),3.,lambda:None)
    assert true_error <= Q(upper) < Q(2000001,1000000)
    p = curve.exact_point(Q(1,2))
    assert p == (Q(0),Q(0),Q(-1,3) if rational else Q(0))
    # Exact analytic weighted numerator is 1-6t+6t² and denominator
    # 1+2t-2t²; the polynomial curve is (2t-1)². The continuous upper
    # proof comes from all positive-weight controls, not test samples.


def test_small_parabolic_iso_chord_uses_quadratic_hull_without_aabb_tiling():
    curve = iso_curve()
    a,b = Q(1,4),Q(9,32)
    # z=1-4t+4t², so the exact midpoint departure is (b-a)².
    # For controls (1,-1,1) Bernstein interpolation gives that same square.
    h = b-a
    with intersection_curve_operation():
        upper,pieces = proof._world_chord(curve,a,b,ends(curve,a,b),.0025,lambda:None)
    assert 0 < Q(upper) <= 3*h*h
    assert pieces == ((a,b),)


@pytest.mark.parametrize('reverse',(False,True))
def test_knotted_positive_hull_covers_joint_without_c2_assumption(reverse):
    curve = iso_curve(reversed=reverse,knotted=True)
    assert curve.exact_point(Q(1,2)) == (Q(0),Q(0),Q(0))
    with intersection_curve_operation():
        upper,_ = proof._world_chord(curve,Q(0),Q(1),ends(curve,Q(0),Q(1)),2.,lambda:None)
    assert Q(1) <= Q(upper) < Q(1000001,1000000)


@pytest.mark.parametrize('warm',(False,True))
@pytest.mark.parametrize('family',('outer','support','pcurve','also_on','unsupported'))
def test_iso_admission_precedes_even_warmed_lookup(warm,family):
    curve = iso_curve()
    with intersection_curve_operation():
        if warm:
            prepare_iso_hull(curve)
        if family == 'unsupported':
            candidate = object()
        elif family == 'also_on':
            candidate = iso_curve()
            object.__setattr__(candidate,'also_on',[])
        else:
            base = {'outer':CurveOnSurface,'support':NurbsSurface,'pcurve':NurbsCurve}[family]
            class Foreign(base):
                def evaluate(self,*args):
                    raise AssertionError('foreign semantics must never supply a proof')
            original = curve if family == 'outer' else getattr(curve,family)
            foreign = object.__new__(Foreign)
            for field in fields(original):
                object.__setattr__(foreign,field.name,getattr(original,field.name))
            candidate = foreign if family == 'outer' else iso_curve()
            if family != 'outer':
                object.__setattr__(candidate,family,foreign)
        with pytest.raises(IsoHullUnavailable):
            prepare_iso_hull(candidate)


def test_mutable_global_iso_cache_is_never_proof_authority(monkeypatch):
    import anygeometry.nurbs as nurbs
    curve = iso_curve()
    targets = ends(curve,Q(0),Q(1))
    monkeypatch.setitem(nurbs._ISO_CACHE,curve,object())
    with intersection_curve_operation():
        upper,_ = proof._world_chord(curve,Q(0),Q(1),targets,3.,lambda:None)
    assert Q(upper) >= 1


@pytest.mark.parametrize('warm',(False,True))
def test_numeric_alias_cannot_reuse_canonical_degree_proof(warm):
    curve = iso_curve()
    with intersection_curve_operation():
        if warm:
            prepare_iso_hull(curve)
        object.__setattr__(curve.pcurve,'degree',True)
        with pytest.raises(GeometryError):
            prepare_iso_hull(curve)


@pytest.mark.parametrize('warm',(False,True))
def test_same_name_float_subclass_cannot_alias_issued_literal_proof(warm):
    curve = iso_curve()
    # Equality/hash and even the name agree with float. Raw public definitions
    # must still not retain this ordinary caller object in an owner cache key.
    Impostor = type('float',(float,),{})
    with intersection_curve_operation():
        if warm:
            prepare_iso_hull(curve)
        controls = curve.support.controls
        first = controls[0][0]
        row = ((Impostor(first[0]),*first[1:]),*controls[0][1:])
        object.__setattr__(curve.support,'controls',(row,*controls[1:]))
        with pytest.raises(GeometryError,match='unsupported scalar'):
            prepare_iso_hull(curve)


def test_unsupported_non_iso_retains_original_proof_route(monkeypatch):
    curve = iso_curve()
    object.__setattr__(curve,'pcurve',NurbsCurve(1,(0.,0.,1.,1.),((0.,0.),(1.,1.))))
    with pytest.raises(IsoHullUnavailable):
        prepare_iso_hull(curve)
    calls = []
    def original_route(*args):
        calls.append(args)
        return 7.,((Q(0),Q(1)),)
    monkeypatch.setattr(proof,'_compute_world_chord',original_route)
    with intersection_curve_operation():
        for _ in range(2):
            assert proof._world_chord(curve,Q(0),Q(1),((0.,0.,0.),(1.,1.,1.)),8.,lambda:None)[0] == 7.
    assert len(calls) == 2  # unsupported results are deliberately uncached


def test_success_reuse_is_scoped_detached_and_checks_callbacks(monkeypatch):
    curve = iso_curve()
    targets = ends(curve,Q(0),Q(1))
    calls = []
    original = proof._compute_world_chord
    def compute(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(proof,'_compute_world_chord',compute)
    with intersection_curve_operation():
        from anygeometry.nurbs_intersection_curve import operation_fact,operation_cached
        key = lambda:proof._world_chord_key(curve,Q(0),Q(1),targets,3.,lambda:None)
        operation_fact(key,lambda:(0.,()))
        operation_cached(key)(lambda:(0.,()))()
        first = proof._world_chord(curve,Q(0),Q(1),targets,3.,lambda:None)
        assert first[0] > 0  # unrelated producer namespaces cannot seed authority
        object.__setattr__(first[1][0][0],'_numerator',123)
        second = proof._world_chord(curve,Q(0),Q(1),targets,3.,lambda:None)
        assert second[1] == ((Q(0),Q(1)),)
        assert len(calls) == 1
        proof._world_chord(curve,Q(0),Q(1),targets,4.,lambda:None)
        assert len(calls) == 2
        def cancel():
            raise GeometryError('cancelled proof hit')
        with pytest.raises(GeometryError,match='cancelled proof hit'):
            proof._world_chord(curve,Q(0),Q(1),targets,3.,cancel)
    with intersection_curve_operation():
        proof._world_chord(curve,Q(0),Q(1),targets,3.,lambda:None)
    assert len(calls) == 3


def test_success_hit_revalidates_current_owner_issuer():
    from test_nurbs_boundary_certification import graph_curve
    import anygeometry.nurbs_intersection_curve as curves
    curve = graph_curve(False)
    targets = tuple(map(tuple,curve.evaluate(np.asarray([0.,1.]))))
    with intersection_curve_operation():
        proof._world_chord(curve,Q(0),Q(1),targets,.001,lambda:None)
        curves._OPERATIONS.get()[('shape',curve)] = object()
        with pytest.raises(GeometryError,match='cached prepared shape'):
            proof._world_chord(curve,Q(0),Q(1),targets,.001,lambda:None)


def test_success_hit_rejects_callback_input_mutation():
    curve = iso_curve()
    targets = [list(p) for p in ends(curve,Q(0),Q(1))]
    with intersection_curve_operation():
        proof._world_chord(curve,Q(0),Q(1),targets,3.,lambda:None)
        def mutate():
            targets[0][2] += .25
        with pytest.raises(GeometryError,match='inputs changed'):
            proof._world_chord(curve,Q(0),Q(1),targets,3.,mutate)
