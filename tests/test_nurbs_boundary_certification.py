"""Pinned source frontiers and exact single-C2-piece chord remainder proofs."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import (NurbsIntersectionCurve,IntersectionStatus,
    query_nurbs_surface_intersection)
from anygeometry.nurbs_surface_intersection._intervals import Interval
from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
import anygeometry.prepared_material_curvilinear as owner
from test_nurbs_pair_public import model_pair


def graph_curve(parabola=True):
    knots = (0.,0.,0.,1.,1.,1.)
    square = (1.,-1.,1.) if parabola else (0.,0.,0.)
    surfaces = tuple(NurbsSurface(2,2,knots,knots,
        tuple(tuple((float(i-1),float(j-1),square[i]+.5*offset*(j-1))
                    for j in range(3)) for i in range(3))) for offset in (0,1))
    result = query_nurbs_surface_intersection(*surfaces,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED
    curve = NurbsSurfaceIntersectionCurve(NurbsIntersectionCurve.from_result(result))
    assert len(curve.owner.segments) == 1
    return curve


@pytest.mark.parametrize('shift',(Q(0),Q(1,8)))
def test_world_chord_c2_bound_covers_exact_parabolic_remainder_and_endpoint_error(shift,monkeypatch):
    import anygeometry.plane_nurbs as work
    curve = graph_curve()
    with intersection_curve_operation():
        curve._shape()
        ends = [tuple(map(float,p)) for p in curve.evaluate(np.asarray([0.,1.]))]
        ends[0] = (*ends[0][:2],ends[0][2]+float(shift))
        charged = []
        original_charge = work.charge_box
        def charge():
            charged.append(1)
            return original_charge()
        monkeypatch.setattr(work,'charge_box',charge)
        bound,pieces = owner._world_chord(curve,Q(0),Q(1),tuple(ends),1+shift,lambda:None)
        # C(t)=(+/- (2t-1),0,(2t-1)^2), ||C''||_1=8.
        # The exact parametric chord remainder with shifted first target is
        # max[-4t^2+(4-shift)t+shift] = shift+(4-shift)^2/16.
        exact = shift+(4-shift)**2/16
        assert Q(bound) == 1+shift and Q(bound) >= exact
        assert pieces == ((Q(0),Q(1)),)
        assert len(charged) == 3  # two endpoint queries plus one whole C2 query


def test_loose_c2_upper_falls_back_without_falsely_rejecting_exact_straight_chord(monkeypatch):
    curve = graph_curve(parabola=False)
    original = NurbsSurfaceIntersectionCurve.enclose
    def loose(self,*args,**kwargs):
        return tuple(replace(bound,world_second_derivative=(Interval(-10,10),)*3)
                     for bound in original(self,*args,**kwargs))
    with intersection_curve_operation():
        ends = tuple(map(tuple,curve.evaluate(np.asarray([0.,1.]))))
        monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'enclose',loose)
        bound,pieces = owner._world_chord(curve,Q(0),Q(1),ends,.001,lambda:None)
        assert bound == 0. and pieces == ((Q(0),Q(1)),)


def test_c2_fast_bound_never_spans_mandatory_continuation_station(monkeypatch):
    from anygeometry.intersections import query_intersection
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    assert len(curve.owner.segments) > 1
    def forbidden(*args,**kwargs):
        raise AssertionError('multi-piece curve requested a global C2 proof')
    monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'enclose',forbidden)
    assert owner._single_c2_world_chord_bound(curve,Q(0),Q(1),Q(0),lambda:None) is None


def test_mutating_returned_interval_fields_cannot_poison_cached_chord_or_derivative_proofs():
    curve = graph_curve()
    with intersection_curve_operation():
        bound = curve.enclose(Q(0),Q(1))[0]
        names = ('first_chart_derivative','second_chart_derivative','world_derivative',
                 'first_chart_second_derivative','second_chart_second_derivative','world_second_derivative')
        expected = {name:tuple((x.lo,x.hi) for x in getattr(bound,name)) for name in names}
        for name in names:
            for interval in getattr(bound,name):
                object.__setattr__(interval,'lo',Q(0))
                object.__setattr__(interval,'hi',Q(0))
        fresh = curve.enclose(Q(0),Q(1))[0]
        for name in names:
            assert tuple((x.lo,x.hi) for x in getattr(fresh,name)) == expected[name]
        ends = tuple(map(tuple,curve.evaluate(np.asarray([0.,1.]))))
        # The true midpoint lies distance 1 from this chord. Forged zero
        # acceleration in a previous public result must not certify .001.
        with pytest.raises(owner._WorldChordNeedsRefinement):
            owner._world_chord(curve,Q(0),Q(1),ends,.001,lambda:None)


@pytest.mark.parametrize('change',('controls','vertex','reverse_incidence','operand'))
def test_pinned_boundary_callback_rejects_raw_same_revision_mutation(change):
    model,faces = model_pair()
    revision = model.revision
    operand = owner.MaterialBoundaryChord(model.handle('edge',next(iter(model.edges))),
        (Q(0),Q(1)),(),(),(),0.,0.)
    def callback(phase):
        if change == 'controls':
            surface = model.faces[faces[0]].surface
            controls = [list(row) for row in surface.controls]
            x,y,z = controls[1][1]
            controls[1][1] = (x,y,z+.125)
            object.__setattr__(surface,'controls',tuple(tuple(row) for row in controls))
        elif change == 'vertex':
            vertex = next(iter(model.vertices.values()))
            object.__setattr__(vertex,'position',vertex.position+np.asarray([.125,0.,0.]))
        elif change == 'reverse_incidence':
            model._edge_coedges.setdefault(next(iter(model.edges)),set()).add(987654321)
        else:
            object.__setattr__(operand,'world_error',.125)
        return False
    pinned = owner._pinned_callback(model,('regions',),callback,(operand,))
    with pytest.raises(GeometryError,match='changed during callback'):
        pinned('boundary regression')
    assert model.revision == revision


def test_unchanged_pinned_callback_reuses_complete_content_qualification(monkeypatch):
    import anygeometry.serialization as serialization
    model,_faces = model_pair()
    qualified = []
    actual = serialization._qualify_document
    def qualify(*args,**kwargs):
        qualified.append(1)
        return actual(*args,**kwargs)
    monkeypatch.setattr(serialization,'_qualify_document',qualify)
    pinned = owner._pinned_callback(model,('regions',),lambda phase:False,())
    initial = len(qualified)
    assert pinned('first') is False
    assert pinned('second') is False
    assert len(qualified) == initial
