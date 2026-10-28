"""Source-curvature proposals never replace boundary or whole-cell proofs."""
from dataclasses import replace
from fractions import Fraction as Q
import numpy as np
import math
import pytest

from anygeometry import GeometryError
from anygeometry.nurbs_intersection_curve import intersection_curve_operation,NurbsSurfaceIntersectionCurve
from anygeometry.nurbs_surface_intersection._mesh_proposals import propose_c2_boundary_parameters as propose
from anygeometry.nurbs_surface_intersection._source import (
    NurbsSurfaceIntersectionCancelled,NurbsSurfaceIntersectionIntegrityError,NurbsSurfaceIntersectionRefused)
from test_nurbs_boundary_certification import graph_curve


@pytest.mark.parametrize('reverse',(False,True))
def test_exact_parabolic_c2_criterion_monotone_binary_stations_and_reversal(reverse):
    curve = graph_curve()
    if reverse:
        curve = curve.subcurve(Q(1),Q(0))
    original = (Q(float(np.float64(0))),Q(1,4),Q(1,2),Q(float(np.float64(1))))
    with intersection_curve_operation():
        parameters = propose(curve,original,chord_error=Q(1,16))
        assert type(parameters) is tuple and set(map(Q,original)) <= set(parameters)
        assert all(Q(float(t)) == t for t in parameters)
        assert all(a < b for a,b in zip(parameters,parameters[1:]))
        assert len(parameters) > len(original)
        # Exact continuous oracle C''=(0,0,8): ||C''||_1 Δt²/8=Δt².
        assert all((b-a)**2 <= Q(3,4)*Q(1,16) for a,b in zip(parameters,parameters[1:]))
        assert propose(curve,parameters,chord_error=Q(1,16)) == parameters


def test_zero_curvature_retains_original_stations_exactly():
    curve = graph_curve(False)
    original = (Q(0),Q(1,4),Q(3,4),Q(1))
    with intersection_curve_operation():
        assert propose(curve,original,chord_error=.000625) == original


def test_nonuniform_float_seed_midpoints_round_only_new_proposals_and_keep_actual_c2_criterion():
    curve = graph_curve()
    original = (Q(0),Q(.1),Q(.2),Q(1))
    assert Q(float((original[1]+original[2])/2)) != (original[1]+original[2])/2
    error = Q(1,1000)
    with intersection_curve_operation():
        parameters = propose(curve,original,chord_error=error)
        assert set(original) <= set(parameters) and len(parameters) > len(original)
        assert all(Q(float(point)) == point for point in parameters)
        for a,b in zip(parameters,parameters[1:]):
            # Exact continuous C''=8 oracle on the ACTUAL rounded intervals.
            assert (b-a)**2 <= Q(3,4)*error
            bound = curve.enclose(a,b)[0]
            maximum = sum((max(abs(x.lo),abs(x.hi)) for x in bound.world_second_derivative),Q(0))
            assert maximum*(b-a)**2/8 <= Q(3,4)*error


def test_adjacent_binary64_stations_refuse_if_the_c2_proposal_requires_more_resolution():
    a,b = Q(.5),Q(math.nextafter(.5,1.))
    error = (b-a)**2/16
    with pytest.raises(NurbsSurfaceIntersectionRefused,match='no interior binary64'):
        propose(graph_curve(),(a,b),chord_error=error)


@pytest.mark.parametrize('invalid',((0.,Q(1,3),1.),(0.,0.,1.),(1.,0.),(0.,float('nan'),1.)))
def test_bad_or_nonbinary_input_is_not_rounded_or_dropped(invalid):
    with pytest.raises(GeometryError):
        propose(graph_curve(),invalid,chord_error=.001)


def test_phase_callback_cancellation_exception_and_owner_mutation_propagate():
    curve = graph_curve()
    phases = []
    def cancel(phase):
        phases.append(phase)
        return True
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        propose(curve,(0.,1.),chord_error=.001,cancellation_check=cancel)
    assert phases and all(type(phase) is str for phase in phases)
    failure = RuntimeError('proposal caller failure')
    def fail(phase):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        propose(curve,(0.,1.),chord_error=.001,cancellation_check=fail)
    assert caught.value is failure
    anchor = curve.owner.branch_anchor
    def change(phase):
        object.__setattr__(curve.owner,'branch_anchor',('changed',))
    try:
        with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
            propose(curve,(0.,1.),chord_error=.001,cancellation_check=change)
    finally:
        object.__setattr__(curve.owner,'branch_anchor',anchor)


def test_owner_refusal_is_not_hidden_as_a_fallback(monkeypatch):
    curve = graph_curve()
    failure = NurbsSurfaceIntersectionRefused('original owner budget refusal')
    def refuse(*args,**kwargs):
        raise failure
    monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'enclose',refuse)
    with pytest.raises(NurbsSurfaceIntersectionRefused) as caught:
        propose(curve,(0.,1.),chord_error=.001)
    assert caught.value is failure


def test_trace_limit_is_not_replaced_by_empty_or_uncertified_success():
    curve = replace(graph_curve(),qualification_values=(('max_trace_segments',1),))
    with pytest.raises(NurbsSurfaceIntersectionRefused):
        propose(curve,(0.,1.),chord_error=Q(1,1024))


def test_nonbinary_mandatory_joint_explicitly_defers_whole_interval(monkeypatch):
    from test_nurbs_pair_public import model_pair
    from anygeometry.intersections import query_intersection
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    count = len(curve.owner.segments)
    assert count >= 2
    view = curve.subcurve(Q(0),Q(3,2*count))  # original joint maps to2/3
    def forbidden(*args,**kwargs):
        raise AssertionError('C2 proposal spanned a nonbinary mandatory joint')
    monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'enclose',forbidden)
    assert propose(view,(0.,1.),chord_error=.001) == (Q(0),Q(1))


def test_binary_mandatory_joint_is_retained_before_single_piece_proposals():
    from test_nurbs_pair_public import model_pair
    from anygeometry.intersections import query_intersection
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    count = len(curve.owner.segments)
    assert count >= 2
    view = curve.subcurve(Q(1,2*count),Q(3,2*count))
    with intersection_curve_operation():
        parameters = propose(view,(0.,1.),chord_error=1000.)
        assert Q(1,2) in parameters and parameters[0] == 0 and parameters[-1] == 1
        assert all(Q(float(t)) == t for t in parameters)


def test_depth_exhaustion_and_warmed_fake_context_refuse(monkeypatch):
    import anygeometry.plane_nurbs as work
    import anygeometry.nurbs_intersection_curve as curves
    curve = graph_curve()
    with intersection_curve_operation():
        propose(curve,(0.,1.),chord_error=2.)
        curves._OPERATIONS.get()[('shape',curve)] = object()
        with pytest.raises(GeometryError,match='cached prepared shape'):
            propose(curve,(0.,1.),chord_error=2.)
    monkeypatch.setattr(work,'depth_limit',lambda default:0)
    with pytest.raises(NurbsSurfaceIntersectionRefused,match='depth exhausted'):
        propose(curve,(0.,1.),chord_error=Q(1,1024))


def test_enclosure_and_split_queries_charge_existing_work(monkeypatch):
    import anygeometry.plane_nurbs as work
    counts = {'box':0,'split':0}
    def box(): counts['box'] += 1
    def split(): counts['split'] += 1
    monkeypatch.setattr(work,'charge_box',box)
    monkeypatch.setattr(work,'charge_split',split)
    parameters = propose(graph_curve(),(0.,1.),chord_error=Q(1,16))
    assert counts['split'] == len(parameters)-2 and counts['box'] >= len(parameters)-1
