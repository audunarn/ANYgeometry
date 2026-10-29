"""Upper-bound inconclusiveness is distinct from violation and hard refusal."""
from fractions import Fraction as Q
from uuid import uuid4
import pytest

from anygeometry import GeometryError,SourceCellRefinementRequired,SourceCellApproximationInconclusive
from anygeometry.errors import _FanApproximationInconclusive
from anygeometry.surfaces import Plane
from anygeometry.regular_source_reference import SourceNode,SourceSide
from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work


def fan(scale=Q(1),offset=Q(1),orientation=1):
    # A genuine affine source fan with FE translated in z. Its approximation
    # is not exact; no custom curve/derivative or fabricated upper bound.
    side=SourceSide(SourceNode(('a',),((Q(0),Q(0)),(Q(0),Q(0)))),
                    SourceNode(('b',),((Q(1),Q(1)),(Q(0),Q(0)))))
    return dict(support=Plane((0.,0.,0.),(float(scale),0.,0.),(0.,float(2*scale),0.)),
        boundary=side,centre_uv=(Q(1,3),Q(2,3)),
        xyz_side=((Q(0),Q(0),offset),(scale,Q(0),offset)),
        centre_xyz=(scale/3,4*scale/3,offset),orientation=orientation)


@pytest.mark.parametrize('scale,tolerance',((Q(1),Q(1,8)),(Q(3,2),Q(1,100)),(Q(1,16),Q(1,1000))))
def test_actual_terminal_fan_positive_wedge_emits_upper_only(scale,tolerance):
    work=Work(100,0);token=CURRENT_WORK.set(work)
    try:
        with pytest.raises(_FanApproximationInconclusive) as caught:
            certify_fan_subtriangle(**fan(scale,2*tolerance),surface_error=tolerance)
    finally:CURRENT_WORK.reset(token)
    error=caught.value
    assert error.upper_error>=2*tolerance>error.error_limit==tolerance
    assert error.jacobian_lower>0 and error.proof_depth==0
    assert error.proof_box==((Q(0),Q(1)),(Q(0),Q(1)))
    assert not isinstance(error,SourceCellRefinementRequired)


@pytest.mark.parametrize('orientation',(0,-1))
def test_nonpositive_or_uncertain_wedge_remains_hard_refusal(orientation):
    token=CURRENT_WORK.set(Work(100,0))
    try:
        with pytest.raises(GeometryError) as caught:
            certify_fan_subtriangle(**fan(orientation=orientation),surface_error=Q(1,8))
    finally:CURRENT_WORK.reset(token)
    assert type(caught.value) is GeometryError


def test_straddling_jacobian_interval_remains_hard_refusal():
    class UncertainBoundary:
        curve=None
        def box(self,lo,hi):return ((Q(0),Q(1)),(Q(-1),Q(1)))
        def derivative(self,lo,hi):return ((Q(1),Q(1)),(Q(0),Q(0)))
    inputs=fan();inputs['boundary']=UncertainBoundary();inputs['centre_uv']=(Q(1,3),Q(0))
    # cross((1,0), centre-Q(t)) = [-1,1], genuinely uncertain.
    token=CURRENT_WORK.set(Work(100,0))
    try:
        with pytest.raises(GeometryError) as caught:
            certify_fan_subtriangle(**inputs,surface_error=Q(1,8))
    finally:CURRENT_WORK.reset(token)
    assert type(caught.value) is GeometryError


def test_box_budget_and_cancellation_stay_hard():
    token=CURRENT_WORK.set(Work(0,0))
    try:
        with pytest.raises(GeometryError) as caught:
            certify_fan_subtriangle(**fan(),surface_error=Q(1,8))
    finally:CURRENT_WORK.reset(token)
    assert not isinstance(caught.value,_FanApproximationInconclusive)
    with pytest.raises(GeometryError,match='cancelled') as caught:
        certify_fan_subtriangle(**fan(),surface_error=Q(1,8),cancellation_check=lambda:True)
    assert type(caught.value) is GeometryError


def test_previous_box_upper_bound_cannot_reclassify_later_uncertain_jacobian():
    class ChangingBoundary:
        curve=None
        calls=0
        def box(self,lo,hi):return ((Q(0),Q(1)),(Q(0),Q(0)))
        def derivative(self,lo,hi):
            self.calls+=1
            return ((Q(1),Q(1)),(Q(0),Q(0))) if self.calls==1 else ((Q(0),Q(0)),)*2
    inputs=fan();inputs['boundary']=ChangingBoundary()
    token=CURRENT_WORK.set(Work(100,1))
    try:
        with pytest.raises(GeometryError) as caught:
            certify_fan_subtriangle(**inputs,surface_error=Q(1,8))
    finally:CURRENT_WORK.reset(token)
    assert type(caught.value) is GeometryError
    assert inputs['boundary'].calls==2


def test_invalid_nurbs_chart_is_not_approximation_inconclusiveness():
    from anygeometry.nurbs import NurbsSurface
    inputs=fan()
    inputs['support']=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))),((1.,2.),(1.,2.)))
    inputs['boundary']=SourceSide(SourceNode(('a',),((Q(2),Q(2)),(Q(0),Q(0)))),
                                  SourceNode(('b',),((Q(3),Q(3)),(Q(0),Q(0)))))
    inputs['centre_uv']=(Q(5,2),Q(1,2))
    token=CURRENT_WORK.set(Work(100,0))
    try:
        with pytest.raises(GeometryError) as caught:
            certify_fan_subtriangle(**inputs,surface_error=Q(1,8))
    finally:CURRENT_WORK.reset(token)
    assert not isinstance(caught.value,_FanApproximationInconclusive)


def signal(**changes):
    fields=dict(face_id=7,node_ids=(1,2,3),error_limit=Q(1,100),upper_error=Q(1,50),
        requested_error=Q(1,80),jacobian_lower=Q(1,4),proof_box=((Q(0),Q(1)),)*2,
        proof_depth=12,source_model_id=uuid4(),source_revision=3,source_checksum='a'*64,chord_checksum='b'*64)
    fields.update(changes)
    return SourceCellApproximationInconclusive(**fields)


def test_public_signal_is_not_a_lower_witness():
    error=signal()
    assert isinstance(error,GeometryError) and not isinstance(error,SourceCellRefinementRequired)
    assert not hasattr(error,'lower_error_squared')


@pytest.mark.parametrize('changes',({'upper_error':Q(1,100)},{'jacobian_lower':Q(0)},
    {'upper_error':float('inf')},{'requested_error':Q(1,200)},
    {'proof_box':((Q(0),Q(0)),)*2},{'proof_depth':True},{'source_checksum':'stale'},
    {'node_ids':(1,1,2)}))
def test_malformed_public_inconclusive_fields_refuse(changes):
    with pytest.raises(GeometryError):signal(**changes)


def test_owner_contextualizes_real_fan_result_after_fresh_binding(monkeypatch):
    # Public source-cell path uses authentic model/regions/chords. The real
    # terminal producer above is delegated only to isolate owner control flow;
    # this test is not material admission or an accuracy assertion.
    from test_prepared_material_curvilinear import square_cells
    from anygeometry import validate_prepared_material_curvilinear_cells
    from anygeometry import source_cell_complex as cells
    from anygeometry.definition_binding import definition_checksum
    model,regions,chords,rows=square_cells(True)
    actual=certify_fan_subtriangle
    def terminal(*args,**kwargs):
        limit=Q(kwargs['surface_error'])
        token=CURRENT_WORK.set(Work(100,0))
        try:return actual(**fan(offset=2*limit),surface_error=limit,
                          cancellation_check=kwargs['cancellation_check'])
        finally:CURRENT_WORK.reset(token)
    monkeypatch.setattr(cells,'certify_fan_subtriangle',terminal)
    with pytest.raises(SourceCellApproximationInconclusive) as caught:
        validate_prepared_material_curvilinear_cells(model,regions,chords,rows,surface_error=.0025)
    error=caught.value
    assert error.face_id==rows[0].face.id and error.node_ids==rows[0].node_ids
    assert error.requested_error==Q(.0025) and 0<error.error_limit<=error.requested_error
    assert error.source_model_id==model.model_id and error.source_revision==model.revision
    assert error.chord_checksum==definition_checksum(chords)


def test_owner_does_not_contextualize_a_callback_private_signal(monkeypatch):
    from test_prepared_material_curvilinear import square_cells
    from anygeometry import validate_prepared_material_curvilinear_cells
    from anygeometry import source_cell_complex as cells
    model,regions,chords,rows=square_cells(True)
    armed=[False]
    private=_FanApproximationInconclusive(Q(1,100),Q(1),Q(1),((Q(0),Q(1)),)*2,12)
    def terminal(*args,**kwargs):
        armed[0]=True
        kwargs['cancellation_check']()
    def callback(phase):
        if armed[0]:raise private
    monkeypatch.setattr(cells,'certify_fan_subtriangle',terminal)
    with pytest.raises(_FanApproximationInconclusive) as caught:
        validate_prepared_material_curvilinear_cells(model,regions,chords,rows,
            surface_error=.0025,cancellation_check=callback)
    assert caught.value is private


@pytest.mark.parametrize('mutation',('chord','transport'))
def test_owner_rejects_stale_chord_or_unqualified_transport_instead_of_signal(monkeypatch,mutation):
    from test_prepared_material_curvilinear import square_cells
    from anygeometry import validate_prepared_material_curvilinear_cells
    from anygeometry import source_cell_complex as cells
    model,regions,chords,rows=square_cells(True)
    if mutation=='transport':
        monkeypatch.setattr(cells,'parameter_transport_bound',lambda side:Q(1))
    else:
        def terminal(*args,**kwargs):
            limit=Q(kwargs['surface_error']);token=CURRENT_WORK.set(Work(100,0))
            try:
                try:certify_fan_subtriangle(**fan(offset=2*limit),surface_error=limit)
                except _FanApproximationInconclusive:
                    object.__setattr__(chords,'chord_error',1.)
                    raise
            finally:CURRENT_WORK.reset(token)
        monkeypatch.setattr(cells,'certify_fan_subtriangle',terminal)
    with pytest.raises(GeometryError) as caught:
        validate_prepared_material_curvilinear_cells(model,regions,chords,rows,surface_error=.0025)
    assert not isinstance(caught.value,SourceCellApproximationInconclusive)
