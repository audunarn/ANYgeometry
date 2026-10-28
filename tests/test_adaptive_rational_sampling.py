from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
from anygeometry.nurbs import NurbsCurve,CurveOnSurface
from anygeometry.curvilinear_reference_maps import _SideMemo,certify_fan_subtriangle
from anygeometry.prepared_material_curvilinear import (_world_chord,_parameterized_world_chord,
    _WorldChordNeedsRefinement,query_prepared_material_boundary_sampling,
    validate_prepared_material_boundary_sampling_binding)
from test_rational_source_side import side,parabola
from test_local_curve_geometry import flat
from test_section_boundary_sampling import under_work


def test_rational_memo_reuses_exact_ranges_without_recharging_spans():
    original=side(parabola());memo=_SideMemo(original)
    with under_work() as work:
        box=memo.box(Q(1,8),Q(7,8));charged=work.boxes
        derivative=memo.derivative(Q(1,8),Q(7,8))
        assert charged>0 and work.boxes==charged
        assert box==original.box(Q(1,8),Q(7,8))
        assert derivative==original.derivative(Q(1,8),Q(7,8))
        assert work.boxes>charged


@pytest.mark.parametrize('endpoint',(False,True))
def test_rational_memo_hit_refuses_current_definition_mutation(endpoint):
    original=side(parabola());memo=_SideMemo(original);memo.box(Q(0),Q(1))
    if endpoint:object.__setattr__(original.start,'identity',('foreign',))
    else:object.__setattr__(original.curve.pcurve,'controls',((0.,0.),(.5,.25),(1.,1.)))
    with pytest.raises(GeometryError):memo.box(Q(0),Q(1))


def test_rational_fan_callback_exception_identity_and_endpoint_mutation():
    original=side(parabola());failure=RuntimeError('actual callback')
    def raised():raise failure
    with pytest.raises(RuntimeError) as caught:
        certify_fan_subtriangle(original.curve.support,original,(Q(1,2),Q(3,4)),
            ((0.,0.,0.),(1.,1.,0.)),(.5,.75,0.),orientation=1,surface_error=Q(3,10),cancellation_check=raised)
    assert caught.value is failure
    def mutate():object.__setattr__(original.end,'identity',('changed',));return False
    with pytest.raises(GeometryError,match='changed'):
        certify_fan_subtriangle(original.curve.support,original,(Q(1,2),Q(3,4)),
            ((0.,0.,0.),(1.,1.,0.)),(.5,.75,0.),orientation=1,surface_error=Q(3,10),cancellation_check=mutate)


def test_geometric_distance_does_not_claim_parameterized_interpolation():
    curve=CurveOnSurface(flat(),NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.),(.125,0.),(1.,0.))))
    ends=((0.,0.,0.),(1.,0.,0.));check=lambda:None
    with under_work():
        bound,_=_world_chord(curve,Q(0),Q(1),ends,.01,check)
        assert bound<=.01
    with under_work(),pytest.raises(_WorldChordNeedsRefinement):
        _parameterized_world_chord(curve,Q(0),Q(1),ends,.01,check)


def rounded(support=None):
    import anygeometry as owner
    model=owner.GeometryModel();support=flat() if support is None else support
    points=((.75,.5),(.5,.75),(.25,.5),(.5,.25))
    corners=((.75,.75),(.25,.75),(.25,.25),(.75,.25))
    curves=tuple(CurveOnSurface(support,NurbsCurve(2,(0.,0.,0.,1.,1.,1.),
        (points[i],corners[i],points[(i+1)%4]))) for i in range(4))
    vertices=tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges=tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve) for i,curve in enumerate(curves))
    face=model.add_face(edges,surface=support)
    regions=owner.query_material_surface_regions(model,(face,))
    return model,regions,tuple((model.handle('edge',edge),(Q(0),Q(1))) for edge in edges)


def test_sampling_receipt_binds_measure_fraction_and_preserves_all_seeds():
    model,regions,seeds=rounded()
    result=query_prepared_material_boundary_sampling(model,regions,seeds,chord_error=.02,
        error_measure='parameterized',proposal_fraction=.25)
    assert result.error_measure=='parameterized' and result.proposal_fraction==.25
    assert all(set(seed).issubset(parameters) for (_,seed),(_,parameters) in zip(seeds,result.edge_parameters))
    validate_prepared_material_boundary_sampling_binding(model,regions,result)
    with pytest.raises(GeometryError):validate_prepared_material_boundary_sampling_binding(model,regions,replace(result,proposal_fraction=1.))
    object.__setattr__(result,'error_measure','geometric')
    with pytest.raises(GeometryError):validate_prepared_material_boundary_sampling_binding(model,regions,result)


@pytest.mark.parametrize('fraction',[0.,-1.,float('nan'),float('inf'),True,1.01])
def test_sampling_proposal_fraction_refuses_invalid_before_owner_work(fraction):
    with pytest.raises(GeometryError,match='fraction'):
        query_prepared_material_boundary_sampling(None,None,(),chord_error=.01,proposal_fraction=fraction)


def test_boundary_bearing_cells_with_real_interior_only_owner_witness():
    import anygeometry as owner
    from anygeometry.nurbs import NurbsSurface
    # z=8*((u-.5)**2+(v-.5)**2): all FE nodes lie exactly on the
    # source. Boundary error is below .2, but the triangle-centre error
    # is 2/9>.2. No off-source node manufactures the interior witness.
    heights=(.25,-.25,.25);knots=(0.,0.,0.,1.,1.,1.)
    support=NurbsSurface(2,2,knots,knots,tuple(tuple((i/2,j/2,
        8*(heights[i]+heights[j])) for j in range(3)) for i in range(3)))
    model,regions,seeds=rounded(support);face=next(iter(model.faces))
    chords=owner.query_prepared_material_boundary_chords(model,regions,
        tuple((handle,0.,1.) for handle,_ in seeds),chord_error=.2)
    points=((.75,.5),(.5,.75),(.25,.5),(.5,.25))
    cells=tuple(owner.CurvilinearMaterialCell(model.handle('face',face),(i,(i+1)%4,4),
        (points[i],points[(i+1)%4],(.5,.5)),
        (tuple(map(float,support.exact_point(*points[i]))),
         tuple(map(float,support.exact_point(*points[(i+1)%4]))),
         tuple(map(float,support.exact_point(.5,.5)))),((0,i,True),)) for i in range(4))
    with pytest.raises(owner.SourceCellRefinementRequired) as caught:
        owner.validate_prepared_material_curvilinear_cells(model,regions,chords,cells,surface_error=.2)
    assert caught.value.lower_error_squared>caught.value.error_limit**2
    assert caught.value.boundary_failure_kind=='interior_only'


def test_parameterized_sampling_context_mutation_in_callback_refuses(monkeypatch):
    import anygeometry.prepared_material_curvilinear as module
    model,regions,seeds=rounded();original=module._cos_c2_world_chord_bound
    def callback(*args):
        monkeypatch.setattr(module,'_cos_c2_world_chord_bound',lambda *args,**kwargs:Q(0))
    with pytest.raises(GeometryError,match='context changed'):
        query_prepared_material_boundary_sampling(model,regions,seeds,chord_error=.02,
            error_measure='parameterized',proposal_fraction=.25,cancellation_check=callback)


def test_local_cell_infimum_is_not_boundary_station_impossibility():
    from types import SimpleNamespace
    from anygeometry.source_cell_complex import _boundary_failure_kind
    current=side(parabola())
    cell=SimpleNamespace(node_ids=(0,1,2),xyz=((0.,0.,0.),(1.,1.,0.),(.5,.75,0.)))
    fans=((0,current,Q(0)),);marked={(0,0):0}
    assert _boundary_failure_kind(cell,0,fans,marked,Q(3,10),lambda:None,
        cell_error_limit=Q(1,10))=='unresolved'
    assert _boundary_failure_kind(cell,0,fans,marked,Q(1,10),lambda:None)=='boundary_limited'
