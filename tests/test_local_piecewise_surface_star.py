from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry.nurbs import NurbsSurface
from anygeometry._local_curvature import LocalApproximationNotCertified,bound_surface_curvature
from anygeometry._local_piecewise_surface_star import (SpanBudget,surface_span_atlas,
    certify_piecewise_surface_star,certify_span_continuity,validate_span_coverage,
    restricted_span_pieces,span_hull)
from anygeometry.predicates import IntersectionQualificationPolicy

FULL=((Q(0),Q(1)),)*2
SIDES=((0,0),(0,1),(1,0),(1,1))


def inserted_surface(warp=Q(1,32),rational=False):
    if rational:
        xs=(0.,.375,.625,1.); zs=(0.,0.,float(warp/4),float(warp)); weights=((1.,1.),(2.,2.),(2.,2.),(1.,1.))
    else:
        xs=(0.,.25,.75,1.); zs=(0.,0.,float(warp/2),float(warp)); weights=None
    return NurbsSurface(2,1,(0.,0.,0.,.5,1.,1.,1.),(0.,0.,1.,1.),
        tuple(tuple((x,y,z) for y in (0.,1.)) for x,z in zip(xs,zs)),weights=weights)


def original_surface(warp=Q(1,32),rational=False):
    return NurbsSurface(2,1,(0.,0.,0.,1.,1.,1.),(0.,0.,1.,1.),
        tuple(tuple((x,y,z) for y in (0.,1.)) for x,z in ((0.,0.),(.5,0.),(1.,float(warp)))),
        weights=((1.,1.),(3.,3.),(1.,1.)) if rational else None)


@pytest.mark.parametrize('rational',(False,True))
@pytest.mark.parametrize('ranges',(((0.,1.),(0.,1.)),((.875,.125),(1.,0.)),((.125,.75),(.25,.875))))
def test_complete_c1_projection_and_knot_insertion_image(rational,ranges):
    surface=replace(inserted_surface(rational=rational),u_range=ranges[0],v_range=ranges[1])
    original=replace(original_surface(rational=rational),u_range=ranges[0],v_range=ranges[1])
    budget=SpanBudget();atlas=surface_span_atlas(surface,budget=budget)
    star=certify_piecewise_surface_star(atlas,FULL,FULL,budget=budget,source_boundary_sides=SIDES)
    assert star.strong_monotonicity_lower>0 and star.parameter_buffer is None
    assert sum((p.chart_box[0][1]-p.chart_box[0][0])*(p.chart_box[1][1]-p.chart_box[1][0]) for p in atlas.pieces)==1
    # Exact independent source image equality, not equality of conservative bounds.
    for u in (Q(0),Q(1,8),Q(1,2),Q(7,8),Q(1)):
        for v in (Q(0),Q(1,3),Q(1)):
            assert surface.exact_point(u,v)==original.exact_point(u,v)
    bounds=bound_surface_curvature(surface,_atlas=atlas)
    assert bounds.tiles and all(t.sigma_min_squared_lower>0 for t in bounds.tiles)


def test_internal_knot_is_not_exterior_and_remote_domain_retained():
    budget=SpanBudget();atlas=surface_span_atlas(inserted_surface(0),budget=budget)
    inner=((Q(3,8),Q(5,8)),(Q(3,8),Q(5,8)))
    outer=((Q(1,4),Q(3,4)),)*2
    star=certify_piecewise_surface_star(atlas,inner,outer,budget=budget)
    assert star.parameter_buffer==Q(1,8) and star.inner_to_outer_boundary_lower>0
    with pytest.raises(LocalApproximationNotCertified,match='source-chart endpoint'):
        certify_piecewise_surface_star(atlas,inner,outer,budget=budget,source_boundary_sides=((0,0),))
    assert len(restricted_span_pieces(atlas,outer,take=budget.take))==2


@pytest.mark.parametrize('damage',('missing','duplicate'))
def test_incomplete_or_duplicate_atlas_refused(damage):
    atlas=surface_span_atlas(inserted_surface(),budget=SpanBudget())
    pieces=atlas.pieces[:-1] if damage=='missing' else (*atlas.pieces,atlas.pieces[0])
    with pytest.raises(LocalApproximationNotCertified):validate_span_coverage(replace(atlas,pieces=pieces))


def test_c0_corner_and_smooth_fold_are_not_c1_embedded_stars():
    corner=NurbsSurface(1,1,(0.,0.,.5,1.,1.),(0.,0.,1.,1.),
        tuple(tuple((x,y,z) for y in (0.,1.)) for x,z in ((0.,0.),(.5,.25),(1.,0.))))
    budget=SpanBudget();atlas=surface_span_atlas(corner,budget=budget)
    with pytest.raises(LocalApproximationNotCertified,match='C1'):
        certify_piecewise_surface_star(atlas,FULL,FULL,budget=budget,source_boundary_sides=SIDES)
    fold=replace(inserted_surface(0),controls=tuple(tuple((x,y,0.) for y in (0.,1.)) for x in (0.,.25,.25,0.)))
    budget=SpanBudget();atlas=surface_span_atlas(fold,budget=budget)
    certify_span_continuity(atlas,budget=budget)
    with pytest.raises(LocalApproximationNotCertified):
        certify_piecewise_surface_star(atlas,FULL,FULL,budget=budget,source_boundary_sides=SIDES)


def test_work_callback_and_source_mutation_frontiers():
    surface=inserted_surface()
    with pytest.raises(LocalApproximationNotCertified,match='budget'):
        surface_span_atlas(surface,budget=SpanBudget(IntersectionQualificationPolicy(max_boxes_per_pair=1)))
    token=RuntimeError('external cancellation')
    def fail():raise token
    with pytest.raises(RuntimeError) as caught:surface_span_atlas(surface,budget=SpanBudget(check=fail))
    assert caught.value is token
    def mutate():object.__setattr__(surface,'u_range',(.125,.875))
    with pytest.raises(LocalApproximationNotCertified,match='changed'):
        surface_span_atlas(surface,budget=SpanBudget(check=mutate))


def test_changed_subbox_code_context_refuses(monkeypatch):
    from anygeometry.nurbs_surface_intersection import _bernstein
    original=_bernstein.subbox
    monkeypatch.setattr(_bernstein,'subbox',lambda *args:original(*args))
    with pytest.raises(LocalApproximationNotCertified,match='context'):
        surface_span_atlas(inserted_surface(),budget=SpanBudget())


def test_nonuniform_shifted_knots_and_affine_c1_degree_one_seam():
    surface=NurbsSurface(1,1,(-2.,-2.,0.,6.,6.),(0.,0.,1.,1.),
        tuple(tuple((x,y,0.) for y in (0.,1.)) for x in (0.,.25,1.)),u_range=(-2.,6.))
    budget=SpanBudget();atlas=surface_span_atlas(surface,budget=budget)
    star=certify_piecewise_surface_star(atlas,FULL,FULL,budget=budget,source_boundary_sides=SIDES)
    assert star.strong_monotonicity_lower==1
    assert tuple(p.chart_box[0] for p in atlas.pieces)==((Q(0),Q(1,4)),(Q(1,4),Q(1)))


def test_fixed_common_projection_rejects_smooth_reversing_sheet():
    fold=replace(inserted_surface(0),controls=tuple(tuple((x,y,0.) for y in (0.,1.)) for x in (0.,.25,.25,0.)))
    budget=SpanBudget(IntersectionQualificationPolicy(max_subdivision_depth=4))
    atlas=surface_span_atlas(fold,budget=budget)
    with pytest.raises(LocalApproximationNotCertified,match='monotonicity'):
        certify_piecewise_surface_star(atlas,FULL,FULL,budget=budget,
            projection=((1,0,0),(0,1,0)),source_boundary_sides=SIDES)


def test_in_place_subbox_code_change_refuses_before_reuse(monkeypatch):
    from anygeometry.nurbs_surface_intersection import _bernstein
    function=_bernstein.subbox
    def unsupported(grid,u0,u1,v0,v1):return grid
    monkeypatch.setattr(function,'__code__',unsupported.__code__)
    with pytest.raises(LocalApproximationNotCertified,match='context'):
        surface_span_atlas(inserted_surface(),budget=SpanBudget())


@pytest.mark.parametrize('name',('SurfaceSpan','SurfaceSpanAtlas','SurfaceStarBound'))
def test_same_identity_constructor_code_replacement_refuses(monkeypatch,name):
    import anygeometry._local_piecewise_surface_star as spans
    function=getattr(spans,name).__init__
    # Frozen dataclass constructors close over object; retain the closure count
    # while changing the existing function object's code rather than its identity.
    marker=object
    def changed(self,*args,**kwargs):
        if marker is not None:raise RuntimeError('changed constructor must not execute')
    assert len(function.__code__.co_freevars)==len(changed.__code__.co_freevars)
    monkeypatch.setattr(function,'__code__',changed.__code__)
    with pytest.raises(LocalApproximationNotCertified,match='context'):
        surface_span_atlas(inserted_surface(),budget=SpanBudget())
