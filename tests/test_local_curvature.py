from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import Plane
from anygeometry.nurbs import NurbsSurface
from anygeometry.predicates import IntersectionQualificationPolicy
from anygeometry._local_curvature import bound_surface_curvature, LocalApproximationNotCertified

def saddle(scale=1., shift=0., weights=None):
    return NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((shift,0.,0.),(shift,scale,0.)),
         ((shift+scale,0.,0.),(shift+scale,scale,scale))),weights=weights)

def test_exact_saddle_radius_bound_covers_known_origin_curvature():
    result=bound_surface_curvature(saddle())
    assert result.tiles
    assert all(t.curvature_upper>=1 and 0<t.radius_lower<=1 for t in result.tiles)
    assert all(t.sigma_min_squared_lower>0 for t in result.tiles)

@pytest.mark.parametrize('scale',[.125,1.,8.])
def test_curvature_units_scale_and_translation(scale):
    first=bound_surface_curvature(saddle())
    scaled=bound_surface_curvature(saddle(scale,16.))
    assert [t.curvature_upper*Q(scale) for t in scaled.tiles] == [t.curvature_upper for t in first.tiles]
    assert [t.radius_lower/Q(scale) for t in scaled.tiles] == [t.radius_lower for t in first.tiles]

def test_flat_rational_patch_has_zero_curvature_without_discarding_hessian_bound():
    surface=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))),weights=((1.,2.),(2.,1.)))
    result=bound_surface_curvature(surface)
    assert all(t.curvature_upper==0 and t.radius_lower is None for t in result.tiles)
    assert all(t.hessian_norm_upper>=0 for t in result.tiles)

def test_nonuniform_knots_split_complete_chart():
    surface=NurbsSurface(1,1,(0.,0.,.25,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((.25,0.,0.),(.25,1.,.25)),((1.,0.,0.),(1.,1.,1.))))
    result=bound_surface_curvature(surface)
    assert {t.chart_box[0] for t in result.tiles} == {(Q(0),Q(1,4)),(Q(1,4),Q(1))}
    assert all(t.sigma_min_squared_lower>0 for t in result.tiles)

def test_reversed_cropped_chart_is_covered():
    result=bound_surface_curvature(replace(saddle(),u_range=(.75,.25)),((Q(1,4),Q(3,4)),(0,1)))
    assert result.chart_box == ((Q(1,4),Q(3,4)),(Q(0),Q(1)))
    assert result.tiles and all(t.curvature_upper>0 for t in result.tiles)

def test_flat_plane_and_singular_source_refusal():
    result=bound_surface_curvature(Plane((0,0,0),(2,0,0),(.5,3,0)))
    assert result.tiles[0].curvature_upper==0 and result.tiles[0].radius_lower is None
    collapsed=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(1.,0.,0.)),((1.,0.,0.),(2.,0.,0.))))
    with pytest.raises(LocalApproximationNotCertified,match='depth'):
        bound_surface_curvature(collapsed,policy=IntersectionQualificationPolicy(max_subdivision_depth=1))

def test_cancellation_and_source_mutation_refuse():
    with pytest.raises(LocalApproximationNotCertified,match='cancelled'):
        bound_surface_curvature(saddle(),cancellation_check=lambda phase:True)
    surface=saddle(); changed=False
    def check(phase):
        nonlocal changed
        if not changed:
            changed=True;object.__setattr__(surface,'u_range',(.25,.75))
        return False
    with pytest.raises(LocalApproximationNotCertified,match='changed'):
        bound_surface_curvature(surface,cancellation_check=check)

def test_existing_work_and_budget_are_honored():
    from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified
    from anygeometry.section_audit import Work
    token=CURRENT_WORK.set(Work(max_boxes=0))
    try:
        with pytest.raises(PlaneNurbsNotCertified): bound_surface_curvature(saddle())
    finally: CURRENT_WORK.reset(token)
    with pytest.raises(LocalApproximationNotCertified,match='budget'):
        bound_surface_curvature(saddle(),policy=IntersectionQualificationPolicy(max_boxes_per_pair=1))

def test_cubic_quadratic_paraboloid_bounds_known_principal_curvature():
    # Exact Bernstein coefficients for (u,v,u^2+v^2), degree (3,2).
    controls=tuple(tuple((i/3,j/2,i*(i-1)/6+j*(j-1)/2) for j in range(3)) for i in range(4))
    surface=NurbsSurface(3,2,(0.,)*4+(1.,)*4,(0.,)*3+(1.,)*3,controls)
    result=bound_surface_curvature(surface)
    origin=next(t for t in result.tiles if t.chart_box[0][0]==0 and t.chart_box[1][0]==0)
    # Derive exact origin curvatures from the stored double thirds.
    third=Q(1/3)
    assert origin.curvature_upper>=max(Q(2),6*third/(3*third)**2)
    assert all(t.radius_lower is not None and t.radius_lower>0 for t in result.tiles)

def test_positive_rational_quadratic_curvature_and_regular_subdivision():
    # At u=0: x'=2, z'=0, z''=2, hence principal curvature=1/2.
    surface=NurbsSurface(2,1,(0.,0.,0.,1.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((.5,0.,0.),(.5,1.,0.)),((1.,0.,1.),(1.,1.,1.))),
        weights=((1.,1.),(2.,2.),(1.,1.)))
    result=bound_surface_curvature(surface)
    assert result.subdivisions>0
    assert all(t.sigma_min_squared_lower>0 and t.radius_lower is not None for t in result.tiles)
    assert any(t.chart_box[0][0]==0 and t.curvature_upper>=Q(1,2) for t in result.tiles)
    # Exact dyadic chart-area coverage, not a sampled acceptance criterion.
    assert sum((t.chart_box[0][1]-t.chart_box[0][0])*(t.chart_box[1][1]-t.chart_box[1][0])
               for t in result.tiles)==1
