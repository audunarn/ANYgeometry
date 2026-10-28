from fractions import Fraction as Q
import pytest
from anygeometry._local_neighborhood import (certify_surface_star,outside_rectangles,
    projected_hull,separation_squared,curve_curvature_from_jet)
from anygeometry._local_curvature import LocalApproximationNotCertified
from anygeometry.nurbs_surface_intersection._intervals import Interval as I
from anygeometry.nurbs_surface_intersection._bernstein import subbox

def grid():
    return (((Q(0),Q(0),Q(0),Q(1)),(Q(0),Q(1),Q(0),Q(1))),
            ((Q(1),Q(0),Q(0),Q(1)),(Q(1),Q(1),Q(0),Q(1))))

def test_exact_buffered_affine_star_and_complete_outside_cover():
    inner=((Q(2,5),Q(3,5)),)*2;outer=((Q(1,4),Q(3,4)),)*2
    result=certify_surface_star(grid(),inner,outer)
    assert result.strong_monotonicity_lower==1
    assert 0<result.inner_to_outer_boundary_lower<=Q(3,20)
    rectangles=outside_rectangles(outer)
    assert sum((b[0][1]-b[0][0])*(b[1][1]-b[1][0]) for b in rectangles)==Q(3,4)
    inside=projected_hull(subbox(grid(),*inner[0],*inner[1]))
    assert all(separation_squared(inside,projected_hull(subbox(grid(),*b[0],*b[1])))>0 for b in rectangles)

def test_zero_buffer_and_wrong_projection_refuse():
    with pytest.raises(LocalApproximationNotCertified,match='buffer'):
        certify_surface_star(grid(),((0,Q(1,2)),)*2,((0,1),)*2)
    with pytest.raises(LocalApproximationNotCertified,match='monotonicity'):
        certify_surface_star(grid(),((Q(1,3),Q(2,3)),)*2,((0,1),)*2,projection=((0,0,0),(0,0,0)))

def test_remote_competitor_cannot_be_excluded_by_local_star():
    hull=projected_hull(grid())
    assert separation_squared(hull,hull)==0  # folded/remote coincidence remains unresolved

def test_curve_curvature_small_hole_is_not_flat_support_curvature():
    upper,radius=curve_curvature_from_jet((I(0),I(Q(1,10)),I(0)),(I(-Q(1,10)),I(0),I(0)))
    assert upper>=10 and 0<radius<=Q(1,10)
    with pytest.raises(LocalApproximationNotCertified,match='regularity'):
        curve_curvature_from_jet((I(-1,1),I(0),I(0)),(I(0),I(1),I(0)))

def test_one_sided_source_corner_preserves_internal_buffer():
    result=certify_surface_star(grid(),((0,Q(1,4)),)*2,((0,Q(1,2)),)*2,
        source_boundary_sides=((0,0),(1,0)))
    assert result.parameter_buffer==Q(1,4)
    assert result.inner_to_outer_boundary_lower>0
    with pytest.raises(LocalApproximationNotCertified,match='buffer'):
        certify_surface_star(grid(),((0,Q(1,2)),)*2,((0,Q(1,2)),)*2,
            source_boundary_sides=((0,0),(1,0)))

def test_whole_source_star_has_no_invented_outside_buffer():
    result=certify_surface_star(grid(),((0,1),)*2,((0,1),)*2,
        source_boundary_sides=((0,0),(0,1),(1,0),(1,1)))
    assert result.parameter_buffer is None and result.inner_to_outer_boundary_lower is None
    assert result.strong_monotonicity_lower>0

def test_internal_knot_tile_cannot_declare_its_edge_a_source_endpoint():
    with pytest.raises(LocalApproximationNotCertified,match='source-chart endpoint'):
        certify_surface_star(grid(),((Q(1,4),Q(1,2)),)*2,((Q(1,4),Q(3,4)),)*2,
            source_boundary_sides=((0,0),(1,0)))

@pytest.mark.parametrize('u,v',[((1,0,0),(10,1,0)),((2,0,0),(10,3,0))])
def test_affine_sheared_parameterization_uses_exact_left_inverse(u,v):
    made=tuple(tuple(tuple(Q(i*u[k]+j*v[k]) for k in range(3))+(Q(1),) for j in range(2)) for i in range(2))
    result=certify_surface_star(made,((Q(1,4),Q(3,4)),)*2,((0,1),)*2)
    assert result.strong_monotonicity_lower==1
