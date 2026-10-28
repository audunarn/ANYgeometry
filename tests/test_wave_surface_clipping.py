import numpy as np
import pytest

from anygeometry.wave_surface_clipping import clip_polygon_positive, clip_triangle_positive


def test_dry_wet_and_boundary_states():
    assert clip_triangle_positive([-1., -2., 0.]).state == 'dry'
    assert clip_triangle_positive([0., 0., 0.]).state == 'dry'
    full = clip_triangle_positive([1., 2., 0.])
    assert full.state == 'wet' and full.area_fraction == 1.
    assert clip_triangle_positive([1., 1., 1.]).parameters.shape == (3, 3)


def test_one_vertex_wet_cuts_corner_with_exact_area_fraction():
    # Field linear on the triangle: wet corner has value 3, others -1 -> cut at t=3/4 on both edges.
    clip = clip_triangle_positive([3., -1., -1.])
    assert clip.state == 'partial'
    assert clip.area_fraction == pytest.approx((3 / 4) ** 2)
    assert np.allclose(clip.parameters.sum(axis=1), 1.)
    assert np.allclose(clip.values[[1, 2]], 0.)


def test_two_vertices_wet_is_complement_of_corner():
    clip = clip_triangle_positive([2., 2., -1.])
    assert clip.area_fraction == pytest.approx(1. - (1 / 3) ** 2)
    assert len(clip.fan) == len(clip.parameters) - 2


def test_clipped_polygon_lies_exactly_on_zero_level_of_linear_interpolant():
    g = np.array([.7, -.4, -.2])
    clip = clip_triangle_positive(g)
    assert np.allclose(clip.parameters @ g, clip.values)
    assert np.all(clip.values >= -1e-15)


def test_square_polygon_in_reference_coordinates():
    square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    clip = clip_polygon_positive(square, square[:, 0] + 0.)  # x > 0: half
    assert clip.area_fraction == pytest.approx(.5)
    assert clip.parameters[:, 0].min() >= -1e-15


def test_rejects_nonfinite_and_wrong_shapes():
    with pytest.raises(ValueError):
        clip_triangle_positive([1., np.nan, 0.])
    with pytest.raises(ValueError):
        clip_triangle_positive([1., 2.])
