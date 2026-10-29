"""A partial angular chart chooses its nearest winding without new tolerance."""
import math
import numpy as np
import pytest

from anygeometry import Cone, Cylinder


@pytest.mark.parametrize("kind", ["cone", "cylinder"])
@pytest.mark.parametrize("start", [.2, 3., -3., 7.])
@pytest.mark.parametrize("direction", [-1., 1.])
def test_partial_chart_preserves_tiny_outside_start_residual(kind, start, direction):
    args = ((0, 0, 0), (0, 0, 1), (1, 0, 0))
    surface = (Cone(*args, 2., 3., 4., start, direction*math.pi/2)
               if kind == "cone" else Cylinder(*args, 2., 4., start, direction*math.pi/2))
    for u in (-1e-10, 0., .2, .8, 1., 1.+1e-10):
        point = surface.evaluate(u, .7)
        restored = surface.local_uv(point)
        assert restored == pytest.approx((u, .7), abs=2e-14)
        assert np.linalg.norm(surface.evaluate(*restored)-point) < 1e-13
    assert surface.local_uv(surface.evaluate(-.01, .7))[0] < 0


def test_imported_cone_start_seam_uses_nearest_chart_branch():
    surface = Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 2., 3., 4., .2, math.pi/2)
    point = (2.9401997335238117, .5960079923851305, 4.000000000000041)
    u, v = surface.local_uv(point)
    assert abs(u) < 1e-12
    assert v == pytest.approx(1., abs=2e-14)
    # Preserve the nonzero angular residual, rather than expanding a tolerance.
    assert u <= 0


@pytest.mark.parametrize("kind", ["cone", "cylinder"])
@pytest.mark.parametrize("direction", [-1., 1.])
def test_public_batch_inverse_uses_same_partial_chart_branch(kind, direction):
    from anygeometry import GeometryModel, OrientedEdge
    args = ((0, 0, 0), (0, 0, 1), (1, 0, 0))
    surface = (Cone(*args, 2., 3., 4., .2, direction*math.pi/2)
               if kind == "cone" else Cylinder(*args, 2., 4., .2, direction*math.pi/2))
    model = GeometryModel()
    lower = model.add_points(surface.evaluate(u, 0.) for u in (0., .5, 1.))
    upper = model.add_points(surface.evaluate(u, 1.) for u in (0., .5, 1.))
    edges = (model.add_arc(*lower), model.add_line(lower[-1], upper[-1]),
             model.add_arc(*upper), model.add_line(lower[0], upper[0]))
    face = model.add_face_from_loop(tuple(OrientedEdge(e, forward) for e, forward in
        zip(edges, (True, True, False, False))), (0, 1, 2, 3), surface=surface)
    points = np.array([surface.evaluate(u, .7) for u in (-1e-10, 0., .2, .8, 1., 1.+1e-10)])
    expected = np.clip(np.array([surface.local_uv(p) for p in points]), 0, 1)
    assert model.face_local_uv_many(face, points) == pytest.approx(expected, abs=2e-14)
    assert model.face_local_uv_many(face, points)[0, 0] == 0.
