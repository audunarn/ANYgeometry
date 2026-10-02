"""Owner projection preserves exact branch stations and global distance bounds."""

import numpy as np
import pytest

from anygeometry import to_dict
from anygeometry.branch_algebra import BezierRuledSupport
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.exact_curves import project_analytic_curve
from anygeometry.quadric_algebra import QuadricSupport
from test_branch_models import SCENARIOS, run


@pytest.fixture(scope="module")
def cut_wall():
    model, faces = SCENARIOS["a pipe across the cubic wall"][0]()
    run(model, faces)
    return model


@pytest.mark.parametrize("face,edge,parameter", [
    (11, 81, .3454125133797817),
    (11, 86, .2843607609103987),
    (13, 80, .35190340622368593),
])
def test_public_projection_preserves_exact_trim_boundary_station(cut_wall, face, edge, parameter):
    model = cut_wall
    before, revision = to_dict(model), model.revision
    target = model.sample_edge(edge, np.asarray([parameter]))[0]
    saved_target = target.copy()
    local = model.face_local_uv(face, target)
    assert np.linalg.norm(model.face_point(face, *local) - target) < 1e-10
    # These stations lie outside the sampled trim polygon, so projection must
    # exercise the exact boundary-curve path rather than its interior shortcut.
    assert not model._face_contains_uv_polygons(np.asarray(local), model.face_trim_loops_uv(face))
    point, _parameter, distance = model.closest_edge_point(edge, target)
    assert distance < 1e-10
    assert np.linalg.norm(point - target) < 1e-10
    projected, uv, distance = model.project_to_face(face, target)
    assert distance < 1e-10
    assert np.linalg.norm(projected - target) < 1e-10
    assert np.linalg.norm(model.face_point(face, *uv) - target) < 1e-10
    assert model.revision == revision and to_dict(model) == before
    np.testing.assert_array_equal(target, saved_target)


def straight_branch():
    first = BezierRuledSupport(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.)), (0., 0., 1.))
    second = QuadricSupport("general", matrix=(0., 0., 0., 0., 0., 0., 0., 0., 1.),
                            linear=(0., 0., 0.), constant=-1.)
    return BezierQuadricCurve(first, second, 0., 1.)  # c(t)=(t,0,1)


@pytest.mark.parametrize("x", [-.2, .37, 1.2])
@pytest.mark.parametrize("affine", [False, True])
def test_off_curve_projection_retains_global_segment_distance(x, affine):
    curve = straight_branch()
    if affine:
        curve = curve.transformed(np.asarray(((2., .3, 0., 1.), (.5, 1., 0., -2.),
                                               (0., 0., 3., .4), (0., 0., 0., 1.))))
    first, last = curve.evaluate(np.asarray((0., 1.)))
    direction = last - first
    target = first + x * direction + np.asarray((0., 0., .4))
    expected = first + np.clip(x, 0., 1.) * direction
    minimum = float(np.linalg.norm(expected - target))
    tolerance = 1e-9
    point, parameter, distance = project_analytic_curve(curve, target, tolerance)
    assert minimum - 1e-14 <= distance <= minimum + tolerance
    np.testing.assert_allclose(point, curve.evaluate(parameter), rtol=0., atol=1e-14)
    assert abs(np.linalg.norm(point - target) - distance) < 1e-14


def test_polishing_cannot_replace_a_better_certified_candidate(monkeypatch):
    curve = straight_branch()
    # This candidate improves the initial sampled upper bound but is worse than
    # the globally certified result. A proposed polish must not degrade it.
    monkeypatch.setattr(BezierQuadricCurve, "_polish_point", lambda self, point, parameter: .4)
    target = np.asarray((.37, .4, 1.))
    _point, _parameter, distance = project_analytic_curve(curve, target, 1e-9)
    assert .4 <= distance <= .4 + 1e-9
