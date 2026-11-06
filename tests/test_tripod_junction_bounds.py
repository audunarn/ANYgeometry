import numpy as np
import pytest
from anygeometry import Cylinder, Plane, GeometryError
from anygeometry.analytic_supports import cylinder_cylinder_support, plane_cylinder_support
from anygeometry.arrangement_geometry import (LinePath, curve_junctions, _junction_bounds_policy,
                                             _separated_line_pair)
from anygeometry.exact_curves import EllipticArc
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics


@pytest.fixture(autouse=True)
def enable_experimental_rejection():
    with _junction_bounds_policy(True):
        yield


def reference(first, second, tol=1e-10):
    with _junction_bounds_policy(False):
        return curve_junctions(first, second, tolerance=tol)


def test_long_line_normalized_endpoint_slack_is_not_pruned():
    first = LinePath((0.,0.,0.), (100.,0.,0.))
    second = LinePath((100.+5e-9,-1.,0.), (100.+5e-9,1.,0.))
    expected = reference(first, second)
    assert expected
    assert curve_junctions(first, second) == expected


@pytest.mark.parametrize('reverse', (False, True))
def test_degenerate_line_refusal_precedes_rejection(reverse):
    pair = [LinePath((50.,0.,0.), (50.,0.,0.)), EllipticArc((0,0,0),(1,0,0),(0,1,0))]
    if reverse:
        pair.reverse()
    with pytest.raises(GeometryError, match='degenerate'):
        curve_junctions(*pair)


def test_cancellation_precedes_rejection():
    a, b = LinePath((0,0,0),(1,0,0)), LinePath((100,0,0),(101,0,0))
    with pytest.raises(GeometryError, match='cancelled'):
        curve_junctions(a,b,cancellation_check=lambda: True)


def test_random_line_ellipse_pairs_match_unpruned_predicate():
    rng = np.random.default_rng(341)
    with collect_runtime_diagnostics() as counts:
        for i in range(60):
            center = rng.normal(size=3) * (10 if i % 2 else .2)
            arc = EllipticArc(tuple(center), (1.,0.,0.), (0.,.7,0.), -.3, 4.)
            line = LinePath(tuple(rng.normal(size=3)), tuple(rng.normal(size=3)))
            assert curve_junctions(line,arc) == reference(line,arc)
            assert curve_junctions(arc,line) == reference(arc,line)
        assert counts['junction_line_bounds_pruned'] > 0


def test_finite_branch_pairs_match_reference_and_transformed_chart_is_excluded():
    cylinder = Cylinder((0,0,-2),(0,0,1),(1,0,0),1,4)
    plane_curves = plane_cylinder_support(Plane((0,0,0),(1e-16,0,1),(0,1,0)),cylinder).curves
    crossing = Cylinder((-2,0,0),(1,0,0),(0,1,0),1,4)
    curves = (*plane_curves, *cylinder_cylinder_support(cylinder,crossing).curves)
    assert curves
    for curve in curves:
        for z in (0.,20.):
            line = LinePath((-2.,0.,z),(2.,0.,z))
            assert curve_junctions(line,curve) == reference(line,curve)
    matrix = np.diag([100.,100.,100.,1.])
    transformed = plane_curves[0].transformed(matrix)
    assert not _separated_line_pair(LinePath((1000,0,0),(1001,0,0)), transformed, 1e-10)


def test_direct_arrangement_budget_and_outer_policy_disable_rejection(monkeypatch):
    from anygeometry import GeometryModel
    from anygeometry.material_arrangement import MaterialDomain, ArrangementPath, arrange_material
    import anygeometry.arrangement_geometry as predicates
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(5,0,0),(5,5,0),(0,5,0))))
    domain = MaterialDomain.from_model(model, face)
    traces = tuple(ArrangementPath(EllipticArc((2.5,2.5,0),(1,0,0),(0,1,0),start,np.pi),owners=(1,2))
                   for start in (0.,np.pi))
    flags = []
    original = predicates._separated_line_pair
    def observed(*args):
        flags.append(predicates._JUNCTION_BOUNDS.get())
        return original(*args)
    monkeypatch.setattr(predicates, '_separated_line_pair', observed)
    result = arrange_material(domain,traces,tolerance=1e-10,max_predicates=100000)
    assert result.cells and flags and not any(flags)
    flags.clear()
    with _junction_bounds_policy(False):
        arrange_material(domain,traces,tolerance=1e-10)
    assert flags and not any(flags)
