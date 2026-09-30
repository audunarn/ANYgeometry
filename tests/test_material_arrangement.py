import math
import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryError, GeometryModel
from anygeometry.arrangement_geometry import LinePath, curve_junctions
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material


def test_cylinder_tip_ray_has_one_half_open_crossing_and_complete_area():
    from anygeometry import Cylinder,CylinderIntersectionCurve
    support=Cylinder((0,0,-1.5),(0,0,1),(1,0,0),1,3,7*math.pi/4,math.pi/4)
    second=Cylinder((0,-1.5,0),(0,1,0),(1,0,0),1,3)
    lower=CylinderIntersectionCurve(support,second,7*math.pi/4,math.pi/4,-1)
    upper=CylinderIntersectionCurve(support,second,7*math.pi/4,math.pi/4,1)
    boundary=(ArrangementPath(LinePath(tuple(upper.evaluate(0.)),tuple(lower.evaluate(0.)))),
              ArrangementPath(lower.subcurve(0.,.5)),ArrangementPath(lower.subcurve(.5,1.)),
              ArrangementPath(upper.subcurve(1.,.5)),ArrangementPath(upper.subcurve(.5,0.)))
    domain=MaterialDomain(1,support,(boundary,))
    assert domain.contains_loop(boundary,np.asarray((-.0001,.5)),1e-9,boundary=False) is False
    assert domain.contains_loop(boundary,np.asarray((.5,.5)),1e-9,boundary=False) is True
    arranged=arrange_material(domain,(),tolerance=1e-9,area_tolerance=1e-9)
    assert len(arranged.cells)==1
    assert arranged.area*domain.area_jacobian==pytest.approx(2*(1-math.cos(math.pi/4)),abs=1e-10)


def _plate(points=((0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0))):
    model = GeometryModel()
    face = model.add_plate(model.add_points(points))
    return model, MaterialDomain.from_model(model, face)


def test_exact_line_ellipse_and_ellipse_ellipse_roots():
    circle = EllipticArc((0, 0, 0), (1, 0, 0), (0, 1, 0), 0, math.tau)
    line = LinePath((-2, 0, 0), (2, 0, 0))
    hits = curve_junctions(line, circle)
    assert sorted(a for a, _b in hits) == pytest.approx((.25, .75, .75))
    other = EllipticArc((1, 0, 0), (1, 0, 0), (0, 1, 0), 0, math.tau)
    hits = curve_junctions(circle, other)
    assert len(hits) == 2
    for a, b in hits:
        np.testing.assert_allclose(circle.evaluate(a), other.evaluate(b), atol=1e-12)
        assert circle.evaluate(a)[0] == pytest.approx(.5)


def test_crossing_stubs_high_valence_and_retained_material():
    _model, domain = _plate()
    traces = [ArrangementPath(LinePath((1, 2.5, 0), (4, 2.5, 0)), owners=(1, 2)),
              ArrangementPath(LinePath((2.5, 1, 0), (2.5, 4, 0)), owners=(1, 3))]
    result = arrange_material(domain, traces, tolerance=1e-10)
    assert result.area == pytest.approx(1.)  # Native plate parameters are unit-square.
    assert len(result.cells) >= 4
    assert any(path.decomposition for path in result.paths)
    physical = [path for path in result.paths if path.owners]
    assert sum(np.linalg.norm(path.curve.evaluate(1)-path.curve.evaluate(0)) for path in physical) == pytest.approx(6.)
    assert sum(np.linalg.norm(path.curve.evaluate(0)-np.array((2.5, 2.5, 0))) < 1e-10 or
               np.linalg.norm(path.curve.evaluate(1)-np.array((2.5, 2.5, 0))) < 1e-10 for path in physical) == 4


def test_circle_and_crossing_plates_preserve_exact_curves_and_area():
    _model, domain = _plate()
    traces = [ArrangementPath(LinePath((0, 2.5, 0), (5, 2.5, 0)), owners=(1, 2)),
              ArrangementPath(LinePath((2.5, 0, 0), (2.5, 5, 0)), owners=(1, 3))]
    for start in (0., math.pi):
        traces.append(ArrangementPath(EllipticArc((2.5, 2.5, 0), (1, 0, 0), (0, 1, 0), start, math.pi), owners=(1, 4)))
    result = arrange_material(domain, traces, tolerance=1e-10)
    assert result.area == pytest.approx(1.)
    assert len(result.cells) == 8
    arcs = [path for path in result.paths if isinstance(path.curve, EllipticArc)]
    assert sum(abs(path.curve.sweep_angle) for path in arcs) == pytest.approx(math.tau)


def test_closed_loop_retains_inside_and_outside_as_material_cells():
    _model, domain = _plate()
    traces = [ArrangementPath(EllipticArc((2.5, 2.5, 0), (1, 0, 0), (0, 1, 0), start, math.pi), owners=(1, 2))
              for start in (0., math.pi)]
    result = arrange_material(domain, traces, tolerance=1e-10)
    assert result.area == pytest.approx(1.)
    assert len(result.cells) >= 2


def test_concave_domain_and_explicit_resource_failures():
    _model, domain = _plate(((0,0,0), (4,0,0), (4,1,0), (1,1,0), (1,4,0), (0,4,0)))
    trace = ArrangementPath(LinePath((0, .5, 0), (4, .5, 0)), owners=(1, 2))
    result = arrange_material(domain, (trace,), tolerance=1e-10)
    assert len(result.cells) == 2
    with pytest.raises(GeometryError, match='cancelled'):
        arrange_material(domain, (trace,), tolerance=1e-10, cancellation_check=lambda: True)
    with pytest.raises(GeometryError, match='budget'):
        arrange_material(domain, (trace,), tolerance=1e-10, max_predicates=1)
