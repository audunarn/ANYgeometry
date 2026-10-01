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


def test_sheared_concave_chart_keeps_full_vertical_cut_at_reflex_height():
    from anygeometry.material_arrangement import _clip
    _model,domain=_plate(((0,0,0),(6,0,0),(6,2,0),(3,2,0),(3,4,0),(0,4,0)))
    # A horizontal boundary can have a tiny computed affine-chart slope.
    cut=LinePath((1.25,-8.223295267641177,0),(1.25,8.523295267641178,0))
    parts=_clip(domain,cut,1e-8,lambda:None)
    assert len(parts)==1
    np.testing.assert_allclose(parts[0].evaluate(0),(1.25,0,0),atol=1e-12)
    np.testing.assert_allclose(parts[0].evaluate(1),(1.25,4,0),atol=1e-12)
    for y in (2.,2.-1e-12,2.+1e-12):
        assert domain.contains(LinePath((1.25,y,0),(1.25,y,0)),0.,1e-8)


def test_decomposition_stub_directions_use_physical_orthogonality():
    _model,domain=_plate(((0,0,0),(6,0,0),(6,2,0),(3,2,0),(3,4,0),(0,4,0)))
    trace=ArrangementPath(LinePath((.1,.1,0),(2.9,2.9,0)),owners=(1,2))
    arranged=arrange_material(domain,(trace,),tolerance=1e-10)
    assert arranged.area*domain.area_jacobian==pytest.approx(18.)
    authored=np.asarray((1.,0.,0.))
    for endpoint in (trace.curve.evaluate(0.),trace.curve.evaluate(1.)):
        incident=[]
        for path in arranged.paths:
            if not path.decomposition:continue
            if any(np.linalg.norm(path.curve.evaluate(t)-endpoint)<1e-9 for t in (0.,1.)):
                direction=path.curve.derivative(.5);direction/=np.linalg.norm(direction)
                incident.append(direction)
        assert incident
        assert all(abs(np.dot(direction,authored))<1e-12 or
                   abs(abs(np.dot(direction,authored))-1)<1e-12 for direction in incident)


def test_reversed_duplicate_bezier_boundary_is_one_exact_material_edge():
    from anygeometry import Plane
    from anygeometry.arrangement_geometry import BezierPath
    curve = BezierPath(((.5,.5,0),(1.5,1.5,0),(2.5,.5,0)))
    boundary = (
        ArrangementPath(LinePath((0,0,0),(3,0,0))),
        ArrangementPath(LinePath((3,0,0),(3,.5,0))),
        ArrangementPath(LinePath((3,.5,0),(2.5,.5,0))),
        ArrangementPath(curve.subcurve(1.,0.)),
        ArrangementPath(LinePath((.5,.5,0),(0,.5,0))),
        ArrangementPath(LinePath((0,.5,0),(0,0,0))),
    )
    domain = MaterialDomain(1, Plane((0,0,0),(1,0,0),(0,1,0)), (boundary,))
    arranged = arrange_material(domain,(ArrangementPath(curve,owners=(1,2)),),tolerance=1e-10)
    # Integral under quadratic y=.5+2t(1-t), x=.5+2t, plus end strips.
    assert arranged.area == pytest.approx(13/6,abs=1e-12)
    assert len(arranged.cells) == 1
    assert sum(isinstance(path.curve,BezierPath) for path in arranged.paths) == 1
