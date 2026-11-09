"""A plane split that runs out of its qualification budget says so, with typed counters."""
import pytest
from anygeometry import GeometryModel, GeometryError, Plane, plan_plane_split
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.predicates import IntersectionQualificationPolicy


def _bowl_face():
    a = (1., -1., 1.)
    surface = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
        tuple(tuple((i - 1., j - 1., a[i] + a[j]) for j in range(3)) for i in range(3)))
    model = GeometryModel()
    pts = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    vs = [model.add_point(*surface.evaluate(*p)) for p in pts]
    edges = [model.add_curve(vs[i], vs[(i + 1) % 4],
                             CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (pts[i], pts[(i + 1) % 4]))))
             for i in range(4)]
    return model, model.add_face(edges, surface=surface)


def _plan(cap):
    model, face = _bowl_face()
    return plan_plane_split(model, [face], Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.)),
                            qualification=IntersectionQualificationPolicy(max_boxes_per_pair=cap))


@pytest.mark.parametrize("cap", [3000, 7000])
def test_small_budget_refusal_names_the_exhausted_budget(cap):
    from anygeometry.plane_split import QualificationBudgetExhausted
    with pytest.raises(QualificationBudgetExhausted) as info:
        _plan(cap)
    error = info.value
    assert isinstance(error, GeometryError)
    assert "qualification budget exhausted" in str(error)
    assert error.limit == cap and error.spent > cap and error.subdivisions > 0
    assert error.consumer == "extreme"


def test_sufficient_budget_still_plans():
    plan = _plan(8000)
    assert plan is not None
