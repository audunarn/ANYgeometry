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


def test_independent_failure_is_not_relabelled_as_budget_exhaustion():
    from anygeometry import apply_plane_split
    from anygeometry.entities import EntityRef
    from anygeometry.native_support_snapshots import capture_native_supports
    from anygeometry.plane_split import QualificationBudgetExhausted
    from anygeometry.policies import MutationPolicy
    model, face = _bowl_face()
    plane = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    plan = plan_plane_split(model, [face], plane, qualification=IntersectionQualificationPolicy(max_boxes_per_pair=8000))
    apply_plane_split(model, plan, policy=MutationPolicy.IMPRINT)
    capture_native_supports(model)
    model._groups['bad'] = {EntityRef('face', 10 ** 9)}      # an invalid group no budget can repair
    with pytest.raises(GeometryError) as info:
        plan_plane_split(model, [sorted(model.faces)[-1]], Plane((0., 0., .25), (1., 0., 0.), (0., 1., 0.)),
                         qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1))
    error = info.value
    assert not isinstance(error, QualificationBudgetExhausted)
    assert "cannot serialize invalid topology" in str(error)
    # the budget facts stay visible on the original exception
    assert error.budget_denied is True and error.budget_limit == 1 and error.budget_spent > 1


def test_work_take_is_safe_when_called_directly_and_records_the_declared_consumer():
    from anygeometry.section_audit import Work
    assert Work(0).take() is False            # no stack inspection: a module-level call works
    work = Work(1)
    work.consumer = "edge_edge"
    assert work.take() is True and work.take() is False
    assert work.exhausted_by == "edge_edge"
    assert Work(1).exhausted_by is None
