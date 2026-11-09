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


def _assert_budget_facts(error, cap):
    assert error.budget_limit == cap and error.budget_spent > cap and error.budget_denied >= 1
    assert any("qualification budget was denied" in note for note in error.__notes__)


@pytest.mark.parametrize("cap", [3000, 7000])
def test_small_budget_refusal_keeps_its_type_and_carries_the_budget_facts(cap):
    with pytest.raises(GeometryError) as info:
        _plan(cap)
    assert type(info.value) is GeometryError          # the original class and message are untouched
    _assert_budget_facts(info.value, cap)


def test_sufficient_budget_still_plans():
    plan = _plan(8000)
    assert plan is not None


def test_independent_failure_is_not_relabelled_as_budget_exhaustion():
    from anygeometry import apply_plane_split
    from anygeometry.entities import EntityRef
    from anygeometry.native_support_snapshots import capture_native_supports
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
    assert type(error) is GeometryError
    assert "cannot serialize invalid topology" in str(error)
    # the budget facts stay visible on the original exception
    _assert_budget_facts(error, 1)


def test_empty_selection_keeps_its_original_refusal_at_every_cap():
    model, _ = _bowl_face()
    plane = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    for cap in (1, 2):
        with pytest.raises(GeometryError, match="requires selected faces") as info:
            plan_plane_split(model, [], plane, qualification=IntersectionQualificationPolicy(max_boxes_per_pair=cap))
        assert type(info.value) is GeometryError


def test_work_counts_denied_takes_and_keeps_its_old_take_behaviour():
    from anygeometry.section_audit import Work
    assert Work(0).take() is False
    work = Work(1)
    assert [work.take(), work.take(), work.take()] == [True, False, False]
    assert work.boxes == 3 and work.denied == 2
