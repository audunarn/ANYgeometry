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


@pytest.mark.parametrize("cap", [1000, 1500])  # was [3000, 7000] before the _tight_x memo (about 1,700 boxes are needed now)
def test_small_budget_refusal_keeps_its_type_and_carries_the_budget_facts(cap):
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified
    with pytest.raises(GeometryError) as info:
        _plan(cap)
    assert type(info.value) is PlaneNurbsNotCertified and 'budget' in str(info.value)   # G1: typed at the source
    _assert_budget_facts(info.value, cap)


def test_sufficient_budget_still_plans():
    plan = _plan(2500)
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


def _denying_ids(error, takes):
    """A face-id generator that overruns the budget (swallowing the refusal) and then raises ``error``."""
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified, charge_box
    for _ in range(takes):
        try:
            charge_box()
        except PlaneNurbsNotCertified:
            pass
    raise error
    yield  # pragma: no cover


_PLANE = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))


def _plan_ids(model, ids, cap):
    return plan_plane_split(model, ids, _PLANE, qualification=IntersectionQualificationPolicy(max_boxes_per_pair=cap))


def test_shared_exception_keeps_the_facts_of_its_first_plan():
    model, _ = _bowl_face()
    shared = GeometryError("shared refusal")
    with pytest.raises(GeometryError):
        _plan_ids(model, _denying_ids(shared, 2), 1)
    first = (shared.budget_limit, shared.budget_spent, shared.budget_denied)
    assert first == (1, 2, 1) and len(shared.__notes__) == 1
    with pytest.raises(GeometryError):
        _plan_ids(model, _denying_ids(shared, 5), 3)
    assert (shared.budget_limit, shared.budget_spent, shared.budget_denied) == first
    assert len(shared.__notes__) == 1


def test_nested_plans_decorate_the_propagated_exception_once():
    model, _ = _bowl_face()
    shared = GeometryError("inner refusal")

    def outer_ids():
        from anygeometry.plane_nurbs import PlaneNurbsNotCertified, charge_box
        for _ in range(2):
            try:
                charge_box()
            except PlaneNurbsNotCertified:
                pass
        _plan_ids(model, _denying_ids(shared, 2), 1)
        yield  # pragma: no cover

    with pytest.raises(GeometryError):
        _plan_ids(model, outer_ids(), 1)
    assert len(shared.__notes__) == 1 and shared.budget_limit == 1


def test_undecoratable_exception_escapes_unchanged():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Frozen(GeometryError):
        message: str

    model, _ = _bowl_face()
    with pytest.raises(Frozen) as info:
        _plan_ids(model, _denying_ids(Frozen("frozen refusal"), 2), 1)
    assert str(info.value) == "frozen refusal"


# ---- G1 rework: single attempt, no replay (typed from facts of the attempt itself)

def test_one_shot_iterator_input_is_typed_like_a_list():
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified
    for cap in (1000, 1500):
        model, face = _bowl_face()
        with pytest.raises(GeometryError) as info:
            _plan_ids(model, iter([face]), cap)
        assert type(info.value) is PlaneNurbsNotCertified and 'budget' in str(info.value)
        _assert_budget_facts(info.value, cap)


@pytest.mark.parametrize("cap", [1, 64, 1000, 1500])
def test_the_callers_cap_is_never_exceeded_by_more_than_one_denied_take_per_search_step(cap):
    # one attempt: spent counts the attempt's takes (denied ones included); a replay under a larger budget would
    # add thousands. Measured 2026-11-13: cap 1000 -> spent 1001, cap 1500 -> 1502 (the old replay design: +1717).
    with pytest.raises(GeometryError) as info:
        _plan(cap)
    assert info.value.budget_limit == cap and cap < info.value.budget_spent <= cap + 64


def test_independent_refusal_after_a_denial_keeps_its_class_and_gets_the_facts():
    model, _ = _bowl_face()
    plane = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    with pytest.raises(GeometryError, match="requires selected faces") as info:
        plan_plane_split(model, _denying_ids(GeometryError("requires selected faces"), 3), plane,
                         qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1))
    assert type(info.value) is GeometryError and info.value.budget_denied >= 1


# ---- G1 round 3: an independent permanent defect is never relabelled as budget (per-entry provenance)

def _bowtie_child():
    """Bowl split at z=.5, then a self-intersecting straight-pcurve bowtie hole appended to the child with holes."""
    from dataclasses import replace
    from anygeometry import apply_plane_split
    from anygeometry.entities import OrientedEdge
    from anygeometry.policies import MutationPolicy
    model, face = _bowl_face()
    plan = plan_plane_split(model, [face], Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.)),
                            qualification=IntersectionQualificationPolicy(max_boxes_per_pair=200000))
    apply_plane_split(model, plan, policy=MutationPolicy.IMPRINT)
    child = next(i for i, f in model.faces.items() if f.holes)
    support = model.faces[child].surface
    points = [(.02, .02), (.18, .18), (.02, .18), (.18, .02)]
    vertices = [model.add_point(*support.evaluate(*p)) for p in points]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4],
                             CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.), (p, points[(i + 1) % 4]))))
             for i, p in enumerate(points)]
    current = model.faces[child]
    model._put_entity('face', replace(current, holes=current.holes + (tuple(OrientedEdge(e, True) for e in edges),)))
    return model, child


@pytest.mark.parametrize("z", [.25, .5])
def test_independent_self_intersection_keeps_its_class_at_every_cap(z):
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified
    classes = {}
    for cap in (64, 200000):                    # 64: a denial AND the defect coexist; 200000: the defect alone
        model, child = _bowtie_child()
        with pytest.raises(GeometryError) as info:
            plan_plane_split(model, [child], Plane((0., 0., z), (1., 0., 0.), (0., 1., 0.)),
                             qualification=IntersectionQualificationPolicy(max_boxes_per_pair=cap))
        classes[cap] = type(info.value)
        assert 'self-intersects in the chart' in str(info.value)
        if cap == 64:
            assert info.value.budget_limit == 64 and info.value.budget_denied >= 1   # facts still attached
    assert classes[64] is classes[200000] is GeometryError and not issubclass(classes[64], PlaneNurbsNotCertified)
