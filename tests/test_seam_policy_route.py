"""RING-SPLIT RS-0: the opt-in seam_policy axis (plumbing and route identity).

``IntersectionBatchPolicy.seam_policy`` (None sentinel, "baseline", "ring_split") is independent of
``planning_mode``. Exact mode with baseline seams is plan state E (no route field, byte-identical);
every other combination is a route plan whose route checksum binds both axes. Every test here fails
on the base 788b84f (no seam_policy field).
"""
import pickle
from dataclasses import fields

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, from_dict, to_dict
from anygeometry import batch_intersections as bi
from anygeometry import material_arrangement as ma
from anygeometry import separating_plane_certificate as sp
from anygeometry.arrangement_geometry import LinePath
from anygeometry.batch_intersections import (IntersectionBatchPolicy, IntersectionPlan, TopologyCertifiedPlan,
                                             apply_intersections, plan_intersections)
from anygeometry.definition_binding import definition_value
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material
from anygeometry.surfaces import Plane

RS = IntersectionBatchPolicy(seam_policy="ring_split")
T = IntersectionBatchPolicy(planning_mode="topology_certified")
TRS = IntersectionBatchPolicy(planning_mode="topology_certified", seam_policy="ring_split")


@pytest.fixture(autouse=True)
def _no_environment(monkeypatch):
    monkeypatch.delenv(bi.PLANNING_MODE_ENVIRONMENT, raising=False)
    monkeypatch.delenv(bi.SEAM_POLICY_ENVIRONMENT, raising=False)


def _l_joint():
    model = GeometryModel()
    p = model.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2)))
    edges = {}

    def line(i, j):
        key = (min(i, j), max(i, j))
        if key not in edges:
            edges[key] = model.add_line(p[key[0]], p[key[1]])
        return edges[key]
    first = model.add_face([line(0, 1), line(1, 2), line(2, 3), line(3, 0)])
    second = model.add_face([line(0, 1), line(1, 5), line(5, 4), line(4, 0)])
    return model, (first, second)


def _forge(carrier, **changes):
    made = object.__new__(type(carrier))
    for item in fields(carrier):
        object.__setattr__(made, item.name, changes[item.name] if item.name in changes else getattr(carrier, item.name))
    return made


def _resealed(plan, **changes):
    changed = _forge(plan, **changes)
    try:
        return _forge(changed, route_checksum=bi._route_checksum(changed))
    except (ValueError, TypeError):
        return changed


@pytest.fixture
def plans():
    model, faces = _l_joint()
    return (model, plan_intersections(model, faces, policy="connect"), plan_intersections(model, faces, policy=RS),
            plan_intersections(model, faces, policy=TRS))


# ----------------------------------------------------------------------------------- the policy field

def test_seam_policy_field_values_state_and_definition():
    assert bi.SEAM_POLICIES == ("baseline", "ring_split")
    for bad in ("Ring_split", "ring", "", 1, True):
        with pytest.raises(GeometryError, match="seam_policy"):
            IntersectionBatchPolicy(seam_policy=bad)
    default = IntersectionBatchPolicy()
    assert len(default.__getstate__()) == len(fields(IntersectionBatchPolicy)) - 2       # pre-T2-1b state
    assert len(T.__getstate__()) == len(fields(IntersectionBatchPolicy)) - 1             # T2-1b state
    assert len(RS.__getstate__()) == len(fields(IntersectionBatchPolicy))
    for policy in (default, T, RS, TRS):
        assert pickle.loads(pickle.dumps(policy)) == policy
    assert b"seam_policy" not in pickle.dumps(default) and b"ring_split" not in pickle.dumps(T)
    assert "seam_policy" not in definition_value(RS)["fields"]
    broken = IntersectionBatchPolicy.__new__(IntersectionBatchPolicy)
    with pytest.raises(GeometryError, match="seam_policy"):
        broken.__setstate__([*RS.__getstate__()[:-1], "nonsense"])


def test_resolution_environment_precedence_budgets_and_context(monkeypatch):
    from anygeometry.arrangement_geometry import _junction_bounds_policy
    assert bi.resolve_seam_policy(IntersectionBatchPolicy()) == "baseline"
    monkeypatch.setenv(bi.SEAM_POLICY_ENVIRONMENT, "ring_split")
    assert bi.resolve_seam_policy(IntersectionBatchPolicy()) == "ring_split"
    assert bi.resolve_seam_policy(IntersectionBatchPolicy(seam_policy="baseline")) == "baseline"   # field wins
    for bad in ("RING_SPLIT", "", "ring_split "):
        monkeypatch.setenv(bi.SEAM_POLICY_ENVIRONMENT, bad)
        with pytest.raises(GeometryError, match="ANYGEOMETRY_SEAM_POLICY"):
            bi.resolve_seam_policy(IntersectionBatchPolicy())
    monkeypatch.delenv(bi.SEAM_POLICY_ENVIRONMENT)
    for forced in (IntersectionBatchPolicy(seam_policy="ring_split", max_predicates=10**9),
                   IntersectionBatchPolicy(seam_policy="ring_split", max_candidate_pairs=10**6)):
        assert bi.resolve_seam_policy(forced) == "baseline"
    with _junction_bounds_policy(True):
        assert bi.resolve_seam_policy(RS) == "baseline"
    assert bi.resolve_seam_policy(RS) == "ring_split"
    # independent of planning_mode: all four combinations resolve as given
    for mode in (None, "exact", "topology_certified"):
        for seam in ("baseline", "ring_split"):
            policy = IntersectionBatchPolicy(planning_mode=mode, seam_policy=seam)
            assert bi.resolve_seam_policy(policy) == seam


def test_arrange_material_validates_the_policy_and_reads_no_environment(monkeypatch):
    support = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    corners = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))
    loop = tuple(ArrangementPath(LinePath(a, b), source_edge=k + 1)
                 for k, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])))
    domain = MaterialDomain(1, support, (loop,))
    trace = (ArrangementPath(LinePath((.5, -1., 0.), (.5, .5, 0.)), owners=(2,)),)     # one free end at (.5, .5)
    for bad in ("RING_SPLIT", "ring", 1):
        with pytest.raises(GeometryError, match="seam_policy"):
            arrange_material(domain, trace, tolerance=1e-9, seam_policy=bad)
    reference = arrange_material(domain, trace, tolerance=1e-9)
    monkeypatch.setenv(bi.SEAM_POLICY_ENVIRONMENT, "ring_split")
    assert arrange_material(domain, trace, tolerance=1e-9) == reference                  # not read here
    assert arrange_material(domain, trace, tolerance=1e-9, seam_policy="baseline") == reference


# ----------------------------------------------------------------------------------- plan states

def test_default_and_explicit_baseline_plans_are_byte_identical_state_e(plans):
    model, exact, _rs, _trs = plans
    _model, faces = _l_joint()
    explicit = plan_intersections(model, faces, policy=IntersectionBatchPolicy(seam_policy="baseline"))
    budget = plan_intersections(model, faces, policy=IntersectionBatchPolicy(seam_policy="ring_split",
                                                                             max_predicates=10**9))
    assert type(exact) is type(explicit) is type(budget) is IntersectionPlan
    assert explicit.policy.seam_policy is None and budget.policy.seam_policy is None
    assert pickle.dumps(explicit) == pickle.dumps(exact)
    for marker in (b"seam_policy", b"ring_split", b"route_checksum", b"TopologyCertified"):
        assert marker not in pickle.dumps(exact)


def test_ring_split_exact_mode_gives_a_route_plan_bound_to_the_seam_axis(plans):
    _model, exact, rs, trs = plans
    assert type(rs) is TopologyCertifiedPlan
    assert (rs.planning_mode, rs.seam_policy, rs.seam_policy_version) == (
        "exact", "ring_split", bi.SEAM_POLICY_VERSIONS["ring_split"])
    assert (rs.certificate_version, rs.r_max, rs.route_tolerance, rs.r_pairs) == (None, None, None, ())
    assert rs.policy.seam_policy is None and rs.policy.planning_mode is None
    # no free trace end here: identical geometry, identical content checksum, different route
    assert pickle.dumps(rs.arrangements) == pickle.dumps(exact.arrangements)
    assert rs.content_checksum == exact.content_checksum
    assert rs.route_checksum == bi.definition_checksum((rs.content_checksum, "exact", None, None, "ring_split",
                                                        bi.SEAM_POLICY_VERSIONS["ring_split"]))
    assert pickle.loads(pickle.dumps(rs)).route_checksum == rs.route_checksum
    assert (trs.planning_mode, trs.seam_policy, trs.certificate_version) == (
        "topology_certified", "ring_split", sp.CERTIFICATE_VERSION)
    assert trs.route_checksum == bi.definition_checksum((trs.content_checksum, "topology_certified",
                                                         sp.CERTIFICATE_VERSION, trs.r_max, "ring_split",
                                                         bi.SEAM_POLICY_VERSIONS["ring_split"]))
    assert len({exact.content_checksum, rs.content_checksum, trs.content_checksum}) == 1
    assert len({rs.route_checksum, trs.route_checksum}) == 2


def test_environment_selects_ring_split_at_planning(monkeypatch):
    model, faces = _l_joint()
    monkeypatch.setenv(bi.SEAM_POLICY_ENVIRONMENT, "ring_split")
    plan = plan_intersections(model, faces, policy="connect")
    assert type(plan) is TopologyCertifiedPlan and plan.seam_policy == "ring_split"
    assert type(plan_intersections(model, faces, policy=IntersectionBatchPolicy(seam_policy="baseline"))) \
        is IntersectionPlan


def test_planner_passes_the_resolved_policy_to_the_main_arrangement_loop(monkeypatch):
    seen = []
    original = bi.arrange_material

    def spy(*args, **kwargs):
        seen.append(kwargs.get("seam_policy"))
        return original(*args, **kwargs)
    monkeypatch.setattr(bi, "arrange_material", spy)
    model, faces = _l_joint()
    for policy, expected in (("connect", "baseline"), (RS, "ring_split"), (TRS, "ring_split"),
                             (IntersectionBatchPolicy(seam_policy="ring_split", max_predicates=10**9), "baseline")):
        seen.clear()
        plan_intersections(model, faces, policy=policy)
        assert seen and set(seen) == {expected}, (policy, seen)


# ----------------------------------------------------------------------------------- matrix rows (RS-0)

def test_row_e_plan_under_ring_split_is_rejected(plans):
    model, exact, _rs, _trs = plans
    before = to_dict(model)
    for policy in (RS, TRS):
        with pytest.raises(GeometryError, match="baseline-seam intersection plan|topology_certified mode"):
            apply_intersections(model, exact, policy=policy)
    assert to_dict(model) == before


def test_row_ring_split_plan_under_baseline_is_rejected(plans):
    model, _exact, rs, trs = plans
    before = to_dict(model)
    for plan, policy in ((rs, "connect"), (rs, IntersectionBatchPolicy(seam_policy="baseline")), (trs, T),
                         (rs, TRS), (trs, RS),
                         (rs, IntersectionBatchPolicy(seam_policy="ring_split", max_predicates=10**9))):
        with pytest.raises(GeometryError, match="seam intersection plan cannot be applied|never applied as exact"
                                                "|topology_certified mode"):
            apply_intersections(model, plan, policy=policy)
    assert to_dict(model) == before


def test_row_t_plan_with_baseline_seams_under_ring_split_is_rejected():
    model, faces = _l_joint()
    certified = plan_intersections(model, faces, policy=T)
    assert certified.seam_policy == "baseline"
    with pytest.raises(GeometryError, match="'baseline'-seam intersection plan cannot be applied"):
        apply_intersections(model, certified, policy=TRS)


@pytest.mark.parametrize("changes", [
    {"seam_policy": None}, {"seam_policy": "ring"}, {"seam_policy": "RING_SPLIT"}, {"seam_policy": 1},
    {"seam_policy_version": None}, {"seam_policy_version": "seam-ring-split-0"},
    {"seam_policy_version": bi.SEAM_POLICY_VERSIONS["baseline"]},
    {"seam_policy": "baseline"},                         # exact mode + baseline seams = state E: no route fields
])
def test_row_missing_unknown_or_unsupported_seam_fields_are_rejected(plans, changes):
    model, _exact, rs, _trs = plans
    tampered = _resealed(rs, **changes)
    with pytest.raises(GeometryError):
        pickle.loads(pickle.dumps(tampered))
    with pytest.raises(GeometryError):
        TopologyCertifiedPlan(**{item.name: getattr(tampered, item.name) for item in fields(tampered)})
    for policy in (RS, "connect"):
        with pytest.raises(GeometryError):
            apply_intersections(model, tampered, policy=policy)


def test_row_15_has_its_own_unsound_double(plans, monkeypatch):
    model, _exact, rs, _trs = plans
    tampered = _resealed(rs, seam_policy_version="seam-ring-split-0")
    with pytest.raises(GeometryError, match="seam_policy_version"):
        bi._route_row_15(tampered)
    monkeypatch.setattr(bi, "_route_row_15", lambda plan: None)
    loaded = pickle.loads(pickle.dumps(tampered))
    assert type(apply_intersections(model.clone(preserve_identity=True), loaded, policy=RS)) \
        is bi.TopologyCertifiedApplication


def test_row_seam_fields_without_route_checksum_are_rejected(plans):
    model, _exact, rs, trs = plans
    for plan in (rs, trs):
        for value in (None, "", "0"*64):
            tampered = _forge(plan, route_checksum=value)
            with pytest.raises(GeometryError, match="route_checksum|route binding"):
                pickle.loads(pickle.dumps(tampered))
            with pytest.raises(GeometryError):
                apply_intersections(model, tampered, policy=RS if plan is rs else TRS)


def test_row_exact_route_carries_no_certificate_fields(plans):
    model, _exact, rs, trs = plans
    for changes in ({"certificate_version": sp.CERTIFICATE_VERSION}, {"r_max": trs.r_max},
                    {"route_tolerance": 1e-9}, {"r_pairs": ((1, 2, 0.),)}):
        tampered = _resealed(rs, **changes)
        with pytest.raises(GeometryError):
            pickle.loads(pickle.dumps(tampered))
        with pytest.raises(GeometryError):
            apply_intersections(model, tampered, policy=RS)


def test_row_ring_split_with_topology_certified_is_accepted_and_persisted(plans):
    model, _exact, _rs, trs = plans
    applied = apply_intersections(model, trs, policy=TRS)
    assert type(applied) is bi.TopologyCertifiedApplication
    assert (applied.seam_policy, applied.seam_policy_version) == ("ring_split", bi.SEAM_POLICY_VERSIONS["ring_split"])
    change_set = applied.change_set
    assert type(change_set) is bi.TopologyCertifiedChangeSet
    assert (change_set.seam_policy, change_set.planning_mode) == ("ring_split", "topology_certified")
    record = bi.intersection_planning_record(model)
    assert len(record) == 1 and record[0]["seam_policy"] == "ring_split"
    loaded = from_dict(to_dict(model))
    assert bi.intersection_planning_record(loaded) == record


def test_row_exact_ring_split_plan_applies_under_its_policy_and_records_the_route(plans):
    model, exact, rs, _trs = plans
    left, right = model.clone(preserve_identity=True), model.clone(preserve_identity=True)
    applied = apply_intersections(left, rs, policy=RS)
    assert type(applied) is bi.TopologyCertifiedApplication and applied.planning_mode == "exact"
    assert (applied.change_set.certificate_version, applied.change_set.r_max) == (None, None)
    entry = bi.intersection_planning_record(left)[0]
    assert (entry["planning_mode"], entry["seam_policy"], entry["certificate_version"]) == ("exact", "ring_split", None)
    reference = apply_intersections(right, exact, policy="connect")
    assert type(reference) is bi.IntersectionApplication
    strip = lambda document: {k: v for k, v in document.items() if k not in ("extensions", "checksum")}
    assert strip(to_dict(left)) == strip(to_dict(right))          # same geometry on this model


def test_row_13_stripped_ring_split_plan_is_accepted_as_e(plans):
    model, _exact, rs, _trs = plans
    stripped = IntersectionPlan(**{item.name: getattr(rs, item.name) for item in fields(IntersectionPlan)})
    assert type(apply_intersections(model, stripped, policy="connect")) is bi.IntersectionApplication


def test_apply_resolves_the_environment_seam_policy(plans, monkeypatch):
    model, exact, rs, _trs = plans
    monkeypatch.setenv(bi.SEAM_POLICY_ENVIRONMENT, "ring_split")
    with pytest.raises(GeometryError, match="baseline-seam intersection plan"):
        apply_intersections(model.clone(preserve_identity=True), exact, policy="connect")
    assert type(apply_intersections(model.clone(preserve_identity=True), rs, policy="connect")) \
        is bi.TopologyCertifiedApplication
    monkeypatch.setenv(bi.SEAM_POLICY_ENVIRONMENT, "bogus")
    with pytest.raises(GeometryError, match="ANYGEOMETRY_SEAM_POLICY"):
        apply_intersections(model.clone(preserve_identity=True), exact, policy="connect")


def test_mode_agnostic_routes_store_no_seam_policy():
    policy = IntersectionBatchPolicy(planning_mode="topology_certified", seam_policy="ring_split")
    stored = bi._mode_agnostic(policy)
    assert stored.planning_mode is None and stored.seam_policy is None
    plain = IntersectionBatchPolicy()
    assert bi._mode_agnostic(plain) is plain
    only_seam = IntersectionBatchPolicy(seam_policy="ring_split")
    assert bi._mode_agnostic(only_seam).seam_policy is None


def test_route_carriers_with_seam_fields_round_trip(plans):
    model, _exact, rs, trs = plans
    for plan, policy in ((rs, RS), (trs, TRS)):
        applied = apply_intersections(model.clone(preserve_identity=True), plan, policy=policy)
        for carrier in (plan, applied.change_set, applied):
            values = {item.name: getattr(carrier, item.name) for item in fields(carrier)}
            rebuilt = type(carrier)(**values)
            assert pickle.dumps(rebuilt) == pickle.dumps(carrier)
            once = pickle.dumps(pickle.loads(pickle.dumps(carrier)))
            assert pickle.dumps(pickle.loads(once)) == once
