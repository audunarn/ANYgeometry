"""T2-1b: opt-in topology_certified planning route (ADR-T2 v2 sections 1, 4.5, 5, 7; addendum A9).

Every safety rule is checked together with an unsound test double that must fail the same check.
"""
import math
import pickle
from dataclasses import fields, replace

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, from_dict, to_dict
from anygeometry import analytic_supports
from anygeometry import batch_intersections as bi
from anygeometry import separating_plane_certificate as sp
from anygeometry.exact_curves import EllipticArc
from anygeometry.serialization import _checksum
from anygeometry.transactions import ChangeSet
from anygeometry.batch_intersections import (IntersectionBatchPolicy, IntersectionPlan, TopologyCertifiedPlan,
                                             apply_intersections, plan_intersections)
from anygeometry.definition_binding import definition_value
from anygeometry.entities import OrientedEdge
from anygeometry.generators import cylinder
from anygeometry.material_arrangement import MaterialDomain
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.surfaces import Cylinder

T = IntersectionBatchPolicy(planning_mode="topology_certified")


@pytest.fixture(autouse=True)
def _no_mode_environment(monkeypatch):
    monkeypatch.delenv(bi.PLANNING_MODE_ENVIRONMENT, raising=False)


# ----------------------------------------------------------------------------------- fixtures

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
    return model, (first, second), edges[(0, 1)]


def _vertex(model, point):
    return next(v.id for v in model.vertices.values() if np.linalg.norm(np.asarray(v.position)-point) < 1e-12)


def _strips_and_fin():
    """Eight 45-degree strips of one cylinder (coincident supports) and a radial fin plate that
    shares the generator at theta = 0 (a cp pair with two strips)."""
    model = cylinder(1., 1., circumferential_segments=8)
    strips = tuple(model.faces)
    a, b = _vertex(model, (1, 0, 0)), _vertex(model, (1, 0, 1))
    generator = next(e.id for e in model.edges.values() if {e.start, e.end} == {a, b})
    c, d = model.add_point(2, 0, 0), model.add_point(2, 0, 1)
    fin = model.add_face([generator, model.add_line(b, d), model.add_line(d, c), model.add_line(c, a)])
    return model, strips, fin


def _cyl_point(model, theta, z, radius=1.):
    return model.add_point(radius*math.cos(theta), radius*math.sin(theta), z)


def _arc(model, start, end, theta0, theta1, z, radius=1.):
    via = _cyl_point(model, .5*(theta0+theta1), z, radius)
    return model.add_arc(start, via, end)


def _lobe_strips(radius_b=1.):
    """Strip A (theta in [0, pi/4], z in [0, 2]) and a nonconvex face B on the same cylinder that
    shares A's generator theta = pi/4, z in [0, 1], with opposite traversal and an overlap lobe
    (theta in [pi/8, pi/4], z in [1.3, 1.6]) away from the shared edge."""
    model = GeometryModel()
    q = math.pi/4
    a00, a01 = _cyl_point(model, 0, 0), _cyl_point(model, 0, 2)
    s0, s1, s2 = _cyl_point(model, q, 0), _cyl_point(model, q, 1), _cyl_point(model, q, 2)
    shared = model.add_line(s0, s1)
    loop_a = (OrientedEdge(_arc(model, a00, s0, 0, q, 0), True), OrientedEdge(shared, True),
              OrientedEdge(model.add_line(s1, s2), True), OrientedEdge(_arc(model, a01, s2, 0, q, 2), False),
              OrientedEdge(model.add_line(a00, a01), False))
    face_a = model.add_face_from_loop(loop_a, surface=Cylinder(radius=1., origin=(0, 0, 0), axis=(0, 0, 1),
                                      radial_direction=(1, 0, 0), height=2., start_angle=0., sweep_angle=q))
    h, k = math.pi/2, 3*math.pi/8
    r = radius_b
    b1, b2 = _cyl_point(model, h, 0, r), _cyl_point(model, h, 1.6, r)
    b3, b4 = _cyl_point(model, math.pi/8, 1.6, r), _cyl_point(model, math.pi/8, 1.3, r)
    b5, b6 = _cyl_point(model, k, 1.3, r), _cyl_point(model, k, 1., r)
    loop_b = (OrientedEdge(_arc(model, s0, b1, q, h, 0, r), True), OrientedEdge(model.add_line(b1, b2), True),
              OrientedEdge(_arc(model, b3, b2, math.pi/8, h, 1.6, r), False),
              OrientedEdge(model.add_line(b3, b4), True), OrientedEdge(_arc(model, b4, b5, math.pi/8, k, 1.3, r), True),
              OrientedEdge(model.add_line(b5, b6), True), OrientedEdge(_arc(model, s1, b6, q, k, 1., r), False),
              OrientedEdge(shared, False))
    face_b = model.add_face_from_loop(loop_b, surface=Cylinder(radius=r, origin=(0, 0, 0), axis=(0, 0, 1),
                                      radial_direction=(1, 0, 0), height=1.6, start_angle=math.pi/8,
                                      sweep_angle=3*math.pi/8))
    return model, (face_a, face_b)


class _StubCertifier:
    """Certifies every pair with a given coincidence flag (the verdict is the input under test)."""

    def __init__(self, coincident):
        self.coincident = coincident

    def certify(self, first, second, tolerance, *, check=None):
        return sp.Certified(0., {"coincident_support": self.coincident})


def _with_certifier(monkeypatch, certifier):
    original = bi._TopologyRoute.__init__

    def init(self, model, certificate_policy=None):
        original(self, model, certificate_policy)
        self.certifier = certifier
    monkeypatch.setattr(bi._TopologyRoute, "__init__", init)


def _counters(call):
    with collect_runtime_diagnostics() as counters:
        result = call()
    return result, {k: v for k, v in counters.items() if k.startswith(("t2b", "planning_mode")) and v
                    and not k.endswith("_ns")}


def _refusal(call):
    try:
        call()
    except GeometryError as exc:
        return str(exc)
    return None


# ----------------------------------------------------------------------------------- switch

def test_switch_exists_selects_the_route_and_default_stays_exact():
    model, faces, shared = _l_joint()
    exact = plan_intersections(model, faces, policy="connect")
    certified, counters = _counters(lambda: plan_intersections(model, faces, policy=T))
    assert type(exact) is IntersectionPlan
    assert type(certified) is TopologyCertifiedPlan
    assert certified.planning_mode == "topology_certified"
    assert counters["planning_mode:topology_certified"] == 1 and counters["t2b_certified"] == 1
    assert [item[:2] for item in certified.r_pairs] == [tuple(faces)]
    assert 0. <= certified.r_pairs[0][2] <= certified.r_max
    assert certified.content_checksum == exact.content_checksum        # geometry identity unchanged
    assert certified.route_checksum and certified.route_checksum != certified.content_checksum
    domains = [MaterialDomain.from_model(model, f) for f in faces]
    route = bi._TopologyRoute(model)
    traces = bi._pair_paths(domains[0], domains[1], model, 1e-9, lambda: None, point_contacts=[], route=route)
    from anygeometry.arrangement_geometry import freeze_edge
    stored = freeze_edge(model, shared)
    assert len(traces) == 1 and traces[0] is not None
    assert np.array_equal(traces[0].evaluate(np.linspace(0, 1, 5)), stored.evaluate(np.linspace(0, 1, 5)))


def test_environment_variable_and_precedence(monkeypatch):
    model, faces, _ = _l_joint()
    monkeypatch.setenv(bi.PLANNING_MODE_ENVIRONMENT, "topology_certified")
    assert type(plan_intersections(model, faces, policy="connect")) is TopologyCertifiedPlan
    explicit = IntersectionBatchPolicy(planning_mode="exact")
    assert type(plan_intersections(model, faces, policy=explicit)) is IntersectionPlan   # API argument wins
    for bad in ("TOPOLOGY", "", "exact "):
        monkeypatch.setenv(bi.PLANNING_MODE_ENVIRONMENT, bad)
        with pytest.raises(GeometryError, match="ANYGEOMETRY_PLANNING_MODE"):
            plan_intersections(model, faces, policy="connect")
    monkeypatch.delenv(bi.PLANNING_MODE_ENVIRONMENT)
    for bad in ("Exact", 1, True, ""):
        with pytest.raises(GeometryError, match="planning_mode"):
            IntersectionBatchPolicy(planning_mode=bad)


def test_budgets_and_nondefault_context_force_the_exact_route():
    from anygeometry.arrangement_geometry import _junction_bounds_policy
    model, faces, _ = _l_joint()
    forced = (IntersectionBatchPolicy(planning_mode="topology_certified", max_predicates=10**9),
              IntersectionBatchPolicy(planning_mode="topology_certified", max_candidate_pairs=10**6))
    for policy in forced:
        assert bi.resolve_planning_mode(policy) == "exact"
        assert type(plan_intersections(model, faces, policy=policy)) is IntersectionPlan
    with _junction_bounds_policy(True):
        assert bi.resolve_planning_mode(T) == "exact"
    assert bi.resolve_planning_mode(T) == "topology_certified"

    def unsound_resolver(policy):                       # double: ignores budgets
        return policy.planning_mode or "exact"
    assert any(unsound_resolver(policy) != "exact" for policy in forced)


def test_default_planning_never_imports_the_certificate():
    import subprocess, sys, os, pathlib
    src = pathlib.Path(bi.__file__).resolve().parents[1]
    code = ("import sys; from anygeometry import GeometryModel; from anygeometry.batch_intersections import "
            "plan_intersections as p; m=GeometryModel(); a=m.add_plate(m.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))));"
            "b=m.add_plate(m.add_points(((0,.5,-1),(1,.5,-1),(1,.5,1),(0,.5,1)))); p(m,(a,b),policy='connect');"
            "print('anygeometry.separating_plane_certificate' in sys.modules)")
    env = {k: v for k, v in os.environ.items() if k != bi.PLANNING_MODE_ENVIRONMENT}
    env["PYTHONPATH"] = str(src)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr[-2000:]


# ----------------------------------------------------------------------------------- state E identity

def test_exact_plans_carry_no_route_field_under_every_serialization():
    model, faces, _ = _l_joint()
    default = plan_intersections(model, faces, policy="connect")
    explicit = plan_intersections(model, faces, policy=IntersectionBatchPolicy(planning_mode="exact"))
    assert type(default) is type(explicit) is IntersectionPlan
    assert explicit.policy.planning_mode is None                        # normalised to the sentinel
    assert pickle.dumps(default) == pickle.dumps(explicit)
    for blob in (pickle.dumps(default), pickle.dumps(default.policy)):
        for marker in (b"planning_mode", b"route_checksum", b"TopologyCertified", b"topology_certified"):
            assert marker not in blob
    assert len(IntersectionBatchPolicy().__getstate__()) == len(fields(IntersectionBatchPolicy)) - 1
    assert "planning_mode" not in definition_value(IntersectionBatchPolicy())["fields"]
    assert pickle.loads(pickle.dumps(default)).content_checksum == default.content_checksum
    restored = pickle.loads(pickle.dumps(T))
    assert restored.planning_mode == "topology_certified" and restored == T
    legacy = IntersectionBatchPolicy.__new__(IntersectionBatchPolicy)
    legacy.__setstate__(IntersectionBatchPolicy().__getstate__())        # pre-T2-1b state (7 items)
    assert legacy == IntersectionBatchPolicy()

    class AlwaysField(IntersectionBatchPolicy):          # double: always pickles the new field
        __slots__ = ()

        def __getstate__(self):
            return [getattr(self, item.name) for item in fields(self)]
    assert len(AlwaysField().__getstate__()) != len(IntersectionBatchPolicy().__getstate__())


# ----------------------------------------------------------------------------------- matrix (section 7)

@pytest.fixture
def plans():
    model, faces, _ = _l_joint()
    exact = plan_intersections(model, faces, policy="connect")
    certified = plan_intersections(model, faces, policy=T)
    return model, exact, certified


def _forge(carrier, **changes):
    """``carrier`` with ``changes`` set WITHOUT construction: a corrupted object as unpickling, a
    caller or an attacker could present it (fix round 5: construction itself validates every field,
    so tampered plans and change sets can no longer be made with ``dataclasses.replace``)."""
    made = object.__new__(type(carrier))
    for item in fields(carrier):
        object.__setattr__(made, item.name, changes[item.name] if item.name in changes else getattr(carrier, item.name))
    return made


def _resealed(plan, **changes):
    changed = _forge(plan, **changes)
    try:
        return _forge(changed, route_checksum=bi._route_checksum(changed))
    except (ValueError, TypeError):                     # non-finite values cannot be bound at all
        return changed


def _only_this_field_is_wrong(plan, certified):
    """Discrimination check (the double for every row is a validator that ignores the tampered
    field): restoring the tampered fields, and only them, makes the plan valid again."""
    changed = {item.name: getattr(certified, item.name) for item in fields(certified)
               if repr(getattr(plan, item.name)) != repr(getattr(certified, item.name))}
    assert changed, "the tampered plan must differ in a field"
    restored = replace(plan, **changed)
    return bi._route_state(restored) == "T"


def test_matrix_rows_1_and_3_accept_and_apply_identically(plans):
    model, exact, certified = plans
    left, right = model.clone(preserve_identity=True), model.clone(preserve_identity=True)
    applied_e = apply_intersections(left, exact, policy="connect")                       # row 1
    assert type(applied_e) is bi.IntersectionApplication
    applied_t = apply_intersections(right, certified, policy=T)                          # row 3
    assert type(applied_t) is bi.TopologyCertifiedApplication
    assert (applied_t.planning_mode, applied_t.r_max, applied_t.certificate_version) == (
        "topology_certified", certified.r_max, sp.CERTIFICATE_VERSION)
    # Equal geometry checksums: identical models, except the row-3 route record of the T application
    # (fix round 4: the record lives in the document extensions, so the document checksum differs).
    document_e, document_t = to_dict(left), to_dict(right)
    assert document_e["extensions"] == {} and set(document_t["extensions"]) == {bi.INTERSECTION_PLANNING_EXTENSION}
    strip = lambda document: {k: v for k, v in document.items() if k not in ("extensions", "checksum")}
    assert strip(document_e) == strip(document_t)


@pytest.mark.parametrize("row", [2, 4])
def test_matrix_mode_mismatch_rows(plans, row):
    model, exact, certified = plans
    plan, policy = (exact, T) if row == 2 else (certified, IntersectionBatchPolicy())
    before = to_dict(model)
    with pytest.raises(GeometryError, match="topology_certified mode|never applied as exact"):
        apply_intersections(model, plan, policy=policy)
    assert to_dict(model) == before


def _tampered(certified, row):
    if row == 5:
        return [_forge(certified, route_checksum=value) for value in ("", None)]
    if row == 6:
        return [_forge(certified, route_checksum="0"*64), _forge(certified, r_max=math.nextafter(certified.r_max, 0.))]
    if row == 7:
        return [_resealed(certified, certificate_version=value) for value in (None, "sp-exact-t2-1a-0", 1)]
    if row == 8:
        values = [None, math.nan, math.inf, 5e-324, 0., -certified.r_max, 1e-12, "1e-12", 2*certified.r_max]
        return [_resealed(certified, r_max=value) for value in values
                if not (value == certified.r_max and type(value) is float)]
    if row == 9:
        rp = certified.r_pairs[0]
        values = [((rp[0], rp[1], None),), ((rp[0], rp[1], math.nan),), ((rp[0], rp[1], math.inf),),
                  ((rp[0], rp[1], math.nextafter(certified.r_max, 1.)),), ((rp[0], rp[1]),), [rp]]
        return [_resealed(certified, r_pairs=value) for value in values]
    if row == 10:
        return [_resealed(certified, planning_mode=value) for value in ("topology", "", 3)]
    if row == 11:
        return [_resealed(certified, planning_mode=None), _resealed(certified, planning_mode="exact")]
    if row == 12:                       # fix round 5: the content checksum's form and binding (R6)
        return [_resealed(certified, content_checksum=value) for value in ("0"*64, None, "", 123, "0"*63)]
    if row == 14:                       # fix round 5: every other field has its exact type and form
        policy = _PolicySubclass(**{item.name: getattr(certified.policy, item.name)
                                    for item in fields(certified.policy)})
        return [_forge(certified, policy=policy), _forge(certified, revision=np.int64(certified.revision)),
                _forge(certified, revision=True), _forge(certified, source_checksum=certified.source_checksum.upper())]
    raise AssertionError(row)


_ROWS = [5, 6, 7, 8, 9, 10, 11, 12, 14]


@pytest.mark.parametrize("row", _ROWS)
def test_matrix_malformed_route_rows_are_rejected_at_apply_and_load(plans, row, monkeypatch):
    model, _exact, certified = plans
    before = to_dict(model)
    for plan in _tampered(certified, row):
        for policy in (T, IntersectionBatchPolicy()):
            with pytest.raises(GeometryError):
                apply_intersections(model, plan, policy=policy)
        with pytest.raises(GeometryError):
            pickle.loads(pickle.dumps(plan))                  # plan load
        assert _only_this_field_is_wrong(plan, certified)
    assert to_dict(model) == before


_ROW_MESSAGES = {5: "no route_checksum", 6: "route binding changed", 7: "unsupported certificate_version",
                 8: "r_max is not", 9: "r_pair", 10: "is not supported",
                 11: "without planning_mode|carries no route fields",
                 12: "content binding changed|content_checksum is not", 14: "field .* is malformed"}


def _disable_row(monkeypatch, row):
    """The unsound double of ONE section-7 check: that row's check (for row 8 both its load-side and
    its model-side half) is removed; every other check stays active."""
    monkeypatch.setattr(bi, f"_route_row_{row}", lambda plan: None)
    if row == 8:
        monkeypatch.setattr(bi, "_route_row_8_model", lambda plan, model: None)


def _load_and_apply(model, plan):
    """``None`` if the plan loads and applies in topology_certified mode, else the refusal."""
    try:
        loaded = pickle.loads(pickle.dumps(plan))
        apply_intersections(model.clone(preserve_identity=True), loaded, policy=T)
    except Exception as exc:                             # noqa: BLE001 - the outcome is compared
        return f"{type(exc).__name__}: {exc}"
    return None


@pytest.mark.parametrize("row", _ROWS)
def test_matrix_row_checks_each_have_their_own_unsound_double(plans, row, monkeypatch):
    """Review R5 (fix round 4): one double per check, not one global double. For every tampered plan
    of a row, that row's check rejects it on its own; with ONLY that check removed, at least one
    tampered plan of the row loads (pickle) and applies, and every other one is refused by another
    row's check (different reason). Row 8 is not exempt: its double removes both halves of the row
    (the contract value at load and the model tolerance at apply), and the self-consistent r_max
    values that are not the contract value then load and apply.

    Row 5 is the one row whose double cannot reach acceptance: a missing or empty route_checksum can
    never equal the recomputed definition_checksum (a non-empty hex digest), so the row-6 binding
    check rejects every row-5 plan as well. Its double is discriminated by the reason: with row 5
    removed, every row-5 plan is refused with the row-6 reason instead of the row-5 reason."""
    model, _exact, certified = plans
    tampered = _tampered(certified, row)
    for plan in tampered:
        with pytest.raises(GeometryError, match=_ROW_MESSAGES[row]):
            getattr(bi, f"_route_row_{row}")(plan)
    real = [_load_and_apply(model, plan) for plan in tampered]
    assert all(outcome is not None for outcome in real)
    _disable_row(monkeypatch, row)
    double = [_load_and_apply(model, plan) for plan in tampered]
    accepted = sum(outcome is None for outcome in double)
    import re
    for outcome in double:
        assert outcome is None or not re.search(_ROW_MESSAGES[row], outcome), outcome
    if row == 5:
        assert accepted == 0 and all(_ROW_MESSAGES[6] in outcome for outcome in double)
    else:
        assert accepted >= 1, double


def test_matrix_row_8_model_side_check_has_its_own_double(plans, monkeypatch):
    """Row 8 at apply: a self-consistent plan for ANOTHER tolerance passes the load-side half and is
    refused only by the model-side half; removing that half alone applies it."""
    model, _exact, certified = plans
    other = _resealed(certified, route_tolerance=2e-9, r_max=sp.contract_r_max(2e-9), r_pairs=())
    assert bi._route_row_8(other) is None
    assert "contract value" in _load_and_apply(model, other)
    monkeypatch.setattr(bi, "_route_row_8_model", lambda plan, model: None)
    assert _load_and_apply(model, other) is None


def test_matrix_row_8_rejects_a_wrong_contract_value_at_load_and_apply(plans):
    model, _exact, certified = plans
    # Positive normal values that are not tol/1000 rounded down; the float product 1e-3*tol is not
    # the contract value when it differs from the rounded-down rational.
    for value in (certified.r_max/2, math.nextafter(certified.r_max, math.inf), 1e-3*1e-9*(1+2**-50)):
        plan = _resealed(certified, r_max=value, r_pairs=())
        with pytest.raises(GeometryError, match="contract value"):
            pickle.loads(pickle.dumps(plan))
        with pytest.raises(GeometryError, match="contract value"):
            apply_intersections(model, plan, policy=T)
    # A self-consistent plan for ANOTHER tolerance loads, but is refused against this model.
    other = _resealed(certified, route_tolerance=2e-9, r_max=sp.contract_r_max(2e-9), r_pairs=())
    assert pickle.loads(pickle.dumps(other)).route_tolerance == 2e-9
    with pytest.raises(GeometryError, match="contract value"):
        apply_intersections(model, other, policy=T)


def test_plan_load_checks_the_content_checksum(plans):
    _model, _exact, certified = plans
    tampered = _forge(certified, face_contacts=((),))
    with pytest.raises(GeometryError, match="content binding"):
        pickle.loads(pickle.dumps(tampered))
    assert pickle.loads(pickle.dumps(certified)).route_checksum == certified.route_checksum


def test_matrix_row_12_invalid_content_checksum(plans):
    model, exact, certified = plans
    for plan, policy in ((replace(exact, face_contacts=((),)), "connect"),
                         (_forge(certified, content_checksum="0"*64), T)):
        with pytest.raises(GeometryError, match="content binding"):
            apply_intersections(model, plan, policy=policy)


def test_matrix_row_13_stripped_plan_is_an_exact_plan(plans):
    model, exact, certified = plans
    stripped = IntersectionPlan(**{item.name: getattr(certified, item.name) for item in fields(IntersectionPlan)})
    applied = apply_intersections(model, stripped, policy="connect")
    assert type(applied) is bi.IntersectionApplication


def test_route_checksum_binds_mode_version_and_r_max(plans):
    _model, _exact, certified = plans
    expected = bi.definition_checksum((certified.content_checksum, "topology_certified",
                                       sp.CERTIFICATE_VERSION, certified.r_max))
    assert certified.route_checksum == expected
    assert certified.r_max == sp.contract_r_max(1e-9)


# ----------------------------------------------------------------------------------- route behaviour

def test_coincident_pairs_return_the_exact_output_and_plans_stay_identical():
    model, strips, fin = _strips_and_fin()
    faces = (*strips, fin)
    exact = plan_intersections(model, faces, policy="connect")
    certified, counters = _counters(lambda: plan_intersections(model, faces, policy=T))
    assert counters["t2b_coincident_exact_output"] == 8 and counters["t2b_certified"] == 2
    assert certified.content_checksum == exact.content_checksum
    pairs = [item[:2] for item in certified.r_pairs]
    assert pairs == sorted(pairs)                                      # candidate order


def test_certificate_budget_exhaustion_falls_back_to_the_identical_exact_plan():
    model, strips, fin = _strips_and_fin()
    faces = (*strips, fin)
    exact = plan_intersections(model, faces, policy="connect")
    tiny = sp.SPCertificatePolicy(max_interval_operations=1)
    certified, counters = _counters(lambda: bi._plan_intersections(
        model, faces, policy="connect", _planning_mode="topology_certified", _certificate_policy=tiny))
    assert certified.r_pairs == () and certified.content_checksum == exact.content_checksum
    assert counters["t2b_pairs"] == 10 and "t2b_certified" not in counters
    assert any(key.startswith("t2b_fallback:") for key in counters)


def test_records_are_keyed_by_domain_object_not_face_id():
    model, faces, _ = _l_joint()
    route = bi._TopologyRoute(model)
    domain = MaterialDomain.from_model(model, faces[0])
    twin = MaterialDomain.from_model(model, faces[0])
    assert route._record(domain) is route._record(domain)
    assert route._record(twin) is not route._record(domain)
    with pytest.raises(sp.SPRefusal, match="domain_face"):
        sp.face_record_from_model(model, faces[1], domain=domain)


@pytest.mark.parametrize("entry", ["plane_cylinder_support", "cylinder_cylinder_support", "qualified_plane_plane",
                                   "plane_ruled.qualify_plane_cylinder_result"])
def test_support_stage_refusal_is_raised_in_shadow_on_certified_pairs(monkeypatch, entry):
    """D3 / section 5: support-stage refusals of the pp, cp and cc families stay reachable for
    certified pairs (the coincident cc strips reach cylinder_cylinder_support first)."""
    if entry == "qualified_plane_plane":
        model, faces, _ = _l_joint()
    else:
        model, strips, fin = _strips_and_fin()
        faces = (*strips, fin)

    def refuse(*args, **kwargs):
        raise GeometryError(f"injected {entry} refusal")
    if "." in entry:
        import importlib
        module, name = entry.split(".")
        monkeypatch.setattr(importlib.import_module("anygeometry." + module), name, refuse)
    else:
        monkeypatch.setattr(bi, entry, refuse)
    exact = _refusal(lambda: plan_intersections(model, faces, policy="connect"))
    assert exact and "injected" in exact
    assert _refusal(lambda: plan_intersections(model, faces, policy=T)) == exact

    class NoShadow(bi._TopologyRoute):                  # double: skips the support shadow
        def pair(self, first, second, model, tolerance, check, point_contacts, face_tolerances):
            verdict = self._certify(first, second, check)
            return None if verdict is None else self._emit(verdict, first, second, tolerance, point_contacts)
    monkeypatch.setattr(bi, "_TopologyRoute", NoShadow)
    assert _refusal(lambda: plan_intersections(model, faces, policy=T)) != exact


# ----------------------------------------------------------------------------------- A9

class _FlagTrigger(bi._TopologyRoute):
    """Double: triggers the coincident shadow from the certificate's flag (not production's test)."""

    def pair(self, first, second, model, tolerance, check, point_contacts, face_tolerances):
        verdict = self._certify(first, second, check)
        if verdict is None:
            return None
        if verdict.evidence.get("shadow_required") or verdict.evidence.get("coincident_support"):
            return super().pair(first, second, model, tolerance, check, point_contacts, face_tolerances)
        return self._emit(verdict, first, second, tolerance, point_contacts)


@pytest.mark.parametrize("radius_b", [1., math.nextafter(1., 2.)])
def test_a9a_production_coincidence_triggers_the_overlap_shadow(monkeypatch, radius_b):
    """A9 4(a): production says coincident (the one-ulp radius variant is not coincident for the
    exact test), the certificate certifies with its flag off, and the lobe overlap is refused as in
    exact mode; the call to production's overlap check is spied."""
    model, faces = _lobe_strips(radius_b)
    exact = _refusal(lambda: plan_intersections(model, faces, policy="connect"))
    assert exact and "positive-area overlap" in exact
    calls = []
    original = bi._coplanar_traces

    def spy(*args, **kwargs):
        calls.append(args[0].face_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(bi, "_coplanar_traces", spy)
    _with_certifier(monkeypatch, _StubCertifier(coincident=False))
    refused = _refusal(lambda: plan_intersections(model, faces, policy=T))
    assert refused == exact and calls                    # production's overlap check ran in shadow
    calls.clear()
    monkeypatch.setattr(bi, "_TopologyRoute", type("FlagTriggerStub", (_FlagTrigger,), {}))
    _with_certifier(monkeypatch, _StubCertifier(coincident=False))
    assert _refusal(lambda: plan_intersections(model, faces, policy=T)) is None and not calls   # double fails
    # The real certificate does not certify the overlapping pair: exact path, same refusal.
    monkeypatch.undo()
    assert _refusal(lambda: plan_intersections(model, faces, policy=T)) == exact


def test_a9b_unequal_orientation_signs_are_ineligible():
    model, strips, _fin = _strips_and_fin()
    route = bi._TopologyRoute(model)
    first, second = (MaterialDomain.from_model(model, f) for f in strips[:2])
    assert route._coincident_eligible(first, second)
    reversed_second = MaterialDomain(second.face_id, second.support, tuple(
        tuple(replace(path, curve=path.curve.subcurve(1., 0.)) for path in reversed(loop))
        for loop in second.boundaries))
    assert route._orientation(reversed_second) == -route._orientation(second)
    route._records[id(reversed_second)] = (reversed_second, route._record(second))   # same support data
    assert not route._coincident_eligible(first, reversed_second)

    class NoOrientation(bi._TopologyRoute):              # double: support data only
        def _orientation(self, domain):
            return 1
    double = NoOrientation(model)
    double._records = route._records
    assert double._coincident_eligible(first, reversed_second)


def test_a9_radial_direction_is_part_of_identical_support_data():
    model, strips, _fin = _strips_and_fin()
    route = bi._TopologyRoute(model)
    first, second = (MaterialDomain.from_model(model, f) for f in strips[:2])
    record = route._record(second)
    rotated = replace(record, support=replace(record.support, radial_direction=(0., 1., 0.)))
    route._records[id(second)] = (second, rotated)
    assert not route._coincident_eligible(first, second)
    key = lambda s: (s.origin, s.axis, s.radius)                 # double: support_key[:4] only
    assert key(route._record(first).support) == key(rotated.support)


def test_a9c_overlap_refusals_on_coincident_pairs_in_both_modes(monkeypatch):
    """A9 4(c): the nonconvex lobe overlap (above) and a corner overlap within delta give the
    exact mode's outcome in topology_certified mode, with the real and with a certifying stub."""
    model, faces = _lobe_strips()
    exact = _refusal(lambda: plan_intersections(model, faces, policy="connect"))
    assert exact and _refusal(lambda: plan_intersections(model, faces, policy=T)) == exact
    with monkeypatch.context() as patch:
        _with_certifier(patch, _StubCertifier(coincident=True))
        assert _refusal(lambda: plan_intersections(model, faces, policy=T)) == exact
    # corner overlap: B's bottom corner moved delta into A (B shares only A's top-right vertex)
    for delta in (1e-13, 1e-5):
        model, faces = _corner_strips(delta)
        outcome_e = _plan_outcome(model, faces, "connect")
        assert _plan_outcome(model, faces, T) == outcome_e
        with monkeypatch.context() as patch:
            _with_certifier(patch, _StubCertifier(coincident=True))
            assert _plan_outcome(model, faces, T) == outcome_e


def _plan_outcome(model, faces, policy):
    try:
        return ("planned", plan_intersections(model, faces, policy=policy).content_checksum)
    except GeometryError as exc:
        return ("refused", str(exc))


def _corner_strips(delta):
    """Strip A (theta in [0, pi/4], z in [0, 1]) and strip B (theta in [pi/4, pi/2]) whose left
    side runs from A's top-right vertex down to a bottom corner moved delta rad into A."""
    model = GeometryModel()
    q = math.pi/4
    a0, a1 = _cyl_point(model, 0, 0), _cyl_point(model, 0, 1)
    s0, s1 = _cyl_point(model, q, 0), _cyl_point(model, q, 1)
    loop_a = (OrientedEdge(_arc(model, a0, s0, 0, q, 0), True), OrientedEdge(model.add_line(s0, s1), True),
              OrientedEdge(_arc(model, a1, s1, 0, q, 1), False), OrientedEdge(model.add_line(a0, a1), False))
    cyl = dict(radius=1., origin=(0, 0, 0), axis=(0, 0, 1), radial_direction=(1, 0, 0), height=1.)
    face_a = model.add_face_from_loop(loop_a, surface=Cylinder(start_angle=0., sweep_angle=q, **cyl))
    c = _cyl_point(model, q - delta, 0)
    b0, b1 = _cyl_point(model, 2*q, 0), _cyl_point(model, 2*q, 1)
    loop_b = (OrientedEdge(_arc(model, c, b0, q - delta, 2*q, 0), True), OrientedEdge(model.add_line(b0, b1), True),
              OrientedEdge(_arc(model, s1, b1, q, 2*q, 1), False), OrientedEdge(model.add_line(c, s1), False))
    face_b = model.add_face_from_loop(loop_b, surface=Cylinder(start_angle=q - delta, sweep_angle=q + delta, **cyl))
    return model, (face_a, face_b)


class _Cancel(Exception):
    pass


@pytest.mark.parametrize("where", ["certificate", "shadow"])
def test_a9d_cancellation_inside_the_certificate_or_shadow_propagates_unchanged(monkeypatch, where):
    model, strips, fin = _strips_and_fin()
    faces = (*strips, fin)
    sentinel = _Cancel("stop")
    armed = []
    if where == "certificate":
        original = sp.SPCertifier.certify

        def certify(self, *args, **kwargs):
            armed.append(1)
            try:
                return original(self, *args, **kwargs)
            finally:
                armed.pop()
        monkeypatch.setattr(sp.SPCertifier, "certify", certify)
    else:
        original = bi._pair_paths_exact

        def exact(*args, _shadow=None, **kwargs):
            if _shadow is None:
                return original(*args, **kwargs)
            armed.append(1)
            try:
                return original(*args, _shadow=_shadow, **kwargs)
            finally:
                armed.pop()
        monkeypatch.setattr(bi, "_pair_paths_exact", exact)
        support = bi.plane_cylinder_support

        def polling_support(*args, cancellation_check=None, **kwargs):
            cancellation_check()                        # the support stage polls the planning check
            return support(*args, cancellation_check=cancellation_check, **kwargs)
        monkeypatch.setattr(bi, "plane_cylinder_support", polling_support)

    fired = []

    def cancellation():                                 # one cancellation, raised inside the target
        if armed and not fired:
            fired.append(1)
            raise sentinel
        return False
    policy = IntersectionBatchPolicy(planning_mode="topology_certified", cancellation_check=cancellation)
    with pytest.raises(_Cancel) as info:
        plan_intersections(model, faces, policy=policy)
    assert info.value is sentinel

    class Swallow(bi._TopologyRoute):                    # double: converts any exception to a fallback
        def pair(self, *args):
            try:
                return super().pair(*args)
            except Exception:
                return None
    fired.clear()
    monkeypatch.setattr(bi, "_TopologyRoute", Swallow)
    plan = plan_intersections(model, faces, policy=policy)     # the double loses the cancellation
    assert fired and type(plan) is TopologyCertifiedPlan


def test_cancellation_flag_gives_the_same_refusal_text_in_both_modes():
    model, strips, fin = _strips_and_fin()
    faces = (*strips, fin)
    results = []
    for mode in (None, "topology_certified"):
        calls = []

        def cancellation():
            calls.append(1)
            return len(calls) > 3
        results.append(_refusal(lambda: plan_intersections(model, faces, policy=IntersectionBatchPolicy(
            planning_mode=mode, cancellation_check=cancellation))))
    assert all(r and "cancelled" in r for r in results)


# ----------------------------------------------------------------------------------- review round 1

def test_a9b_orientation_check_is_reached_through_the_route(monkeypatch):
    """A9 4(b) end to end: production-coincident strips, the certificate flags coincidence, the
    second face's loop runs the other way (unequal orientation sign): counted as ineligible; the
    output is the exact one. The double without the production orientation call counts it eligible."""
    model, strips, _fin = _strips_and_fin()
    first, second = (MaterialDomain.from_model(model, f) for f in strips[:2])
    flipped = MaterialDomain(second.face_id, second.support, tuple(
        tuple(replace(path, curve=path.curve.subcurve(1., 0.)) for path in reversed(loop))
        for loop in second.boundaries))

    def run(route_type):
        route = route_type(model)
        route.certifier = _StubCertifier(coincident=True)
        route._records[id(flipped)] = (flipped, route._record(second))
        exact = bi._pair_paths(first, flipped, model, 1e-9, lambda: None, point_contacts=[])
        with collect_runtime_diagnostics() as counters:
            out = bi._pair_paths(first, flipped, model, 1e-9, lambda: None, point_contacts=[], route=route)
        assert len(out) == len(exact)
        return counters
    real = run(bi._TopologyRoute)
    assert real.get("t2b_fallback:coincident_support_ineligible:radial_or_orientation") == 1
    assert not real.get("t2b_coincident_exact_output")

    class NoOrientation(bi._TopologyRoute):
        def _orientation(self, domain):
            return 1
    assert run(NoOrientation).get("t2b_coincident_exact_output") == 1      # the double misses it


def test_a9d_cancellation_inside_the_coincident_overlap_shadow(monkeypatch):
    model, faces = _lobe_strips()
    sentinel = _Cancel("stop in overlap check")
    original = bi._coplanar_traces
    fired = []

    def overlap(first, second, tolerance, check, model):
        check()                                         # the overlap check polls the planning check
        return original(first, second, tolerance, check, model)
    monkeypatch.setattr(bi, "_coplanar_traces", overlap)

    def cancellation():
        if inside and not fired:
            fired.append(1)
            raise sentinel
        return False
    inside = []
    shadow = bi._pair_paths_exact

    def tracked(*args, _shadow=None, **kwargs):
        if _shadow is not None:
            inside.append(1)
        return shadow(*args, _shadow=_shadow, **kwargs)
    monkeypatch.setattr(bi, "_pair_paths_exact", tracked)
    _with_certifier(monkeypatch, _StubCertifier(coincident=True))
    policy = IntersectionBatchPolicy(planning_mode="topology_certified", cancellation_check=cancellation)
    with pytest.raises(_Cancel) as info:
        plan_intersections(model, faces, policy=policy)
    assert info.value is sentinel and fired


def test_shared_vertices_are_emitted_by_identity_not_merged_within_tolerance():
    """Review R1: two distinct shared vertices 1e-10 apart (r_max < 1e-10 < pair tolerance) are two
    point contacts; the tolerance-merging double drops one."""
    model, faces, _ = _l_joint()
    route = bi._TopologyRoute(model)
    d1, d2 = (MaterialDomain.from_model(model, f) for f in faces)
    plate = sp.PlaneSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    line = lambda a, b: bi.LinePath(a, b)
    p, q, r = (0., 0., 0.), (1e-10, 0., 0.), (1., 1., 0.)
    loop1 = (sp.BoundaryUse(line(p, q), 101, 1, 2), sp.BoundaryUse(line(q, r), 102, 2, 3),
             sp.BoundaryUse(line(r, p), 103, 3, 1))
    loop2 = (sp.BoundaryUse(line(q, p), 201, 2, 1), sp.BoundaryUse(line(p, (0., -1., 1.)), 202, 1, 4),
             sp.BoundaryUse(line((0., -1., 1.), q), 203, 4, 2))
    r1 = sp.FaceRecord(d1.face_id, plate, (loop1,), ((1, p), (2, q), (3, r)))
    r2 = sp.FaceRecord(d2.face_id, plate, (loop2,), ((1, p), (2, q), (4, (0., -1., 1.))))
    route._records[id(d1)], route._records[id(d2)] = (d1, r1), (d2, r2)
    points = []
    assert route._emit(sp.Certified(0.), d1, d2, 1e-9, points) == ()
    assert points == [p, q]
    merged = []
    for point in points:                               # double: merge within the pair tolerance
        if not any(np.linalg.norm(np.asarray(point)-old) <= 1e-9 for old in merged):
            merged.append(point)
    assert len(merged) == 1


def test_mode_agnostic_routes_and_reapplication_in_topology_mode(monkeypatch):
    """Review R2: plans of the existing-material route (a re-plan of a prepared model) carry no mode
    and apply under a topology_certified or environment-selected mode; an explicit exact is not
    stored."""
    from anygeometry.existing_material_preparation import is_existing_plan
    model, faces, _ = _l_joint()
    apply_intersections(model, plan_intersections(model, faces, policy=T), policy=T)
    replanned = plan_intersections(model, tuple(model.faces), policy=T)
    if is_existing_plan(replanned) or type(replanned) is IntersectionPlan:
        assert replanned.policy.planning_mode is None
    apply_intersections(model, replanned, policy=T)
    monkeypatch.setenv(bi.PLANNING_MODE_ENVIRONMENT, "topology_certified")
    again = plan_intersections(model, tuple(model.faces), policy="connect")
    apply_intersections(model, again, policy="connect")
    explicit = plan_intersections(model, tuple(model.faces), policy=IntersectionBatchPolicy(planning_mode="exact"))
    assert explicit.policy.planning_mode is None and b"planning_mode" not in pickle.dumps(explicit.policy)


def test_cancellation_geometry_error_is_wrapped_exactly_as_in_exact_mode(monkeypatch):
    """Review R6: a GeometryError raised by the callback inside the certificate reaches the caller
    through the same pair-loop wrapper as in exact mode, with the original object as __cause__."""
    model, faces, _ = _l_joint()
    original = sp.SPCertifier.certify
    error = GeometryError("host cancelled")
    state = {"armed": False}

    def certify(self, *args, **kwargs):
        state["armed"] = True
        try:
            return original(self, *args, **kwargs)
        finally:
            state["armed"] = False
    monkeypatch.setattr(sp.SPCertifier, "certify", certify)

    def cancellation():
        if state["armed"]:
            raise error
        return False
    with pytest.raises(GeometryError) as info:
        plan_intersections(model, faces, policy=IntersectionBatchPolicy(planning_mode="topology_certified",
                                                                        cancellation_check=cancellation))
    assert info.value.__cause__ is error
    assert str(info.value) == f"intersection batch face:{faces[0]}/face:{faces[1]}: host cancelled"


@pytest.mark.parametrize("route", ["existing", "bounded"])
def test_mode_agnostic_early_routes_are_selected_and_an_invalid_mode_is_refused(monkeypatch, route):
    """Review R2/R2-new: the existing-material and bounded routes (spied: selected by their own
    predicates) get no stored mode at plan time and an exact validation at apply time, under T,
    env-T, exact and budget-forced exact; an invalid environment value at apply is refused first.
    Fix round 6: they are handed the caller's policy object itself (identity preserved), the
    mode-free copy only as ``stored_policy`` and the exact mode only as ``planning_mode``."""
    import anygeometry.existing_material_preparation as em
    import anygeometry.bounded_nurbs as bn
    model, faces, _ = _l_joint()
    seen = {"plan": [], "apply": []}
    plain = plan_intersections(model, faces, policy="connect")

    def fake_plan(model_, handles, policy, *, stored_policy):
        seen["plan"].append((policy, stored_policy))
        return plain
    def fake_apply(model_, plan, policy, *, planning_mode):
        seen["apply"].append((policy, planning_mode))
        bi._validate_policy(plan, policy, planning_mode=planning_mode)
        return "applied"
    if route == "existing":
        monkeypatch.setattr(em, "try_plan_existing", fake_plan)
        monkeypatch.setattr(em, "is_existing_plan", lambda plan: True)
        monkeypatch.setattr(em, "apply_existing", fake_apply)
    else:
        monkeypatch.setattr(em, "try_plan_existing", lambda *args, **kwargs: None)
        monkeypatch.setattr(bn, "handles_bounded", lambda *args: True)
        monkeypatch.setattr(bn, "plan_bounded", fake_plan)
        monkeypatch.setattr(bn, "is_bounded_plan", lambda plan: True)
        monkeypatch.setattr(bn, "apply_bounded", fake_apply)
    policies = [T, IntersectionBatchPolicy(planning_mode="exact"),
                IntersectionBatchPolicy(planning_mode="topology_certified", max_predicates=10**9)]
    for policy in policies:
        assert plan_intersections(model, faces, policy=policy) is plain
        assert apply_intersections(model, plain, policy=policy) == "applied"
        assert seen["plan"][-1][0] is policy and seen["apply"][-1][0] is policy
        assert seen["plan"][-1][1] == replace(policy, planning_mode=None)
    with pytest.raises(GeometryError, match="cannot be applied in topology_certified mode"):
        bi._validate_policy(plain, T)             # without the override the T policy is refused
    monkeypatch.setenv(bi.PLANNING_MODE_ENVIRONMENT, "topology_certified")
    assert plan_intersections(model, faces, policy="connect") is plain
    assert apply_intersections(model, plain, policy="connect") == "applied"
    assert all(s.planning_mode is None for _, s in seen["plan"]) and len(seen["plan"]) == 4
    assert all(m == "exact" for _, m in seen["apply"]) and len(seen["apply"]) == 4
    monkeypatch.setenv(bi.PLANNING_MODE_ENVIRONMENT, "bogus")
    with pytest.raises(GeometryError, match="ANYGEOMETRY_PLANNING_MODE"):
        apply_intersections(model, plain, policy="connect")
    with pytest.raises(GeometryError, match="ANYGEOMETRY_PLANNING_MODE"):
        plan_intersections(model, faces, policy="connect")
    assert len(seen["apply"]) == 4


# ----------------------------------------------------------------------------------- fix round 4 (R4): row 3 persisted

_IDENTITY = ("planning_mode", "certificate_version", "r_max", "route_tolerance", "content_checksum", "route_checksum")


def test_row_3_persists_the_route_identity_in_change_set_hooks_and_model_record(plans):
    """Review R4 / parent decision (a): a topology-certified application writes the mode, r_max and
    certificate version (with the tolerance and both checksums that bind them) into the change set
    that the application returns, that model.last_change_set holds and that change hooks receive,
    and into the model document's ``anygeometry:intersection_planning`` record, which survives
    to_dict/from_dict. Fails on 2736b9e (no persisted metadata)."""
    model, exact, certified = plans
    model_e = model.clone(preserve_identity=True)
    seen = []
    model.add_change_hook(seen.append)
    applied = apply_intersections(model, certified, policy=T)
    change_set = applied.change_set
    assert type(change_set) is bi.TopologyCertifiedChangeSet and isinstance(change_set, ChangeSet)
    assert {name: getattr(change_set, name) for name in _IDENTITY} == {
        name: getattr(certified, name) for name in _IDENTITY}
    assert change_set.revision_after == model.revision
    exact_set = apply_intersections(model_e, exact, policy="connect").change_set   # the same publication, exact
    assert type(exact_set) is ChangeSet and all(
        getattr(change_set, item.name) == getattr(exact_set, item.name) for item in fields(ChangeSet))
    assert model.last_change_set is change_set and seen == [change_set]
    assert pickle.loads(pickle.dumps(change_set)) == change_set
    record = bi.intersection_planning_record(model)
    assert len(record) == 1
    entry = record[0]
    assert {name: entry[name] for name in _IDENTITY} == {name: getattr(certified, name) for name in _IDENTITY}
    assert (entry["source_revision"], entry["source_checksum"]) == (certified.revision, certified.source_checksum)
    document = to_dict(model)
    assert document["extensions"][bi.INTERSECTION_PLANNING_EXTENSION]["publications"] == [dict(entry)]
    loaded = from_dict(document)
    assert bi.intersection_planning_record(loaded) == record and to_dict(loaded) == document


def test_row_1_exact_application_writes_no_route_metadata(plans):
    """Exact plans carry none: an exact application returns and stores a plain ChangeSet, writes no
    document extension and no record (default outputs unchanged)."""
    model, exact, _certified = plans
    seen = []
    model.add_change_hook(seen.append)
    applied = apply_intersections(model, exact, policy="connect")
    assert type(applied.change_set) is ChangeSet and type(model.last_change_set) is ChangeSet
    assert [type(item) for item in seen] == [ChangeSet]
    assert to_dict(model)["extensions"] == {} and bi.intersection_planning_record(model) == ()
    for marker in (b"planning_mode", b"route_checksum", b"TopologyCertified", b"topology_certified"):
        assert marker not in pickle.dumps(applied.change_set)


def test_second_topology_application_appends_and_an_exact_one_leaves_the_record(plans):
    model, _exact, certified = plans
    apply_intersections(model, certified, policy=T)
    first = bi.intersection_planning_record(model)
    replanned = plan_intersections(model, tuple(model.faces), policy=T)
    apply_intersections(model, replanned, policy=T)
    after_t = bi.intersection_planning_record(model)
    assert after_t[:1] == first and len(after_t) in (1, 2)
    if type(replanned) is TopologyCertifiedPlan and len(after_t) == 2:
        assert after_t[1]["route_checksum"] == replanned.route_checksum
    exact = plan_intersections(model, tuple(model.faces), policy="connect")
    apply_intersections(model, exact, policy="connect")
    assert bi.intersection_planning_record(model) == after_t          # exact writes nothing


def _record_document(plans):
    model, _exact, certified = plans
    apply_intersections(model, certified, policy=T)
    return to_dict(model)


def _reseal_entry(entry):
    entry["route_checksum"] = bi.definition_checksum((entry["content_checksum"], entry["planning_mode"],
                                                      entry["certificate_version"], entry["r_max"]))


# tamper id -> (mutation of one publication entry, reseal the route checksum, the check whose removal
# alone makes the record load, or None when another check also rejects it)
_RECORD_TAMPERS = {
    "extra_field": (lambda e: e.update(note=1), False, "_record_fields"),
    "missing_version": (lambda e: e.pop("certificate_version"), False, None),
    "mode_exact": (lambda e: e.update(planning_mode="exact"), True, "_route_row_11"),
    "mode_other": (lambda e: e.update(planning_mode="topology"), True, "_route_row_10"),
    "version_unsupported": (lambda e: e.update(certificate_version="sp-exact-t2-1a-0"), True, "_route_row_7"),
    "r_max_not_contract": (lambda e: e.update(r_max=2*e["r_max"]), True, "_route_row_8"),
    "route_checksum_mismatch": (lambda e: e.update(route_checksum="0"*64), False, "_route_row_6"),
    "route_checksum_empty": (lambda e: e.update(route_checksum=""), False, None),
    "source_revision_bool": (lambda e: e.update(source_revision=True), False, "_route_row_14"),
    # fix round 5 (R6): the same rows 12 and 14 as for plans and change sets
    "content_checksum_none": (lambda e: e.update(content_checksum=None), True, "_route_row_12"),
    "content_checksum_short": (lambda e: e.update(content_checksum=e["content_checksum"][:-1]), True, "_route_row_12"),
    "source_checksum_upper": (lambda e: e.update(source_checksum=e["source_checksum"].upper()), False, "_route_row_14"),
}


@pytest.mark.parametrize("tamper", sorted(_RECORD_TAMPERS))
def test_row_14_model_record_load_check_and_per_check_double(plans, tamper, monkeypatch):
    """New matrix row 14 (fix round 4): a model document whose topology-certified record has a
    missing, extra or mismatched field is rejected at load (from_dict), with a correct document
    checksum. The double of the named check alone (the same section-7 row functions as for plans,
    plus the record's field check) makes the document load; tampers rejected by two checks are
    discriminated by the reason only."""
    document = _record_document(plans)
    assert from_dict(document) is not None                          # the untampered record loads
    mutate, reseal, check = _RECORD_TAMPERS[tamper]
    entry = document["extensions"][bi.INTERSECTION_PLANNING_EXTENSION]["publications"][0]
    mutate(entry)
    if reseal:
        _reseal_entry(entry)
    document["checksum"] = _checksum(document)
    with pytest.raises(GeometryError, match="intersection planning record") as info:
        from_dict(document)
    if check is None:
        return
    monkeypatch.setattr(bi, check, (lambda value: None))
    if check == "_route_row_8":
        monkeypatch.setattr(bi, "_route_row_8_model", lambda plan, model: None)
    assert from_dict(document) is not None, info.value


@pytest.mark.parametrize("record", [[], {"format": 1}, {"format": 2, "publications": []},
                                    {"format": 1, "publications": []}, "x"])
def test_row_14_malformed_record_container_is_rejected(plans, record):
    document = _record_document(plans)
    document["extensions"][bi.INTERSECTION_PLANNING_EXTENSION] = record
    document["checksum"] = _checksum(document)
    with pytest.raises(GeometryError, match="intersection planning record"):
        from_dict(document)


_CHANGE_SET_TAMPERS = {
    "mode_exact": ({"planning_mode": "exact"}, True, "_route_row_11"),
    "mode_none": ({"planning_mode": None}, True, None),
    "mode_other": ({"planning_mode": "topology"}, True, "_route_row_10"),
    "version_unsupported": ({"certificate_version": "sp-exact-t2-1a-0"}, True, "_route_row_7"),
    "version_missing": ({"certificate_version": None}, True, "_route_row_7"),
    "r_max_not_contract": ({"r_max": 2e-12}, True, "_route_row_8"),
    "r_max_missing": ({"r_max": None}, True, "_route_row_8"),
    "route_checksum_mismatch": ({"route_checksum": "0"*64}, False, "_route_row_6"),
    "route_checksum_missing": ({"route_checksum": None}, False, None),
    # fix round 5 (review R6): content_checksum None, "" and 123 were accepted with a resealed route checksum
    "content_checksum_none": ({"content_checksum": None}, True, "_route_row_12"),
    "content_checksum_empty": ({"content_checksum": ""}, True, "_route_row_12"),
    "content_checksum_int": ({"content_checksum": 123}, True, "_route_row_12"),
    "revision_bool": ({"revision_before": True}, False, "_route_row_14"),
}


@pytest.mark.parametrize("tamper", sorted(_CHANGE_SET_TAMPERS))
def test_row_14_change_set_metadata_is_checked_at_construction_and_load(plans, tamper, monkeypatch):
    model, _exact, certified = plans
    change_set = apply_intersections(model, certified, policy=T).change_set
    changes, reseal, check = _CHANGE_SET_TAMPERS[tamper]
    values = {item.name: getattr(change_set, item.name) for item in fields(change_set)}
    values.update(changes)
    if reseal:
        values["route_checksum"] = bi.definition_checksum((values["content_checksum"], values["planning_mode"],
                                                           values["certificate_version"], values["r_max"]))
    with pytest.raises(GeometryError):
        bi.TopologyCertifiedChangeSet(**values)
    forged = object.__new__(bi.TopologyCertifiedChangeSet)
    for name, value in values.items():
        object.__setattr__(forged, name, value)
    blob = pickle.dumps(forged)
    with pytest.raises(GeometryError):
        pickle.loads(blob)
    if check is None:
        return
    monkeypatch.setattr(bi, check, (lambda value: None))
    assert pickle.loads(blob) == forged


def test_topology_certified_plan_is_refused_inside_a_caller_transaction(plans, monkeypatch):
    """The route record cannot join the caller's journal (a rollback would keep a record of a
    publication that never happened), so a state-T plan is applied only through the committed owner
    path; exact plans still join the caller's transaction. The double without this refusal applies
    the T plan inside the transaction and leaves the model WITHOUT its row-3 record."""
    model, exact, certified = plans
    before = to_dict(model)
    with pytest.raises(GeometryError, match="committed owner path"):
        with model.transaction():
            apply_intersections(model, certified, policy=T)
    assert to_dict(model) == before
    other = model.clone(preserve_identity=True)
    with other.transaction():
        apply_intersections(other, exact, policy="connect")
    assert other.revision == model.revision + 1
    monkeypatch.setattr(bi, "_refuse_journaled_route", lambda plan: None)
    with model.transaction():
        apply_intersections(model, certified, policy=T)
    assert model.revision == certified.revision + 1 and bi.intersection_planning_record(model) == ()


# ----------------------------------------------------------------------------------- fix round 5 (R6): one validator
#
# Review R6 found content_checksum None/""/123 accepted by change-set construction and unpickling.
# The class: every field of every route carrier (TopologyCertifiedPlan, TopologyCertifiedChangeSet,
# a model record entry, TopologyCertifiedApplication) is validated by one shared validator in every
# path (construction, unpickling, apply, document load, record read). The kind of every field is
# listed here; the tables must cover the dataclass fields exactly, so a new field needs a kind.

class _PolicySubclass(IntersectionBatchPolicy):           # equal values, not the exact type
    __slots__ = ()


_IDENTITY_KINDS = {"planning_mode": "mode", "certificate_version": "version", "r_max": "r_max",
                   "route_tolerance": "tolerance", "content_checksum": "checksum", "route_checksum": "checksum"}
_KEY_FIELDS = ("added", "removed", "modified", "ownership_changes", "member_changes", "attachment_changes",
               "tag_changes", "invalidated_caches", "spatial_updates")
_CARRIER_KINDS = {
    "plan": {"model_id": "uuid", "revision": "revision", "source_checksum": "checksum", "operands": "tuple",
             "policy": "policy", "arrangements": "tuple", "axes": "tuple", "contacts": "tuple",
             "face_contacts": "tuple", "r_pairs": "r_pairs", **_IDENTITY_KINDS},
    "change_set": {"revision_before": "revision", "revision_after": "revision", "replacements": "tuple",
                   "group_changes": "tuple", "affected_aabbs": "tuple", "feature_history_changed": "bool",
                   "document_settings_changed": "bool", **{name: "tuple" for name in _KEY_FIELDS}, **_IDENTITY_KINDS},
    "record": {"source_revision": "revision", "source_checksum": "checksum", **_IDENTITY_KINDS},
    "application": {"plan": "plan", "change_set": "change_set", "joint_edges": "tuple", "reused": "bool",
                    "planning_mode": "mode", "r_max": "r_max", "certificate_version": "version"},
}
_COMMON_BAD = (None, "", 123, True, math.nan, math.inf, -1, -1.0, [], {})


def _bad_values(kind, valid, exact):
    """Bad values of one field kind: the common ones that are not valid for the kind, then the
    kind's own (wrong form, wrong range, inconsistent, numpy scalars, wrong container elements)."""
    common = [v for v in _COMMON_BAD if not ((kind == "revision" and type(v) is int and v >= 0)
                                             or (kind == "bool" and type(v) is bool))]
    own = {
        "checksum": lambda: ["0"*63, "0"*65, "A"*64, "g"*64, ("0"*64,)],
        "mode": lambda: ["exact", "topology", "TOPOLOGY_CERTIFIED", ("topology_certified",)],
        "version": lambda: ["sp-exact-t2-1a-0", sp.CERTIFICATE_VERSION.upper()],
        "r_max": lambda: [0., 5e-324, -valid, 2*valid, math.nextafter(valid, math.inf), np.float64(valid)],
        "tolerance": lambda: [0., 5e-324, -valid, 2*valid, np.float64(valid)],
        "revision": lambda: [-5, 1.0, np.int64(valid)],
        "uuid": lambda: [str(valid), valid.hex],
        "bool": lambda: [0, 1, "False", np.bool_(valid)],
        "tuple": lambda: [(1,), (("face", True),), [*valid]],
        "r_pairs": lambda: [("x",), ((1, 2),), ((1, 2, None),), ((1, 2, math.nan),), ((0, 2, 0.),),
                            ((True, 2, 0.),), ((1, 2, -0.5),)],
        "policy": lambda: [_PolicySubclass(), "connect"],
        "plan": lambda: [exact],
        "change_set": lambda: [ChangeSet(0, 0)],
    }[kind]()
    return [v for v in [*common, *own] if not (type(v) is type(valid) and repr(v) == repr(valid))]


@pytest.fixture
def carriers(plans):
    model, exact, certified = plans
    applied = apply_intersections(model.clone(preserve_identity=True), certified, policy=T)
    document = _record_document((model.clone(preserve_identity=True), exact, certified))   # model stays current
    return model, exact, certified, applied, document


def _refused(call):
    """``None`` when ``call`` raises GeometryError, else what happened instead."""
    try:
        call()
    except GeometryError:
        return None
    except Exception as exc:                             # noqa: BLE001 - wrong refusal type is a finding
        return f"{type(exc).__name__}: {exc}"
    return "accepted"


def _dataclass_paths(carrier):
    """Construction, unpickling under every protocol, and unpickling a state with a missing or an
    extra item, for a dataclass carrier."""
    cls = type(carrier)
    values = {item.name: getattr(carrier, item.name) for item in fields(carrier)}
    paths = {"construct": lambda: cls(**values)}
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        paths[f"unpickle-p{protocol}"] = (lambda p: lambda: pickle.loads(pickle.dumps(carrier, p)))(protocol)
    return paths


def _record_paths(document, entry_changes, model):
    """Document load (from_dict with a correct document checksum), the record validator itself and
    the model record read, for a document whose first publication entry has ``entry_changes``."""
    import copy
    document = copy.deepcopy(document)
    record = document["extensions"][bi.INTERSECTION_PLANNING_EXTENSION]
    entry = record["publications"][0]
    for name, value in entry_changes.items():
        if value is _MISSING:
            entry.pop(name)
        else:
            entry[name] = value
    if "route_checksum" not in entry_changes and all(n in entry for n in ("content_checksum", "planning_mode",
                                                                          "certificate_version", "r_max")):
        try:
            _reseal_entry(entry)
        except (ValueError, TypeError):
            pass
    paths = {"validate": lambda: bi.validate_intersection_planning_record(record)}
    reader = model.clone(preserve_identity=True)
    reader._serialization_extensions = {bi.INTERSECTION_PLANNING_EXTENSION: record}
    paths["record-read"] = lambda: bi.intersection_planning_record(reader)
    try:
        document["checksum"] = _checksum(document)
    except (ValueError, TypeError):                      # a value no JSON document can hold
        return paths
    paths["from_dict"] = lambda: from_dict(document)
    return paths


_MISSING = object()


def _route_resealed(values):
    if all(name in values for name in ("content_checksum", "planning_mode", "certificate_version", "r_max")) \
            and "route_checksum" in values:
        try:
            values["route_checksum"] = bi.definition_checksum((values["content_checksum"], values["planning_mode"],
                                                               values["certificate_version"], values["r_max"]))
        except (ValueError, TypeError):
            pass
    return values


def test_route_carrier_kind_tables_cover_every_field():
    for name, cls in (("plan", TopologyCertifiedPlan), ("change_set", bi.TopologyCertifiedChangeSet),
                      ("application", bi.TopologyCertifiedApplication)):
        assert set(_CARRIER_KINDS[name]) == {item.name for item in fields(cls)}, name
    assert set(_CARRIER_KINDS["record"]) == set(bi._RECORD_FIELDS)


@pytest.mark.parametrize("carrier_name, field_name", [(c, f) for c in ("plan", "change_set", "application")
                                                      for f in _CARRIER_KINDS[c]])
def test_every_carrier_field_is_validated_in_every_path(carriers, carrier_name, field_name):
    """Each bad value of the field, in each path, is refused with GeometryError (never accepted,
    never another exception type). Route fields are resealed (route checksum recomputed) wherever
    possible, so the field's own check is the one under test. Fails on 1ea67e2: construction of a
    plan and an application validated nothing; content_checksum None/""/123 passed the change set."""
    model, exact, certified, applied, _document = carriers
    carrier = {"plan": certified, "change_set": applied.change_set, "application": applied}[carrier_name]
    valid = getattr(carrier, field_name)
    failures = []
    for value in _bad_values(_CARRIER_KINDS[carrier_name][field_name], valid, exact):
        values = {item.name: getattr(carrier, item.name) for item in fields(carrier)}
        values[field_name] = value
        if field_name != "route_checksum":
            _route_resealed(values)
        forged = _forge(carrier, **values)
        paths = _dataclass_paths(forged)
        if carrier_name == "plan":
            paths["apply"] = lambda: apply_intersections(model.clone(preserve_identity=True), forged, policy=T)
        for path, call in paths.items():
            outcome = _refused(call)
            if outcome is not None:
                failures.append((field_name, repr(value), path, outcome[:120]))
    assert not failures, failures


@pytest.mark.parametrize("carrier_name", ["plan", "change_set", "application"])
def test_carrier_state_with_a_missing_or_extra_item_is_refused(carriers, carrier_name):
    _model, _exact, certified, applied, _document = carriers
    carrier = {"plan": certified, "change_set": applied.change_set, "application": applied}[carrier_name]
    state = [getattr(carrier, item.name) for item in fields(carrier)]
    for bad in (state[:-1], [*state, None], []):
        blank = object.__new__(type(carrier))
        assert _refused(lambda: blank.__setstate__(bad)) is None


@pytest.mark.parametrize("field_name", sorted(_CARRIER_KINDS["record"]))
def test_every_record_field_is_validated_in_every_path(carriers, field_name):
    """The model record entry (extensions['anygeometry:intersection_planning']): every bad value and
    the missing field are refused at document load, by the record validator and by the record read."""
    model, exact, certified, _applied, document = carriers
    entry = document["extensions"][bi.INTERSECTION_PLANNING_EXTENSION]["publications"][0]
    valid = entry[field_name]
    failures = []
    for value in [*_bad_values(_CARRIER_KINDS["record"][field_name], valid, exact), _MISSING]:
        for path, call in _record_paths(document, {field_name: value}, model).items():
            outcome = _refused(call)
            if outcome is not None:
                failures.append((field_name, "missing" if value is _MISSING else repr(value), path, outcome[:120]))
    assert not failures, failures


def test_record_container_and_extra_entry_field_are_refused(carriers):
    model, _exact, _certified, _applied, document = carriers
    for changes in ({"note": 1}, {"source_revision ": 1}):
        for path, call in _record_paths(document, changes, model).items():
            assert _refused(call) is None, (changes, path)
    record = document["extensions"][bi.INTERSECTION_PLANNING_EXTENSION]
    for bad in ({**record, "format": True}, {**record, "format": 1.0}, {**record, "publications": tuple(record["publications"])},
                {**record, "publications": [tuple(record["publications"][0].items())]}, dict(record, extra=1)):
        assert _refused(lambda: bi.validate_intersection_planning_record(bad)) is None, bad


def test_route_carriers_round_trip_unchanged(carriers):
    """Positive round trip: valid carriers construct, unpickle under every protocol to equal bytes,
    and the record survives to_dict/from_dict."""
    model, _exact, certified, applied, document = carriers
    for carrier in (certified, applied.change_set, applied):
        values = {item.name: getattr(carrier, item.name) for item in fields(carrier)}
        rebuilt = type(carrier)(**values)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            blob = pickle.dumps(carrier, protocol)
            assert pickle.dumps(rebuilt, protocol) == blob
            # The first trip may change object sharing inside the arrangements (pickle memo, as on
            # 1ea67e2); from then on the bytes are stable.
            once = pickle.dumps(pickle.loads(blob), protocol)
            assert pickle.dumps(pickle.loads(once), protocol) == once
    assert pickle.loads(pickle.dumps(applied.change_set)) == applied.change_set
    loaded = from_dict(document)
    assert to_dict(loaded) == document
    assert bi.intersection_planning_record(loaded) == tuple(document["extensions"][
        bi.INTERSECTION_PLANNING_EXTENSION]["publications"])


# ----------------------------------------------------------------------------------- fix round 4 (R5): cc_cross

def _mitre(beta=0.35, half=0.3):
    """Two equal-radius cylinder strips with NONPARALLEL axes (angle 2*beta) meeting on the mitre
    plane z = tan(beta)*x: the shared edge is an ellipse arc (theta in [-half, half], away from the
    two double points at theta = +-pi/2). The exact path reaches the nonparallel branch of
    cylinder_cylinder_support (cc_cross); the real certificate certifies the pair."""
    model = GeometryModel()
    t = math.tan(beta)
    n = (-math.sin(beta), 0., math.cos(beta))

    def reflect(p):
        d = sum(p[i]*n[i] for i in range(3))
        return tuple(p[i]-2*d*n[i] for i in range(3))
    P = lambda th: (math.cos(th), math.sin(th), t*math.cos(th))
    Q = lambda th: (math.cos(th), math.sin(th), -1.)
    p0, p1 = model.add_point(*P(-half)), model.add_point(*P(half))
    q0, q1 = model.add_point(*Q(-half)), model.add_point(*Q(half))
    ellipse = model.add_curve(p0, p1, EllipticArc((0., 0., 0.), (1., 0., t), (0., 1., 0.), -half, 2*half))
    bottom = model.add_arc(q1, model.add_point(*Q(0.)), q0)
    first = model.add_face_from_loop(
        (OrientedEdge(ellipse, True), OrientedEdge(model.add_line(p1, q1), True), OrientedEdge(bottom, True),
         OrientedEdge(model.add_line(q0, p0), True)),
        surface=Cylinder(radius=1., origin=(0., 0., -1.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
                         height=1+t, start_angle=-half, sweep_angle=2*half))
    r0, r1 = model.add_point(*reflect(Q(-half))), model.add_point(*reflect(Q(half)))
    bottom2 = model.add_arc(r0, model.add_point(*reflect(Q(0.))), r1)
    origin = reflect((0., 0., -1.))
    second = model.add_face_from_loop(
        (OrientedEdge(ellipse, False), OrientedEdge(model.add_line(p0, r0), True), OrientedEdge(bottom2, True),
         OrientedEdge(model.add_line(r1, p1), True)),
        surface=Cylinder(radius=1., origin=origin, axis=tuple(-2*n[i]*n[2] + (i == 2) for i in range(3)),
                         radial_direction=tuple((i == 0) - 2*n[i]*n[0] for i in range(3)), height=1+t,
                         start_angle=-half, sweep_angle=2*half))
    return model, (first, second)


class _NoShadow(bi._TopologyRoute):
    """Double: certified pairs skip the support shadow."""

    def pair(self, first, second, model, tolerance, check, point_contacts, face_tolerances):
        verdict = self._certify(first, second, check)
        return None if verdict is None else self._emit(verdict, first, second, tolerance, point_contacts)


def test_cc_cross_mitre_is_certified_and_its_nonparallel_support_runs_in_shadow(monkeypatch):
    model, faces = _mitre()
    calls = []
    original = bi.cylinder_cylinder_support

    def spy(first, second, **kwargs):
        calls.append(float(np.linalg.norm(np.cross(first.axis, second.axis))))
        return original(first, second, **kwargs)
    monkeypatch.setattr(bi, "cylinder_cylinder_support", spy)
    plan_intersections(model, faces, policy="connect")
    assert len(calls) == 1 and calls[0] > .5                         # exact path: nonparallel branch
    calls.clear()
    plan, counters = _counters(lambda: plan_intersections(model, faces, policy=T))
    assert counters["t2b_certified"] == 1 and [item[:2] for item in plan.r_pairs] == [tuple(faces)]
    assert len(calls) == 1 and calls[0] > .5                         # the same call, in shadow
    calls.clear()
    monkeypatch.setattr(bi, "_TopologyRoute", _NoShadow)
    plan_intersections(model, faces, policy=T)
    assert calls == []                                               # the double never reaches it


class _InjectedZero(Exception):
    pass


def _cc_injections():
    """The nonparallel cc support stage, analytic_supports.cylinder_cylinder_support (line refs at
    t21b fix round 4), and every refusal it can raise:
      L162-163  type guard (two Cylinders): unreachable for certified pairs (records exist only for
                Plane/Cylinder supports and the family dispatch sends Cylinder/Cylinder here); the
                shadow runs it anyway.
      L166-167  parallel axes -> _parallel (L220-252): not cc_cross (|a1 x a2| != 0 here).
      L174-176  transition events: _linear_events -> _events (L28-35) -> analytic_roots.
                trigonometric_roots: GeometryError (invalid constraint/interval, zero divisor,
                inconsistent division/isolation, cancellation "analytic root isolation cancelled").
      L179-190  cap and seam plane events: _plane_branch_events (L144-157) -> _events/_linear_events:
                the same root-isolation refusals and cancellation.
      L203-205  CylinderIntersectionCurve(...) (exact_curves.py:337): GeometryError (branch,
                parameterization, parallel axes, chart not split at a discriminant transition) and
                its evaluate(.5).
      L212      float division by w.w: ZeroDivisionError if w.w rounds to 0 (not a GeometryError;
                injected as a non-GeometryError stand-in at the curve constructor).
      Any exception of the callback (cancellation) raised through cancellation_check.
    batch_intersections cc branch (_pair_paths_exact, L1048-1054 at fix round 4): result.coincident is
    False on the nonparallel branch, so the coincident sub-branch is unreachable; segments is always ()
    there, so no LinePath is built.
    Parity argument: on a certified pair the route calls _pair_paths_exact(_shadow=...) which makes
    the identical call (same supports, tolerance and cancellation callback) at the same point of the
    pair loop with no handler interposed, so every exception above propagates exactly as in exact
    mode. Each family is injected below (the type guard by replacing the whole entry point)."""
    def raising(error):
        def refuse(*args, **kwargs):
            raise error
        return refuse
    return {
        "entry": (bi, "cylinder_cylinder_support", raising(GeometryError("injected cc entry refusal"))),
        "transition_events": (analytic_supports, "_linear_events",
                              raising(GeometryError("injected transition root refusal"))),
        "plane_branch_events": (analytic_supports, "_plane_branch_events",
                                raising(GeometryError("injected plane-branch root refusal"))),
        "root_isolation": (analytic_supports, "trigonometric_roots",
                           raising(GeometryError("analytic root isolation cancelled"))),
        "branch_curve": (analytic_supports, "CylinderIntersectionCurve",
                         raising(GeometryError("intersection branch chart must split at every discriminant "
                                               "transition"))),
        "zero_division": (analytic_supports, "CylinderIntersectionCurve", raising(_InjectedZero("w.w == 0"))),
    }


@pytest.mark.parametrize("family", sorted(_cc_injections()))
def test_cc_cross_support_refusals_are_raised_in_shadow_on_a_certified_pair(monkeypatch, family):
    """Review R5: each enumerated cc_cross refusal family, injected inside the nonparallel support of
    the certified mitre pair, is raised in topology_certified mode exactly as in exact mode (same
    type and text); the double without the shadow does not raise it."""
    model, faces = _mitre()
    module, name, replacement = _cc_injections()[family]
    monkeypatch.setattr(module, name, replacement)

    def outcome(policy):
        try:
            plan_intersections(model, faces, policy=policy)
        except Exception as exc:                         # noqa: BLE001 - compared below
            return type(exc), str(exc)
        return None
    exact = outcome("connect")
    assert exact is not None and ("injected" in exact[1] or exact[1].endswith(("cancelled", "transition", "w.w == 0")))
    assert outcome(T) == exact
    monkeypatch.setattr(bi, "_TopologyRoute", _NoShadow)
    assert outcome(T) != exact


def test_cc_cross_cancellation_inside_the_nonparallel_support_propagates_unchanged(monkeypatch):
    model, faces = _mitre()
    sentinel = _Cancel("stop inside cc support")
    inside = []
    original = analytic_supports._plane_branch_events

    def polling(first, second, normal, offset, tolerance, cancellation_check):
        inside.append(1)
        cancellation_check()
        return original(first, second, normal, offset, tolerance, cancellation_check)
    monkeypatch.setattr(analytic_supports, "_plane_branch_events", polling)

    def cancellation():
        if inside:
            raise sentinel
        return False
    for mode in (None, "topology_certified"):
        inside.clear()
        with pytest.raises(_Cancel) as info:
            plan_intersections(model, faces, policy=IntersectionBatchPolicy(planning_mode=mode,
                                                                            cancellation_check=cancellation))
        assert info.value is sentinel
    inside.clear()
    monkeypatch.setattr(bi, "_TopologyRoute", _NoShadow)
    plan = plan_intersections(model, faces, policy=IntersectionBatchPolicy(planning_mode="topology_certified",
                                                                           cancellation_check=cancellation))
    assert type(plan) is TopologyCertifiedPlan and not inside


# ----------------------------------------------------------------------------------- naming

def test_on_is_described_as_experimental_topology_certified_planning_not_as_equivalent():
    import pathlib
    texts = [pathlib.Path(bi.__file__).read_text(encoding="utf-8"), pathlib.Path(__file__).read_text(encoding="utf-8")]
    assert "topology-certified planning (experimental)" in texts[0]
    forbidden = "certified" + "-equivalent"
    assert all(forbidden not in text.lower() for text in texts)
