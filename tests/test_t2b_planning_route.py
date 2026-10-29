"""T2-1b: opt-in topology_certified planning route (ADR-T2 v2 sections 1, 4.5, 5, 7; addendum A9).

Every safety rule is checked together with an unsound test double that must fail the same check.
"""
import math
import pickle
from dataclasses import fields, replace

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, to_dict
from anygeometry import batch_intersections as bi
from anygeometry import separating_plane_certificate as sp
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


def _resealed(plan, **changes):
    changed = replace(plan, **changes)
    try:
        return replace(changed, route_checksum=bi._route_checksum(changed))
    except (ValueError, TypeError):                     # non-finite values cannot be bound at all
        return changed


_ROUTE_FIELDS = ("planning_mode", "route_checksum", "certificate_version", "r_max", "r_pairs", "route_tolerance")


def _only_this_field_is_wrong(plan, certified):
    """Discrimination check (the double for every route row is a validator that ignores the
    tampered field): restoring the tampered fields, and only them, makes the plan valid again."""
    changed = {name: getattr(certified, name) for name in _ROUTE_FIELDS
               if repr(getattr(plan, name)) != repr(getattr(certified, name))}
    assert changed, "the tampered plan must differ in a route field"
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
    assert to_dict(left) == to_dict(right)              # equal geometry checksums: identical models


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
        return [replace(certified, route_checksum=value) for value in ("", None)]
    if row == 6:
        return [replace(certified, route_checksum="0"*64), replace(certified, r_max=math.nextafter(certified.r_max, 0.))]
    if row == 7:
        return [_resealed(certified, certificate_version=value) for value in (None, "sp-exact-t2-1a-0", 1)]
    if row == 8:
        tol = 1e-9
        values = [None, math.nan, math.inf, 5e-324, 0., -certified.r_max, 1e-12, "1e-12"]
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
    raise AssertionError(row)


@pytest.mark.parametrize("row", [5, 6, 7, 8, 9, 10, 11])
def test_matrix_malformed_route_rows_are_rejected_at_apply_and_load(plans, row):
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
    tampered = replace(certified, face_contacts=((),))
    with pytest.raises(GeometryError, match="content binding"):
        pickle.loads(pickle.dumps(tampered))
    assert pickle.loads(pickle.dumps(certified)).route_checksum == certified.route_checksum


def test_matrix_row_12_invalid_content_checksum(plans):
    model, exact, certified = plans
    for plan, policy in ((replace(exact, face_contacts=((),)), "connect"),
                         (replace(certified, content_checksum="0"*64), T)):
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


@pytest.mark.parametrize("entry", ["plane_cylinder_support", "cylinder_cylinder_support", "qualified_plane_plane"])
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
