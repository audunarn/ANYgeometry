"""R0 structural audit of rounding-sensitive decision sites (reporting mode).

GENERALITY_ARCHITECTURE.md R.4 instrument 1 / slice R0: the audit is
"initially reporting, not failing". These tests check that the heuristic
scanner (tools/decision_audit.py) finds each kind of site on small snippets,
that the known inventory anchors are detected in the engine, and write the
report next to pytest's basetemp. They do not fail on uncertified sites unless
strict mode is requested with ``ANYGEOMETRY_DECISION_AUDIT_STRICT=1`` (or
``check_certified(report, strict=True)``), which later R-slices switch on.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
STRICT_ENV = "ANYGEOMETRY_DECISION_AUDIT_STRICT"


def _load_audit():
    spec = importlib.util.spec_from_file_location("decision_audit", ROOT / "tools" / "decision_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("decision_audit", module)  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def scan(code, **kwargs):
    return audit.scan_source(textwrap.dedent(code), **kwargs)


def kinds(code):
    return [(site.kind, site.subkind) for site in scan(code)]


def check_certified(report, *, strict=None):
    """Reporting mode returns the uncertified count; strict mode fails on any."""
    strict = os.environ.get(STRICT_ENV) == "1" if strict is None else strict
    uncertified = [site for site in report["sites"] if not site["certified"]]
    if strict:
        assert not uncertified, (
            f"{len(uncertified)} uncertified decision sites, first: "
            + "; ".join(f"{s['file']}:{s['line']} {s['text']}" for s in uncertified[:5]))
    return len(uncertified)


# -- kind (a): tolerance comparisons ----------------------------------------

def test_tolerance_comparison_is_found():
    assert kinds("""
        def f(point, tolerance):
            if np.linalg.norm(point) <= tolerance:
                return 1
    """) == [("tolerance", "tolerance-compare")]
    assert kinds("""
        def f(a, b):
            x = float(a @ b)
            return abs(x) <= 1e-12 or abs(x) < 64*np.finfo(float).eps
    """) == [("tolerance", "tolerance-compare")] * 2
    assert kinds("def f(a, b):\n    return np.isclose(a, b)\n") == [("tolerance", "isclose-call")]


def test_tolerance_only_checks_and_integer_comparisons_are_not_sites():
    assert kinds("""
        def f(items, tolerance):
            if tolerance <= 0:
                raise ValueError
            if len(items) > 3 and items is not None:
                return items
            return foo(items, tolerance=tolerance) == "x"
    """) == []


def test_shifted_bounds_are_thresholds():
    assert kinds("""
        def f(lo, hi, tolerance):
            reject = np.nextafter(lo - 2*tolerance, -np.inf)
            return hi < reject
    """) == [("tolerance", "tolerance-compare")]


# -- kind (b): ordering and selection without thresholds --------------------

def test_ordering_and_selection_sites_are_found():
    found = kinds("""
        def f(loop, count, values):
            deviations = []
            for index in range(count):
                deviations.append(float(np.arccos(loop[index])))
            ranked = sorted(range(count), key=lambda i: (-deviations[i], i))
            cell = math.floor(float(values[0]) / 2.)
            key = round(float(values[1]) / 1e-9)
            area = float(values[2])
            if area > 0:
                pass
            if area == 1.0:
                pass
            return min(1., max(0., area)), np.argmin(values)
    """)
    assert ("ordering", "select-by-float-key") in found
    assert found.count(("ordering", "round-floor")) == 2
    assert ("ordering", "sign-branch") in found
    assert ("ordering", "float-equality") in found
    assert found.count(("ordering", "min-max-value")) == 2
    assert found.count(("ordering", "select-by-float-key")) == 2  # sorted + argmin


def test_integer_orderings_are_not_sites():
    assert kinds("""
        def f(edges, count):
            ids = [edge.id for edge in edges]
            if count == 4:
                return sorted(ids), sorted(edges, key=lambda edge: edge.id), max(len(ids), 2)
    """) == []


# -- kind (c): construction provenance --------------------------------------

def test_derived_constructions_are_found():
    sites = scan("""
        def f(curve, result, model, a, b, u, v):
            first = LinePath(tuple(curve.evaluate(0.)), tuple(curve.evaluate(1.)))
            point = result.witnesses[0]
            plane = Plane(point, u, v)
            stored = LinePath(tuple(model.vertex_position(a)), tuple(model.vertex_position(b)))
            return first, plane, stored
    """)
    assert [(s.kind, s.detail) for s in sites] == [
        ("provenance", "LinePath from evaluate()"), ("provenance", "Plane from point")]


# -- kind (d): library predicates ------------------------------------------

def test_library_predicates_are_found():
    assert kinds("""
        def f(polygon, plane, world):
            return polygon.covers(Point(plane.local_uv(world)))
    """) == [("library", "library-predicate")]


# -- certification and inventory assignment ---------------------------------

def test_coordinate_only_annotation_certifies():
    sites = scan("""
        def f(point, tolerance):
            # coordinate-only: display value, never decides
            return np.linalg.norm(point) <= tolerance
    """)
    assert [(s.certified, s.certification) for s in sites] == [(True, "coordinate-only")]
    assert not scan("""
        def f(point, tolerance):
            # coordinate-only:
            return np.linalg.norm(point) <= tolerance
    """)[0].certified  # a reason is required


def test_allowlisted_exact_predicate_call_certifies(monkeypatch):
    code = """
        from .exact_predicates import certified_compare
        def f(point, tolerance):
            return certified_compare(np.linalg.norm(point), tolerance) and point[0] <= tolerance
    """
    assert not scan(code)[0].certified
    monkeypatch.setattr(audit, "CERTIFIED_EXACT_PREDICATES", frozenset({"certified_compare"}))
    assert [(s.certified, s.certification) for s in scan(code)] == [(True, "exact_predicates")]


def test_inventory_assignment_by_function_region_and_text():
    code = textwrap.dedent("""
        def g(curve, point, tolerance):
            a = np.linalg.norm(point) <= tolerance
            # begin block
            b = abs(float(point[0])) <= tolerance
            # end block
            c = abs(float(point[1])) <= tolerance
            return a, b, c
    """)
    inventory = (audit.InventoryEntry("T1", "m", "g", region=(r"begin block", r"end block")),
                 audit.InventoryEntry("T2", "m", "g", text=r"point\[1\]"))
    sites = audit.assign_inventory(audit.scan_source(code), "m", code, inventory)
    assert [s.inventory_id for s in sites] == ["unassigned", "T1", "T2"]


# -- the engine: anchors and the baseline report ----------------------------

@pytest.fixture(scope="module")
def report(tmp_path_factory):
    made = audit.run_audit(ROOT)
    out = tmp_path_factory.getbasetemp() / "decision_audit"
    out.mkdir(exist_ok=True)
    (out / "decision_sites.json").write_text(json.dumps(made, indent=1) + "\n", encoding="utf-8")
    (out / "decision_sites.md").write_text(audit.markdown_summary(made), encoding="utf-8")
    return made


def _has(report, file, function, subkind, inventory_id):
    return any(site["file"].endswith(file) and site["function"] == function
               and site["subkind"] == subkind and inventory_id in site["inventory_ids"]
               for site in report["sites"])


@pytest.mark.parametrize("file, function, subkind, inventory_id", [
    # Plane.local_uv feeding shapely covers / boundary.covers (R.0 tangency flip).
    ("intersections.py", "clip_line_to_face", "library-predicate", "A1"),
    ("intersections.py", "clip_line_to_face", "library-predicate", "E4"),
    # material_arrangement vertex lookup (C1).
    ("material_arrangement.py", "_VertexLookup.find", "tolerance-compare", "C1"),
    # batch_intersections positive-area overlap probe (section-10 X4).
    ("batch_intersections.py", "_coplanar_traces", "derived-construction", "X4"),
    # model._detect_corners ranking without a tolerance (A9).
    ("model.py", "GeometryModel._detect_corners", "select-by-float-key", "A9"),
    # endpoint identity, broad phase, arc frames, derived tolerances.
    ("arrangement_geometry.py", "point_parameters", "tolerance-compare", "B3"),
    ("batch_intersections.py", "_plan_intersections", "tolerance-compare", "X2"),
    ("curves.py", "arc_frame", "tolerance-compare", "X1"),
    ("tolerance.py", "TolerancePolicy.effective_area", "min-max-value", "X3"),
])
def test_inventory_anchors_are_detected(report, file, function, subkind, inventory_id):
    assert _has(report, file, function, subkind, inventory_id)


def test_report_is_complete_and_reporting_mode_does_not_fail(report, tmp_path_factory):
    assert report["missing_scope_functions"] == []
    assert report["inventory_ids_without_sites"] == []
    assert {"tolerance", "ordering", "provenance"} <= set(report["by_kind"])
    assert report["totals"]["sites"] == len(report["sites"]) > 0
    for site in report["sites"]:
        assert site["inventory_id"] == (site["inventory_ids"][0] if site["inventory_ids"] else "unassigned")
    out = tmp_path_factory.getbasetemp() / "decision_audit"
    assert (out / "decision_sites.json").is_file() and (out / "decision_sites.md").is_file()
    uncertified = check_certified(report)  # strict only when requested
    print(f"decision audit: {report['totals']}; report in {out}")
    assert uncertified >= 0


def test_strict_mode_fails_on_uncertified_sites():
    sample = {"sites": [{"certified": False, "file": "m.py", "line": 1, "text": "x <= tol"}]}
    with pytest.raises(AssertionError, match="1 uncertified"):
        check_certified(sample, strict=True)
    assert check_certified(sample, strict=False) == 1
