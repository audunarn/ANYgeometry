"""The R0 perturbation harness itself works (not that the engine is robust)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import perturbation_harness as harness  # noqa: E402

from anygeometry import Cylinder, Plane  # noqa: E402
from anygeometry.arrangement_geometry import LinePath  # noqa: E402
from anygeometry.exact_curves import EllipticArc  # noqa: E402
from anygeometry import surfaces  # noqa: E402


VALUES = np.asarray([0.0, -0.0, 1.0, -2.5, 1e-300, 5e-324, 1e300, 0.1, -7.25e-17])


@pytest.mark.parametrize("pattern", harness.PATTERNS)
def test_nudge_moves_every_finite_float_by_exactly_one_ulp(pattern):
    perturber = harness.Perturber(pattern, 7)
    moved = perturber.nudge(VALUES)
    assert moved.dtype == np.float64 and moved.shape == VALUES.shape
    for before, after in zip(VALUES, moved):
        assert after != before
        assert after in (math.nextafter(before, math.inf), math.nextafter(before, -math.inf))
        assert harness.ulp_distance(before, after) == 1 or before == 0.0
    if pattern == "+1":
        assert np.all(moved > VALUES)
    if pattern == "-1":
        assert np.all(moved < VALUES)
    scalar = perturber.nudge(1.0)
    assert type(scalar) is float and abs(scalar - 1.0) in (2.0**-52, 2.0**-53)
    assert type(perturber.nudge(np.float64(3.0))) is np.float64
    nested = perturber.nudge((1.0, [2.0, np.asarray([3.0])], "x", 4))
    assert nested[2:] == ("x", 4) and nested[0] != 1.0 and nested[1][0] != 2.0


def test_nudge_leaves_non_finite_and_non_float_values():
    perturber = harness.Perturber("random", 1)
    special = np.asarray([np.inf, -np.inf, np.nan])
    moved = perturber.nudge(special)
    assert np.isinf(moved[0]) and np.isinf(moved[1]) and np.isnan(moved[2])
    assert perturber.nudge(math.inf) == math.inf
    assert perturber.nudge(True) is True and perturber.nudge(3) == 3
    integers = np.arange(3)
    assert perturber.nudge(integers) is integers


def test_random_pattern_is_deterministic_per_seed_and_mixes_signs():
    values = np.linspace(1.0, 2.0, 64)
    first = harness.Perturber("random", 42).nudge(values)
    again = harness.Perturber("random", 42).nudge(values)
    other = harness.Perturber("random", 43).nudge(values)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert 0 < int(np.sum(first > values)) < len(values)
    assert harness.seed_for(0, "a", "b") == harness.seed_for(0, "a", "b") != harness.seed_for(1, "a", "b")


def test_ulp_distance():
    assert harness.ulp_distance(1.0, 1.0) == 0
    assert harness.ulp_distance(1.0, math.nextafter(1.0, 2.0)) == 1
    assert harness.ulp_distance(-0.0, 0.0) == 0
    assert harness.ulp_distance(-5e-324, 5e-324) == 2


def test_perturbation_wraps_restores_and_counts_calls():
    plane = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    original = Plane.__dict__["local_uv"]
    reference = plane.local_uv((0.25, 0.5, 0.0))
    with harness.Perturbation(["Plane.local_uv", "LinePath.evaluate"], "+1", 0) as active:
        moved = plane.local_uv((0.25, 0.5, 0.0))
        line = LinePath((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)).evaluate(0.5)
    assert Plane.__dict__["local_uv"] is original
    assert moved == (math.nextafter(reference[0], 1.0), math.nextafter(reference[1], 1.0))
    assert np.all(line == np.nextafter(0.5, 1.0))
    assert active.calls == {"Plane.local_uv": 1, "LinePath.evaluate": 1}
    assert plane.local_uv((0.25, 0.5, 0.0)) == reference


def test_nested_wrapped_calls_are_perturbed_once():
    plane = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    uv = np.asarray([[0.5, 0.25]])
    reference = surfaces._evaluate_surface_many(plane, uv)
    with harness.Perturbation(["_evaluate_surface_many", "Plane.evaluate"], "+1", 0):
        moved = surfaces._evaluate_surface_many(plane, uv)
    finite = reference != 0.0
    assert np.all(moved[finite] == np.nextafter(reference[finite], np.inf))


def test_module_function_patched_in_every_importing_module_and_restored():
    from anygeometry import arrangement_geometry, batch_intersections, material_arrangement
    original = arrangement_geometry.freeze_edge
    with harness.Perturbation(["freeze_edge.arc"], "+1", 0):
        assert batch_intersections.freeze_edge is not original
        assert material_arrangement.freeze_edge is batch_intersections.freeze_edge
    assert batch_intersections.freeze_edge is original
    assert material_arrangement.freeze_edge is original


def test_freeze_edge_arc_definition_is_perturbed_by_one_ulp():
    from anygeometry import GeometryModel, arrangement_geometry
    model = GeometryModel()
    a, b, c = model.add_points(((1.0, 0.0, 0.0), (0.6, 0.8, 0.0), (0.0, 1.0, 0.0)))
    arc = model.add_arc(a, b, c)
    line = model.add_line(a, c)
    reference = arrangement_geometry.freeze_edge(model, arc)
    with harness.Perturbation(["freeze_edge.arc"], "-1", 0) as active:
        moved = arrangement_geometry.freeze_edge(model, arc)
        straight = arrangement_geometry.freeze_edge(model, line)
    assert isinstance(moved, EllipticArc)
    for name in ("center", "u_vector", "v_vector"):
        for before, after in zip(getattr(reference, name), getattr(moved, name)):
            assert after == math.nextafter(before, -math.inf)
    assert straight == arrangement_geometry.freeze_edge(model, line)
    assert active.calls == {"freeze_edge.arc": 1}


def _ulps(a, b, scale):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / (np.spacing(scale))))


@pytest.mark.parametrize("route", harness.ROUTES)
def test_routes_stay_within_a_few_ulps_of_the_default(route):
    rng = np.random.default_rng(5)
    plane = Plane((0.3, -0.2, 1.0), (1.0, 0.1, 0.0), (0.0, 0.9, 0.2))
    cylinder = Cylinder((0.0, 0.0, 0.0), (0.1, 0.2, 1.0), (1.0, 0.0, 0.0), 1.5, 2.0, 0.3, 4.0)
    line = LinePath((0.1, 0.2, 0.3), (1.7, -0.4, 2.2))
    ellipse = EllipticArc((0.5, 0.0, 0.0), (1.0, 0.0, 0.2), (0.0, 0.8, 0.0), 0.2, 2.5)
    uvs = rng.uniform(0.05, 0.95, (40, 2))
    with np.errstate(all="ignore"):
        default = ([plane.local_uv(plane.evaluate(*uv)) for uv in uvs],
                   [cylinder.local_uv(cylinder.evaluate(*uv)) for uv in uvs],
                   [plane.evaluate(*uv) for uv in uvs], [cylinder.evaluate(*uv) for uv in uvs],
                   [line.evaluate(float(t)) for t in uvs[:, 0]], line.evaluate(uvs[:, 0]),
                   [ellipse.evaluate(float(t)) for t in uvs[:, 0]], ellipse.evaluate(uvs[:, 0]))
        with harness.RouteSwitch(route) as switch:
            routed = ([plane.local_uv(plane.evaluate(*uv)) for uv in uvs],
                      [cylinder.local_uv(cylinder.evaluate(*uv)) for uv in uvs],
                      [plane.evaluate(*uv) for uv in uvs], [cylinder.evaluate(*uv) for uv in uvs],
                      [line.evaluate(float(t)) for t in uvs[:, 0]], line.evaluate(uvs[:, 0]),
                      [ellipse.evaluate(float(t)) for t in uvs[:, 0]], ellipse.evaluate(uvs[:, 0]))
    assert switch.calls  # the route was actually used
    for before, after in zip(default, routed):
        assert np.shape(before) == np.shape(after)
        assert _ulps(before, after, 4.0) <= 8.0
    # the routes are genuinely different arithmetic for at least one value
    if route != "numpy":
        assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(default, routed))
    assert Plane.__dict__["local_uv"] is not None and Plane.local_uv.__name__ == "local_uv"


def test_scalar_route_reproduces_the_seam_case():
    """R.0 / review s1c case 2: NumPy snaps u to 0.0, the scalar route does not."""
    import adversarial_corpus as corpus
    seam = corpus.SEAM_CYLINDER
    cylinder = Cylinder(seam["origin"], seam["axis"], seam["radial"], 1, 1, 0, 1)
    assert cylinder.local_uv(corpus.SEAM_POINT)[0] == 0.0
    with harness.RouteSwitch("scalar"):
        u, _ = cylinder.local_uv(corpus.SEAM_POINT)
    assert 0.0 < u < 1e-13


def test_canonical_outcome_is_float_free():
    a = harness.canonical({"kind": "cross", "x": 1.25, "msg": "distance 1.5e-09 exceeds 1e-9",
                           "nested": [0.5, {"b": True}]})
    b = harness.canonical({"kind": "cross", "x": 3.0, "msg": "distance 2.5e-09 exceeds 1e-9",
                           "nested": [0.25, {"b": True}]})
    assert a == b
    assert harness.canonical({"kind": "cross"}) != harness.canonical({"kind": "touch_point"})


def test_report_format_on_a_small_case(tmp_path):
    import adversarial_corpus as corpus
    case = corpus.CASES["exact_tol_vertex/tol"]
    variants = ["none", "outputs/-1", "route/scalar", "reverse"]
    rows = harness.run_case(case, variants=variants, seed=0, op_filter="points[face]", log=lambda text: None)
    assert set(rows) == {"points[face]"}
    op = rows["points[face]"]
    assert set(op["variants"]) == {"none", "outputs/-1", "route/scalar"}
    assert op["variants"]["none"]["flipped"] is False
    assert op["reversal"] is None  # point membership is not a reversible operation
    flipped = op["variants"]["outputs/-1"]
    if flipped["flipped"]:
        assert flipped["reproducer"].startswith("python tools/perturbation_harness.py --case")
        assert all(row["site"] in harness.SITES and row["inventory"] for row in flipped["attribution"])
    report = {"meta": {"anygeometry": "x", "python": "3", "numpy": "2", "seed": 0, "variants": variants,
                       "seconds": 1.0},
              "cases": {case.name: {"group": case.group, "note": case.note, "inventory": [],
                                    "operations": rows}}}
    text = harness.markdown(report)
    assert text.startswith("# Perturbation harness report")
    assert "## Per case" in text and "## Per perturbation site" in text and "## Site coverage" in text
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, default=str), encoding="utf-8")
    merged = harness.merge([path, path])
    assert set(merged["cases"]) == {case.name} and merged["meta"]["seconds"] == 2.0
    for flip in harness.summarize(report):
        assert {"case", "operation", "variant", "sites", "inventory", "reproducer"} <= set(flip)


def test_reversal_is_run_for_reversible_operations():
    import adversarial_corpus as corpus
    case = corpus.CASES["plane_family/plane"]
    rows = harness.run_case(case, variants=["reverse"], seed=0, log=lambda text: None)
    op = next(iter(rows.values()))
    assert op["reversal"] is not None and op["reversal"]["flipped"] is False


def test_every_site_target_exists():
    for site in harness.SITES.values():
        if isinstance(site.owner, type):
            assert site.attribute in site.owner.__dict__, site.name
        else:
            assert callable(getattr(site.owner, site.attribute)), site.name
