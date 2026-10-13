"""NURBS definitions (slice A0): validation, evaluation, parameter maps and schema-7 records.

Independent references: Piegl and Tiller's worked basis example (Ex. 2.3), the
exact rational circle, the analytic cylinder patch (radius and height), the
bilinear patch formula, finite differences for the parameter chain rule, and
the released 0.4.5 reader for the version gate.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, to_dict
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface, nurbs_from_record, nurbs_record
from anygeometry.serialization import _NURBS_VERSION, required_version

HALF = math.sqrt(0.5)
CIRCLE_KNOTS = (0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1)
CIRCLE_CONTROLS = ((1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0), (1, -1, 0), (1, 0, 0))
CIRCLE_WEIGHTS = (1, HALF, 1, HALF, 1, HALF, 1, HALF, 1)


def circle(**changes):
    values = dict(degree=2, knots=CIRCLE_KNOTS, controls=CIRCLE_CONTROLS, weights=CIRCLE_WEIGHTS)
    values.update(changes)
    return NurbsCurve(**values)


def cylinder_patch(radius=2.0, height=3.0):
    """Quarter cylinder: rational quadratic arc in u, linear in v (exact)."""
    arc = ((radius, 0.), (radius, radius), (0., radius))
    controls = tuple(tuple((x, y, z) for z in (0.0, height)) for x, y in arc)
    weights = tuple((w, w) for w in (1.0, HALF, 1.0))
    return NurbsSurface(2, 1, (0, 0, 0, 1, 1, 1), (0, 0, 1, 1), controls, weights)


# ----------------------------------------------------------------------------------- curves


def test_textbook_basis_combination():
    # Piegl & Tiller Ex. 2.3: at u = 5/2 the non-zero basis values are 1/8, 6/8, 1/8 on P2, P3, P4.
    knots = (0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5)
    controls = tuple((float(i), float(i * i), 0.0) for i in range(8))
    curve = NurbsCurve(2, knots, controls)
    expected = (np.array(controls[2]) + 6 * np.array(controls[3]) + np.array(controls[4])) / 8
    np.testing.assert_allclose(curve.evaluate(0.5), expected, rtol=0, atol=1e-14)   # t = 0.5 -> u = 2.5


def test_exact_circle_and_parameter_map():
    curve = circle()
    points = curve.evaluate(np.linspace(0, 1, 4001))
    assert np.max(np.abs(np.linalg.norm(points, axis=1) - 1)) <= 4 * np.finfo(float).eps
    part = curve.subcurve(0.25, 0.5)
    np.testing.assert_allclose(part.evaluate([0.0, 1.0]), curve.evaluate([0.25, 0.5]), atol=1e-15)
    reverse = curve.subcurve(0.5, 0.25)
    np.testing.assert_allclose(reverse.evaluate(0.0), curve.evaluate(0.5), atol=1e-15)
    # Chain rule: derivative with respect to t is (domain length) x derivative in the knot parameter.
    t, h = 0.37, 1e-6
    numeric = (part.evaluate(t + h) - part.evaluate(t - h)) / (2 * h)
    np.testing.assert_allclose(part.derivative(t), numeric, rtol=1e-7, atol=1e-9)
    with pytest.raises(GeometryError, match="positive length"):
        curve.subcurve(0.3, 0.3)
    with pytest.raises(GeometryError, match=r"in \[0, 1\]"):
        curve.evaluate(1.5)


def test_bounds_contain_the_curve():
    rng = np.random.default_rng(4)
    controls = tuple(map(tuple, rng.standard_normal((9, 3))))
    knots = (0, 0, 0, 0, .2, .4, .5, .7, .9, 1, 1, 1, 1)
    curve = NurbsCurve(3, knots, controls, tuple(rng.uniform(.5, 2, 9)))
    for lower, upper in ((0., 1.), (.3, .45), (.61, .62)):
        low, high = curve.bounds(lower, upper)
        points = curve.evaluate(np.linspace(lower, upper, 2000))
        assert np.all(points >= low) and np.all(points <= high)


def test_transform_commutes_with_evaluation():
    curve = circle()
    angle = 0.7
    matrix = np.array([[math.cos(angle), -math.sin(angle), 0, 1.5], [math.sin(angle), math.cos(angle), 0, -2],
                       [0, 0, 2, 0.25], [0, 0, 0, 1]])
    t = np.linspace(0, 1, 101)
    moved = curve.transformed(matrix).evaluate(t)
    expected = curve.evaluate(t) @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_allclose(moved, expected, rtol=0, atol=1e-13)
    pcurve = NurbsCurve(1, (0, 0, 1, 1), ((0, 0), (1, 1)))
    with pytest.raises(GeometryError, match="only 3D"):
        pcurve.transformed(matrix)


@pytest.mark.parametrize("changes,message", [
    (dict(degree=0), "integer from 1 to 7"),
    (dict(degree=8), "integer from 1 to 7"),
    (dict(degree=True), "integer from 1 to 7"),
    (dict(knots=(0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1, 1)), "clamped"),
    (dict(knots=(0, 0, 0, .25, .25, .25, .5, .75, .75, 1, 1, 1)), "multiplicity"),
    (dict(weights=(1, -1, 1, HALF, 1, HALF, 1, HALF, 1)), "positive weight"),
    (dict(weights=(1, 1e-7, 1, HALF, 1, HALF, 1, HALF, 1)), "1e6"),
    (dict(controls=CIRCLE_CONTROLS[:-1] + ((1, 0),)), "dimension 2 or 3"),
    (dict(controls=CIRCLE_CONTROLS[:-1] + ((1e-160, 0, 0),)), r"2\*\*-500"),
    (dict(controls=CIRCLE_CONTROLS[:-1] + ((math.nan, 0, 0),)), "finite"),
    (dict(knots=tuple(k * 2.0 ** 101 for k in CIRCLE_KNOTS)), r"2\*\*100"),
    (dict(knots=(0, 0, 0, .25, .25, .5, .5, .75, .75 + 1e-15, 1, 1, 1)), "64 eps"),
    (dict(domain=(0.5, 1.5)), "inside"),
    (dict(domain=(0.5, 0.5)), "distinct"),
])
def test_invalid_curves_are_refused(changes, message):
    with pytest.raises(GeometryError, match=message):
        circle(**changes)


# ----------------------------------------------------------------------------------- surfaces


def test_bilinear_patch_is_exact():
    corners = (((0., 0., 0.), (0., 2., 1.)), ((3., 0., 0.5), (3., 2., -1.)))
    surface = NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1), corners)
    rng = np.random.default_rng(1)
    uv = rng.uniform(0, 1, (500, 2))
    a, b, c, d = (np.array(p) for p in (corners[0][0], corners[0][1], corners[1][0], corners[1][1]))
    u, v = uv[:, :1], uv[:, 1:]
    expected = (1 - u) * (1 - v) * a + (1 - u) * v * b + u * (1 - v) * c + u * v * d
    np.testing.assert_allclose(surface.evaluate_many(uv), expected, rtol=0, atol=1e-14)


def test_rational_cylinder_patch_is_exact_to_rounding():
    surface = cylinder_patch(radius=2.0, height=3.0)
    rng = np.random.default_rng(2)
    uv = rng.uniform(0, 1, (5000, 2))
    points = surface.evaluate_many(uv)
    radius = np.hypot(points[:, 0], points[:, 1])
    assert np.max(np.abs(radius - 2.0)) <= 8 * np.finfo(float).eps
    np.testing.assert_allclose(points[:, 2], 3.0 * uv[:, 1], rtol=0, atol=1e-14)
    assert np.all(points[:, 0] >= -1e-15) and np.all(points[:, 1] >= -1e-15)   # the first quadrant
    np.testing.assert_allclose(surface.evaluate(1.0, 1.0), [0.0, 2.0, 3.0], atol=1e-15)


def test_surface_chart_sub_rectangle_and_transform():
    surface = cylinder_patch()
    part = NurbsSurface(*[getattr(surface, name) for name in
                          ("degree_u", "degree_v", "knots_u", "knots_v", "controls", "weights")],
                        u_range=(0.25, 0.75), v_range=(0.5, 1.0))
    np.testing.assert_allclose(part.evaluate(0.0, 0.0), surface.evaluate(0.25, 0.5), atol=1e-15)
    assert part.support_key() == surface.support_key()
    matrix = np.diag([1., 1., 1., 1.]); matrix[:3, 3] = (1., 2., 3.)
    np.testing.assert_allclose(surface.transformed(matrix).evaluate(0.4, 0.6), surface.evaluate(0.4, 0.6) + (1, 2, 3),
                               atol=1e-14)


@pytest.mark.parametrize("controls,weights,message", [
    ((((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.),)), None, "full grid"),
    ((((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))), ((1., 1.),), "match the control grid"),
    ((((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))), ((1., 1.), (1., 2e6)), "1e6"),
])
def test_invalid_surfaces_are_refused(controls, weights, message):
    with pytest.raises(GeometryError, match=message):
        NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1), controls, weights)


# ----------------------------------------------------------------------------------- curves on surfaces


def test_curve_on_surface_is_the_composition():
    surface = cylinder_patch()
    iso = CurveOnSurface(surface, NurbsCurve(1, (0, 0, 1, 1), ((0.3, 0.0), (0.3, 1.0))))
    t = np.linspace(0, 1, 11)
    np.testing.assert_allclose(iso.evaluate(t), surface.evaluate_many(np.column_stack((np.full(11, .3), t))),
                               atol=1e-15)
    assert np.max(np.abs(np.hypot(*iso.evaluate(t)[:, :2].T) - 2.0)) <= 8 * np.finfo(float).eps
    with pytest.raises(GeometryError, match="inside the chart"):
        CurveOnSurface(surface, NurbsCurve(1, (0, 0, 1, 1), ((0.3, -0.1), (0.3, 1.0))))
    with pytest.raises(GeometryError, match="also_on"):
        CurveOnSurface(surface, iso.pcurve, ((surface, iso.pcurve),))
    with pytest.raises(GeometryError, match="2D NurbsCurve"):
        CurveOnSurface(surface, circle())


# ----------------------------------------------------------------------------------- records


def definitions():
    surface = cylinder_patch()
    pcurve = NurbsCurve(2, (0, 0, 0, 1, 1, 1), ((0.1, 0.2), (0.5, 0.9), (0.8, 0.3)), (1.0, 2.0, 1.0))
    return (circle(), circle().subcurve(0.8, 0.1), pcurve, surface,
            CurveOnSurface(surface, pcurve, (), ((1, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))))


@pytest.mark.parametrize("index", range(5))
def test_records_round_trip_exactly(index):
    value = definitions()[index]
    record = nurbs_record(value)
    text = json.dumps(record, sort_keys=True, allow_nan=False)
    back = nurbs_from_record(json.loads(text))
    assert back == value
    assert json.dumps(nurbs_record(back), sort_keys=True) == text
    assert required_version(record) == _NURBS_VERSION == 7


def test_records_are_strict():
    record = nurbs_record(circle())
    with pytest.raises(GeometryError, match="unexpected"):
        nurbs_from_record({**record, "extra": 1})
    with pytest.raises(GeometryError, match="missing"):
        nurbs_from_record({key: value for key, value in record.items() if key != "domain"})
    with pytest.raises(GeometryError, match="dimension"):
        nurbs_from_record({**record, "type": "nurbs_pcurve"})
    with pytest.raises(GeometryError, match="unsupported NURBS record"):
        nurbs_from_record({**record, "type": "nurbs_volume"})
    with pytest.raises(GeometryError, match="real number"):
        nurbs_from_record({**record, "knots": [True] + record["knots"][1:]})


def test_required_version_matches_the_writer_for_existing_documents():
    plain = GeometryModel()
    plain.add_plate(plain.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    from test_serialization_quadric import _curve, _model_with
    quadric, _ = _model_with(_curve())
    for model, expected in ((plain, 5), (quadric, 6)):
        document = to_dict(model)
        assert document["version"] == expected
        assert required_version(document) == expected
    nested = {"type": "edge", "curve": {"type": "curve_on_surface", "support": {"type": "nurbs_surface"}}}
    assert required_version([{"type": "plane"}, nested]) == 7


def test_the_released_reader_refuses_schema_7(tmp_path):
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "release.tar"
    result = subprocess.run(["git", "-C", str(root), "archive", "-o", str(archive), "v0.4.5", "src"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"release tag v0.4.5 unavailable: {result.stderr.strip()}")
    with tarfile.open(archive) as bundle:
        bundle.extractall(tmp_path / "release", filter="data")
    model = GeometryModel()
    model.add_plate(model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    from test_serialization_quadric import _resign
    document = dict(to_dict(model))
    document["version"] = 7
    _resign(document)                      # a correctly signed schema-7 document: only the version is new
    path = tmp_path / "document.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    script = ("import json, sys; from anygeometry import from_dict\n"
              "try:\n    from_dict(json.load(open(sys.argv[1])))\nexcept Exception as error:\n"
              "    print(type(error).__name__, error); sys.exit(3)\nprint('accepted')")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "release" / "src"))
    run = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, env=env,
                         timeout=120)
    assert run.returncode == 3, run.stdout + run.stderr
    assert "unsupported ANYgeometry version 7" in run.stdout


# ----------------------------------------------------------------------------------- review regressions (A0 round 1)


def test_bounds_hold_for_tiny_equal_weights():
    curve = NurbsCurve(2, (0, 0, 0, .5, 1, 1, 1), ((0, 0, 0), (1e-100, 0, 0), (1e-100, 0, 0), (0, 0, 0)),
                       (1e-300,) * 4)
    low, high = curve.bounds()
    point = curve.evaluate(.5)
    assert np.all(point >= low) and np.all(point <= high)


def test_tiny_knot_ranges_are_refused_and_derivatives_scale_per_factor():
    with pytest.raises(GeometryError, match="knot range"):
        NurbsCurve(1, (0, 0, 2.0 ** -540, 2.0 ** -540), ((0, 0, 0), (2.0 ** -100, 0, 0)), (1, 2))
    # Engine-parameter derivatives of a tiny subcurve: x(t) = 2 C t/(1+t) scaled; compare with the
    # knot-parameter derivatives times the domain length per order.
    curve = NurbsCurve(1, (0, 0, 1, 1), ((0, 0, 0), (2.0 ** -100, 0, 0)), (1, 2))
    part = curve.subcurve(0.0, 2.0 ** -200)
    values = part.derivatives(0.5, 2)
    exact = curve.derivatives(2.0 ** -201, 2)
    scale = 2.0 ** -200
    np.testing.assert_allclose(values[1], exact[1] * scale, rtol=1e-14)
    np.testing.assert_allclose(values[2], exact[2] * scale * scale, rtol=1e-14)


def test_domain_end_maps_exactly():
    curve = NurbsCurve(1, (-1, -1, .1, .1), ((0, 0, 0), (1, 0, 0)))
    np.testing.assert_array_equal(curve.evaluate(1.0), [1, 0, 0])
    np.testing.assert_array_equal(curve.subcurve(0, 1).evaluate([0.0, 1.0]), [[0, 0, 0], [1, 0, 0]])
    low, high = curve.bounds(1.0, 1.0)
    assert low[0] <= 1.0 <= high[0]


def test_curve_on_surface_refuses_overflow_after_transform():
    corners = (((1e150, 0., 0.), (1e150, 1., 0.)), ((1e150, 0., 1.), (1e150, 1., 1.)))
    surface = NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1), corners)
    pcurve = NurbsCurve(1, (0, 0, 1, 1), ((0., 0.), (1., 1.)))
    stretched = CurveOnSurface(surface, pcurve, (), ((1e200, 0, 0, 0), (0, 1e200, 0, 0), (0, 0, 1e200, 0), (0, 0, 0, 1)))
    with pytest.raises(GeometryError, match="not representable"):
        stretched.evaluate(0.5)


@pytest.mark.parametrize("arguments", [
    dict(controls=(0, 0)),
    dict(weights=1),
    dict(weights=(None, (1, 1))),
])
def test_malformed_surface_inputs_raise_typed_errors(arguments):
    values = dict(degree_u=1, degree_v=1, knots_u=(0, 0, 1, 1), knots_v=(0, 0, 1, 1),
                  controls=(((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    values.update(arguments)
    with pytest.raises(GeometryError):
        NurbsSurface(**values)


def test_huge_integer_coordinates_raise_typed_errors():
    with pytest.raises(GeometryError, match="finite"):
        NurbsCurve(1, (0, 0, 1, 1), ((10 ** 400, 0, 0), (1, 0, 0)))


def test_transform_records_are_numerically_strict_and_rank_based():
    surface = cylinder_patch()
    pcurve = NurbsCurve(1, (0, 0, 1, 1), ((0.3, 0.0), (0.3, 1.0)))
    record = nurbs_record(CurveOnSurface(surface, pcurve))
    for bad in (True, "1"):
        damaged = json.loads(json.dumps(record))
        damaged["transform"][0][0] = bad
        with pytest.raises(GeometryError, match="real number"):
            nurbs_from_record(damaged)
    tiny = ((1e-110, 0, 0, 0), (0, 1e-110, 0, 0), (0, 0, 1e-110, 0), (0, 0, 0, 1))
    CurveOnSurface(surface, pcurve, (), tiny)     # invertible although its determinant underflows


def test_pcurve_chart_rule_is_the_documented_sufficient_condition():
    surface = cylinder_patch()
    inside_but_polygon_outside = NurbsCurve(2, (0, 0, 0, 1, 1, 1), ((0., 0.), (2., 0.), (0., 0.)))
    with pytest.raises(GeometryError, match="control polygon"):
        CurveOnSurface(surface, inside_but_polygon_outside)


# ----------------------------------------------------------------------------------- review regressions (A0 round 2)


def test_weight_ratio_is_decided_exactly_at_subnormal_magnitudes():
    tiny = 2.0 ** -1074
    with pytest.raises(GeometryError, match="1e6"):
        NurbsCurve(1, (0, 0, 1, 1), ((0, 0, 0), (1, 0, 0)), (tiny, 1499999 * tiny))
    NurbsCurve(1, (0, 0, 1, 1), ((0, 0, 0), (1, 0, 0)), (tiny, 1000000 * tiny))     # exactly 1e6 is allowed
    with pytest.raises(GeometryError, match="1e6"):
        NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1), (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))),
                     ((tiny, tiny), (tiny, 1499999 * tiny)))


def test_nonsingular_transforms_are_accepted_exactly():
    curve = NurbsCurve(1, (0, 0, 1, 1), ((0, 0, 1), (1, 0, 1)))
    moved = curve.transformed(np.diag((1., 1., 2.0 ** -100, 1.)))
    np.testing.assert_array_equal(moved.evaluate(0.5), [0.5, 0.0, 2.0 ** -100])
    with pytest.raises(GeometryError, match="nonsingular"):
        curve.transformed(np.diag((1., 1., 0., 1.)))
    with pytest.raises(GeometryError, match="nonsingular"):
        curve.transformed(((1, 2, 3, 0), (2, 4, 6, 0), (0, 0, 1, 0), (0, 0, 0, 1)))


@pytest.mark.parametrize("call", [
    lambda c: c.evaluate(10 ** 400),
    lambda c: c.evaluate([[0], [0, 1]]),
    lambda c: c.evaluate("0.5"),
    lambda c: c.derivatives(.5, 1.0),
    lambda c: c.derivatives(.5, True),
])
def test_malformed_parameters_raise_typed_errors(call):
    with pytest.raises(GeometryError):
        call(circle())


def test_zero_dimensional_controls_raise_typed_errors():
    with pytest.raises(GeometryError, match="sequence"):
        NurbsCurve(1, (0, 0, 1, 1), np.array(1.0))
