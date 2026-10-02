"""Schema policy for Bezier quadric intersection curves: schema 6 only when a document holds one.

The same contract as the angular curves: ordinary documents keep schema 5, older schemas refuse the record,
the record round-trips bit for bit (including affine images and slices), edits and splits keep the exact curve,
and a malformed record is refused whole (the documents below are re-signed, so the refusal comes from the record
and not from the checksum).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from anygeometry import BezierQuadricCurve, EllipticArc, GeometryError, GeometryModel, from_dict, to_dict
from anygeometry.branch_supports import bezier_quadric_support
from anygeometry.extrusions import BezierDirectrix
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))


def _curve(kind="sine", controls=CUBIC):
    """One chart of each parameterization: a narrow pipe gives sine charts, a cone square and linear ones."""
    wall = ExtrudedSurface(BezierDirectrix(controls), (.25, 0., 1.5))
    if kind == "sine":
        second, index = Cylinder((-2., .5, .7), (1., 0., 0.), (0., 1., 0.), .3, 8.), 0
    else:
        second, index = Cone((-2., .5, .8), (1., 0., 0.), (0., 1., 0.), .1, 1., 8.), 0 if kind == "square" else 1
    curve = list(bezier_quadric_support(wall, second).curves)[index]
    assert curve.parameterization == {"sine": "both_sine", "square": "left_square", "linear": "linear"}[kind]
    return curve


def _model_with(curve):
    model = GeometryModel()
    start, end = model.add_points(curve.evaluate(np.asarray((0., 1.))))
    return model, model.add_curve(start, end, curve)


def _resign(document):
    """Re-sign a hand-edited document the way the writer does."""
    body = {k: v for k, v in document.items() if k != "checksum"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    document["checksum"] = {"algorithm": "sha256", "value": hashlib.sha256(encoded).hexdigest()}
    return document


@pytest.mark.parametrize("kind", ["sine", "square", "linear"])
def test_a_document_with_a_bezier_branch_is_schema_six_and_round_trips_exactly(kind):
    curve = _curve(kind)
    model, edge = _model_with(curve)
    document = to_dict(model)
    assert document["version"] == 6
    restored = from_dict(json.loads(json.dumps(document)))
    assert to_dict(restored) == document
    again = restored.edges[edge].curve
    assert isinstance(again, BezierQuadricCurve) and again == curve and hash(again) == hash(curve)
    t = np.linspace(0., 1., 11)
    assert np.array_equal(again.evaluate(t), curve.evaluate(t))
    assert restored.validate_topology() == ()


def test_the_record_names_the_wall_the_quadric_the_chart_and_the_image():
    model, _edge = _model_with(_curve())
    (record,) = [e["curve"] for e in to_dict(model)["edges"]]
    assert set(record) == {"type", "first", "second", "start", "sweep", "branch", "parameterization", "transform"}
    assert record["type"] == "bezier_quadric_intersection"
    assert record["first"] == {"kind": "bezier", "controls": [list(p) for p in CUBIC], "direction": [.25, 0., 1.5]}
    assert record["second"]["kind"] == "cylinder" and record["second"]["radius"] == .3


def test_the_affine_image_and_slices_survive_serialization():
    curve = _curve()
    matrix = np.eye(4)
    matrix[:3, :3] = 2. * np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    matrix[:3, 3] = (1., -2., .5)
    moved = curve.transformed(matrix).subcurve(.2, .8)
    model, edge = _model_with(moved)
    restored = from_dict(json.loads(json.dumps(to_dict(model))))
    assert restored.edges[edge].curve == moved
    t = np.linspace(0., 1., 7)
    assert np.allclose(restored.sample_edge(edge, t), curve.transformed(matrix).subcurve(.2, .8).evaluate(t), atol=1e-14)


def test_splitting_an_edge_keeps_the_exact_curve_on_both_sides():
    curve = _curve("linear")
    model, edge = _model_with(curve)
    _vertex, (left, right) = model.split_edge(edge, .37)
    t = np.linspace(0., 1., 9)
    assert np.allclose(model.sample_edge(left, t), curve.subcurve(0., .37).evaluate(t), atol=1e-13)
    assert np.allclose(model.sample_edge(right, t), curve.subcurve(.37, 1.).evaluate(t), atol=1e-13)
    assert isinstance(model.edges[left].curve, BezierQuadricCurve)
    assert model.validate_topology() == ()
    assert from_dict(json.loads(json.dumps(to_dict(model)))).validate_topology() == ()


def test_reversal_and_endpoint_checks_keep_the_curve_honest():
    curve = _curve()
    model = GeometryModel()
    start, end = model.add_points(curve.evaluate(np.asarray((0., 1.))))
    with pytest.raises(GeometryError, match="endpoints"):
        model.add_curve(end, end, curve)                                         # vertices that disagree with the definition
    edge = model.add_curve(start, end, curve)
    assert model.edge_length(edge) > 0 and model.validate_topology() == ()


def test_ordinary_documents_are_still_schema_five():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)
    model = GeometryModel()
    start, end = model.add_points(arc.evaluate(np.asarray((0., 1.))))
    model.add_curve(start, end, arc)
    assert to_dict(model)["version"] == 5


def test_older_schemas_refuse_the_branch_record():
    model, _edge = _model_with(_curve())
    document = to_dict(model)
    for version in (4, 5):
        stale = deepcopy(document)
        stale["version"] = version
        with pytest.raises(GeometryError, match="require schema 6"):
            from_dict(_resign(stale))


@pytest.mark.parametrize("damage", [
    lambda c: c.pop("first"),
    lambda c: c.pop("branch"),
    lambda c: c.pop("start"),
    lambda c: c.update(extra=1),
    lambda c: c["first"].update(kind="cylinder"),                    # a quadric as the Bezier wall
    lambda c: c["first"].pop("direction"),
    lambda c: c["first"].update(controls=5),
    lambda c: c["first"].update(controls=[[0., 0., 0.], [1., 0., 0.]]),   # a line is not a Bezier wall of this engine
    lambda c: c["first"].update(direction=[0., 0., 0.]),
    lambda c: c["second"].pop("matrix"),
    lambda c: c.update(branch=2),
    lambda c: c.update(branch="plus"),
    lambda c: c.update(parameterization="cubic"),
    lambda c: c.update(sweep=0.),
    lambda c: c.update(sweep=3.),                                    # leaves the parameter interval
    lambda c: c.update(start="north"),
    lambda c: c.update(transform=[[1, 0, 0, 0]] * 4),
    lambda c: c.update(start=.5, sweep=.1),                          # a chart that is not a branch of the pair
])
def test_malformed_branch_records_are_refused_whole(damage):
    model, _edge = _model_with(_curve())
    document = to_dict(model)
    damage(_find_record(document))
    with pytest.raises(GeometryError):
        from_dict(_resign(document))


def _find_record(node):
    if isinstance(node, dict):
        if node.get("type") == "bezier_quadric_intersection":
            return node
        for value in node.values():
            found = _find_record(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_record(value)
            if found is not None:
                return found
    return None
