"""Schema policy for quadric intersection curves: schema 6 only when a document holds one.

Ordinary documents keep schema 5 and byte-identical checksums; a schema-5 reader never meets a record it
cannot read; older schemas refuse the new record; malformed records are refused whole.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryError, GeometryModel, from_dict, to_dict
from anygeometry.quadric_curves import QuadricIntersectionCurve
from anygeometry.quadric_supports import cone_support
from anygeometry.surfaces import Cone, Cylinder


def _curve(branch_index=0):
    cone = Cone((0., 0., 0.), (math.cos(math.radians(10)), 0., math.sin(math.radians(10))), (0., 1., 0.), .5, 1., 5., 0., math.tau)
    cylinder = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau)
    charts = [c for c in cone_support(cone, cylinder).curves if isinstance(c, QuadricIntersectionCurve)]
    return charts[branch_index]


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


def test_a_document_with_a_quadric_curve_is_schema_six_and_round_trips_exactly():
    curve = _curve()
    model, edge = _model_with(curve)
    document = to_dict(model)
    assert document["version"] == 6
    restored = from_dict(json.loads(json.dumps(document)))
    assert to_dict(restored) == document
    again = restored.edges[edge].curve
    assert isinstance(again, QuadricIntersectionCurve) and again == curve
    t = np.linspace(0., 1., 11)
    assert np.array_equal(again.evaluate(t), curve.evaluate(t))
    assert restored.validate_topology() == ()


def test_the_affine_image_and_subcurves_survive_serialization():
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
    curve = _curve()
    model, edge = _model_with(curve)
    _vertex, (left, right) = model.split_edge(edge, .37)
    t = np.linspace(0., 1., 9)
    assert np.allclose(model.sample_edge(left, t), curve.evaluate(.37 * t), atol=1e-13)
    assert np.allclose(model.sample_edge(right, t), curve.evaluate(.37 + .63 * t), atol=1e-13)
    assert isinstance(model.edges[left].curve, QuadricIntersectionCurve)
    assert model.validate_topology() == ()
    assert from_dict(json.loads(json.dumps(to_dict(model)))).validate_topology() == ()


def test_ordinary_documents_are_still_schema_five():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)
    model, _edge = _model_with(arc)
    assert to_dict(model)["version"] == 5


def test_older_schemas_refuse_the_quadric_record():
    model, _edge = _model_with(_curve())
    document = to_dict(model)
    for version in (4, 5):
        stale = deepcopy(document)
        stale["version"] = version
        with pytest.raises(GeometryError, match="require schema 6"):
            from_dict(_resign(stale))


def test_schema_six_is_a_superset_of_schema_five():
    model = GeometryModel()
    model.add_plate(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))))
    document = to_dict(model)
    document["version"] = 6
    restored = from_dict(_resign(document))
    assert restored.model_id == model.model_id and restored.revision == model.revision
    assert to_dict(restored)["version"] == 5                    # written again at the lowest version that expresses it


@pytest.mark.parametrize("damage", [
    lambda c: c.pop("first"),
    lambda c: c.pop("branch"),
    lambda c: c.update(extra=1),
    lambda c: c["first"].update(kind="plane"),
    lambda c: c["first"].update(kind="cylinder"),                    # slope of a cone but called a cylinder
    lambda c: c["first"].pop("radial_direction"),
    lambda c: c["second"].pop("matrix"),
    lambda c: c.update(branch=2),
    lambda c: c.update(parameterization="cubic"),
    lambda c: c.update(sweep_angle=0.),
    lambda c: c.update(start_angle="north"),
    lambda c: c.update(transform=[[1, 0, 0, 0]] * 4),
])
def test_malformed_quadric_records_are_refused_whole(damage):
    model, _edge = _model_with(_curve())
    document = to_dict(model)
    damage(_find_record(document))
    with pytest.raises(GeometryError):
        from_dict(_resign(document))


def _find_record(node):
    if isinstance(node, dict):
        if node.get("type") == "quadric_intersection":
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
