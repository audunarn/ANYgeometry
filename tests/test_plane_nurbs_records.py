"""Schema-7 ``plane_nurbs_intersection`` records (generality slice B1-P3).

The record holds the definition (plane, surface, path, domain) and the certified
branch (path roots and arc boxes as exact rationals). Loading recomputes the
section and compares; tampered branches, vertices or spellings are refused.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from anygeometry import Cylinder, GeometryError, GeometryModel, Plane, from_dict, to_dict
from anygeometry.nurbs import cylinder_to_nurbs
from anygeometry.serialization import _NURBS_VERSION, _checksum, read_geometry, required_version, write_geometry
from anygeometry.support_curves import PlaneNurbsCurve, plane_nurbs_from_record, plane_nurbs_record

OBLIQUE = Plane((0., 0., 1.5), (1., 0., .3), (0., 1., .2))


def section(domain=(0., 1.)):
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., math.pi))
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, surface, 0)
    return curve if domain == (0., 1.) else curve.subcurve(*domain)


def edge_model(curve):
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    return model, model.add_curve(start, end, curve)


def resealed(document):
    document = dict(document)
    document["checksum"] = _checksum(document)
    return document


def tampered(document, edit):
    document = deepcopy(document)
    edit(document["edges"][0]["curve"])
    return resealed(document)


@pytest.fixture(scope="module")
def stored():
    model, edge = edge_model(section())
    return model, edge, to_dict(model)


def test_round_trip_is_identical(stored, tmp_path):
    model, edge, document = stored
    assert document["version"] == _NURBS_VERSION == 7
    record = document["edges"][0]["curve"]
    assert record["type"] == "plane_nurbs_intersection" and required_version(record) == 7
    restored = from_dict(document)
    assert restored.edges[edge].curve == model.edges[edge].curve
    assert json.dumps(to_dict(restored), sort_keys=True) == json.dumps(document, sort_keys=True)
    first, second = tmp_path / "a.anygeometry", tmp_path / "b.anygeometry"
    write_geometry(first, model)
    write_geometry(second, read_geometry(first))
    assert first.read_bytes() == second.read_bytes()


def test_sub_curves_keep_their_domain():
    curve = section((.2, .7))
    model, edge = edge_model(curve)
    restored = from_dict(to_dict(model)).edges[edge].curve
    assert restored == curve and restored.domain == curve.domain
    assert (restored.evaluate(.5) == curve.evaluate(.5)).all()


def test_records_hold_exact_canonical_rationals():
    record = plane_nurbs_record(section())
    branch = record["branch"]
    assert branch["algorithm"] == "plane_nurbs_exact_v1"
    assert len(branch["roots"]) == len(branch["arcs"]) + 1
    for root in branch["roots"]:
        assert root[0] in (0, 1) and all(isinstance(value, str) for value in root[1:])
    assert plane_nurbs_from_record(json.loads(json.dumps(record))) == section()


@pytest.mark.parametrize("edit, message", [
    (lambda c: c["branch"]["roots"][1].__setitem__(2, "1/3"), "root 1 differs"),
    (lambda c: c["branch"]["arcs"][0].__setitem__(0, c["branch"]["arcs"][0][0] + 1), "arc 0 differs"),
    (lambda c: c["branch"]["arcs"][0].__setitem__(2, 1 - c["branch"]["arcs"][0][2]), "arc 0 differs"),
    (lambda c: c["branch"]["roots"].pop(), "roots, the recomputed section"),
    (lambda c: c["branch"].__setitem__("algorithm", "plane_nurbs_v0"), "algorithm"),
    (lambda c: c["branch"]["roots"][0].__setitem__(1, "2/4"), "canonical"),
    (lambda c: c["branch"]["roots"][0].__setitem__(1, "-0"), "canonical"),
    (lambda c: c["branch"]["roots"][0].__setitem__(1, "1/0"), "zero denominator"),
    (lambda c: c["branch"]["roots"][0].__setitem__(1, 0.5), "rational text"),
    (lambda c: c["branch"]["roots"][0].__setitem__(0, 0.0), r"\[axis, value, lo, hi\]"),
    (lambda c: c["branch"].__setitem__("boxes", []), "exactly the fields"),
    (lambda c: c.__setitem__("path", 1), "no such section path"),
    (lambda c: c.__setitem__("path", True), "path"),
    (lambda c: c.__setitem__("domain", [.5, .5]), "domain"),
    (lambda c: c.__setitem__("domain", [0., 1.5]), "domain"),
    (lambda c: c["plane"].__setitem__("type", "cylinder"), "plane record"),
])
def test_tampered_records_are_refused_with_their_index(stored, edit, message):
    _model, _edge, document = stored
    with pytest.raises(GeometryError, match=message):
        from_dict(tampered(document, edit))


def test_a_moved_plane_is_refused_by_the_recomputation_or_the_vertices(stored):
    _model, _edge, document = stored
    moved = tampered(document, lambda c: c["plane"].__setitem__("origin", [0., 0., 1.6]))
    with pytest.raises(GeometryError):
        from_dict(moved)


def test_vertices_must_match_the_section_roots(stored):
    _model, _edge, document = stored
    document = deepcopy(document)
    document["vertices"][0]["position"][0] += 1e-3
    with pytest.raises(GeometryError, match="ends do not match"):
        from_dict(resealed(document))


def test_older_schema_readers_refuse(stored):
    _model, _edge, document = stored
    with pytest.raises(GeometryError, match="7"):
        from_dict(resealed(dict(document, version=6)))
