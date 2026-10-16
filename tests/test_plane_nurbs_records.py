"""Schema-7 ``plane_nurbs_intersection`` records (generality slice B1-P3).

The record holds the definition (plane, surface, path, domain) and the certified
branch (path roots and arc boxes as exact rationals). Loading recomputes the
section and compares; tampered branches, vertices or spellings are refused.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import sys

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
    assert branch["algorithm"] == "plane_nurbs_exact_v2"
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


# ------------------------------------------------------------------ review B1-P3 round 1


@pytest.mark.parametrize("edit, message", [
    (lambda c: c.__setitem__("domain", [10 ** 400, 1.]), "domain"),
    (lambda c: c["plane"].__setitem__("origin", [10 ** 400, 0., 0.]), "plane"),
    (lambda c: c.__setitem__("surface", {"type": "curve_on_surface", "support": {}, "pcurve": {},
                                         "also_on": [], "transform": []}), "nurbs_surface record"),
])
def test_malformed_numbers_and_nesting_are_typed_refusals(stored, edit, message):
    _model, _edge, document = stored
    with pytest.raises(GeometryError, match=message):
        from_dict(tampered(document, edit))


def test_deeply_nested_surfaces_are_refused_before_decoding(stored):
    _model, _edge, document = stored
    nested = {"type": "nurbs_pcurve"}
    for _ in range(990):
        nested = {"type": "curve_on_surface", "support": nested, "pcurve": {}, "also_on": [], "transform": []}
    # the seal of this input is computed with room on the stack: on Python 3.11 the JSON encoder
    # shares the interpreter's recursion limit with the test's own frames (3.12+: a separate C limit)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 5000))
    try:
        forged = tampered(document, lambda c: c.__setitem__("surface", nested))
    finally:
        sys.setrecursionlimit(limit)
    with pytest.raises(GeometryError, match="nurbs_surface record"):
        from_dict(forged)


def test_documents_nested_beyond_json_are_refused(tmp_path):
    nested = {}
    for _ in range(200_000):
        nested = {"next": nested}
    with pytest.raises(GeometryError, match="nested too deeply"):
        _checksum({"payload": nested})
    path = tmp_path / "deep.json"
    path.write_text('{"schema": "anygeometry", "x": ' + "[" * 200_000 + "]" * 200_000 + "}", encoding="utf-8")
    with pytest.raises(GeometryError, match="nested too deeply"):
        read_geometry(path)


def test_integer_domains_round_trip_byte_identically():
    curve = PlaneNurbsCurve(section().plane_record, section().surface, 0, (0, 1))
    assert curve.domain == (0., 1.) and all(type(value) is float for value in curve.domain)
    model, _edge = edge_model(curve)
    document = to_dict(model)
    assert to_dict(from_dict(document)) == document


def test_the_standalone_decoder_checks_its_type():
    record = plane_nurbs_record(section())
    with pytest.raises(GeometryError, match="not a plane_nurbs_intersection"):
        plane_nurbs_from_record(dict(record, type="bogus"))


def test_huge_patch_grids_are_refused_before_extraction():
    import time
    import numpy as np
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
    count = 150
    knots = (0., *np.linspace(0., 1., count), 1.)
    controls = tuple(tuple((i / (count - 1), j / (count - 1), j / (count - 1) - .5) for j in range(count))
                     for i in range(count))
    surface = NurbsSurface(1, 1, knots, knots, controls)
    started = time.perf_counter()
    with pytest.raises(PlaneNurbsNotCertified, match="sub-patches exceed"):
        plane_nurbs_section(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), surface)
    assert time.perf_counter() - started < 5.


def test_loading_recomputes_on_a_cold_cache(stored):
    from anygeometry import support_curves
    model, edge, document = stored
    support_curves._SECTIONS.clear()
    assert from_dict(document).edges[edge].curve == model.edges[edge].curve
    support_curves._SECTIONS.clear()
    with pytest.raises(GeometryError, match="root 1 differs"):
        from_dict(tampered(document, lambda c: c["branch"]["roots"][1].__setitem__(2, "1/3")))


def test_transformed_curves_round_trip():
    import numpy as np
    matrix = np.eye(4)
    c, s = math.cos(.4), math.sin(.4)
    matrix[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 2.))
    matrix[:3, 3] = (5., -1., 3.)
    curve = section().transformed(matrix)
    model, edge = edge_model(curve)
    document = to_dict(model)
    restored = from_dict(document)
    assert restored.edges[edge].curve == curve and to_dict(restored) == document


def test_missing_end_vertices_are_refused(stored):
    _model, _edge, document = stored
    document = deepcopy(document)
    document["edges"][0]["end"] = 99
    with pytest.raises(GeometryError):
        from_dict(resealed(document))


# ------------------------------------------------------------------ review B1-P3 round 2


def test_domains_outside_the_unit_interval_are_refused_at_construction():
    curve = section()
    for domain in ((-1, 2), (.5, .5), (0., float("nan")), (0., float("inf"))):
        with pytest.raises(GeometryError, match="domain"):
            PlaneNurbsCurve(curve.plane_record, curve.surface, 0, domain)


def test_large_inactive_control_grids_are_refused_before_expansion(monkeypatch):
    import numpy as np
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
    count = 300
    knots = (0., *np.linspace(0., 1., count), 1.)
    controls = tuple(tuple((i / (count - 1), j / (count - 1), j / (count - 1) - .5) for j in range(count))
                     for i in range(count))
    surface = NurbsSurface(1, 1, knots, knots, controls, u_range=(knots[150], knots[151]),
                           v_range=(knots[150], knots[151]))

    def expanded(_self):
        raise AssertionError("the control grid was expanded before the budget check")

    monkeypatch.setattr(NurbsSurface, "exact_net", expanded)
    with pytest.raises(PlaneNurbsNotCertified, match="extraction budget"):
        plane_nurbs_section(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), surface)
