"""K6: the new floating tripod's arm-2 column/transition mitre (face 153 / face 203), an F550-class contact.

Fixture ``data/k6_tripod_face153.json`` (captured from ANYfem's tripod example with ANYgeometry 39e2e2e, R 6.0,
column radius 46 m, draft 16 m, zc 14 m, 48 webs): the arrangement inputs of face 153 (the column skin, r = 6) with
main's pair traces, and the domain of face 203 (the transition skin, r = 6). The two equal-radius skins touch by
design at the double points of their mitre, through which their shared stored edge 505 (an ellipse) passes. The
rotation by 120 degrees leaves the stored doubles with fold discriminants of -3.4 and +3.7 units of
|a1 x a2| ulp(r2), inside the rigorous bound of the float fold coefficients (about 8.8 units): main splits one fold,
misses the other and refuses with an unresolved bridge; the K6 policy charts both as double contacts because edge
505 passes through their contact points.
"""
from __future__ import annotations

from dataclasses import replace
import json
import pathlib

import numpy as np
import pytest

from anygeometry.arrangement_geometry import LinePath, point_parameters
from anygeometry.batch_intersections import UNDECIDABLE_CONTACT, _pair_paths_exact, _undecidable_contact_points
from anygeometry.errors import GeometryError
from anygeometry.exact_curves import CylinderIntersectionCurve, EllipticArc, _CylinderSupport, _double_nodes, \
    _fold_undecidable
from anygeometry.material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain, arrange_material
from anygeometry.surfaces import Cylinder

DATA = pathlib.Path(__file__).resolve().parent / "data" / "k6_tripod_face153.json"
BRIDGE = "arrangement contains an unresolved bridge"
SHARED = 505


def _cylinder(record):
    return {key: record[key] for key in ("origin", "axis", "radial_direction", "radius", "height",
                                         "start_angle", "sweep_angle")}


def _curve(record):
    kind = record["type"]
    if kind == "line":
        return LinePath(tuple(record["start"]), tuple(record["end"]))
    if kind == "elliptic_arc":
        return EllipticArc(tuple(record["center"]), tuple(record["u_vector"]), tuple(record["v_vector"]),
                           record["start_angle"], record["sweep_angle"])
    assert kind == "cylinder_intersection", kind
    first, second = (_CylinderSupport(**{key: tuple(value) if isinstance(value, list) else value
                                         for key, value in _cylinder(record[name]).items()})
                     for name in ("first", "second"))
    return CylinderIntersectionCurve(first, second, record["start_angle"], record["sweep_angle"], record["branch"],
                                     record["parameterization"], tuple(tuple(row) for row in record["transform"]))


def _path(record):
    return ArrangementPath(_curve(record["curve"]), record["source_edge"], tuple(record["owners"]),
                           record["decomposition"], tuple(record["member_ids"]))


def _domain(record, relabel=False):
    loops = []
    for loop in record["boundaries"]:
        paths = [_path(item) for item in loop]
        if relabel:
            paths = [replace(path, source_edge=-path.source_edge) if path.source_edge == SHARED else path
                     for path in paths]
        loops.append(tuple(paths))
    return MaterialDomain(record["face_id"], Cylinder(**_cylinder(record["support"])), tuple(loops))


def _record():
    return json.loads(DATA.read_text())


def _arrange(traces, policy=None):
    record = _record()
    points = tuple(ArrangementPoint(tuple(item["position"]), tuple(item["member_ids"])) for item in record["points"])
    return arrange_material(_domain(record["face"]), traces, points=points, tolerance=record["tolerance"],
                            area_tolerance=record["area_tolerance"], seam_policy=policy)


def _stored_traces():
    return tuple(_path(item) for item in _record()["traces"])


def _production_traces(relabel=False):
    """Face 153's traces with the 153/203 pair recomputed by the production pair stage."""
    record = _record()
    first, second = _domain(record["face"]), _domain(record["other"], relabel)
    paths = _pair_paths_exact(first, second, None, record["pair_tolerance"], lambda: None, point_contacts=[],
                              face_tolerances=tuple(record["face_tolerances"]))
    kept = [trace for trace in _stored_traces() if set(trace.owners) != {153, 203}]
    return tuple(kept)+tuple(ArrangementPath(path, owners=(153, 203)) for path in paths)


def _supports():
    record = _record()
    return (_CylinderSupport.from_surface(Cylinder(**_cylinder(record["face"]["support"]))),
            _CylinderSupport.from_surface(Cylinder(**_cylinder(record["other"]["support"]))))


@pytest.mark.parametrize("policy", [None, "ring_split"])
def test_main_pair_traces_refuse_with_an_unresolved_bridge(policy):
    with pytest.raises(GeometryError, match=BRIDGE):
        _arrange(_stored_traces(), policy)


def test_both_mitre_folds_are_certified_undecidable_and_edge_505_passes_through_their_contacts():
    first, second = _supports()
    assert _fold_undecidable(first, second) == {-1: -1, 1: 1}
    curve = next(trace.curve for trace in _stored_traces() if set(trace.owners) == {153, 203})
    assert [(node.sign, node.exact) for node in _double_nodes(curve)] == [(-1, False), (1, False)]
    record = _record()
    face, other = _domain(record["face"]), _domain(record["other"])
    edge = next(path.curve for loop in face.boundaries for path in loop if path.source_edge == SHARED)
    points = _undecidable_contact_points(face.support, other.support)
    tolerance = record["tolerance"]
    inside = [all(domain.contains(LinePath(tuple(point), tuple(point)), 0., tolerance) for domain in (face, other))
              for point in points]
    # One double point of the mitre is in both faces, on the shared edge 505 (at its middle); the other (the far
    # side of the column) is in neither face and needs no edge.
    assert sorted(inside) == [False, True]
    for point, both in zip(points, inside):
        assert bool(point_parameters(edge, point, tolerance=tolerance)) is both
        if not both:
            assert not any(domain.contains(LinePath(tuple(point), tuple(point)), 0., tolerance)
                           for domain in (face, other))


@pytest.mark.parametrize("policy", [None, "ring_split"])
def test_face153_arranges_with_the_production_pair_traces(policy):
    arrangement = _arrange(_production_traces(), policy)
    assert arrangement.cells


def test_production_joint_pieces_meet_at_the_contact_points():
    record = _record()
    points = _undecidable_contact_points(*(Cylinder(**_cylinder(record[name]["support"])) for name in ("face", "other")))
    pieces = [trace.curve for trace in _production_traces() if set(trace.owners) == {153, 203}]
    for point in points:
        ends = [end for piece in pieces for end in (piece.evaluate(0.), piece.evaluate(1.))
                if np.linalg.norm(end-point) < 1e-5]
        # Both branch pieces at a contact end on it (no 1e-7 m stubs), or the contact is not in the faces.
        assert all(np.linalg.norm(end-point) < 1e-12 for end in ends), ends


def test_without_the_shared_stored_edge_the_pair_refuses():
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _production_traces(relabel=True)
