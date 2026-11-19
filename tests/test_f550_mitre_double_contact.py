"""F550: an equal-radius cylinder mitre whose double contact the stored doubles do not represent exactly.

Fixture ``data/f550_face1094_arrangement.json``: N2 source face 1094 (production face 550), the vertical column
skin (r = 6.5) of a mitre with the inclined skin 1096 (r = 6.5, axes intersecting by design), with the traces of
every incident pair, captured by the T-10 diagnosis. The two supports touch tangentially at the design double
point, through which the stored shared trim edge 2433 (an ellipse) and the web trace 392 pass. In the stored doubles
the double root of the transition equation is not exact: main splits it into two simple folds 3.9e-8 rad apart, so
the cylinder/cylinder branch ends 1.8e-7 m on either side of the double point (tolerance 1e-9) and the arrangement
refuses with an unresolved bridge (the stored traces of the fixture are main's). The exact rational discriminant of
the stored data is positive, but inside the rigorous error bound of the kernel's float fold coefficients: the sign is
undecidable at stored precision. K6 policy (owner decision 2026-11-19, F550 option B with the stored-edge check):
such a fold is taken as an exact double contact (both folds undecidable: the pair is an equal-radius mitre
and its intersection the two bisector ellipses), and the pair stage accepts it only when a stored edge shared by
both faces passes through the contact point; otherwise it refuses. These tests pin the diagnosis, main's refusal,
the accepted face (production pair traces), a near miss the bound decides (32 ulp) that keeps refusing, the 8-ulp
near miss inside the bound (accepted with the edge, refused without it).
Second fixture ``data/f550_face1096_domain.json``: the domain of face 1096 (for the production pair stage).
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import math
import pathlib

import numpy as np
import pytest

from anygeometry.analytic_roots import trigonometric_roots
from anygeometry.arrangement_geometry import LinePath, point_parameters
from anygeometry.errors import GeometryError
from anygeometry.analytic_supports import _ellipse_cylinder_residual, undecidable_contact
from anygeometry.batch_intersections import UNDECIDABLE_CONTACT, _pair_paths_exact
from anygeometry.exact_curves import (CylinderIntersectionCurve, EllipticArc, _CylinderSupport, _double_nodes,
                                      _fold_undecidable)
from anygeometry.material_arrangement import (ArrangementPath, ArrangementPoint, MaterialDomain,
                                              arrange_material)
from anygeometry.serialization import _decode_bezier_quadric_curve
from anygeometry.surfaces import Cylinder

DATA = pathlib.Path(__file__).resolve().parent / "data" / "f550_face1094_arrangement.json"
OTHER = pathlib.Path(__file__).resolve().parent / "data" / "f550_face1096_domain.json"
BRIDGE = "arrangement contains an unresolved bridge"
#: Fixture indices of the six 1094/1096 branch pieces (ordered along the trim) and of the web 392 trace.
JOINT = range(35, 41)
FOLD_BEFORE, FOLD_AFTER, WEB392 = 37, 38, 15


def _cylinder_record(record):
    return {key: record[key] for key in ("origin", "axis", "radial_direction", "radius", "height",
                                         "start_angle", "sweep_angle")}


def _curve(record):
    kind = record["type"]
    if kind == "line":
        return LinePath(tuple(record["start"]), tuple(record["end"]))
    if kind == "elliptic_arc":
        return EllipticArc(tuple(record["center"]), tuple(record["u_vector"]), tuple(record["v_vector"]),
                           record["start_angle"], record["sweep_angle"])
    if kind == "cylinder_intersection":
        first, second = (_CylinderSupport(**{key: tuple(value) if isinstance(value, list) else value
                                             for key, value in _cylinder_record(record[name]).items()})
                         for name in ("first", "second"))
        return CylinderIntersectionCurve(first, second, record["start_angle"], record["sweep_angle"],
                                         record["branch"], record["parameterization"],
                                         tuple(tuple(row) for row in record["transform"]))
    if kind == "bezier_quadric_intersection":
        return _decode_bezier_quadric_curve(record)
    raise AssertionError(kind)


def _path(record):
    return ArrangementPath(_curve(record["curve"]), record["source_edge"], tuple(record["owners"]),
                           record["decomposition"], tuple(record["member_ids"]))


def _fixture():
    record = json.loads(DATA.read_text())
    support = Cylinder(**_cylinder_record(record["support"]))
    domain = MaterialDomain(record["face_id"], support,
                            tuple(tuple(_path(item) for item in loop) for loop in record["boundaries"]))
    traces = tuple(_path(item) for item in record["traces"])
    points = tuple(ArrangementPoint(tuple(item["position"]), tuple(item["member_ids"])) for item in record["points"])
    return domain, traces, points, record["tolerance"], record["area_tolerance"]


def _arrange(traces=None, seam_policy=None):
    domain, stored, points, tolerance, area_tolerance = _fixture()
    return arrange_material(domain, stored if traces is None else traces, points=points, tolerance=tolerance,
                            area_tolerance=area_tolerance, seam_policy=seam_policy)


def _transition_coefficients(first, second, sign):
    """``(c0, nc, ns)`` of the fold equation c0 + nc cos + ns sin = 0, as ``cylinder_cylinder_support`` forms it."""
    cross = np.cross(first.axis, second.axis)
    magnitude = float(np.linalg.norm(cross))
    e1, e2 = np.asarray(first.radial_direction), np.cross(first.axis, first.radial_direction)
    n0 = float((np.asarray(first.origin)-np.asarray(second.origin)) @ cross)
    return n0-sign*magnitude*second.radius, first.radius*float(e1 @ cross), first.radius*float(e2 @ cross)


def _exact_discriminant_sign(first, second, sign):
    """Exact sign of nc^2 + ns^2 - (n0 - sign*|a1 x a2|*r2)^2 for the stored doubles (rational arithmetic;
    the irrational |a1 x a2| is decided by comparing squares). Positive: two simple folds, a real gap."""
    q = lambda vector: [Fraction(float(value)) for value in vector]
    o1, a1, e1 = q(first.origin), q(first.axis), q(first.radial_direction)
    o2, a2 = q(second.origin), q(second.axis)
    e2 = q(np.cross(first.axis, first.radial_direction))
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    dot = lambda u, v: sum(x*y for x, y in zip(u, v))
    c = cross(a1, a2)
    r1, r2 = Fraction(float(first.radius)), Fraction(float(second.radius))
    n0, m2 = dot([x-y for x, y in zip(o1, o2)], c), dot(c, c)
    nc, ns = r1*dot(e1, c), r1*dot(e2, c)
    rational, factor = nc*nc+ns*ns-n0*n0-m2*r2*r2, 2*sign*n0*r2     # value = rational + factor*sqrt(m2)
    signum = lambda value: (value > 0) - (value < 0)
    if factor == 0 or signum(rational) in (0, signum(factor)):
        return signum(rational) or signum(factor)
    left, right = rational*rational, factor*factor*m2
    return signum(rational) if left > right else signum(factor) if left < right else 0


def _design_double_point(first, second):
    """The double point of the mitre: the axes' common point offset by the radius along a1 x a2 (the side the
    face 1094 trim passes through)."""
    o1, a1 = np.asarray(first.origin), np.asarray(first.axis)
    o2, a2 = np.asarray(second.origin), np.asarray(second.axis)
    w0 = o1-o2
    b, d, e = a1 @ a2, a1 @ w0, a2 @ w0
    s = (b*e-(a2 @ a2)*d)/((a1 @ a1)*(a2 @ a2)-b*b)
    normal = np.cross(a1, a2)/np.linalg.norm(np.cross(a1, a2))
    return o1+s*a1-first.radius*normal


def test_fixture_reproduces_the_face550_refusal_under_both_seam_policies():
    for policy in (None, "ring_split"):
        with pytest.raises(GeometryError, match=BRIDGE):
            _arrange(seam_policy=policy)


def test_mitre_double_contact_splits_into_two_folds_in_the_stored_doubles():
    domain, traces, _points, tolerance, _area = _fixture()
    before, after = traces[FOLD_BEFORE].curve, traces[FOLD_AFTER].curve
    first, second = before.first, before.second
    assert first.radius == second.radius == 6.5
    c0, nc, ns = _transition_coefficients(first, second, -1)
    roots = trigonometric_roots((c0, nc, ns, 0., 0.), start=first.start_angle, sweep=first.sweep_angle,
                                tolerance=4*np.finfo(float).eps)
    # One design double root at pi/2 is returned as two simple roots 3.9e-8 rad apart (far above the
    # arrangement's angular tolerance), and the branch pieces end exactly at those two witnesses.
    assert len(roots) == 2 and roots[0] < .5*math.pi < roots[1]
    assert 3.9e-8 < roots[1]-roots[0] < 4.0e-8
    assert roots[1]-roots[0] > 100*tolerance/first.radius
    assert before.start_angle+before.sweep_angle == roots[0] and after.start_angle == roots[1]
    assert (before.parameterization, after.parameterization) == ("right_square", "left_square")
    # The exact stored data agrees that there are two folds: the gap is not an evaluation artefact.
    assert _exact_discriminant_sign(first, second, -1) == 1
    # The stored trim edge and the web trace meet the design double point; the branch ends do not.
    point = _design_double_point(first, second)
    trim = domain.boundaries[0][2]
    assert trim.source_edge == 2433 and type(trim.curve) is EllipticArc
    assert point_parameters(trim.curve, point, tolerance=tolerance)
    assert np.linalg.norm(traces[WEB392].curve.evaluate(1.)-point) < 1e-14
    for end in (before.evaluate(1.), after.evaluate(0.)):
        assert 1.8e-7 < np.linalg.norm(end-point) < 1.81e-7
    assert all(traces[index].owners == (1094, 1096) and traces[index].source_edge is None for index in JOINT)


def _near_miss_traces(ulps):
    """The fixture with the inclined skin's radius lowered by ``ulps`` units in the last place: a genuinely
    open mitre (larger exact discriminant, certified positive), the joint pieces rebuilt on the new folds."""
    domain, traces, _points, tolerance, _area = _fixture()
    first, second = traces[FOLD_BEFORE].curve.first, traces[FOLD_BEFORE].curve.second
    radius = second.radius
    for _ in range(ulps):
        radius = float(np.nextafter(radius, 0.))
    second = replace(second, radius=radius)
    c0, nc, ns = _transition_coefficients(first, second, -1)
    folds = trigonometric_roots((c0, nc, ns, 0., 0.), start=first.start_angle, sweep=first.sweep_angle,
                                tolerance=4*np.finfo(float).eps)
    rebuilt = list(traces)
    for index in JOINT:
        old = traces[index].curve
        start, end = old.start_angle, old.start_angle+old.sweep_angle
        start = folds[1] if index == FOLD_AFTER else start
        end = folds[0] if index == FOLD_BEFORE else end
        rebuilt[index] = replace(traces[index], curve=CylinderIntersectionCurve(
            first, second, start, end-start, old.branch, old.parameterization))
    return tuple(rebuilt), first, second, folds


def _undecidable(first, second):
    return _fold_undecidable(_CylinderSupport.from_surface(first), _CylinderSupport.from_surface(second))


def test_a_genuinely_larger_near_miss_of_the_same_kind_still_refuses():
    """32 ulp (K6, owner decision 2026-11-19: moved from 8 ulp, which is inside the rigorous bound): both folds are
    certified decided (two simple folds each), so the policy does not apply; the production pair stage splits the
    branch at the float folds as on main and the arrangement refuses, under both seam policies."""
    traces, first, second, folds = _near_miss_traces(32)
    assert _exact_discriminant_sign(first, second, -1) == 1 and _exact_discriminant_sign(first, second, 1) == 1
    assert _undecidable(first, second) == {}
    point = _design_double_point(first, second)
    gaps = [float(np.linalg.norm(traces[index].curve.evaluate(t)-point))
            for index, t in ((FOLD_BEFORE, 1.), (FOLD_AFTER, 0.))]
    # Five times the face-550 gap (1.8e-7): 8.5e-7 .. 9e-7 from the double point, far above tolerance.
    assert all(8.5e-7 < gap < 9e-7 for gap in gaps), gaps
    assert folds[1]-folds[0] > 1.9e-7
    for policy in (None, "ring_split"):
        with pytest.raises(GeometryError, match=BRIDGE):
            _arrange(traces, seam_policy=policy)
        with pytest.raises(GeometryError, match=BRIDGE):
            _arrange(_production_traces(32), seam_policy=policy)


def _other_domain(ulps=0, relabel=False):
    """Face 1096 (the inclined skin) from its fixture, its radius lowered by ``ulps`` ulp; ``relabel`` gives its
    copy of the shared trim 2433 another source edge, so no stored edge is shared any more."""
    record = json.loads(OTHER.read_text())
    support = _cylinder_record(record["support"])
    for _ in range(ulps):
        support["radius"] = float(np.nextafter(support["radius"], 0.))
    loops = []
    for loop in record["boundaries"]:
        paths = [_path(item) for item in loop]
        if relabel:
            paths = [replace(path, source_edge=-path.source_edge) if path.source_edge == 2433 else path
                     for path in paths]
        loops.append(tuple(paths))
    return MaterialDomain(record["face_id"], Cylinder(**support), tuple(loops))


def _production_traces(ulps=0, relabel=False):
    """The fixture with the 1094/1096 pair traces recomputed by the production pair stage (support + clipping
    to both domains), the inclined skin's radius lowered by ``ulps`` ulp."""
    domain, traces, _points, tolerance, _area = _fixture()
    paths = _pair_paths_exact(domain, _other_domain(ulps, relabel), None, tolerance, lambda: None,
                              point_contacts=[], face_tolerances=(tolerance, tolerance))
    kept = [trace for trace in traces if set(trace.owners) != {1094, 1096}]
    return tuple(kept)+tuple(ArrangementPath(path, owners=(1094, 1096)) for path in paths)


def test_the_stored_folds_are_certified_undecidable_and_the_pair_is_the_two_mitre_ellipses():
    domain, traces, _points, tolerance, _area = _fixture()
    curve = traces[FOLD_BEFORE].curve
    # Both folds: exact discriminant nonzero (two folds on the trim side, see above) but inside the rigorous error
    # bound of the float fold coefficients (K6-DESIGN.md: about 9.2 units of |a1 x a2| ulp(r2) here).
    assert _fold_undecidable(curve.first, curve.second) == {-1: -1, 1: 1}
    # The branch charts themselves are main's: no node treatment for an inexact pair.
    assert _double_nodes(curve) == ()
    first, second = domain.support, _other_domain().support
    contact = undecidable_contact(first, second, tolerance=tolerance)
    assert len(contact.points) == 2 and contact.support is not None
    ellipses = contact.support.curves
    assert len(ellipses) == 2 and all(type(item) is EllipticArc for item in ellipses)
    assert all(_ellipse_cylinder_residual(item, cylinder) < 1e-11 for item in ellipses for cylinder in (first, second))
    # Both ellipses pass through both contact points (the design double points).
    for point in contact.points:
        assert all(point_parameters(item, np.asarray(point), tolerance=1e-12) for item in ellipses)
    point = _design_double_point(curve.first, curve.second)
    assert min(np.linalg.norm(np.asarray(item)-point) for item in contact.points) < 1e-12
    # The production pair traces are arcs of those ellipses, on the stored trim 2433.
    production = [trace.curve for trace in _production_traces() if set(trace.owners) == {1094, 1096}]
    assert production and all(type(item) is EllipticArc for item in production)
    trim = domain.boundaries[0][2].curve
    for item in production:
        for t in (0., .5, 1.):
            assert point_parameters(trim, item.evaluate(t), tolerance=1e-12)


def test_face1094_mitre_arrangement_is_accepted_once_the_double_contact_is_resolved():
    """Formerly the strict xfail: the production pair traces of the certified-undecidable mitre, with the stored
    shared trim 2433 through the contact point, give a complete arrangement (K6 policy)."""
    for policy in (None, "ring_split"):
        arrangement = _arrange(_production_traces(), seam_policy=policy)
        assert arrangement.cells


def test_the_8_ulp_near_miss_is_inside_the_bound_and_accepted_as_a_contact():
    """Owner decision 2026-11-19: 8 ulp is inside the rigorous bound (folds 8.24 and 8.72 units against about 9.2),
    so with the stored edge through the contact point it is a double contact and the face arranges."""
    _traces, first, second, _folds = _near_miss_traces(8)
    assert _exact_discriminant_sign(first, second, -1) == 1
    assert _undecidable(first, second) == {-1: -1, 1: 1}
    for policy in (None, "ring_split"):
        assert _arrange(_production_traces(8), seam_policy=policy).cells


@pytest.mark.parametrize("ulps", [0, 8])
def test_without_a_stored_shared_edge_through_the_contact_point_the_pair_refuses(ulps):
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _production_traces(ulps, relabel=True)


def test_a_shared_edge_that_stops_short_of_the_contact_point_is_no_witness():
    """Codex K6 R1 H1: the shared trim cut 0.75e-9 rad before the contact point (about 5e-9 m short, tolerance
    1e-9): the clamped end parameter of the shared piece is not a witness; the continuation has a different id in each face."""
    domain, _traces, _points, tolerance, _area = _fixture()
    other = _other_domain()
    target = undecidable_contact(domain.support, other.support, tolerance=tolerance).points

    def cut(loops, continuation):
        result = []
        for loop in loops:
            paths = []
            for path in loop:
                if path.source_edge != 2433:
                    paths.append(path)
                    continue
                arc = path.curve
                hits = [t for p in target for t in point_parameters(arc, np.asarray(p), tolerance=tolerance)]
                assert hits
                angle = arc.start_angle+hits[0]*arc.sweep_angle
                at = angle-math.copysign(.75e-9, arc.sweep_angle)
                head = EllipticArc(arc.center, arc.u_vector, arc.v_vector, arc.start_angle, at-arc.start_angle)
                tail = EllipticArc(arc.center, arc.u_vector, arc.v_vector, at, arc.start_angle+arc.sweep_angle-at)
                gap = min(np.linalg.norm(head.evaluate(1.)-np.asarray(p)) for p in target)
                assert 2*tolerance < gap < 1e-8
                paths += [replace(path, curve=head), replace(path, curve=tail, source_edge=continuation)]
            result.append(tuple(paths))
        return tuple(result)

    first = MaterialDomain(domain.face_id, domain.support, cut(domain.boundaries, -1))
    second = MaterialDomain(other.face_id, other.support, cut(other.boundaries, -2))
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _pair_paths_exact(first, second, None, tolerance, lambda: None, point_contacts=[],
                          face_tolerances=(tolerance, tolerance))


def test_the_t2b_shadow_refuses_like_exact_planning():
    """Codex K6 R2 H1: the T2b support shadow runs the same policy, so topology_certified planning cannot accept a
    contact that exact planning refuses."""
    domain, _traces, _points, tolerance, _area = _fixture()
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _pair_paths_exact(domain, _other_domain(relabel=True), None, tolerance, lambda: None, point_contacts=[],
                          face_tolerances=(tolerance, tolerance), _shadow={})
    shadow = {}
    assert _pair_paths_exact(domain, _other_domain(), None, tolerance, lambda: None, point_contacts=[],
                             face_tolerances=(tolerance, tolerance), _shadow=shadow) is None
    assert not shadow.get("coincident")


def test_every_contact_near_both_faces_needs_its_own_witness(monkeypatch):
    """Codex K6 R2 H2: the second double point is 2.4 m from face 1094's box, beyond the near radius (r / 4 = 1.6 m):
    no witness needed. Were it within the near radius of both faces, the witness at the first contact would not
    cover it: the pair refuses."""
    import anygeometry.batch_intersections as batch
    domain, _traces, _points, tolerance, _area = _fixture()
    real = batch.undecidable_contact

    def widened(first, second, *, tolerance):
        contact = real(first, second, tolerance=tolerance)
        return replace(contact, near_radius=10.)

    monkeypatch.setattr(batch, "undecidable_contact", widened)
    with pytest.raises(GeometryError, match=UNDECIDABLE_CONTACT):
        _pair_paths_exact(domain, _other_domain(), None, tolerance, lambda: None, point_contacts=[],
                          face_tolerances=(tolerance, tolerance))
