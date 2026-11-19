"""F550: an equal-radius cylinder mitre whose double contact the stored doubles do not represent exactly.

Fixture ``data/f550_face1094_arrangement.json``: N2 source face 1094 (production face 550), the vertical column
skin (r = 6.5) of a mitre with the inclined skin 1096 (r = 6.5, axes intersecting by design), with the traces of
every incident pair, captured by the T-10 diagnosis. The two supports touch tangentially at the design double
point, through which the stored shared trim edge 2433 (an ellipse) and the web trace 392 pass. In the stored doubles
the double root of the transition equation is not exact: it splits into two simple folds 3.9e-8 rad apart, so the
cylinder/cylinder branch ends 1.8e-7 m on either side of the double point (tolerance 1e-9) and the arrangement
refuses with an unresolved bridge. The exact rational discriminant of the stored data is positive, so the split is
a property of the stored doubles, not of their float evaluation: no certificate that the ends coincide exists and
nothing may be welded. These tests pin that diagnosis, the refusal, a genuinely larger near miss that must keep
refusing, and (strict xfail) the accepted outcome a certified resolution would have to deliver.
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
from anygeometry.exact_curves import CylinderIntersectionCurve, EllipticArc, _CylinderSupport
from anygeometry.material_arrangement import (ArrangementPath, ArrangementPoint, MaterialDomain,
                                              arrange_material)
from anygeometry.serialization import _decode_bezier_quadric_curve
from anygeometry.surfaces import Cylinder

DATA = pathlib.Path(__file__).resolve().parent / "data" / "f550_face1094_arrangement.json"
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


def test_a_genuinely_larger_near_miss_of_the_same_kind_still_refuses():
    traces, first, second, folds = _near_miss_traces(8)
    assert _exact_discriminant_sign(first, second, -1) == 1
    point = _design_double_point(first, second)
    gaps = [float(np.linalg.norm(traces[index].curve.evaluate(t)-point))
            for index, t in ((FOLD_BEFORE, 1.), (FOLD_AFTER, 0.))]
    # Same order as the face-550 gap (1.8e-7): 2.5e-7 .. 1e-6 from the double point, far above tolerance.
    assert all(2.5e-7 < gap < 1e-6 for gap in gaps), gaps
    assert folds[1]-folds[0] > 4e-8
    for policy in (None, "ring_split"):
        with pytest.raises(GeometryError, match=BRIDGE):
            _arrange(traces, seam_policy=policy)


@pytest.mark.xfail(strict=True, raises=GeometryError,
                   reason="F550: no certified resolution of the inexact mitre double contact yet (owner decision)")
def test_face1094_mitre_arrangement_is_accepted_once_the_double_contact_is_resolved():
    arrangement = _arrange()
    assert arrangement.cells
