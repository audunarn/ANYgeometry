"""Read-only member-axis events in the shared material arrangement engine."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np

from .analytic_roots import isolate_real_roots, trigonometric_roots
from .arrangement_geometry import (LinePath, BezierPath, freeze_edge, plane_roots,
                                   point_parameters, curve_junctions)
from .exact_curves import EllipticArc, CylinderIntersectionCurve
from .branch_curves import BezierQuadricCurve
from .quadric_curves import QuadricIntersectionCurve
from .errors import GeometryError
from .material_arrangement import ArrangementPath, ArrangementPoint, _clip
from .structural import Orientation
from .surfaces import Cone, ExtrudedSurface, Plane


@dataclass(frozen=True, slots=True)
class MemberAxisArrangement:
    edge_id: int
    member_ids: tuple[int, ...]
    curve: object
    split_parameters: tuple[float, ...]
    world_tolerance: float


@dataclass(frozen=True, slots=True)
class MemberPointContact:
    face_id: int | None
    position: tuple[float, float, float]
    member_parameters: tuple[tuple[int, float], ...]
    edge_ids: tuple[int, ...]
    world_tolerance: float


def _support_roots(curve, support, tolerance, check):
    if isinstance(support, ExtrudedSurface):
        # No algebraic implicit form here: a curve is either on the surface (sampled residual) or unsupported.
        points = np.asarray(curve.evaluate(np.linspace(0., 1., 17)))
        if float(np.max(np.linalg.norm(support.evaluate_many(support.local_uv_many(points))-points, axis=1))) <= tolerance:
            return None
        raise GeometryError("a curve leaving an extruded support (a beam or axis contact) is not supported yet")
    if isinstance(support, Plane):
        return plane_roots(curve, support.normal, float(support.normal @ support.origin),
                           tolerance=tolerance, cancellation_check=lambda: (check() or False))
    if isinstance(support, Cone) or isinstance(curve, (QuadricIntersectionCurve, BezierQuadricCurve)):
        from .quadric_events import curve_quadric_roots
        return curve_quadric_roots(curve, support, tolerance=tolerance,
                                   cancellation_check=lambda: (check() or False))
    if isinstance(curve, CylinderIntersectionCurve):
        from .cylinder_curve_events import cylinder_roots
        return cylinder_roots(curve, support, tolerance=tolerance,
                              cancellation_check=lambda: (check() or False))
    project = np.eye(3)-np.outer(support.axis, support.axis)
    implicit_tolerance = tolerance*(2*support.radius+tolerance)
    if isinstance(curve, EllipticArc):
        q0 = project @ (np.asarray(curve.center)-support.origin)
        qc, qs = project @ curve.u_vector, project @ curve.v_vector
        coefficients = (float(q0 @ q0+.5*(qc @ qc+qs @ qs)-support.radius**2),
                        float(2*q0 @ qc), float(2*q0 @ qs),
                        float(.5*(qc @ qc-qs @ qs)), float(qc @ qs))
        if sum(abs(value) for value in coefficients) <= implicit_tolerance:
            return None
        # A short arc of the support circle is rebuilt from its end points with a centre off by many
        # ulp, which spoils the full-circle coefficients but not the arc itself: bound the deviation
        # on the arc's own angular range (sampled, plus the Lipschitz slack between samples).
        samples = curve.start_angle+curve.sweep_angle*np.linspace(0., 1., 17)
        deviation = (coefficients[0]+coefficients[1]*np.cos(samples)+coefficients[2]*np.sin(samples)
                     +coefficients[3]*np.cos(2*samples)+coefficients[4]*np.sin(2*samples))
        slack = (abs(coefficients[1])+abs(coefficients[2])+2*abs(coefficients[3])+2*abs(coefficients[4])
                 )*abs(curve.sweep_angle)/32
        if float(np.max(np.abs(deviation)))+slack <= implicit_tolerance:
            return None
        angles = trigonometric_roots(coefficients, start=curve.start_angle,
            sweep=curve.sweep_angle, tolerance=4*np.finfo(float).eps,
            cancellation_check=lambda: (check() or False))
        return tuple(min(1., max(0., (angle-curve.start_angle)/curve.sweep_angle)) for angle in angles)
    if isinstance(curve, LinePath):
        q0 = project @ (np.asarray(curve.start)-support.origin)
        delta = project @ (np.asarray(curve.end)-curve.start)
        coefficients = (float(q0 @ q0-support.radius**2), float(2*q0 @ delta), float(delta @ delta))
        if sum(abs(value) for value in coefficients) <= implicit_tolerance:
            return None
    elif isinstance(curve, BezierPath):
        controls = (np.asarray(curve.controls)-support.origin) @ project.T
        degree = len(controls)-1
        powers = [[sum(Fraction(float(controls[i, coordinate]))*((-1)**(k-i)*math.comb(degree,i)
                          * math.comb(degree-i,k-i)) for i in range(k+1)) for k in range(degree+1)]
                  for coordinate in range(3)]
        coefficients = [Fraction(0)]*(2*degree+1)
        coefficients[0] = -Fraction(float(support.radius))**2
        for values in powers:
            for i, first in enumerate(values):
                for j, second in enumerate(values):
                    coefficients[i+j] += first*second
        if sum(abs(float(value)) for value in coefficients) <= implicit_tolerance:
            return None
    else:
        raise GeometryError("member support predicate is unsupported")
    return tuple(min(1., max(0., root.witness)) for root in isolate_real_roots(
        coefficients, tolerance=4*np.finfo(float).eps, cancellation_check=lambda: (check() or False))
        if -32*np.finfo(float).eps <= root.witness <= 1+32*np.finfo(float).eps)


def _member_parameters(model, edge_id, parameter, selected):
    values = []
    for use_id in model._edge_member_uses.get(edge_id, ()):
        use = model.member_edge_uses[use_id]
        if use.member_id not in selected:
            continue
        t = parameter if use.orientation is Orientation.FORWARD else 1-parameter
        values.append((use.member_id, use.parent_range.start+t*use.parent_range.length))
    return tuple(sorted(values))


def plan_member_arrangements(model, selected, domains, traces, points, check, *,
                             include_axis_axis=True, include_axis_face=True, check_pair=None,
                             domain_bounds=None):
    """Collect all original axis/axis and axis/material events before editing."""
    edge_ids = sorted({use.edge_id for use in model.member_edge_uses.values() if use.member_id in selected})
    curves = {edge_id: freeze_edge(model, edge_id) for edge_id in edge_ids}
    bounds = {edge_id: curve.bounds() for edge_id, curve in curves.items()}
    tolerances = {edge_id: model.tolerance.effective_length(float(np.linalg.norm(hi-lo)))
                  for edge_id, (lo, hi) in bounds.items()}
    split = {edge_id: {0., 1.} for edge_id in edge_ids}
    contacts = []
    if domain_bounds is None:
        domain_bounds = {}
        for face_id, domain in domains.items():
            boxes = [path.curve.bounds() for loop in domain.boundaries for path in loop]
            domain_bounds[face_id] = (np.min([box[0] for box in boxes], axis=0),
                                      np.max([box[1] for box in boxes], axis=0))
    for edge_id, curve in curves.items():
        members = tuple(sorted(set(member for member, _value in _member_parameters(model, edge_id, 0., selected))))
        lo1, hi1 = bounds[edge_id]
        for face_id, domain in domains.items():
            if not include_axis_face:
                break
            check()
            lo2, hi2 = domain_bounds[face_id]
            tolerance = model.tolerance.effective_length(max(np.linalg.norm(hi1-lo1),np.linalg.norm(hi2-lo2)))
            if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
                continue
            if check_pair is not None:
                check_pair()
            # Exact shared axis topology is already an explicit connection.
            if any(path.source_edge == edge_id for loop in domain.boundaries for path in loop):
                continue
            roots = _support_roots(curve, domain.support, tolerance, check)
            if roots is None:
                for part in _clip(domain, curve, tolerance, check):
                    traces[face_id].append(ArrangementPath(part, edge_id, (face_id,), False, members))
                    for t in (0.,1.):
                        split[edge_id].update(point_parameters(curve,part.evaluate(t),tolerance=tolerance))
                continue
            for parameter in roots:
                if not domain.contains(curve, parameter, tolerance):
                    continue
                position = tuple(curve.evaluate(parameter))
                parameters = _member_parameters(model, edge_id, parameter, selected)
                split[edge_id].add(parameter)
                contacts.append(MemberPointContact(face_id,position,parameters,(edge_id,),tolerance))
                points[face_id].append(ArrangementPoint(position,members))
    for index, first_id in enumerate(edge_ids):
        if not include_axis_axis:
            break
        first = curves[first_id]; lo1, hi1 = bounds[first_id]
        for second_id in edge_ids[index+1:]:
            check()
            second = curves[second_id]; lo2, hi2 = bounds[second_id]
            tolerance = max(tolerances[first_id],tolerances[second_id])
            if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
                continue
            if check_pair is not None:
                check_pair()
            for a,b in curve_junctions(first,second,tolerance=tolerance,
                                     cancellation_check=lambda: (check() or False)):
                parameters = (*_member_parameters(model,first_id,a,selected),
                              *_member_parameters(model,second_id,b,selected))
                first_edge, second_edge = model.edges[first_id], model.edges[second_id]
                first_vertex = first_edge.start if a == 0. else first_edge.end if a == 1. else None
                second_vertex = second_edge.start if b == 0. else second_edge.end if b == 1. else None
                if (first_vertex is not None and first_vertex == second_vertex
                        and all(value in (0., 1.) for _member, value in parameters)):
                    # Authored endpoint topology already gives these axes one
                    # station; preserve the established no-imprint contract.
                    continue
                split[first_id].add(a); split[second_id].add(b)
                position = tuple(first.evaluate(a))
                contacts.append(MemberPointContact(None,position,tuple(sorted(set(parameters))),
                                                   (first_id,second_id),tolerance))
    axes = tuple(MemberAxisArrangement(edge_id,
        tuple(sorted(set(member for member,_value in _member_parameters(model,edge_id,0.,selected)))),
        curves[edge_id],tuple(sorted(split[edge_id])),tolerances[edge_id]) for edge_id in edge_ids)
    return axes, tuple(contacts)
