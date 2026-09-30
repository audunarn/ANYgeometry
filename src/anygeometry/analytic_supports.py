"""Analytic Plane/Cylinder support intersections and finite chart clipping.

This layer intersects the original supports. Face-material arrangements apply
their additional trim constraints separately. Polynomial roots define chart
events; midpoint evaluations classify intervals only after all events are
isolated. No display polyline defines the intersection.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .analytic_roots import trigonometric_roots
from .errors import GeometryError
from .exact_curves import CylinderIntersectionCurve, EllipticArc
from .surfaces import Cylinder, Plane


@dataclass(frozen=True, slots=True)
class SupportIntersection:
    curves: tuple = ()
    segments: tuple = ()
    points: tuple = ()
    coincident: bool = False


def _events(coefficients, cylinder, tolerance, cancellation_check):
    if not np.any(coefficients):
        # A coincident constraint contributes no isolated chart events.
        return ()
    return trigonometric_roots(coefficients, start=cylinder.start_angle,
                              sweep=cylinder.sweep_angle,
                              tolerance=min(tolerance, 4*np.finfo(float).eps),
                              cancellation_check=cancellation_check)


def _linear_events(center, cosine, sine, cylinder, tolerance, cancellation_check):
    return _events((center, cosine, sine, 0, 0), cylinder, tolerance, cancellation_check)


def _ordered(values, cylinder, tolerance):
    values = sorted(values, reverse=cylinder.sweep_angle < 0)
    result = []
    for value in values:
        if not result or abs(value-result[-1]) > tolerance:
            result.append(value)
    return result


def _inside(cylinder, point, tolerance):
    u, v = cylinder.local_uv(point)
    angular = tolerance/max(cylinder.radius*abs(cylinder.sweep_angle), tolerance)
    axial = tolerance/max(abs(cylinder.height), tolerance)
    return -angular <= u <= 1+angular and -axial <= v <= 1+axial


def plane_cylinder_support(plane, cylinder, *, tolerance=1e-10, cancellation_check=None):
    """All exact intersections within a Cylinder's native finite rectangle."""
    if not isinstance(plane, Plane) or not isinstance(cylinder, Cylinder):
        raise GeometryError("analytic plane/cylinder intersection needs Plane and Cylinder")
    normal = plane.normal
    e1, e2 = cylinder.radial_direction, cylinder.circumferential_direction
    center = float((cylinder.origin-plane.origin) @ normal)
    cosine, sine = cylinder.radius*float(e1 @ normal), cylinder.radius*float(e2 @ normal)
    axial = float(cylinder.axis @ normal)
    angle_tolerance = tolerance/max(cylinder.radius, 1.)
    if axial == 0:
        angles = _linear_events(center, cosine, sine, cylinder, angle_tolerance,
                                cancellation_check)
        segments = []
        for angle in angles:
            start = cylinder.origin+cylinder.radius*(math.cos(angle)*e1+math.sin(angle)*e2)
            end = start+cylinder.height*cylinder.axis
            if not any(np.linalg.norm(start-np.asarray(item[0])) <= tolerance for item in segments):
                segments.append((tuple(start), tuple(end)))
        return SupportIntersection(segments=tuple(segments))
    ellipse = EllipticArc(
        cylinder.origin-center/axial*cylinder.axis,
        cylinder.radius*e1-cosine/axial*cylinder.axis,
        cylinder.radius*e2-sine/axial*cylinder.axis,
        cylinder.start_angle, cylinder.sweep_angle)
    events = [cylinder.start_angle, cylinder.start_angle+cylinder.sweep_angle]
    for height in (0., cylinder.height):
        events.extend(_linear_events(center+height*axial, cosine, sine, cylinder,
                                     angle_tolerance, cancellation_check))
    # Every returned edge has an open parameter interval with distinct ends.
    # A periodic seam is represented by two charts, even without another cut.
    events.append(cylinder.start_angle+.5*cylinder.sweep_angle)
    events = _ordered(events, cylinder, angle_tolerance)
    curves, points = [], []
    for start, end in zip(events, events[1:]):
        curve = EllipticArc(ellipse.center, ellipse.u_vector, ellipse.v_vector, start, end-start)
        if _inside(cylinder, curve.evaluate(.5), tolerance):
            curves.append(curve)
    for angle in events:
        point = ellipse.evaluate((angle-cylinder.start_angle)/cylinder.sweep_angle)
        if _inside(cylinder, point, tolerance) and not any(
            np.linalg.norm(point-curve.evaluate(t)) <= tolerance
            for curve in curves for t in (0., 1.)):
            points.append(tuple(point))
    return SupportIntersection(curves=tuple(curves), points=tuple(points))


def _quadratic_trig(center, cosine, sine, radius):
    return (float(center @ center+.5*(cosine @ cosine+sine @ sine)-radius**2),
            float(2*center @ cosine), float(2*center @ sine),
            float(.5*(cosine @ cosine-sine @ sine)), float(cosine @ sine))


def _plane_branch_events(first, second, normal, offset, tolerance, cancellation_check):
    """Both quadratic branches crossing an infinite plane."""
    e1, e2 = first.radial_direction, first.circumferential_direction
    denominator = float(normal @ first.axis)
    c0 = float(normal @ first.origin-offset)
    cc, cs = first.radius*float(normal @ e1), first.radius*float(normal @ e2)
    if denominator == 0:
        return _linear_events(c0, cc, cs, first, tolerance, cancellation_check)
    center = first.origin-c0/denominator*first.axis-second.origin
    cosine = first.radius*e1-cc/denominator*first.axis
    sine = first.radius*e2-cs/denominator*first.axis
    project = np.eye(3)-np.outer(second.axis, second.axis)
    return _events(_quadratic_trig(project @ center, project @ cosine, project @ sine,
                                  second.radius), first, tolerance, cancellation_check)


def cylinder_cylinder_support(first, second, *, tolerance=1e-10, cancellation_check=None):
    """All analytic nonparallel branches, or parallel generators/coincidence."""
    if not isinstance(first, Cylinder) or not isinstance(second, Cylinder):
        raise GeometryError("analytic cylinder intersection needs two Cylinders")
    cross = np.cross(first.axis, second.axis)
    magnitude = float(np.linalg.norm(cross))
    if magnitude == 0:
        return _parallel(first, second, tolerance)
    e1, e2 = first.radial_direction, first.circumferential_direction
    # D/4 = |a1 x a2|^2*r2^2 - ((base-o2).(a1 x a2))^2.
    n0 = float((first.origin-second.origin) @ cross)
    nc, ns = first.radius*float(e1 @ cross), first.radius*float(e2 @ cross)
    angle_tolerance = tolerance/max(first.radius, second.radius, 1.)
    transition = []
    for sign in (-1, 1):
        transition.extend(_linear_events(n0-sign*magnitude*second.radius, nc, ns,
                                         first, angle_tolerance, cancellation_check))
    events = [first.start_angle, first.start_angle+first.sweep_angle,
              first.start_angle+.5*first.sweep_angle, *transition]
    for owner in (first, second):
        for height in (0., owner.height):
            point = owner.origin+height*owner.axis
            events.extend(_plane_branch_events(first, second, owner.axis,
                                               float(owner.axis @ point), angle_tolerance,
                                               cancellation_check))
    # Second-cylinder seam and finite angular boundaries are radial planes.
    for angle in (second.start_angle, second.start_angle+second.sweep_angle):
        radial = math.cos(angle)*second.radial_direction+math.sin(angle)*second.circumferential_direction
        normal = np.cross(second.axis, radial)
        events.extend(_plane_branch_events(first, second, normal, float(normal @ second.origin),
                                           angle_tolerance, cancellation_check))
    events = _ordered(events, first, angle_tolerance)
    curves, points = [], []
    for start, end in zip(events, events[1:]):
        angle = .5*(start+end)
        if abs(n0+nc*math.cos(angle)+ns*math.sin(angle)) > magnitude*second.radius:
            continue
        def regular(at):
            # Simple transitions need a square chart; a double contact has
            # a finite one-sided derivative in the linear angular chart.
            return any(abs(at-item) <= angle_tolerance for item in transition) and abs(-nc*math.sin(at)+ns*math.cos(at)) > angle_tolerance
        left, right = regular(start), regular(end)
        mode = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
        for branch in (-1, 1):
            curve = CylinderIntersectionCurve(first, second, start, end-start, branch, mode)
            point = curve.evaluate(.5)
            if _inside(first, point, tolerance) and _inside(second, point, tolerance):
                curves.append(curve)
    for angle in transition:
        base = first.origin+first.radius*(math.cos(angle)*e1+math.sin(angle)*e2)
        project = np.eye(3)-np.outer(second.axis, second.axis)
        q, w = project @ (base-second.origin), project @ first.axis
        point = base-float(q @ w)/float(w @ w)*first.axis
        if _inside(first, point, tolerance) and _inside(second, point, tolerance) and not any(
            np.linalg.norm(point-curve.evaluate(t)) <= tolerance
            for curve in curves for t in (0., 1.)):
            points.append(tuple(point))
    return SupportIntersection(curves=tuple(curves), points=tuple(points))


def _parallel(first, second, tolerance):
    offset = second.origin-first.origin
    radial_offset = offset-float(offset @ first.axis)*first.axis
    distance = float(np.linalg.norm(radial_offset))
    if distance == 0:
        return SupportIntersection(coincident=abs(first.radius-second.radius) <= tolerance)
    if distance > first.radius+second.radius+tolerance or distance < abs(first.radius-second.radius)-tolerance:
        return SupportIntersection()
    x = (first.radius**2-second.radius**2+distance**2)/(2*distance)
    height2 = first.radius**2-x*x
    if height2 < -tolerance*max(first.radius, second.radius):
        return SupportIntersection()
    along = radial_offset/distance
    across = np.cross(first.axis, along)
    segments, points = [], []
    roots = (0.,) if height2 <= 0 else (-math.sqrt(height2), math.sqrt(height2))
    for root in roots:
        base = first.origin+x*along+root*across
        second_start = float(offset @ first.axis)
        second_end = second_start+float(second.axis @ first.axis)*second.height
        lo = max(min(0., first.height), min(second_start, second_end))
        hi = min(max(0., first.height), max(second_start, second_end))
        if hi < lo-tolerance:
            continue
        middle = base+.5*(lo+hi)*first.axis
        if not _inside(first, middle, tolerance) or not _inside(second, middle, tolerance):
            continue
        start, end = base+lo*first.axis, base+hi*first.axis
        if hi-lo <= tolerance:
            points.append(tuple(start))
        else:
            segments.append((tuple(start), tuple(end)))
    return SupportIntersection(segments=tuple(segments), points=tuple(points))
