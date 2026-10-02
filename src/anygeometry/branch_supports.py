"""Exact support intersection of a Bezier extrusion with a quadric patch, as polynomial-chart branch curves.

The counterpart of :func:`anygeometry.quadric_supports.charts_support` for a first support whose chart is the
directrix parameter ``t`` of a Bezier curve of any degree: the events that cut the chart (folds and double
contacts of the discriminant, the poles of a linear branch, the patch limits of the extrusion and the boundary
planes of the other patch) are real roots of exact polynomials, every interval between them is kept when its
midpoint lies in both patches, and whole generators that lie on the quadric are segments by themselves.
"""
from __future__ import annotations

import numpy as np

from .analytic_supports import SupportIntersection
from .branch_algebra import BezierRuledSupport, get_poly_plan
from .branch_curves import BezierQuadricCurve
from .errors import GeometryError
from .quadric_supports import _boundary_planes, _inside_patch, _inside_patch_many, _merge, _poll
from .surfaces import Cone, Cylinder, ExtrudedSurface

_TOLERANCE = 1e-12                    # chart tolerance in the directrix parameter (the unit interval)


def _in_range(roots, lo, hi):
    return [min(hi, max(lo, t)) for t, _m in roots if lo - _TOLERANCE <= t <= hi + _TOLERANCE]


def _branch_values(plan, ts, branch):
    """``s`` of a branch at the parameters ``ts`` (no regularization; used only to qualify exact events)."""
    fs = plan.floats()
    ts = np.asarray(ts, dtype=float)
    a, b, c = fs.at(ts)
    with np.errstate(divide="ignore", invalid="ignore"):
        if plan.linear:
            return -c / b
        radical = np.sqrt(np.maximum(fs.discriminant(ts), 0.0))
        sign_b = np.where(b >= 0, 1.0, -1.0)
        q = -.5 * (b + sign_b * radical)
        big = q / a
        small = np.where(q != 0, c / q, 0.0)
        plus = np.where(sign_b > 0, small, big)
        minus = np.where(sign_b > 0, big, small)
        return plus if branch > 0 else minus


def _points(plan, ts, s):
    fs = plan.floats()
    return fs.origin + fs.frame(np.asarray(ts, dtype=float))[0] + np.asarray(s)[..., None] * fs.direction


def _fold_split(plan):
    """``(folds, nodes)`` over ``[0, 1]``; higher-order contact is refused."""
    roots = plan.discriminant_roots()
    if any(m > 2 for _t, m in roots):
        raise GeometryError("higher-order contact of the supports (a cusp) is not supported")
    return [(t, m) for t, m in roots if m == 1], [(t, m) for t, m in roots if m == 2]


def bezier_quadric_support(first_surface, second_surface, *, tolerance=1e-10, cancellation_check=None):
    """All exact branch charts of a Bezier extrusion and a quadric patch inside both native rectangles."""
    if not isinstance(second_surface, (Cylinder, Cone, ExtrudedSurface)):
        raise GeometryError("a Bezier extrusion meets a Cylinder, a Cone or a quadric extrusion exactly")
    first = BezierRuledSupport.from_surface(first_surface)
    plan = get_poly_plan(first, second_surface)
    if plan.coincident:
        return SupportIntersection(coincident=True)
    lo, hi = sorted(first_surface.u_range)
    _poll(cancellation_check)
    folds, nodes = _fold_split(plan)
    fold_ts = _in_range(folds, lo, hi)
    node_ts = _in_range(nodes, lo, hi)
    pole_ts = _in_range(plan.pole_roots(), lo, hi)
    structural = [lo, hi, *fold_ts, *node_ts]
    s_lo, s_hi = sorted(first_surface.v_range)
    boundary_slack = max(tolerance, 1e-8)
    candidate_events = []                          # (t, kind, reference) crossings of a patch boundary
    for s0 in (s_lo, s_hi):
        candidate_events.extend((t, "height", s0) for t in _in_range(plan.bound_roots(s0), lo, hi))
    for plane in _boundary_planes(second_surface):
        _poll(cancellation_check)
        roots = plan.resultant_roots(plane)
        if roots:
            candidate_events.extend((t, "plane", plane) for t in _in_range(roots, lo, hi))
    fs = plan.floats()
    branches = () if plan.linear and all(v == 0 for v in plan.B) else (1,) if plan.linear else (-1, 1)
    second = plan.second
    curves = []
    for branch in branches:
        events = list(structural)
        if candidate_events:
            # Keep only genuine crossings of a patch boundary by THIS branch inside the other patch.
            ts = np.array([t for t, _k, _r in candidate_events])
            s = _branch_values(plan, ts, branch)
            pts = _points(plan, ts, s)
            in_second = _inside_patch_many(second_surface, pts, boundary_slack)
            in_first = (s >= s_lo - boundary_slack) & (s <= s_hi + boundary_slack)
            for index, (t, kind, reference) in enumerate(candidate_events):
                if not (in_second[index] and in_first[index]):
                    continue
                if kind == "height":
                    on_boundary = abs(s[index] - reference) <= boundary_slack
                else:
                    on_boundary = (abs(float(reference.value(pts[index]))) / float(reference.gradient_norm(pts[index]))
                                   <= boundary_slack)
                if on_boundary:
                    events.append(t)
        split = _merge(events, _TOLERANCE)
        if pole_ts:
            split = _merge([*split, *pole_ts], _TOLERANCE)          # a pole of the linear branch ends every chart
        for e0, e1 in zip(split, split[1:]):
            if e1 - e0 <= _TOLERANCE:
                continue
            mid = .5 * (e0 + e1)
            if not plan.linear and fs.discriminant_scalar(float(mid)) <= 0.0:
                continue
            begin, length = (e0, e1 - e0) if first_surface.u_range[1] >= first_surface.u_range[0] else (e1, e0 - e1)
            left = any(abs(begin - f) <= _TOLERANCE for f in fold_ts)
            right = any(abs(begin + length - f) <= _TOLERANCE for f in fold_ts)
            mode = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
            curve = BezierQuadricCurve._make(first, second, begin, length, branch, mode)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):          # a chart may run to a pole
                ends = curve.evaluate(np.array([0., .5, 1.]))
            if not np.all(np.isfinite(ends)):
                continue                                  # a chart that runs to a pole is not a curve of this pair
            if np.ptp(ends, axis=0).max() <= tolerance:
                continue
            point = ends[1]
            t_mid = float(curve._chart(np.array([.5]))[0][0])
            s_mid = float(np.dot(point - fs.origin - fs.frame(t_mid)[0], fs.direction) / np.dot(fs.direction, fs.direction))
            if not (s_lo - tolerance <= s_mid <= s_hi + tolerance):
                continue
            if not _inside_patch(second_surface, point, tolerance):
                continue
            curves.append(curve)
    points = []
    for t in (*fold_ts, *node_ts):
        a, b, _c = fs.abc_scalar(float(t))
        if plan.linear or a == 0:
            continue
        s = -b / (2 * a)
        point = _points(plan, [t], [s])[0]
        if (s_lo - tolerance <= s <= s_hi + tolerance and _inside_patch(second_surface, point, tolerance)
                and not any(np.linalg.norm(point - curve.evaluate(u)) <= tolerance for curve in curves for u in (0., 1.))):
            points.append(tuple(float(v) for v in point))
    segments = []
    for t in _in_range(plan.common_roots(), lo, hi):
        for start, end in _generator_segments(first_surface, second_surface, plan, float(t), tolerance):
            if not any(np.linalg.norm(np.asarray(start) - np.asarray(a)) <= tolerance
                       and np.linalg.norm(np.asarray(end) - np.asarray(b)) <= tolerance for a, b in segments):
                segments.append((start, end))
    return SupportIntersection(curves=tuple(curves), segments=tuple(segments), points=tuple(points))


def _generator_segments(first_surface, second_surface, plan, t, tolerance):
    """The stretches of the first support's generator at ``t`` that lie in both patches."""
    fs = plan.floats()
    s_lo, s_hi = sorted(first_surface.v_range)
    start = _points(plan, [t], [0.0])[0]
    direction = np.asarray(fs.direction)
    cuts = [s_lo, s_hi]
    for plane in _boundary_planes(second_surface):
        normal = np.asarray(plane.axis)
        rate = float(normal @ direction)
        if abs(rate) > 1e-14:
            s = float(normal @ (np.asarray(plane.origin) - start)) / rate
            if s_lo < s < s_hi:
                cuts.append(s)
    cuts = _merge(cuts, 1e-14)
    stretches = []
    for lower, upper in zip(cuts, cuts[1:]):
        if (upper - lower) * float(np.linalg.norm(direction)) <= tolerance:
            continue
        middle = start + .5 * (lower + upper) * direction
        if _inside_patch(second_surface, middle, tolerance):
            if stretches and abs(stretches[-1][1] - lower) <= 1e-14:
                stretches[-1][1] = upper
            else:
                stretches.append([lower, upper])
    return [(tuple(float(v) for v in start + a * direction), tuple(float(v) for v in start + b * direction))
            for a, b in stretches]
