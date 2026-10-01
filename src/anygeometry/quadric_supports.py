"""Patch-level exact intersections of a Cone with a Plane, Cylinder or Cone.

``cone_support`` is the Cone counterpart of ``plane_cylinder_support`` and
``cylinder_cylinder_support``: it intersects the *supports* inside their native
rectangles and returns exact curves; face material trims are applied afterwards by
the arrangement. Generic pairs return :class:`QuadricIntersectionCurve` charts cut
at every exact event; a plane meeting a cone in an ellipse (or circle) and coaxial
rings are returned as :class:`~anygeometry.exact_curves.EllipticArc`, which the
rest of the engine already integrates analytically.

All the exact work is shared per *support pair* through
:func:`~anygeometry.quadric_algebra.get_plan`, so the facets of a shell cost a
dictionary lookup and a handful of midpoint tests each.
"""
from __future__ import annotations

import math

import numpy as np

from .analytic_supports import SupportIntersection
from .errors import GeometryError
from .exact_curves import EllipticArc
from .quadric_algebra import QuadricSupport, RuledSupport, TWO_PI, circle_angles, get_plan
from .quadric_curves import IDENTITY, QuadricIntersectionCurve
from .surfaces import Cone, Cylinder, Plane

_EPS = float(np.finfo(float).eps)


def _poll(check, message="quadric support intersection cancelled"):
    if check is not None and check():
        raise GeometryError(message)


def _axial_range(surface):
    height = float(surface.height)
    return min(0.0, height), max(0.0, height)


def _max_radius(surface):
    if isinstance(surface, Cone):
        return max(float(surface.radius_start), float(surface.radius_end))
    return float(surface.radius)


def _inside_patch(surface, point, tolerance):
    """Whether ``point`` (on the support) lies in the surface's native rectangle."""
    if isinstance(surface, Plane):
        return True
    u, v = surface.local_uv(point)
    angular = tolerance / max(_max_radius(surface) * abs(surface.sweep_angle), tolerance)
    axial = tolerance / max(abs(surface.height), tolerance)
    return -angular <= u <= 1 + angular and -axial <= v <= 1 + axial


def _inside_patch_many(surface, points, tolerance):
    """Vectorized :func:`_inside_patch` (the same angle-on-sweep rule as ``Cylinder.local_uv``)."""
    points = np.atleast_2d(points)
    if isinstance(surface, Plane):
        return np.ones(len(points), dtype=bool)
    rel = points - np.asarray(surface.origin)
    axis = np.asarray(surface.axis)
    axial = rel @ axis
    radial = rel - axial[:, None] * axis
    angle = np.arctan2(radial @ np.asarray(surface.circumferential_direction), radial @ np.asarray(surface.radial_direction))
    sweep = float(surface.sweep_angle)
    raw = angle - float(surface.start_angle)
    wrap_tol = 64.0 * _EPS * np.maximum(1.0, np.maximum(np.abs(angle), abs(float(surface.start_angle))))
    if sweep > 0.0:
        delta = np.mod(raw, TWO_PI)
        delta = np.where(np.abs(delta - TWO_PI) <= wrap_tol, 0.0, delta)
    else:
        delta = -np.mod(-raw, TWO_PI)
        delta = np.where(np.abs(delta + TWO_PI) <= wrap_tol, 0.0, delta)
    delta = np.where(np.abs(raw) <= wrap_tol, 0.0, delta)
    u, v = delta / sweep, axial / float(surface.height)
    angular = tolerance / max(_max_radius(surface) * abs(sweep), tolerance)
    axial_tol = tolerance / max(abs(float(surface.height)), tolerance)
    return (u >= -angular) & (u <= 1 + angular) & (v >= -axial_tol) & (v <= 1 + axial_tol)


def _branch_points(plan, angles, branch):
    """Branch points at ``angles`` (no regularization; used only to qualify exact events)."""
    fs = plan.floats()
    theta = np.asarray(angles, dtype=float)
    a, b, c = fs.at(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        if plan.linear:
            s = -c / b
        else:
            radical = np.sqrt(np.maximum(fs.discriminant(theta), 0.0))
            sign_b = np.where(b >= 0, 1.0, -1.0)
            q = -.5 * (b + sign_b * radical)
            big = q / a
            small = np.where(q != 0, c / q, 0.0)
            plus = np.where(sign_b > 0, small, big)
            minus = np.where(sign_b > 0, big, small)
            s = plus if branch > 0 else minus
    cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
    p0, p1, p2 = (np.asarray(row) for row in fs.p_rows)
    d0, d1, d2 = (np.asarray(row) for row in fs.d_rows)
    return fs.origin + p0 + cos * p1 + sin * p2 + s[:, None] * (d0 + cos * d1 + sin * d2)


def _boundary_planes(surface):
    """Planes carrying a ruled patch's end rings and angular seams (as exact quadric descriptors)."""
    axis = tuple(float(v) for v in surface.axis)
    origin = np.asarray(surface.origin)
    planes = []
    for z in (0.0, float(surface.height)):
        planes.append(QuadricSupport("plane", tuple(origin + z * np.asarray(axis)), axis))
    e1, e2 = np.asarray(surface.radial_direction), np.asarray(surface.circumferential_direction)
    for angle in (surface.start_angle, surface.start_angle + surface.sweep_angle):
        radial = math.cos(angle) * e1 + math.sin(angle) * e2
        normal = np.cross(np.asarray(axis), radial)
        planes.append(QuadricSupport("plane", tuple(origin), tuple(float(v) for v in normal)))
    return planes


def _in_range(angles, lo, hi, tolerance):
    return [a for a, _m in circle_angles(angles, lo, hi, tolerance)]


def _merge(values, tolerance):
    values = sorted(values)
    merged = []
    for value in values:
        if not merged or value - merged[-1] > tolerance:
            merged.append(value)
    return merged


# ---------------------------------------------------------------------------
# Generic branch charts


def _fold_split(plan):
    """``(fold angles, node angles)`` over the whole circle; higher-order contact is refused."""
    roots = plan.discriminant_roots()
    if any(m > 2 for _a, m in roots):
        raise GeometryError("higher-order contact of the supports (a cusp) is not supported")
    return [(a, m) for a, m in roots if m == 1], [(a, m) for a, m in roots if m == 2]


def charts_support(first_surface, second_surface, *, tolerance=1e-10, cancellation_check=None):
    """All exact branch charts of two supports inside both native rectangles.

    ``first_surface`` (a Cylinder or Cone) supplies the angle; the second is any
    quadric support. Events (folds, double contacts, poles, patch heights, the
    second patch's ring and seam planes) cut the angular range; each interval is
    kept only if its midpoint lies in both rectangles.
    """
    first = RuledSupport.from_surface(first_surface)
    plan = get_plan(first, second_surface)
    start, sweep = float(first_surface.start_angle), float(first_surface.sweep_angle)
    lo, hi = sorted((start, start + sweep))
    tol = tolerance / max(1.0, _max_radius(first_surface))
    _poll(cancellation_check)
    folds, nodes = _fold_split(plan)
    fold_angles = _in_range(folds, lo, hi, tol)
    node_angles = _in_range(nodes, lo, hi, tol)
    pole_angles = _in_range(plan.pole_roots(), lo, hi, tol)
    structural = [lo, hi, *fold_angles, *node_angles]
    if abs(abs(sweep) - TWO_PI) <= tol:
        structural.append(start + .5 * sweep)
    z_lo, z_hi = _axial_range(first_surface)
    boundary_slack = max(tolerance, 1e-8)
    height = float(first_surface.height)
    first_origin, first_axis = np.asarray(first.origin), np.asarray(first.axis)
    candidate_events = []                       # (angle, kind, reference) crossings of a patch boundary
    for z in (0.0, height):
        candidate_events.extend((a, "height", z) for a in _in_range(plan.bound_roots(z), lo, hi, tol))
    if isinstance(second_surface, (Cylinder, Cone)):
        for plane in _boundary_planes(second_surface):
            _poll(cancellation_check)
            roots = plan.resultant_roots(plane)
            if roots:
                candidate_events.extend((a, "plane", plane) for a in _in_range(roots, lo, hi, tol))
    fs = plan.floats()
    branches = (1,) if plan.linear else (-1, 1)
    second = plan.second
    curves = []
    for branch in branches:
        events = list(structural)
        if candidate_events:
            # Keep only genuine crossings of a patch boundary by THIS branch inside the other patch.
            angles = np.array([a for a, _k, _r in candidate_events])
            pts = _branch_points(plan, angles, branch)
            axial = (pts - first_origin) @ first_axis
            in_second = _inside_patch_many(second_surface, pts, boundary_slack)
            in_first = (axial >= z_lo - boundary_slack) & (axial <= z_hi + boundary_slack)
            for index, (angle, kind, reference) in enumerate(candidate_events):
                if not (in_second[index] and in_first[index]):
                    continue
                if kind == "height":
                    on_boundary = abs(axial[index] - reference) <= boundary_slack
                else:
                    on_boundary = (abs(float(reference.value(pts[index]))) / float(reference.gradient_norm(pts[index]))
                                   <= boundary_slack)
                if on_boundary:
                    events.append(angle)
        split = _merge(events, tol)
        if pole_angles:
            diverging = []
            for angle in pole_angles:
                if plan.linear:
                    diverging.append(angle)
                    continue
                b_at = fs.abc_scalar(float(angle))[1]
                if abs(b_at) <= 1e-9 * fs.scale or branch != (1 if b_at >= 0 else -1):
                    diverging.append(angle)
            split = _merge([*split, *diverging], tol)
        for e0, e1 in zip(split, split[1:]):
            if e1 - e0 <= tol:
                continue
            mid = .5 * (e0 + e1)
            if not plan.linear and fs.discriminant_scalar(float(mid)) <= 0.0:
                continue
            begin, length = (e0, e1 - e0) if sweep >= 0 else (e1, e0 - e1)
            left = any(abs(begin - f) <= tol for f in fold_angles)
            right = any(abs(begin + length - f) <= tol for f in fold_angles)
            mode = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
            curve = QuadricIntersectionCurve._make(first, second, begin, length, branch, mode)
            ends = curve.evaluate(np.array([0., .5, 1.]))
            if not np.all(np.isfinite(ends)):
                continue                                  # a chart that runs to a pole is not a curve of this pair
            if np.ptp(ends, axis=0).max() <= tolerance:
                continue                                  # the apex branch (s = 0 for every angle) is a point, not a curve
            point = ends[1]
            axial_mid = float((point - first_origin) @ first_axis)
            if not (z_lo - tolerance <= axial_mid <= z_hi + tolerance):
                continue
            if not _inside_patch(second_surface, point, tolerance):
                continue
            curves.append(curve)
    points = []
    for angle in (*fold_angles, *node_angles):
        a, b, _c = fs.abc_scalar(float(angle))
        if plan.linear or a == 0:
            continue
        s = -b / (2 * a)
        point = _point_at(first, fs, float(angle), s)
        axial = float((point - np.asarray(first.origin)) @ np.asarray(first.axis))
        if (z_lo - tolerance <= axial <= z_hi + tolerance and _inside_patch(second_surface, point, tolerance)
                and not any(np.linalg.norm(point - curve.evaluate(t)) <= tolerance for curve in curves for t in (0., 1.))):
            points.append(tuple(point))
    segments = []
    for angle in _in_range(plan.common_roots(), lo, hi, tol):
        for start, end in _generator_segments(first_surface, second_surface, plan, float(angle), tolerance):
            if not any(np.linalg.norm(np.asarray(start) - np.asarray(a)) <= tolerance
                       and np.linalg.norm(np.asarray(end) - np.asarray(b)) <= tolerance for a, b in segments):
                segments.append((start, end))               # the seam of a full turn lists its generator twice
    for apex in _apex_contacts(first_surface, second_surface, tolerance):
        if not any(np.linalg.norm(np.asarray(apex) - np.asarray(old)) <= tolerance for old in points):
            points.append(apex)
    return SupportIntersection(curves=tuple(curves), segments=tuple(segments), points=tuple(points))


def _generator_segments(first_surface, second_surface, plan, angle, tolerance):
    """The stretches of the first support's generator at ``angle`` that lie in both patches.

    The generator is a component of the intersection by itself when ``A``, ``B`` and ``C`` all vanish at
    ``angle``: every point of the line satisfies the second support's equation.
    """
    fs = plan.floats()
    first = plan.first
    z_lo, z_hi = _axial_range(first_surface)
    start = _point_at(first, fs, angle, 0.0)
    direction = _point_at(first, fs, angle, 1.0) - start
    cuts = [z_lo, z_hi]
    if isinstance(second_surface, (Cylinder, Cone)):
        for plane in _boundary_planes(second_surface):
            normal = np.asarray(plane.axis)
            rate = float(normal @ direction)
            if abs(rate) > 1e-14:
                s = float(normal @ (np.asarray(plane.origin) - start)) / rate
                if z_lo < s < z_hi:
                    cuts.append(s)
    cuts = _merge(cuts, 1e-14)
    stretches = []
    for lower, upper in zip(cuts, cuts[1:]):
        if (upper - lower) * float(np.linalg.norm(direction)) <= tolerance:
            continue
        middle = start + .5 * (lower + upper) * direction
        if _inside_patch(first_surface, middle, tolerance) and _inside_patch(second_surface, middle, tolerance):
            if stretches and abs(stretches[-1][1] - lower) <= 1e-14:
                stretches[-1][1] = upper
            else:
                stretches.append([lower, upper])
    return [(tuple(start + a * direction), tuple(start + b * direction)) for a, b in stretches]


def _apex_contacts(first_surface, second_surface, tolerance):
    """Apexes of the two supports that lie on the other support, inside both patches.

    A cone's apex is a singular point of the intersection: the branch ``s = 0`` of its rulings meets every
    other branch there, and when no curve ends at it the apex is an isolated contact.
    """
    contacts = []
    for cone, other in ((first_surface, second_surface), (second_surface, first_surface)):
        if not isinstance(cone, Cone):
            continue
        slope = (float(cone.radius_end) - float(cone.radius_start)) / float(cone.height)
        apex = np.asarray(cone.origin) - (float(cone.radius_start) / slope) * np.asarray(cone.axis)
        quadric = QuadricSupport.from_surface(other)
        scale = float(quadric.gradient_norm(apex)) + tolerance
        if abs(float(quadric.value(apex))) <= tolerance * scale and _inside_patch(cone, apex, tolerance)                 and _inside_patch(other, apex, tolerance):
            contacts.append(tuple(float(v) for v in apex))
    return contacts


def _point_at(first, fs, theta, s):
    c, sn = math.cos(theta), math.sin(theta)
    (p0, p1, p2), (d0, d1, d2) = fs.p_rows, fs.d_rows
    return np.array([fs.origin[i] + p0[i] + c * p1[i] + sn * p2[i] + s * (d0[i] + c * d1[i] + sn * d2[i])
                     for i in range(3)])


# ---------------------------------------------------------------------------
# Plane meeting a Cone


def plane_cone_support(plane, cone, *, tolerance=1e-10, cancellation_check=None):
    """Exact plane/cone intersections inside the cone's native rectangle.

    Ellipses and circles are :class:`EllipticArc`; hyperbola and parabola branches are
    :class:`QuadricIntersectionCurve` charts split at the generators parallel to the
    plane; a plane through the apex gives generator segments or the apex point.
    """
    if not isinstance(plane, Plane) or not isinstance(cone, Cone):
        raise GeometryError("analytic plane/cone intersection needs a Plane and a Cone")
    first = RuledSupport.from_surface(cone)
    plan = get_plan(first, plane)
    a_hp, b_hp, c_hp = plan.hp
    if c_hp.is_zero():
        return _plane_through_apex(plane, cone, plan, tolerance, cancellation_check)
    if not plan.pole_roots():
        return _plane_cone_ellipse(plane, cone, plan, tolerance, cancellation_check)
    return charts_support(cone, plane, tolerance=tolerance, cancellation_check=cancellation_check)


def _plane_through_apex(plane, cone, plan, tolerance, check):
    """The plane contains the apex: the section is the generators it contains (or just the apex)."""
    first = plan.first
    start, sweep = float(cone.start_angle), float(cone.sweep_angle)
    lo, hi = sorted((start, start + sweep))
    tol = tolerance / max(1.0, _max_radius(cone))
    roots = plan.pole_roots()                                   # B(theta) = 0: generators inside the plane
    segments = []
    fs = plan.floats()
    z_lo, z_hi = _axial_range(cone)
    for angle in _in_range(roots, lo, hi, tol):
        a = np.asarray(first.axis)
        c, s = math.cos(angle), math.sin(angle)
        rows = fs.d_rows
        direction = np.asarray(rows[0]) + c * np.asarray(rows[1]) + s * np.asarray(rows[2])
        ring = np.asarray(first.origin) + first.radius * (c * np.asarray(first.radial_direction)
                                                          + s * np.asarray(first.circumferential_direction))
        start_point, end_point = ring + z_lo * direction, ring + z_hi * direction
        if np.linalg.norm(end_point - start_point) <= tolerance:
            continue
        if any(np.linalg.norm(np.asarray(a) - start_point) <= tolerance and np.linalg.norm(np.asarray(b) - end_point) <= tolerance
               for a, b in segments):
            continue                                            # the seam of a full turn lists its generator twice
        segments.append((tuple(start_point), tuple(end_point)))
    points = []
    if not segments:
        apex = np.asarray(first.origin) - float(first.radius / first.slope) * np.asarray(first.axis)
        if z_lo - tolerance <= float((apex - np.asarray(first.origin)) @ np.asarray(first.axis)) <= z_hi + tolerance:
            points.append(tuple(apex))
    return SupportIntersection(segments=tuple(segments), points=tuple(points))


def _plane_cone_ellipse(plane, cone, plan, tolerance, check):
    """A plane crossing every generator of a nappe: build the ellipse exactly and cut it at the events."""
    first = plan.first
    origin = np.asarray(first.origin)
    axis = np.asarray(first.axis)
    e1 = np.asarray(first.radial_direction)
    e2 = np.asarray(first.circumferential_direction)
    radius, slope = first.radius, first.slope
    normal = np.asarray(plane.normal)
    offset = float(normal @ (np.asarray(plane.origin) - origin))
    # in-plane orthonormal basis
    seed = e1 - (e1 @ normal) * normal
    if np.linalg.norm(seed) < 1e-3:
        seed = e2 - (e2 @ normal) * normal
    w_u = seed / np.linalg.norm(seed)
    w_v = np.cross(normal, w_u)
    basis = np.column_stack((w_u, w_v))
    m_full = np.eye(3) - (1 + slope * slope) * np.outer(axis, axis)
    l_full = -radius * slope * axis
    c_full = -radius * radius
    p0 = offset * normal
    m2 = basis.T @ m_full @ basis
    l2 = basis.T @ (m_full @ p0 + l_full)
    c2 = float(p0 @ m_full @ p0 + 2 * l_full @ p0 + c_full)
    center2 = -np.linalg.solve(m2, l2)
    q0 = c2 + float(l2 @ center2)
    eigenvalues, vectors = np.linalg.eigh(m2)
    if np.any(eigenvalues * (-q0) <= 0):
        return SupportIntersection()                              # no real ellipse for this nappe pair
    semi = np.sqrt(-q0 / eigenvalues)
    center = origin + p0 + basis @ center2
    u_vec = semi[0] * (basis @ vectors[:, 0])
    v_vec = semi[1] * (basis @ vectors[:, 1])
    if np.cross(u_vec, v_vec) @ normal < 0:
        v_vec = -v_vec
    # event points from the exact branch events, mapped to the ellipse parameter
    start, sweep = float(cone.start_angle), float(cone.sweep_angle)
    lo, hi = sorted((start, start + sweep))
    tol = tolerance / max(1.0, _max_radius(cone))
    fs = plan.floats()
    thetas = [lo, hi]
    for z in (0.0, float(cone.height)):
        thetas.extend(_in_range(plan.bound_roots(z), lo, hi, tol))
    u2, v2 = float(u_vec @ u_vec), float(v_vec @ v_vec)

    def psi_of(point):
        rel = point - center
        return math.atan2(float(rel @ v_vec) / v2, float(rel @ u_vec) / u2)

    psis = sorted((psi_of(_point_at(first, fs, float(t), -fs.abc_scalar(float(t))[2] / fs.abc_scalar(float(t))[1]))
                   % TWO_PI) for t in thetas)
    full_turn = abs(abs(sweep) - TWO_PI) <= tol
    if full_turn and len(set(round(p, 9) for p in psis)) < 2:
        psis = sorted({*psis, (psis[0] + math.pi) % TWO_PI}) if psis else [0.0, math.pi]
    merged = _merge(psis, tol)
    if len(merged) == 1:
        merged = [merged[0], (merged[0] + math.pi) % TWO_PI]
        merged.sort()
    z_lo, z_hi = _axial_range(cone)
    curves = []
    for index, begin in enumerate(merged):
        end = merged[(index + 1) % len(merged)]
        length = end - begin if end > begin else end + TWO_PI - begin
        if length <= tol:
            continue
        mid = begin + .5 * length
        point = center + u_vec * math.cos(mid) + v_vec * math.sin(mid)
        axial = float((point - origin) @ axis)
        if not (z_lo - tolerance <= axial <= z_hi + tolerance):
            continue
        if not full_turn:
            theta = float(first.angle_of(point))
            if not any(lo - tol <= theta + TWO_PI * k <= hi + tol
                       for k in range(math.ceil((lo - tol - theta) / TWO_PI), math.floor((hi + tol - theta) / TWO_PI) + 1)):
                continue
        curves.append(EllipticArc(tuple(center), tuple(u_vec), tuple(v_vec), begin, length))
    return SupportIntersection(curves=tuple(curves))


# ---------------------------------------------------------------------------
# Cone meeting a Cylinder or Cone


def _coaxial(a, b, tolerance):
    """``(sigma, offset)`` when the two ruled supports share an axis line, else ``None``."""
    ax, bx = np.asarray(a.axis), np.asarray(b.axis)
    if np.linalg.norm(np.cross(ax, bx)) > 64 * _EPS:
        return None
    delta = np.asarray(b.origin) - np.asarray(a.origin)
    along = float(delta @ ax)
    if np.linalg.norm(delta - along * ax) > tolerance:
        return None
    return (1.0 if float(ax @ bx) > 0 else -1.0), along


def _slope(surface):
    if isinstance(surface, Cone):
        return (float(surface.radius_end) - float(surface.radius_start)) / float(surface.height)
    return 0.0


def _ring_radius(surface):
    return float(surface.radius_start) if isinstance(surface, Cone) else float(surface.radius)


def _coaxial_rings(a, b, relation, tolerance):
    """Rings where two coaxial ruled supports meet (``a`` is a Cone), as arcs over ``a``'s angular range."""
    sigma, along = relation
    ka, kb = _slope(a), _slope(b)
    ra, rb = _ring_radius(a), _ring_radius(b)
    denominator = ka - kb * sigma
    constant = ra - rb + kb * sigma * along
    scale = max(1.0, ra, rb)
    if abs(denominator) <= 64 * _EPS * max(1.0, abs(ka), abs(kb)):
        return SupportIntersection(coincident=abs(constant) <= tolerance)
    z = -constant / denominator
    radius = ra + ka * z
    if radius <= tolerance:
        return SupportIntersection()
    za_lo, za_hi = _axial_range(a)
    zb = sigma * (z - along)
    zb_lo, zb_hi = _axial_range(b)
    if not (za_lo - tolerance <= z <= za_hi + tolerance and zb_lo - tolerance <= zb <= zb_hi + tolerance):
        return SupportIntersection()
    center = np.asarray(a.origin) + z * np.asarray(a.axis)
    arc = EllipticArc(tuple(center), tuple(radius * np.asarray(a.radial_direction)),
                      tuple(radius * np.asarray(a.circumferential_direction)),
                      float(a.start_angle), float(a.sweep_angle))
    return SupportIntersection(curves=(arc,))


_FIRST_CHOICE = {}


def _prefer_first(a, b):
    """Which of two ruled supports should supply the angle: the one whose branches fold least."""
    key = (RuledSupport.from_surface(a), QuadricSupport.from_surface(a), RuledSupport.from_surface(b),
           QuadricSupport.from_surface(b))
    if key not in _FIRST_CHOICE:
        costs = []
        for first, second in ((a, b), (b, a)):
            plan = get_plan(RuledSupport.from_surface(first), second)
            costs.append(sum(m for _a, m in plan.discriminant_roots()) + 2 * len(plan.pole_roots()))
        _FIRST_CHOICE[key] = costs[0] <= costs[1]
        if len(_FIRST_CHOICE) > 4096:
            _FIRST_CHOICE.pop(next(iter(_FIRST_CHOICE)))
    return _FIRST_CHOICE[key]


def cone_support(a, b, *, tolerance=1e-10, cancellation_check=None):
    """Exact support intersection for a pair that includes a Cone.

    ``a`` and ``b`` may be given in either order. The result is a
    :class:`~anygeometry.analytic_supports.SupportIntersection`.
    """
    if isinstance(b, Cone) and not isinstance(a, Cone):
        a, b = b, a
    if not isinstance(a, Cone):
        raise GeometryError("cone_support needs a Cone")
    if isinstance(b, Plane):
        return plane_cone_support(b, a, tolerance=tolerance, cancellation_check=cancellation_check)
    if not isinstance(b, (Cylinder, Cone)):
        raise GeometryError("a Cone meets a Plane, Cylinder or Cone")
    relation = _coaxial(RuledSupport.from_surface(a), RuledSupport.from_surface(b), tolerance)
    if relation is not None:
        return _coaxial_rings(a, b, relation, tolerance)
    first, second = (a, b) if _prefer_first(a, b) else (b, a)
    return charts_support(first, second, tolerance=tolerance, cancellation_check=cancellation_check)
