"""Face moments in a frame, optionally clipped below a plane (generality slice H1a).

A face contributes ``int phi(P) (n . N) dA`` over the face (or its part below the frame's plane
``zeta = 0``), where ``(xi, eta, zeta)`` are coordinates in an orthonormal frame with unit normal
``n`` (third axis), ``N`` is the face's area normal oriented by its loop, and ``phi`` is a vector of
fields of the frame coordinates. With ``phi = (zeta, xi zeta, eta zeta, zeta^2 / 2)`` over a closed
outward shell this is the volume and its first moments (divergence theorem, ``F = phi n``); clipped
below ``zeta = 0`` it is the submerged volume, because ``F`` vanishes on the waterplane. With
``phi = (1, xi, eta, xi^2, xi eta, eta^2)`` (divergence-free fields) the clipped sums are minus the
waterplane area and its moments (H1b).

Every face is a chart: a NURBS face its own chart; a plane face the plane's own ``(u, v)`` frame,
in any orientation. Green's theorem in the chart, with an inner coordinate ``s`` and an outer ``w``
(``(u, v)`` or, swapped, ``(v, u)``), gives ``int int g ds dw = oint G dw`` with ``g = phi(S) (n .
(S_s x S_w))`` and ``G(s, w) = int_0^s g(r, w) H(-zeta) dr``. The inner Gauss-Legendre rule is split
at the chart's knots in ``s`` and, clipped, at the iso-line's roots of ``zeta`` (plane: the linear
root; NURBS: the roots of the polynomial ``zeta W`` per knot span). Clipped, ``G`` is only piecewise
smooth along the boundary, so the boundary rule is split at knots, arc pieces, the edges' crossings
of the plane and the levels ``w*`` where the waterline crosses the base line ``s = 0`` (there the
iso-line's root leaves ``[0, s]``). The inner coordinate is the one along which ``zeta`` varies most
over the face (the waterline then crosses the iso-lines transversally; a waterline turning back in
``w`` inside the face leaves a square-root kink that the order-doubling estimate reports).
Values are quadratures (the caller compares two orders), not rigorous bounds.

Boundaries: straight lines and plane x NURBS section edges on plane faces; section edges and
straight chart segments on NURBS faces. Anything else refuses (``GeometryError``).
"""
from __future__ import annotations

import numpy as np

from .curves import Straight
from .errors import GeometryError
from .support_curves import PlaneNurbsCurve


class Frame:
    """An orthonormal frame: ``origin``, longitudinal ``e1``, transverse ``e2`` and normal ``n``
    (``e1 x e2 = n``); ``local(points)`` gives ``(xi, eta, zeta)``."""

    def __init__(self, origin, e1, e2):
        e1 = np.asarray(e1, dtype=float)
        e2 = np.asarray(e2, dtype=float)
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 - (e2 @ e1) * e1
        norm = float(np.linalg.norm(e2))
        if not norm > 0 or not np.all(np.isfinite(e1)):
            raise GeometryError("a frame needs two independent directions")
        e2 = e2 / norm
        self.origin = np.asarray(origin, dtype=float)
        self.axes = np.stack((e1, e2, np.cross(e1, e2)))

    @classmethod
    def world(cls, origin=(0., 0., 0.)) -> "Frame":
        return cls(origin, (1., 0., 0.), (0., 1., 0.))

    @property
    def normal(self) -> np.ndarray:
        return self.axes[2]

    def local(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.axes.T

    def zeta(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.axes[2]


def volume_fields(local) -> np.ndarray:
    """``zeta, xi zeta, eta zeta, zeta^2 / 2``: volume and its first moments (``xi, eta, zeta``)."""
    xi, eta, zeta = local[:, 0], local[:, 1], local[:, 2]
    return np.stack((zeta, xi * zeta, eta * zeta, .5 * zeta * zeta), axis=1)


def waterplane_fields(local) -> np.ndarray:
    """``1, xi, eta, xi^2, xi eta, eta^2``: minus the waterplane area and its moments when clipped."""
    xi, eta = local[:, 0], local[:, 1]
    return np.stack((np.ones_like(xi), xi, eta, xi * xi, xi * eta, eta * eta), axis=1)


# ------------------------------------------------------------------ charts (in (u, v))


def _span_roots(function, degree, cuts, lo, hi):
    """Roots in ``(lo, hi)`` of a function that is a polynomial of ``degree`` between ``cuts``
    (interpolated at Chebyshev points per piece, real roots kept)."""
    found = []
    for a, b in zip(cuts, cuts[1:]):
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        if degree == 0:
            continue
        k = degree + 1
        x = np.cos((2 * np.arange(k) + 1) * np.pi / (2 * k))
        h = function(.5 * (a + b) + .5 * (b - a) * x)
        scale = float(np.max(np.abs(h)))
        if not scale > 0:
            continue
        coefficients = np.polynomial.chebyshev.chebfit(x, h / scale, degree)
        for root in np.polynomial.chebyshev.chebroots(coefficients):
            if abs(root.imag) <= 1e-12 and -1 < root.real < 1:
                t = .5 * (a + b) + .5 * (b - a) * float(root.real)
                if lo < t < hi:
                    found.append(t)
    return sorted(found)


class _PlaneChart:
    def __init__(self, plane, frame):
        self.origin = np.asarray(plane.origin, dtype=float)
        self.u = np.asarray(plane.u_vector, dtype=float)
        self.v = np.asarray(plane.v_vector, dtype=float)
        self.inverse = np.linalg.pinv(np.stack((self.u, self.v), axis=1))      # 2 x 3
        n = frame.normal
        self.depth = (float(frame.zeta(self.origin[None])[0]), float(n @ self.u), float(n @ self.v))
        self.knots = ((), ())

    def to_chart(self, points, rates):
        return (np.asarray(points) - self.origin) @ self.inverse.T, np.asarray(rates) @ self.inverse.T

    def evaluate(self, uv):
        return self.origin + uv[:, :1] * self.u + uv[:, 1:] * self.v

    def jacobian(self, uv, n):
        return np.full(len(uv), float(n @ np.cross(self.u, self.v)))

    def gradient_scale(self, axis):
        return abs(self.depth[1 + axis])

    def roots(self, axis, fixed, lo, hi):
        """Roots of zeta along the line where coordinate ``axis`` runs and the other is ``fixed``."""
        c0, *c = self.depth
        slope, other = c[axis], c[1 - axis]
        if slope == 0.0:
            return []
        root = -(c0 + other * fixed) / slope
        return [root] if lo < root < hi else []


class _NurbsChart:
    def __init__(self, surface, frame):
        from .nurbs import NurbsSurface
        self.surface = surface
        knots = []
        for values, (r0, r1) in ((surface.knots_u, surface.u_range), (surface.knots_v, surface.v_range)):
            knots.append(tuple(sorted({(k - r0) / (r1 - r0) for k in values if 0 < (k - r0) / (r1 - r0) < 1})))
        self.knots = tuple(knots)
        self.degrees = (surface.degree_u, surface.degree_v)
        # zeta W as a polynomial B-spline surface: controls w (n . P - n . o), unweighted
        n, d = frame.normal, float(frame.normal @ frame.origin)
        weights = surface.weights or tuple((1.,) * len(row) for row in surface.controls)
        heights = tuple(tuple((w * (float(n @ np.asarray(p)) - d), 0., 0.) for p, w in zip(row, wrow))
                        for row, wrow in zip(surface.controls, weights))
        self.height = NurbsSurface(surface.degree_u, surface.degree_v, surface.knots_u, surface.knots_v, heights,
                                   None, surface.u_range, surface.v_range)

    def evaluate(self, uv):
        return self.surface.evaluate_many(uv)

    def jacobian(self, uv, n):
        s_u, s_v = self.surface.derivatives_many(uv)
        return np.cross(s_u, s_v) @ n

    def gradient_scale(self, axis):
        grid = np.linspace(0., 1., 9)
        uv = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
        h = self.height.evaluate_many(uv)[:, 0]
        rates = self.height.derivatives_many(uv)[axis][:, 0]
        # weight by nearness to the waterline: where |zeta W| is small relative to its range
        span = float(np.max(np.abs(h))) or 1.0
        near = np.exp(-(h / (.1 * span)) ** 2)
        return float(np.sum(near * np.abs(rates)) / max(float(np.sum(near)), 1e-300))

    def roots(self, axis, fixed, lo, hi):
        def function(t):
            uv = np.empty((len(t), 2))
            uv[:, axis], uv[:, 1 - axis] = t, fixed
            return self.height.evaluate_many(uv)[:, 0]
        return _span_roots(function, self.degrees[axis], [0.0, *self.knots[axis], 1.0], lo, hi)


def _chart(face, frame):
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    if isinstance(face.surface, Plane):
        return _PlaneChart(face.surface, frame)
    if isinstance(face.surface, NurbsSurface):
        return _NurbsChart(face.surface, frame)
    raise GeometryError(f"face {face.id}: only plane and NURBS faces have face moments")


# ------------------------------------------------------------------ boundaries


def _section_float(curve, t):
    """Float points of a plane x NURBS section edge for quadrature: ``(chart, chart rates, world, world
    rates)`` at parameters ``t``. The same arcs as the exact definition (slicing coordinate affine in
    ``t``); the dependent knot coordinate by a vectorised float bisection of the plane function over
    the arc's box, polished by Newton. Points whose box shows no sign change (inside an end root's
    interval) fall back to the exact evaluation."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    arcs = curve._arcs()
    count = len(arcs)
    a, b = (float(value) for value in curve.domain)
    scaled = (a + t * (b - a)) * count
    index = np.clip(np.floor(scaled).astype(int), 0, count - 1)
    tau = np.clip(scaled - index, 0.0, 1.0)
    m, d, *_rest = curve._plane_slab()
    m, d = np.array([float(x) for x in m]), float(d)
    (r0, r1), (q0, q1) = (tuple(map(float, item)) for item in curve._chart_ranges())
    surface = curve.surface
    knots = np.empty((len(t), 2))
    rates_knot = np.empty((len(t), 2))
    exact = np.zeros(len(t), dtype=bool)
    for k in np.unique(index):
        arc = arcs[k]
        rows = np.flatnonzero(index == k)
        s0, s1 = float(arc.s0), float(arc.s1)
        s = s0 + tau[rows] * (s1 - s0)
        box = arc.box
        x0, x1 = (float(box.v_range[0]), float(box.v_range[1])) if arc.slicing == 0 else             (float(box.u_range[0]), float(box.u_range[1]))

        def plane(x, s=s, arc=arc, rows=rows):
            knot = np.stack((s, x), axis=1) if arc.slicing == 0 else np.stack((x, s), axis=1)
            chart = np.stack(((knot[:, 0] - r0) / (r1 - r0), (knot[:, 1] - q0) / (q1 - q0)), axis=1)
            return surface.evaluate_many(np.clip(chart, 0.0, 1.0)) @ m - d
        lo, hi = np.full(len(rows), x0), np.full(len(rows), x1)
        f_lo, f_hi = plane(lo), plane(hi)
        bad = (f_lo > 0) == (f_hi > 0)
        bad &= (f_lo != 0) & (f_hi != 0)
        for _ in range(12):
            middle = .5 * (lo + hi)
            f_mid = plane(middle)
            left = (f_mid > 0) == (f_lo > 0)
            lo, f_lo = np.where(left, middle, lo), np.where(left, f_mid, f_lo)
            hi = np.where(left, hi, middle)
        x = .5 * (lo + hi)
        axis = 1 if arc.slicing == 0 else 0
        scale = (q1 - q0) if axis == 1 else (r1 - r0)
        for _ in range(5):                                      # Newton inside the bracket
            knot = np.stack((s, x), axis=1) if arc.slicing == 0 else np.stack((x, s), axis=1)
            chart = np.stack(((knot[:, 0] - r0) / (r1 - r0), (knot[:, 1] - q0) / (q1 - q0)), axis=1)
            chart = np.clip(chart, 0.0, 1.0)
            value = surface.evaluate_many(chart) @ m - d
            slope = (surface.derivatives_many(chart)[axis] @ m) / scale
            step = np.divide(value, slope, out=np.zeros_like(value), where=slope != 0)
            candidate = x - step
            x = np.clip(candidate, np.minimum(lo, hi), np.maximum(lo, hi))
        knot = np.stack((s, x), axis=1) if arc.slicing == 0 else np.stack((x, s), axis=1)
        chart = np.stack(((knot[:, 0] - r0) / (r1 - r0), (knot[:, 1] - q0) / (q1 - q0)), axis=1)
        s_u, s_v = surface.derivatives_many(np.clip(chart, 0.0, 1.0))
        g_u, g_v = (s_u @ m) / (r1 - r0), (s_v @ m) / (q1 - q0)          # knot-coordinate partials
        g_s, g_x = (g_u, g_v) if arc.slicing == 0 else (g_v, g_u)
        ds = (s1 - s0) * count * (b - a)
        dx = -g_s / np.where(g_x == 0, np.inf, g_x) * ds
        du, dv = (np.full(len(rows), ds), dx) if arc.slicing == 0 else (dx, np.full(len(rows), ds))
        knots[rows] = knot
        rates_knot[rows] = np.stack((du, dv), axis=1)
        exact[rows] = bad | (g_x == 0)
    chart = np.clip(np.stack(((knots[:, 0] - r0) / (r1 - r0), (knots[:, 1] - q0) / (q1 - q0)), axis=1), 0.0, 1.0)
    chart_rates = np.stack((rates_knot[:, 0] / (r1 - r0), rates_knot[:, 1] / (q1 - q0)), axis=1)
    world = surface.evaluate_many(chart)
    s_u, s_v = surface.derivatives_many(chart)
    world_rates = s_u * chart_rates[:, :1] + s_v * chart_rates[:, 1:]
    for i in np.flatnonzero(exact):
        chart[i], chart_rates[i] = curve.chart(t[i]), curve.chart_derivative(t[i])
        world[i], world_rates[i] = curve.evaluate(t[i]), curve.derivative(t[i])
    return chart, chart_rates, world, world_rates


def _edge_map(model, item, chart):
    """``(function, breakpoints)``: ``function(t)`` gives the chart points ``(u, v)``, chart rates and
    world points along the oriented edge for ``t`` in ``[0, 1]`` (loop direction); breakpoints in t."""
    from .nurbs import CurveOnSurface
    edge = model.edges[item.edge]
    curve = edge.curve
    flip = not item.forward

    def oriented(raw, breaks):
        def function(t):
            t = np.asarray(t, dtype=float)
            points, rates, world = raw(1.0 - t if flip else t)
            return points, (-rates if flip else rates), world
        return function, sorted({1.0 - b if flip else b for b in breaks if 0.0 < b < 1.0})

    if isinstance(chart, _PlaneChart):
        if isinstance(curve, Straight):
            a, b = model.vertices[edge.start].position, model.vertices[edge.end].position

            def raw(t):
                world = a + t[:, None] * (b - a)
                points, rates = chart.to_chart(world, np.tile(b - a, (len(t), 1)))
                return points, rates, world
            return oriented(raw, [])
        if isinstance(curve, PlaneNurbsCurve):
            def raw(t):
                _c, _r, world, rates = _section_float(curve, t)
                points, chart_rates = chart.to_chart(world, rates)
                return points, chart_rates, world
            return oriented(raw, curve.breakpoints())
        raise GeometryError(f"edge {item.edge}: only straight and section edges bound plane faces here")
    if isinstance(curve, PlaneNurbsCurve):
        def raw(t):
            points, rates, world, _w = _section_float(curve, t)
            return points, rates, world
        return oriented(raw, curve.breakpoints())
    if isinstance(curve, CurveOnSurface) and curve.pcurve.degree == 1 and len(curve.pcurve.controls) == 2:
        p0, p1 = (np.asarray([float(v) for v in curve.pcurve.exact_point(s)]) for s in (0, 1))
        surface = chart.surface
        cuts = set()
        for axis in (0, 1):
            if p1[axis] != p0[axis]:
                cuts.update(c for c in ((k - p0[axis]) / (p1[axis] - p0[axis]) for k in chart.knots[axis])
                            if 0.0 < c < 1.0)

        def raw(t):
            points = p0 + t[:, None] * (p1 - p0)
            return points, np.tile(p1 - p0, (len(t), 1)), surface.evaluate_many(points)
        return oriented(raw, cuts)
    raise GeometryError(f"edge {item.edge}: only section edges and straight chart segments bound NURBS faces here")


def _sign_changes(value, lo, hi, samples: int = 64):
    """Parameters in ``(lo, hi)`` where the scalar ``value(t)`` changes sign or vanishes at a sample
    (sampled, refined by bisection; a touch without a sign change between samples is not found)."""
    t = np.linspace(lo, hi, samples + 1)
    f = value(t)
    found = [float(t[k]) for k in range(1, samples) if f[k] == 0.0]
    brackets = np.flatnonzero(f[:-1] * f[1:] < 0)
    if len(brackets):
        a, b, fa, fb = t[brackets], t[brackets + 1], f[brackets], f[brackets + 1]
        for step in range(40):                                  # all brackets at once
            # bisection first, then false position kept inside the bracket (superlinear on smooth f)
            secant = a - fa * (b - a) / np.where(fb != fa, fb - fa, 1.0)
            m = .5 * (a + b) if step < 8 else np.clip(secant, np.minimum(a, b), np.maximum(a, b))
            fm = value(m)
            left = (fm < 0) == (fa < 0)
            a, fa, b, fb = (np.where(left, m, a), np.where(left, fm, fa), np.where(left, b, m),
                            np.where(left, fb, fm))
            if np.all((fm == 0) | (np.abs(b - a) <= 4 * np.finfo(float).eps * np.maximum(1.0, np.abs(a)))) or                     (step >= 8 and np.all(np.abs(fm) <= 1e-15 * np.maximum(np.abs(f).max(), 1e-300))):
                break
        found += [float(x) for x in np.where(np.abs(fa) <= np.abs(fb), a, b)]
    return found


def _boundary_pieces(model, face, chart):
    pieces = []
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            function, breaks = _edge_map(model, item, chart)
            pieces.append((function, sorted({0.0, 1.0, *breaks})))
    return pieces


# ------------------------------------------------------------------ moments


def face_moments(model, face_id: int, frame: Frame, fields=volume_fields, order: int = 24,
                 clip: bool = False) -> np.ndarray:
    """``int phi (n . N) dA`` over the face (``clip``: its part with ``zeta < 0``), ``phi = fields``."""
    face = model.faces[face_id]
    chart = _chart(face, frame)
    n = frame.normal
    nodes, weights = np.polynomial.legendre.leggauss(order)
    pieces = _boundary_pieces(model, face, chart)
    # the inner coordinate s (axis) and the outer w: zeta varies most along s
    axis = 0
    if clip:
        sample = np.vstack([function(np.linspace(0., 1., 17))[0] for function, _b in pieces])
        extent = np.ptp(sample, axis=0)
        scores = [chart.gradient_scale(k) * float(extent[k]) for k in (0, 1)]
        axis = 0 if scores[0] >= scores[1] else 1
    other = 1 - axis
    # boundary cuts: knots/arcs; clipped also the plane crossings and the base-line levels w*
    nodes_out, rates_out, ws_out = [], [], []
    levels = []
    if clip:
        sample = np.vstack([function(np.linspace(0., 1., 17))[0] for function, _b in pieces])
        w_lo, w_hi = float(np.min(sample[:, other])), float(np.max(sample[:, other]))
        margin = 1e-9 * max(1.0, w_hi - w_lo)
        levels = chart.roots(other, 0.0, w_lo - margin, w_hi + margin)
    for function, cuts in pieces:
        if clip:
            extra = []
            for lo, hi in zip(cuts, cuts[1:]):
                extra += _sign_changes(lambda t: frame.zeta(function(t)[2]), lo, hi)
                for level in levels:
                    extra += _sign_changes(lambda t, level=level: function(t)[0][:, other] - level, lo, hi)
            cuts = sorted(set(cuts) | set(extra))
        for lo, hi in zip(cuts, cuts[1:]):
            if hi <= lo:
                continue
            t = .5 * (hi - lo) * (nodes + 1.) + lo
            p, r, _w = function(t)
            nodes_out.append(p)
            rates_out.append(r)
            ws_out.append(.5 * (hi - lo) * weights)
    points, rates, ws = np.vstack(nodes_out), np.vstack(rates_out), np.concatenate(ws_out)
    # Green: oint G dw with G = int_0^s g dr along the inner coordinate; swapping (u, v) reverses the
    # orientation, so the outer integral changes sign (int int g du dv = -oint G_v du)
    width = fields(np.zeros((1, 3))).shape[1]
    inner = np.zeros((len(points), width))
    knots = chart.knots[axis]
    for k, point in enumerate(points):
        s, w = float(point[axis]), float(point[other])
        low, high = min(0.0, s), max(0.0, s)
        cuts = {low, high, *(c for c in knots if low < c < high)}
        if clip:
            cuts.update(chart.roots(axis, w, low, high))
        cuts = sorted(cuts)
        value = np.zeros(width)
        for lo, hi in zip(cuts, cuts[1:]):
            if hi <= lo:
                continue
            r = .5 * (hi - lo) * (nodes + 1.) + lo
            uv = np.empty((order, 2))
            uv[:, axis], uv[:, other] = r, w
            if clip:
                middle = np.empty((1, 2))
                middle[0, axis], middle[0, other] = .5 * (lo + hi), w
                if frame.zeta(chart.evaluate(middle))[0] >= 0:
                    continue
            g = fields(frame.local(chart.evaluate(uv))) * chart.jacobian(uv, n)[:, None]
            value += (.5 * (hi - lo) * weights) @ g
        inner[k] = value if s >= 0 else -value
    total = ws @ (inner * rates[:, other][:, None])
    return total if axis == 0 else -total


def shell_moments(model, faces, frame: Frame, fields=volume_fields, order: int = 24, clip: bool = False):
    total = None
    for face_id in faces:
        value = face_moments(model, face_id, frame, fields, order, clip)
        total = value if total is None else total + value
    return total
