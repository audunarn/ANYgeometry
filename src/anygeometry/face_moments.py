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
in any orientation. Green's theorem in the chart gives ``int int g du dv = oint G dv`` with
``g = phi(S) (n . (S_u x S_v))`` and ``G(u, v) = int_{0}^{u} g(s, v) H(-zeta) ds``. The inner
Gauss-Legendre rule is split at the chart's u knots and, clipped, at the iso-line's roots of
``zeta`` (plane: the linear root; NURBS: the roots of the polynomial ``zeta W`` per knot span); the
boundary rule is split at knots, arc pieces and, clipped, at the edges' crossings of the plane.
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


# ------------------------------------------------------------------ charts


class _PlaneChart:
    def __init__(self, plane, frame):
        self.origin = np.asarray(plane.origin, dtype=float)
        self.u = np.asarray(plane.u_vector, dtype=float)
        self.v = np.asarray(plane.v_vector, dtype=float)
        self.inverse = np.linalg.pinv(np.stack((self.u, self.v), axis=1))      # 2 x 3
        self.u_knots = ()
        n = frame.normal
        self.depth = (float(frame.zeta(self.origin[None])[0]), float(n @ self.u), float(n @ self.v))

    def to_chart(self, points, rates):
        return (np.asarray(points) - self.origin) @ self.inverse.T, np.asarray(rates) @ self.inverse.T

    def evaluate(self, uv):
        return self.origin + uv[:, :1] * self.u + uv[:, 1:] * self.v

    def jacobian(self, uv, n):
        return np.full(len(uv), float(n @ np.cross(self.u, self.v)))

    def roots(self, v, lo, hi):
        """Roots of zeta along the iso-line ``v`` in ``(lo, hi)``."""
        c0, cu, cv = self.depth
        if cu == 0.0:
            return []
        root = -(c0 + cv * v) / cu
        return [root] if lo < root < hi else []


class _NurbsChart:
    def __init__(self, surface, frame):
        from .nurbs import NurbsSurface
        self.surface = surface
        (r0, r1) = surface.u_range
        self.u_knots = tuple(sorted({(k - r0) / (r1 - r0) for k in surface.knots_u if 0 < (k - r0) / (r1 - r0) < 1}))
        # zeta W as a polynomial B-spline surface: controls w (n . P - n . o), unweighted
        n, d = frame.normal, float(frame.normal @ frame.origin)
        weights = surface.weights or tuple((1.,) * len(row) for row in surface.controls)
        heights = tuple(tuple((w * (float(n @ np.asarray(p)) - d), 0., 0.) for p, w in zip(row, wrow))
                        for row, wrow in zip(surface.controls, weights))
        self.height = NurbsSurface(surface.degree_u, surface.degree_v, surface.knots_u, surface.knots_v, heights,
                                   None, surface.u_range, surface.v_range)
        self.degree = surface.degree_u

    def evaluate(self, uv):
        return self.surface.evaluate_many(uv)

    def jacobian(self, uv, n):
        s_u, s_v = self.surface.derivatives_many(uv)
        return np.cross(s_u, s_v) @ n

    def roots(self, v, lo, hi):
        """Roots of ``zeta W`` along the iso-line ``v`` in ``(lo, hi)``: per knot span a polynomial of
        degree ``degree_u`` (interpolated at Chebyshev points, real roots kept)."""
        cuts = [0.0, *self.u_knots, 1.0]
        found = []
        for a, b in zip(cuts, cuts[1:]):
            if b <= lo or a >= hi:
                continue
            k = self.degree + 1
            x = np.cos((2 * np.arange(k) + 1) * np.pi / (2 * k))                # in (-1, 1)
            s = .5 * (a + b) + .5 * (b - a) * x
            h = self.height.evaluate_many(np.stack((s, np.full_like(s, v)), axis=1))[:, 0]
            scale = float(np.max(np.abs(h))) or 1.0
            coefficients = np.polynomial.chebyshev.chebfit(x, h / scale, self.degree)
            for root in np.polynomial.chebyshev.chebroots(coefficients) if self.degree else []:
                if abs(root.imag) <= 1e-12 and -1 - 1e-12 <= root.real <= 1 + 1e-12:
                    t = .5 * (a + b) + .5 * (b - a) * float(root.real)
                    if lo < t < hi:
                        found.append(t)
        return sorted(found)


def _chart(face, frame):
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    if isinstance(face.surface, Plane):
        return _PlaneChart(face.surface, frame)
    if isinstance(face.surface, NurbsSurface):
        return _NurbsChart(face.surface, frame)
    raise GeometryError(f"face {face.id}: only plane and NURBS faces have face moments")


# ------------------------------------------------------------------ boundaries


def _edge_map(model, item, chart):
    """``(function, breakpoints)``: ``function(t)`` gives the chart points, chart rates and world points
    along the oriented edge for ``t`` in ``[0, 1]`` (loop direction)."""
    from .nurbs import CurveOnSurface
    edge = model.edges[item.edge]
    curve = edge.curve
    flip = not item.forward

    def oriented(raw):
        def function(t):
            t = np.asarray(t, dtype=float)
            points, rates, world = raw(1.0 - t if flip else t)
            return points, (-rates if flip else rates), world
        return function

    if isinstance(chart, _PlaneChart):
        if isinstance(curve, Straight):
            a, b = model.vertices[edge.start].position, model.vertices[edge.end].position

            def raw(t):
                world = a + t[:, None] * (b - a)
                points, rates = chart.to_chart(world, np.tile(b - a, (len(t), 1)))
                return points, rates, world
            return oriented(raw), []
        if isinstance(curve, PlaneNurbsCurve):
            def raw(t):
                world, rates = curve.evaluate(t), curve.derivative(t)
                points, chart_rates = chart.to_chart(world, rates)
                return points, chart_rates, world
            return oriented(raw), sorted(set(curve.breakpoints()))
        raise GeometryError(f"edge {item.edge}: only straight and section edges bound plane faces here")
    if isinstance(curve, PlaneNurbsCurve):
        def raw(t):
            return curve.chart(t), curve.chart_derivative(t), curve.evaluate(t)
        return oriented(raw), sorted(set(curve.breakpoints()))
    if isinstance(curve, CurveOnSurface) and curve.pcurve.degree == 1 and len(curve.pcurve.controls) == 2:
        p0, p1 = (np.asarray([float(v) for v in curve.pcurve.exact_point(s)]) for s in (0, 1))
        surface = chart.surface
        cuts = set()
        (q0, q1) = surface.v_range
        v_knots = [(k - q0) / (q1 - q0) for k in surface.knots_v]
        for axis, knots in ((0, chart.u_knots), (1, v_knots)):
            if p1[axis] != p0[axis]:
                cuts.update(c for c in ((k - p0[axis]) / (p1[axis] - p0[axis]) for k in knots) if 0.0 < c < 1.0)

        def raw(t):
            points = p0 + t[:, None] * (p1 - p0)
            return points, np.tile(p1 - p0, (len(t), 1)), surface.evaluate_many(points)
        return oriented(raw), sorted(cuts)
    raise GeometryError(f"edge {item.edge}: only section edges and straight chart segments bound NURBS faces here")


def _plane_crossings(function, lo, hi, frame, samples: int = 64):
    """Parameters in ``(lo, hi)`` where the edge crosses ``zeta = 0`` (sign changes of sampled
    ``zeta``, refined by bisection; touching without a sign change does not split)."""
    t = np.linspace(lo, hi, samples + 1)
    zeta = frame.zeta(function(t)[2])
    found = []
    for k in range(samples):
        a, b, za, zb = t[k], t[k + 1], zeta[k], zeta[k + 1]
        if za == 0.0 and lo < a:
            found.append(a)
        if za * zb < 0:
            for _ in range(80):
                m = .5 * (a + b)
                zm = frame.zeta(function(np.array([m]))[2])[0]
                if zm == 0.0 or m in (a, b):
                    break
                if (zm < 0) == (za < 0):
                    a, za = m, zm
                else:
                    b = m
            found.append(.5 * (a + b))
    return found


def _boundary_nodes(model, face, chart, frame, order, clip):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points, rates, ws = [], [], []
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            function, breaks = _edge_map(model, item, chart)
            if not item.forward:
                breaks = [1.0 - b for b in breaks]
            cuts = sorted({0.0, 1.0, *(b for b in breaks if 0.0 < b < 1.0)})
            if clip:
                extra = []
                for lo, hi in zip(cuts, cuts[1:]):
                    extra += _plane_crossings(function, lo, hi, frame)
                cuts = sorted(set(cuts) | set(extra))
            for lo, hi in zip(cuts, cuts[1:]):
                if hi <= lo:
                    continue
                t = .5 * (hi - lo) * (nodes + 1.) + lo
                p, r, _w = function(t)
                points.append(p)
                rates.append(r)
                ws.append(.5 * (hi - lo) * weights)
    return np.vstack(points), np.vstack(rates), np.concatenate(ws)


# ------------------------------------------------------------------ moments


def face_moments(model, face_id: int, frame: Frame, fields=volume_fields, order: int = 24,
                 clip: bool = False) -> np.ndarray:
    """``int phi (n . N) dA`` over the face (``clip``: its part with ``zeta < 0``), ``phi = fields``."""
    face = model.faces[face_id]
    chart = _chart(face, frame)
    n = frame.normal
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points, rates, ws = _boundary_nodes(model, face, chart, frame, order, clip)
    width = fields(np.zeros((1, 3))).shape[1]
    inner = np.zeros((len(points), width))
    for k, (u, v) in enumerate(points):
        low, high = min(0.0, u), max(0.0, u)
        cuts = {low, high, *(c for c in chart.u_knots if low < c < high)}
        if clip:
            cuts.update(chart.roots(v, low, high))
        cuts = sorted(cuts)
        value = np.zeros(width)
        for lo, hi in zip(cuts, cuts[1:]):
            if hi <= lo:
                continue
            if clip:
                middle = chart.evaluate(np.array([[.5 * (lo + hi), v]]))
                if frame.zeta(middle)[0] >= 0:
                    continue
            s = .5 * (hi - lo) * (nodes + 1.) + lo
            uv = np.stack((s, np.full_like(s, v)), axis=1)
            g = fields(frame.local(chart.evaluate(uv))) * chart.jacobian(uv, n)[:, None]
            value += (.5 * (hi - lo) * weights) @ g
        inner[k] = value if u >= 0 else -value
    return ws @ (inner * rates[:, 1][:, None])


def shell_moments(model, faces, frame: Frame, fields=volume_fields, order: int = 24, clip: bool = False):
    total = None
    for face_id in faces:
        value = face_moments(model, face_id, frame, fields, order, clip)
        total = value if total is None else total + value
    return total
