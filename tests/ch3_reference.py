"""Independent references for the CH3c curved query tests (no production code is imported or called).

Only the *data* of the B1 bow (control points, weights) is read from ``b1_example.b1_bow_surface``; everything
else is written here from the geometry:

* **Station closed forms** (``station``): for the upright plane ``z = h`` the section of the hull is
  ``R(h) = {x1 <= x <= x2, |y| <= y(x)}`` (the half hull: ``0 <= y``), bounded by the frames ``x = 1.5, 3.5``
  and the iso-``v`` conic arc ``S(u, h/4)``, which is a rational quadratic in ``u``. Its quantities are single
  ``mpmath`` integrals over ``u`` of ``y x'`` with weights; ``V`` and ``J`` are ``mpmath`` integrals of the
  station areas over ``z``. (``v = z / 4`` because the bow's control rows lie at ``z = 0`` and ``z = 4``.)
* **Independent slice integration** (``slice_volume_stats``, ``slice_cap_stats``): for any plane ``n . r = d``
  the hull is cut into horizontal slices ``z = const``; in each slice the region ``R(z)`` is clipped by the
  half-plane ``n_x x + n_y y <= d - n_z z`` with its *own* 2-D section computation (intersection parameters of
  the line with the boundary pieces by closed-form roots, kept sub-pieces by a midpoint test, chords closing
  exit -> enter after sorting the cut points along the line) and Green's theorem; the slice results are
  integrated over ``z`` by composite Gauss-Legendre between the heights at which the line meets a frame corner.
  The cap (waterplane) quantities integrate the chord weights over ``z`` with the exact area element
  ``|n| / sqrt(n_x^2 + n_y^2) dt dz`` of the plane parametrised by ``(t, z)``.

Accuracy: ``mpmath`` at 30 digits for the arithmetic and roots, Gauss-Legendre nodes of ``numpy`` (double
precision), so the references are good to about ``1e-13`` relative, far inside the 1e-9 requests. The tests
check the reference itself against the known closed values (H1 hydrostatics numbers of design section 1)."""
from __future__ import annotations

import mpmath as mp
import numpy as np

mp.mp.dps = 30

_NODES = {}


def _gl(order: int):
    if order not in _NODES:
        x, w = np.polynomial.legendre.leggauss(order)
        _NODES[order] = ([mp.mpf(float(t)) for t in x], [mp.mpf(float(t)) for t in w])
    return _NODES[order]


def gauss(f, a, b, order: int = 20):
    """``int_a^b f`` (``f`` returns a tuple of mp numbers) by Gauss-Legendre of the given order."""
    xs, ws = _gl(order)
    half, mid = (b - a) / 2, (b + a) / 2
    total = None
    for x, w in zip(xs, ws):
        value = f(mid + half * x)
        if total is None:
            total = [w * v for v in value]
        else:
            total = [t + w * v for t, v in zip(total, value)]
    return [half * t for t in total]


class Hull:
    """The B1 hull (full, or the starboard half ``y >= 0``) between the frames and ``z = bottom, deck``."""

    def __init__(self, half: bool = False, frames=(1.5, 3.5), bottom=0.5, deck=3.0) -> None:
        from anygeometry.b1_example import b1_bow_surface
        surface = b1_bow_surface()
        self.p = [[[mp.mpf(c) for c in surface.controls[i][j]] for j in (0, 1)] for i in range(3)]
        self.w = [mp.mpf(surface.weights[i][0]) for i in range(3)]
        assert all(surface.weights[i][0] == surface.weights[i][1] for i in range(3))
        self.half = half
        self.x1, self.x2 = mp.mpf(frames[0]), mp.mpf(frames[1])
        self.bottom, self.deck = mp.mpf(bottom), mp.mpf(deck)

    # ---- the iso-v arc S(u, z/4): rational quadratic, power-basis numerators
    def coeffs(self, z):
        v = mp.mpf(z) / 4
        basis = [[1, -2, 1], [0, 2, -2], [0, 0, 1]]         # B0, B1, B2 in the power basis of u
        out = []
        for comp in (0, 1):
            c = [mp.mpf(0)] * 3
            for i in range(3):
                q = (1 - v) * self.p[i][0][comp] + v * self.p[i][1][comp]
                for e in range(3):
                    c[e] += self.w[i] * q * basis[i][e]
            out.append(c)
        wc = [mp.mpf(0)] * 3
        for i in range(3):
            for e in range(3):
                wc[e] += self.w[i] * basis[i][e]
        return out[0], out[1], wc

    @staticmethod
    def _ev(c, u):
        return c[0] + u * (c[1] + u * c[2])

    @staticmethod
    def _dv(c, u):
        return c[1] + 2 * c[2] * u

    def xy(self, z, u):
        nx, ny, w = self.coeffs(z)
        wv = self._ev(w, u)
        return self._ev(nx, u) / wv, self._ev(ny, u) / wv

    def xy_d(self, z, u):
        """``(x, y, dx/du, dy/du)``."""
        nx, ny, w = self.coeffs(z)
        wv, dw = self._ev(w, u), self._dv(w, u)
        x, y = self._ev(nx, u) / wv, self._ev(ny, u) / wv
        return x, y, (self._dv(nx, u) - x * dw) / wv, (self._dv(ny, u) - y * dw) / wv

    def u_at_x(self, z, x):
        """The ``u`` in [0, 1] with ``x(u, z) = x`` (``x`` increases with ``u``)."""
        nx, _ny, w = self.coeffs(z)
        q = [nx[0] - x * w[0], nx[1] - x * w[1], nx[2] - x * w[2]]
        roots = mp.polyroots(q[::-1], maxsteps=200, extraprec=60)
        good = [r.real for r in roots if abs(r.imag) < mp.mpf(10) ** -20 and -mp.mpf(10) ** -20 <= r.real <= 1]
        assert len(good) == 1, (z, x, roots)
        return good[0]

    def arc_ends(self, z):
        return self.u_at_x(z, self.x1), self.u_at_x(z, self.x2)

    # ---- the boundary of R(z), counterclockwise, as pieces
    def pieces(self, z):
        ua, ub = self.arc_ends(z)
        (xa, ya), (xb, yb) = self.xy(z, ua), self.xy(z, ub)
        if self.half:
            return [("line", (xa, mp.mpf(0)), (xb, mp.mpf(0))), ("line", (xb, mp.mpf(0)), (xb, yb)),
                    ("arc", +1, ub, ua), ("line", (xa, ya), (xa, mp.mpf(0)))]
        return [("arc", -1, ua, ub), ("line", (xb, -yb), (xb, yb)), ("arc", +1, ub, ua),
                ("line", (xa, ya), (xa, -ya))]

    def point(self, z, piece, t):
        """``(x, y, dx/dt, dy/dt)`` on a piece at the parameter ``t`` in [0, 1]."""
        if piece[0] == "line":
            (x0, y0), (x1, y1) = piece[1], piece[2]
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0), x1 - x0, y1 - y0
        _k, sign, u0, u1 = piece
        u = u0 + t * (u1 - u0)
        x, y, dx, dy = self.xy_d(z, u)
        return x, sign * y, dx * (u1 - u0), sign * dy * (u1 - u0)

    def cut_params(self, z, piece, a, b, c):
        """Parameters ``t`` in (0, 1) where ``a x + b y = c`` meets the piece."""
        if piece[0] == "line":
            (x0, y0), (x1, y1) = piece[1], piece[2]
            s0, s1 = a * x0 + b * y0 - c, a * x1 + b * y1 - c
            if s0 == s1:
                return []
            t = -s0 / (s1 - s0)
            return [t] if 0 < t < 1 else []
        _k, sign, u0, u1 = piece
        nx, ny, w = self.coeffs(z)
        q = [a * nx[e] + b * sign * ny[e] - c * w[e] for e in range(3)]
        if q[2] == 0 and q[1] == 0:
            return []
        roots = [-q[0] / q[1]] if q[2] == 0 else mp.polyroots(q[::-1], maxsteps=200, extraprec=60)
        out = []
        for r in roots:
            if abs(mp.im(r)) > mp.mpf(10) ** -20:
                continue
            t = (mp.re(r) - u0) / (u1 - u0)
            if 0 < t < 1:
                out.append(t)
        return sorted(out)

    # ---- one slice: area, int x dA, int y dA of R(z) clipped by a x + b y <= c
    def slice_stats(self, z, a, b, c, order: int = 16):
        pcs = self.pieces(z)
        subs = []                                    # (piece, t0, t1, kept)
        for pc in pcs:
            cuts = [mp.mpf(0)] + self.cut_params(z, pc, a, b, c) + [mp.mpf(1)]
            for t0, t1 in zip(cuts, cuts[1:]):
                x, y, _dx, _dy = self.point(z, pc, (t0 + t1) / 2)
                subs.append((pc, t0, t1, a * x + b * y - c <= 0))
        area = sx = sy = mp.mpf(0)
        cut_nodes = []                               # (kind, point)
        for i, (pc, t0, t1, kept) in enumerate(subs):
            if kept:
                if pc[0] == "line":
                    xa, ya = self.point(z, pc, t0)[:2]
                    xb, yb = self.point(z, pc, t1)[:2]
                    ar_, sx_, sy_ = self._straight(xa, ya, xb, yb)
                else:
                    def f(t, pc=pc):
                        x, y, dx, dy = self.point(z, pc, t)
                        return (x * dy - y * dx) / 2, x * x * dy / 2, -y * y * dx / 2
                    ar_, sx_, sy_ = gauss(f, t0, t1, order)
                area, sx, sy = area + ar_, sx + sx_, sy + sy_
            nxt = subs[(i + 1) % len(subs)]
            if kept != nxt[3]:
                x, y = self.point(z, pc, t1)[:2]
                cut_nodes.append(("exit" if kept else "enter", (x, y)))
        if cut_nodes:
            rho = mp.sqrt(a * a + b * b)
            along = lambda p: (-b * p[0] + a * p[1]) / rho
            cut_nodes.sort(key=lambda item: along(item[1]))
            assert len(cut_nodes) % 2 == 0
            for j in range(0, len(cut_nodes), 2):
                (k1, p), (k2, q) = cut_nodes[j], cut_nodes[j + 1]
                assert {k1, k2} == {"exit", "enter"}, cut_nodes
                ex, en = (p, q) if k1 == "exit" else (q, p)
                ar_, sx_, sy_ = self._straight(ex[0], ex[1], en[0], en[1])
                area, sx, sy = area + ar_, sx + sx_, sy + sy_
        return area, sx, sy

    @staticmethod
    def _straight(x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        return ((x0 * y1 - x1 * y0) / 2, dy * (x0 * x0 + x0 * x1 + x1 * x1) / 6,
                -dx * (y0 * y0 + y0 * y1 + y1 * y1) / 6)

    def chords(self, z, a, b, c):
        """The chords of ``a x + b y = c`` inside ``R(z)`` as ``(t_start, t_end)`` along ``(-b, a) / rho``."""
        pcs = self.pieces(z)
        rho = mp.sqrt(a * a + b * b)
        points = []
        for pc in pcs:
            for t in self.cut_params(z, pc, a, b, c):
                x, y = self.point(z, pc, t)[:2]
                points.append((-b * x + a * y) / rho)
        points.sort()
        assert len(points) % 2 == 0, points
        return [(points[j], points[j + 1]) for j in range(0, len(points), 2)]

    # ---- breakpoints in z: the line passes through a frame corner
    def breakpoints(self, a, b, nz, d):
        lo, hi = self.bottom, self.deck
        corners = []

        def height(z, which):
            ua, ub = self.arc_ends(z)
            u, x = (ua, self.x1) if which[0] == "a" else (ub, self.x2)
            y = self.xy(z, u)[1]
            y = y if which[1] == "+" else -y
            if self.half and which[1] == "-":
                y = mp.mpf(0)
            return a * x + b * y + nz * z - d

        zs = [lo + (hi - lo) * i / 60 for i in range(61)]
        out = [lo, hi]
        for which in ("a+", "a-", "b+", "b-"):
            prev = height(zs[0], which)
            for z0, z1 in zip(zs, zs[1:]):
                cur = height(z1, which)
                if prev == 0 or (prev < 0) != (cur < 0):
                    r = mp.findroot(lambda z: height(z, which), (z0, z1), solver="anderson", tol=1e-25, maxsteps=100)
                    if lo < r < hi:
                        out.append(r)
                prev = cur
        self.singular = self.tangent_heights(a, b, nz, d)
        out.extend(self.singular)
        return sorted(set(out))

    def tangent_heights(self, a, b, nz, d):
        """Heights at which the line is tangent to the arc: the discriminant of the (z-linear) quadratic
        ``a Nx + b (+-Ny) - (d - nz z) W`` in ``u`` is itself quadratic in ``z`` (chord ends have a square-root
        behaviour there, so these heights are panel boundaries)."""
        lo, hi = self.bottom, self.deck
        out = []
        for sign in (1, -1):
            def disc(z):
                nx, ny, w = self.coeffs(z)
                q = [a * nx[e] + b * sign * ny[e] - (d - nz * z) * w[e] for e in range(3)]
                return q[1] * q[1] - 4 * q[0] * q[2]
            d0, d1, d2 = disc(mp.mpf(0)), disc(mp.mpf(1)), disc(mp.mpf(2))
            c2, c1, c0 = (d0 - 2 * d1 + d2) / 2, (-3 * d0 + 4 * d1 - d2) / 2, d0
            if c2 == 0:
                roots = [-c0 / c1] if c1 != 0 else []
            else:
                disc2 = c1 * c1 - 4 * c2 * c0
                if disc2 < 0:
                    continue
                r = mp.sqrt(disc2)
                roots = [(-c1 - r) / (2 * c2), (-c1 + r) / (2 * c2)]
            out.extend(z for z in roots if lo < z < hi)
        return out

    def z_integral(self, f, breaks, order: int = 24, panels: int = 6):
        """Composite Gauss-Legendre between the breakpoints; an interval that ends at a tangent height (where the
        integrand behaves like a square root) is integrated by tanh-sinh (``mp.quad``) instead."""
        total = None
        singular = getattr(self, "singular", [])
        for z0, z1 in zip(breaks, breaks[1:]):
            if z1 - z0 < mp.mpf(10) ** -25:
                continue
            if any(abs(z0 - t) < mp.mpf(10) ** -20 or abs(z1 - t) < mp.mpf(10) ** -20 for t in singular):
                cache = {}

                def cached(z):
                    if z not in cache:
                        cache[z] = f(z)
                    return cache[z]
                size = len(cached((z0 + z1) / 2))
                value = [mp.quad(lambda z, i=i: cached(z)[i], [z0, z1]) for i in range(size)]
                total = value if total is None else [t + v for t, v in zip(total, value)]
                continue
            for i in range(panels):
                a = z0 + (z1 - z0) * i / panels
                b = z0 + (z1 - z0) * (i + 1) / panels
                value = gauss(f, a, b, order)
                total = value if total is None else [t + v for t, v in zip(total, value)]
        return total

    # ---- V and J of the clipped hull by slices
    def slice_volume_stats(self, n, d):
        """``(V, Jx, Jy, Jz)`` of ``hull & {n . r <= d}``."""
        a, b, nz = (mp.mpf(c) for c in (n[0], n[1], n[2]))
        d = mp.mpf(d)
        breaks = self.breakpoints(a, b, nz, d)

        def f(z):
            area, sx, sy = self.slice_stats(z, a, b, d - nz * z)
            return area, sx, sy, z * area
        v, jx, jy, jz = self.z_integral(f, breaks)
        return v, jx, jy, jz

    # ---- the cap of an inclined plane (n_x or n_y nonzero)
    def slice_cap_stats(self, n, d, o, e1, e2):
        """``(A, Mu, Mv, Iuu, Ivv, Iuv)`` of the section ``hull & {n . r = d}`` in the basis ``(o, e1, e2)``."""
        a, b, nz = (mp.mpf(c) for c in (n[0], n[1], n[2]))
        d = mp.mpf(d)
        rho = mp.sqrt(a * a + b * b)
        assert rho > 0
        nn = mp.sqrt(a * a + b * b + nz * nz)
        o = [mp.mpf(c) for c in o]
        e1 = [mp.mpf(c) for c in e1]
        e2 = [mp.mpf(c) for c in e2]
        breaks = self.breakpoints(a, b, nz, d)

        def f(z):
            c = d - nz * z
            r0 = (c * a / rho ** 2, c * b / rho ** 2, z)
            tau = (-b / rho, a / rho, mp.mpf(0))
            u0 = sum(e1[i] * (r0[i] - o[i]) for i in range(3))
            u1 = sum(e1[i] * tau[i] for i in range(3))
            v0 = sum(e2[i] * (r0[i] - o[i]) for i in range(3))
            v1 = sum(e2[i] * tau[i] for i in range(3))
            acc = [mp.mpf(0)] * 6
            for t0, t1 in self.chords(z, a, b, c):
                m0, m1, m2 = t1 - t0, (t1 ** 2 - t0 ** 2) / 2, (t1 ** 3 - t0 ** 3) / 3

                def integ(p0, p1, q0, q1):                    # int (p0 + p1 t)(q0 + q1 t) dt
                    return p0 * q0 * m0 + (p0 * q1 + p1 * q0) * m1 + p1 * q1 * m2
                one = (mp.mpf(1), mp.mpf(0))
                for k, (p, q) in enumerate(((one, one), ((u0, u1), one), ((v0, v1), one), ((u0, u1), (u0, u1)),
                                            ((v0, v1), (v0, v1)), ((u0, u1), (v0, v1)))):
                    acc[k] += integ(p[0], p[1], q[0], q[1])
            return tuple(t * nn / rho for t in acc)
        return self.z_integral(f, breaks)

    # ---- upright stations (analytic through the iso-v conic)
    def station_moments(self, h, o=(0, 0), order: int = 24):
        """``(A, Mu, Mv, Iuu, Ivv, Iuv)`` at ``z = h`` in the axes ``x``, ``y`` from the origin ``o`` (xy only):
        integrals over the conic parameter ``u`` of the strip ``0 <= y' <= y(u)`` (and its mirror image for the
        full hull)."""
        ox, oy = mp.mpf(o[0]), mp.mpf(o[1])
        ua, ub = self.arc_ends(h)

        def strip(y, ex, dx, lo, hi):
            # int over y' in [lo, hi] of (1, ex, y'-oy, ex^2, (y'-oy)^2, ex (y'-oy)) dy'  times dx
            m0 = hi - lo
            m1 = (hi * hi - lo * lo) / 2 - oy * m0
            m2 = ((hi - oy) ** 3 - (lo - oy) ** 3) / 3
            return (m0 * dx, ex * m0 * dx, m1 * dx, ex * ex * m0 * dx, m2 * dx, ex * m1 * dx)

        def f(u):
            x, y, dx, _dy = self.xy_d(h, u)
            ex = x - ox
            upper = strip(y, ex, dx, mp.mpf(0), y)
            if self.half:
                return upper
            lower = strip(y, ex, dx, -y, mp.mpf(0))
            return tuple(p + q for p, q in zip(upper, lower))
        return tuple(gauss(f, ua, ub, order))

    def station_volume_stats(self, h, order: int = 24, panels: int = 6):
        """``(V, Jx, Jy, Jz)`` of ``hull & {z <= h}`` from the station areas."""
        def f(z):
            a, mu, mv, _iuu, _ivv, _iuv = self.station_moments(z, (0, 0))
            return a, mu, mv, z * a
        return self.z_integral(f, [self.bottom, mp.mpf(h)], order, panels)
