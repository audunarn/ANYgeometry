"""Generic local arc certificate for ``S1(u, v) = S2(s, t)`` (three equations, four parameters).

Solved in factored homogeneous form
``F(u,v,s,t) = A_xyz(u,v) B_w(s,t) - B_xyz(s,t) A_w(u,v)``
(``F = A_w B_w (S1 - S2)`` for positive weights, so the zero sets agree and the
Jacobians agree up to a positive factor on solutions).  Each surface is an
exact homogeneous Bezier piece; enclosures use Bernstein ranges of sub-boxes of
the *two* 2D pieces and their derivative nets, combined by interval products.
No 4D tensor exists.

Parameterized Krawczyk: with free coordinate ``f`` and unknown ``y`` (the other
three), ``K(Y, Phi) = y0 - C F(y0, Phi) + (I - C J_y(Y x Phi)) (Y - y0)``.  If
``K`` lies strictly inside ``Y`` then for EVERY ``f`` in ``Phi`` there is exactly
one solution ``y(f)`` in ``Y`` (and the Jacobian is regular over the whole box).

THIS IS A LOCAL CERTIFICATE ONLY.  It proves one graph over the free interval
inside the given box.  It does not say the box holds the whole component, that
arcs close up, or that other components do not exist; the global cover,
stitching and accounting are separate (not yet implemented) obligations.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import time

import numpy as np

from ..errors import GeometryError
from ..nurbs import NurbsSurface
from ._bernstein import bezier_piece, derivative_grid, grid_ranges, subbox
from ._intervals import Interval
from ._source import SourceDefinition

__all__ = ["LocalArcCertificate", "certify_local_arc"]

_ONE = Interval(1)


@dataclass(frozen=True, slots=True)
class LocalArcCertificate:
    certified: bool
    reason: str
    free_coordinate: int
    knot_box: tuple                 # ((u0,u1),(v0,v1),(s0,s1),(t0,t1)) Fractions, knot coordinates
    free_interval: tuple            # Fractions in the box-local [0,1] parameter of the free coordinate
    unknown_enclosure: tuple = ()   # three Intervals (box-local) of K, strictly inside the unknown box
    weights_positive: bool = False
    immersed: bool = False          # both normals certified non-zero over the whole box
    regular: bool = False           # Krawczyk strict inclusion (implies a regular 3x3 minor everywhere)
    uses_dense_4d_tensor: bool = False
    preconditioner: tuple = ()      # exact rational C, useful for independent enclosure replay

    def world_free_interval(self):
        lo, hi = self.knot_box[self.free_coordinate]
        return lo + self.free_interval[0] * (hi - lo), lo + self.free_interval[1] * (hi - lo)


class _Pair:
    @classmethod
    def from_pieces(cls, piece_a, piece_b):
        """Pair from cached ``(grid, d_u, d_v)`` pieces (no recomputation, no 4D tensor)."""
        made = cls.__new__(cls)
        # Exact common translation removes origin-dependent interval cancellation.
        # A common power-of-two world scale preserves the zero set and all ranks.
        origin = tuple(piece_a[0][0][0][k] / piece_a[0][0][0][3] for k in range(3))
        def translate(grid):
            return tuple(tuple(tuple(p[k] - origin[k] * p[3] for k in range(3)) + (p[3],)
                               for p in row) for row in grid)
        a, b = translate(piece_a[0]), translate(piece_b[0])
        largest = max(abs(p[k] / p[3]) for g in (a, b) for row in g for p in row for k in range(3))
        power = largest.numerator.bit_length() - largest.denominator.bit_length() if largest else 0
        scale = Fraction(2) ** (-power)
        def scaled(grid):
            return tuple(tuple(tuple(p[k] * scale for k in range(3)) + (p[3],) for p in row) for row in grid)
        made.a, made.b = scaled(a), scaled(b)
        made.origin = None
        made.float_cache = [None]
        made.da = (derivative_grid(made.a, 0), derivative_grid(made.a, 1))
        made.db = (derivative_grid(made.b, 0), derivative_grid(made.b, 1))
        return made

    def __init__(self, first, second, knot_box):
        net_a, net_b = first.exact_net(), second.exact_net()
        self.a = bezier_piece(first, net_a, knot_box[0], knot_box[1])
        self.b = bezier_piece(second, net_b, knot_box[2], knot_box[3])
        prepared = self.from_pieces((self.a,), (self.b,))
        self.a, self.b, self.da, self.db = prepared.a, prepared.b, prepared.da, prepared.db
        self.origin = None
        self.float_cache = prepared.float_cache

    def anchored(self, box):
        """Common exact local world origin; coefficient/derivative grids remain shared."""
        u,v = ((Fraction(a)+Fraction(b))/2 for a,b in box[:2])
        p = subbox(self.a,u,u,v,v)[0][0]
        made = self.__class__.__new__(self.__class__)
        made.a,made.b,made.da,made.db = self.a,self.b,self.da,self.db
        made.float_cache = self.float_cache
        made.origin = tuple(p[k]/p[3] for k in range(3))
        return made

    def ranges(self, grid):
        if self.origin is not None:
            grid = tuple(tuple(tuple(p[k]-self.origin[k]*p[3] for k in range(3))+(p[3],)
                               for p in row) for row in grid)
        return grid_ranges(grid)

    def preconditioned(self, matrix):
        """Exact CF/CJ coefficients before ranges; original immersion is checked separately."""
        def transform(grid):
            made=[]
            for row in grid:
                out=[]
                for p in row:
                    xyz=[p[k]-(self.origin[k]*p[3] if self.origin is not None else 0) for k in range(3)]
                    out.append(tuple(sum((matrix[i][k]*xyz[k] for k in range(3) if matrix[i][k]),Fraction(0))
                                     for i in range(3))+(p[3],))
                made.append(tuple(out))
            return tuple(made)
        made=self.__class__.__new__(self.__class__)
        made.a,made.b=transform(self.a),transform(self.b)
        made.da=(derivative_grid(made.a,0),derivative_grid(made.a,1))
        made.db=(derivative_grid(made.b,0),derivative_grid(made.b,1))
        made.origin=None
        made.float_cache=[None]
        return made

    def section_value(self, free, y0, phi):
        """Exact one-variable Bernstein F(y0,Phi), preserving coordinate/weight correlation."""
        box=_box_with(free,[Interval(x) for x in y0],phi)
        (u0,u1),(v0,v1),(s0,s1),(t0,t1)=box
        a=subbox(self.a,u0,u1,v0,v1)
        b=subbox(self.b,s0,s1,t0,t1)
        varying,point=(a,b[0][0]) if free<2 else (b,a[0][0])
        sign=1 if free<2 else -1
        coefficients=[tuple(sign*(p[k]*point[3]-point[k]*p[3]) for k in range(3))
                      for row in varying for p in row]
        return tuple(Interval(min(p[k] for p in coefficients),max(p[k] for p in coefficients))
                     for k in range(3))

    def enclose(self, box):
        """Ranges over a box-local sub-box ``[(lo, hi)] * 4`` (degenerate intervals give exact values)."""
        (u0, u1), (v0, v1), (s0, s1), (t0, t1) = box
        a = self.ranges(subbox(self.a, u0, u1, v0, v1))
        b = self.ranges(subbox(self.b, s0, s1, t0, t1))
        da = tuple(self.ranges(subbox(g, u0, u1, v0, v1)) for g in self.da)
        db = tuple(self.ranges(subbox(g, s0, s1, t0, t1)) for g in self.db)
        return a, b, da, db

    def enclose_values(self,box):
        """Same exact value/weight bounds as enclose, without unused derivatives."""
        (u0,u1),(v0,v1),(s0,s1),(t0,t1)=box
        return (self.ranges(subbox(self.a,u0,u1,v0,v1)),
                self.ranges(subbox(self.b,s0,s1,t0,t1)))

    @staticmethod
    def value_and_jacobian(enc):
        a, b, da, db = enc
        value = [a[k] * b[3] - b[k] * a[3] for k in range(3)]
        jacobian = []
        for k in range(3):
            row = [da[c][k] * b[3] - b[k] * da[c][3] for c in range(2)]
            row += [a[k] * db[c][3] - db[c][k] * a[3] for c in range(2)]
            jacobian.append(row)
        return value, jacobian

    @staticmethod
    def immersed(enc) -> bool:
        a, b, da, db = enc

        def normal_nonzero(p, dp):
            n_u = [dp[0][k] * p[3] - p[k] * dp[0][3] for k in range(3)]
            n_v = [dp[1][k] * p[3] - p[k] * dp[1][3] for k in range(3)]
            cross = [n_u[1] * n_v[2] - n_u[2] * n_v[1], n_u[2] * n_v[0] - n_u[0] * n_v[2],
                     n_u[0] * n_v[1] - n_u[1] * n_v[0]]
            return any(c.excludes_zero() for c in cross)

        return normal_nonzero(a, da) and normal_nonzero(b, db)


def _box_with(free: int, y: list, phi: Interval):
    """Four (lo, hi) pairs: the three unknown intervals in order and ``phi`` at ``free``."""
    others = iter(y)
    return [(phi.lo, phi.hi) if k == free else (lambda iv: (iv.lo, iv.hi))(next(others)) for k in range(4)]


def certify_local_arc(first: NurbsSurface, second: NurbsSurface, knot_box, free_coordinate: int,
                      unknown=None, free_interval=None) -> LocalArcCertificate:
    """Try to certify one graph over the free coordinate inside ``knot_box``.

    ``knot_box`` holds four Fraction pairs in knot coordinates, each inside one
    knot span.  ``unknown`` (three box-local Intervals, default [0, 1]) and
    ``free_interval`` (box-local pair, default (0, 1)) select the Krawczyk
    region.  Returns an uncertified record with a reason instead of raising for
    mathematical failures (singular Jacobian, non-contraction, weights).
    """
    if type(free_coordinate) is not int or free_coordinate not in (0, 1, 2, 3):
        raise GeometryError("free coordinate must be the integer 0, 1, 2 or 3")
    # trusted values rebuilt from complete frozen definitions (same path as the query)
    first = SourceDefinition.capture(first).rebuild()
    second = SourceDefinition.capture(second).rebuild()
    from ._dependencies import verify_dependencies
    verify_dependencies()
    try:
        if len(knot_box) != 4 or any(len(pair) != 2 for pair in knot_box):
            raise ValueError
        box = tuple((Fraction(lo), Fraction(hi)) for lo, hi in knot_box)
        phi = Interval(0, 1) if free_interval is None else Interval(*free_interval)
        if unknown is None:
            y = [Interval(0, 1)] * 3
        else:
            if len(unknown) != 3 or not all(isinstance(item, Interval) for item in unknown):
                raise ValueError
            y = list(unknown)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError("malformed knot box, free interval or unknown region") from error
    if not all(0 <= i.lo <= i.hi <= 1 for i in y) or not 0 <= phi.lo <= phi.hi <= 1:
        raise GeometryError("certificate region must lie in the box")
    # the box must lie inside the SELECTED (possibly reversed or cropped) chart ranges, in knot coordinates
    for (lo, hi), (r0, r1) in zip(box, (first.u_range, first.v_range, second.u_range, second.v_range)):
        low, high = sorted((Fraction(r0), Fraction(r1)))
        if not low <= lo < hi <= high:
            raise GeometryError("knot box leaves the selected support range or has no positive size")
    pair = _Pair(first, second, box)
    pair = pair.anchored(_box_with(free_coordinate,y,phi))
    return certify_prepared(pair, box, free_coordinate, y, phi)


def certify_prepared(pair, box, free_coordinate: int, y, phi, *, for_root_filter=False,
                     interval_mode="filtered",stats=None) -> LocalArcCertificate:
    reuse_c = None
    if interval_mode=="filtered":
        from ._filtered import certificate,Unknown
        began=time.perf_counter()
        if stats is not None:
            stats.float_attempts+=1
        try:
            filtered=certificate(pair,box,free_coordinate,y,phi,for_root_filter=for_root_filter)
            if filtered.certified or (for_root_filter and filtered.unknown_enclosure):
                return filtered
            if not for_root_filter:
                reuse_c = _completed_filter_preconditioner(filtered, box, free_coordinate, phi)
        except (Unknown,OverflowError,ZeroDivisionError,FloatingPointError):
            pass  # unknown arithmetic cannot exclude or certify; use exact reference
        finally:
            if stats is not None:
                stats.filtered_seconds+=time.perf_counter()-began
        if stats is not None:
            stats.exact_escalations+=1
    began=time.perf_counter()
    try:
        if reuse_c is not None and stats is not None:
            stats.exact_preparations_reused += 1
        return _certify_exact(pair,box,free_coordinate,y,phi,for_root_filter=for_root_filter,
                              _reuse_c=reuse_c)
    finally:
        if stats is not None:
            stats.exact_seconds+=time.perf_counter()-began


def _completed_filter_preconditioner(certificate, box, free, phi):
    """Invocation-local facts only, after a completed finite filtered proof.

    The caller just computed this result on this same pair and unknown box.
    Unknown/exception and root-filter paths never supply reusable preparation.
    """
    if (type(certificate) is not LocalArcCertificate or not certificate.weights_positive
            or not certificate.immersed or certificate.knot_box != box
            or certificate.free_coordinate != free or certificate.free_interval != (phi.lo, phi.hi)
            or len(certificate.unknown_enclosure) != 3
            or not all(type(x) is Interval for x in certificate.unknown_enclosure)):
        return None
    matrix = certificate.preconditioner
    if (len(matrix) != 3 or any(len(row) != 3 for row in matrix)
            or not all(type(x) is Fraction for row in matrix for x in row)):
        return None
    return matrix


def _certify_exact(pair, box, free_coordinate: int, y, phi, *, for_root_filter=False,
                   _reuse_c=None) -> LocalArcCertificate:
    """Krawczyk certificate on an already prepared pair (cached pieces); ``y``/``phi`` are box-local Intervals."""

    def fail(reason, **flags):
        return LocalArcCertificate(False, reason, free_coordinate, box, (phi.lo, phi.hi), **flags)

    region = _box_with(free_coordinate, y, phi)
    unknown_columns = [c for c in range(4) if c != free_coordinate]
    y0 = [i.mid for i in y]
    if _reuse_c is not None:
        # Both facts were rigorously proved on this very region by the
        # completed filtered attempt. Any fixed rational C is valid below.
        immersed, c_matrix = True, _reuse_c
    else:
        enc = pair.enclose(region)
        positive = enc[0][3].lo > 0 and enc[1][3].lo > 0
        if not positive:
            return fail("weights are not certified positive over the box")
        immersed = pair.immersed(enc)
        if not immersed and not for_root_filter:
            return fail("a surface normal is not certified non-zero over the box", weights_positive=True)
        mid_box = _box_with(free_coordinate, [Interval(m) for m in y0], Interval(phi.mid))
        _, mid_jacobian = pair.value_and_jacobian(pair.enclose(mid_box))
        # Exact power-of-two scaling keeps the floating proposal finite.
        exact_minor = [[mid_jacobian[k][c].lo for c in unknown_columns] for k in range(3)]
        largest = max((abs(x) for row in exact_minor for x in row), default=Fraction(0))
        if largest == 0:
            return fail("midpoint Jacobian minor is zero", weights_positive=True, immersed=immersed)
        exponent = largest.numerator.bit_length() - largest.denominator.bit_length()
        scale = Fraction(2) ** exponent
        try:
            matrix = np.array([[float(x / scale) for x in row] for row in exact_minor], dtype=float)
            if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 3:
                return fail("midpoint Jacobian minor is singular", weights_positive=True, immersed=immersed)
            inverse = np.linalg.inv(matrix)
            if not np.all(np.isfinite(inverse)):
                return fail("midpoint Jacobian minor is singular", weights_positive=True, immersed=immersed)
            c_matrix = [[Fraction(float(inverse[i, k])) / scale for k in range(3)] for i in range(3)]
        except (np.linalg.LinAlgError, OverflowError, ValueError, FloatingPointError):
            return fail("float preconditioner failed; exact certificate not attempted", weights_positive=True,
                        immersed=immersed)
    # Apply the fixed exact preconditioner to homogeneous coefficients BEFORE
    # intervalization. Interval sums C*F and C*J would discard the shared
    # polynomial cancellations needed by correlated surface coordinates.
    conditioned=pair.preconditioned(c_matrix)
    f_enclosure=conditioned.section_value(free_coordinate,y0,phi)
    _value,conditioned_jacobian=conditioned.value_and_jacobian(conditioned.enclose(region))
    jy = [[conditioned_jacobian[k][c] for c in unknown_columns] for k in range(3)]
    result = []
    for i in range(3):
        cf = f_enclosure[i]
        total = Interval(y0[i]) - cf
        for j in range(3):
            m_ij = jy[i][j]
            r_ij = Interval(1 if i == j else 0) - m_ij
            total = total + r_ij * (y[j] - Interval(y0[j]))
        result.append(total)
    if not all(result[i].strictly_inside(y[i]) for i in range(3)):
        return fail("Krawczyk operator is not strictly inside the unknown box",
                    weights_positive=True, immersed=immersed, unknown_enclosure=tuple(result),
                    preconditioner=tuple(tuple(row) for row in c_matrix))
    if not immersed:
        return fail("root filter only: a surface normal is not certified non-zero",
                    weights_positive=True,immersed=False,unknown_enclosure=tuple(result),
                    preconditioner=tuple(tuple(row) for row in c_matrix))
    return LocalArcCertificate(True, "unique regular graph over the full free interval", free_coordinate, box,
                               (phi.lo, phi.hi), tuple(result), True, True, True, False,
                               tuple(tuple(row) for row in c_matrix))
