"""Immutable affine carriers with exact, directed retained subranges.

Topology vertices are finite proxies; they never redefine this support.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
import math

import numpy as np

from .errors import GeometryError


@dataclass(frozen=True, slots=True)
class LineRange:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    domain: tuple[Q, Q] = (Q(0), Q(1))

    def __post_init__(self):
        for name in ('a', 'b'):
            try:
                value = tuple(float(x) for x in getattr(self, name))
            except (TypeError, ValueError, OverflowError) as error:
                raise GeometryError('line carrier endpoints must be finite 3D points') from error
            if len(value) != 3 or not all(math.isfinite(x) for x in value):
                raise GeometryError('line carrier endpoints must be finite 3D points')
            object.__setattr__(self, name, value)
        try:
            domain = tuple(Q(x) for x in self.domain)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
            raise GeometryError('line range domain must contain exact finite rational parameters') from error
        if (len(domain) != 2 or domain[0] == domain[1] or self.a == self.b
                or any(not 0 <= x <= 1 for x in domain)):
            raise GeometryError('line range must be nondegenerate')
        object.__setattr__(self, 'domain', domain)

    @property
    def exact_domain(self):
        self.validate_definition()
        return tuple((x.numerator, x.denominator) for x in self.domain)

    def validate_definition(self):
        if (type(self.domain) is not tuple or len(self.domain) != 2
                or any(type(x) is not Q or not 0 <= x <= 1 for x in self.domain) or self.domain[0] == self.domain[1]
                or any(type(p) is not tuple or len(p) != 3 or any(type(x) is not float or not math.isfinite(x)
                    for x in p) for p in (self.a, self.b)) or self.a == self.b):
            raise GeometryError('retained line definition is malformed')

    def exact_point(self, parameter):
        self.validate_definition()
        t = Q(parameter)
        if not 0 <= t <= 1:
            raise GeometryError('line parameter must lie in [0, 1]')
        lam = self.domain[0] + t * (self.domain[1] - self.domain[0])
        return tuple(Q(a) + lam * (Q(b) - Q(a)) for a, b in zip(self.a, self.b))

    def evaluate(self, parameters):
        values = np.asarray(parameters)
        points = [tuple(float(x) for x in self.exact_point(t)) for t in values.flat]
        return np.asarray(points, dtype=float).reshape(values.shape + (3,))

    def derivative(self, parameters):
        values = np.asarray(parameters)
        # Validate parameters without deriving the carrier from finite proxies.
        for t in values.flat:
            self.exact_point(t)
        direction = tuple(float((Q(b)-Q(a))*(self.domain[1]-self.domain[0]))
                          for a, b in zip(self.a, self.b))
        return np.broadcast_to(direction, values.shape + (3,)).copy()

    def bounds(self, lower=0, upper=1):
        if Q(lower) > Q(upper):
            raise GeometryError('curve bound interval is reversed')
        points = (self.exact_point(lower), self.exact_point(upper))
        from .nurbs import _float_down, _float_up
        return (np.asarray([_float_down(min(p[k] for p in points)) for k in range(3)]),
                np.asarray([_float_up(max(p[k] for p in points)) for k in range(3)]))

    def subcurve(self, lower, upper):
        lower, upper = Q(lower), Q(upper)
        if not (0 <= lower <= 1 and 0 <= upper <= 1) or lower == upper:
            raise GeometryError('line subrange needs distinct parameters in [0, 1]')
        start, delta = self.domain[0], self.domain[1]-self.domain[0]
        return replace(self, domain=(start+lower*delta, start+upper*delta))

    def transformed(self, matrix):
        # A transform authors a new finite carrier; retained domains survive.
        from .exact_curves import _affine
        matrix = _affine(matrix)
        return replace(self, a=matrix[:3, :3] @ self.a + matrix[:3, 3],
                       b=matrix[:3, :3] @ self.b + matrix[:3, 3])

    def certify_plane(self, frame, tolerance=1.e-8):
        """Bound the complete affine R-S span independently of joint tolerance."""
        normal = tuple(Q(x) for x in frame.normal)
        norm2 = sum(x*x for x in normal)
        if norm2 == 0:
            raise GeometryError('line range has a degenerate incident plane')
        points = (self.exact_point(0), self.exact_point(1))
        if frame.uv(points[0]) == frame.uv(points[1]):
            raise GeometryError('line range projection is degenerate')
        for point in points:
            residual = sum(n*x for n, x in zip(normal, point))-frame.offset()
            if residual*residual > Q(tolerance)**2 * norm2:
                raise GeometryError('line carrier exceeds the native plane residual')
