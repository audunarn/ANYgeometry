"""Complete real-root isolation for low-degree intersection polynomials.

Coefficients are interpreted as exact rational representations of the supplied
numbers. Sturm counts, rather than display samples or complex-root filtering,
certify the number of distinct real roots. Repeated roots are retained once.
The result records isolating rational intervals as well as float witnesses.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from .errors import GeometryError


def _trim(values):
    values = list(values)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _derivative(p):
    return _trim([i*p[i] for i in range(1, len(p))] or [Fraction(0)])


def _value(p, x):
    value = Fraction(0)
    for coefficient in reversed(p):
        value = value*x+coefficient
    return value


def _integer_value(polynomial,point):
    """Homogeneous Horner numerator; a positive denominator preserves zeros."""
    numerator,denominator=point.numerator,point.denominator
    value=polynomial[-1]
    power=denominator
    for coefficient in reversed(polynomial[:-1]):
        value=value*numerator+coefficient*power
        power*=denominator
    return value


def _division(p, q):
    remainder = list(p)
    if q == (0,):
        raise GeometryError("zero polynomial divisor")
    quotient = [Fraction(0)]*max(1, len(p)-len(q)+1)
    while len(remainder) >= len(q) and remainder != [0]:
        index = len(remainder)-len(q)
        scale = remainder[-1]/q[-1]
        quotient[index] += scale
        for i, coefficient in enumerate(q):
            remainder[index+i] -= scale*coefficient
        remainder = list(_trim(remainder))
    return _trim(quotient), _trim(remainder)


def _square_free(p):
    first, second = p, _derivative(p)
    while second != (0,):
        first, second = second, _division(first, second)[1]
    return _division(p, first)[0]


def _sturm(p):
    sequence = [p, _derivative(p)]
    while sequence[-1] != (0,):
        remainder = _division(sequence[-2], sequence[-1])[1]
        if remainder == (0,):
            break
        # Positive rescaling reduces rational arithmetic growth without
        # changing any variation count.
        magnitude = abs(remainder[-1])
        sequence.append(tuple(-value/magnitude for value in remainder))
    # Integerize each row once. Positive scaling preserves its signs and
    # every Sturm variation, while avoiding repeated Fraction normalization
    # at every bisection point for degree-eight cylinder resultants.
    integer_rows=[]
    for row in sequence:
        denominator=math.lcm(*(value.denominator for value in row))
        values=tuple(value.numerator*(denominator//value.denominator) for value in row)
        divisor=math.gcd(*values)
        integer_rows.append(tuple(value//divisor for value in values))
    return tuple(integer_rows)


def _variations(sequence, point):
    numerator,denominator=point.numerator,point.denominator
    signs=[]
    for polynomial in sequence:
        value=polynomial[-1]
        power=denominator
        for coefficient in reversed(polynomial[:-1]):
            value=value*numerator+coefficient*power
            power*=denominator
        if value:
            signs.append(1 if value>0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


@dataclass(frozen=True, slots=True)
class IsolatedRoot:
    lower: Fraction
    upper: Fraction

    @property
    def witness(self):
        return float((self.lower+self.upper)/2)


def isolate_real_roots(coefficients, *, tolerance=1e-13, cancellation_check=None, interval=None):
    """Isolate every distinct real root; coefficients ascend by power.

    The zero polynomial represents a coincident constraint and is rejected,
    rather than being reported as an empty root set. There is no root-count
    cap. Cancellation raises before a partial result can be returned. An
    explicit closed rational interval selects only roots inside that interval.
    """
    try:
        p = _trim([Fraction(value) for value in coefficients])
        tolerance = Fraction(tolerance)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise GeometryError("root isolation needs finite coefficients and tolerance") from exc
    if not p or p == (0,) or tolerance <= 0:
        raise GeometryError("root isolation needs a nonzero polynomial and positive tolerance")
    if cancellation_check is not None and cancellation_check():
        raise GeometryError("analytic root isolation cancelled")
    if interval is not None:
        try:
            lower,upper=(Fraction(value) for value in interval)
            if lower>=upper:
                raise ValueError('empty interval')
        except (TypeError,ValueError,OverflowError,ZeroDivisionError) as exc:
            raise GeometryError('root isolation interval must have two finite increasing endpoints') from exc
    if len(p) == 1:
        return ()
    p = _square_free(p)
    sequence = _sturm(p)
    if interval is None:
        bound = Fraction(2)+max(abs(value/p[-1]) for value in p[:-1])
        # The strict Cauchy bound puts all roots inside, not on the endpoints.
        lower,upper=-bound,bound
    else:
        for endpoint in (lower,upper):
            if _integer_value(sequence[0],endpoint)==0:
                quotient,remainder=_division(p,(-endpoint,Fraction(1)))
                if remainder!=(0,):
                    raise GeometryError('inconsistent interval endpoint root')
                remaining=isolate_real_roots(quotient,tolerance=tolerance,interval=(lower,upper),
                                             cancellation_check=cancellation_check)
                return tuple(sorted((IsolatedRoot(endpoint,endpoint),*remaining),key=lambda item:item.lower))
    pending = [(lower,upper)]
    roots = []
    while pending:
        if cancellation_check is not None and cancellation_check():
            raise GeometryError("analytic root isolation cancelled")
        lower, upper = pending.pop()
        count = _variations(sequence, lower)-_variations(sequence, upper)
        if count == 0:
            continue
        if count == 1 and upper-lower <= tolerance:
            roots.append(IsolatedRoot(lower, upper))
            continue
        middle = (lower+upper)/2
        if _integer_value(sequence[0],middle) == 0:
            # Remove an exact dyadic root and restart on its quotient. This
            # keeps the isolating endpoints away from every remaining root.
            roots.append(IsolatedRoot(middle, middle))
            quotient, remainder = _division(p, (-middle, Fraction(1)))
            if remainder != (0,):
                raise GeometryError("inconsistent exact root isolation")
            remaining = isolate_real_roots(quotient, tolerance=tolerance,
                                          cancellation_check=cancellation_check,interval=interval)
            # The quotient includes roots isolated earlier in this traversal;
            # discard those earlier intervals rather than duplicate them.
            return tuple(sorted((IsolatedRoot(middle, middle), *remaining),
                                key=lambda item: item.lower))
        pending.extend(((middle, upper), (lower, middle)))
    return tuple(sorted(roots, key=lambda item: item.lower))


def trigonometric_roots(coefficients, *, start=0.0, sweep=math.tau,
                        tolerance=1e-12, cancellation_check=None):
    """Roots of c0+c1*cos(t)+c2*sin(t)+c3*cos(2t)+c4*sin(2t).

    A rational tan-half-angle polynomial supplies completeness. Its missing
    projective point (odd multiples of pi) is tested exactly. Roots at either
    interval endpoint are included; duplicate periodic endpoints are kept as
    distinct chart boundaries.
    """
    try:
        c0, c1, c2, c3, c4 = [Fraction(value) for value in coefficients]
        start, sweep, tolerance = float(start), float(sweep), float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError("invalid trigonometric root constraint") from exc
    if not all(math.isfinite(item) for item in (start, sweep, tolerance)) or sweep == 0 or tolerance <= 0:
        raise GeometryError("invalid trigonometric root interval")
    p = (c0+c1+c3, 2*c2+4*c4, 2*c0-6*c3, 2*c2-4*c4, c0-c1+c3)
    roots = isolate_real_roots(p, tolerance=tolerance/4,
                              cancellation_check=cancellation_check)
    angles = [2*math.atan(root.witness) for root in roots]
    if p[-1] == 0:
        angles.append(math.pi)
    lower, upper = sorted((start, start+sweep))
    result = []
    for angle in angles:
        first = math.ceil((lower-angle-tolerance)/math.tau)
        last = math.floor((upper-angle+tolerance)/math.tau)
        for turn in range(first, last+1):
            value = angle+turn*math.tau
            result.append(min(upper, max(lower, value)))
    result.sort(reverse=sweep < 0)
    return tuple(result)
