"""Returned forms of a quantity: exact, binary64 enclosure, decimal enclosure (design section 3).

A quantity is a rational ``Q`` (volume, volume moments: ``norm_squared is None``) or ``Q / |n|`` with
``|n|^2`` rational. Enclosures are computed with exact comparisons only (Fractions, integer square roots)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ._errors import RefusalCategory, refuse
from ._work import Meter

__all__ = ["QuantityResult", "build_quantity", "binary64_enclosure", "decimal_enclosure", "DEFAULT_DECIMAL_WIDTH"]

DEFAULT_DECIMAL_WIDTH = Fraction(1, 10 ** 12)    # decimal width when "decimal" is requested without an accuracy


@dataclass(frozen=True)
class QuantityResult:
    """One quantity in its requested forms.

    ``numerator``/``norm_squared``: the exact form (value ``numerator`` when ``norm_squared is None``, else
    ``numerator / sqrt(norm_squared)``); ``None`` when the exact form was not requested (volume and volume
    moments always carry it). ``binary64``: ``(lo, hi)`` floats with ``lo == hi`` or adjacent. ``decimal``:
    ``(lo, hi, width)`` decimal strings and the width used (requested accuracy, or the default width).
    ``accuracy``: requested accuracy or ``None``; ``accuracy_met``: ``True`` when requested and met, else ``None``
    (a request that cannot be met is refused ``ACCURACY_UNACHIEVABLE``)."""

    name: str
    numerator: Fraction | None
    norm_squared: Fraction | None
    binary64: tuple | None
    decimal: tuple | None
    accuracy: Fraction | None
    accuracy_met: bool | None


def _isqrt_counted(n: int, meter: Meter) -> int:
    meter.unit("sqrt_step", 1 + n.bit_length() // 1024)
    return math.isqrt(n)


def _rational_value(numerator: Fraction, norm_squared, meter: Meter):
    """The value as a Fraction when it is rational, else ``None`` (irrational ``Q / sqrt(N)``)."""
    if norm_squared is None:
        return numerator
    if numerator == 0:
        return Fraction(0)
    square = meter.fit(numerator * numerator / norm_squared)
    root_n, root_d = _isqrt_counted(square.numerator, meter), _isqrt_counted(square.denominator, meter)
    if root_n * root_n == square.numerator and root_d * root_d == square.denominator:
        return Fraction(root_n, root_d) if numerator > 0 else -Fraction(root_n, root_d)
    return None


def _compare_root(numerator: Fraction, norm_squared: Fraction, f: float, meter: Meter) -> int:
    """Sign of ``numerator / sqrt(norm_squared) - f`` (``numerator != 0``), exactly."""
    meter.unit("sqrt_step")
    sq = 1 if numerator > 0 else -1
    sf = (f > 0) - (f < 0)
    if sq > 0 and sf <= 0:
        return 1
    if sq < 0 and sf >= 0:
        return -1
    ff = Fraction(f)
    left, right = meter.fit(numerator * numerator), meter.fit(ff * ff * norm_squared)
    difference = (left > right) - (left < right)
    return difference if sq > 0 else -difference


def _out_of_range(name: str):
    return refuse(RefusalCategory.RESULT_OUT_OF_RANGE, f"{name} is outside the binary64 range")


def binary64_enclosure(numerator: Fraction, norm_squared, meter: Meter, name: str = "value") -> tuple:
    """``(lo, hi)``: the exact value as a double twice, or the adjacent doubles around it (outward, exact)."""
    value = _rational_value(numerator, norm_squared, meter)
    try:
        if value is not None:
            meter.unit("enclosure_step")
            f = float(value)
            if math.isinf(f):
                raise OverflowError
            exact = Fraction(f)
            if exact == value:
                return (f + 0.0, f + 0.0)
            lo, hi = (f, math.nextafter(f, math.inf)) if exact < value else (math.nextafter(f, -math.inf), f)
        else:
            # irrational x = Q.num sqrt(M) / (Q.den N.num), M = N.num N.den: a ~100-bit approximation first,
            # then exact adjustment to the bracketing doubles
            product = norm_squared.numerator * norm_squared.denominator
            shift = max(0, (200 - product.bit_length()) // 2 + 1)
            root = _isqrt_counted(product << (2 * shift), meter)              # sqrt(M) 2^shift
            approx = Fraction(numerator.numerator * root,
                              numerator.denominator * norm_squared.numerator * 2 ** shift)
            lo = float(approx)
            if math.isinf(lo):
                raise OverflowError
            while _compare_root(numerator, norm_squared, lo, meter) < 0:
                lo = math.nextafter(lo, -math.inf)
            while _compare_root(numerator, norm_squared, math.nextafter(lo, math.inf), meter) > 0:
                lo = math.nextafter(lo, math.inf)
            hi = math.nextafter(lo, math.inf)
    except OverflowError:
        raise _out_of_range(name) from None
    if math.isinf(lo) or math.isinf(hi):
        raise _out_of_range(name)
    return (lo + 0.0, hi + 0.0)


def decimal_enclosure(numerator: Fraction, norm_squared, width: Fraction, meter: Meter) -> tuple:
    """``(lo, hi)`` decimal strings containing the value with ``hi - lo <= width`` (multiples of ``10^-k``)."""
    inverse = -(-width.denominator // width.numerator)         # ceil(1 / width)
    digits = 0 if inverse <= 1 else len(str(inverse - 1))      # smallest k >= 0 with 10^k >= ceil(1 / width)
    scale = 10 ** digits
    meter.unit("enclosure_step")
    value = _rational_value(numerator, norm_squared, meter)
    if value is not None:
        scaled = meter.fit(value * scale)
        low, high = scaled.numerator // scaled.denominator, -((-scaled.numerator) // scaled.denominator)
    else:
        # |x| 10^k = T sqrt(M) / D with T = |Q.num| 10^k, M = N.num N.den, D = Q.den N.num
        t = abs(numerator.numerator) * scale
        d = numerator.denominator * norm_squared.numerator
        radicand = meter.fit(t * t * norm_squared.numerator * norm_squared.denominator)
        s = _isqrt_counted(radicand, meter)
        floor_abs = s // d
        ceil_abs = floor_abs + 1                                # the root is irrational here
        low, high = (floor_abs, ceil_abs) if numerator > 0 else (-ceil_abs, -floor_abs)
    return _decimal_string(low, digits), _decimal_string(high, digits)


def _decimal_string(scaled: int, digits: int) -> str:
    sign = "-" if scaled < 0 else ""
    text = str(abs(scaled))
    if digits == 0:
        return sign + text
    text = text.rjust(digits + 1, "0")
    return f"{sign}{text[:-digits]}.{text[-digits:]}"


def build_quantity(name: str, numerator: Fraction, norm_squared, forms, accuracy, meter: Meter,
                   *, always_exact: bool = False) -> QuantityResult:
    """The requested forms of one quantity; refuses ``ACCURACY_UNACHIEVABLE`` / ``RESULT_OUT_OF_RANGE``."""
    binary64 = decimal = None
    if "binary64" in forms:
        binary64 = binary64_enclosure(numerator, norm_squared, meter, name)
        if accuracy is not None and Fraction(binary64[1]) - Fraction(binary64[0]) > accuracy:
            raise refuse(RefusalCategory.ACCURACY_UNACHIEVABLE,
                         f"binary64 cannot meet the accuracy {float(accuracy)!r} for {name}; its bound is "
                         f"{binary64[1] - binary64[0]!r} (the decimal form remains available)")
    if "decimal" in forms:
        width = DEFAULT_DECIMAL_WIDTH if accuracy is None else accuracy
        decimal = (*decimal_enclosure(numerator, norm_squared, width, meter), width)
    keep_exact = always_exact or "exact" in forms
    return QuantityResult(name, numerator if keep_exact else None, norm_squared if keep_exact else None, binary64,
                          decimal, accuracy, None if accuracy is None else True)
