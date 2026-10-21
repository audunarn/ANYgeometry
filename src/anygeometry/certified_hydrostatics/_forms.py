"""Returned forms of a quantity: exact, binary64 enclosure, decimal enclosure (design section 3).

A quantity is a rational ``Q`` (volume, volume moments: ``norm_squared is None``) or ``Q / |n|`` with
``|n|^2`` rational. Enclosures are computed with exact comparisons only (Fractions, integer square roots)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ._errors import RefusalCategory, refuse
from ._arith import Ar, int_to_str
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
    (a request that cannot be met is refused ``ACCURACY_UNACHIEVABLE``). ``enclosure``: ``(lo, hi)`` Fractions, the
    published claim of a quantity of a curved preparation (CH3c, design 4.4: core +- width); ``None`` for planar
    quantities. For a curved quantity ``numerator`` is present only when the computation was exact."""

    name: str
    numerator: Fraction | None
    norm_squared: Fraction | None
    binary64: tuple | None
    decimal: tuple | None
    accuracy: Fraction | None
    accuracy_met: bool | None
    enclosure: tuple | None = None


def _rational_value(ar: Ar, numerator: Fraction, norm_squared):
    """The value as a Fraction when it is rational, else ``None`` (irrational ``Q / sqrt(N)``)."""
    if norm_squared is None:
        return numerator
    if numerator == 0:
        return Fraction(0)
    square = ar.div(ar.mul(numerator, numerator), norm_squared)
    root_n, root_d = ar.isqrt(square.numerator), ar.isqrt(square.denominator)
    if root_n * root_n == square.numerator and root_d * root_d == square.denominator:
        return Fraction(root_n, root_d) if numerator > 0 else -Fraction(root_n, root_d)
    return None


def _compare_root(ar: Ar, numerator: Fraction, norm_squared: Fraction, f: float) -> int:
    """Sign of ``numerator / sqrt(norm_squared) - f`` (``numerator != 0``), exactly."""
    ar.meter.unit("sqrt_step")
    sq = 1 if numerator > 0 else -1
    sf = (f > 0) - (f < 0)
    if sq > 0 and sf <= 0:
        return 1
    if sq < 0 and sf >= 0:
        return -1
    ff = ar.exact_float(f)
    difference = ar.cmp(ar.mul(numerator, numerator), ar.mul(ar.mul(ff, ff), norm_squared))
    return difference if sq > 0 else -difference


def _out_of_range(name: str):
    return refuse(RefusalCategory.RESULT_OUT_OF_RANGE, f"{name} is outside the binary64 range")


def binary64_enclosure(numerator: Fraction, norm_squared, meter: Meter, name: str = "value") -> tuple:
    """``(lo, hi)``: the exact value as a double twice, or the adjacent doubles around it (outward, exact)."""
    ar = Ar(meter)
    value = _rational_value(ar, numerator, norm_squared)
    try:
        if value is not None:
            meter.unit("enclosure_step")
            f = float(value)
            if math.isinf(f):
                raise OverflowError
            order = ar.cmp(ar.exact_float(f), value)
            if order == 0:
                return (f + 0.0, f + 0.0)
            lo, hi = (f, math.nextafter(f, math.inf)) if order < 0 else (math.nextafter(f, -math.inf), f)
        else:
            # irrational x = Q.num sqrt(M) / (Q.den N.num), M = N.num N.den: a ~100-bit approximation first,
            # then exact adjustment to the bracketing doubles
            product = ar.mul(norm_squared.numerator, norm_squared.denominator)
            shift = max(0, (200 - product.bit_length()) // 2 + 1)
            ar.guard(product.bit_length() + 2 * shift)
            root = ar.isqrt(product << (2 * shift))                       # sqrt(M) 2^shift
            approx = Fraction(ar.mul(numerator.numerator, root),
                              ar.mul(numerator.denominator, ar.mul(norm_squared.numerator, 2 ** shift)))
            lo = float(approx)
            if math.isinf(lo):
                raise OverflowError
            while _compare_root(ar, numerator, norm_squared, lo) < 0:
                lo = math.nextafter(lo, -math.inf)
            while _compare_root(ar, numerator, norm_squared, math.nextafter(lo, math.inf)) > 0:
                lo = math.nextafter(lo, math.inf)
            hi = math.nextafter(lo, math.inf)
    except OverflowError:
        raise _out_of_range(name) from None
    if math.isinf(lo) or math.isinf(hi):
        raise _out_of_range(name)
    return (lo + 0.0, hi + 0.0)


def decimal_enclosure(numerator: Fraction, norm_squared, width: Fraction, meter: Meter) -> tuple:
    """``(lo, hi)`` decimal strings containing the value with ``hi - lo <= width`` (multiples of ``10^-k``)."""
    ar = Ar(meter)
    inverse = -(-width.denominator // width.numerator)         # ceil(1 / width)
    digits = 0
    if inverse > 1:                                            # smallest k >= 0 with 10^k >= ceil(1 / width)
        digits = max(0, int(inverse.bit_length() * 0.30103) - 1)
        while ar.pow10(digits) < inverse:
            digits += 1
    scale = ar.pow10(digits)
    meter.unit("enclosure_step")
    value = _rational_value(ar, numerator, norm_squared)
    if value is not None:
        scaled = ar.mul(value, scale)
        low, high = scaled.numerator // scaled.denominator, -((-scaled.numerator) // scaled.denominator)
    else:
        # |x| 10^k = T sqrt(M) / D with T = |Q.num| 10^k, M = N.num N.den, D = Q.den N.num
        t = ar.mul(abs(numerator.numerator), scale)
        d = ar.mul(numerator.denominator, norm_squared.numerator)
        radicand = ar.mul(ar.mul(t, t), ar.mul(norm_squared.numerator, norm_squared.denominator))
        s = ar.isqrt(radicand)
        floor_abs = s // d
        ceil_abs = floor_abs + 1                                # the root is irrational here
        low, high = (floor_abs, ceil_abs) if numerator > 0 else (-ceil_abs, -floor_abs)
    meter.unit("digit_conversion", 1)
    return _decimal_string(low, digits, meter), _decimal_string(high, digits, meter)


def _decimal_string(scaled: int, digits: int, meter=None) -> str:
    sign = "-" if scaled < 0 else ""
    text = int_to_str(abs(scaled), None if meter is None else (lambda: meter.unit("digit_chunk")))
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
        ar = Ar(meter)
        if accuracy is not None and ar.cmp(ar.sub(ar.exact_float(binary64[1]), ar.exact_float(binary64[0])),
                                           accuracy) > 0:
            raise refuse(RefusalCategory.ACCURACY_UNACHIEVABLE,
                         f"binary64 cannot meet the accuracy {float(accuracy)!r} for {name}; its bound is "
                         f"{binary64[1] - binary64[0]!r} (the decimal form remains available)")
    if "decimal" in forms:
        width = DEFAULT_DECIMAL_WIDTH if accuracy is None else accuracy
        decimal = (*decimal_enclosure(numerator, norm_squared, width, meter), width)
    keep_exact = always_exact or "exact" in forms or "enclosure" in forms
    return QuantityResult(name, numerator if keep_exact else None, norm_squared if keep_exact else None, binary64,
                          decimal, accuracy, None if accuracy is None else True)


def interval_binary64(lo: Fraction, hi: Fraction, meter: Meter, name: str = "value") -> tuple:
    """Doubles ``(f_lo, f_hi)`` with ``f_lo <= lo`` and ``hi <= f_hi``, as tight as possible (exact comparisons)."""
    ar = Ar(meter)
    try:
        meter.unit("enclosure_step")
        f_lo, f_hi = float(lo), float(hi)
        if math.isinf(f_lo) or math.isinf(f_hi):
            raise OverflowError
        while ar.cmp(ar.exact_float(f_lo), lo) > 0:
            meter.unit("enclosure_step")
            f_lo = math.nextafter(f_lo, -math.inf)
        while ar.cmp(ar.exact_float(f_hi), hi) < 0:
            meter.unit("enclosure_step")
            f_hi = math.nextafter(f_hi, math.inf)
    except OverflowError:
        raise _out_of_range(name) from None
    if math.isinf(f_lo) or math.isinf(f_hi):
        raise _out_of_range(name)
    return (f_lo + 0.0, f_hi + 0.0)


def interval_decimal(lo: Fraction, hi: Fraction, width: Fraction, meter: Meter) -> tuple:
    """``(low, high, k)``: integers with ``low / 10^k <= lo`` and ``hi <= high / 10^k``, ``k`` fixed by ``width / 8``;
    the caller checks that the rounded interval still fits ``width``."""
    ar = Ar(meter)
    inverse = -(-(8 * width.denominator) // width.numerator)
    digits = 0
    if inverse > 1:
        digits = max(0, int(inverse.bit_length() * 0.30103) - 1)
        while ar.pow10(digits) < inverse:
            digits += 1
    scale = ar.pow10(digits)
    meter.unit("enclosure_step")
    low_scaled, high_scaled = ar.mul(lo, scale), ar.mul(hi, scale)
    low = low_scaled.numerator // low_scaled.denominator
    high = -((-high_scaled.numerator) // high_scaled.denominator)
    return low, high, digits


def build_enclosed_quantity(name: str, claim: tuple, exact, forms, accuracy, width: Fraction,
                            meter: Meter) -> QuantityResult:
    """A quantity of a curved preparation: the claim ``[lo, hi]`` (design 4.4, width at most ``3/4`` of ``width``,
    the accuracy asked for or the default width) in the requested forms. ``exact`` is ``(numerator,
    norm_squared)`` when the computation was exact (then the exact form is carried as for planar quantities),
    else ``None``. ``binary64`` and ``decimal`` are outward roundings of the claim (the exact value when
    ``exact``); one that cannot meet ``width`` refuses ``ACCURACY_UNACHIEVABLE`` for that form."""
    ar = Ar(meter)
    lo, hi = claim
    binary64 = decimal = None
    if "binary64" in forms:
        if exact is not None:
            binary64 = binary64_enclosure(exact[0], exact[1], meter, name)
        else:
            binary64 = interval_binary64(lo, hi, meter, name)
        if ar.cmp(ar.sub(ar.exact_float(binary64[1]), ar.exact_float(binary64[0])), width) > 0:
            raise refuse(RefusalCategory.ACCURACY_UNACHIEVABLE,
                         f"binary64 cannot meet the accuracy {float(width)!r} for {name}; its bound is "
                         f"{binary64[1] - binary64[0]!r} (the decimal form remains available)")
    if "decimal" in forms:
        low, high, digits = interval_decimal(lo, hi, width, meter)
        if ar.cmp(Fraction(high - low), ar.mul(width, ar.pow10(digits))) > 0:
            raise refuse(RefusalCategory.ACCURACY_UNACHIEVABLE,
                         f"decimal cannot meet the accuracy {float(width)!r} for {name}")
        meter.unit("digit_conversion", 1)
        d_lo, d_hi = _decimal_string(low, digits, meter), _decimal_string(high, digits, meter)
        decimal = (d_lo, d_hi, width)
    keep_exact = exact is not None
    return QuantityResult(name, exact[0] if keep_exact else None, exact[1] if keep_exact else None, binary64,
                          decimal, accuracy, None if accuracy is None else True, (lo, hi))
