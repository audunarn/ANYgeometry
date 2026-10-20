"""Bit-capped exact arithmetic and unlimited-digit integer text (design section 7).

``Ar`` checks the *raw* size of every product, sum and quotient before it is formed (sum of the operand bit
lengths, numerator and denominator separately), then checks the reduced result, so an intermediate that is
larger than the cap is refused even when it would cancel. ``int_to_str`` converts integers without the
interpreter's global int/str digit limit."""
from __future__ import annotations

from fractions import Fraction

from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._work import Meter

__all__ = ["Ar", "int_to_str"]


def _parts(x) -> tuple:
    return abs(x.numerator).bit_length(), x.denominator.bit_length()


class Ar:
    """Metered, bit-capped arithmetic on ints and Fractions bound to one ``Meter``."""

    __slots__ = ("meter", "budget")

    def __init__(self, meter: Meter) -> None:
        self.meter = meter
        self.budget = meter.budget

    def guard(self, bits: int) -> None:
        budget = self.budget
        if bits > budget._bits:
            budget._bits = bits
        if budget.max_bits is not None and bits > budget.max_bits:
            raise CertifiedHydrostaticsRefusal(
                RefusalCategory.BIT_LENGTH_EXCEEDED,
                f"an exact intermediate needs {bits} bits, the cap is {budget.max_bits}")

    def fit(self, value):
        return self.meter.fit(value)

    def mul(self, a, b):
        an, ad = _parts(a)
        bn, bd = _parts(b)
        self.guard(max(an + bn, ad + bd))
        return self.meter.fit(a * b)

    def add(self, a, b):
        an, ad = _parts(a)
        bn, bd = _parts(b)
        self.guard(max(an + bd, bn + ad, ad + bd) + 1)
        return self.meter.fit(a + b)

    def sub(self, a, b):
        an, ad = _parts(a)
        bn, bd = _parts(b)
        self.guard(max(an + bd, bn + ad, ad + bd) + 1)
        return self.meter.fit(a - b)

    def div(self, a, b):
        an, ad = _parts(a)
        bn, bd = _parts(b)
        self.guard(max(an + bd, ad + bn))
        return self.meter.fit(Fraction(a) / b)

    def cmp(self, a, b) -> int:
        """Sign of ``a - b``; the cross-multiplication of the comparison is size-guarded first."""
        an, ad = _parts(a)
        bn, bd = _parts(b)
        self.guard(max(an + bd, bn + ad))
        return (a > b) - (a < b)

    def exact_float(self, f: float) -> Fraction:
        """The exact value of a double, size-guarded before the Fraction is formed."""
        numerator, denominator = f.as_integer_ratio()
        self.guard(max(abs(numerator).bit_length(), denominator.bit_length()))
        return Fraction(numerator, denominator)

    def sub3(self, a, b):
        return (self.sub(a[0], b[0]), self.sub(a[1], b[1]), self.sub(a[2], b[2]))

    def dot(self, a, b):
        return self.add(self.add(self.mul(a[0], b[0]), self.mul(a[1], b[1])), self.mul(a[2], b[2]))

    def cross(self, a, b):
        return (self.sub(self.mul(a[1], b[2]), self.mul(a[2], b[1])),
                self.sub(self.mul(a[2], b[0]), self.mul(a[0], b[2])),
                self.sub(self.mul(a[0], b[1]), self.mul(a[1], b[0])))

    def det(self, a, b, c):
        return self.dot(a, self.cross(b, c))

    def isqrt(self, n: int) -> int:
        """Integer square root; one unit per 1024 bits of the radicand, reserved one at a time."""
        import math
        self.guard(n.bit_length())
        for _ in range(1 + n.bit_length() // 1024):
            self.meter.unit("sqrt_step")
        return math.isqrt(n)

    def pow10(self, digits: int) -> int:
        self.guard(int(digits * 3.3220) + 1)
        return 10 ** digits


_CHUNK_DIGITS = 500        # str() of <= 500 digits is below every legal limit (>= 640)
_CHUNK = 10 ** _CHUNK_DIGITS


def int_to_str(n: int) -> str:
    """Decimal text of an integer of any size, independent of ``sys.set_int_max_str_digits``."""
    if n < 0:
        return "-" + int_to_str(-n)
    if n < _CHUNK:
        return str(n)
    powers = [(_CHUNK_DIGITS, _CHUNK)]
    while powers[-1][1] * powers[-1][1] <= n:
        digits, power = powers[-1]
        powers.append((2 * digits, power * power))

    def rec(value: int, level: int) -> str:
        if value < _CHUNK:
            return str(value)
        while powers[level][1] > value:
            level -= 1
        digits, power = powers[level]
        high, low = divmod(value, power)
        return rec(high, level) + rec(low, level).rjust(digits, "0")

    return rec(n, len(powers) - 1)
