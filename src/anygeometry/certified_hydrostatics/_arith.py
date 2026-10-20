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


# log2(10) = 3.32192809488736...; the two constants bracket it, so the bounds below are exact integer arithmetic
_LOG2_10_LOW = 3321928094
_LOG2_10_HIGH = 3321928095
_LOG2_10_SCALE = 10 ** 9


def digits_exceed(n: int, max_digits: int) -> bool:
    """``abs(n) >= 10 ** max_digits`` (more than ``max_digits`` digits) without ever building ``10 ** max_digits``.

    The bit length decides unless it lies in the narrow band around ``max_digits * log2(10)``; only then the exact
    decimal length is counted (in chunks, see ``int_to_str``)."""
    bits = abs(n).bit_length()
    if bits == 0:
        return False
    # 2**(bits-1) <= abs(n) < 2**bits and 10**d has between floor(d*log2 10)+1 bits and ceil(d*log2 10)+1 bits
    low = max_digits * _LOG2_10_LOW // _LOG2_10_SCALE            # <= max_digits * log2(10)
    high = -(-max_digits * _LOG2_10_HIGH // _LOG2_10_SCALE)       # >= max_digits * log2(10)
    if bits <= low:
        return False                         # abs(n) < 2**bits <= 2**low <= 10**max_digits
    if bits - 1 >= high:
        return True                          # abs(n) >= 2**(bits-1) >= 2**high >= 10**max_digits
    return len(int_to_str(abs(n))) > max_digits


_CHUNK_DIGITS = 500        # str() of <= 500 digits is below every legal limit (>= 640)
_CHUNK = 10 ** _CHUNK_DIGITS


def int_to_str(n: int, tick=None) -> str:
    """Decimal text of an integer of any size, independent of ``sys.set_int_max_str_digits``; ``tick`` (if given) is
    called once per squaring and once per chunk (the checkpoints of this loop)."""
    if n < 0:
        return "-" + int_to_str(-n, tick)
    if n < _CHUNK:
        return str(n)
    powers = [(_CHUNK_DIGITS, _CHUNK)]
    while powers[-1][1] * powers[-1][1] <= n:
        if tick is not None:
            tick()
        digits, power = powers[-1]
        powers.append((2 * digits, power * power))

    def rec(value: int, level: int) -> str:
        if tick is not None:
            tick()
        if value < _CHUNK:
            return str(value)
        while powers[level][1] > value:
            level -= 1
        digits, power = powers[level]
        high, low = divmod(value, power)
        return rec(high, level) + rec(low, level).rjust(digits, "0")

    return rec(n, len(powers) - 1)
