"""The half-space query (design section 3): validated exactly, immutable.

All numbers are converted to exact rationals when the query is built: ints and ``Fraction`` as given,
floats (and numpy scalars) by their exact binary value (``Fraction(float)``), never through decimal text
or normalisation. A query that exists is therefore valid: finite, ``n != 0`` and ``(e1 x e2) . n != 0``."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Mapping, Sequence

from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory

__all__ = ["HalfSpaceQuery", "QUANTITY_NAMES", "FORM_NAMES", "VOLUME_NAMES", "WATERPLANE_NAMES"]

VOLUME_NAMES = ("V", "Jx", "Jy", "Jz")
WATERPLANE_NAMES = ("A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")   # area, first moments, second moments u^2, v^2, uv
QUANTITY_NAMES = VOLUME_NAMES + WATERPLANE_NAMES
FORM_NAMES = ("exact", "binary64", "decimal")


def _refusal(category: RefusalCategory, message: str) -> CertifiedHydrostaticsRefusal:
    return CertifiedHydrostaticsRefusal(category, message, phase="query")


def _rational(value, what: str) -> Fraction:
    if isinstance(value, bool):
        raise _refusal(RefusalCategory.INVALID_QUERY, f"{what} is a bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise _refusal(RefusalCategory.NONFINITE_INPUT, f"{what} is not finite")
        return Fraction(number)
    raise _refusal(RefusalCategory.INVALID_QUERY, f"{what} is not a number ({type(value).__name__})")


def _vector(value, what: str) -> tuple:
    try:
        items = tuple(value)
    except TypeError:
        raise _refusal(RefusalCategory.INVALID_QUERY, f"{what} is not a sequence of three numbers") from None
    if len(items) != 3:
        raise _refusal(RefusalCategory.INVALID_QUERY, f"{what} needs three components, got {len(items)}")
    return tuple(_rational(item, f"{what}[{i}]") for i, item in enumerate(items))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass(frozen=True)
class HalfSpaceQuery:
    """``n . r <= d`` with the waterplane basis ``(origin, e1, e2)``, requested accuracies and forms.

    ``absolute_accuracy`` maps quantity names (``V``, ``Jx``, ``Jy``, ``Jz``, ``A``, ``Mu``, ``Mv``,
    ``Iuu``, ``Ivv``, ``Iuv``; ``J`` stands for the three ``J`` components) to positive numbers.
    ``forms`` is a non-empty subset of ``exact``, ``binary64``, ``decimal`` (stored in that order).
    After construction every numeric field is an exact ``Fraction``; ``absolute_accuracy`` is a tuple of
    ``(name, Fraction)`` pairs in quantity order."""

    normal: tuple
    offset: Fraction
    waterplane_origin: tuple = (0, 0, 0)
    waterplane_axes: tuple = ((1, 0, 0), (0, 1, 0))
    absolute_accuracy: Mapping | Sequence = ()
    forms: Sequence = ("exact",)

    def __post_init__(self) -> None:
        normal = _vector(self.normal, "normal")
        offset = _rational(self.offset, "offset")
        origin = _vector(self.waterplane_origin, "waterplane_origin")
        try:
            axes = tuple(self.waterplane_axes)
        except TypeError:
            axes = ()
        if len(axes) != 2:
            raise _refusal(RefusalCategory.INVALID_QUERY, "waterplane_axes needs two vectors")
        e1, e2 = _vector(axes[0], "waterplane_axes[0]"), _vector(axes[1], "waterplane_axes[1]")
        if not any(normal):
            raise _refusal(RefusalCategory.INVALID_QUERY, "the normal is the zero vector")
        if _dot(_cross(e1, e2), normal) == 0:
            raise _refusal(RefusalCategory.INVALID_QUERY, "(e1 x e2) . n is zero: the waterplane basis is degenerate")
        forms = self.forms
        if isinstance(forms, str):
            forms = (forms,)
        forms = tuple(forms)
        if not forms or len(set(forms)) != len(forms) or any(f not in FORM_NAMES for f in forms):
            raise _refusal(RefusalCategory.INVALID_QUERY, f"forms must be a non-empty subset of {FORM_NAMES}")
        items = self.absolute_accuracy.items() if isinstance(self.absolute_accuracy, Mapping) else self.absolute_accuracy
        accuracy: dict = {}
        for name, value in items:
            names = ("Jx", "Jy", "Jz") if name == "J" else (name,)
            for each in names:
                if each not in QUANTITY_NAMES:
                    raise _refusal(RefusalCategory.INVALID_QUERY, f"unknown quantity {name!r} in absolute_accuracy")
                if each in accuracy:
                    raise _refusal(RefusalCategory.INVALID_QUERY, f"accuracy for {each} given twice")
                width = _rational(value, f"accuracy of {name}")
                if width <= 0:
                    raise _refusal(RefusalCategory.INVALID_QUERY, f"accuracy of {name} must be positive")
                accuracy[each] = width
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "waterplane_origin", origin)
        object.__setattr__(self, "waterplane_axes", (e1, e2))
        object.__setattr__(self, "absolute_accuracy", tuple((name, accuracy[name]) for name in QUANTITY_NAMES
                                                            if name in accuracy))
        object.__setattr__(self, "forms", tuple(f for f in FORM_NAMES if f in forms))

    def accuracy(self, name: str):
        """The requested absolute accuracy of ``name`` (a Fraction) or ``None``."""
        for key, value in self.absolute_accuracy:
            if key == name:
                return value
        return None

    @property
    def normal_squared(self) -> Fraction:
        return _dot(self.normal, self.normal)

    def as_dict(self) -> dict:
        """Canonical JSON-able form (rationals as ``[numerator, denominator]``)."""
        def pair(x: Fraction) -> list:
            return [x.numerator, x.denominator]

        return {"normal": [pair(c) for c in self.normal], "offset": pair(self.offset),
                "waterplane_origin": [pair(c) for c in self.waterplane_origin],
                "waterplane_axes": [[pair(c) for c in axis] for axis in self.waterplane_axes],
                "absolute_accuracy": {name: pair(value) for name, value in self.absolute_accuracy},
                "forms": list(self.forms)}
