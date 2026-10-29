"""Finite canonical rational admission; no authority from a live module alias."""
from fractions import Fraction


def fraction_kind(alias, _canonical=Fraction):
    return _canonical if alias is _canonical else None


_DECLARATION = (fraction_kind, fraction_kind.__code__, fraction_kind.__defaults__)


def binding(aliases, _declaration=_DECLARATION):
    """Validate the import-time declaration and consumer import edges."""
    if _DECLARATION is not _declaration:
        return None
    function, code, defaults = _declaration
    if (fraction_kind is not function or function.__code__ is not code
            or function.__defaults__ is not defaults):
        return None
    for alias in aliases:
        if alias is not function:
            return None
    return _declaration
