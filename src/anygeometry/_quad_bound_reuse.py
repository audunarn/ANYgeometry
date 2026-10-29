"""One-operation reuse of pure quadratic Q4 algebra; no receipt authority."""
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from types import FunctionType, MethodType
import math

from . import nurbs, polynomial_cell_bounds as polynomial
from . import curvilinear_reference_maps as maps

_ACTIVE = ContextVar('quadratic_quad_bound_operation', default=None)
_SURFACE = nurbs.NurbsSurface
_UNSUPPORTED = object()


def _keyword_defaults(function):
    values = function.__kwdefaults__
    if values is None:
        return None
    if type(values) is not dict or any(type(key) is not str for key in values):
        return _UNSUPPORTED
    # Registered defaults are literal scalars. Unknown/mutable defaults take
    # the fresh route; their equality operators are never invoked here.
    if any(value is not None and type(value) not in (bool, int, float, str) for value in values.values()):
        return _UNSUPPORTED
    return tuple((key, type(value), value.hex() if type(value) is float else value)
                 for key, value in sorted(values.items()))


_FIELDS = ('degree_u', 'degree_v', 'knots_u', 'knots_v', 'controls', 'weights', 'u_range', 'v_range')
_FUNCTIONS = tuple((polynomial, name, value, value.__code__, value.__defaults__)
    for name, value in vars(polynomial).items() if type(value) is FunctionType)
_FUNCTIONS += tuple((owner, name, getattr(owner, name), getattr(owner, name).__code__,
                    getattr(owner, name).__defaults__) for owner, name in (
    (nurbs.NurbsSurface, 'exact_point'), (nurbs.NurbsSurface, 'exact_net'),
    (nurbs, '_exact_chart'), (nurbs, '_exact_span'), (nurbs, '_exact_basis'), (maps, '_norm_upper')))
_ISQRT = math.isqrt
_CEIL = math.ceil
_KEYWORD_DEFAULTS = tuple(_keyword_defaults(row[2]) for row in _FUNCTIONS)


@contextmanager
def quad_bound_operation():
    # Nested public validations get their own storage and cannot poison an
    # outer operation. At most one complete literal bound is retained.
    state = {'entry': None, 'hits': 0, 'calculations': 0}
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def _context(support):
    if (nurbs.NurbsSurface is not _SURFACE or type(support) is not _SURFACE
            or math.isqrt is not _ISQRT or math.ceil is not _CEIL):
        return False
    if nurbs.Fraction is not Fraction or polynomial.Q is not Fraction or maps.Q is not Fraction:
        return False
    for (owner, name, function, code, defaults), keywords in zip(_FUNCTIONS, _KEYWORD_DEFAULTS):
        actual = getattr(owner, name)
        if actual is not function or actual.__code__ is not code or actual.__defaults__ is not defaults:
            return False
        if keywords is _UNSUPPORTED or _keyword_defaults(actual) != keywords:
            return False
    for name in ('exact_point', 'exact_net'):
        method = getattr(support, name)
        if (type(method) is not MethodType or method.__self__ is not support
                or method.__func__ is not getattr(nurbs.NurbsSurface, name)):
            return False
    return True


def _literal(value):
    if value is None:
        return ('none',)
    if type(value) is int:
        return ('int', value)
    if type(value) is float and math.isfinite(value):
        return ('float', value.hex())
    if type(value) is Fraction:
        return ('fraction', value.numerator, value.denominator)
    if type(value) is tuple:
        return ('tuple', tuple(_literal(item) for item in value))
    raise TypeError('nonliteral quadratic bound input')


def _key(support, uv, xyz, orientation):
    try:
        return _literal((tuple(getattr(support, name) for name in _FIELDS), uv, xyz, orientation))
    except (TypeError, AttributeError):
        return None


def quadratic_quad_bound(support, uv, xyz, orientation):
    state = _ACTIVE.get()
    if state is None or not _context(support):
        return polynomial.quad_residual_bound(support, uv, xyz, orientation)
    p, q = support.degree_u, support.degree_v
    if (type(p) is not int or type(q) is not int or p not in (1, 2) or q not in (1, 2)
            or type(support.controls) is not tuple or len(support.controls) != p+1
            or any(type(row) is not tuple or len(row) != q+1 for row in support.controls)):
        # Do not serialize potentially large cubic/multispan nets for a cache
        # whose only reusable calculation is the single-span quadratic branch.
        return polynomial.quad_residual_bound(support, uv, xyz, orientation)
    key = _key(support, uv, xyz, orientation)
    if key is None:
        return polynomial.quad_residual_bound(support, uv, xyz, orientation)
    entry = state['entry']
    if entry is not None and entry[0] == key:
        state['hits'] += 1
        return entry[1]
    # Only the pure quadratic branch is reusable. Cubic/general routes may
    # charge Work internally and must keep their original fresh execution.
    boxes = tuple(polynomial._box(point) for point in uv)
    if not all(polynomial._inside(box) for box in boxes):
        return polynomial.quad_residual_bound(support, uv, xyz, orientation)
    table = polynomial._quadratic_coefficients(support)
    if table is None:
        return polynomial.quad_residual_bound(support, uv, xyz, orientation)
    state['calculations'] += 1
    result = polynomial._quad_residual_from_table(table, boxes, xyz, orientation)
    # No callbacks occur in this pure calculation. Recheck literal/context
    # bindings before storing, so an unsupported implementation never seeds it.
    if (type(result) is Fraction and result >= 0 and _context(support)
            and _key(support, uv, xyz, orientation) == key):
        state['entry'] = (key, result)
    return result
