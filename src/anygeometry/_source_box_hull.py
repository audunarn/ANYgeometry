"""Same rounded NURBS enclosure, without an unused Jacobian calculation.

No receipts or persistent memo. Unknown effective implementations use the
original capability path, including its exceptions and override behaviour.
"""
from types import MethodType
from fractions import Fraction
import math

from . import nurbs
from . import surface_capabilities as caps

_SURFACE = nurbs.NurbsSurface
_UNKNOWN = object()


def _keywords(function):
    values = function.__kwdefaults__
    if values is None:
        return None
    if type(values) is not dict or any(type(k) is not str for k in values):
        return _UNKNOWN
    if any(v is not None and type(v) not in (bool, int, float, str) for v in values.values()):
        return _UNKNOWN
    return tuple((k, type(v), v.hex() if type(v) is float else v) for k, v in sorted(values.items()))


_METHODS = ('patch_bounds', 'knot_patch_hull', 'exact_net', 'support_key')
_FUNCTIONS = tuple((owner, name, getattr(owner, name), getattr(owner, name).__code__,
                    getattr(owner, name).__defaults__, _keywords(getattr(owner, name)))
    for owner, names in (
        (_SURFACE, _METHODS),
        (nurbs, ('_knot_interval', '_real', '_exact_chart', '_exact_refine', '_exact_insert',
                 '_exact_span', '_derivative_bound', '_sqrt_up')),
        (caps, ('capabilities', '_nurbs_capabilities', '_nurbs_net', '_down', '_up',
                '_evaluate', '_derivatives', '_transformed')))
    for name in names)
_BINDINGS = tuple((owner, name, getattr(owner, name)) for owner, names in (
    (_SURFACE, ('__getattribute__',)),
    (math, ('ulp', 'isfinite', 'nextafter', 'isqrt', 'inf')),
    (caps.np, ('max', 'abs', 'asarray', 'finfo', 'nextafter', 'errstate', 'inf')),
    (nurbs, ('Fraction', 'NurbsSurface', 'math', 'np')),
    (caps, ('Fraction', 'NurbsSurface', '_FLOAT_MAX', 'math', 'np', 'sys', 'Enclosure',
            'SurfaceCapabilities', 'SurfaceKind')))
    for name in names)


def _context(surface):
    if type(surface) is not _SURFACE:
        return False
    if any(getattr(owner, name) is not value for owner, name, value in _BINDINGS):
        return False
    for owner, name, function, code, defaults, keywords in _FUNCTIONS:
        current = getattr(owner, name)
        if (current is not function or current.__code__ is not code or current.__defaults__ is not defaults
                or keywords is _UNKNOWN or _keywords(current) != keywords):
            return False
    for name in _METHODS:
        method = getattr(surface, name)
        if (type(method) is not MethodType or method.__self__ is not surface
                or method.__func__ is not getattr(_SURFACE, name)):
            return False
    return True


def _canonical_literals(surface):
    """Sufficient regular control-net structure; unknown raw edits take fallback.

    This does not grant native validity or replace the caller's model checks.
    It excludes malformed nets for which omitted derivative work could refuse.
    """
    p, q = surface.degree_u, surface.degree_v
    controls, weights = surface.controls, surface.weights
    if type(p) is not int or type(q) is not int or not (1 <= p <= 3 and 1 <= q <= 3):
        return False
    if type(controls) is not tuple or len(controls) < p+1 or not controls:
        return False
    if type(controls[0]) is not tuple or len(controls[0]) < q+1:
        return False
    n, m = len(controls), len(controls[0])
    finite = lambda x: type(x) is float and math.isfinite(x)
    if any(type(row) is not tuple or len(row) != m or any(
            type(point) is not tuple or len(point) != 3 or not all(map(finite, point))
            for point in row) for row in controls):
        return False
    if weights is not None and (type(weights) is not tuple or len(weights) != n or any(
            type(row) is not tuple or len(row) != m or any(not finite(w) or w <= 0 for w in row)
            for row in weights)):
        return False
    for degree, count, knots, interval in ((p,n,surface.knots_u,surface.u_range),
                                          (q,m,surface.knots_v,surface.v_range)):
        if (type(knots) is not tuple or len(knots) != count+degree+1 or not all(map(finite, knots))
                or any(a > b for a,b in zip(knots,knots[1:])) or knots[degree] >= knots[count]
                or any(x != knots[0] for x in knots[:degree+1])
                or any(x != knots[-1] for x in knots[-degree-1:])
                or any(knots.count(x) != 1 for x in set(knots[degree+1:-degree-1]))
                or type(interval) is not tuple or len(interval) != 2 or not all(map(finite, interval))
                or interval[0] == interval[1] or not all(knots[degree] <= x <= knots[count] for x in interval)):
            return False
    return True


def source_enclosure(surface, box):
    """Return the original Enclosure exactly, or call the original path."""
    if not _context(surface) or not _canonical_literals(surface):
        return caps.capabilities(surface).enclosure(box)
    # Same order and operations as _nurbs_capabilities, including NumPy's
    # floating margin, followed by the same interval mapping as patch_bounds.
    scale = caps.np.max(caps.np.abs(caps.np.asarray(surface.controls, dtype=float)), axis=(0, 1))
    margin = 16 * (surface.degree_u + surface.degree_v + 2) * caps.np.finfo(float).eps * scale
    (u0,u1),(v0,v1) = box
    u = nurbs._knot_interval(u0,u1,surface.u_range)
    v = nurbs._knot_interval(v0,v1,surface.v_range)
    if u[0] >= u[1] or v[0] >= v[1]:
        return caps.capabilities(surface).enclosure(box)
    net = caps._nurbs_net(surface)
    # The capability cache is mutable and not authority for specialization.
    # Unknown/tampered cached nets must retain the complete original path.
    fresh = surface.exact_net()
    if (type(net) is not tuple or any(type(row) is not tuple or any(
            type(point) is not tuple or any(type(x) is not Fraction for x in point)
            for point in row) for row in net) or net != fresh):
        return caps.capabilities(surface).enclosure(box)
    points = surface.knot_patch_hull(u,v,net)
    lower = tuple(min(point[k] for point in points) for k in range(3))
    upper = tuple(max(point[k] for point in points) for k in range(3))
    with caps.np.errstate(over='ignore'):
        return caps.Enclosure(
            tuple(float(caps.np.nextafter(caps._down(value)-m, -caps.np.inf)) for value,m in zip(lower,margin)),
            tuple(float(caps.np.nextafter(caps._up(value)+m, caps.np.inf)) for value,m in zip(upper,margin)))
