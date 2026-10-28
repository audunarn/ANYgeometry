"""Value-only canonical Q4 point probes; no proof or cache authority.

The interval expressions deliberately retain the original fan operation
order, including dependency overestimation and the corner-hull intersection.
"""
from fractions import Fraction as Q
import math

from . import curvilinear_reference_maps as maps
from . import regular_source_reference as source
from . import nurbs_metric as metric

_SIDE = source.SourceSide
_UNKNOWN = object()


def _keywords(function):
    values = function.__kwdefaults__
    if values is None:
        return None
    if type(values) is not dict or any(type(k) is not str for k in values):
        return _UNKNOWN
    if any(v is not None and type(v) not in (bool,int,float,str) for v in values.values()):
        return _UNKNOWN
    return tuple((k,type(v),v.hex() if type(v) is float else v) for k,v in sorted(values.items()))


_FUNCTIONS = tuple((maps,name,function,code,defaults,
                    None if keywords is None else tuple((k,type(v),v.hex() if type(v) is float else v)
                                                         for k,v in sorted(keywords)))
    for name,function,code,defaults,keywords in maps._POINT_FAN_FUNCTIONS)
# scale's effective global mul is in the metric module, not the maps alias.
_FUNCTIONS += tuple((metric,name,function,code,defaults,keywords)
    for _owner,name,function,code,defaults,keywords in _FUNCTIONS if name in ('add','sub','mul','scale'))
_GETATTRIBUTE = _SIDE.__getattribute__
_CURVE_DESCRIPTOR = vars(_SIDE).get('curve')


def _context(boundary):
    if (source.SourceSide is not _SIDE or type(boundary) is not _SIDE
            or _SIDE.__getattribute__ is not _GETATTRIBUTE or vars(_SIDE).get('curve') is not _CURVE_DESCRIPTOR
            or boundary.curve is not None
            or maps.Q is not Q or metric.Q is not Q):
        return False
    for owner,name,function,code,defaults,keywords in _FUNCTIONS:
        current = getattr(owner,name)
        if (current is not function or current.__code__ is not code or current.__defaults__ is not defaults
                or keywords is _UNKNOWN or _keywords(current) != keywords):
            return False
    return True


def canonical_point_values(uv_boxes, boundary, side, s, t, xyz):
    """(source UV interval, FE XYZ interval) or None for the original route.

    Only the two current point probes and literal exact straight-side inputs
    are specialized. No boundary box/derivative call occurs on that old path.
    """
    if not _context(boundary) or type(side) is not int or not 0 <= side < 4:
        return None
    if type(s) is not Q or type(t) is not Q or s not in (0,1) or t != Q(1,2):
        return None
    if (type(uv_boxes) is not tuple or len(uv_boxes) != 4 or any(
            type(point) is not tuple or len(point) != 2 or any(
                type(box) is not tuple or len(box) != 2 or any(type(x) is not Q for x in box)
                or box[0] > box[1] for box in point) for point in uv_boxes)):
        return None
    if (type(xyz) is not tuple or len(xyz) != 4 or any(
            type(point) is not tuple or len(point) != 3 or any(
                type(x) not in (Q,int,float) or (type(x) is float and not math.isfinite(x))
                for x in point) for point in xyz)):
        return None
    corners = ((-1,-1),(1,-1),(1,1),(-1,1))
    first,last = corners[side],corners[(side+1)%4]
    edge = tuple(maps.interval((t,t),Q(a),Q(b)) for a,b in zip(first,last))
    one_minus_s = (1-s,1-s)
    reference = tuple(maps.mul(one_minus_s,x) for x in edge)
    base = []
    for a,b,c,d in zip(*uv_boxes):
        c0 = maps.scale(maps.add(maps.add(a,b),maps.add(c,d)),Q(1,4))
        cx = maps.scale(maps.add(maps.sub(b,a),maps.sub(c,d)),Q(1,4))
        cy = maps.scale(maps.add(maps.sub(c,b),maps.sub(d,a)),Q(1,4))
        cxy = maps.scale(maps.sub(maps.add(a,c),maps.add(b,d)),Q(1,4))
        x,y = reference
        base.append(maps.add(maps.add(c0,maps.mul(cx,x)),
                             maps.add(maps.mul(cy,y),maps.mul(cxy,maps.mul(x,y)))))
    base = [(max(value[0],min(p[axis][0] for p in uv_boxes)),
             min(value[1],max(p[axis][1] for p in uv_boxes))) for axis,value in enumerate(base)]
    correction = (maps.constant(Q(0)),)*2
    uv = tuple(maps.add(b,maps.mul(one_minus_s,e)) for b,e in zip(base,correction))
    output = []
    for values in zip(*xyz):
        a,b,c,d = map(Q,values)
        c0,cx,cy,cxy = (a+b+c+d)/4,(-a+b+c-d)/4,(-a-b+c+d)/4,(a-b+c-d)/4
        x,y = reference
        output.append(maps.add(maps.add(maps.constant(c0),maps.scale(x,cx)),
                      maps.add(maps.scale(y,cy),maps.scale(maps.mul(x,y),cxy))))
    return uv,tuple(output)
