"""Exact owner-loop qualification for embedded scalar intersection branches.

This first connected family admits a complete simple closed branch strictly
inside an authored chart rectangle. Crosscuts, multiple/nested traces and
pretrimmed operands remain explicit refusals until their event graph is proved.
"""
from fractions import Fraction as Q

from .errors import GeometryError
from .nurbs import CurveOnSurface, _IDENTITY
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
from .nurbs_surface_intersection._intervals import Interval


def has_intersection_edges(model,face):
    return any(isinstance(model.edges[e.edge].curve,NurbsSurfaceIntersectionCurve)
               for loop in (face.loop,*face.holes) for e in loop)


def rectangle_loop(model,loop,surface):
    if len(loop) != 4:
        raise GeometryError('NURBS pair imprint requires an authored complete chart rectangle')
    points = []
    ends = []
    for use in loop:
        curve = model.edges[use.edge].curve
        if (not isinstance(curve,CurveOnSurface) or curve.support != surface or curve.transform != _IDENTITY
                or curve.pcurve.degree != 1 or len(curve.pcurve.controls) != 2):
            raise GeometryError('NURBS pair imprint requires exact chart-segment source boundaries')
        a,b = curve.pcurve.exact_point(0),curve.pcurve.exact_point(1)
        if not use.forward:
            a,b = b,a
        points.append(tuple(a)); ends.append(tuple(b))
    corners = {(Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1)),(Q(0),Q(1))}
    if set(points) != corners or any(b != points[(i+1)%4] for i,b in enumerate(ends)):
        raise GeometryError('NURBS pair imprint source is not its complete selected chart rectangle')
    area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,ends))/2
    if area == 0:
        raise GeometryError('authored chart rectangle has zero oriented area')
    return area


def qualify_closed_loop(model,loop,surface):
    curves = [model.edges[e.edge].curve for e in loop]
    if len(curves) < 3 or not all(isinstance(c,NurbsSurfaceIntersectionCurve) for c in curves):
        raise GeometryError('intersection loop must consist of at least three persistent owner intervals')
    owner = curves[0].owner
    if not owner.closed or any(c.owner != owner for c in curves):
        raise GeometryError('intersection loop must cover one complete closed owner branch')
    if not owner.segments or any(s[0] != 'scalar' for s in owner.segments):
        raise GeometryError('generic chart embedding is not yet qualified for fragment publication')
    intervals = []
    oriented = []
    for use,curve in zip(loop,curves):
        view = curve.for_support(surface)
        view._shape()  # canonical complete topology and exact support correspondence
        intervals.append(tuple(sorted(curve.domain)))
        oriented.append(curve.domain if use.forward else tuple(reversed(curve.domain)))
    reach = Q(0)
    for a,b in sorted(intervals):
        if a != reach or b <= a:
            raise GeometryError('intersection owner intervals overlap or omit a source interval')
        reach = b
    if reach != 1:
        raise GeometryError('intersection loop does not cover its entire source branch')
    for (a,b),(c,d) in zip(oriented,oriented[1:]+oriented[:1]):
        if b != c and {b,c} != {Q(0),Q(1)}:
            raise GeometryError('intersection loop source occurrences are discontinuous')
    boxes = [curve.for_support(surface).chart_box() for curve in curves]
    if not all(0 < lo <= hi < 1 for box in boxes for lo,hi in box):
        raise GeometryError('closed intersection branch is not strictly inside its authored support rectangle')
    return owner


def _loop_area_key(model,loop,surface):
    from .nurbs_surface_intersection import SourceDefinition
    definitions = []
    for use in loop:
        curve = model.edges[use.edge].curve
        if isinstance(curve,NurbsSurfaceIntersectionCurve):
            definitions.append((curve.source_key,tuple(curve.domain),use.forward))
        else:
            from .nurbs import nurbs_record
            def literal(value):
                if isinstance(value,dict):
                    return tuple(sorted((k,literal(v)) for k,v in value.items()))
                if isinstance(value,(list,tuple)):
                    return tuple(literal(v) for v in value)
                return value
            definitions.append((literal(nurbs_record(curve)),use.forward))
    return ('intersection-loop-green',SourceDefinition.capture(surface).values,tuple(definitions))


def _uncached_loop_area(model,loop,surface):
    """Signed Green integral interval, using continuous chart/derivative proofs."""
    if all(isinstance(model.edges[e.edge].curve,CurveOnSurface) for e in loop):
        area = rectangle_loop(model,loop,surface)
        return area,area
    with intersection_curve_operation():
        qualify_closed_loop(model,loop,surface)
        for subdivisions in (2,4,8,16,32):
            total = Interval(0)
            for use in loop:
                curve = model.edges[use.edge].curve.for_support(surface)
                for i in range(subdivisions):
                    a,b = Q(i,subdivisions),Q(i+1,subdivisions)
                    for bound in curve.enclose(a,b):
                        chart = (bound.first_chart,bound.second_chart)[curve.chart_support]
                        derivative = (bound.first_chart_derivative,bound.second_chart_derivative)[curve.chart_support]
                        u,v = (Interval(*row) for row in chart)
                        integral = (u*derivative[1]-v*derivative[0]).scale(
                            (bound.parameter_range[1]-bound.parameter_range[0])/2)
                        total = total+integral.scale(1 if use.forward else -1)
            if total.excludes_zero():
                return total.lo,total.hi
    raise GeometryError('intersection loop orientation is not certified within its work bound')


from .nurbs_intersection_curve import operation_cached
loop_area = operation_cached(_loop_area_key)(_uncached_loop_area)


def validate_intersection_face(model,face_id):
    face = model.faces[face_id]
    try:
        with intersection_curve_operation():
            loops = (face.loop,*face.holes)
            flags = [any(isinstance(model.edges[e.edge].curve,NurbsSurfaceIntersectionCurve) for e in loop) for loop in loops]
            if flags[0]:
                if len(loops) != 1:
                    raise GeometryError('nested intersection loops are not yet qualified')
                qualify_closed_loop(model,face.loop,face.surface)
                loop_area(model,face.loop,face.surface)
            else:
                rectangle_loop(model,face.loop,face.surface)
                if len(face.holes) != 1:
                    raise GeometryError('one complete interior intersection loop is required')
                qualify_closed_loop(model,face.holes[0],face.surface)
                outer = loop_area(model,face.loop,face.surface)
                hole = loop_area(model,face.holes[0],face.surface)
                if (outer[0] > 0) == (hole[0] > 0):
                    raise GeometryError('intersection hole orientation matches its outer boundary')
    except GeometryError as error:
        return [f'face {face_id}: {error}']
    return []
