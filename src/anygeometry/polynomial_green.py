"""Exact Bernstein Green integrals for registered polynomial world curves.

No fitted controls, display geometry or cached evaluation samples are used.
Unsupported rational/implicit families return None to the existing backend.
"""
from fractions import Fraction as Q
import math

from .errors import GeometryError


def _checkpoint(cancellation_check=None):
    from .plane_nurbs import charge_box
    if cancellation_check is not None and cancellation_check():
        raise GeometryError('polynomial Green area cancelled')
    charge_box()


def _uniform(weights):
    return weights is None or len(set(Q(w) for w in weights)) == 1


def bezier_spans(degree, knots, controls, start, end, *, cancellation_check=None):
    """Exact oriented Bezier pieces of an actual polynomial B-spline range."""
    from .nurbs import _exact_insert
    _checkpoint(cancellation_check)
    degree = int(degree)
    knots = list(map(Q,knots))
    points = [tuple(map(Q,p)) for p in controls]
    start,end = Q(start),Q(end)
    lo,hi = sorted((start,end))
    if not knots[degree] <= lo < hi <= knots[len(points)]:
        raise GeometryError('polynomial area range is outside the registered knot domain')
    for knot in sorted({lo,hi}|{x for x in knots if lo < x < hi}):
        if knots[degree] < knot < knots[len(points)]:
            for _ in range(max(0,degree-knots.count(knot))):
                _checkpoint(cancellation_check)
                knots,points = _exact_insert(degree,knots,points,knot,1)
    pieces = []
    for k in range(degree,len(points)):
        if lo <= knots[k] < knots[k+1] <= hi:
            _checkpoint(cancellation_check)
            pieces.append(tuple(points[k-degree:k+1]))
    if not pieces:
        raise GeometryError('polynomial area range has no nonzero span')
    if end < start:
        pieces = [piece[::-1] for piece in pieces[::-1]]
    return tuple(pieces)


def bernstein_green(controls, *, cancellation_check=None):
    """Exact 1/2 integral(x dy-y dx), plus a coefficient absolute bound."""
    _checkpoint(cancellation_check)
    degree = len(controls)-1
    if degree < 1:
        return Q(0),Q(0)
    value = magnitude = Q(0)
    for i,(x,y) in enumerate(controls):
        for j in range(degree):
            _checkpoint(cancellation_check)
            dx = controls[j+1][0]-controls[j][0]
            dy = controls[j+1][1]-controls[j][1]
            weight = Q(math.comb(degree,i)*math.comb(degree-1,j),4*math.comb(2*degree-1,i+j))
            term = (x*dy-y*dx)*weight
            value += term
            magnitude += (abs(x*dy)+abs(y*dx))*weight
    return value,magnitude


def _world_pieces(curve, *, cancellation_check=None):
    from .arrangement_geometry import LinePath,BezierPath
    from .line_ranges import LineRange
    from .nurbs import NurbsCurve,CurveOnSurface,_exact_basis
    if type(curve) in (LinePath,BezierPath,LineRange,NurbsCurve,CurveOnSurface):
        _checkpoint(cancellation_check)
    if type(curve) is LinePath:
        return ((tuple(map(Q,curve.start)),tuple(map(Q,curve.end))),)
    if type(curve) is BezierPath:
        return (tuple(tuple(map(Q,p)) for p in curve.controls),)
    if type(curve) is LineRange:
        return ((curve.exact_point(0),curve.exact_point(1)),)
    if type(curve) is NurbsCurve:
        if curve.dimension != 3 or not _uniform(curve.weights):
            return None
        from .nurbs_runtime_binding import validate_curve_runtime
        validate_curve_runtime(curve)
        return bezier_spans(curve.degree,curve.knots,curve.controls,*curve.domain,cancellation_check=cancellation_check)
    if type(curve) is not CurveOnSurface:
        return None
    p,s = curve.pcurve,curve.support
    if (p.degree != 1 or len(p.controls) != 2 or not _uniform(p.weights)
            or s.weights is not None and len({Q(w) for row in s.weights for w in row}) != 1):
        return None
    from .nurbs_runtime_binding import validate_curve_runtime
    validate_curve_runtime(p)
    fixed = [axis for axis in (0,1) if p.controls[0][axis] == p.controls[1][axis]]
    if len(fixed) != 1:
        return None  # arbitrary polynomial compositions stay on the existing path
    fixed = fixed[0]; running = 1-fixed
    ranges = s.u_range,s.v_range
    degrees = s.degree_u,s.degree_v
    knots = s.knots_u,s.knots_v
    at = Q(ranges[fixed][0])+Q(p.controls[0][fixed])*(Q(ranges[fixed][1])-Q(ranges[fixed][0]))
    degree = degrees[fixed]; count = len(s.controls) if fixed == 0 else len(s.controls[0])
    span = next((i for i in range(degree,count) if Q(knots[fixed][i]) <= at < Q(knots[fixed][i+1])),count-1)
    basis = _exact_basis(degree,list(map(Q,knots[fixed])),span,at)
    controls = []
    for k in range(len(s.controls[0]) if fixed == 0 else len(s.controls)):
        _checkpoint(cancellation_check)
        values = [s.controls[span-degree+j][k] if fixed == 0 else s.controls[k][span-degree+j]
                  for j in range(degree+1)]
        point = tuple(sum(w*Q(v[axis]) for w,v in zip(basis,values)) for axis in range(3))
        matrix = curve.transform
        controls.append(tuple(sum(Q(matrix[i][j])*point[j] for j in range(3))+Q(matrix[i][3]) for i in range(3)))
    ends = tuple(p.exact_point(t)[running] for t in (0,1))
    mapped = tuple(Q(ranges[running][0])+t*(Q(ranges[running][1])-Q(ranges[running][0])) for t in ends)
    return bezier_spans(degrees[running],knots[running],controls,*mapped,cancellation_check=cancellation_check)


def _maximum(controls,axis):
    return max(abs(p[axis]) for p in controls)


def _derivative_controls(controls,width):
    degree = len(controls)-1
    if not degree:
        return ((Q(0),)*len(controls[0]),)
    return tuple(tuple(degree*(b-a)/width for a,b in zip(first,last))
                 for first,last in zip(controls,controls[1:]))


def _round_bound(value):
    if not value:
        return Q(0)
    from .nurbs import _float_up
    rounded = _float_up(abs(value))
    if not math.isfinite(rounded):
        raise GeometryError('section Green rounding enclosure is not finite')
    return Q(math.ulp(rounded))/2


def section_green(frame,curve, *, cancellation_check=None):
    """Exact structural image integral plus current midpoint/tangent-law error.

    Only uniform polynomial translational tensors and profile-u slicing are
    registered. No evaluator, authored curve, endpoint or parameter is changed.
    """
    from .nurbs_runtime_binding import validate_section_runtime
    _checkpoint(cancellation_check)
    surface = curve.surface
    if (surface.degree_v != 1 or len(surface.knots_v) != 4
            or surface.knots_v[:2] != (surface.knots_v[0],)*2
            or surface.knots_v[2:] != (surface.knots_v[2],)*2
            or any(len(row) != 2 for row in surface.controls)
            or surface.weights is not None and len({Q(w) for row in surface.weights for w in row}) != 1):
        return None
    profile = tuple(tuple(map(Q,row[0])) for row in surface.controls)
    direction = tuple(Q(b)-Q(a) for a,b in zip(*surface.controls[0]))
    if not any(direction) or any(tuple(Q(b)-Q(a) for a,b in zip(*row)) != direction for row in surface.controls):
        return None
    validate_section_runtime(curve)
    normal,offset,_norm2,_slab2,slab = curve._plane_slab()
    denominator = sum(a*b for a,b in zip(normal,direction))
    if not denominator or any(frame.normal[i]*normal[j] != frame.normal[j]*normal[i]
                              for i in range(3) for j in range(3)):
        return None
    if sum(n*x for n,x in zip(normal,frame.origin)) != offset:
        return None
    arcs = curve._arcs()
    start,end = map(Q,curve.domain)
    low,high = sorted((start,end))
    total = error = magnitude = Q(0)
    world_scale = tuple(max(abs(Q(p[i])) for row in surface.controls for p in row) for i in range(3))
    projection = tuple(tuple(n/frame._scale for n in row) for row in (frame._dual_u,frame._dual_v))
    point_round = tuple(sum(abs(a)*_round_bound(m) for a,m in zip(row,world_scale)) for row in projection)
    world_uv = tuple(frame.uv(p) for row in surface.controls for p in row)
    point_size = tuple(_maximum(world_uv,i) for i in (0,1))
    projected_direction = frame.uv_direction(direction)
    span_v = Q(surface.knots_v[2])-Q(surface.knots_v[1])
    for index,arc in enumerate(arcs):
        a,b = max(low,Q(index,len(arcs))),min(high,Q(index+1,len(arcs)))
        if a >= b:
            continue
        _checkpoint(cancellation_check)
        if arc.slicing != 0:
            return None
        box_low,box_high = map(Q,arc.box.u_range)
        root_rects = tuple(curve._root_rect(root) for root in (arc.first,arc.second))
        if (not box_low <= arc.s0 <= box_high or not box_low <= arc.s1 <= box_high
                or any(not box_low <= rect[0][0] <= rect[0][1] <= box_high for rect in root_rects)):
            return None
        pieces = bezier_spans(surface.degree_u,surface.knots_u,profile,box_low,box_high,cancellation_check=cancellation_check)
        if len(pieces) != 1:
            return None  # no Lipschitz assumption across a tangent-discontinuous knot
        projected = tuple(tuple(x+d*(offset-sum(n*y for n,y in zip(normal,p)))/denominator
                                 for x,d in zip(p,direction)) for p in pieces[0])
        uv = tuple(frame.uv(p) for p in projected)
        du = _derivative_controls(uv,box_high-box_low)
        ddu = _derivative_controls(du,box_high-box_low)
        derivative_world = _derivative_controls(projected,box_high-box_low)
        tau0,tau1 = a*len(arcs)-index,b*len(arcs)-index
        u0,u1 = (arc.s0+t*(arc.s1-arc.s0) for t in (tau0,tau1))
        if u0 == u1:
            return None
        nominal = bezier_spans(surface.degree_u,surface.knots_u,profile,u0,u1,cancellation_check=cancellation_check)
        for controls in nominal:
            points = tuple(frame.uv(tuple(x+d*(offset-sum(n*y for n,y in zip(normal,p)))/denominator
                                         for x,d in zip(p,direction))) for p in controls)
            value,mass = bernstein_green(points,cancellation_check=cancellation_check)
            total += value; magnitude += mass
        delta_u = max(rect[0][1]-rect[0][0] for rect in root_rects)
        delta_beta = slab/abs(denominator)
        if all(root.axis == 0 for root in (arc.first,arc.second)):
            bracket = max((Q(arc.box.v_range[1])-Q(arc.box.v_range[0]))/Q(2)**(curve._INITIAL_ROOT_BISECTIONS+1),
                          *(rect[1][1]-rect[1][0] for rect in root_rects))
            delta_beta = min(delta_beta,bracket/span_v)
        lipschitz = (_maximum(uv,0)*_maximum(ddu,1)+_maximum(uv,1)*_maximum(ddu,0))/2
        transverse = (abs(projected_direction[0])*_maximum(du,1)+abs(projected_direction[1])*_maximum(du,0))/2
        local_error = lipschitz*delta_u+transverse*delta_beta
        # Public evaluate/derivative round exact world coordinates once.
        # Enclose those fields too through the EXACT plane projection; no
        # floating least-squares projection or unqualified roundoff slack.
        rate = abs((end-start)*len(arcs)*(arc.s1-arc.s0))
        if not rate:
            return None
        derivative_round = tuple(sum(abs(c)*_round_bound(_maximum(derivative_world,j)*rate)
                                     for j,c in enumerate(row))/rate for row in projection)
        local_error += (point_round[0]*_maximum(du,1)+point_round[1]*_maximum(du,0)
                        +point_size[0]*derivative_round[1]+point_size[1]*derivative_round[0]
                        +point_round[0]*derivative_round[1]+point_round[1]*derivative_round[0])/2
        error += abs(u1-u0)*local_error
    return (total if end > start else -total),error,magnitude


def polynomial_planar_loop_area(plane,loop,tolerance, *, cancellation_check=None):
    """Exact native-chart Green area, or None without a registered polynomial map."""
    try:
        requested = float(tolerance)
    except (TypeError,ValueError,OverflowError) as cause:
        raise GeometryError('polynomial Green tolerance must be finite and nonnegative') from cause
    if not math.isfinite(requested) or requested < 0:
        raise GeometryError('polynomial Green tolerance must be finite and nonnegative')
    requested = Q(requested)  # detach mutable scalar inputs BEFORE callbacks or proof work
    from .planar_exact import ExactPlane
    from .plane_nurbs import charge_box
    charge_box()  # refuse before extraction even when no callback is supplied
    supplied_loop = loop if type(loop) in (list,tuple) else None
    loop = tuple(loop)
    from .cancellation import is_cancellation_signal
    if cancellation_check is not None and not is_cancellation_signal(cancellation_check):
        from .definition_binding import definition_checksum
        from .nurbs import NurbsCurve,CurveOnSurface
        from .support_curves import PlaneNurbsCurve
        from .nurbs_runtime_binding import validate_curve_runtime,validate_section_runtime
        original_paths = tuple(loop)
        original_curves = tuple(path.curve for path in original_paths)
        nested = tuple((curve,curve.support,curve.pcurve) if type(curve) is CurveOnSurface
                       else (curve,curve.surface,curve.path) if type(curve) is PlaneNurbsCurve else (curve,)
                       for curve in original_curves)
        signature = definition_checksum((plane,original_curves))
        original_callback = cancellation_check
        def unchanged():
            if (len(loop) != len(original_paths)
                    or any(path is not original for path,original in zip(loop,original_paths))
                    or supplied_loop is not None and (len(supplied_loop) != len(original_paths)
                        or any(path is not original for path,original in zip(supplied_loop,original_paths)))
                    or any(path.curve is not curve for path,curve in zip(original_paths,original_curves))
                    or any((type(row[0]) is CurveOnSurface and
                            (row[0].support is not row[1] or row[0].pcurve is not row[2]))
                           or (type(row[0]) is PlaneNurbsCurve and
                               (row[0].surface is not row[1] or row[0].path is not row[2]))
                           for row in nested)):
                raise GeometryError('polynomial Green consumed operand objects changed during callback')
            if definition_checksum((plane,original_curves)) != signature:
                raise GeometryError('polynomial Green operands changed during callback')
            for curve in original_curves:
                if type(curve) is NurbsCurve:
                    validate_curve_runtime(curve)
                elif type(curve) is CurveOnSurface:
                    validate_curve_runtime(curve.pcurve)
                elif type(curve) is PlaneNurbsCurve:
                    validate_section_runtime(curve)
        def guarded():
            unchanged()
            cancelled = original_callback()
            unchanged()
            return cancelled
        cancellation_check = guarded
    _checkpoint(cancellation_check)
    frame = ExactPlane.from_plane(plane)
    value = magnitude = error = Q(0)
    for path in loop:
        from .support_curves import PlaneNurbsCurve
        if type(path.curve) is PlaneNurbsCurve:
            result = section_green(frame,path.curve,cancellation_check=cancellation_check)
            if result is None:
                return None
            part,bound,mass = result
            value += part; error += bound; magnitude += mass
            continue
        pieces = _world_pieces(path.curve,cancellation_check=cancellation_check)
        if pieces is None:
            return None
        for controls in pieces:
            _checkpoint(cancellation_check)
            part,mass = bernstein_green(tuple(frame.uv(point) for point in controls),cancellation_check=cancellation_check)
            value += part; magnitude += mass
    result = float(value)
    if not math.isfinite(result):
        raise GeometryError('polynomial Green area is not representable')
    if error+abs(Q(result)-value) > requested:
        return None
    _checkpoint(cancellation_check)
    return result
