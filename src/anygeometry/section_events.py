"""Isolated crossings of an implicit section with a declared chart segment.

Records retain exact parameter and chart enclosures. Rounded station values are
not used here to identify or merge roots.
"""
from dataclasses import dataclass
from fractions import Fraction as Q

from .branch_predicates import _engine
from .errors import GeometryError
from .nurbs import CurveOnSurface, _IDENTITY
from .nurbs_section_faces import _Chart, _line_patches, _context_rect, _corners
from .plane_nurbs import PlaneNurbsNotCertified, CURRENT_WORK
from .support_curves import PlaneNurbsCurve


@dataclass(frozen=True)
class SectionSegmentEvent:
    section_interval: tuple
    segment_interval: tuple
    chart_enclosure: tuple


def rational_segment_parameter(pcurve, axis, coordinate):
    """Exact inverse on a declared rational degree-one chart segment."""
    x0, x1 = (Q(p[axis]) for p in pcurve.controls)
    w0, w1 = (Q(1), Q(1)) if pcurve.weights is None else map(Q, pcurve.weights)
    numerator = w0*(coordinate-x0)
    denominator = w1*(x1-coordinate)+numerator
    if not denominator:
        raise GeometryError('chart segment inverse is singular')
    lo, hi = Q(pcurve.knots[1]), Q(pcurve.knots[2])
    knot = lo+(hi-lo)*numerator/denominator
    a, b = map(Q, pcurve.domain)
    return (knot-a)/(b-a)


def contract_crossing(context, *, accept=None):
    """Exact interval Newton enclosure of all roots in a regular graph piece.

    A nonzero derivative interval is sufficient for the enclosure theorem;
    endpoint signs need not prove existence or a strict interior crossing.
    The caller must not infer existence from the returned root enclosure.

    Newton iterations are bounded separately from subdivision depth by the
    active qualification policy; every iteration charges the work counter.
    Exact Newton quotients are enclosed on a fixed dyadic grid of the initial
    interval before reuse. This bounds rational coefficient growth without
    discarding any root. If that grid cannot decide a caller's predicate, the
    caller must subdivide or refuse; the grid grants no geometric tolerance.
    """
    from .branch_predicates import _range, _multiply, _subtract
    from .plane_nurbs import charge_box
    view = context.view
    s_range, x_range = context.s_range, context.x_range
    origin = s_range[0]
    quantum = (s_range[1]-s_range[0])/2**64
    if not quantum:
        return view.uv(s_range, x_range)
    work = CURRENT_WORK.get()
    iterations = 16 if work is None else getattr(work, 'max_newton_iterations', 16)
    for _ in range(iterations):
        charge_box()
        d1 = view.derivative_ranges(view.g1, s_range, x_range)
        d2 = view.derivative_ranges(view.g2, s_range, x_range)
        gx = d1[1-view.slicing]
        if gx[0] <= 0 <= gx[1]:
            break
        inverse = tuple(sorted((Q(1)/gx[0], Q(1)/gx[1])))
        derivative = _multiply(_subtract(_multiply(d2[view.slicing], gx),
                                        _multiply(d2[1-view.slicing], d1[view.slicing])), inverse)
        if derivative[0] <= 0 <= derivative[1]:
            break
        middle = sum(s_range)/2
        bracket = view.bracket(middle, max((s_range[1]-s_range[0])/2**16, Q(1, 2**100)))
        if bracket is None:
            break
        value = _range(view.over(view.g2, (middle, middle), bracket))
        quotient = _multiply(value, tuple(sorted((Q(1)/derivative[0], Q(1)/derivative[1]))))
        narrowed = max(s_range[0], middle-quotient[1]), min(s_range[1], middle-quotient[0])
        if narrowed[0] > narrowed[1]:
            raise PlaneNurbsNotCertified('section root interval Newton became empty')
        # Repeated exact rational Newton division otherwise grows denominators
        # through every Bernstein evaluation. Outward rounding is exact and
        # keeps the root enclosure inside the previous enclosing interval.
        low_grid, high_grid = ((value-origin)/quantum for value in narrowed)
        low_index = low_grid.numerator//low_grid.denominator
        high_index = -((-high_grid.numerator)//high_grid.denominator)
        narrowed = (max(s_range[0], origin+low_index*quantum),
                    min(s_range[1], origin+high_index*quantum))
        if narrowed == s_range:
            break
        s_range, x_range = view.rectangle(*narrowed, context.depth, x_range)
        if accept is not None and accept(view.uv(s_range, x_range)):
            break
    return view.uv(s_range, x_range)


def section_segment_events(section, segment, *, cancellation_check=None):
    """All transverse interior crossings; contact/tangency/coincidence refuses.

    The entire graph is accounted by exact exclusion or one monotone root in a
    disjoint isolating rectangle. No display polyline supplies topology.
    """
    if isinstance(segment, PlaneNurbsCurve):
        return section_section_events(section, segment, cancellation_check=cancellation_check)
    if (not isinstance(section, PlaneNurbsCurve) or not isinstance(segment, CurveOnSurface)
            or segment.support != section.surface or segment.transform != _IDENTITY
            or segment.pcurve.degree != 1 or len(segment.pcurve.controls) != 2):
        raise GeometryError('section crossings require an own-support chart segment')
    pcurve = segment.pcurve
    p0, p1 = pcurve.exact_point(0), pcurve.exact_point(1)
    direction = tuple(b-a for a, b in zip(p0, p1))
    if not any(direction):
        raise GeometryError('section trim segment is degenerate')
    axis = 0 if abs(direction[0]) >= abs(direction[1]) else 1
    chart = _Chart(section.surface)
    a, b, c = chart.line(p0, p1)
    events = []
    work = CURRENT_WORK.get()
    refine = 48 if work is None else max(1, min(48, work.max_depth-1))
    endpoint_events = {}
    for t in (Q(0), Q(1)):
        rect = chart.rect(*section.chart_enclosure(float(t)))
        corners = _corners(rect)
        # Exact line identity of the entire end-root enclosure; an uncertain
        # nearby endpoint is not an event identity and remains a refusal.
        if all((p[0]-p0[0])*direction[1]-(p[1]-p0[1])*direction[0] == 0 for p in corners):
            positions = sorted((p[axis]-p0[axis])/direction[axis] for p in corners)
            if 0 < positions[0] <= positions[-1] < 1:
                interval = sorted(rational_segment_parameter(pcurve, axis, p[axis]) for p in corners)
                event = SectionSegmentEvent((t, t), (interval[0], interval[-1]), rect)
                events.append(event)
                endpoint_events[t] = event

    def classify(context):
        if cancellation_check is not None and cancellation_check():
            raise GeometryError('section crossing cancelled')
        low, high = context.g2_range()
        if low > 0 or high < 0:
            return 'done'
        rect = _context_rect(chart, context)
        positions = sorted((p[axis]-p0[axis])/direction[axis] for p in _corners(rect))
        if positions[-1] < 0 or positions[0] > 1:
            return 'done'
        if context.view.monotone(context.s_range, context.x_range):
            first, last = context.sign_from, context.sign_to
            t0, t1 = context.t_exact()
            from_t, to_t = (t1, t0) if section.domain[1] < section.domain[0] else (t0, t1)
            if ((first == 0 and last and from_t in endpoint_events)
                    or (last == 0 and first and to_t in endpoint_events)):
                return 'done'  # Unique endpoint root already retained above.
            if first and last:
                if first == last:
                    return 'done'
                if 0 < positions[0] <= positions[-1] < 1 and t1-t0 <= Q(1, 2**refine):
                    segment_range = sorted(rational_segment_parameter(pcurve, axis, p[axis])
                                           for p in _corners(rect))
                    if not 0 < segment_range[0] <= segment_range[-1] < 1:
                        return 'split'
                    events.append(SectionSegmentEvent((t0, t1),
                        (segment_range[0], segment_range[-1]), rect))
                    return 'done'
        return 'split'

    _engine(section, None, None, classify, 64, patches=_line_patches(a, b, c))
    events.sort(key=lambda event: event.section_interval)
    for first, second in zip(events, events[1:]):
        if first.section_interval[1] >= second.section_interval[0]:
            raise PlaneNurbsNotCertified('section crossing identities are not separated')
    return tuple(events)


def section_section_events(first, second, *, cancellation_check=None):
    """Transverse crossings of two owner paths on the same declared support."""
    if not isinstance(first, PlaneNurbsCurve) or first.surface != second.surface:
        raise GeometryError('implicit trim crossings need the same declared NURBS support')
    plane = second.plane
    a, b = [tuple(Q(float(x)) for x in v) for v in (plane.u_vector, plane.v_vector)]
    n = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    d = sum(x*Q(float(y)) for x, y in zip(n, plane.origin))
    events = []
    work = CURRENT_WORK.get()
    refine = 48 if work is None else max(1, min(48, work.max_depth-1))
    count = len(second._arcs())
    domain0, domain1 = map(Q, second.domain)
    chart = _Chart(first.surface)
    active_boxes = [(s, x) if arc.slicing == 0 else (x, s)
                    for arc, s, x in second._arc_rects(0., 1.)]

    def other_parameter(rect):
        if all(any(a[1] < b[0] or b[1] < a[0] for a, b in zip(rect, active))
               for active in active_boxes):
            return False
        candidates = []
        for index, arc in enumerate(second._arcs()):
            u, v = rect
            if not (arc.box.u_range[0] <= u[0] <= u[1] <= arc.box.u_range[1]
                    and arc.box.v_range[0] <= v[0] <= v[1] <= arc.box.v_range[1]):
                continue
            slicing = rect[arc.slicing]
            parameter = sorted(((Q(index)+(s-arc.s0)/(arc.s1-arc.s0))/count-domain0)
                               /(domain1-domain0) for s in slicing)
            if parameter[1] < 0 or parameter[0] > 1:
                continue
            if 0 < parameter[0] <= parameter[1] < 1:
                candidates.append(tuple(parameter))
        if len(candidates) == 1:
            return candidates[0]
        return None

    def classify(context):
        if cancellation_check is not None and cancellation_check():
            raise GeometryError('implicit trim crossing cancelled')
        lo, hi = context.g2_range()
        if lo > 0 or hi < 0:
            return 'done'
        if context.view.monotone(context.s_range, context.x_range):
            before, after = context.sign_from, context.sign_to
            if before and after:
                if before == after:
                    return 'done'
                t0, t1 = context.t_exact()
                rect = context.view.uv(context.s_range, context.x_range)
                parameter = other_parameter(rect)
                if parameter is False:
                    return 'done'
                if parameter is not None and t1-t0 <= Q(1, 2**refine):
                    events.append(SectionSegmentEvent((t0, t1), parameter, chart.rect(*rect)))
                    return 'done'
        return 'split'

    _engine(first, n, d, classify, 64)
    events.sort(key=lambda event: event.section_interval)
    if any(a.section_interval[1] >= b.section_interval[0] for a, b in zip(events, events[1:])):
        raise GeometryError('implicit crossing event identities are not separated')
    return tuple(events)
