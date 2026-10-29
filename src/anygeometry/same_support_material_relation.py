"""Fresh exact material-chain disjointness; no historical/persisted authority.

Strict slice: exact segment cycles and complete authentic closed P1 cycles.
Uncancelled curved chains and mixed open tolerance joints are unsupported.
"""
from dataclasses import dataclass
from fractions import Fraction as Q

from .errors import GeometryError
from .definition_binding import definition_checksum
from .identity import EntityHandle
from .nurbs import NurbsSurface, CurveOnSurface
from .support_curves import PlaneNurbsCurve
from .surfaces import Plane
from .plane_nurbs import CURRENT_WORK, charge_box, plane_nurbs_section
from .exact_predicates import point_in_polygon, Outcome
from .material_arrangement import _segments_meet


class _Unsupported(Exception):
    pass


def _audit_curve(curve):
    """Dispatch only to the registered validator of the ACTUAL kernel."""
    from .nurbs import NurbsCurve
    from .curves import LineRange
    from .nurbs_runtime_binding import validate_curve_runtime, validate_section_runtime
    if type(curve) is PlaneNurbsCurve:
        if type(curve._INITIAL_ROOT_BISECTIONS) is not int or curve._INITIAL_ROOT_BISECTIONS != 60:
            raise GeometryError('fresh material relation native root law is not certified')
        validate_section_runtime(curve)
    elif type(curve) is CurveOnSurface:
        validate_curve_runtime(curve.pcurve)
        from .native_support_snapshots import _material_support_read_pin
        _material_support_read_pin(curve.support)
    elif type(curve) is NurbsCurve:
        validate_curve_runtime(curve)
    elif type(curve) is LineRange:
        curve.validate_definition()
    # Straight has no separate evaluated kernel: actual endpoints are bound
    # by the complete current model definition, never a curve vertex proxy.


@dataclass(frozen=True)
class _Loop:
    points: tuple = ()
    curve: object = None
    key: object = None
    direction: int = 1
    sources: tuple = ()


def _area(points):
    return sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(points, points[1:]+points[:1]))


def _edges(points):
    return tuple(zip(points, points[1:]+points[:1]))


def _simple(points, tick, required=True):
    def invalid(message):
        if required:
            raise GeometryError(message)
        raise _Unsupported()
    if len(points) < 3 or not _area(points):
        invalid('fresh material loop is degenerate')
    edges = _edges(points)
    for i, (a, b) in enumerate(edges):
        if a == b:
            invalid('fresh material loop has a zero segment')
        for j in range(i+1, len(edges)):
            tick()
            c, d = edges[j]
            if not _segments_meet(a, b, c, d):
                continue
            common = {a, b} & {c, d}
            adjacent = j == i+1 or (i == 0 and j == len(edges)-1)
            if not adjacent or len(common) != 1:
                invalid('fresh material loop has a non-simple contact')
            # Collinear adjacent segments must not fold back/overlap.
            joint = next(iter(common))
            p = b if a == joint else a
            q = d if c == joint else c
            cross = (p[0]-joint[0])*(q[1]-joint[1])-(p[1]-joint[1])*(q[0]-joint[0])
            if not cross and sum((x-z)*(y-z) for x, y, z in zip(p, q, joint)) >= 0:
                invalid('fresh material loop folds back')


def _line_section(curve, support, tick):
    """Whole affine G with exact native root endpoint correspondence."""
    from ._cubic_cell_bounds import coefficients
    from .nurbs_section_faces import _Chart
    from .nurbs_runtime_binding import validate_section_runtime
    validate_section_runtime(curve)
    tick()
    table = coefficients(support)
    if table is None:
        return None
    normal, offset, *_ = curve._plane_slab()
    scalar = {(i, j): sum(normal[k]*table[k][i][j] for k in range(3))
              for i in range(4) for j in range(4)}
    scalar[0, 0] -= offset
    if any(value for (i, j), value in scalar.items() if i+j > 1):
        return None
    a, b, c = scalar[1, 0], scalar[0, 1], scalar[0, 0]
    if not (a or b):
        return None
    arcs = curve._arcs()
    chart = _Chart(support)
    start, finish = map(Q, curve.domain)
    low, high = sorted((start, finish))
    if low == high or low < 0 or high > 1:
        return None
    for i, arc in enumerate(arcs):
        if Q(i+1, len(arcs)) <= low or Q(i, len(arcs)) >= high:
            continue
        tick()
        axis = arc.slicing
        # Affine G's nonzero transverse derivative establishes unique root.
        if not (b if axis == 0 else a):
            return None
        if any(root.axis != axis or root.value != value for root, value in
               ((arc.first, arc.s0), (arc.second, arc.s1))):
            return None  # no algebraic midpoint substitution at a free endpoint
    def endpoint(parameter):
        if parameter == 1:
            index, local = len(arcs)-1, Q(1)
        else:
            index = int(parameter*len(arcs)); local = parameter*len(arcs)-index
        arc = arcs[index]
        s = arc.s0+(arc.s1-arc.s0)*local
        if arc.slicing == 0:
            u = chart.point(s, chart.q0)[0]
            return u, -(a*u+c)/b
        v = chart.point(chart.r0, s)[1]
        return -(b*v+c)/a, v
    first, last = endpoint(start), endpoint(finish)
    # Every native piece lies on the SAME unique affine line. No endpoint
    # fitting, topology coordinates or isolator centres enter its carrier.
    for i in range(1, len(arcs)):
        if low < Q(i, len(arcs)) < high:
            left, right = arcs[i-1], arcs[i]
            if left.second != right.first:
                return None
    return first, last


def _loop(view, loop, support, tick, pin_curve):
    from .curves import Straight, LineRange
    from .planar_exact import ExactPlane
    segments, sections, sources = [], [], []
    for use in loop:
        tick()
        edge = view.edges[use.edge]
        curve = edge.curve
        _audit_curve(curve)
        pair = None
        if type(support) is NurbsSurface and type(curve) is CurveOnSurface:
            if (definition_checksum(curve.support) != definition_checksum(support)
                    or curve.transform != tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
                    or curve.pcurve.degree != 1 or len(curve.pcurve.controls) != 2):
                raise _Unsupported()
            pair = tuple(tuple(map(Q, curve.pcurve.exact_point(t))) for t in (0, 1))
        elif type(support) is NurbsSurface and type(curve) is PlaneNurbsCurve:
            if definition_checksum(curve.surface) != definition_checksum(support):
                raise _Unsupported()
            pair = _line_section(curve, support, tick)
            if pair is None:
                sections.append((curve, use.forward))
        elif type(support) is Plane and type(curve) in (Straight, LineRange):
            frame = ExactPlane.from_plane(support)
            if type(curve) is Straight:
                pair = tuple(frame.uv(view.vertex_position(v)) for v in (edge.start, edge.end))
            else:
                pair = tuple(frame.uv(curve.exact_point(t)) for t in (0, 1))
        else:
            raise _Unsupported()
        if pair is not None:
            oriented = pair if use.forward else pair[::-1]
            vertices = (edge.start, edge.end) if use.forward else (edge.end, edge.start)
            segments.append(oriented)
            sources.append((edge.id, 'segment', oriented, vertices))
        else:
            values = tuple(map(Q, curve.domain))
            vertices = (edge.start, edge.end) if use.forward else (edge.end, edge.start)
            sources.append((edge.id, 'cycle', values if use.forward else values[::-1], vertices))
    if not sections:
        if any(b != c for (_, b), (c, _) in zip(segments, segments[1:]+segments[:1])):
            raise _Unsupported()  # exact endpoint closure; no tolerance welding
        points = tuple(a for a, _ in segments)
        _simple(points, tick)
        return _Loop(points=points, sources=tuple(sources))
    if segments:
        raise _Unsupported()
    key = definition_checksum((sections[0][0].plane_record, support, sections[0][0].path))
    intervals = []
    direction = None
    for curve, forward in sections:
        if definition_checksum((curve.plane_record, curve.surface, curve.path)) != key:
            raise _Unsupported()
        values = tuple(map(Q, curve.domain))
        values = values if forward else values[::-1]
        sign = 1 if values[1] > values[0] else -1
        if values[0] == values[1] or min(values) < 0 or max(values) > 1 or direction not in (None, sign):
            raise _Unsupported()
        direction = sign
        intervals.append(values)
    if any(b != c and {b, c} != {Q(0), Q(1)} for (_, b), (c, _) in zip(intervals, intervals[1:]+intervals[:1])):
        raise _Unsupported()
    reach = Q(0)
    for a, b in sorted(tuple(sorted(pair)) for pair in intervals):
        if a != reach or b <= a:
            raise _Unsupported()
        reach = b
    if reach != 1:
        raise _Unsupported()
    original = sections[0][0]
    tick()
    fresh = plane_nurbs_section(original.plane, support)
    if fresh.coincident or original.path >= len(fresh.paths) or not fresh.paths[original.path].closed:
        raise _Unsupported()
    # Native P1 constructs a complete regular simple cycle; float endpoints
    # and persisted closed flags are not used to infer its closure.
    canonical = PlaneNurbsCurve(original.plane_record, support, original.path)
    pin_curve(canonical)
    from .nurbs_section_faces import chart_loop_area
    for level in (3, 5, 7):
        tick()
        area, error = chart_loop_area(view, loop, support, level)
        if abs(Q(area)) > Q(error):
            native_sign = (1 if area > 0 else -1)*direction
            return _Loop(curve=canonical, key=key, direction=native_sign, sources=tuple(sources))
    raise GeometryError('fresh closed material cycle orientation is undecidable')


def _contacts(first, second, support, tick):
    from .nurbs_section_faces import segment_section, _sections_apart, _Chart
    if first.points and second.points:
        for a, b in _edges(first.points):
            for c, d in _edges(second.points):
                tick()
                if _segments_meet(a, b, c, d):
                    raise GeometryError('fresh input material loops contact')
    elif first.points or second.points:
        polygon, cycle = (first, second) if first.points else (second, first)
        for a, b in _edges(polygon.points):
            tick()
            verdict = segment_section(cycle.curve, a, b, (), Q(0))
            if verdict is not True:
                raise GeometryError('fresh input segment/closed-cycle separation is not certified')
    else:
        tick()
        if first.key == second.key or not _sections_apart(first.curve, second.curve, _Chart(support), Q(0)):
            raise GeometryError('fresh input closed-cycle separation is not certified')


def _witness(loop):
    if loop.points:
        return loop.points[0], Q(0)
    from .nurbs_section_faces import _Chart
    arc = loop.curve._arcs()[0]
    box = _Chart(loop.curve.surface).rect(*loop.curve._root_rect(arc.first))
    point = tuple((a+b)/2 for a, b in box)
    radius = sum((b-a)/2 for a, b in box)  # encloses the entire connected root box
    return point, radius


def _contains(loop, other, tick):
    point, radius = _witness(other)
    tick()
    if loop.points:
        # Prove the entire witness root box clear before selecting its rational
        # centre. Segment clipping against its L-infinity enclosing square.
        for a, b in _edges(loop.points):
            tick()
            lo, hi = Q(0), Q(1)
            for x, y, p in zip(a, b, point):
                if x == y:
                    if not p-radius <= x <= p+radius: lo, hi = Q(1), Q(0); break
                else:
                    u, v = sorted(((p-radius-x)/(y-x), (p+radius-x)/(y-x)))
                    lo, hi = max(lo, u), min(hi, v)
            if lo <= hi:
                raise GeometryError('fresh nesting witness is not strictly clear')
        value = point_in_polygon(point, loop.points)
        if value is Outcome.EQUAL:
            raise GeometryError('fresh nesting witness touches boundary')
        return value is Outcome.ABOVE
    from .nurbs_section_faces import chart_boundary_contact, contains_chart_curves
    if chart_boundary_contact((loop.curve,), point, radius) is not False:
        raise GeometryError('fresh curved nesting witness is not strictly clear')
    result = contains_chart_curves((loop.curve,), point, Q(0))
    if result is None:
        raise GeometryError('fresh curved nesting is undecidable')
    return result


def _domain(view, face, support, tick, pin_curve):
    loops = tuple(_loop(view, loop, support, tick, pin_curve) for loop in (face.loop, *face.holes))
    for i, first in enumerate(loops):
        for second in loops[i+1:]:
            _contacts(first, second, support, tick)
    for i, hole in enumerate(loops[1:], 1):
        if not _contains(loops[0], hole, tick):
            raise GeometryError('fresh material hole is not inside outer loop')
        if any(_contains(other, hole, tick) for j, other in enumerate(loops[1:], 1) if i != j):
            raise GeometryError('fresh material holes overlap or nest')
    return tuple((loop, 1 if i == 0 else -1) for i, loop in enumerate(loops))


def _reduced_segments(domains, tick):
    carriers, cycles = {}, {}
    for domain in domains:
        for loop, role in domain:
            if loop.curve is not None:
                cycles[loop.key] = cycles.get(loop.key, 0)+role*loop.direction
                continue
            points = loop.points if (_area(loop.points) > 0) == (role > 0) else loop.points[::-1]
            for a, b in _edges(points):
                tick()
                dx, dy = b[0]-a[0], b[1]-a[1]
                axis = 0 if dx else 1
                direction = (Q(1), dy/dx) if axis == 0 else (dx/dy, Q(1))
                offset = a[1]-direction[1]*a[0] if axis == 0 else a[0]-direction[0]*a[1]
                carriers.setdefault((axis, direction, offset), []).append((a[axis], b[axis]))
    if any(cycles.values()):
        raise _Unsupported()  # no uncancelled implicit contributions discarded
    result = []
    for (axis, direction, offset), rows in carriers.items():
        def point(x):
            return (x, direction[1]*x+offset) if axis == 0 else (direction[0]*x+offset, x)
        for low, high, weight in _atomic_intervals(rows, tick):
            if abs(weight) > 1:
                raise _Unsupported()  # same-sense multiplicity / observed overlap
            if weight:
                pair = point(low), point(high)
                result.append(pair if weight > 0 else pair[::-1])
    return result


def _atomic_intervals(rows, tick):
    """Charged exact event sweep; no hidden quadratic coverage scan."""
    events = {}
    for a, b in rows:
        tick()
        lo, hi = sorted((a, b))
        sign = 1 if b > a else -1
        events[lo] = events.get(lo, 0)+sign
        events[hi] = events.get(hi, 0)-sign
    # Charge bounded sort work BEFORE comparing an unbounded cut catalogue.
    for _ in range(len(events)*max(1, len(events).bit_length())):
        tick()
    cuts = sorted(events)
    weight = 0
    for lo, hi in zip(cuts, cuts[1:]):
        tick()
        weight += events[lo]
        yield lo, hi, weight


def _reduced_winding(edges, tick):
    outgoing, incoming = {}, {}
    for a, b in edges:
        tick()
        if a in outgoing or b in incoming:
            raise _Unsupported()  # branch/touching components not in this slice
        outgoing[a], incoming[b] = b, a
    if set(outgoing) != set(incoming):
        raise GeometryError('fresh reduced material chain is open')
    cycles = []
    while outgoing:
        start = next(iter(outgoing)); points = [start]; current = outgoing.pop(start)
        while current != start:
            tick()
            if current not in outgoing or current in points:
                raise GeometryError('fresh reduced material chain is not a cycle')
            points.append(current); current = outgoing.pop(current)
        polygon = tuple(points)
        _simple(polygon, tick, required=False)
        cycles.append(_Loop(points=polygon))
    for i, first in enumerate(cycles):
        for second in cycles[i+1:]:
            for a, b in _edges(first.points):
                for c, d in _edges(second.points):
                    tick()
                    if _segments_meet(a, b, c, d):
                        return False
    for i, loop in enumerate(cycles):
        outside = sum((1 if _area(other.points) > 0 else -1) for j, other in enumerate(cycles)
                      if i != j and _contains(other, loop, tick))
        inside = outside+(1 if _area(loop.points) > 0 else -1)
        if outside not in (0, 1) or inside not in (0, 1):
            return False
    return True  # unbounded component has winding0


def query_same_support_material_disjoint_interiors(model, first_face, second_face, *, cancellation_check=None):
    return _query_material_pair(model, first_face, second_face, cancellation_check=cancellation_check)


def _query_material_pair(model, first_face, second_face, *, cancellation_check=None, contact_ledger=None, _tick_hook=None):
    """Fresh strict-slice proof. False means overlap/unsupported/unknown.

    No lineage, parent, UUID or persisted proof establishes material truth.
    Required undecidable predicates, stale/native corruption, mutation,
    cancellation and proof exhaustion are typed refusals.
    """
    from .model import GeometryModel
    from .prepared_material_source_transfer import _state
    from .nurbs_sweep_chart import has_sweep_chart, chart_model
    from .section_audit import Work
    if not isinstance(model, GeometryModel) or any(type(h) is not EntityHandle or h.kind != 'face'
        or h.model_id != model.model_id or type(h.id) is not int or h.id not in model.faces for h in (first_face, second_face)):
        raise GeometryError('fresh material relation requires current model-bound face handles')
    if cancellation_check is not None and not callable(cancellation_check):
        raise GeometryError('fresh material relation cancellation_check must be callable')
    token = CURRENT_WORK.set(Work()) if CURRENT_WORK.get() is None else None
    try:
        initial = _state(model)
        faces = tuple(model.faces[h.id] for h in (first_face, second_face))
        pins = tuple((face, face.surface, face.parameterization, face.loop, face.holes,
                     tuple((use, model.edges[use.edge], model.edges[use.edge].curve)
                     for loop in (face.loop, *face.holes) for use in loop)) for face in faces)
        pinned_signature = definition_checksum((first_face, second_face, pins))
        derived_pins, curve_pins = [], []
        def pin_curve(curve):
            curve_pins.append((curve, definition_checksum(curve)))
        def unchanged():
            if _state(model) != initial or definition_checksum((first_face, second_face, pins)) != pinned_signature:
                raise GeometryError('fresh material relation source changed')
            from .native_support_snapshots import _material_support_read_pin
            for handle, (face, surface, parameterization, loop, holes, rows) in zip((first_face, second_face), pins):
                if (model.faces[handle.id] is not face or face.loop is not loop or face.holes is not holes
                    or face.surface is not surface or face.parameterization is not parameterization):
                    raise GeometryError('fresh material relation consumed face changed')
                _material_support_read_pin(surface)
                if parameterization is not None:
                    _material_support_read_pin(parameterization)
                for use, edge, curve in rows:
                    if model.edges[use.edge] is not edge or edge.curve is not curve:
                        raise GeometryError('fresh material relation consumed curve changed')
                    _audit_curve(curve)
            for view, face, support, parameterization, edge_rows, signature in derived_pins:
                if (view.faces.get(face.id) is not face or set(view.edges) != {edge.id for edge in edge_rows}
                    or face.surface is not support
                    or face.parameterization is not parameterization
                    or any(view.edges[edge.id] is not edge for edge in edge_rows)
                    or definition_checksum((face, support, edge_rows)) != signature):
                    raise GeometryError('fresh material relation derived chart changed')
                _material_support_read_pin(support)
            for curve, signature in curve_pins:
                if definition_checksum(curve) != signature:
                    raise GeometryError('fresh material relation consumed native curve changed')
                _audit_curve(curve)
        callback_failure = [None]
        def check(phase='fresh material relation'):
            unchanged()
            if cancellation_check is not None:
                try:
                    cancelled = cancellation_check(phase)
                except BaseException as error:
                    callback_failure[0] = error
                    raise
                unchanged()
                if cancelled:
                    raise GeometryError('fresh material relation cancelled')
        def tick():
            charge_box()
            if _tick_hook is not None:
                _tick_hook()
            if cancellation_check is not None:
                check()
        check('fresh material relation binding')
        result = False
        try:
            if any(not has_sweep_chart(face) and face.parameterization is not None
                   and face.parameterization is not face.surface for face in faces):
                raise _Unsupported()
            class ProofView:
                def __init__(self, source, face_id):
                    self.source = source
                    face = source.faces[face_id]
                    self.faces = {face_id: face}
                    self.edges = {use.edge: source.edges[use.edge] for loop in (face.loop, *face.holes) for use in loop}
                def __getattr__(self, name):
                    return getattr(self.source, name)
            views = tuple(ProofView(chart_model(model, face.id) if has_sweep_chart(face) else model, face.id) for face in faces)
            for view, face in zip(views, faces):
                actual = view.faces[face.id]
                edge_rows = tuple(view.edges.values())
                derived_pins.append((view, actual, actual.surface, actual.parameterization, edge_rows,
                                     definition_checksum((actual, actual.surface, edge_rows))))
                for edge in edge_rows:
                    pin_curve(edge.curve)
            supports = tuple(view.faces[face.id].surface for view, face in zip(views, faces))
            if any(type(support) not in (Plane, NurbsSurface) for support in supports):
                raise _Unsupported()
            if definition_checksum(supports[0]) != definition_checksum(supports[1]):
                raise _Unsupported()
            support = supports[0]
            if type(support) is NurbsSurface:
                from .nurbs_metric import certify_projected_chart
                callback_error = [None]
                def metric_check():
                    try: check('fresh material relation injectivity')
                    except BaseException as error:
                        callback_error[0] = error; raise
                    return False
                try:
                    certify_projected_chart(support, cancellation_check=None if cancellation_check is None else metric_check)
                except GeometryError as error:
                    if (callback_error[0] is error or type(error) is not GeometryError
                        or str(error) != 'NURBS support immersion and injective projection are not certified'):
                        raise
                    raise _Unsupported()
            else:
                from .planar_exact import ExactPlane
                ExactPlane.from_plane(support)
            domains = tuple(_domain(view, view.faces[face.id], support, tick, pin_curve) for view, face in zip(views, faces))
            result = _reduced_winding(_reduced_segments(domains, tick), tick)
            if result and contact_ledger is not None:
                contact_ledger.extend(_existing_contacts(domains, tick))
        except _Unsupported as error:
            if callback_failure[0] is error:
                raise
            result = False
        check('fresh material relation complete')
        return result
    finally:
        if token is not None:
            CURRENT_WORK.reset(token)


def _existing_contacts(domains, tick):
    """Complete cross-face contact accounting, authenticated by world topology.

    Chart equality establishes mathematical world-point equality on the SAME
    injective support. Current edge/vertex IDs then establish actual existing
    connectivity; equal geometry alone never creates a connection.
    """
    first, second = domains
    result = []
    def endpoint(row, point):
        points, vertices = row[2], row[3]
        matches = [vertex for p, vertex in zip(points, vertices) if p == point]
        if len(matches) != 1:
            raise GeometryError('existing preparation has an unsplit endpoint-on-segment contact')
        return matches[0]
    for a, _ in first:
        for b, _ in second:
            if a.points and b.points:
                for ra in a.sources:
                    for rb in b.sources:
                        tick()
                        p, q = ra[2]; r, s = rb[2]
                        if not _segments_meet(p, q, r, s):
                            continue
                        d, e = tuple(y-x for x, y in zip(p, q)), tuple(y-x for x, y in zip(r, s))
                        cross = d[0]*e[1]-d[1]*e[0]
                        common = {x for x in (p, q, r, s) if
                            min(p[0], q[0]) <= x[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= x[1] <= max(p[1], q[1])
                            and min(r[0], s[0]) <= x[0] <= max(r[0], s[0]) and min(r[1], s[1]) <= x[1] <= max(r[1], s[1])
                            and (x[0]-p[0])*d[1] == (x[1]-p[1])*d[0]
                            and (x[0]-r[0])*e[1] == (x[1]-r[1])*e[0]}
                        if not cross and len(common) >= 2:
                            if ra[0] != rb[0] or set(ra[2]) != set(rb[2]):
                                raise GeometryError('shared geometric interval is not one current world edge')
                            if any(endpoint(ra, x) != endpoint(rb, x) for x in common):
                                raise GeometryError('shared edge chart endpoints have different vertex incidences')
                            result.append(('edge', ra[0], (Q(0), Q(1)), tuple(sorted(common))))
                        else:
                            if len(common) != 1:
                                raise GeometryError('existing preparation has a crossing contact requiring fragmentation')
                            point = next(iter(common))
                            va, vb = endpoint(ra, point), endpoint(rb, point)
                            if va != vb:
                                raise GeometryError('equal chart endpoints are different current world vertices')
                            result.append(('vertex', va, point))
            elif a.curve is not None and b.curve is not None:
                if a.key != b.key:
                    _contacts(a, b, a.curve.surface, tick)
                    continue
                for ra in a.sources:
                    for rb in b.sources:
                        tick()
                        lo, hi = max(min(ra[2]), min(rb[2])), min(max(ra[2]), max(rb[2]))
                        if lo < hi:
                            if ra[0] != rb[0] or sorted(ra[2]) != sorted(rb[2]):
                                raise GeometryError('shared closed-path interval is not one current world edge')
                            result.append(('edge', ra[0], (lo, hi), a.key))
                        # P1 closed seam is one authenticated native root, not
                        # approximate world-endpoint equality.
                        for ta, va in zip(ra[2], ra[3]):
                            for tb, vb in zip(rb[2], rb[3]):
                                if ta % 1 == tb % 1:
                                    if va != vb:
                                        raise GeometryError('shared native path endpoint has different world vertices')
                                    result.append(('vertex', va, (a.key, ta % 1)))
            else:
                # Every surviving curved cycle was cancelled with its exact
                # counterpart. Still classify all cross polygon/cycle contacts
                # through the existing whole-curve predicate, not by omission.
                _contacts(a, b, a.curve.surface if a.curve is not None else b.curve.surface, tick)
    return tuple(sorted(set(result), key=repr))
