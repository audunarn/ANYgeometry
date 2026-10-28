"""Certified closed-trace planar cells for the bounded section owner.

Crossing a plate boundary requires a further station graph; this first
production slice refuses that case rather than extending the physical cut.
"""
from fractions import Fraction as Q
from itertools import product, combinations
from dataclasses import replace

from .errors import GeometryError
from .entities import OrientedEdge, EntityRef
from .curves import Straight, LineRange
from .support_curves import PlaneNurbsCurve
from .planar_exact import ExactPlane, point_in_region, region_is_simple
from .exact_predicates import Outcome
from .plane_nurbs import charge_box, charge_split, depth_limit
from .nurbs_metric import add, sub, mul, scale, derivative_intervals


class PlateBoundaryCrossingRequired(GeometryError):
    """The bounded trace needs a plate-boundary event graph, not extension."""


def _corners(box):
    return product(*box)


def _box(frame, bounds):
    values = [frame.uv(p) for p in product(*zip(*bounds))]
    return tuple((min(p[i] for p in values), max(p[i] for p in values)) for i in (0, 1))


def _curve_box(model, edge, frame, lo=0., hi=1.):
    item = model.edges[edge]
    charge_box()
    if type(item.curve) in (Straight, LineRange):
        if type(item.curve) is LineRange:
            item.curve.certify_plane(frame, min(1.e-8, model.tolerance.effective_curve_fit_residual(1.)))
            a, b = item.curve.exact_point(0), item.curve.exact_point(1)
        else:
            a, b = [tuple(Q(float(x)) for x in model.vertex_position(v)) for v in (item.start, item.end)]
        points = [frame.uv(tuple(x+Q(t)*(y-x) for x, y in zip(a, b))) for t in (lo, hi)]
        return tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in (0, 1))
    if not isinstance(item.curve, PlaneNurbsCurve):
        raise GeometryError('bounded plate boundary curve has no true source chart')
    from .regular_source_reference import source_node, source_side
    from .curvilinear_reference_maps import _source_box
    first = source_node(item.curve, Q(0), vertex=item.start)
    last = source_node(item.curve, Q(1), vertex=item.end)
    side = source_side(item.curve, (Q(0), Q(1)), first, last)
    return _project_interval(frame, _source_box(item.curve.surface, side.box(Q(lo), Q(hi))))


def plate_loops(model, face):
    frame = ExactPlane.from_plane(face.surface)
    loops = []
    for loop in (face.loop, *face.holes):
        if any(type(model.edges[e.edge].curve) not in (Straight, LineRange) for e in loop):
            raise GeometryError('bounded section currently requires straight authored plate boundaries')
        points = []
        for use in loop:
            edge = model.edges[use.edge]
            if type(edge.curve) is LineRange:
                edge.curve.certify_plane(frame, min(1.e-8, model.tolerance.effective_curve_fit_residual(1.)))
                point = edge.curve.exact_point(0 if use.forward else 1)
            else:
                point = model.vertex_position(model.oriented_start_vertex(use))
            points.append(frame.uv(point))
        loops.append(tuple(points))
    if not region_is_simple(loops):
        raise GeometryError('bounded section plate material is not a simple polygon with holes')
    return frame, tuple(loops)


def _line_box_clear(a, b, box):
    if any(max(a[i], b[i]) < box[i][0] or min(a[i], b[i]) > box[i][1] for i in (0, 1)):
        return True
    values = [(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) for p in _corners(box)]
    return min(values) > 0 or max(values) < 0


def trace_in_plate(model, edge, frame, loops, check):
    statuses, pending = set(), [(Q(0), Q(1), 0)]
    segments = [(a, b) for loop in loops for a, b in zip(loop, loop[1:]+loop[:1])]
    from .nurbs import _float_down, _float_up
    while pending:
        check()
        a, b, depth = pending.pop()
        box = _curve_box(model, edge, frame, _float_down(a), _float_up(b))
        if all(_line_box_clear(p, q, box) for p, q in segments):
            status = point_in_region(tuple((lo+hi)/2 for lo, hi in box), loops)
            if status is not Outcome.EQUAL:
                statuses.add(status is Outcome.ABOVE)
                if len(statuses) > 1:
                    raise PlateBoundaryCrossingRequired('trace crosses bounded plate material; boundary-event clipping required')
                continue
        if depth >= depth_limit(32):
            raise PlateBoundaryCrossingRequired('trace/plate boundary contact is not certified')
        charge_split()
        middle = (a+b)/2
        pending.extend(((a, middle, depth+1), (middle, b, depth+1)))
    return statuses == {True}


def certify_trace_embedding(source, parents):
    """Prove simplicity using source immersion or exact translation structure."""
    if len(parents) == 1:
        from .nurbs import NurbsSurface
        from .nurbs_sweep_chart import has_sweep_chart, chart_model, _tensor
        from ._translational_pairs import _sweep
        support = source.faces[parents[0]].surface
        if has_sweep_chart(source.faces[parents[0]]):
            support = chart_model(source, parents[0]).support
        elif not isinstance(support, NurbsSurface):
            recognized = _sweep(source, parents[0])
            if recognized is None:
                raise GeometryError('bounded trace has no injective source support proof')
            support = _tensor(recognized[0], recognized[1])
        from .nurbs_metric import certify_projected_chart
        certify_projected_chart(support)
        return
    from ._translational_pairs import separated_sweep_connectors
    for first, second in combinations(parents, 2):
        if separated_sweep_connectors(source, first, second) is None:
            raise GeometryError('bounded multi-support trace embedding is not certified by exact structural correspondence')


def _translation_cycle_embedding(model, loops, frame):
    """Re-derive the exact affine image of a common transverse profile plane."""
    from .nurbs import NurbsCurve
    from .regular_source_reference import source_node
    from .nurbs_audit import _certify_pieces, _monotone, _BUDGET
    from ._translational_pairs import _cross, _dot, _sub
    profiles, vertices, common = [], [], None
    for loop in loops:
        for use in loop:
            edge = model.edges[use.edge]
            curve, s = edge.curve, edge.curve.surface
            if (s.degree_v != 1 or s.knots_v != (0., 0., 1., 1.) or s.v_range != (0., 1.)
                    or any(len(row) != 2 for row in s.controls) or set(curve.domain) != {0., 1.}):
                return False
            d = tuple(Q(b)-Q(a) for a, b in zip(s.controls[0][0], s.controls[0][1]))
            if not any(d) or (common is not None and d != common):
                return False
            common = d
            if any(tuple(Q(b)-Q(a) for a, b in zip(row[0], row[1])) != d for row in s.controls):
                return False
            if s.weights is not None and any(row[0] != row[1] for row in s.weights):
                return False
            profile = NurbsCurve(s.degree_u, s.knots_u, tuple(row[0] for row in s.controls),
                None if s.weights is None else tuple(row[0] for row in s.weights), s.u_range)
            parameters = (Q(0), Q(1)) if use.forward else (Q(1), Q(0))
            nodes = [source_node(curve, t) for t in parameters]
            u = [node.box[0] for node in nodes]
            if u == [(Q(1), Q(1)), (Q(0), Q(0))]:
                profile = replace(profile, domain=profile.domain[::-1])
            elif u != [(Q(0), Q(0)), (Q(1), Q(1))]:
                return False
            profiles.append(profile)
            vertices.append((model.oriented_start_vertex(use), model.oriented_end_vertex(use)))
    if not _dot(frame.normal, common):
        return False
    points = [tuple(map(Q, p)) for profile in profiles for p in profile.controls]
    anchor = points[0]
    normal = next((n for a in points[1:] for b in points[1:]
                   if any(n := _cross(_sub(a, anchor), _sub(b, anchor)))), None)
    if normal is None or not _dot(normal, common) or any(_dot(normal, _sub(p, anchor)) for p in points):
        return False
    def hull(curve, lo, hi):
        charge_box()
        return curve.exact_hull(lo, hi)
    def simple(curve, lo=0., hi=1., depth=0):
        if curve.exact_point(lo) == curve.exact_point(hi):
            return False
        if _monotone(hull(curve, lo, hi)):
            return True
        if depth >= depth_limit(24):
            return False
        charge_split()
        mid = (lo+hi)/2
        return (simple(curve, lo, mid, depth+1) and simple(curve, mid, hi, depth+1)
                and _certify_pieces(curve, (lo, mid), curve, (mid, hi), [(mid, mid)], _BUDGET,
                    _hull=lambda c, a, b: c.exact_hull(a, b), _qualification=True, _depth=depth+1) is True)
    for profile in profiles:
        if not simple(profile):
            raise GeometryError('transverse source profile simplicity is not certified')
    for i, j in combinations(range(len(profiles)), 2):
        contacts = []
        for a, b in product((0, 1), repeat=2):
            equal = profiles[i].exact_point(a) == profiles[j].exact_point(b)
            if equal != (vertices[i][a] == vertices[j][b]):
                raise GeometryError('transverse exact profile joints disagree with declared trace vertices')
            if equal:
                contacts.append((a, b))
        if _certify_pieces(profiles[i], (0., 1.), profiles[j], (0., 1.), contacts, _BUDGET,
                _hull=lambda c, a, b: c.exact_hull(a, b), _qualification=True) is not True:
            raise GeometryError('transverse profile pair separation is not certified')
    # T(P)=P+d*(offset-n.P)/(n.d) is injective on the certified
    # common profile plane (its kernel is span(d), transverse to that plane).
    return True


def validate_complete_path_face(model, face_id):
    """Fresh Plane trim proof for full closed paths on one injective support.

    Returns None outside this domain; the existing general validator remains
    responsible there. No receipt, metadata or sampled polygon grants proof.
    """
    face = model.faces[face_id]
    loops = (face.loop, *face.holes)
    curved = [loop for loop in loops if any(isinstance(model.edges[e.edge].curve, PlaneNurbsCurve) for e in loop)]
    if not curved or any(any(not isinstance(model.edges[e.edge].curve, PlaneNurbsCurve) for e in loop) for loop in curved):
        return None
    support = model.edges[curved[0][0].edge].curve.surface
    plane_record = model.edges[curved[0][0].edge].curve.plane_record
    if any(model.edges[e.edge].curve.plane_record != plane_record for loop in curved for e in loop):
        return None
    frame = ExactPlane.from_plane(face.surface)
    translated = any(model.edges[e.edge].curve.surface != support for loop in curved for e in loop)
    if translated and not _translation_cycle_embedding(model, curved, frame):
        return None
    seen = set()
    from .plane_nurbs import plane_nurbs_section
    from .nurbs_metric import certify_projected_chart
    for loop in (() if translated else curved):
        curves = [model.edges[e.edge].curve for e in loop]
        first = curves[0]
        if any(c.surface != support or c.plane_record != first.plane_record or c.path != first.path for c in curves):
            return None
        section = plane_nurbs_section(first.plane, support)
        if not section.paths[first.path].closed:
            return None
        if first.path in seen:
            return [f'face {face_id} repeats a complete section boundary']
        seen.add(first.path)
        spans = sorted(tuple(sorted(map(Q, c.domain))) for c in curves)
        if spans[0][0] != 0 or spans[-1][1] != 1 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
            return None
        directed = [tuple(map(Q, c.domain)) if e.forward else tuple(reversed(tuple(map(Q, c.domain)))) for e, c in zip(loop, curves)]
        if any(a[1] != b[0] and {a[1], b[0]} != {Q(0), Q(1)} for a, b in zip(directed, directed[1:]+directed[:1])):
            return [f'face {face_id} closed section stations are not ordered continuously']
    if not translated:
        certify_projected_chart(support)
    straight = [loop for loop in loops if loop not in curved]
    if any(any(type(model.edges[e.edge].curve) is not Straight for e in loop) for loop in straight):
        return None
    # Authored straight boundaries retain their exact simplicity gate.
    if straight:
        polygons = [tuple(frame.uv(model.vertex_position(model.oriented_start_vertex(e))) for e in loop) for loop in straight]
        from .planar_exact import _segments_cross
        if any(not region_is_simple((polygon,)) for polygon in polygons):
            return [f'face {face_id} straight material boundaries are not simple']
        for first, second in combinations(polygons, 2):
            if any(_segments_cross(a, b, c, d) for a, b in zip(first, first[1:]+first[:1])
                   for c, d in zip(second, second[1:]+second[:1])):
                return [f'face {face_id} straight material boundaries intersect']
    segments = [(frame.uv(model.vertex_position(model.oriented_start_vertex(e))),
                 frame.uv(model.vertex_position(model.oriented_end_vertex(e)))) for loop in straight for e in loop]
    for loop in curved:
        for use in loop:
            pending = [(Q(0), Q(1), 0)]
            while pending:
                lo, hi, depth = pending.pop()
                box = _curve_box(model, use.edge, frame, lo, hi)
                if all(_line_box_clear(a, b, box) for a, b in segments):
                    continue
                if depth >= depth_limit(32):
                    raise GeometryError('complete section boundary is not separated from authored plate edges')
                charge_split()
                mid = (lo+hi)/2
                pending.extend(((lo, mid, depth+1), (mid, hi, depth+1)))
    areas = [loop_area(model, loop, frame, lambda: None) for loop in loops]
    sign = 1 if areas[0][0] > 0 else -1
    if any((area[0] > 0) == (sign > 0) for area in areas[1:]):
        return [f'face {face_id} hole orientation disagrees with material boundary']
    for i, loop in enumerate(loops[1:], 1):
        witness = _curve_box(model, loop[0].edge, frame, .5, .5)
        if not _witness_inside(model, loops[0], witness, frame, lambda: None):
            return [f'face {face_id} hole is outside the outer boundary']
        if any(_witness_inside(model, other, witness, frame, lambda: None) for j, other in enumerate(loops[1:], 1) if i != j):
            return [f'face {face_id} holes overlap or nest']
    return []


def trace_cycles(model, traces):
    uses = {}
    for edge in traces:
        item = model.edges[edge]
        uses.setdefault(item.start, []).append(edge)
        uses.setdefault(item.end, []).append(edge)
    if any(len(edges) != 2 for edges in uses.values()):
        raise PlateBoundaryCrossingRequired('bounded trace has interior-ended seams or unresolved branch stations')
    remaining, loops = set(traces), []
    while remaining:
        first = min(remaining)
        current, start = model.edges[first].start, model.edges[first].start
        edge, loop = first, []
        while edge in remaining:
            remaining.remove(edge)
            item = model.edges[edge]
            forward = item.start == current
            loop.append(OrientedEdge(edge, forward))
            current = item.end if forward else item.start
            if current == start:
                break
            options = [e for e in uses[current] if e in remaining]
            if len(options) != 1:
                raise GeometryError('bounded section cycle station identity is inconsistent')
            edge = options[0]
        if current != start or len(loop) < 2:
            raise GeometryError('bounded section cycle is not closed with distinct edges')
        loops.append(tuple(loop))
    return tuple(loops)


def _reverse(loop):
    return tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(loop))


def _polygon_area(model, loop, frame):
    points = [_line_endpoints(model, e, frame)[0] for e in loop]
    value = sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(points, points[1:]+points[:1]))/2
    return value, value


def _line_endpoints(model, use, frame):
    edge = model.edges[use.edge]
    if type(edge.curve) is LineRange:
        edge.curve.certify_plane(frame, min(1.e-8, model.tolerance.effective_curve_fit_residual(1.)))
        points = tuple(frame.uv(edge.curve.exact_point(t)) for t in (0, 1))
        for t, vertex, uv in zip((0, 1), (edge.start, edge.end), points):
            raw = edge.curve.exact_point(t)
            projected = tuple(frame.world(*uv))
            proxy = tuple(Q(float(x)) for x in model.vertex_position(vertex))
            if sum(max(r,s,q)-min(r,s,q) for r,s,q in zip(raw, projected, proxy)) > Q(model.tolerance.effective_length(1.)):
                raise GeometryError('retained boundary native/projected/proxy transport exceeds joint tolerance')
        return points if use.forward else points[::-1]
    return (frame.uv(model.vertex_position(model.oriented_start_vertex(use))),
            frame.uv(model.vertex_position(model.oriented_end_vertex(use))))


def _line_world_endpoints(model, use, frame):
    """Exact projected retained segment used by chart contact/clearance proofs."""
    return tuple(tuple(frame.world(*uv)) for uv in _line_endpoints(model, use, frame))


def _project_interval(frame, values, *, direction=False):
    projected = [(frame.uv_direction(p) if direction else frame.uv(p)) for p in _corners(values)]
    return tuple((min(p[i] for p in projected), max(p[i] for p in projected)) for i in (0, 1))


def loop_area(model, loop, frame, check):
    if all(type(model.edges[e.edge].curve) in (Straight, LineRange) for e in loop):
        return _polygon_area(model, loop, frame)
    from .regular_source_reference import source_node, source_side
    from .curvilinear_reference_maps import _source_box
    for level in range(depth_limit(8)+1):
        if level:
            for _ in range(len(loop)*2**(level-1)):
                charge_split()
        total = (Q(0), Q(0))
        for use in loop:
            edge = model.edges[use.edge]
            curve = edge.curve
            if type(curve) in (Straight, LineRange):
                # a straight piece of a mixed loop contributes its exact shoelace term
                a, b = _line_endpoints(model, use, frame)
                term = (a[0]*b[1] - a[1]*b[0])/2
                total = add(total, (term, term))
                continue
            if not isinstance(curve, PlaneNurbsCurve):
                raise GeometryError('bounded plate cycle has an unsupported curved boundary')
            params = (Q(0), Q(1)) if use.forward else (Q(1), Q(0))
            nodes = tuple(source_node(curve, t, vertex=v) for t, v in zip(params,
                (edge.start, edge.end) if use.forward else (edge.end, edge.start)))
            side = source_side(curve, params, *nodes)
            size = 2**level
            for k in range(size):
                check()
                charge_box()
                lo, hi = Q(k, size), Q(k+1, size)
                chart, dchart = side.box(lo, hi), side.derivative(lo, hi)
                uv = _project_interval(frame, _source_box(curve.surface, chart))
                du, dv = derivative_intervals(curve.surface, chart)
                world_d = tuple(add(mul(a, dchart[0]), mul(b, dchart[1])) for a, b in zip(du, dv))
                delta = _project_interval(frame, world_d, direction=True)
                total = add(total, scale(sub(mul(uv[0], delta[1]), mul(uv[1], delta[0])), (hi-lo)/2))
        if total[0] > 0 or total[1] < 0:
            return total
    raise GeometryError('bounded plate curved cycle orientation is not certified')


class _UndecidedRay(Exception):
    pass


def _contains(model, loop, point, frame, check):
    """Exact branch-root parity in the plane, never a sampled polygon."""
    if all(type(model.edges[e.edge].curve) in (Straight, LineRange) for e in loop):
        return point_in_region(point, (tuple(_line_endpoints(model, e, frame)[0] for e in loop),)) is Outcome.ABOVE
    from .branch_predicates import _engine
    for slope in (Q(0), Q(1, 3), Q(-2, 5), Q(7, 11)):
        count = [0]
        direction = (Q(1), slope)
        normal = tuple((slope*u-v)/frame._scale for u, v in zip(frame._dual_u, frame._dual_v))
        offset = sum(n*x for n, x in zip(normal, frame.origin))+slope*point[0]-point[1]
        try:
            for use in loop:
                curve = model.edges[use.edge].curve
                if type(curve) in (Straight, LineRange):
                    # exact ray/segment parity term of a straight piece in a mixed loop
                    item = model.edges[use.edge]
                    a, b = _line_endpoints(model, OrientedEdge(item.id, True), frame)
                    e = (b[0]-a[0], b[1]-a[1])
                    w = (a[0]-point[0], a[1]-point[1])
                    den = direction[0]*e[1] - direction[1]*e[0]
                    if den == 0:
                        if w[0]*direction[1] - w[1]*direction[0] == 0:
                            raise _UndecidedRay()           # collinear with the ray
                        continue
                    s = (w[0]*e[1] - w[1]*e[0])/den
                    r = (w[0]*direction[1] - w[1]*direction[0])/den
                    if r == 0 or r == 1 or s == 0:
                        raise _UndecidedRay()               # vertex or start contact: try another slope
                    if 0 < r < 1 and s > 0:
                        count[0] += 1
                    continue
                def classify(context):
                    check()
                    low, high = context.g2_range()
                    if low > 0 or high < 0:
                        return 'done'
                    if context.sign_from == 0 or context.sign_to == 0:
                        raise _UndecidedRay()
                    if context.view.monotone(context.s_range, context.x_range):
                        if context.sign_from == context.sign_to:
                            return 'done'
                        box = _box(frame, context.world())
                        values = [(p[0]-point[0])+slope*(p[1]-point[1]) for p in _corners(box)]
                        if min(values) > 0:
                            count[0] += 1
                            return 'done'
                        if max(values) < 0:
                            return 'done'
                    return 'split'
                _engine(curve, normal, offset, classify, 32)
            return bool(count[0]%2)
        except _UndecidedRay:
            continue
    raise GeometryError('bounded plate cycle nesting is not certified')


def _witness_inside(model, loop, witness, frame, check):
    # The witness is an ENCLOSURE of a true source boundary point. Prove
    # the entire box clear of the candidate boundary before using parity
    # at its rational centre; approximate station coordinates never decide
    # topology or nesting by themselves.
    for use in loop:
        pending = [(Q(0), Q(1), 0)]
        while pending:
            check()
            lo, hi, depth = pending.pop()
            box = _curve_box(model, use.edge, frame, lo, hi)
            if any(a[1] < b[0] or b[1] < a[0] for a, b in zip(box, witness)):
                continue
            if depth >= depth_limit(32):
                raise GeometryError('bounded plate nesting witness is not separated from another true boundary')
            charge_split()
            mid = (lo+hi)/2
            pending.extend(((lo, mid, depth+1), (mid, hi, depth+1)))
    return _contains(model, loop, tuple((a+b)/2 for a, b in witness), frame, check)


def fragment_plate(model, parent, traces, frame, check):
    face = model.faces[parent]
    source_loops = (face.loop, *face.holes)
    source_area = [_polygon_area(model, loop, frame)[0] for loop in source_loops]
    positive = [source_loops[0] if source_area[0] > 0 else _reverse(source_loops[0])]
    negative = [loop if area < 0 else _reverse(loop) for loop, area in zip(source_loops[1:], source_area[1:])]
    for loop in trace_cycles(model, traces):
        area = loop_area(model, loop, frame, check)
        loop = loop if area[0] > 0 else _reverse(loop)
        positive.append(loop)
        negative.append(_reverse(loop))
    holes = [[] for _ in positive]
    for loop in negative:
        witness = _curve_box(model, loop[0].edge, frame, .5, .5)
        candidates = [i for i, outer in enumerate(positive) if {e.edge for e in outer} != {e.edge for e in loop}
                      and _witness_inside(model, outer, witness, frame, check)]
        if len(candidates) > 1:
            innermost = []
            for i in candidates:
                box = _curve_box(model, positive[i][0].edge, frame, .5, .5)
                if all(i == j or _witness_inside(model, positive[j], box, frame, check) for j in candidates):
                    innermost.append(i)
            candidates = innermost
        if len(candidates) != 1:
            raise GeometryError('bounded plate hole owner is not uniquely certified')
        holes[candidates[0]].append(loop)
    children = []
    for outer, inner in zip(positive, holes):
        if source_area[0] < 0:
            outer, inner = _reverse(outer), [_reverse(loop) for loop in inner]
        child = model.add_face_from_loop(outer, surface=face.surface)
        model._put_entity('face', replace(model.faces[child], holes=tuple(inner), metadata=dict(face.metadata),
            parameterization=face.parameterization))
        model.tag(EntityRef('face', child), *model.tags_for(face.ref))
        children.append(child)
    model._delete_entity('face', parent)
    model.record_replacement(face.ref, tuple(EntityRef('face', child) for child in children))
    return tuple(children)
