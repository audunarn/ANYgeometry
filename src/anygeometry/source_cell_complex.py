"""Oriented source-cell complex and whole-reference-map qualification.

Admission currently requires pure segment loops or complete regular closed
section paths. Mixed algebraic trim junctions require their own true-event
reconstruction and are explicitly refused.
"""
from collections import defaultdict
from fractions import Fraction as Q
import math

from .errors import (GeometryError, SourceCellRefinementRequired,
                     _FanApproximationInconclusive, SourceCellApproximationInconclusive)
from .identity import EntityHandle
from .nurbs import NurbsSurface, CurveOnSurface
from .support_curves import PlaneNurbsCurve
from .regular_source_reference import SourceNode, SourceSide, source_node, source_side, parameter_transport_bound
from .curvilinear_reference_maps import certify_fan_subtriangle, _source_box
from .nurbs_plane_arrangement import _area
from .nurbs_metric import sub, mul, add, scale
from .plane_nurbs import charge_box, charge_split


def _cross(a, b):
    return sub(mul(a[0], b[1]), mul(a[1], b[0]))


def _dot(a, b):
    return add(mul(a[0], b[0]), mul(a[1], b[1]))


def _gap(a, b):
    return any(x[1] < y[0] or y[1] < x[0] for x, y in zip(a, b))


def _difference(a, b):
    return tuple(sub(x, y) for x, y in zip(a, b))


def _orientation(a, b, c):
    return _cross(_difference(b, a), _difference(c, a))


def _same_strict_sign(a, b):
    return a[0] > 0 and b[0] > 0 or a[1] < 0 and b[1] < 0


class _SideEnclosures:
    """Proof-local, immutable-operand memo of one source side's exact enclosures.

    The key is the exact (Fraction) parameter interval and the value is the tuple the side's own
    deterministic ``box``/``derivative`` returned for it, so a repeated query during one skeleton proof
    reuses the identical enclosure. It lives only inside one ``_skeleton`` call (no module, model or
    global state), is never consulted for identity, and every examined box is still charged by the caller.
    """
    def __init__(self, side):
        self.side, self._boxes, self._derivatives = side, {}, {}

    def box(self, lo, hi):
        key = (lo, hi)
        if key not in self._boxes:
            self._boxes[key] = self.side.box(lo, hi)
        return self._boxes[key]

    def derivative(self, lo, hi):
        key = (lo, hi)
        if key not in self._derivatives:
            self._derivatives[key] = self.side.derivative(lo, hi)
        return self._derivatives[key]


def _skeleton(cells, boundary_sides, sides, nodes, check):
    """Certify noncollapsed source skeleton and absence of undeclared contact."""
    def _boundary_index(boundary, take):
        """Complete balanced partition of boundary *occurrences*, with exact unions.

        Every occurrence is one leaf, even if two occurrences use the same side.
        Each internal node is the coordinatewise union of its two children. Thus
        strict separation from a node box proves separation from ALL its leaves;
        overlap/touching is never evidence of separation. No proof escapes this call.
        ``take`` charges every leaf enclosure and every composite construction.
        """
        leaves = []
        for occurrence, (_, _, curve) in enumerate(boundary):
            take()
            leaves.append((curve.box(Q(0), Q(1)), None, None, occurrence))
        # Explicit postorder construction keeps every callback frontier in
        # this pinned calculator. There is no mutable recursive callee or
        # self-recursive closure binding capable of dropping a later subtree.
        pending, completed = [(leaves, False)] if leaves else [], []
        while pending:
            rows, combine = pending.pop()
            if combine:
                right, left = completed.pop(), completed.pop()
                take()
                box = tuple((min(a[0], b[0]), max(a[1], b[1]))
                            for a, b in zip(left[0], right[0]))
                completed.append((box, left, right, None))
                continue
            if len(rows) == 1:
                completed.append(rows[0])
                continue
            # Sorting only partitions the complete occurrence list; it is not a
            # geometric predicate. Exact midpoint ordering makes the tree balanced.
            axis = max(range(2), key=lambda k:
                max(row[0][k][1] for row in rows)-min(row[0][k][0] for row in rows))
            rows.sort(key=lambda row: (sum(row[0][axis]), row[3]))
            middle = len(rows)//2
            pending.extend(((None, True), (rows[middle:], False), (rows[:middle], False)))
        return completed[0] if completed else None


    def _boundary_candidates(tree, box, take):
        """Yield exactly the unseparated leaves; every node comparison is charged."""
        pending = [] if tree is None else [tree]
        while pending:
            node = pending.pop()
            take()  # cache hits, composite comparisons and surviving leaves count
            if _gap(box, node[0]):
                continue
            if node[3] is not None:
                yield node[3]
            else:
                pending.extend((node[2], node[1]))

    internal, boundary = {}, []
    for i, cell in enumerate(cells):
        for side, (a, b) in enumerate(zip(cell.node_ids, cell.node_ids[1:]+cell.node_ids[:1])):
            if (i, side) in boundary_sides:
                boundary.append((a, b, _SideEnclosures(sides[(i, side)])))
            else:
                key = tuple(sorted((a, b)))
                internal[key] = key
    rows = []
    for a, b in internal:
        first, last = nodes[a].box, nodes[b].box
        delta = _difference(last, first)
        if not any(lo > 0 or hi < 0 for lo, hi in delta):
            raise GeometryError('source internal edge is not certified noncollapsed')
        box = tuple((min(x[0], y[0]), max(x[1], y[1])) for x, y in zip(first, last))
        rows.append((a, b, first, last, box))
    rows.sort(key=lambda row: row[4][0][0])
    examined = 0
    builder, candidates, gap = _boundary_index, _boundary_candidates, _gap
    def take():
        nonlocal examined
        check()
        if ((_boundary_index, _boundary_candidates, _gap, take) != calculators
                or tuple(f.__code__ for f in calculators) != codes):
            raise GeometryError('source skeleton boundary index calculator changed')
        charge_box()
        examined += 1
        if examined > 2000000:
            raise GeometryError('source skeleton proof pair budget exhausted')
    calculators = builder, candidates, gap, take
    codes = tuple(f.__code__ for f in calculators)
    tree = builder(boundary, take) if rows else None
    for index, (a, b, first, last, box) in enumerate(rows):
        check()
        for c, d, p, q, other in rows[index+1:]:
            if other[0][0] > box[0][1]:
                break
            if _gap(box, other):
                continue
            charge_box()
            examined += 1
            if examined > 2000000:
                raise GeometryError('source skeleton proof pair budget exhausted')
            shared = {a, b}&{c, d}
            if shared:
                joint = next(iter(shared))
                va = _difference(nodes[b if a == joint else a].box, nodes[joint].box)
                vb = _difference(nodes[d if c == joint else c].box, nodes[joint].box)
                cross = _cross(va, vb)
                if cross[0] > 0 or cross[1] < 0 or _dot(va, vb)[1] < 0:
                    continue
            elif (_same_strict_sign(_orientation(first, last, p), _orientation(first, last, q)) or
                  _same_strict_sign(_orientation(p, q, first), _orientation(p, q, last))):
                continue
            raise GeometryError('source internal edges have an undeclared or unresolved contact')
        for occurrence in candidates(tree, box, take):
            c, d, curve = boundary[occurrence]
            pending = [(Q(0), Q(1), 0)]
            while pending:
                take()
                lo, hi, depth = pending.pop()
                curved_box = curve.box(lo, hi)
                side = _orientation(first, last, curved_box)
                if _gap(box, curved_box) or side[0] > 0 or side[1] < 0:
                    continue
                joint = a if a in (c, d) else b if b in (c, d) else None
                if joint is not None and ((joint == c and lo == 0) or (joint == d and hi == 1)):
                    line = _difference(nodes[b if a == joint else a].box, nodes[joint].box)
                    derivative = curve.derivative(lo, hi)
                    if joint == d:
                        derivative = tuple(scale(x, Q(-1)) for x in derivative)
                    crossing = _cross(line, derivative)
                    if crossing[0] > 0 or crossing[1] < 0 or _dot(line, derivative)[1] < 0:
                        continue
                if depth >= 20:
                    raise GeometryError('source skeleton boundary contact is not certified')
                charge_split()
                middle = (lo+hi)/2
                pending.extend(((lo, middle, depth+1), (middle, hi, depth+1)))


def _row(values, size):
    if type(values) is not tuple or len(values) != size or any(
            type(x) not in (float, int) or not math.isfinite(x) for x in values):
        raise GeometryError('source-cell coordinates require finite literal tuples')
    return tuple(Q(x) for x in values)


def _support_owner(model, face):
    from .nurbs_sweep_chart import has_sweep_chart, chart_model
    return chart_model(model, face) if has_sweep_chart(model.faces[face]) else model


def _rational_loop_area(owner, loop, reference):
    from .plane_nurbs import depth_limit
    from .regular_source_reference import RationalSourceSide
    sides = []
    for use in loop:
        edge = owner.edges[use.edge]
        params = (Q(0), Q(1)) if use.forward else (Q(1), Q(0))
        verts = (edge.start, edge.end) if use.forward else (edge.end, edge.start)
        nodes = tuple(reference.node(use.edge, t, v) for t, v in zip(params, verts))
        side = reference.side(use.edge, params, *nodes)
        if type(side) not in (SourceSide, RationalSourceSide):
            raise GeometryError('source-cell polynomial boundary family is not qualified')
        sides.append(side)
    for level in range(depth_limit(8)+1):
        if level:
            for side in sides:
                if type(side) is RationalSourceSide:
                    for _ in range(2**(level-1)):
                        charge_split()
        total = (Q(0), Q(0))
        size = 2**level
        for side in sides:
            if type(side) is SourceSide and side.curve is None:
                a, b = side.start.box, side.end.box
                total = add(total, scale(sub(mul(a[0], b[1]), mul(a[1], b[0])), Q(1, 2)))
                continue
            for k in range(size):
                lo, hi = Q(k, size), Q(k+1, size)
                (u, v), (du, dv) = side._ranges(lo, hi)
                total = add(total, scale(sub(mul(u, dv), mul(v, du)), (hi-lo)/2))
        if total[0] > 0 or total[1] < 0:
            return total
    raise GeometryError('source-cell polynomial boundary orientation is not certified')


def _boundary_proof(owner, face, support):
    """Reuse exact owner embedding only for unchanged complete source images."""
    directions = {}
    from .nurbs_intersection_faces import has_intersection_edges, validate_intersection_face,loop_area
    if has_intersection_edges(owner,face):
        problems = validate_intersection_face(owner,face.id)
        if problems:
            raise GeometryError('; '.join(problems))
        for index,loop in enumerate((face.loop,*face.holes)):
            area = loop_area(owner,loop,support)
            sign,desired = (1 if area[0] > 0 else -1),(1 if index == 0 else -1)
            for use in loop:
                directions[use.edge] = use.forward if sign == desired else not use.forward
        return directions
    from .surfaces import Plane
    if isinstance(support, Plane):
        from .plane_section_plate import loop_area
        from .planar_exact import ExactPlane
        problems = owner._validate_face_geometry(face.id)
        if problems:
            raise GeometryError('; '.join(problems))
        frame = ExactPlane.from_plane(support)
        for index, loop in enumerate((face.loop, *face.holes)):
            area = loop_area(owner, loop, frame, lambda: None)
            sign, desired = (1 if area[0] > 0 else -1), (1 if index == 0 else -1)
            for use in loop:
                directions[use.edge] = use.forward if sign == desired else not use.forward
        return directions
    for loop_index, loop in enumerate((face.loop, *face.holes)):
        curves = [owner.edges[use.edge].curve for use in loop]
        if all(isinstance(curve, CurveOnSurface) for curve in curves):
            if any(curve.support != support for curve in curves):
                raise GeometryError('source-cell polynomial boundary family is not qualified')
            if any(curve.pcurve.degree != 1 or len(curve.pcurve.controls) != 2 for curve in curves):
                problems = owner._validate_face_geometry(face.id)
                if problems:
                    raise GeometryError('; '.join(problems))
                from .regular_source_reference import NurbsReference
                area = _rational_loop_area(owner, loop, NurbsReference(owner, face))[0]
                sign = 1 if area > 0 else -1
                desired = 1 if loop_index == 0 else -1
                for use in loop:
                    directions[use.edge] = use.forward if sign == desired else not use.forward
                continue
        elif all(isinstance(curve, PlaneNurbsCurve) for curve in curves):
            first = curves[0]
            if any((curve.surface, curve.plane_record, curve.path) !=
                   (support, first.plane_record, first.path) for curve in curves):
                raise GeometryError('source-cell section loop must be one complete regular path')
            from .plane_nurbs import plane_nurbs_section
            if not plane_nurbs_section(first.plane, support).paths[first.path].closed:
                raise GeometryError('source-cell section boundary is not closed')
            intervals = sorted(tuple(sorted(map(Q, curve.domain))) for curve in curves)
            reach = Q(0)
            for a, b in intervals:
                if a != reach or b <= a:
                    raise GeometryError('source-cell section loop omits or overlaps source arcs')
                reach = b
            if reach != 1:
                raise GeometryError('source-cell section loop does not cover its complete source path')
        else:
            if not all((isinstance(c, PlaneNurbsCurve) and c.surface == support)
                       or (isinstance(c, CurveOnSurface) and c.support == support
                           and c.pcurve.degree == 1 and len(c.pcurve.controls) == 2) for c in curves):
                raise GeometryError('mixed source-boundary family is not qualified')
            problems = owner._validate_face_geometry(face.id)
            if problems:
                raise GeometryError('; '.join(problems))
            from .regular_source_reference import NurbsReference
            NurbsReference(owner, face)  # exact axis incidence/true-joint proof
        area, _bound = _area(owner, loop, support)
        sign = 1 if area > 0 else -1
        desired = 1 if loop_index == 0 else -1
        for use in loop:
            directions[use.edge] = use.forward if sign == desired else not use.forward
    return directions


def _manifold(cells, boundary_sides, hole_count):
    """An oriented connected disk/holed-disk complex with no vertex pinches."""
    edge_uses, links = defaultdict(list), defaultdict(list)
    for index, cell in enumerate(cells):
        ids = cell.node_ids
        for side, (a, b) in enumerate(zip(ids, ids[1:]+ids[:1])):
            edge_uses[tuple(sorted((a, b)))].append((index, side, a, b))
        for i, node in enumerate(ids):
            links[node].append((ids[i-1], ids[(i+1)%len(ids)]))
    adjacency = defaultdict(set)
    boundary_degree = defaultdict(int)
    for uses in edge_uses.values():
        marked = [(i, side) in boundary_sides for i, side, a, b in uses]
        if len(uses) == 1 and marked == [True]:
            _i, _s, a, b = uses[0]
            boundary_degree[a] += 1
            boundary_degree[b] += 1
        elif len(uses) == 2 and not any(marked) and uses[0][2:] == tuple(reversed(uses[1][2:])):
            adjacency[uses[0][0]].add(uses[1][0])
            adjacency[uses[1][0]].add(uses[0][0])
        else:
            raise GeometryError('source-cell sides lack exact opposite interior incidence or unique boundary use')
    visited, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current not in visited:
            visited.add(current)
            pending.extend(adjacency[current]-visited)
    if len(visited) != len(cells):
        raise GeometryError('source-cell face complex is disconnected')
    for vertex, pairs in links.items():
        graph = defaultdict(list)
        for a, b in pairs:
            graph[a].append(b)
            graph[b].append(a)
        degrees = sorted(map(len, graph.values()))
        boundary = boundary_degree[vertex]
        if boundary not in (0, 2) or (boundary == 0 and any(x != 2 for x in degrees)) or (
                boundary == 2 and (degrees.count(1) != 2 or any(x not in (1, 2) for x in degrees))):
            raise GeometryError('source-cell vertex link is not a disk or half-disk')
        seen, stack = set(), [next(iter(graph))]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node])
        if len(seen) != len(graph):
            raise GeometryError('source-cell vertex link has a pinch')
    if len(links)-len(edge_uses)+len(cells) != 1-hole_count:
        raise GeometryError('source-cell Euler characteristic does not match the owner holes')


from ._quad_bound_reuse import quad_bound_operation


@quad_bound_operation()
def _boundary_failure_kind(cell, cell_index, fans, boundary_sides, error_limit, check, *, cell_error_limit=None):
    """Diagnose only the exact side law; boundary incidence alone proves nothing.

    This is a refusal classification, never an admission or replacement cell
    proof. A boundary witness must exceed the GLOBAL requested ceiling, which
    cannot rise after local footprint refinement. Proving only that ceiling is
    insufficient to diagnose an interior-only local failure; the returned upper
    must also meet the current cell limit. Unsupported side parameterizations
    preserve the original retries.
    """
    from .regular_source_reference import RationalSourceSide
    from .prepared_material_curvilinear import (_parameterized_world_chord, _WorldChordNeedsRefinement,
        _ParameterizedChordInconclusive)
    found = False
    complete = True
    cell_error_limit = error_limit if cell_error_limit is None else cell_error_limit
    callback_errors = []
    def checked():
        try:
            check()
        except BaseException as error:
            callback_errors.append(error)
            raise
    for side, current, transport in fans:
        if (cell_index,side) not in boundary_sides:
            continue
        found = True
        if type(current) is not RationalSourceSide:
            complete = False
            continue
        current._validate()
        try:
            upper, _ = _parameterized_world_chord(current.curve,*current.parameters,
                (cell.xyz[side],cell.xyz[(side+1)%len(cell.node_ids)]),
                error_limit-transport,checked)
            if Q(upper) > cell_error_limit-transport:
                complete = False
        except _WorldChordNeedsRefinement as error:
            if any(error is cause for cause in callback_errors):
                raise
            current._validate()
            return 'boundary_limited'
        except _ParameterizedChordInconclusive as error:
            if any(error is cause for cause in callback_errors):
                raise
            complete = False
        current._validate()
    return 'interior_only' if not found or complete else 'unresolved'


def certify_source_cells(model, regions, chords, cells, surface_error, check, *, face_error_limits=(), local_limit_query=None, local_allocation_records=None):
    from .prepared_material_curvilinear import CurvilinearMaterialCell
    sources = {chart.face.id: chart for region in regions.regions for chart in region.sources}
    if any(len(region.sources) != 1 for region in regions.regions):
        raise GeometryError('curvilinear source cells currently require singleton material regions')
    grouped, xyz_by_node, uv_by_node = defaultdict(list), {}, {}
    for cell in cells:
        check()
        if (type(cell) is not CurvilinearMaterialCell or type(cell.face) is not EntityHandle
                or cell.face.model_id != model.model_id or cell.face.kind != 'face' or cell.face.id not in sources
                or type(cell.node_ids) is not tuple or len(cell.node_ids) not in (3, 4)
                or any(type(n) is not int or n < 0 for n in cell.node_ids) or len(set(cell.node_ids)) != len(cell.node_ids)
                or type(cell.uv) is not tuple or type(cell.xyz) is not tuple
                or len(cell.uv) != len(cell.node_ids) or len(cell.xyz) != len(cell.node_ids)
                or type(cell.boundary_intervals) is not tuple):
            raise GeometryError('source-cell literal connectivity or ownership is invalid')
        for node, uv, xyz in zip(cell.node_ids, cell.uv, cell.xyz):
            _row(uv, 2)
            _row(xyz, 3)
            if node in xyz_by_node and xyz_by_node[node] != xyz:
                raise GeometryError('one source-cell global node has conflicting literal XYZ')
            key = (cell.face.id, node)
            if key in uv_by_node and uv_by_node[key] != uv:
                raise GeometryError('one face node has conflicting literal source charts')
            xyz_by_node[node], uv_by_node[key] = xyz, uv
        grouped[cell.face.id].append(cell)
    if set(grouped) != set(sources):
        raise GeometryError('source-cell proof does not cover every selected material face')
    node_records, reference_uv, bounds, boundary_records = [], [], [], []
    global_station_nodes = {}
    requested_surface_error=surface_error
    local_limits=dict(face_error_limits)
    for face_id, face_cells in sorted(grouped.items()):
        surface_error=min(requested_surface_error,local_limits[face_id]) if local_limits else requested_surface_error
        check()
        chart = sources[face_id]
        from .surfaces import Plane
        if not isinstance(chart.support, (NurbsSurface, Plane)):
            raise GeometryError('source-cell support family is not qualified')
        owner = _support_owner(model, face_id)
        face = owner.faces[face_id]
        directions = _boundary_proof(owner, face, chart.support)
        from .plane_source_reference import PlaneReference
        from .regular_source_reference import NurbsReference
        plane_reference = PlaneReference(owner, face) if isinstance(chart.support, Plane) else NurbsReference(owner, face)
        source_edges = set(directions)
        boundary_sides, intervals, nodes, sides, source_station_nodes = {}, defaultdict(list), {}, {}, {}
        orientation = None
        for cell_index, cell in enumerate(face_cells):
            points = [_row(row, 2) for row in cell.uv]
            area = sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(points, points[1:]+points[:1]))
            sign = 1 if area > 0 else -1 if area < 0 else 0
            if not sign or (orientation is not None and sign != orientation):
                raise GeometryError('source-cell face orientations are zero or inconsistent')
            orientation = sign
            for reference in cell.boundary_intervals:
                if (type(reference) is not tuple or len(reference) != 3
                        or type(reference[0]) is not int or not 0 <= reference[0] < len(cell.node_ids)
                        or type(reference[1]) is not int or not 0 <= reference[1] < len(chords.intervals)
                        or type(reference[2]) is not bool):
                    raise GeometryError('source-cell boundary interval reference is invalid')
                side, index, forward = reference
                if (cell_index, side) in boundary_sides:
                    raise GeometryError('source-cell boundary side has duplicate references')
                chord = chords.intervals[index]
                edge_id = chord.edge.id
                if edge_id not in source_edges:
                    raise GeometryError('source-cell boundary does not belong to this source face')
                occurrences = [row for row in chord.occurrences if row.face == cell.face]
                if len(occurrences) != 1:
                    raise GeometryError('source-cell boundary has no unique owner chart occurrence')
                occurrence = occurrences[0]
                parameters = chord.parameter_range if forward else chord.parameter_range[::-1]
                expected_xyz = chord.xyz if forward else chord.xyz[::-1]
                expected_uv = occurrence.uv if forward else occurrence.uv[::-1]
                a, b = parameters
                expected_forward = directions[edge_id] if orientation > 0 else not directions[edge_id]
                if (a < b) != expected_forward:
                    raise GeometryError('source-cell boundary orientation disagrees with the owner material cycle')
                edge = owner.edges[edge_id]
                curve = edge.curve
                endpoint_nodes = []
                for offset, parameter in enumerate(parameters):
                    vertex = edge.start if parameter == 0 else edge.end if parameter == 1 else None
                    node = cell.node_ids[(side+offset)%len(cell.node_ids)]
                    if cell.xyz[(side+offset)%len(cell.node_ids)] != expected_xyz[offset] or cell.uv[(side+offset)%len(cell.node_ids)] != expected_uv[offset]:
                        raise GeometryError('source-cell boundary station differs from the owner literal target')
                    station = ('vertex', vertex) if vertex is not None else ('edge_station', edge_id, parameter)
                    if station in global_station_nodes and global_station_nodes[station] != node:
                        raise GeometryError('shared owner boundary station has different global node IDs')
                    global_station_nodes[station] = node
                    true = (plane_reference.node(edge_id, parameter, vertex) if plane_reference is not None
                            else source_node(curve, parameter, vertex=vertex))
                    if node in nodes and nodes[node].identity != true.identity:
                        raise GeometryError('source-cell boundary node has incompatible true algebraic identities')
                    nodes[node] = true
                    endpoint_nodes.append(true)
                sides[(cell_index, side)] = (plane_reference.side(edge_id, parameters, *endpoint_nodes)
                    if plane_reference is not None else source_side(curve, parameters, *endpoint_nodes))
                boundary_sides[(cell_index, side)] = index
                intervals[edge_id].append(tuple(sorted(parameters)))
                boundary_records.append((cell.face, edge_id, tuple(parameters), tuple(n.identity for n in endpoint_nodes)))
        for edge_id in source_edges:
            reach = Q(0)
            for a, b in sorted(intervals[edge_id]):
                if a != reach or b <= a:
                    raise GeometryError('source-cell boundary intervals have a gap, duplicate or overlap')
                reach = b
            if reach != 1:
                raise GeometryError('source-cell boundary intervals omit owner material')
        _manifold(face_cells, boundary_sides, len(face.holes))
        identity_nodes = {}
        for node, true in nodes.items():
            if true.identity in identity_nodes and identity_nodes[true.identity] != node:
                raise GeometryError('one true source vertex has duplicate mesh node identities')
            identity_nodes[true.identity] = node
        for cell in face_cells:
            for node, uv in zip(cell.node_ids, cell.uv):
                if node not in nodes:
                    point = _row(uv, 2)
                    nodes[node] = SourceNode(('interior_node', face_id, node), tuple((x, x) for x in point))
        _skeleton(face_cells, boundary_sides, sides, nodes, check)
        # Precision remains a hard geometry refusal, including on cells whose
        # interior approximation will subsequently request finer resolution.
        for node, true in sorted(nodes.items()):
            check()
            literal_xyz = xyz_by_node[node]
            world = _source_box(chart.support, true.box)
            residual = sum(max(abs(a-Q(x)), abs(b-Q(x))) for (a, b), x in zip(world, literal_xyz))
            if residual > Q(chart.world_tolerance) or residual > surface_error:
                raise GeometryError('true source node target exceeds the existing geometry precision')
            node_records.append((chart.face, node, true.identity, true.box))
        # The unchanged exact owner boundaries are simple, correctly nested
        # source images. The complete oriented disk/holed-disk complex above,
        # shared true nodes/internal sides, and strict Jacobians below give
        # degree1 inside those boundaries and0 outside: no overlap or gap.
        for cell_index, cell in enumerate(face_cells):
            check()
            ids = cell.node_ids
            for node, uv in zip(ids, cell.uv):
                if node not in nodes:
                    point = _row(uv, 2)
                    nodes[node] = SourceNode(('interior_node', face_id, node), tuple((x, x) for x in point))
            centre = tuple(Q(float(sum(Q(row[i]) for row in cell.uv)/len(ids))) for i in (0, 1))
            if not chart.domain.contains_loop(chart.boundaries[0], centre, chart.world_tolerance, boundary=False) or any(
                    chart.domain.contains_loop(loop, centre, chart.world_tolerance, boundary=True) for loop in chart.boundaries[1:]):
                raise GeometryError('source-cell reference centre is not certified in source material')
            centre_xyz = tuple(sum(Q(row[i]) for row in cell.xyz)/len(ids) for i in range(3))
            maximum = Q(0)
            fans = []
            for side in range(len(ids)):
                current = sides.get((cell_index, side), SourceSide(nodes[ids[side]], nodes[ids[(side+1)%len(ids)]]))
                transport = parameter_transport_bound(current)
                if transport > Q(chart.world_tolerance):
                    raise GeometryError('true source endpoint transport exceeds the existing precision contract')
                fans.append((side, current, transport))
            if local_limit_query is not None:
                # The authentic corrected source maps, not the caller's vertex
                # hull, supply the complete footprint enclosure.
                footprints = []
                for side, current, transport in fans:
                    if len(ids) == 4:
                        from .curvilinear_reference_maps import _source_quad_fan
                        footprint = _source_quad_fan(tuple(nodes[node].box for node in ids),
                            current, side, (Q(0), Q(1)), (Q(0), Q(1)))[0]
                    else:
                        edge_box = current.box(Q(0), Q(1))
                        footprint = tuple((min(a, c), max(b, c))
                            for (a, b), c in zip(edge_box, centre))
                    footprints.append(footprint)
                limit, mapped = local_limit_query(face_id, tuple(footprints))
                surface_error = min(requested_surface_error, limit)
                if local_allocation_records is not None:
                    key = (face_id, tuple(ids))
                    if key in local_allocation_records:
                        raise GeometryError('duplicate local cell allocation identity')
                    local_allocation_records[key] = (surface_error, mapped)
            if any(transport >= surface_error for _, _, transport in fans):
                raise GeometryError('true source endpoint transport exceeds the existing precision contract')
            for side, current, transport in fans:
                from ._source_cell_point_error import fan_point_error_lower_squared
                lower = fan_point_error_lower_squared(chart.support, current, centre,
                    (cell.xyz[side], cell.xyz[(side+1)%len(ids)]), centre_xyz, check=check,
                    quad_xyz=cell.xyz if len(ids) == 4 else None, quad_side=side,
                    quad_uv=tuple(nodes[node].box for node in ids) if len(ids) == 4 else None)
                allocated = surface_error-transport
                if lower is not None and lower > allocated*allocated:
                    failure_kind = _boundary_failure_kind(cell,cell_index,fans,boundary_sides,requested_surface_error,check,cell_error_limit=surface_error)
                    # Initial public validation plus guarded arithmetic and a
                    # fresh final complete source/chord binding are mandatory.
                    check()
                    from .prepared_material_curvilinear import validate_prepared_material_boundary_chords_binding
                    validate_prepared_material_boundary_chords_binding(model, regions, chords,
                                                                       expected_revision=model.revision)
                    check()
                    failure = SourceCellRefinementRequired(face_id, cell.node_ids, allocated, lower,
                                                       requested_error=requested_surface_error)
                    failure.boundary_failure_kind = failure_kind
                    raise failure
            for side, current, transport in fans:
                callback_error = None
                from .curvilinear_reference_maps import _closed_fan_window
                def fan_check():
                    nonlocal callback_error
                    try:
                        check()
                    except BaseException as cause:
                        callback_error = cause
                        raise
                    return False
                try:
                    # An issued check verifies the finite window/helper slice
                    # after any opaque local-limit query, before entering it.
                    from .curvilinear_reference_maps import _CLOSED_FAN_STEPS, _ClosedFunction
                    if type(check) is _ClosedFunction and check in _CLOSED_FAN_STEPS:
                        check()
                    with _closed_fan_window(check) as closed:
                        error, _pieces = certify_fan_subtriangle(chart.support, current, centre,
                            (cell.xyz[side], cell.xyz[(side+1)%len(ids)]), centre_xyz,
                            orientation=orientation, surface_error=surface_error-transport,
                            quad_xyz=cell.xyz if len(ids) == 4 else None, quad_side=side,
                            quad_uv=tuple(nodes[node].box for node in ids) if len(ids) == 4 else None,
                            quad_nodes=tuple(nodes[node] for node in ids) if len(ids) == 4 else None,
                            cancellation_check=closed if closed is not None else fan_check)
                except _FanApproximationInconclusive as uncertain:
                    if uncertain is callback_error or type(uncertain) is not _FanApproximationInconclusive:
                        raise
                    if uncertain.error_limit != surface_error-transport:
                        raise GeometryError('inconclusive fan error allocation changed') from uncertain
                    failure_kind = _boundary_failure_kind(cell,cell_index,fans,boundary_sides,requested_surface_error,check,cell_error_limit=surface_error)
                    check()
                    from .prepared_material_curvilinear import validate_prepared_material_boundary_chords_binding
                    from .definition_binding import definition_checksum
                    validate_prepared_material_boundary_chords_binding(model, regions, chords,
                                                                       expected_revision=model.revision)
                    check()
                    failure = SourceCellApproximationInconclusive(face_id, cell.node_ids,
                        uncertain.error_limit, uncertain.upper_error, requested_error=requested_surface_error,
                        jacobian_lower=uncertain.jacobian_lower, proof_box=uncertain.proof_box,
                        proof_depth=uncertain.proof_depth, source_model_id=model.model_id,
                        source_revision=model.revision, source_checksum=chords.source_checksum,
                        chord_checksum=definition_checksum(chords))
                    failure.boundary_failure_kind = failure_kind
                    raise failure from uncertain
                maximum = max(maximum, error+transport)
            reference_uv.append((cell.face, cell.node_ids, tuple(map(float, centre))))
            bounds.append((cell.face, cell.node_ids, maximum))
    return tuple(node_records), tuple(reference_uv), tuple(bounds), tuple(boundary_records)
