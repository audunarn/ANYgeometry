"""Plate-domain boundary splitting, straight tip completion and directed cell extraction (bounded sections).

Everything runs on the planning clone inside the caller's transaction. Vertex identities come from
certified station matches made by ``bounded_cut_graph`` (never from distance); the only geometric
checks here are exact rational ones, the owner's existing ``certify_line_section`` for traces and the
edge-partition tolerance check of ``_record_edge_partition_split``.
"""
from collections import Counter
from dataclasses import replace
from fractions import Fraction as Q

from .curves import Straight, LineRange  # noqa: E402
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .plane_section_plate import (PlateBoundaryCrossingRequired, _curve_box, _polygon_area, _reverse,
                                  _witness_inside, loop_area, plate_loops, _line_endpoints)


def split_edge_at_vertex(model, edge_id, vertex_id, t, edge_draft, source_draft):
    """Split a STRAIGHT edge at an EXISTING vertex whose identity was certified by the caller.

    The same primitives and bookkeeping as ``GeometryModel.split_edge`` (face loop replacement with FaceUse
    synchronization, replacement history, source edge-partition and subcurve-preimage records), but the
    split point is the supplied vertex instead of a new one. ``t`` is the parameter of that vertex's
    projection; ``_record_edge_partition_split`` refuses it unless the vertex lies within the joint tolerance
    of the straight edge's interpolation at ``t``. Attachments and member uses are refused."""
    from .prepared_material_source_transfer import _before_edge_partition_split, _record_edge_partition_split
    edge = model._require_edge(edge_id)
    if type(edge.curve) not in (Straight, LineRange):
        raise GeometryError('split at a certified vertex requires a straight plate edge')
    if model._target_attachments.get(('edge', edge_id)) or model._edge_member_uses.get(edge_id):
        raise GeometryError('plate edge carries attachments or member uses')
    if vertex_id in (edge.start, edge.end) or vertex_id not in model.vertices or not 0 < float(t) < 1:
        raise GeometryError('certified plate station is not an interior split vertex')
    before = _before_edge_partition_split(source_draft, model, edge_id)
    parent_definition = None
    if edge_draft is not None:
        from .edge_subcurve_preimages import _edge_ancestry_definition
        parent_definition = _edge_ancestry_definition(model, edge_id)
    # route C: when the vertex is EXACTLY collinear with the edge its exact rational parameter is derived from the
    # stored coordinates (both children are then exact subsets of the original line and their spans union
    # exactly to the parent); otherwise the legacy float parameter and its tolerance check apply unchanged
    a = tuple(Q(float(x)) for x in model.vertex_position(edge.start))
    b = tuple(Q(float(x)) for x in model.vertex_position(edge.end))
    p = tuple(Q(float(x)) for x in model.vertex_position(vertex_id))
    direction = tuple(y - x for x, y in zip(a, b))
    offset = tuple(y - x for x, y in zip(a, p))
    cross = (offset[1]*direction[2] - offset[2]*direction[1], offset[2]*direction[0] - offset[0]*direction[2],
             offset[0]*direction[1] - offset[1]*direction[0])
    exact_t = (sum(x*y for x, y in zip(offset, direction))/sum(x*x for x in direction)
               if not any(cross) and any(direction) else None)
    if type(edge.curve) is Straight and exact_t is not None and not 0 < exact_t < 1:
        raise GeometryError('certified plate station is outside its plate edge')
    if not 0 < Q(t) < 1:
        raise GeometryError('certified carrier station is outside its plate edge')
    if type(edge.curve) is Straight and exact_t is not None and exact_t == Q(t):
        first, second = model.add_line(edge.start, vertex_id), model.add_line(vertex_id, edge.end)
    else:
        carrier = edge.curve if type(edge.curve) is LineRange else LineRange(a, b)
        exact_t = Q(t)
        if sum(abs(x-y) for x, y in zip(p, carrier.exact_point(exact_t))) > Q(model.tolerance.effective_length(1.)):
            raise GeometryError('retained carrier station exceeds joint correspondence tolerance')
        first = model.add_curve(edge.start, vertex_id, carrier.subcurve(0, exact_t))
        second = model.add_curve(vertex_id, edge.end, carrier.subcurve(exact_t, 1))
    for face_id in model.faces_using_edge(edge_id):
        model._put_entity('face', model._replace_edge_in_loop(model.faces[face_id], edge_id, first, second))
    model._delete_entity('edge', edge_id)
    model.record_replacement(EntityRef('edge', edge_id), (EntityRef('edge', first), EntityRef('edge', second)))
    _record_edge_partition_split(source_draft, model, edge_id, t, (first, second), before, exact=exact_t)
    if edge_draft is not None:
        from .edge_subcurve_preimages import _record_edge_subcurve_split
        _record_edge_subcurve_split(edge_draft, edge_id, t, (first, second),
                                    model.tolerance.effective_length(1.), model=model,
                                    parent_definition=parent_definition)
    return first, second


def split_edge_new_vertex(model, edge_id, t, edge_draft, source_draft):
    """``GeometryModel.split_edge`` of a straight plate edge with the same ancestry records."""
    edge = model.edges[edge_id]
    carrier = edge.curve if type(edge.curve) is LineRange else LineRange(
        model.vertex_position(edge.start), model.vertex_position(edge.end))
    vertex = model.add_point(*carrier.evaluate(Q(t)))
    children = split_edge_at_vertex(model, edge_id, vertex, Q(t), edge_draft, source_draft)
    return vertex, children


def split_generated_edge(model, edge_id, source_draft):
    """One extra owned station on a staging-created (generated) trace: ``model.split_edge`` at the middle of its
    own regular curve (EXACT_CURVES subcurves, FaceUse synchronization, replacement history); the source
    edge-partition draft only keeps the children generated. Returns the two children."""
    from .prepared_material_source_transfer import _before_edge_partition_split, _record_edge_partition_split
    before = _before_edge_partition_split(source_draft, model, edge_id)
    _vertex, (first, second) = model.split_edge(edge_id, 0.5)
    _record_edge_partition_split(source_draft, model, edge_id, 0.5, (first, second), before)
    return first, second


def _boundary_edges(model, plate_id):
    face = model.faces[plate_id]
    return [use.edge for loop in (face.loop, *face.holes) for use in loop]


def complete_tips(model, plate_id, traces, check, edge_draft, source_draft):
    """Straight auxiliary partitions from every trace tip that ends in the plate's interior to the plate
    boundary, so that no dangling trace is left in a plate cell. Returns the new auxiliary edge ids.

    A candidate is the perpendicular foot on a boundary edge (nearest first); it is accepted only when its
    segment crosses no other boundary segment, lies in the plate material, meets every trace only at the tip
    (``certify_line_section`` True) and crosses no earlier auxiliary segment. Nothing certifies -> refusal."""
    from .branch_predicates import certify_line_section
    from .exact_predicates import Outcome
    from .planar_exact import _segments_cross, point_in_region
    face = model.faces[plate_id]
    boundary = {model.oriented_start_vertex(u) for loop in (face.loop, *face.holes) for u in loop}
    degree = Counter()
    for edge in traces:
        item = model.edges[edge]
        degree[item.start] += 1
        degree[item.end] += 1
    if any(d > 2 and v not in boundary for v, d in degree.items()):
        raise PlateBoundaryCrossingRequired('bounded trace has an unresolved branch station')
    tips = sorted(v for v, d in degree.items() if d == 1 and v not in boundary)
    created, accepted, accepted_actual = [], [], []
    for tip in tips:
        check()
        frame, loops = plate_loops(model, model.faces[plate_id])
        p = frame.uv(model.vertex_position(tip))
        segments = []
        for edge_id in _boundary_edges(model, plate_id):
            item = model.edges[edge_id]
            segments.append((edge_id, *_line_endpoints(model, OrientedEdge(edge_id, True), frame)))
        candidates = []
        for edge_id, a, b in segments:
            e = (b[0]-a[0], b[1]-a[1])
            length = e[0]*e[0] + e[1]*e[1]
            lam = ((p[0]-a[0])*e[0] + (p[1]-a[1])*e[1])/length
            if 0 < lam < 1:
                foot = (a[0] + lam*e[0], a[1] + lam*e[1])
                candidates.append(((p[0]-foot[0])**2 + (p[1]-foot[1])**2, edge_id, lam, foot))
        candidates.sort(key=lambda row: (row[0], row[1]))
        chosen = None
        for _distance, edge_id, lam, foot in candidates:
            check()
            if any(_segments_cross(p, foot, a, b) for other, a, b in segments if other != edge_id):
                continue
            middle = ((p[0] + foot[0])/2, (p[1] + foot[1])/2)
            if point_in_region(middle, loops) is not Outcome.ABOVE:
                continue
            start3 = tuple(float(x) for x in model.vertex_position(tip))
            end3 = tuple(float(x) for x in frame.world(*foot))
            # only a trace INCIDENT (by vertex id) to this tip may touch the auxiliary at the tip; every other
            # trace must not meet it at all
            if any(certify_line_section(model.edges[t].curve, start3, end3,
                                        (start3,) if tip in (model.edges[t].start, model.edges[t].end) else (),
                                        float(model.tolerance.length)) is not True for t in traces):
                continue
            if any(_segments_cross(p, foot, q0, q1) for q0, q1 in accepted):
                continue
            chosen = (edge_id, lam, foot)
            break
        if chosen is None:
            raise PlateBoundaryCrossingRequired(
                'a trace tip inside the plate has no certified straight completion to the plate boundary')
        edge_id, lam, foot = chosen
        vertex, _pieces = split_edge_new_vertex(model, edge_id, lam, edge_draft, source_draft)
        aux = model.add_line(tip, vertex)
        # The candidate above used the IDEAL foot. The mutated model holds the ACTUAL float-rounded vertex, so
        # the segment that really exists is certified now (refusal aborts the whole candidate, which the
        # caller discards): exact incidence by vertex id is the only exemption.
        actual_frame, actual_loops = plate_loops(model, model.faces[plate_id])
        a_uv = actual_frame.uv(model.vertex_position(tip))
        b_uv = actual_frame.uv(model.vertex_position(vertex))
        if any(_segments_cross(a_uv, b_uv, *_line_endpoints(model, OrientedEdge(other, True), actual_frame))
               for other in _boundary_edges(model, plate_id) for item in (model.edges[other],)
               if vertex not in (item.start, item.end)):
            raise PlateBoundaryCrossingRequired('the created auxiliary crosses a plate boundary segment')
        if point_in_region(((a_uv[0] + b_uv[0])/2, (a_uv[1] + b_uv[1])/2), actual_loops) is not Outcome.ABOVE:
            raise PlateBoundaryCrossingRequired('the created auxiliary leaves the plate material')
        start3 = tuple(float(x) for x in model.vertex_position(tip))
        end3 = tuple(float(x) for x in model.vertex_position(vertex))
        for trace in traces:
            item = model.edges[trace]
            shared = (start3,) if tip in (item.start, item.end) else ()
            if certify_line_section(item.curve, start3, end3, shared, float(model.tolerance.length)) is not True:
                raise PlateBoundaryCrossingRequired('the created auxiliary is not certified against a trace')
        for other_uv, other_vertices in accepted_actual:
            if not {tip, vertex} & other_vertices and _segments_cross(a_uv, b_uv, *other_uv):
                raise PlateBoundaryCrossingRequired('the created auxiliary crosses an earlier auxiliary')
        created.append(aux)
        accepted.append((p, foot))
        accepted_actual.append(((a_uv, b_uv), {tip, vertex}))
    return created


def extract_plate_cells(model, plate_id, cut_edges, check):
    """Replace the plate by its directed cells: boundary pieces (material to the left) and the cut edges
    (physical traces and auxiliary completions, both directions) form the station graph; each directed
    walk is a cell loop, holes are attached by certified nesting. Returns the child face ids.

    Embedding argument (each part is a gate or refusal, never a sample): the plate boundary is a simple
    region (``plate_loops``); kept traces meet the plate boundary only at their end stations because no
    plate-edge root exists in a retained interval's interior (``_Clip.keep``) and distinct paths/arcs of one
    regular injective support are disjoint (``certify_trace_embedding``); straight tip auxiliaries are
    certified against every other boundary segment, trace and earlier auxiliary; at a boundary station
    exactly one cut edge is admitted (the interior side of the material-left boundary), at an interior
    station exactly two; a cell loop that would repeat an edge refuses; nesting uses ``_witness_inside``;
    certified signed areas must contain the exact oriented source area."""
    face = model.faces[plate_id]
    frame, _loops = plate_loops(model, face)
    source_loops = (face.loop, *face.holes)
    links = []
    source_area = Q(0)
    for index, loop in enumerate(source_loops):
        area = _polygon_area(model, loop, frame)[0]
        source_area += abs(area) if index == 0 else -abs(area)
        directed = loop if (area > 0) == (index == 0) else _reverse(loop)
        for use in directed:
            links.append((model.oriented_start_vertex(use), model.oriented_end_vertex(use), use, 'boundary'))
    for edge in cut_edges:
        item = model.edges[edge]
        links.append((item.start, item.end, OrientedEdge(edge, True), 'cut'))
        links.append((item.end, item.start, OrientedEdge(edge, False), 'cut'))
    outgoing = {}
    for i, link in enumerate(links):
        outgoing.setdefault(link[0], []).append(i)
    successor = {}
    for i, (_a, b, use, kind) in enumerate(links):
        options = [j for j in outgoing.get(b, ()) if links[j][2].edge != use.edge]
        if len(options) > 1:
            options = [j for j in options if links[j][3] != kind]
        if len(options) != 1:
            raise PlateBoundaryCrossingRequired('plate cut graph has an uncertified fan or a dangling station')
        successor[i] = options[0]
    cycles, visited = [], set()
    for seed in range(len(links)):
        if seed in visited:
            continue
        current, chain = seed, []
        while current not in visited:
            visited.add(current)
            chain.append(links[current][2])
            current = successor[current]
        if current != seed:
            raise GeometryError('plate cut graph contains an unaccounted directed walk')
        loop = tuple(chain)
        if len({use.edge for use in loop}) != len(loop):
            raise PlateBoundaryCrossingRequired('plate cell would repeat an edge (a bridge or dangling cut)')
        cycles.append((loop, loop_area(model, loop, frame, check)))
    positive = [loop for loop, area in cycles if area[0] > 0]
    negative = [loop for loop, area in cycles if area[1] < 0]
    if len(positive) + len(negative) != len(cycles) or not positive:
        raise PlateBoundaryCrossingRequired('plate cell orientation is not certified')
    low = sum(area[0] for _loop, area in cycles)
    high = sum(area[1] for _loop, area in cycles)
    # Consistency check (not the coverage proof): every cut edge is walked once in each direction, so the
    # exact signed cell areas sum to the oriented boundary areas; the certified interval sum must therefore
    # contain the exactly known source area, with no slack.
    if not low <= source_area <= high:
        raise PlateBoundaryCrossingRequired('plate cells do not conserve the plate material area')
    holes = [[] for _ in positive]
    for loop in negative:
        witness = _curve_box(model, loop[0].edge, frame, .5, .5)
        candidates = [i for i, outer in enumerate(positive)
                      if {use.edge for use in outer} != {use.edge for use in loop}
                      and _witness_inside(model, outer, witness, frame, check)]
        if len(candidates) > 1:
            innermost = []
            for i in candidates:
                box = _curve_box(model, positive[i][0].edge, frame, .5, .5)
                if all(i == j or _witness_inside(model, positive[j], box, frame, check) for j in candidates):
                    innermost.append(i)
            candidates = innermost
        if len(candidates) != 1:
            raise PlateBoundaryCrossingRequired('plate cell hole owner is not uniquely certified')
        holes[candidates[0]].append(loop)
    source_positive = _polygon_area(model, source_loops[0], frame)[0] > 0
    children = []
    for outer, inner in zip(positive, holes):
        if not source_positive:
            outer, inner = _reverse(outer), [_reverse(loop) for loop in inner]
        child = model.add_face_from_loop(outer, surface=face.surface)
        model._put_entity('face', replace(model.faces[child], holes=tuple(inner), metadata=dict(face.metadata),
                                          parameterization=face.parameterization))
        model.tag(EntityRef('face', child), *model.tags_for(face.ref))
        children.append(child)
    model._delete_entity('face', plate_id)
    model.record_replacement(face.ref, tuple(EntityRef('face', child) for child in children))
    return tuple(children)
