"""Source/event-identity graph for one regular plane cut of chart-segment trims.

Boundary and trace nodes are declared source vertices or certified root events.
Only material-left directed boundaries participate. At transverse boundary
events the successor switches between trim and trace; no angle/proximity sort
or floating area sign determines connectivity.
"""
from dataclasses import replace
from fractions import Fraction as Q

from .entities import OrientedEdge, EntityRef
from .errors import GeometryError
from .nurbs import CurveOnSurface, _IDENTITY
from .nurbs_section_faces import chart_loop_area, contains_chart_curves, chart_boundary_contact
from .section_events import section_segment_events
from .support_curves import PlaneNurbsCurve


_EVENT_KINDS = ('event', 'straight_plane_root', 'sweep_source_plane_root', 'plate')


def _straight_event_key(model, source_edge, plane, event):
    """One exact world-plane root on a shared source straight edge."""
    from .curves import Straight
    if not isinstance(source_edge.curve, Straight):
        return None
    a, b = [tuple(Q(float(x)) for x in vector) for vector in (plane.u_vector, plane.v_vector)]
    normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    first, last = [tuple(Q(float(x)) for x in model.vertex_position(v))
                   for v in (source_edge.start, source_edge.end)]
    numerator = sum(n*(Q(float(o))-x) for n, o, x in zip(normal, plane.origin, first))
    denominator = sum(n*(y-x) for n, x, y in zip(normal, first, last))
    if not denominator:
        raise GeometryError('straight trim crossing has no isolated world-plane root')
    parameter = numerator/denominator
    if not event.segment_interval[0] <= parameter <= event.segment_interval[1]:
        raise GeometryError('chart trim crossing does not enclose its declared straight-edge root')
    return ('straight_plane_root', source_edge.id, tuple(normal),
            sum(n*Q(float(o)) for n, o in zip(normal, plane.origin)), parameter)


def _area(model, loop, surface):
    for level in (2, 4, 6, 8):
        value, error = chart_loop_area(model, loop, surface, level)
        if abs(value) > error:
            return value, error
    raise GeometryError('plane cut cycle orientation is not certified')


def _inside(model, loops, point, clearance=Q(0)):
    answers = []
    for loop in loops:
        curves = [model.edges[e.edge].curve for e in loop]
        contact = chart_boundary_contact(curves, point, clearance)
        if contact is not False:
            return None
        answer = contains_chart_curves(curves, point, clearance)
        if answer is None:
            return None
        answers.append(answer)
    return answers[0] and not any(answers[1:])


def fragment_nurbs_plane(model, parent, plane, section, *, source_context=None, coalesce_sections=False,
                         min_closed_pieces=0, keep=None):
    """Retain every material cell, with certified chart-segment trim crossings.

    Previous implicit trims use exact same-support plane-equation events.
    """
    from .plane_split import _plane_sign
    context = source_context if source_context is not None else {
        'faces': dict(model.faces), 'edges': dict(model.edges), 'ledgers': {}, 'events': {}}
    face = context['faces'][parent]
    if hasattr(model, 'chart_face'):
        face = model.chart_face(face)
    surface = face.surface
    original = (face.loop, *face.holes)
    for loop in original:
        for use in loop:
            edge = context['edges'][use.edge]
            curve = (model.chart_edge(edge) if hasattr(model, 'chart_edge') else edge).curve
            if isinstance(curve, PlaneNurbsCurve) and curve.surface == surface:
                continue
            if (not isinstance(curve, CurveOnSurface) or curve.support != surface
                    or curve.transform != _IDENTITY or curve.pcurve.degree != 1
                    or len(curve.pcurve.controls) != 2):
                raise GeometryError('plane arrangement currently requires own-support chart-segment trims')
    curves = [PlaneNurbsCurve.from_plane(plane, surface, i) for i in range(len(section.paths))]
    events_by_edge, events_by_path = {}, [[] for _ in curves]
    for path_index, curve in enumerate(curves):
        for edge_id in sorted({e.edge for loop in original for e in loop}):
            source_edge = context['edges'][edge_id]
            boundary = (model.chart_edge(source_edge) if hasattr(model, 'chart_edge') else source_edge).curve
            cache_key = (curve.plane_record, curve.surface, curve.path, edge_id)
            if cache_key not in context['events']:
                context['events'][cache_key] = section_segment_events(curve, boundary)
            events = context['events'][cache_key]
            for index, event in enumerate(events):
                endpoint = event.section_interval[0] == event.section_interval[1] and event.section_interval[0] in (0, 1)
                key = ('event', path_index, edge_id, 'seam' if endpoint and section.paths[path_index].closed else index)
                key = ((model.source_straight_event_key(source_edge, plane, event, context)
                        if hasattr(model, 'source_straight_event_key') else
                        _straight_event_key(model, source_edge, plane, event)) or key)
                events_by_edge.setdefault(edge_id, []).append((key, event))
                events_by_path[path_index].append((key, event))
    # Bounded-section plate-boundary stations (``('plate', path, plate_edge, k)``): canonical exact roots of the
    # section path against the plate edge's exact chart equation, injected into the path's cut list BEFORE any
    # topology exists. They are not face-boundary events (never added to ``events_by_edge``): they only split
    # the full section path into ordered blocks, whose vertices are created by the same station constructor
    # below and recorded in ``station_vertices`` so the plate side can reuse the identical vertex.
    plate_stations = (context.get('plate_events') or {}) if source_context is not None else {}
    for path_index, rows in plate_stations.items():
        for key, event in rows:
            events_by_path[path_index].append((key, event))
    plate_positions = {key: event.position for rows in plate_stations.values() for key, event in rows
                       if getattr(event, 'position', None) is not None}
    trim_event_count = [sum(1 for key, _event in rows if key[0] != 'plate') for rows in events_by_path]
    vertices, boundary_links, trace_links, trace_ids = {}, [], [], []
    boundary_pieces = {}
    for edge_id in sorted({e.edge for loop in original for e in loop}):
        if edge_id in context['ledgers']:
            pieces, saved_vertices = context['ledgers'][edge_id]
            boundary_pieces[edge_id] = pieces
            vertices.update(saved_vertices)
            continue
        edge = context['edges'][edge_id]
        if hasattr(model, 'chart_edge'):
            edge = model.chart_edge(edge)
        first_key, last_key = ('vertex', edge.start), ('vertex', edge.end)
        vertices[first_key], vertices[last_key] = edge.start, edge.end
        entries = sorted(events_by_edge.get(edge_id, ()), key=lambda pair: pair[1].segment_interval)
        unique = []
        for key, event in entries:
            if unique and key == unique[-1][0]:
                continue  # Closed-path seam is one declared root.
            if unique and unique[-1][1].segment_interval[1] >= event.segment_interval[0]:
                raise GeometryError('trim crossing event order is not certified')
            unique.append((key, event))
        current, previous, keys, pieces = edge_id, Q(0), [first_key], []
        for event_index, (key, event) in enumerate(unique):
            parameter = sum(event.segment_interval)/2
            local = float((parameter-previous)/(1-previous))
            if not 0 < local < 1:
                raise GeometryError('trim root station is not representable')
            # an edge CREATED by this staging (a trace / auxiliary partition of an earlier cut) has no source
            # ancestry to record; its split children stay generated
            generated = context.get('generated')
            is_generated = generated is not None and current in generated
            draft = None if is_generated else context.get('edge_draft')
            if draft is not None:
                from .edge_subcurve_preimages import _edge_ancestry_definition
                parent_definition = _edge_ancestry_definition(model.raw if hasattr(model, 'raw') else model, current)
            source_draft = None if is_generated else context.get('source_edge_draft')
            from .prepared_material_source_transfer import (_before_edge_partition_split,
                _record_edge_partition_split)
            actual = model.raw if hasattr(model, 'raw') else model
            source_before = _before_edge_partition_split(source_draft, actual, current)
            vertex, (left, right) = model.split_edge(current, local)
            _record_edge_partition_split(source_draft, actual, current, local, (left, right), source_before)
            if draft is not None:
                from .edge_subcurve_preimages import _record_edge_subcurve_split
                actual = model.raw if hasattr(model, 'raw') else model
                _record_edge_subcurve_split(draft, current, local, (left, right),
                    actual.tolerance.effective_length(1.), model=actual, parent_definition=parent_definition)
            original_domain = tuple(map(Q, edge.curve.domain if isinstance(edge.curve, PlaneNurbsCurve)
                                        else edge.curve.pcurve.domain))
            left_curve = model.edges[left].curve
            stored_knot = Q(left_curve.domain[1] if isinstance(left_curve, PlaneNurbsCurve) else left_curve.pcurve.domain[1])
            represented = (stored_knot-original_domain[0])/(original_domain[1]-original_domain[0])
            preceding = unique[event_index-1][1].segment_interval[1] if event_index else Q(0)
            following = unique[event_index+1][1].segment_interval[0] if event_index+1 < len(unique) else Q(1)
            if not preceding < represented < following:
                raise GeometryError('rounded source station changes certified event order')
            if is_generated:
                generated.discard(current)
                generated.update((left, right))
            vertices[key] = vertex
            if 'station_log' in context:                  # certified event vertices for the bounded owner
                context['station_log'].append((key, vertex, tuple(event.section_interval), edge_id))
            pieces.append(left)
            keys.append(key)
            current, previous = right, represented
        pieces.append(current)
        keys.append(last_key)
        boundary_pieces[edge_id] = list(zip(pieces, keys, keys[1:]))
        context['ledgers'][edge_id] = (boundary_pieces[edge_id], {key: vertices[key] for key in keys})
    # Normalize material to the left, preserving the original face orientation
    # separately when the child loops are published.
    source_positive = _area(model, model.faces[parent].loop, surface)[0] > 0
    for loop_index, loop in enumerate(original):
        directed = []
        for use in loop:
            pieces = boundary_pieces[use.edge]
            directed.extend((edge, a, b, True) for edge, a, b in pieces) if use.forward else directed.extend(
                (edge, b, a, False) for edge, a, b in reversed(pieces))
        current_loop = tuple(OrientedEdge(edge, forward) for edge, a, b, forward in directed)
        positive = _area(model, current_loop, surface)[0] > 0
        if positive != (loop_index == 0):
            directed = [(edge, b, a, not forward) for edge, a, b, forward in reversed(directed)]
        boundary_links.extend((a, b, OrientedEdge(edge, forward), 'boundary')
                              for edge, a, b, forward in directed)
    for index, (curve, path) in enumerate(zip(curves, section.paths)):
        entries = sorted(events_by_path[index], key=lambda pair: pair[1].section_interval)
        for (first_key, first), (last_key, last) in zip(entries, entries[1:]):
            if first.section_interval[1] >= last.section_interval[0] and first_key != last_key:
                raise GeometryError('section crossings on different trims are not separated')
        cuts = [(Q(0), ('arc', index, 0), (Q(0), Q(0))),
                (Q(1), ('arc', index, 0 if path.closed else len(path.arcs)), (Q(1), Q(1)))]
        breaks = tuple(enumerate(curve.breakpoints()))
        # Internal certified arcs remain curve evaluation/sampling pieces.
        # They are not mandatory topological vertices of an open world trace.
        # A closed component keeps one certified interior arc joint so its
        # topology has at least two edges with distinct end vertices.
        if coalesce_sections:
            # min_closed_pieces > 0 (sheet-owned parent) keeps enough certified interior arc joints for that
            # many closed-loop pieces; 0 reproduces the single-joint legacy behaviour exactly.
            breaks = breaks[:max(1, min_closed_pieces-1)] if path.closed else ()
        cuts.extend((Q(t), ('arc', index, k+1), (Q(t), Q(t))) for k, t in breaks)
        for key, event in entries:
            cuts.append((sum(event.section_interval)/2, key, event.section_interval))
        cuts.sort(key=lambda pair: pair[0])
        merged = []
        for t, key, interval in cuts:
            if merged and t == merged[-1][0]:
                previous = merged[-1]
                if previous[2] != (t, t) or interval != (t, t):
                    raise GeometryError('approximate graph stations cannot establish event equivalence')
                if key[0] in _EVENT_KINDS and previous[1][0] == 'arc':
                    merged[-1] = (t, key, interval)  # Exact endpoint of this owner path.
                elif key != previous[1] and not (key[0] == 'arc' and previous[1][0] in _EVENT_KINDS):
                    raise GeometryError('graph station identities are not equivalent')
                continue
            if merged and merged[-1][2][1] >= interval[0]:
                raise GeometryError('section event cannot be separated from an arc station')
            merged.append((t, key, interval))
        if min_closed_pieces and path.closed and not entries and len(merged)-1 < min_closed_pieces:
            raise GeometryError('sheet-owned closed section loop has fewer than the structural minimum of '
                                'certified arc partitions; no extra station is certified for it')
        for (lo, start, lower_interval), (hi, end, upper_interval) in zip(merged, merged[1:]):
            gap_low, gap_high = lower_interval[1], upper_interval[0]
            witness = float((gap_low+gap_high)/2)
            if not gap_low < Q(witness) < gap_high:
                raise GeometryError('section interval has no representable certified gap witness')
            point = tuple(map(Q, curve.chart(witness)))
            from .nurbs_section_faces import _Chart, _corners
            rect = _Chart(surface).rect(*curve.chart_enclosure(witness))
            radius = max(sum(abs(p[i]-point[i]) for i in (0, 1)) for p in _corners(rect))
            material = _inside(model, (model.faces[parent].loop, *model.faces[parent].holes), point,
                               radius)
            if material is None:
                raise GeometryError('section material interval is not certified')
            if not material:
                continue
            # bounded-section clipping hook: ``keep(path_index, lo, hi, start_key, end_key)`` is a certified
            # decision that the material interval (with its two authenticated end stations) is also inside the
            # second operand (the plate); it may only DROP intervals, so an actual bounded cut is never
            # extended. ``None`` (default) keeps every material interval.
            if keep is not None and not keep(index, lo, hi, start, end):
                continue
            for t, key in ((lo, start), (hi, end)):
                if key not in vertices:
                    # a plate station carries its canonical exact position (route C: on the original plate
                    # line, inside the root enclosure); every other station keeps the evaluated curve point
                    position = plate_positions.get(key)
                    vertices[key] = model.add_point(*(position if position is not None else curve.evaluate(float(t))))
            edge = model.add_curve(vertices[start], vertices[end], curve.subcurve(float(lo), float(hi)))
            trace_ids.append(edge)
            if source_context is not None and 'block_log' in context:
                context['block_log'].append((edge, index, lo, hi, start, end))
                context.setdefault('station_vertices', {}).update({start: vertices[start], end: vertices[end]})
            trace_links.extend(((start, end, OrientedEdge(edge, True), 'trace'),
                                (end, start, OrientedEdge(edge, False), 'trace')))
    if not trace_ids:
        if any(trim_event_count):
            raise GeometryError('transverse trim events produced no material section interval')
        return (parent,), ()
    links = boundary_links+trace_links
    outgoing = {}
    for i, link in enumerate(links):
        outgoing.setdefault(link[0], []).append(i)
    successor = {}
    for i, (a, b, use, kind) in enumerate(links):
        options = [j for j in outgoing.get(b, ()) if links[j][2].edge != use.edge]
        if len(options) > 1:
            # A transverse trim contact has exactly one boundary and one
            # interior trace. Switching preserves each material-left cell.
            options = [j for j in options if links[j][3] != kind]
        if len(options) != 1:
            raise GeometryError('plane arrangement vertex has an uncertified fan')
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
            raise GeometryError('plane arrangement contains an unaccounted directed walk')
        loop = tuple(chain)
        area, error = _area(model, loop, surface)
        cycles.append((loop, area, error))
    positive = [loop for loop, area, error in cycles if area > 0]
    source_area = source_error = 0.
    for loop_index, loop in enumerate((model.faces[parent].loop, *model.faces[parent].holes)):
        area, error = _area(model, loop, surface)
        source_area += abs(area) * (1 if loop_index == 0 else -1)
        source_error += error
    retained_area = sum(area for loop, area, error in cycles)
    retained_error = sum(error for loop, area, error in cycles)
    if abs(source_area-retained_area) > source_error+retained_error:
        raise GeometryError('plane arrangement bounded chart areas do not conserve material')
    holes = [[] for _ in positive]
    for loop, area, error in cycles:
        if area > 0:
            continue
        point = tuple(map(Q, model.edges[loop[0].edge].curve.chart(.5)))
        candidates = []
        for i, outer in enumerate(positive):
            if {e.edge for e in loop} == {e.edge for e in outer}:
                continue
            if _inside(model, (outer,), point) is True:
                candidates.append(i)
        if len(candidates) > 1:
            # The source outer region and an inner section cycle can both
            # contain an authored hole. Choose the certified innermost cycle,
            # using nesting rather than estimated area or proximity.
            innermost = []
            for candidate in candidates:
                witness = tuple(map(Q, model.edges[positive[candidate][0].edge].curve.chart(.5)))
                if all(other == candidate or _inside(model, (positive[other],), witness) is True
                       for other in candidates):
                    innermost.append(candidate)
            candidates = innermost
        if len(candidates) != 1:
            raise GeometryError('plane arrangement hole owner is not uniquely certified')
        holes[candidates[0]].append(loop)
    made = []
    for outer, inner in zip(positive, holes):
        if not source_positive:
            outer = tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(outer))
            inner = [tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(loop)) for loop in inner]
        child = model.add_face_from_loop(outer, surface=surface)
        live_metadata = dict(model.raw.faces[parent].metadata) if hasattr(model, 'raw') else dict(face.metadata)
        model._put_entity('face', replace(model.faces[child], holes=tuple(inner), metadata=live_metadata,
                                          parameterization=face.parameterization))
        model.tag(EntityRef('face', child), *model.tags_for(face.ref))
        made.append(child)
    model._delete_entity('face', parent)
    model.record_replacement(face.ref, tuple(EntityRef('face', child) for child in made))
    return tuple(made), tuple(trace_ids)
