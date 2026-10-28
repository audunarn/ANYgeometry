"""Bounded plate / regular NURBS sections by shared exact boundary-event stations (milestone M1).

A bounded section is the part of a regular plane section of a NURBS support that lies inside an authored
straight plate lying in that plane (one NURBS parent, any number of open/closed components, nonconvex plate
with holes). The route (every step exact or typed-refusal; no tolerance, slack or auxiliary plane is added):

1. Exact plate chart. For each plate edge take the exact projected endpoints ``a', b'`` of the material
   definition used for containment (``ExactPlane.uv`` then ``world``: Fractions), and the exact line equation
   ``n = (b' - a') x nP``, ``d = n . a'`` (``nP`` the plate plane's exact normal). Both endpoints satisfy it
   exactly. ``section_roots`` receives ``(n, d)`` directly: every root of the section path against the plate
   edge's line, with a complete set of isolating intervals, an exact world enclosure and an exact edge
   parameter interval. Roots beyond the segment are not boundary events; roots on a segment endpoint,
   non-isolated roots and overlapping isolators (plate vertices, coincident events) are refused.
2. Shared stations. Each on-segment root becomes the canonical station ``('plate', path, plate_edge, k)`` with
   its isolator, injected into the existing full plane split of the NURBS parent as an extra station of the
   path's cut list BEFORE any topology exists (``fragment_nurbs_plane`` ``plate_events``). The split keeps ALL
   material blocks of the path; its vertex constructor creates the one vertex of the station.
3. Blocks. No plate station lies inside a block, so a block cannot cross the plate boundary and is uniformly
   inside or outside; ``interval_inside`` decides it on its middle third by exact box separation and exact
   point-in-region. A used station must change the status of its two incident blocks. Inside blocks are the
   physical joint traces (shared by NURBS cells and plate cells); outside blocks stay as NURBS-only auxiliary
   partitions (tag + owner registry), never physical cuts.
4. Plate side. Each plate edge is split AT the used station vertices (``split_edge_at_vertex``): the vertex
   must lie in the root's exact world enclosure, so its edge parameter lies in the root's exact parameter
   interval; the existing straight-split joint transport bounds the boundary drift. Tips inside the plate get
   certified straight auxiliaries, then the directed plate cells with holes are extracted.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
import weakref

from .errors import GeometryError
from .plane_nurbs import charge_split, depth_limit
from .plane_section_plate import PlateBoundaryCrossingRequired, _box, _line_box_clear

AUXILIARY_TAG = 'anygeometry.auxiliary_partition'


class PlateContactRefused(PlateBoundaryCrossingRequired):
    """Tangent, coincident, vertex-incident or otherwise uncertified plate/section contact."""


@dataclass(frozen=True)
class PlateRoot:
    path: int
    loop: int
    plate_edge: int
    index: int               # order of the root among this path's roots against this plate edge
    isolator: tuple          # exact curve-parameter interval (Fractions) holding exactly this root
    segment_interval: tuple  # exact parameter interval on the plate edge
    on_segment: bool         # strictly inside the open segment (else clearly outside it)
    lower: tuple             # exact world enclosure of the root (section enclosure, evaluation slack included)
    upper: tuple
    line: tuple = None       # the exact plate-chart line equation (normal, offset) it is a root of


@dataclass(frozen=True, eq=False)
class _ContactLedger:
    rows: tuple
    binding: str
    model_ref: object
    plate: int
    plate_binding: str


# Issuance truth lives OUTSIDE the raw-mutable public proof object. Values
# contain only weak owner references and detached immutable signatures.
_CONTACT_LEDGERS = {}


def _ledger_binding(curve, frame, loops, rows):
    from .definition_binding import definition_checksum
    return definition_checksum((curve, frame, loops, rows))


def _plate_binding(model, plate):
    from .definition_binding import definition_checksum
    face = model.faces[plate]
    edges = sorted({use.edge for loop in (face.loop, *face.holes) for use in loop})
    return definition_checksum((face, tuple((model.edges[e], model.vertices[model.edges[e].start],
                                            model.vertices[model.edges[e].end]) for e in edges)))


def _ledger_gaps(ledger, curve, frame, loops):
    from .planar_exact import ExactPlane
    from .nurbs_runtime_binding import validate_section_runtime
    fresh = ExactPlane(frame.origin, frame.u_vector, frame.v_vector)
    if any(getattr(frame, name) != getattr(fresh, name) for name in ('normal', '_scale', '_dual_u', '_dual_v')):
        raise PlateContactRefused('plate contact chart runtime changed')
    validate_section_runtime(curve)
    issued = _CONTACT_LEDGERS.get(id(ledger))
    if (issued is None or issued[0]() is not ledger
            or ledger.binding != issued[1]
            or _ledger_binding(curve, frame, loops, ledger.rows) != issued[1]
            or ledger.model_ref is not issued[2] or ledger.plate != issued[3]
            or ledger.plate_binding != issued[4]):
        raise PlateContactRefused('plate contact ledger is not freshly owner-issued for these definitions')
    model = issued[2]()
    if model is None or issued[3] not in model.faces or _plate_binding(model, issued[3]) != issued[4]:
        raise PlateContactRefused('plate contact ledger source boundary changed')
    roots = sorted((r for r in ledger.rows if r.on_segment), key=lambda r: r.isolator)
    result, previous = [], Q(0)
    for root in roots:
        lo, hi = root.isolator
        if not previous < lo <= hi < 1:
            raise PlateContactRefused('plate contact ledger has overlapping or endpoint events')
        result.append((previous, lo))
        previous = hi
    result.append((previous, Q(1)))
    return tuple(result)


def _gap_inside(ledger, curve, frame, loops, lo, hi, check):
    """Classify a verified subset of one complete-ledger contact-free gap."""
    from .nurbs import _float_down, _float_up
    from .plane_nurbs import charge_box
    from .planar_exact import point_in_region
    from .exact_predicates import Outcome
    lo, hi = Q(lo), Q(hi)
    gaps = _ledger_gaps(ledger, curve, frame, loops)
    if not lo < hi or sum(a <= lo < hi <= b for a, b in gaps) != 1:
        raise PlateContactRefused('requested interval is not inside one certified contact-free plate gap')
    segments = [(a, b) for loop in loops for a, b in zip(loop, loop[1:]+loop[:1])]
    width = hi-lo
    for fraction in (Q(1, 2), Q(1, 3), Q(2, 3), Q(1, 4), Q(3, 4)):
        centre, half = lo+fraction*width, width/8
        for _ in range(depth_limit(20)):
            check()
            low, high = _float_down(centre-half), _float_up(centre+half)
            if lo < Q(low) < Q(high) < hi:
                charge_box()
                box = _box(frame, curve.bounds(low, high))
                if all(_line_box_clear(a, b, box) for a, b in segments):
                    outcome = point_in_region(tuple((a+b)/2 for a, b in box), loops)
                    if outcome is not Outcome.EQUAL:
                        _ledger_gaps(ledger, curve, frame, loops)
                        return outcome is Outcome.ABOVE
            half /= 2
            charge_split()
    raise PlateContactRefused('no strictly clear charged witness certifies the contact-free plate gap')


@dataclass(frozen=True)
class PlateEvent:
    """The event object ``fragment_nurbs_plane`` needs: the exact section interval, and the canonical station
    position (route C: a float point exactly on the original 3D plate line and the plate support, inside the
    root's exact enclosure) that its vertex constructor uses instead of the evaluated curve point."""
    section_interval: tuple
    segment_interval: tuple
    plate_edge: int
    position: tuple = None


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def plane_equation(plane):
    """Exact rational ``(normal, offset)`` of a float-defined ``Plane`` (the construction every plane section
    of this package uses: normal = u x v of ``Fraction(float(...))``, offset = normal . origin)."""
    u, v, o = (tuple(Q(float(x)) for x in vector) for vector in (plane.u_vector, plane.v_vector, plane.origin))
    normal = _cross(u, v)
    return normal, _dot(normal, o)


def _projected(frame, position):
    """The exact point of the plate support used by the containment definition (Fractions)."""
    return tuple(Q(x) for x in frame.world(*frame.uv(position)))


def plate_roots(model, plate_face, plane, frame, curve, path_index, check, *, issue_ledger=False):
    """Every root of the section path against every plate edge's exact line, classified against its segment."""
    from .branch_predicates import section_roots
    face = model.faces[plate_face]
    plate_normal, _offset = plane_equation(plane)
    result = []
    for loop_index, loop in enumerate((face.loop, *face.holes)):
        for use in loop:
            check()
            edge = model.edges[use.edge]
            from .line_ranges import LineRange
            if type(edge.curve) is LineRange:
                edge.curve.certify_plane(frame, min(1.e-8, model.tolerance.effective_curve_fit_residual(1.)))
                a = _projected(frame, edge.curve.exact_point(0))
                b = _projected(frame, edge.curve.exact_point(1))
            else:
                a = _projected(frame, model.vertex_position(edge.start))
                b = _projected(frame, model.vertex_position(edge.end))
            e = tuple(y - x for x, y in zip(a, b))
            normal = _cross(e, plate_normal)
            if not any(normal):
                raise PlateContactRefused('degenerate plate edge or edge parallel to the plate normal')
            offset = _dot(normal, a)
            length = _dot(e, e)
            for index, root in enumerate(section_roots(curve, normal, offset)):
                lower = tuple(Q(float(x)) for x in root.lower)
                upper = tuple(Q(float(x)) for x in root.upper)
                corners = list(product(*zip(lower, upper)))
                lam = [_dot(tuple(p[k] - a[k] for k in range(3)), e)/length for p in corners]
                lo, hi = min(lam), max(lam)
                off = hi < 0 or lo > 1
                if not off and (not root.isolated or len(root.exact) != 1):
                    raise PlateContactRefused('section/plate-edge contact is not an isolated simple root')
                if not off and (lo <= 0 or hi >= 1):
                    raise PlateContactRefused('section meets a plate vertex or is not certified inside the edge')
                isolator = tuple(root.exact[0]) if root.isolated and len(root.exact) == 1 else (Q(root.t_lo), Q(root.t_hi))
                result.append(PlateRoot(path_index, loop_index, use.edge, index, isolator, (lo, hi), not off,
                                        lower, upper, (normal, offset)))
    on = sorted((r for r in result if r.on_segment), key=lambda r: r.isolator)
    for first, second in zip(on, on[1:]):
        if first.isolator[1] >= second.isolator[0]:
            raise PlateContactRefused('plate-boundary events on one section path are not separated')
    if issue_ledger:
        from .plane_section_plate import plate_loops
        actual_frame, loops = plate_loops(model, face)
        if actual_frame != frame:
            raise PlateContactRefused('plate root chart changed while issuing its complete contact ledger')
        rows = tuple(result)
        ledger = _ContactLedger(rows, _ledger_binding(curve, frame, loops, rows),
                                weakref.ref(model), plate_face, _plate_binding(model, plate_face))
        identity = id(ledger)
        _CONTACT_LEDGERS[identity] = (weakref.ref(ledger, lambda ref, key=identity: _CONTACT_LEDGERS.pop(key, None)),
                                     ledger.binding, ledger.model_ref, ledger.plate, ledger.plate_binding)
        _ledger_gaps(ledger, curve, frame, loops)
        return ledger
    return result


def interval_inside(curve, frame, loops, lo, hi, check):
    """``True``/``False`` when the whole parameter interval ``[lo, hi]`` is certified inside/outside the plate
    material; ``PlateBoundaryCrossingRequired`` when a boundary contact is not separated (never a guess)."""
    from .nurbs import _float_down, _float_up
    from .planar_exact import point_in_region
    from .exact_predicates import Outcome
    segments = [(a, b) for loop in loops for a, b in zip(loop, loop[1:] + loop[:1])]
    statuses, pending = set(), [(Q(lo), Q(hi), 0)]
    while pending:
        check()
        a, b, depth = pending.pop()
        box = _box(frame, curve.bounds(max(0., _float_down(a)), min(1., _float_up(b))))
        if all(_line_box_clear(p, q, box) for p, q in segments):
            status = point_in_region(tuple((x + y)/2 for x, y in box), loops)
            if status is not Outcome.EQUAL:
                statuses.add(status is Outcome.ABOVE)
                if len(statuses) > 1:
                    raise PlateBoundaryCrossingRequired('section interval is not uniformly inside the plate')
                continue
        if depth >= depth_limit(32):
            raise PlateBoundaryCrossingRequired('section/plate boundary contact is not certified')
        charge_split()
        middle = (a + b)/2
        pending.extend(((a, middle, depth + 1), (middle, b, depth + 1)))
    return statuses == {True}


def clip_path(curve, frame, loops, roots, closed, check, *, ledger=None):
    """``[(lo, hi, inside)]`` between the on-segment root isolators of one path (a transversality gate: every
    crossing must change the status and a closed path's wrap intervals must agree)."""
    pieces, previous = [], Q(0)
    for root in sorted(roots, key=lambda r: r.isolator):
        pieces.append((previous, root.isolator[0]))
        previous = root.isolator[1]
    pieces.append((previous, Q(1)))
    decided = []
    for lo, hi in pieces:
        if lo >= hi:
            raise PlateContactRefused('section interval between plate events is empty')
        status = (interval_inside(curve, frame, loops, lo, hi, check) if ledger is None
                  else _gap_inside(ledger, curve, frame, loops, lo, hi, check))
        decided.append((lo, hi, status))
    if any(a[2] == b[2] for a, b in zip(decided, decided[1:])):
        raise PlateBoundaryCrossingRequired('a plate-boundary event does not change the inside/outside status')
    if closed and roots and decided[0][2] != decided[-1][2]:
        raise PlateBoundaryCrossingRequired('closed section wrap intervals disagree about the plate material')
    return decided


def _curve_clear(curve, lower, upper, check, depth=0, lo=Q(0), hi=Q(1)):
    """Whether the curve over ``[lo, hi]`` stays out of the box ``[lower, upper]`` (exact; refuses at depth)."""
    from .nurbs import _float_down, _float_up
    check()
    low, high = curve.bounds(max(0., _float_down(lo)), min(1., _float_up(hi)))
    if any(Q(float(h)) < lw or Q(float(l)) > up for l, h, lw, up in zip(low, high, lower, upper)):
        return True
    if depth >= depth_limit(24):
        return False
    charge_split()
    middle = (lo + hi)/2
    return (_curve_clear(curve, lower, upper, check, depth + 1, lo, middle)
            and _curve_clear(curve, lower, upper, check, depth + 1, middle, hi))


def clearance_gate(model, plate, frame, curves, on_roots, tolerance, check, closed=None, parent=None):
    """Isolation of each canonical joint: its root enclosure expanded by the existing length tolerance must stay
    clear of every NON-incident plate edge (holes included), of every other on-segment root's expanded
    enclosure and of every OTHER section path, and the SAME path must pass ``local_certificate`` (all contacts
    with the neighbourhood accounted, one regular transverse inside component holding exactly the intended
    plate-line event over its whole local envelope, every other piece strictly outside) and the PARENT's own
    source trims (outer, holes, prior section trims) must be strictly clear of it (existing conservative world
    enclosures, bounded subdivision). Failure refuses (typed)."""
    from ._canonical_plate_joint import local_certificate
    from .plane_section_plate import _box, _line_box_clear, _line_endpoints
    tol = Q(float(tolerance))
    face = model.faces[plate]
    segments = []
    for loop in (face.loop, *face.holes):
        for use in loop:
            item = model.edges[use.edge]
            segments.append((use.edge, *_line_endpoints(model, use, frame)))
    flat = [(i, root) for i, rows in on_roots.items() for root in rows]
    for i, root in flat:
        check()
        lower = tuple(x - tol for x in root.lower)
        upper = tuple(x + tol for x in root.upper)
        box = _box(frame, (lower, upper))
        for edge_id, a, b in segments:
            if edge_id != root.plate_edge and not _line_box_clear(a, b, box):
                raise PlateContactRefused('a joint neighbourhood is not clear of a non-incident plate edge')
        for j, other in flat:
            if other is not root and all(max(l1, l2) <= min(u1, u2) for l1, u1, l2, u2 in
                                         zip(lower, upper, (x - tol for x in other.lower),
                                             (x + tol for x in other.upper))):
                raise PlateContactRefused('two joint neighbourhoods are not separated')
        for j, curve in enumerate(curves):
            if j != i and not _curve_clear(curve, lower, upper, check):
                raise PlateContactRefused('a joint neighbourhood is not clear of another section path')
        local_certificate(curves[i], root.isolator, root.lower, root.upper, root.line[0], root.line[1], tol,
                          bool(closed[i]) if closed else False,
                          (Q(float(sum(root.isolator)/2)),), check)       # the station vertex is made at this proxy
        if parent is not None:
            from ._canonical_plate_joint import trims_clear
            from .definition_binding import definition_checksum
            own = (curves[i].plane_record, curves[i].path, definition_checksum(curves[i].surface))
            trims_clear(model, parent, lower, upper, own, check)           # original outer / hole / prior trims


def vertex_in_root_enclosure(model, vertex, root):
    """The station vertex (a float evaluation at the isolator midpoint) must lie in the root's exact world
    enclosure; the enclosure is convex, so the vertex's projection onto the plate edge then lies in the root's
    exact segment interval. Exact Fractions, no slack."""
    position = tuple(Q(float(x)) for x in model.vertex_position(vertex))
    from ._canonical_plate_joint import joint_l1
    return joint_l1(position, root.lower, root.upper) <= Q(model.tolerance.effective_length(1.))


def stage_bounded(candidate, plate, parent, qualification, check, edge_draft, source_edge_draft):
    """The connected bounded-section route for ONE NURBS parent and one authored straight plate.

    Returns ``None`` when the legacy route applies (no on-segment plate root on any path: the plate contains
    or misses every path); otherwise ``(parent_mappings, physical_traces, plate_children, auxiliary_edges)``
    staged in the caller's transaction (no physical trace inside the plate: empty traces, the caller then
    discards the candidate). See the module docstring for the proof route."""
    from .nurbs import NurbsSurface
    from .plane_nurbs import plane_nurbs_section
    from .plane_section_plate import plate_loops
    from .plane_split import _stage_plane_split
    from .support_curves import PlaneNurbsCurve
    from .bounded_cut_cells import (split_edge_at_vertex, complete_tips, extract_plate_cells, split_generated_edge)
    from .prepared_material_source_transfer import _mark_generated_edges
    face = candidate.faces[parent]
    if not isinstance(face.surface, NurbsSurface) or (
            face.parameterization is not None and face.parameterization != face.surface):
        return None
    plane = candidate.faces[plate].surface
    section = plane_nurbs_section(plane, face.surface)
    if section.coincident or len(section.paths) > qualification.max_components:
        raise GeometryError('bounded section component budget or coincident support refused')
    if not section.paths:
        return None
    frame, loops = plate_loops(candidate, candidate.faces[plate])
    curves = [PlaneNurbsCurve.from_plane(plane, face.surface, i) for i in range(len(section.paths))]
    ledgers = {i: plate_roots(candidate, plate, plane, frame, curve, i, check, issue_ledger=True)
               for i, curve in enumerate(curves)}
    roots = {i: ledger.rows for i, ledger in ledgers.items()}
    on_roots = {i: [r for r in rows if r.on_segment] for i, rows in roots.items()}
    if not any(on_roots.values()):
        return None
    for i, curve in enumerate(curves):
        if on_roots[i]:
            clip_path(curve, frame, loops, on_roots[i], section.paths[i].closed, check,
                      ledger=ledgers[i])   # complete-ledger transversality gates
    from ._canonical_plate_joint import choose_station, exact, JointRefused
    tolerance = candidate.tolerance.effective_length(1.)
    clearance_gate(candidate, plate, frame, curves, on_roots, tolerance, check,
                   closed=[path.closed for path in section.paths], parent=parent)
    plate_normal, plate_offset = plane_equation(plane)
    events, root_of, carrier_parameters = {}, {}, {}
    for i, rows in on_roots.items():
        events[i] = []
        for root in sorted(rows, key=lambda r: r.isolator):
            key = ('plate', i, root.plate_edge, root.index)
            item = candidate.edges[root.plate_edge]
            from .line_ranges import LineRange
            carrier = item.curve if type(item.curve) is LineRange else LineRange(
                candidate.vertex_position(item.start), candidate.vertex_position(item.end))
            carrier.certify_plane(frame, min(1.e-8, candidate.tolerance.effective_curve_fit_residual(1.)))
            try:
                position, _lam = choose_station(carrier.exact_point(0), carrier.exact_point(1),
                    plate_normal, plate_offset, root.lower, root.upper, tolerance)
            except JointRefused:
                position, _lam = choose_station(carrier.exact_point(0), carrier.exact_point(1),
                    plate_normal, plate_offset, root.lower, root.upper, tolerance, retained=True)
            carrier_parameters[key] = _lam
            events[i].append((key, PlateEvent(root.isolator, root.segment_interval, root.plate_edge, position)))
            root_of[key] = root
    generated = set()
    context = {'generated': generated, 'plate_events': events, 'block_log': [], 'station_vertices': {}}
    mappings, traces, _positive, _negative = _stage_plane_split(candidate, (parent,), plane, qualification,
        edge_draft=edge_draft, source_edge_draft=source_edge_draft, extra_context=context)
    children = tuple(child for _parent, group in mappings for child in group)
    traces = tuple(traces)
    if not traces:
        return ([(parent, children)], (), (), ())
    _mark_generated_edges(source_edge_draft, traces)
    blocks = {row[0]: row for row in context['block_log']}
    if set(blocks) != set(traces):
        raise GeometryError('bounded section block accounting is inconsistent with the created traces')
    status = {}
    for edge, row in blocks.items():
        check()
        _edge, path, lo, hi, start, end = row
        lo, hi = Q(lo), Q(hi)
        core_lo, core_hi = lo, hi
        # The complete root ledger excludes every interior crossing. The ONLY
        # discarded bands are authenticated endpoint events whose full local
        # C2 proof (including rounded section proxies) passed clearance_gate.
        for root in on_roots[path]:
            key = ('plate', path, root.plate_edge, root.index)
            a, b = root.isolator
            if start == key:
                if lo != (a+b)/2:
                    raise PlateContactRefused('start proxy does not belong to its certified contact band')
                core_lo = max(core_lo, b)
            elif end == key:
                if hi != (a+b)/2:
                    raise PlateContactRefused('end proxy does not belong to its certified contact band')
                core_hi = min(core_hi, a)
            elif not (b <= lo or a >= hi):
                raise PlateContactRefused('an unaccounted plate contact band meets a staged block')
        if (start[0] == 'plate' and start not in root_of
                or end[0] == 'plate' and end not in root_of):
            raise PlateContactRefused('staged block endpoint has no authenticated plate event')
        if tuple(candidate.edges[edge].curve.domain) != (float(lo), float(hi)):
            raise PlateContactRefused('staged section block changed its certified source interval')
        status[edge] = _gap_inside(ledgers[path], curves[path], frame, loops, core_lo, core_hi, check)
    incident = {}
    for edge, (_e, _path, _lo, _hi, start, end) in blocks.items():
        for key in (start, end):
            if key[0] == 'plate':
                incident.setdefault(key, []).append(edge)
    for key, edges in incident.items():
        if len(edges) > 2 or (len(edges) == 2 and status[edges[0]] == status[edges[1]]):
            raise PlateBoundaryCrossingRequired('a plate-boundary station does not change the inside/outside status')
    inside = [edge for edge in traces if status[edge]]
    outside = [edge for edge in traces if not status[edge]]
    if not inside:
        return ([(parent, children)], (), (), ())
    stations = context['station_vertices']
    if bool(candidate._face_structural_uses.get(plate)) or any(candidate._face_structural_uses.get(c) for c in children):
        # sheet cells need >= 3 coedges: a physical block joining two stations of the SAME plate edge would give a
        # two-coedge lens, so it gets one more owned station (an interior point of its own regular curve)
        grown = []
        for edge in inside:
            _e, _path, _lo, _hi, start, end = blocks[edge]
            if start[0] == 'plate' and end[0] == 'plate' and start[2] == end[2]:
                grown.extend(split_generated_edge(candidate, edge, source_edge_draft))
            else:
                grown.append(edge)
        inside = grown
    used = sorted({key for edge in blocks if status[edge] for key in (blocks[edge][4], blocks[edge][5])
                   if key[0] == 'plate'})
    by_edge = {}
    for key in used:
        root, vertex = root_of[key], stations[key]
        if not vertex_in_root_enclosure(candidate, vertex, root):
            raise PlateBoundaryCrossingRequired('a station vertex is not inside its root enclosure')
        by_edge.setdefault(root.plate_edge, []).append((root, vertex, carrier_parameters[key]))
    for edge_id, rows in by_edge.items():
        rows.sort(key=lambda row: row[2])
        for first, second in zip(rows, rows[1:]):
            if first[0].segment_interval[1] >= second[0].segment_interval[0]:
                raise PlateBoundaryCrossingRequired('plate-edge stations are not separated along their edge')
        current = edge_id
        previous = Q(0)
        for _root, vertex, parameter in rows:
            local = (parameter-previous)/(1-previous)
            if not 0 < local < 1:
                raise PlateBoundaryCrossingRequired('certified plate station is not interior to its plate edge')
            _first, current = split_edge_at_vertex(candidate, current, vertex, local, edge_draft, source_edge_draft)
            previous = parameter
    aux_plate = complete_tips(candidate, plate, tuple(inside), check, edge_draft, source_edge_draft)
    plate_children = extract_plate_cells(candidate, plate, (*inside, *aux_plate), check)
    auxiliary = tuple(sorted({e for e in outside if e in candidate.edges} | set(aux_plate)))
    return ([(parent, children)], tuple(inside), plate_children, auxiliary)


def tag_auxiliary(candidate, auxiliary):
    """Persist the auxiliary tag on every auxiliary edge (tags are labels, never lineage authority)."""
    from .entities import EntityRef
    for edge in auxiliary:
        if edge in candidate.edges:
            candidate.tag(EntityRef('edge', edge), AUXILIARY_TAG)
