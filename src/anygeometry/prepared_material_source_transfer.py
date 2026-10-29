"""Authenticated live source/clone lineage and curvilinear cell transport.

Ordinary cloning, serialization, support matching and historical authored
snapshots do not issue lineage. Only the owner clone boundary and qualified
whole-material publications below can compose it.
"""
from dataclasses import dataclass, fields
from fractions import Fraction as Q
from weakref import WeakKeyDictionary, ref

from .errors import GeometryError
from .definition_binding import definition_checksum
from .native_support_snapshots import capture_native_supports
from .nurbs import NurbsSurface
from .surfaces import Plane
from .prepared_material_curvilinear import (_literal, _detached, CurvilinearMaterialCell,
    _source_transfer_cell_inputs, validate_prepared_material_curvilinear_cells_binding)

_LINKS = WeakKeyDictionary()
_AUXILIARY = WeakKeyDictionary()      # model -> {edge id: complete edge definition checksum}, owner-issued
_TRANSFERS = WeakKeyDictionary()
_EDGE_PARTITIONS = WeakKeyDictionary()
_PARTITION_COMMITS = WeakKeyDictionary()  # tracked model -> (complete owner-commit state, failure reason)


def _track_partition_commit(model, state):
    # A read or another clone must never bless raw changes on an operand
    # already tracked by an older lineage. Only publication can advance it.
    if model not in _PARTITION_COMMITS:
        _PARTITION_COMMITS[model] = (state, None)
        model._prepared_partition_lineage_tracked = True


def _capture_partition_commit(model):
    """Owner publication witness, captured before arbitrary consumer hooks."""
    if model not in _PARTITION_COMMITS:
        return
    _PARTITION_COMMITS[model] = (None, 'committed partition state capture did not complete')
    try:
        state = _state(model)
    except Exception as error:
        # The model commit has already succeeded. Keep its behavior and make
        # optional proof reuse unusable; qualification failure is not a seal.
        _PARTITION_COMMITS[model] = (None, f'{type(error).__name__}: {error}')
        return
    _PARTITION_COMMITS[model] = (state, None)


def _state(model):
    if model._transaction_journal is not None:
        raise GeometryError('source transport requires committed models')
    try:
        from .serialization import _qualified_model_state
        # Full current content and reverse incidence are bound on EVERY read.
        # Only unchanged topology qualification is reused; native state below
        # is still freshly captured, independently of persisted definitions.
        document = _qualified_model_state(model)
        return model.revision, document['checksum']['value'], capture_native_supports(model)
    except (AttributeError, TypeError, ValueError, KeyError) as error:
        raise GeometryError('source transport complete definitions are malformed or changed') from error


def _effective(model, face_id):
    from .nurbs_sweep_chart import has_sweep_chart, chart_model, _tensor
    from ._translational_pairs import _sweep
    face = model.faces[face_id]
    if has_sweep_chart(face):
        return chart_model(model, face_id).support
    if isinstance(face.surface, (Plane, NurbsSurface)) and (face.parameterization is None or face.parameterization is face.surface):
        return face.surface
    found = _sweep(model, face_id)
    if found is not None:
        return _tensor(found[0], found[1])
    raise GeometryError('source chart transport support is not qualified')


def _chart_map(source, face, support):
    if definition_checksum(_effective(source, face)) != definition_checksum(support):
        raise GeometryError('qualified descendant changed its original physical support')
    if isinstance(support, Plane):
        return ((Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0)))
    from .nurbs_sweep_chart import authored_query_uv
    origin, u, v = authored_query_uv(source, face, support, ((0., 0.), (1., 0.), (0., 1.)))
    return tuple((Q(u[i])-Q(origin[i]), Q(v[i])-Q(origin[i]), Q(origin[i])) for i in (0, 1))


def _map_uv(matrix, uv):
    return tuple(float(a*Q(u)+b*Q(v)+c) for a, b, c in matrix for u, v in (uv,))


def _edge_definition(model, identifier):
    edge = model.edges[identifier]
    points = tuple(tuple(float(x) for x in model.vertex_position(vertex)) for vertex in (edge.start, edge.end))
    return definition_checksum((edge, points))


def _begin_edge_partition(model):
    """Private transaction-local producer ledger, never reconstructed by fitting."""
    initial = tuple((edge, _edge_definition(model, edge)) for edge in sorted(model.edges))
    return {'initial': initial, 'generated': set(),
            'current': {edge: (edge, Q(0), Q(1), signature) for edge, signature in initial}}


def _mark_generated_edges(draft, edge_ids):
    """Edges CREATED by this staging (section traces, auxiliary partitions) have no source ancestry; they are
    tracked separately so their later splits (by a second cut of the same staging) never touch the ledger of
    initial source edges."""
    if draft is not None:
        draft['generated'].update(edge_ids)


def _before_edge_partition_split(draft, model, identifier):
    if draft is None:
        return None
    if identifier in draft.get('generated', ()):
        return ('generated',)
    row = draft['current'].get(identifier)
    if row is None or row[3] != _edge_definition(model, identifier):
        raise GeometryError('producer edge partition source changed before split')
    edge = model.edges[identifier]
    points = tuple(tuple(Q(float(x)) for x in model.vertex_position(vertex)) for vertex in (edge.start, edge.end))
    return edge, points


def _record_edge_partition_split(draft, model, identifier, parameter, children, before, exact=None):
    """``exact``: the EXACT rational split parameter of a straight edge whose split vertex is exactly collinear
    (route C); the recorded child spans and the vertex check then use it instead of the rounded float."""
    if draft is None:
        return
    if isinstance(before, tuple) and before == ('generated',):
        draft['generated'].discard(identifier)
        draft['generated'].update(children)
        return
    from .curves import Straight, LineRange
    from .nurbs import NurbsCurve, CurveOnSurface
    from .support_curves import PlaneNurbsCurve
    old, points = before
    root, first, last, signature = draft['current'].pop(identifier)
    left, right = (model.edges[child] for child in children)
    if (left.start != old.start or right.end != old.end or left.end != right.start
            or len(set(children)) != 2):
        raise GeometryError('producer edge partition endpoint identity changed')
    if isinstance(old.curve, LineRange) or (isinstance(old.curve, Straight)
            and all(isinstance(child.curve, LineRange) for child in (left, right))):
        carrier = old.curve if isinstance(old.curve, LineRange) else LineRange(*points)
        middle = Q(exact) if exact is not None else Q(parameter)
        expected = (carrier.subcurve(0, middle), carrier.subcurve(middle, 1))
        if any(child.curve != curve for child, curve in zip((left, right), expected)):
            raise GeometryError('retained line children do not partition their exact carrier')
        target = tuple(Q(float(x)) for x in model.vertex_position(left.end))
        if sum(abs(x-y) for x, y in zip(target, carrier.exact_point(middle))) > Q(model.tolerance.effective_length(1.)):
            raise GeometryError('retained carrier station exceeds joint correspondence tolerance')
        local = ((Q(0), middle), (middle, Q(1)))
    elif isinstance(old.curve, (NurbsCurve, CurveOnSurface, PlaneNurbsCurve)):
        expected = (old.curve.subcurve(0., float(parameter)), old.curve.subcurve(float(parameter), 1.))
        if any(definition_checksum(child.curve) != definition_checksum(curve)
               for child, curve in zip((left, right), expected)):
            raise GeometryError('producer edge partition is not the actual source restriction')
        def domain(curve):
            return tuple(map(Q, curve.pcurve.domain if isinstance(curve, CurveOnSurface) else curve.domain))
        a, b = domain(old.curve)
        domains = tuple(domain(child.curve) for child in (left, right))
        if a == b or domains[0][0] != a or domains[0][1] != domains[1][0] or domains[1][1] != b:
            raise GeometryError('producer restriction domains do not exactly partition their source')
        local = tuple(tuple((value-a)/(b-a) for value in interval) for interval in domains)
    elif isinstance(old.curve, Straight) and all(isinstance(child.curve, Straight) for child in (left, right)):
        middle = Q(exact) if exact is not None else Q(float(parameter))
        local = ((Q(0), middle), (middle, Q(1)))
        target = tuple(Q(float(x)) for x in model.vertex_position(left.end))
        bound = sum(abs(x-((1-middle)*a+middle*b)) for x, a, b in zip(target, *points))
        if bound > Q(model.tolerance.effective_length(1.)):
            raise GeometryError('source straight split exceeds existing joint correspondence tolerance')
    else:
        raise GeometryError('producer edge partition curve family is not qualified')
    for child, (lo, hi) in zip(children, local):
        if not 0 <= lo < hi <= 1:
            raise GeometryError('producer edge partition parameter order is invalid')
        draft['current'][child] = (root, first+(last-first)*lo, first+(last-first)*hi,
                                   _edge_definition(model, child))


def _finish_edge_partition(draft, model):
    entries = tuple((edge, *row) for edge, row in sorted(draft['current'].items()) if edge in model.edges)
    result = (draft['initial'], entries)
    _validate_edge_partition(result, None, model, issuing=True)
    _EDGE_PARTITIONS[model] = result
    return result


def _validate_edge_partition(partition, source, prepared, *, issuing=False):
    if not issuing and _EDGE_PARTITIONS.get(prepared) is not partition:
        raise GeometryError('producer edge partition was copied or not owner-issued')
    if type(partition) is not tuple or len(partition) != 2:
        raise GeometryError('producer edge partition is malformed')
    initial, entries = partition
    if source is not None and initial != tuple((edge, _edge_definition(source, edge)) for edge in sorted(source.edges)):
        raise GeometryError('producer edge partition belongs to changed source edges')
    covers = {edge: [] for edge, signature in initial}
    seen = set()
    for child, parent, a, b, signature in entries:
        if (child in seen or parent not in covers or type(a) is not Q or type(b) is not Q
                or not 0 <= a < b <= 1 or child not in prepared.edges
                or signature != _edge_definition(prepared, child)):
            raise GeometryError('producer edge partition definitions or intervals changed')
        seen.add(child)
        covers[parent].append((a, b))
    for intervals in covers.values():
        reach = Q(0)
        for a, b in sorted(intervals):
            if a != reach:
                raise GeometryError('producer edge partition omits or overlaps source material')
            reach = b
        if reach != 1:
            raise GeometryError('producer edge partition omits source material')
    return entries

def _edge_transport_error(source, prepared, source_edge, prepared_edge, first, last):
    """Full-span error of producer-partitioned straight world targets.

    Parameter identity is issued by the split producer. This bound qualifies
    its rounded world representation; it never discovers edge ownership.
    Exact native restrictions need no world-curve refit or sampled comparison.
    """
    from .curves import Straight, LineRange
    original, child = source.edges[source_edge], prepared.edges[prepared_edge]
    if isinstance(child.curve, LineRange) and isinstance(original.curve, (Straight, LineRange)):
        carrier = original.curve if isinstance(original.curve, LineRange) else LineRange(
            source.vertex_position(original.start), source.vertex_position(original.end))
        if child.curve != carrier.subcurve(first, last):
            raise GeometryError('retained line descendant changed its raw source carrier')
        error = max(sum(abs(Q(float(x))-y) for x, y in zip(prepared.vertex_position(v), child.curve.exact_point(t)))
                    for v, t in ((child.start, 0), (child.end, 1)))
        if error > min(Q(source.tolerance.effective_length(1.)), Q(prepared.tolerance.effective_length(1.))):
            raise GeometryError('retained source carrier transport exceeds existing joint tolerance')
        return error
    if not isinstance(original.curve, Straight):
        return Q(0)
    if not isinstance(child.curve, Straight):
        raise GeometryError('straight source descendant changed curve family')
    endpoints = tuple(tuple(Q(float(x)) for x in source.vertex_position(vertex))
                      for vertex in (original.start, original.end))
    error = Q(0)
    for vertex, parameter in zip((child.start, child.end), (first, last)):
        target = tuple(Q(float(x)) for x in prepared.vertex_position(vertex))
        error = max(error, sum(abs(x-((1-parameter)*a+parameter*b))
                               for x, a, b in zip(target, *endpoints)))
    if error > min(Q(source.tolerance.effective_length(1.)), Q(prepared.tolerance.effective_length(1.))):
        raise GeometryError('composed source edge transport exceeds existing joint tolerance')
    return error

def _register_auxiliary(model, edge_ids):
    """Called ONLY by the bounded-section owner at publication: the straight auxiliary partition edges it
    created, bound to their complete edge definitions. A persisted tag alone never grants this authority;
    loaded or copied models start with none and must be regenerated by the owner. Returns the prior entry
    (for rollback)."""
    previous = _AUXILIARY.get(model)
    _AUXILIARY[model] = {edge: _edge_definition(model, edge) for edge in edge_ids if edge in model.edges}
    return previous


def _restore_auxiliary(model, previous):
    if previous is None:
        _AUXILIARY.pop(model, None)
    else:
        _AUXILIARY[model] = previous


def _qualified_auxiliary(model, edge_id):
    entry = _AUXILIARY.get(model)
    return bool(entry) and edge_id in entry and entry[edge_id] == _edge_definition(model, edge_id)


def _register_source_clone(source, prepared):
    """Called ONLY by clone_prepared_geometry after its genuine clone."""
    from .serialization import to_dict, _clone_content_payload
    before, copied = to_dict(source), to_dict(prepared)
    if _clone_content_payload(before) != _clone_content_payload(copied):
        raise GeometryError('immediate source clone changed complete definitions')
    source_state, prepared_state = _state(source), _state(prepared)
    if source_state[2] != prepared_state[2]:
        raise GeometryError('immediate source clone changed native definitions')
    routes = [(ref(source), source_state, prepared_state,
               tuple((face, (face,)) for face in sorted(source.faces)),
               tuple((edge, edge, Q(0), Q(1)) for edge in sorted(source.edges)))]
    # Carry authenticated ancestor routes across this exact clone, rather
    # than confusing authored history with the immediate live source.
    for owner, original, current, mapping, edge_map in _LINKS.get(source, ()):
        ancestor = owner()
        if ancestor is not None and current == source_state and _state(ancestor) == original:
            routes.append((owner, original, prepared_state, mapping, edge_map))
    if len(routes) > 64:
        raise GeometryError('source clone lineage depth limit exceeded')
    if _state(source) != source_state or _state(prepared) != prepared_state:
        raise GeometryError('source clone changed while issuing lineage')
    _LINKS[prepared] = tuple(routes)
    _track_partition_commit(source, source_state)
    _track_partition_commit(prepared, prepared_state)
    # genuine owner clone boundary: qualified auxiliary edges keep their authority only when the clone
    # reproduces each complete edge definition
    carried = {edge: signature for edge, signature in _AUXILIARY.get(source, {}).items()
               if edge in prepared.edges and _edge_definition(prepared, edge) == signature}
    if carried:
        _AUXILIARY[prepared] = carried


def _register_source_closure(source, work, faces, edges, source_state, work_state):
    """Called ONLY by ``closure_source.qualify_model_closure_source`` after its independent reconstruction.

    ``faces``: ``(source_face, work_face)`` pairs; ``edges``: ``(work_edge, source_edge)`` pairs (whole
    edges, parameters 0..1). Nothing is published unless both live models still equal the qualified
    states; the prior lineage is replaced only at the final assignment, and a prior route from the same
    source is replaced rather than accumulated.
    """
    if _state(source) != source_state or _state(work) != work_state:
        raise GeometryError('closure source definitions changed before lineage registration')
    route = (ref(source), source_state, work_state,
             tuple((parent, (child,)) for parent, child in sorted(faces)),
             tuple((child, parent, Q(0), Q(1)) for child, parent in sorted(edges)))
    routes = [route]
    for owner, original, current, mapping, edge_map in _LINKS.get(work, ()):
        ancestor = owner()
        if (ancestor is not None and ancestor is not source and current == work_state
                and _state(ancestor) == original):
            routes.append((owner, original, current, mapping, edge_map))
    if len(routes) > 64:
        raise GeometryError('source clone lineage depth limit exceeded')
    if _state(source) != source_state or _state(work) != work_state:
        raise GeometryError('closure source definitions changed while issuing lineage')
    complete = tuple(routes)
    if _LINKS.get(work) == complete:
        return                  # exact same current routes: keep the existing entry (and issued receipts) intact
    _LINKS[work] = complete
    _track_partition_commit(source, source_state)
    _track_partition_commit(work, work_state)


def _prepare_source_lineage(model, candidate, replacements, edge_partition=None):
    """Stage only after a qualified owner's COMPLETE material partition."""
    old = _LINKS.get(model)
    if old is None:
        return None
    current, future = _state(model), _state(candidate)
    changes = dict(replacements)
    if (any(parent not in model.faces or not children or any(child not in candidate.faces for child in children)
            for parent, children in changes.items())
            or {child for parent in model.faces for child in changes.get(parent, (parent,))} != set(candidate.faces)):
        raise GeometryError('source lineage publication omits or invents material descendants')
    routes = []
    parts = _validate_edge_partition(edge_partition, model, candidate) if edge_partition is not None else None
    from .serialization import to_dict
    old_edges = {row["id"]: row for row in to_dict(model)["edges"]}
    new_edges = {row["id"]: row for row in to_dict(candidate)["edges"]}
    for owner, original, bound, mapping, edge_map in old:
        source = owner()
        if source is None or bound != current or _state(source) != original:
            continue  # stale ancestry never becomes new authority
        descendants = tuple((parent, tuple(child for face in faces for child in changes.get(face, (face,))))
                            for parent, faces in mapping)
        for parent, children in descendants:
            for child in children:
                _chart_map(source, parent, _effective(candidate, child))
        if parts is None:
            retained = tuple((child, parent, a, b) for child, parent, a, b in edge_map
                             if child in new_edges and new_edges[child] == old_edges.get(child))
        else:
            previous = {edge: (parent, a, b) for edge, parent, a, b in edge_map}
            retained = tuple((child, previous[edge][0],
                              previous[edge][1]+(previous[edge][2]-previous[edge][1])*a,
                              previous[edge][1]+(previous[edge][2]-previous[edge][1])*b)
                             for child, edge, a, b, signature in parts if edge in previous)
        routes.append((owner, original, future, descendants, retained))
    return old, tuple(routes)


def _publish_source_lineage(model, staged):
    if staged is None:
        return
    old, routes = staged
    if _LINKS.get(model) is not old:
        raise GeometryError('source lineage changed during publication')
    now = _state(model)
    if any(current != now or owner() is None or _state(owner()) != original
           for owner, original, current, mapping, edge_map in routes):
        raise GeometryError('source lineage source or published definitions changed')
    _LINKS[model] = routes
    _track_partition_commit(model, now)


def _restore_source_lineage(model, staged):
    if staged is not None:
        _LINKS[model] = staged[0]


@dataclass(frozen=True, eq=False)
class PreparedMaterialSourceTransfer:
    source_model_id: object
    prepared_model_id: object
    source_revision: int
    prepared_revision: int
    source_checksum: str
    prepared_checksum: str
    source_to_prepared_faces: tuple
    # Transferred carrier rows retain literal XYZ/connectivity; source UV is
    # an evaluation target. Exact chart_correspondence, live partition lineage
    # and the registered prepared proof supply authority, not this row alone.
    source_cells: tuple
    prepared_cells: tuple
    reference_uv_by_cell: tuple
    prepared_reference_uv_by_cell: tuple
    chart_correspondence: tuple
    surface_errors: tuple
    source_map_bounds: tuple
    edge_parameter_maps: tuple
    source_trace_ownership: tuple
    boundary_station_correspondence: tuple
    local_field_fingerprints: tuple = ()
    local_boundary_errors: tuple = ()


def _signature(receipt):
    return tuple((f.name, _literal(getattr(receipt, f.name))) for f in fields(receipt))


def _frontier(source, prepared, receipts, callback):
    states = (_state(source), _state(prepared))
    literal = _literal(receipts)
    links = _LINKS.get(prepared)
    operands = tuple(_source_transfer_cell_inputs(prepared, receipt) for receipt in receipts)
    pins = tuple((definition_checksum(regions), _literal(chords)) for regions, chords in operands)
    def check(phase):
        cancelled = callback(phase) if callback is not None else False
        if (_state(source), _state(prepared)) != states or _literal(receipts) != literal or _LINKS.get(prepared) is not links or tuple(
                (definition_checksum(regions), _literal(chords)) for regions, chords in operands) != pins:
            raise GeometryError('source transfer operands changed during callback')
        if cancelled:
            raise GeometryError('source transfer cancelled')
    return check, states


def _qualify_two_support_virtual_trace(source, prepared, edge_id, represented_faces,
                                      selected_faces, occurrence_faces, check,continuation_check=None):
    """Classify one new-family edge using live owner lineage, never its type alone.

    This helper issues no receipt. Its caller supplies faces from independently
    authenticated cell/chord receipts, then retains the normal whole-cell,
    station, XYZ, source-map and final publication checks.
    """
    from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from .nurbs_surface_intersection import SourceDefinition
    from .joint_edges import query_joint_edge
    from .structural import AttachmentKind,AttachmentEvidence,ConnectionIntent,ParameterRange
    check('two-support virtual trace qualification')
    states = (_state(source),_state(prepared))
    links = _LINKS.get(prepared)
    routes = [row for row in links or () if row[0]() is source]
    if len(routes) != 1 or routes[0][1:3] != states:
        raise GeometryError('two-support virtual trace needs current authenticated source lineage')
    edge = prepared.edges.get(edge_id)
    if (edge is None or edge_id in source.edges
            or type(edge.curve) is not NurbsSurfaceIntersectionCurve):
        raise GeometryError('two-support virtual trace needs its native generated owner curve')
    joint = query_joint_edge(prepared,edge_id)
    if not joint.declared:
        raise GeometryError('two-support virtual trace requires a current explicit CONNECT joint declaration')
    joint_sheets = {handle.id for handle in joint.sheets}
    joint_attachments = {identifier for handle in joint.junctions
        for identifier in prepared.junctions[handle.id].attachment_ids
        for item in (prepared.attachments[identifier],)
        if (item.kind is AttachmentKind.SHEET_ON_JOINT and item.source_kind == 'sheet'
            and item.source_id in joint_sheets and item.target_kind == 'edge' and item.target_id == edge_id
            and item.connection_intent is ConnectionIntent.CONNECT and item.evidence is AttachmentEvidence.EXACT
            and item.target_parameters == (ParameterRange(0.,1.),) and item.max_residual <= item.tolerance_used)}
    if prepared.members_using_edge(edge_id) or any(
            ((item.source_kind == 'edge' and item.source_id == edge_id)
             or (item.target_kind == 'edge' and item.target_id == edge_id))
            and item.id not in joint_attachments for item in prepared.attachments.values()):
        raise GeometryError('two-support virtual trace cannot reinterpret member or attachment ownership')
    # Scan actual loops as well as reverse incidence: neither a stale index nor
    # an invented chord declaration can hide an incident material face.
    counts = {face.id:sum(use.edge == edge_id for loop in (face.loop,*face.holes) for use in loop)
              for face in prepared.faces.values()}
    incident = {face for face,count in counts.items() if count}
    if (len(incident) != 4 or any(counts[face] != 1 for face in incident)
            or set(prepared.faces_using_edge(edge_id)) != incident
            or not incident <= set(represented_faces)
            or len(occurrence_faces) != 4 or set(occurrence_faces) != incident):
        raise GeometryError('two-support virtual trace needs all four declared current material uses')
    mapping = routes[0][3]
    parents = {child:parent for parent,children in mapping for child in children}
    if len(parents) != sum(len(children) for _,children in mapping) or not incident <= set(parents):
        raise GeometryError('two-support virtual trace has missing or overlapping parent lineage')
    originals = {parents[child] for child in incident}
    if (len(originals) != 2 or any(sum(parents[child] == parent for child in incident) != 2 for parent in originals)
            or any(not set(children) <= set(selected_faces) for parent,children in mapping if parent in originals)):
        raise GeometryError('two-support virtual trace requires complete descendants of two original parents')
    curve = edge.curve
    expected = (curve.owner.first_source,curve.owner.second_source)
    supports = {}
    for parent in originals:
        support = _effective(source,parent)
        if type(support) is not NurbsSurface:
            raise GeometryError('two-support virtual trace original parent is not native NURBS')
        definition = SourceDefinition.capture(support)
        indices = [i for i,item in enumerate(expected) if item == definition]
        if len(indices) != 1:
            raise GeometryError('two-support virtual trace owner does not match its source parents')
        supports[parent] = (support,indices[0])
    if {index for _,index in supports.values()} != {0,1}:
        raise GeometryError('two-support virtual trace lacks one original support')
    with intersection_curve_operation(continuation_check):
        # Revalidate the canonical branch, issued context/configuration and
        # both oriented pcurves. No samples qualify an interval here.
        curve._shape()
        for parent,(support,_) in supports.items():
            chart = curve.for_support(support)
            chart.chart_box(Q(0),Q(1))
            for child in incident:
                if parents[child] == parent:
                    _chart_map(source,parent,_effective(prepared,child))
        if not curve.enclose(Q(0),Q(1)):
            raise GeometryError('two-support virtual trace has no certified source interval')
        check('two-support virtual trace qualification complete')
        curve._shape()
    if _LINKS.get(prepared) is not links or (_state(source),_state(prepared)) != states:
        raise GeometryError('two-support virtual trace lineage changed before classification')
    return tuple(sorted(originals)),tuple(sorted(incident))


def query_prepared_material_source_transfer(source, prepared, cell_receipts, *,
        expected_source_revision=None, expected_prepared_revision=None, cancellation_check=None):
    receipts = tuple(cell_receipts)
    check, (source_state, prepared_state) = _frontier(source, prepared, receipts, cancellation_check)
    if ((expected_source_revision is not None and expected_source_revision != source_state[0])
            or (expected_prepared_revision is not None and expected_prepared_revision != prepared_state[0])):
        raise GeometryError('source transfer expected revision is stale')
    check('source transfer qualification')
    routes = [row for row in _LINKS.get(prepared, ()) if row[0]() is source]
    if len(routes) != 1 or routes[0][1:3] != (source_state, prepared_state):
        raise GeometryError('no current authenticated immediate source/clone lineage')
    mapping = routes[0][3]
    edge_map = {edge: (parent, a, b) for edge, parent, a, b in routes[0][4]}
    edge_parameters, trace_ownership, stations = {}, {}, {}
    parents = {child: parent for parent, children in mapping for child in children}
    if len(parents) != sum(len(children) for parent, children in mapping):
        raise GeometryError('source lineage contains overlapping ownership')
    # Authenticate complete selected source descendants before inspecting edge
    # ancestry, so a missing child cannot be hidden by an unrelated edge refusal.
    selected_faces = set()
    for owner_receipt in receipts:
        owner_faces = {cell.face.id for cell in owner_receipt.cells}
        if selected_faces & owner_faces:
            raise GeometryError('source transfer has duplicate prepared material coverage')
        if not owner_faces <= set(parents):
            raise GeometryError('source transfer cell has no owner lineage')
        selected_faces.update(owner_faces)
    if not selected_faces or any(selected_faces.intersection(children) and not set(children) <= selected_faces
                                 for parent, children in mapping):
        raise GeometryError('source transfer omits required material descendants')
    represented_edges = {}
    for owner_receipt in receipts:
        _,owner_chords = _source_transfer_cell_inputs(prepared,owner_receipt)
        for cell in owner_receipt.cells:
            for _,index,_ in cell.boundary_intervals:
                edge_id = owner_chords.intervals[index].edge.id
                represented_edges.setdefault(edge_id,set()).add(cell.face.id)
    qualified_two_support = {}
    local_fingerprints=[];local_boundary_errors=[]
    cells, source_cells, references, prepared_refs, maps, errors, bounds = [], [], [], [], [], [], []
    covered, station_nodes, xyz_nodes, operands = set(), {}, {}, []
    for receipt in receipts:
        check('source transfer cell binding')
        regions, chords = _source_transfer_cell_inputs(prepared, receipt)
        local_errors=dict(zip(((cell.face.id,cell.node_ids) for cell in receipt.cells),receipt.surface_errors_by_cell))
        if receipt.local_field_fingerprint:
            if len(local_errors)!=len(receipt.cells):raise GeometryError('source transfer lost local cell allocations')
            local_fingerprints.append(receipt.local_field_fingerprint)
            if len(chords.interval_errors)!=len(chords.intervals):
                raise GeometryError('source transfer lost local boundary interval allocations')
        local_chord_limits={(edge,a,b):limit for edge,a,b,limit in chords.interval_errors}
        operands.append((receipt, regions, chords))
        faces = {cell.face.id for cell in receipt.cells}
        if covered & faces:
            raise GeometryError('source transfer has duplicate prepared material coverage')
        covered.update(faces)
        centres = {(face.id, ids): uv for face, ids, uv in receipt.reference_uv_by_cell}
        cell_bounds = {(face.id, ids): bound for face, ids, bound in receipt.source_map_bounds}
        for cell in receipt.cells:
            if cell.face.id not in parents:
                raise GeometryError('source transfer cell has no owner lineage')
            parent = parents[cell.face.id]
            edge_transport = Q(0)
            transform = _chart_map(source, parent, _effective(prepared, cell.face.id))
            source_face = source.handle('face', parent)
            for node, xyz in zip(cell.node_ids, cell.xyz):
                if node in xyz_nodes and xyz_nodes[node] != xyz:
                    raise GeometryError('source transfer global node has conflicting literal XYZ')
                xyz_nodes[node] = xyz
            for side, index, forward in cell.boundary_intervals:
                chord = chords.intervals[index]
                chord_transport=Q(0)
                edge = prepared.edges[chord.edge.id]
                if edge.id in edge_map:
                    ancestor, first_parameter, last_parameter = edge_map[edge.id]
                    target = source.handle('edge', ancestor)
                    edge_parameters[edge.id] = (chord.edge, target, first_parameter, last_parameter)
                    chord_transport = _edge_transport_error(source, prepared, ancestor,
                        edge.id, first_parameter, last_parameter)
                    edge_transport = max(edge_transport,chord_transport)
                    owners = ()
                else:
                    from .support_curves import PlaneNurbsCurve
                    from .curves import Straight
                    from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
                    # a virtual edge: a section trace / same-support auxiliary partition (PlaneNurbsCurve), or
                    # a STRAIGHT auxiliary partition only when the owner registered it with its complete
                    # definition (a persisted tag is never authority)
                    if type(edge.curve) is NurbsSurfaceIntersectionCurve:
                        declared = tuple(occurrence.face.id for occurrence in chord.occurrences)
                        if edge.id not in qualified_two_support:
                            qualified_two_support[edge.id] = _qualify_two_support_virtual_trace(
                                source,prepared,edge.id,represented_edges.get(edge.id,()),selected_faces,declared,check,
                                (lambda:check('two-support virtual trace continuation'))
                                if cancellation_check is not None else None)
                        original_ids,incident = qualified_two_support[edge.id]
                        if len(declared) != 4 or set(declared) != set(incident):
                            raise GeometryError('two-support virtual trace chord omits a declared material use')
                        owners = tuple(source.handle('face',value) for value in original_ids)
                    elif edge.id in source.edges or not (isinstance(edge.curve, PlaneNurbsCurve) or (
                            isinstance(edge.curve, Straight) and _qualified_auxiliary(prepared, edge.id))):
                        raise GeometryError('source edge split station transport is not qualified')
                    else:
                        owners = tuple(source.handle('face', value) for value in sorted({
                            parents[item.face.id] for item in receipt.cells
                            if any(chords.intervals[i].edge.id == edge.id for _, i, _ in item.boundary_intervals)}))
                    target = None
                    previous = trace_ownership.get(edge.id, (chord.edge, ()))
                    trace_ownership[edge.id] = (chord.edge, tuple(sorted(set(previous[1]+owners), key=lambda h: h.id)))
                if receipt.local_field_fingerprint:
                    limit=local_chord_limits[(edge.id,*chord.parameter_range)]
                    if Q(chord.world_error)+chord_transport>limit:
                        raise GeometryError('source boundary transport exceeds local interval allowance')
                    local_boundary_errors.append((chord.edge,chord.parameter_range,target,limit,chord_transport))
                parameters = chord.parameter_range if forward else chord.parameter_range[::-1]
                for offset, t in enumerate(parameters):
                    key = ('vertex', edge.start if t == 0 else edge.end) if t in (0, 1) else ('edge', edge.id, t)
                    node = cell.node_ids[(side+offset)%len(cell.node_ids)]
                    if key in station_nodes and station_nodes[key] != node:
                        raise GeometryError('source transfer shared station has different global node IDs')
                    station_nodes[key] = node
                    station = (chord.edge, t, target, first_parameter+(last_parameter-first_parameter)*t if target is not None else None, owners)
                    prior = stations.get((edge.id, t))
                    if prior is not None:
                        if prior[:4] != station[:4]:
                            raise GeometryError('source transfer station ancestry conflicts')
                        station = (*station[:4], tuple(sorted(set(prior[4]+owners), key=lambda h: h.id)))
                    stations[(edge.id, t)] = station
            centre = centres[(cell.face.id, cell.node_ids)]
            cells.append(cell)
            source_cells.append(CurvilinearMaterialCell(source_face, cell.node_ids,
                tuple(_map_uv(transform, uv) for uv in cell.uv), cell.xyz, ()))
            references.append((source_face, cell.node_ids, _map_uv(transform, centre)))
            prepared_refs.append((cell.face, cell.node_ids, centre))
            maps.append((source_face, cell.face, transform))
            from .nurbs import _float_down
            local_error=_float_down(local_errors[(cell.face.id,cell.node_ids)]) if receipt.local_field_fingerprint else receipt.surface_error
            errors.append((source_face, cell.node_ids, local_error))
            from .nurbs import _float_up
            combined = Q(cell_bounds[(cell.face.id, cell.node_ids)])+edge_transport
            if combined > Q(local_error):
                raise GeometryError('source edge transport exceeds supplied surface approximation error')
            bounds.append((source_face, cell.node_ids, _float_up(combined)))
    if not covered or any(covered.intersection(children) and not set(children) <= covered for parent, children in mapping):
        raise GeometryError('source transfer omits required material descendants')
    selected = tuple((source.handle('face', parent), tuple(prepared.handle('face', child) for child in children))
                     for parent, children in mapping if covered.intersection(children))
    check('source transfer qualification complete')
    # All user callbacks have completed. Revalidate owner operands before issue.
    for receipt, regions, chords in operands:
        validate_prepared_material_curvilinear_cells_binding(prepared, regions, receipt)
    if (_state(source), _state(prepared)) != (source_state, prepared_state):
        raise GeometryError('source transfer definitions changed before issue')
    result = PreparedMaterialSourceTransfer(source.model_id, prepared.model_id, source_state[0], prepared_state[0],
        source_state[1], prepared_state[1], selected, _detached(tuple(source_cells)), _detached(tuple(cells)),
        tuple(references), tuple(prepared_refs), tuple(maps), tuple(errors), tuple(bounds),
        tuple(edge_parameters[key] for key in sorted(edge_parameters)),
        tuple(trace_ownership[key] for key in sorted(trace_ownership)),
        tuple(stations[key] for key in sorted(stations)),tuple(local_fingerprints),tuple(local_boundary_errors))
    _TRANSFERS[result] = (ref(source), ref(prepared), receipts, _literal(receipts), tuple(operands), _signature(result), _LINKS.get(prepared), (source_state, prepared_state))
    return result


def validate_prepared_material_source_transfer_binding(source, prepared, receipt, *, cell_receipts, cancellation_check=None):
    if type(receipt) is not PreparedMaterialSourceTransfer or receipt not in _TRANSFERS:
        raise GeometryError('source transfer receipt was not issued by this owner')
    first, second, issued, literals, operands, signature, links, issued_states = _TRANSFERS[receipt]
    supplied = tuple(cell_receipts)
    if first() is not source or second() is not prepared or supplied != issued or any(a is not b for a, b in zip(supplied, issued)):
        raise GeometryError('source transfer belongs to other model or receipt objects')
    check, states = _frontier(source, prepared, supplied, cancellation_check)
    def bound():
        if (states != issued_states or _LINKS.get(prepared) is not links or _signature(receipt) != signature or _literal(supplied) != literals
                or states[0][:2] != (receipt.source_revision, receipt.source_checksum)
                or states[1][:2] != (receipt.prepared_revision, receipt.prepared_checksum)):
            raise GeometryError('source transfer receipt is stale or changed')
    bound()
    check('source transfer binding')
    bound()
    for cell, regions, chords in operands:
        validate_prepared_material_curvilinear_cells_binding(prepared, regions, cell)
    check('source transfer binding complete')
    bound()


def _source_station_transport_inputs(source, prepared, transfer):
    """Live owner operands for the nominal-alias certifier; never reconstructed tokens."""
    if type(transfer) is not PreparedMaterialSourceTransfer or transfer not in _TRANSFERS:
        raise GeometryError('source station transport requires a live owner source transfer')
    row = _TRANSFERS[transfer]
    receipts = row[2]
    validate_prepared_material_source_transfer_binding(source, prepared, transfer, cell_receipts=receipts)
    return receipts, row[4]


def _has_current_prepared_partition_lineage(model):
    """Select optional proof reuse without refreshing or discarding lineage."""
    current = _state(model)
    links = _LINKS.get(model)
    observed = []
    obsolete = False
    for owner, original, bound, mapping, edge_map in links or ():
        source = owner()
        if source is None:
            continue
        children = [child for _, group in mapping for child in group]
        if (type(mapping) is not tuple or len(children) != len(set(children))
                or len({parent for parent, _ in mapping}) != len(mapping)
                or any(type(parent) is not int or parent <= 0 or type(group) is not tuple or not group
                       or any(type(child) is not int or child <= 0 for child in group)
                       for parent, group in mapping)):
            raise GeometryError('prepared partition complete lineage is malformed')
        source_state = _state(source)
        observed.append((source, source_state))
        for operand, actual, issued in ((model, current, bound), (source, source_state, original)):
            if actual == issued:
                continue
            # A normal committed edit makes an optional proof unavailable.
            # A raw checksum/native change at the issued revision, or an
            # invented revision without an owner ChangeSet, is still refusal.
            change = operand.last_change_set
            if (type(actual[0]) is not int or type(issued[0]) is not int
                    or actual[0] <= issued[0] or change.revision_after != actual[0]
                    or not issued[0] <= change.revision_before < change.revision_after):
                raise GeometryError('prepared partition source lineage is stale or changed')
            committed, failure = _PARTITION_COMMITS.get(operand, (None, 'no owner commit witness'))
            if failure is not None or committed != actual:
                raise GeometryError('prepared partition current content has no unchanged owner commit witness'
                                    + (f': {failure}' if failure else ''))
            obsolete = True
    if (_LINKS.get(model) is not links or _state(model) != current
            or any(_state(source) != state for source, state in observed)):
        raise GeometryError('prepared partition lineage changed during readiness query')
    if obsolete:
        return False
    # Reuse the strict complete-lineage validator; readiness is not a second
    # way to infer partition authority from topology or historical metadata.
    fresh, live, routes = _prepared_partition_context(model)
    if fresh != current or live is not links:
        raise GeometryError('prepared partition lineage changed during readiness query')
    return bool(routes)


def _prepared_partition_context(model):
    """Fresh live same-parent partition context; no topology/history inference."""
    current = _state(model)
    links = _LINKS.get(model)
    routes = []
    for owner,original,bound,mapping,edge_map in links or ():
        source = owner()
        if source is None:
            continue
        if bound != current or _state(source) != original:
            raise GeometryError('prepared partition source lineage is stale or changed')
        children = [child for _,group in mapping for child in group]
        if (len(children) != len(set(children)) or set(children) != set(model.faces)
                or len({parent for parent,_ in mapping}) != len(mapping)
                or any(parent not in source.faces or not group or any(child not in model.faces for child in group)
                       for parent,group in mapping)):
            raise GeometryError('prepared partition complete lineage is malformed')
        routes.append((source,original,mapping))
    return current,links,tuple(routes)
