"""Owner-bound pre-mesh material footprints; no cell or clearance authority.

Only owner-qualified effective native charts are eligible. Domains retain
exact native boundary paths (including holes), not fitted UV polygons. With the
exact source-from-child map these represent authored-chart material footprints.
Receipts are local to their issuing objects and cannot be copied or serialized
into authority. Validate again at every consequential use of a receipt.
"""
from dataclasses import dataclass
from weakref import WeakKeyDictionary, ref

from .definition_binding import definition_checksum
from .errors import GeometryError
from .cancellation import is_cancellation_signal
from .material_regions import (MaterialSurfaceRegions,
    validate_material_surface_regions_binding)
from .prepared_material_source_transfer import _prepared_partition_context, _state, _effective
from ._local_adaptation_lineage import (query_local_adaptation_lineage,
    validate_local_adaptation_lineage)

_ISSUED = WeakKeyDictionary()


@dataclass(frozen=True)
class LocalFieldFootprint:
    prepared_face: object
    source_face: object
    child_domain: object
    source_from_child: tuple
    source_coordinate_system: str
    region_index: int
    face_uses: tuple


@dataclass(frozen=True, eq=False)
class LocalFieldTransport:
    source_model_id: object
    prepared_model_id: object
    source_revision: int
    prepared_revision: int
    source_checksum: str
    prepared_checksum: str
    footprints: tuple[LocalFieldFootprint, ...]
    source_to_prepared_faces: tuple
    regions: MaterialSurfaceRegions


def _partition(source, prepared):
    if source is prepared:
        state = _state(source)
        return state, None, tuple((f, (f,)) for f in sorted(source.faces))
    current, links, routes = _prepared_partition_context(prepared)
    matches = [mapping for owner, original, mapping in routes if owner is source]
    if len(matches) != 1:
        raise GeometryError('local field transport needs one current owner source partition')
    return current, links, matches[0]


def _selection(source, prepared, regions, mapping):
    charts = tuple(chart for region in regions.regions for chart in region.sources)
    ids = tuple(chart.face.id for chart in charts)
    if not ids or len(set(ids)) != len(ids):
        raise GeometryError('local field transport requires unique nonempty child coverage')
    if any(c.face.model_id != prepared.model_id or c.face.kind != 'face' for c in charts):
        raise GeometryError('local field footprint belongs to another model')
    selected = set(ids)
    parents = tuple((parent, children) for parent, children in mapping if selected.intersection(children))
    required = {child for _, children in parents for child in children}
    if selected != required:
        raise GeometryError('local field transport omits required material descendants')
    if any(parent not in source.faces for parent, _ in parents):
        raise GeometryError('local field transport source parent is missing')
    return tuple(sorted(ids)), parents


def _guard(source, prepared, regions, lineage, initial, receipt=None):
    a, b, links, mapping, region_signature, lineage_signature = initial
    if _state(source) != a or _state(prepared) != b:
        raise GeometryError('local field transport source or prepared definition changed')
    _, current_links, current_mapping = _partition(source, prepared)
    if current_links is not links or current_mapping != mapping:
        raise GeometryError('local field transport owner partition changed')
    if definition_checksum(regions) != region_signature:
        raise GeometryError('local field transport regions changed')
    if definition_checksum(lineage) != lineage_signature:
        raise GeometryError('local field transport lineage changed')
    if receipt is not None and definition_checksum(receipt[0]) != receipt[1]:
        raise GeometryError('local field transport receipt changed')
    validate_local_adaptation_lineage(source, prepared, lineage)


def _frontier(guard, callback):
    def check(phase):
        guard()
        cancelled = callback(phase) if callback is not None else False
        guard()
        if cancelled:
            raise GeometryError('local field transport cancelled')
        return False
    return check


def _inner_callback(callback, guarded):
    # Preserve the owner's no-caller-code paths. Arbitrary callables still owe
    # the complete transport frontier before and after every invocation.
    return callback if callback is None or is_cancellation_signal(callback) else guarded


def query_local_field_transport(source, prepared, regions, *, cancellation_check=None):
    """Issue complete selected-parent material footprints before cell generation.

    Selection is exactly the supplied owner's regions. Selecting one descendant
    requires every current descendant of its authored parent. The region object
    retains cancelled seams, constraints and structural uses; footprint domains
    retain each child boundary independently, including any shared seam.
    """
    if type(regions) is not MaterialSurfaceRegions:
        raise GeometryError('local field transport requires owner material regions')
    a = _state(source)
    b, links, mapping = _partition(source, prepared)
    region_signature = definition_checksum(regions)
    faces, parents = _selection(source, prepared, regions, mapping)
    # This primitive supplies the established owner-qualified exact chart map.
    # No caller callback occurs before all query operands have been pinned.
    lineage = query_local_adaptation_lineage(source, prepared, faces)
    initial = (a, b, links, mapping, region_signature, definition_checksum(lineage))
    guard = lambda: _guard(source, prepared, regions, lineage, initial)
    check = _frontier(guard, cancellation_check)
    check('local field transport qualification')
    validate_material_surface_regions_binding(prepared, regions,
        cancellation_check=_inner_callback(cancellation_check, check))
    rows = {face: (parent, matrix) for face, parent, matrix in lineage.face_maps}
    footprints = []
    for index, region in enumerate(regions.regions):
        for chart in region.sources:
            parent, matrix = rows[chart.face.id]
            # Both chart supports must be literally the authenticated native chart.
            native = _effective(source, parent)
            if definition_checksum(chart.support) != definition_checksum(native):
                raise GeometryError('local field footprint changed its native chart')
            footprints.append(LocalFieldFootprint(chart.face, source.handle('face', parent),
                chart.domain, matrix, 'authored_query_uv', index, chart.face_uses))
    result = LocalFieldTransport(source.model_id, prepared.model_id, a[0], b[0], a[1], b[1],
        tuple(footprints), tuple((source.handle('face', p),
            tuple(prepared.handle('face', c) for c in children)) for p, children in parents), regions)
    signature = definition_checksum(result)
    final_guard = lambda: _guard(source, prepared, regions, lineage, initial, (result, signature))
    _frontier(final_guard, cancellation_check)('local field transport qualification complete')
    final_guard()  # No user code after the final binding gate.
    _ISSUED[result] = (ref(source), ref(prepared), lineage, initial, signature)
    return result


def validate_local_field_transport(source, prepared, result, *, cancellation_check=None):
    """Validate an issued live footprint receipt, including callback mutations."""
    stored = _ISSUED.get(result) if type(result) is LocalFieldTransport else None
    if stored is None or stored[0]() is not source or stored[1]() is not prepared:
        raise GeometryError('local field transport receipt was not issued for these models')
    _, _, lineage, initial, signature = stored
    guard = lambda: _guard(source, prepared, result.regions, lineage, initial, (result, signature))
    check = _frontier(guard, cancellation_check)
    check('local field transport binding')
    validate_material_surface_regions_binding(prepared, result.regions,
        cancellation_check=_inner_callback(cancellation_check, check))
    check('local field transport binding complete')
    guard()

