"""Owner qualification for two connected, nonperiodic cylinder sectors.

Each face retains the independent cylinder-patch proof.  This contract adds
the source-topology and angular-union proof needed to consume the faces as one
open component; it does not relax the full-period cylinder atlas contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import json
import math

import numpy as np

from .cylinder_patch import (
    CylinderPatchOccurrenceRequest,
    CylinderPatchResult,
    CylinderPatchSample,
    CylinderPatchStatus,
    evaluate_cylinder_patch_occurrences,
    query_cylinder_patch,
    validate_cylinder_patch_binding,
)
from .errors import GeometryError
from .identity import EntityHandle
from .model import GeometryModel


class CylinderOpenComponentErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    WRONG_MODEL = "WRONG_MODEL"
    STALE_RESULT = "STALE_RESULT"
    INVALID_RESULT = "INVALID_RESULT"
    UNQUALIFIED_RESULT = "UNQUALIFIED_RESULT"


class CylinderOpenComponentError(GeometryError):
    def __init__(self, code: CylinderOpenComponentErrorCode, message: str):
        self.code = CylinderOpenComponentErrorCode(code)
        super().__init__(f"{self.code.value}: {message}")


@dataclass(frozen=True, slots=True)
class CylinderOpenComponentResult:
    contract_version: int
    model_id: object
    revision: int
    requested_face_uses: tuple[EntityHandle, EntityHandle]
    status: CylinderPatchStatus
    patches: tuple[CylinderPatchResult, ...]
    shared_edge: EntityHandle | None
    interface_coedges: tuple[EntityHandle, ...]
    exterior_cycle: tuple[tuple[EntityHandle, int], ...]
    external_coedges: tuple[EntityHandle, ...]
    diagnostics: tuple[str, ...]
    digest: str

    @property
    def complete(self) -> bool:
        return self.status is CylinderPatchStatus.QUALIFIED


@dataclass(frozen=True, slots=True)
class CylinderOpenComponentEvaluation:
    model_id: object
    revision: int
    component_digest: str
    samples: tuple[CylinderPatchSample, ...]


def _selection(model, face_uses, expected_revision, cancellation_check):
    if not isinstance(model, GeometryModel) or type(expected_revision) is not int:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "model and integer revision required")
    if cancellation_check is not None and not callable(cancellation_check):
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "cancellation callback must be callable")
    if isinstance(face_uses, (str, bytes)):
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "two ordered FaceUses required")
    try:
        selected = tuple(face_uses)
    except TypeError as error:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "two ordered FaceUses required") from error
    if (len(selected) != 2 or any(type(use) is not EntityHandle or use.kind != "face_use" for use in selected)
            or selected[0] == selected[1]):
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "two distinct FaceUses required")
    if any(use.model_id != model.model_id for use in selected):
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.WRONG_MODEL, "FaceUse belongs to another model")
    if model.revision != expected_revision:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.STALE_RESULT, "model revision changed")
    if any(use.id not in model.face_uses for use in selected):
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "inactive FaceUse")
    if cancellation_check is not None:
        cancellation_check("open cylinder component selection")
    return selected


def _digest(result):
    def handle(value):
        return None if value is None else (value.model_id.hex, value.kind, value.id)
    payload = (
        result.contract_version, result.model_id.hex, result.revision,
        tuple(handle(use) for use in result.requested_face_uses), result.status.value,
        tuple(patch.digest for patch in result.patches), handle(result.shared_edge),
        tuple(handle(use) for use in result.interface_coedges),
        tuple((handle(use), direction) for use, direction in result.exterior_cycle),
        tuple(handle(use) for use in result.external_coedges), result.diagnostics,
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _component(model, selected, patches, cancellation_check):
    """Conservatively prove one open two-sector union from source identities."""
    if any(not patch.certificate.complete or patch.status is not CylinderPatchStatus.QUALIFIED
           or len(patch.loops) != 1 for patch in patches):
        return None, "sector_patch_unqualified"
    uses = [model.face_uses[use.id] for use in selected]
    if uses[0].sheet_id != uses[1].sheet_id:
        return None, "sectors_do_not_share_sheet"
    first, second = patches
    if not all(np.array_equal(np.asarray(getattr(first, name)), np.asarray(getattr(second, name)))
               for name in ("source_origin", "axis", "radial_direction")) or (
               first.radius != second.radius or first.height != second.height):
        return None, "sector_frames_differ"
    if first.sweep_sign != second.sweep_sign:
        return None, "sector_sweep_directions_differ"
    angular_tolerance = max(first.certificate.tolerance_angular, second.certificate.tolerance_angular)
    if abs(first.sweep_angle) + abs(second.sweep_angle) >= 2 * math.pi - angular_tolerance:
        return None, "component_is_not_strictly_open"
    endpoints = ((first.start_angle, first.start_angle + first.sweep_angle),
                 (second.start_angle, second.start_angle + second.sweep_angle))
    touching = (abs(math.remainder(endpoints[0][1] - endpoints[1][0], 2 * math.pi)) <= angular_tolerance
                or abs(math.remainder(endpoints[1][1] - endpoints[0][0], 2 * math.pi)) <= angular_tolerance)
    if not touching:
        return None, "sector_angular_union_is_not_contiguous"
    by_edge = [{occ.edge.id: occ for occ in patch.occurrences} for patch in patches]
    shared = set(by_edge[0]) & set(by_edge[1])
    if len(shared) != 1:
        return None, "one_shared_generator_required"
    edge_id = next(iter(shared))
    a, b = by_edge[0][edge_id], by_edge[1][edge_id]
    if (a.carrier != "AXIAL" or b.carrier != "AXIAL"
            or a.coedge_forward == b.coedge_forward
            or set(model.face_uses_using_edge(edge_id)) != {use.id for use in selected}):
        return None, "shared_generator_incidence_unqualified"
    if (b.coedge not in a.external_coedges or a.coedge not in b.external_coedges):
        return None, "shared_generator_occurrences_unqualified"
    exterior = [occ for patch in patches for occ in patch.occurrences if occ.edge.id != edge_id]
    if len(exterior) != 6 or len({occ.edge.id for occ in exterior}) != 6:
        return None, "open_exterior_requires_six_source_edges"
    outgoing = {}
    for occ in exterior:
        forward = occ.coedge_forward == (occ.face_use_orientation == 1)
        start = occ.start_vertex.id if forward else occ.end_vertex.id
        end = occ.end_vertex.id if forward else occ.start_vertex.id
        if start in outgoing:
            return None, "open_exterior_is_not_simple"
        outgoing[start] = (occ, 1 if forward else -1, end)
    initial = min(outgoing)
    vertex = initial
    visited = set()
    cycle = []
    while vertex not in visited:
        if cancellation_check is not None:
            cancellation_check("open cylinder exterior proof")
        if vertex not in outgoing:
            return None, "open_exterior_is_not_closed"
        visited.add(vertex)
        occ, direction, vertex = outgoing[vertex]
        cycle.append((occ.coedge, direction))
    if vertex != initial or len(cycle) != len(exterior):
        return None, "open_exterior_is_not_one_cycle"
    external = tuple(sorted((use for occ in exterior for use in occ.external_coedges), key=lambda value: value.id))
    return (model.handle("edge", edge_id), (a.coedge, b.coedge), tuple(cycle), external), None


def query_cylinder_open_component(model, face_uses, *, expected_revision, cancellation_check=None):
    """Qualify exactly two connected nonperiodic sectors without source edits."""
    selected = _selection(model, face_uses, expected_revision, cancellation_check)
    patches = tuple(query_cylinder_patch(model, (use,), expected_revision=expected_revision,
                                          cancellation_check=cancellation_check) for use in selected)
    if any(patch.status is not CylinderPatchStatus.QUALIFIED for patch in patches):
        status = (CylinderPatchStatus.CAPABILITY_MISSING if any(patch.status is CylinderPatchStatus.CAPABILITY_MISSING
                  for patch in patches) else CylinderPatchStatus.UNRESOLVED)
        reason = tuple(f"face_use/{use.id}: {patch.diagnostics}" for use, patch in zip(selected, patches)
                       if patch.status is not CylinderPatchStatus.QUALIFIED)
        components = None
    else:
        components, problem = _component(model, selected, patches, cancellation_check)
        status = CylinderPatchStatus.QUALIFIED if components is not None else CylinderPatchStatus.CAPABILITY_MISSING
        reason = () if components is not None else (problem,)
    if components is None:
        result = CylinderOpenComponentResult(1, model.model_id, expected_revision, selected, status,
                                             (), None, (), (), (), reason, "")
    else:
        edge, interface, cycle, external = components
        result = CylinderOpenComponentResult(1, model.model_id, expected_revision, selected, status,
                                             patches, edge, interface, cycle, external, (), "")
    result = replace(result, digest=_digest(result))
    if cancellation_check is not None:
        cancellation_check("open cylinder component result")
    if model.revision != expected_revision:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.STALE_RESULT, "model revision changed")
    return result


def validate_cylinder_open_component_binding(model, result, face_uses, *, expected_revision, cancellation_check=None):
    selected = _selection(model, face_uses, expected_revision, cancellation_check)
    if type(result) is not CylinderOpenComponentResult or result.requested_face_uses != selected:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_RESULT, "component selection differs")
    if result.model_id != model.model_id or result.revision != expected_revision:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.STALE_RESULT, "component source differs")
    if result.contract_version != 1 or _digest(result) != result.digest:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_RESULT, "component digest differs")
    fresh = query_cylinder_open_component(model, selected, expected_revision=expected_revision,
                                          cancellation_check=cancellation_check)
    if result != fresh:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_RESULT, "component differs from fresh qualification")
    for patch, use in zip(result.patches, selected):
        validate_cylinder_patch_binding(model, patch, (use,), expected_revision=expected_revision,
                                        cancellation_check=cancellation_check)


def evaluate_cylinder_open_component_occurrences(model, result, requests, *, face_uses,
                                                 expected_revision, cancellation_check=None):
    validate_cylinder_open_component_binding(model, result, face_uses,
                                             expected_revision=expected_revision,
                                             cancellation_check=cancellation_check)
    if not result.complete:
        raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.UNQUALIFIED_RESULT, "open component is not qualified")
    ownership = {occ.coedge.id: (patch, use) for patch, use in zip(result.patches, result.requested_face_uses)
                 for occ in patch.occurrences}
    samples = []
    for request in requests:
        if type(request) is not CylinderPatchOccurrenceRequest or request.coedge.id not in ownership:
            raise CylinderOpenComponentError(CylinderOpenComponentErrorCode.INVALID_REQUEST, "occurrence is not selected")
        patch, use = ownership[request.coedge.id]
        sample = evaluate_cylinder_patch_occurrences(model, patch, (request,), face_uses=(use,),
            expected_revision=expected_revision, cancellation_check=cancellation_check).samples[0]
        identity = (model.model_id.hex, expected_revision, sample.edge.id, request.parameter.hex())
        key = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        samples.append(replace(sample, identity_key=key))
    return CylinderOpenComponentEvaluation(model.model_id, expected_revision, result.digest, tuple(samples))
