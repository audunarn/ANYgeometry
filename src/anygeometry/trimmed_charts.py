"""Geometry-owner charts for analytic Plane and Cylinder material domains.

Trim loops are exact curve definitions. Evaluation samples describe the support;
material membership is a separate analytic trim predicate, never a sampled
polygon certificate. A chart collection binds the complete source document.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
from weakref import WeakKeyDictionary
import numpy as np

from .arrangement_geometry import LinePath, point_parameters
from .errors import GeometryError
from .identity import EntityHandle
from .material_arrangement import MaterialDomain, arrange_material
from .member_arrangements import _support_roots
from .serialization import to_dict, _serialized_model_state
from .surfaces import Plane, Cylinder
from .definition_binding import definition_checksum

# Reuse completed qualification only while both immutable evidence and the
# complete live document retain their content bindings. The cache is external
# to model state. Retain a bounded set of recent collections so alternating
# component/face queries reuse qualification. This bounds cache memory only;
# evicted collections are fully validated, never refused. Changed evidence and
# direct document edits still require full validation.
_validated_collections = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class TrimmedSurfaceChart:
    face: EntityHandle
    face_uses: tuple[EntityHandle, ...]
    domain: MaterialDomain
    world_tolerance: float
    material_area: float

    @property
    def support(self):
        return self.domain.support

    @property
    def boundaries(self):
        return self.domain.boundaries

    @property
    def face_use(self):
        if len(self.face_uses) != 1:
            raise GeometryError("chart has no unique FaceUse; select an explicit structural occurrence")
        return self.face_uses[0]


@dataclass(frozen=True, slots=True)
class TrimmedSurfaceCharts:
    model_id: UUID
    revision: int
    source_checksum: str
    charts: tuple[TrimmedSurfaceChart, ...]


def _check(callback, phase):
    if callback is not None and callback(phase):
        raise GeometryError("trimmed surface chart query cancelled")


def _handle(model, handle):
    if not isinstance(handle,EntityHandle):
        raise GeometryError("trimmed charts require model-bound handles")
    if handle.model_id != model.model_id:
        raise GeometryError("trimmed chart handle belongs to another model")
    model.handle(handle.kind,handle.id)


def query_trimmed_surface_charts(model, operands=None, *, expected_revision=None,
                                cancellation_check=None):
    """Return all selected material charts, including every hole and trim.

    Faces, FaceUses and Sheets are accepted; omitted operands select all faces.
    Plane charts use their stored affine basis. Cylinder charts use the native
    angular/axial basis, with the original face boundaries retaining seam sides.
    There is no face, hole or intersection count limit.
    """
    revision = model.revision
    if expected_revision is not None and expected_revision != revision:
        raise GeometryError("trimmed surface chart query revision is stale")
    checksum = to_dict(model)["checksum"]["value"]
    faces = set()
    for operand in (tuple(model.faces) if operands is None else operands):
        handle = model.handle("face", operand) if isinstance(operand, int) else operand
        _handle(model,handle)
        if handle.kind == "face":
            faces.add(handle.id)
        elif handle.kind == "face_use":
            faces.add(model.face_uses[handle.id].face_id)
        elif handle.kind == "sheet":
            faces.update(model.face_uses[identifier].face_id
                         for identifier in model.sheets[handle.id].face_use_ids)
        else:
            raise GeometryError("trimmed charts require faces, FaceUses or Sheets")
    charts = []
    for face_id in sorted(faces):
        _check(cancellation_check, "trimmed surface chart qualification")
        domain = MaterialDomain.from_model(model, face_id)
        boxes = [path.curve.bounds() for loop in domain.boundaries for path in loop]
        length = float(np.linalg.norm(np.max([box[1] for box in boxes],axis=0)
                                      - np.min([box[0] for box in boxes],axis=0)))
        tolerance = model.tolerance.effective_length(length)
        for loop in domain.boundaries:
            for path in loop:
                roots = _support_roots(path.curve, domain.support, tolerance,
                    lambda: _check(cancellation_check, "trim support predicate"))
                if roots is not None:
                    raise GeometryError(f"face {face_id} trim is not contained in its analytic support")
        arrangement = arrange_material(domain, (), tolerance=tolerance,
            area_tolerance=model.tolerance.effective_area(length),
            cancellation_check=lambda: (_check(cancellation_check, "trim arrangement") or False))
        charts.append(TrimmedSurfaceChart(model.handle("face",face_id), tuple(
            model.handle("face_use",identifier) for identifier,use in sorted(model.face_uses.items())
            if use.face_id == face_id), domain, tolerance, domain.material_world_area(arrangement)))
    if model.revision != revision or to_dict(model)["checksum"]["value"] != checksum:
        raise GeometryError("geometry changed during trimmed chart query")
    return TrimmedSurfaceCharts(model.model_id,revision,checksum,tuple(charts))


def validate_trimmed_surface_charts_binding(model, result, *, expected_revision=None,
                                            cancellation_check=None):
    if not isinstance(result, TrimmedSurfaceCharts):
        raise TypeError("trimmed chart binding needs a TrimmedSurfaceCharts result")
    _check(cancellation_check, "trimmed surface chart binding")
    if result.model_id != model.model_id:
        raise GeometryError("trimmed chart binding belongs to another model")
    if result.revision != model.revision or (expected_revision is not None
                                            and expected_revision != model.revision):
        raise GeometryError("trimmed chart binding is stale")
    if result.source_checksum != _serialized_model_state(model)["checksum"]["value"]:
        raise GeometryError("trimmed chart source binding changed")
    signature=(result.source_checksum,definition_checksum(result))
    if signature in _validated_collections.get(model,()):
        return
    # A changed evidence collection must receive full source qualification.
    to_dict(model)
    for chart in result.charts:
        _handle(model,chart.face)
        if chart.face.kind != 'face':
            raise GeometryError("trimmed chart binding requires a face")
        live = MaterialDomain.from_model(model,chart.face.id)
        if definition_checksum(live) != definition_checksum(chart.domain):
            raise GeometryError("trimmed chart analytic definition binding changed")
        uses = tuple(model.handle('face_use',identifier) for identifier,use in sorted(model.face_uses.items())
                     if use.face_id==chart.face.id)
        if chart.face_uses != uses:
            raise GeometryError("trimmed chart occurrence binding changed")
        boxes=[path.curve.bounds() for loop in live.boundaries for path in loop]
        length=float(np.linalg.norm(np.max([box[1] for box in boxes],axis=0)
                                    -np.min([box[0] for box in boxes],axis=0)))
        if chart.world_tolerance != model.tolerance.effective_length(length):
            raise GeometryError("trimmed chart tolerance binding changed")
        area_tolerance=model.tolerance.effective_area(length)
        native_tolerance=area_tolerance/live.area_jacobian
        area=live.original_world_area(native_tolerance*.05*live.area_jacobian)
        if not np.isfinite(chart.material_area) or abs(area-chart.material_area)>area_tolerance:
            raise GeometryError("trimmed chart material area binding changed")
        for use in chart.face_uses:
            _handle(model,use)
            if model.face_uses[use.id].face_id != chart.face.id:
                raise GeometryError("trimmed chart FaceUse binding changed")
    _check(cancellation_check,"trimmed surface chart validation complete")
    if model.revision != result.revision or _serialized_model_state(model)["checksum"]["value"] != result.source_checksum:
        raise GeometryError("trimmed chart source binding changed during validation")
    previous=tuple(item for item in _validated_collections.get(model,())
                   if item[0]==result.source_checksum and item!=signature)
    _validated_collections[model]=(*previous[-7:],signature)


def _rows(parameters):
    try:
        values=np.asarray(parameters,dtype=float)
    except (TypeError,ValueError) as error:
        raise GeometryError("trimmed chart parameters must be finite (..., 2) arrays") from error
    if values.ndim < 1 or values.shape[-1] != 2 or not np.all(np.isfinite(values)):
        raise GeometryError("trimmed chart parameters must be finite (..., 2) arrays")
    return values, values.reshape(-1,2)


def evaluate_trimmed_surface_chart(model,result,face,parameters,*,require_material=False,
                                   cancellation_check=None):
    """Evaluate a bound support in batches, optionally rejecting void samples.

    With require_material=False this also evaluates the support extension used
    by metric and projection algorithms. It does not assert material coverage.
    """
    validate_trimmed_surface_charts_binding(model,result,cancellation_check=cancellation_check)
    identifier=face if isinstance(face,int) else face.id
    if not isinstance(face,int):
        _handle(model,face)
        if face.kind != "face":
            raise GeometryError("trimmed chart evaluation requires a face")
    chart=next((chart for chart in result.charts if chart.face.id==identifier),None)
    if chart is None:
        raise GeometryError("face is not selected by the trimmed chart binding")
    values,rows=_rows(parameters)
    positions=np.asarray([chart.support.evaluate(*uv) for uv in rows],dtype=float).reshape((-1,3))
    if require_material:
        for point in positions:
            boundary=any(point_parameters(path.curve,point,tolerance=chart.world_tolerance)
                         for loop in chart.boundaries for path in loop)
            if not boundary and not chart.domain.contains(LinePath(tuple(point),tuple(point)),0.,chart.world_tolerance):
                raise GeometryError("trimmed chart sample is outside material")
    if not np.all(np.isfinite(positions)):
        raise GeometryError("trimmed chart evaluation produced nonfinite results")
    validate_trimmed_surface_charts_binding(model,result,cancellation_check=cancellation_check)
    return positions.reshape((*values.shape[:-1],3)).copy()
