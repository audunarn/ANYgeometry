"""Source-bound trim enclosure reference lengths; no approximation admission."""
from dataclasses import dataclass
import math
import numpy as np
from .errors import GeometryError
from .tolerance import feature_extent
from .trimmed_charts import (query_trimmed_surface_charts,
    validate_trimmed_surface_charts_binding, _query_start, _query_end)
from .nurbs import CurveOnSurface

@dataclass(frozen=True, slots=True)
class ApproximationSourceBinding:
    model_id: object
    revision: int
    source_checksum: str

def query_approximation_source_binding(model, *, cancellation_check=None):
    revision, checksum = _query_start(model, None, cancellation_check)
    _query_end(model, revision, checksum, cancellation_check)
    return ApproximationSourceBinding(model.model_id, revision, checksum)

def validate_approximation_source_binding(model, binding, *, cancellation_check=None):
    if type(binding) is not ApproximationSourceBinding or binding.model_id != model.model_id:
        raise GeometryError('approximation source binding belongs to another model')
    _query_end(model, binding.revision, binding.source_checksum, cancellation_check)

@dataclass(frozen=True, slots=True)
class TrimEnclosureScale:
    source: ApproximationSourceBinding
    face_ids: tuple
    loop_spans: tuple
    reference_scale: float
    policy_floor: float
    charts: object
    scale_definition: str = 'minimum_trim_enclosure_span'

def _values(model, charts, callback):
    spans = []
    for chart in charts.charts:
        if not math.isfinite(chart.material_area) or chart.material_area <= 0:
            raise GeometryError('automatic approximation requires qualified positive material')
        for loop in chart.boundaries:
            if callback is not None and callback('approximation trim enclosure'):
                raise GeometryError('automatic approximation query cancelled')
            boxes = []
            for path in loop:
                curve = path.curve
                box = curve.bounds(fresh=True) if type(curve) is CurveOnSurface else curve.bounds()
                low, high = (np.asarray(part, dtype=float) for part in box)
                if (low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite((low, high)))
                        or np.any(low > high)):
                    raise GeometryError('automatic approximation requires finite ordered trim enclosures')
                boxes.append((low, high))
            if not boxes:
                raise GeometryError('automatic approximation requires nonempty qualified trim loops')
            low = np.min([b[0] for b in boxes], axis=0)
            high = np.max([b[1] for b in boxes], axis=0)
            span = feature_extent((low, high))
            if not math.isfinite(span) or span <= 0:
                raise GeometryError('automatic approximation trim enclosure is degenerate')
            spans.append(span)
    if not spans:
        raise GeometryError('automatic approximation requires selected material faces')
    largest = max(spans)
    floor = max(model.tolerance.effective_length(largest),
                model.tolerance.effective_surface_residual(largest),
                model.tolerance.effective_curve_fit_residual(largest))
    return tuple(spans), min(spans), floor

def query_trim_enclosure_scale(model, faces, *, cancellation_check=None):
    """Smallest qualified boundary-loop enclosure diagonal, including holes.

    Enclosures may be loose: this is not intrinsic diameter, clearance, minimum
    feature size or a curvature measure. The floor is owner numerical policy,
    not a theorem of achievable approximation. Whole-cell proof is separate.
    """
    charts = query_trimmed_surface_charts(model, faces, cancellation_check=cancellation_check)
    values = _values(model, charts, cancellation_check)
    validate_trimmed_surface_charts_binding(model, charts, cancellation_check=cancellation_check)
    source = ApproximationSourceBinding(charts.model_id, charts.revision, charts.source_checksum)
    return TrimEnclosureScale(source, tuple(c.face.id for c in charts.charts), *values, charts)

def validate_trim_enclosure_scale_binding(model, result, *, cancellation_check=None):
    if type(result) is not TrimEnclosureScale:
        raise GeometryError('invalid approximation scale binding')
    validate_approximation_source_binding(model, result.source, cancellation_check=cancellation_check)
    validate_trimmed_surface_charts_binding(model, result.charts, cancellation_check=cancellation_check)
    expected = _values(model, result.charts, cancellation_check)
    if (result.source != ApproximationSourceBinding(result.charts.model_id, result.charts.revision,
            result.charts.source_checksum) or result.face_ids != tuple(c.face.id for c in result.charts.charts)
            or (result.loop_spans, result.reference_scale, result.policy_floor) != expected
            or result.scale_definition != 'minimum_trim_enclosure_span'):
        raise GeometryError('approximation scale binding changed')
    validate_approximation_source_binding(model, result.source, cancellation_check=cancellation_check)
