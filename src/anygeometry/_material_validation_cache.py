"""Transient, weak-model validation scopes; never persisted proof authority."""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping

from .definition_binding import definition_checksum
from .errors import GeometryError


def validation_epoch(model, checksum):
    from .native_support_snapshots import capture_material_validation_read_pin
    from .support_curves import PlaneNurbsCurve
    laws = tuple((edge.id, type(edge.curve._INITIAL_ROOT_BISECTIONS), edge.curve._INITIAL_ROOT_BISECTIONS)
                 for edge in model.edges.values() if type(edge.curve) is PlaneNurbsCurve)
    if any(kind is not int or count != 60 for _, kind, count in laws):
        raise GeometryError('material validation section evaluator law is not certified')
    native = capture_material_validation_read_pin(model)
    if native is None:
        return None  # unknown native dependencies always take full qualification
    return checksum, native, laws


@dataclass
class _Scopes:
    epoch: tuple
    singletons: dict
    recent: tuple = ()


def _scopes(cache, model, epoch):
    current = cache.get(model)
    if type(current) is _Scopes and current.epoch[0] == epoch[0] and current.epoch != epoch:
        raise GeometryError('material validation native binding changed without a source definition change')
    if type(current) is not _Scopes or current.epoch != epoch:
        current = _Scopes(epoch, {})
        cache[model] = current  # retire EVERY old-epoch slot together
    return current


def singleton_face(model, collection):
    """Canonical singleton selection; the full result signature remains its key."""
    from .trimmed_charts import TrimmedSurfaceCharts, TrimmedSurfaceChart
    from .identity import EntityHandle
    if type(collection) is not TrimmedSurfaceCharts or type(collection.charts) is not tuple or len(collection.charts) != 1:
        return None
    chart = collection.charts[0]
    if type(chart) is not TrimmedSurfaceChart or type(chart.face) is not EntityHandle:
        return None
    face = chart.face
    return (face.id if face.kind == 'face' and face.model_id == model.model_id
            and type(face.id) is int and face.id in model.faces else None)


def contains(cache, model, epoch, face, signature):
    if epoch is None:
        cache.pop(model, None)
        return False
    current = _scopes(cache, model, epoch)
    return (current.singletons.get(face) == signature if face is not None else signature in current.recent)


def remember(cache, model, epoch, face, signature):
    if epoch is None:
        cache.pop(model, None)
        return
    current = _scopes(cache, model, epoch)
    if face is not None:
        current.singletons[face] = signature  # exactly one proof key per LIVE face
    else:
        previous = tuple(item for item in current.recent if item != signature)
        current.recent = (*previous[-7:], signature)


def _identities(value):
    """Pin consumed public object graph only across arbitrary callbacks."""
    seen, rows = set(), []
    def visit(item):
        if item is None or type(item) in (str, int, float, bool, bytes):
            return
        key = id(item)
        if key in seen:
            return
        seen.add(key)
        if is_dataclass(item):
            rows.append((key, type(item)))
            for field in fields(item):
                if field.metadata.get('definition') is not False:
                    visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            rows.append((key, type(item)))
            for key, child in item.items():
                visit(key); visit(child)
        elif isinstance(item, (tuple, list)):
            rows.append((key, type(item)))
            for child in item:
                visit(child)
        else:
            # Includes native public ndarray fields; their contents are also
            # in definition_checksum. Primitive value wrappers need no pin.
            from numpy import ndarray
            if isinstance(item, ndarray):
                rows.append((key, type(item)))
    visit(value)
    return tuple(rows)


def guarded_callback(model, result, callback, checksum):
    if callback is None:
        return None
    from .cancellation import is_cancellation_signal
    if is_cancellation_signal(callback):
        return callback
    from .serialization import _qualified_model_state
    validate_evidence_native(result)  # audit BEFORE caller code can repair a poisoned input
    literal, identities = definition_checksum(result), _identities(result)
    epoch, revision = validation_epoch(model, checksum), model.revision
    def guarded(phase):
        cancelled = callback(phase)  # caller exceptions retain their identity
        _qualified_model_state(model, expected_checksum=checksum,
                               changed_message='material validation source changed during callback')
        if (model.revision != revision or validation_epoch(model, checksum) != epoch
                or definition_checksum(result) != literal or _identities(result) != identities):
            raise GeometryError('material validation operands changed during callback')
        validate_evidence_native(result)
        return cancelled
    return guarded


def validate_evidence_native(result):
    """Check native kernels actually consumed from equivalent result objects.

    No result identity is an authority: equivalent definitions remain reusable.
    A detached equal-definition object still owes its actual evaluator audit.
    """
    from .native_support_snapshots import _material_support_read_pin
    from .nurbs_runtime_binding import validate_curve_runtime, validate_section_runtime
    from .nurbs import NurbsCurve, CurveOnSurface
    from .support_curves import PlaneNurbsCurve
    from .line_ranges import LineRange
    domains, extra = [], []
    if hasattr(result, 'charts'):
        domains.extend(chart.domain for chart in result.charts)
    else:
        domains.extend(chart.domain for chart in result.source.charts)
        domains.extend(region.domain for region in result.regions)
        domains.extend(chart.domain for region in result.regions for chart in region.sources)
        extra.extend(path.curve for region in result.regions for path in region.interior_constraints)
    reusable, seen = True, set()
    for domain in domains:
        _validate_domain_memos(domain)
        if _material_support_read_pin(domain.support) is None:
            reusable = False
        extra.extend(path.curve for loop in domain.boundaries for path in loop)
    for curve in extra:
        if id(curve) in seen:
            continue
        seen.add(id(curve))
        if type(curve) is NurbsCurve:
            validate_curve_runtime(curve)
        elif type(curve) is CurveOnSurface:
            validate_curve_runtime(curve.pcurve)
            if _material_support_read_pin(curve.support) is None:
                reusable = False
        elif type(curve) is PlaneNurbsCurve:
            if type(curve._INITIAL_ROOT_BISECTIONS) is not int or curve._INITIAL_ROOT_BISECTIONS != 60:
                raise GeometryError('material validation section evaluator law is not certified')
            validate_section_runtime(curve)
        elif type(curve) is LineRange:
            curve.validate_definition()
    return reusable


def _validate_domain_memos(domain):
    """Audit only the exact Plane memo entries its current consumer will use."""
    from fractions import Fraction
    from .surfaces import Plane
    from .planar_exact import ExactPlane
    from .material_arrangement import (MaterialDomain, ArrangementPath, LinePath,
                                        _ExactBoundaryLine, _prepare_exact_boundary_line)
    if type(domain.support) is not Plane:
        return
    def same(actual, expected):
        if type(expected) is Fraction:
            return type(actual) is Fraction and actual == expected
        return (type(actual) is tuple and len(actual) == len(expected)
                and all(same(a, b) for a, b in zip(actual, expected)))
    fresh = None
    def chart():
        nonlocal fresh
        if fresh is None:
            fresh = ExactPlane.from_plane(domain.support)
        return fresh
    cached = domain._exact_chart_cache
    if cached is not None:
        if type(cached) is not tuple or not cached:
            raise GeometryError('material validation exact chart cache is malformed')
        if cached[0] is domain.support:
            expected = chart()
            if (len(cached) != 2 or type(cached[1]) is not ExactPlane
                    or any(not same(getattr(cached[1], name, None), getattr(expected, name))
                           for name in ('origin', 'u_vector', 'v_vector', 'normal', '_scale', '_dual_u', '_dual_v'))):
                raise GeometryError('material validation exact chart cache differs from its definition')
    if (type(domain) is not MaterialDomain or type(domain.boundaries) is not tuple
            or any(vector.flags.writeable for vector in
                   (domain.support.origin, domain.support.u_vector, domain.support.v_vector))):
        return  # the actual _prepared_exact_boundary disables this memo
    cached = domain._exact_boundary_cache
    if cached is None:
        return
    if type(cached) is not tuple or len(cached) < 2:
        raise GeometryError('material validation exact boundary cache is malformed')
    if cached[0] is not domain.support or cached[1] is not domain.boundaries:
        return  # the actual consumer replaces this stale identity entry
    if len(cached) != 3 or type(cached[2]) is not dict:
        raise GeometryError('material validation exact boundary cache is malformed')
    for loop in domain.boundaries:
        if type(loop) is not tuple or not all(type(path) is ArrangementPath and type(path.curve) is LinePath
                and type(path.curve.start) is tuple and type(path.curve.end) is tuple for path in loop):
            continue
        prepared = cached[2].get(id(loop))
        if prepared is None:
            continue
        if type(prepared) is not dict:
            raise GeometryError('material validation exact boundary memo is malformed')
        for path in loop:
            entry = prepared.get(id(path.curve))
            if entry is None:
                continue
            if type(entry) is not tuple or not entry:
                raise GeometryError('material validation exact boundary entry is malformed')
            if entry[0] is not path.curve:
                continue  # ignored/replaced by _contains_loop_exact.line_data
            expected = _prepare_exact_boundary_line(chart(), path.curve)
            if (len(entry) != 2 or type(entry[1]) is not _ExactBoundaryLine
                    or any(not same(getattr(entry[1], field.name, None), getattr(expected, field.name))
                           for field in fields(expected))):
                raise GeometryError('material validation exact boundary cache differs from its definition')
