"""Private detached operands and explicit unpublished numerical boundaries.

No receipt/admission authority is issued here. The enclosing owner must call
finish before publishing; actual callbacks retain full before/after guards.
"""
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from uuid import UUID
import numpy as np
from .errors import GeometryError


class DetachedOperandUnavailable(GeometryError):
    pass


def detach_numeric(value):
    """Copy canonical numerical records, discarding evaluated instance caches."""
    kind = type(value)
    if value is None or kind in (str, bool, int, float, Fraction, UUID):
        return value
    if isinstance(value, Enum):
        return value
    if kind is tuple:
        return tuple(detach_numeric(x) for x in value)
    if kind is list:
        return tuple(detach_numeric(x) for x in value)
    if kind is np.ndarray:
        result = value.copy()
        result.setflags(write=False)
        return result
    if kind in (dict, MappingProxyType):
        return MappingProxyType({detach_numeric(k): detach_numeric(v) for k, v in value.items()})
    if (is_dataclass(value) and kind.__module__.startswith('anygeometry.')
            and getattr(kind, '__dataclass_params__').frozen):
        return replace(value, **{f.name: detach_numeric(getattr(value, f.name))
                                 for f in fields(value) if f.init})
    raise DetachedOperandUnavailable('detached geometry operand has an unsupported numerical type')


class NumericalBoundary:
    """Private operation lifetime; not a cache and never a proof receipt."""
    __slots__ = ('_guard', '_callback', '_active', 'source', '_source_state', '_detached', '_error')
    def __init__(self, guard, callback, *, source=None, error_type=GeometryError):
        from .prepared_material_source_transfer import _state
        self._error = error_type
        self.source = source
        self._detached = True
        self._source_state = None if source is None else _state(source)
        self._guard, self._callback, self._active = guard, callback, True
        guard()

    def guard(self):
        if not self._active:
            raise self._error('detached numerical operation is closed')
        if self.source is not None:
            from .prepared_material_source_transfer import _state
            if _state(self.source) != self._source_state:
                raise self._error('detached numerical source changed')
        self._guard()

    def tick(self, phase='local approximation field'):
        if not self._active:
            raise self._error('detached numerical operation is closed')
        if self._callback is None:
            if not self._detached: self.guard()
            return False
        # Even passive tokens are conservatively treated as external here;
        # no assumption about an arbitrary callable's mutation capability.
        self.guard()
        cancelled = self._callback(phase)
        self.guard()
        if cancelled:
            raise self._error('detached numerical operation cancelled')
        return False

    def finish(self):
        self.guard()
        self._active = False

    def close(self):
        self._active = False


def _material_context():
    from . import material_regions as mr, nurbs as n, surfaces as s
    from .definition_binding import definition_checksum
    functions = (mr._rows,mr._support_positions,mr._surface_derivatives_many,
        s._surface_derivatives_many,s.Plane.evaluate,n.NurbsSurface.evaluate,
        n.NurbsSurface.evaluate_many,n.NurbsSurface.derivatives_many,
        n._parameters,n._map,n._pair,n._real,n._scaled_weights,n._inward_spans,
        n._reference.find_spans,n._reference.basis_derivatives)
    return ((n.NurbsSurface,s.Plane,n._reference,mr.np,n.np,s.np),
        tuple((f,getattr(f,'__code__',None),definition_checksum(getattr(f,'__defaults__',None)),
               definition_checksum(getattr(f,'__kwdefaults__',None))) for f in functions),
        tuple(getattr(np,name) for name in ('asarray','all','isfinite','repeat','ones',
              'concatenate','arange','einsum','clip','searchsorted','zeros','empty','any','array','max','abs')))

_MATERIAL_CONTEXT = _material_context()

def canonical_material_context():
    return _material_context() == _MATERIAL_CONTEXT

class MaterialNumericalRead:
    """One unpublished attempt's detached canonical support and regular anchor.

    The caller must validate on successful scope exit and close on every exit.
    Public material evaluation and final mesh proof do not use this shortcut.
    """
    def __init__(self, model, collection, face, chords, allocation=None):
        from threading import get_ident
        from .nurbs import NurbsSurface
        from .surfaces import Plane
        from .definition_binding import definition_checksum
        from .prepared_material_source_transfer import _state
        from ._local_contact_stars import _runtime
        if not canonical_material_context():
            raise GeometryError('material numerical implementation context unavailable')
        region = next((r for r in collection.regions if face in {h.id for h in r.faces}), None)
        if region is None or type(region.support) not in (NurbsSurface, Plane):
            raise GeometryError('material numerical read support unavailable')
        self.model, self.collection, self.face = model, collection, face
        self.chords, self.allocation = chords, allocation
        self._thread, self._active = get_ident(), True
        self._state, self._runtime = _state(model), _runtime()
        self._literal = definition_checksum((collection, chords, allocation))
        self.validate()
        self.support = detach_numeric(region.support)
        self._support_pin = definition_checksum(self.support)
        anchors = [uv for handle, uv in chords.reference_uv_by_face
                   if handle == model.handle('face', face)]
        if len(anchors) != 1:
            raise GeometryError('material numerical read lacks one owner anchor')
        self.reference_uv = tuple(anchors[0])
        self.validate()
        self.evaluations = 0

    def check_open(self):
        from threading import get_ident
        if not self._active or get_ident() != self._thread:
            raise GeometryError('material numerical read is closed or belongs to another thread')

    def validate(self):
        from .definition_binding import definition_checksum
        from .prepared_material_source_transfer import _state
        from ._local_contact_stars import _runtime
        from .material_regions import validate_material_surface_regions_binding
        from .prepared_material_curvilinear import validate_prepared_material_boundary_chords_binding
        self.check_open()
        if not canonical_material_context():
            raise GeometryError('material numerical implementation context changed')
        if (_state(self.model) != self._state or _runtime() != self._runtime
                or definition_checksum((self.collection,self.chords,self.allocation)) != self._literal):
            raise GeometryError('material numerical read source or authority changed')
        validate_material_surface_regions_binding(self.model,self.collection)
        validate_prepared_material_boundary_chords_binding(self.model,self.collection,self.chords)
        if self.allocation is not None:
            from .local_approximation_field import validate_local_approximation_allocation
            validate_local_approximation_allocation(self.model,self.allocation)
        if (_state(self.model) != self._state or _runtime() != self._runtime
                or definition_checksum((self.collection,self.chords,self.allocation)) != self._literal):
            raise GeometryError('material numerical read changed during qualification')

    def evaluate(self, parameters, *, derivatives=False):
        from .material_regions import _rows, _support_positions, _surface_derivatives_many
        from .definition_binding import definition_checksum
        self.check_open()
        if not canonical_material_context():
            raise GeometryError('material numerical implementation context changed')
        if definition_checksum(self.support) != self._support_pin:
            raise GeometryError('detached material support changed')
        values,rows = _rows(parameters)
        # Match the original public calculation, including its position/domain
        # refusal even for derivative-only requests and finite result checks.
        positions = _support_positions(self.support,rows)
        arrays = _surface_derivatives_many(self.support,rows) if derivatives else (positions,)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise GeometryError('material region evaluation produced nonfinite results')
        self.evaluations += 1
        arrays = tuple(array.reshape((*values.shape[:-1],3)).copy() for array in arrays)
        return arrays if derivatives else arrays[0]

    def close(self):
        self._active = False
