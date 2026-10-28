"""Live owner qualification of a renumbered ``ModelClosure`` as an immediate source route.

A closure's maps are claims. Authority is granted only after this module independently reconstructs the
selected closure from the live source with the existing ``extract_model_closure`` and finds the supplied
working model (public AND native definitions) and both identity maps equal to that reconstruction. Copies,
transported or loaded closures hold no authority until qualified here against a live source.
"""
from __future__ import annotations

import numpy as np

from .closure import ModelClosure, extract_model_closure
from .errors import GeometryError
from .identity import EntityHandle
from .model import GeometryModel
from .definition_binding import definition_checksum

__all__ = ["qualify_model_closure_source"]

_IGNORED = ("model_id", "checksum", "revision")


def _public(model):
    from .serialization import to_dict
    return {k: v for k, v in to_dict(model).items() if k not in _IGNORED}


def _rows(maps, source_id, work_id):
    """Normalised ``(kind, work_id, source_id)`` rows of both maps, with the handle models verified."""
    forward, backward = maps
    rows = set()
    for source, target in forward.items():
        if (type(source) is not EntityHandle or type(target) is not EntityHandle
                or source.model_id != source_id or target.model_id != work_id or source.kind != target.kind):
            raise GeometryError('closure source map has an inconsistent handle identity or kind')
        rows.add((source.kind, target.id, source.id))
    inverse = set()
    for target, source in backward.items():
        if (type(source) is not EntityHandle or type(target) is not EntityHandle
                or source.model_id != source_id or target.model_id != work_id or source.kind != target.kind):
            raise GeometryError('closure source map has an inconsistent handle identity or kind')
        inverse.add((target.kind, target.id, source.id))
    if rows != inverse or len(rows) != len(forward) or len(rows) != len(backward):
        raise GeometryError('closure source maps are not complete bijective inverses')
    return frozenset(rows)


def _validate_shape(source, closure):
    """Typed refusal of malformed closure fields before any read or callback (never an AttributeError)."""
    from collections.abc import Mapping
    if not isinstance(closure.source_to_work, Mapping) or not isinstance(closure.work_to_source, Mapping):
        raise GeometryError('closure source maps must be mappings')
    if type(closure.source_handles) is not tuple:
        raise GeometryError('closure source selection must be a tuple')
    for handle in closure.source_handles:
        if (type(handle) is not EntityHandle or handle.model_id != source.model_id
                or type(handle.kind) is not str or type(handle.id) is not int):
            raise GeometryError('closure source selection contains a malformed or wrong-model handle')
    if type(closure.source_revision) is not int or type(closure.source_model_id) is not type(source.model_id):
        raise GeometryError('closure source identity or revision is malformed')


def _pins(closure):
    work = closure.working_model
    return (_rows((closure.source_to_work, closure.work_to_source), closure.source_model_id, work.model_id),
            tuple((h.kind, h.id, str(h.model_id)) for h in closure.source_handles),
            str(closure.source_model_id), closure.source_revision)


def qualify_model_closure_source(source, closure, *, cancellation_check=None):
    """Qualify ``closure`` against the live ``source`` and register the immediate source route.

    Returns ``closure`` unchanged after success. On any failure or cancellation both models and the prior
    source-lineage registry are unchanged. Ordinary extraction never calls this (opt-in).
    """
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    from . import prepared_material_source_transfer as transfer
    if type(closure) is not ModelClosure or not isinstance(source, GeometryModel):
        raise GeometryError('closure source qualification requires a GeometryModel and a ModelClosure')
    work = closure.working_model
    if (not isinstance(work, GeometryModel) or work is source or work.model_id == source.model_id
            or closure.source_model_id != source.model_id or closure.source_revision != source.revision):
        raise GeometryError('closure source qualification belongs to another or stale source')
    _validate_shape(source, closure)
    # no callback exists yet: everything below until the first frontier is caller-independent
    validate_model_curve_runtimes(source)
    validate_model_curve_runtimes(work)
    source_state, work_state = transfer._state(source), transfer._state(work)
    pins = _pins(closure)
    links = transfer._LINKS.get(work)
    objects = (work, closure.source_to_work, closure.work_to_source, closure.source_handles)

    def current():
        # object identity first, before any read through the closure: a callback may substitute a
        # same-UUID, equal-definition working model or replacement maps/selection
        if (type(closure) is not ModelClosure or closure.working_model is not objects[0]
                or closure.source_to_work is not objects[1] or closure.work_to_source is not objects[2]
                or closure.source_handles is not objects[3]):
            raise GeometryError('closure working model or maps were substituted during qualification')
        validate_model_curve_runtimes(source)
        validate_model_curve_runtimes(work)
        if (transfer._state(source) != source_state or transfer._state(work) != work_state
                or _pins(closure) != pins or transfer._LINKS.get(work) is not links):
            raise GeometryError('closure source operands changed during qualification')

    def check(phase):
        cancelled = cancellation_check(phase) if cancellation_check is not None else False
        current()
        if cancelled:
            raise GeometryError('closure source qualification cancelled')

    check('closure source qualification')
    expected = None
    for structural in (True, False):             # the closure stores no option flag: try each exact variant
        try:
            candidate = extract_model_closure(source, closure.source_handles,
                                              include_structural_closure=structural, include_features=False)
        except GeometryError:
            continue
        check('closure source reconstruction')
        if (candidate.source_handles == closure.source_handles
                and _public(candidate.working_model) == _public(work)
                and transfer.capture_native_supports(candidate.working_model) == transfer.capture_native_supports(work)
                and _rows((candidate.source_to_work, candidate.work_to_source), source.model_id,
                          candidate.working_model.model_id) == pins[0]
                and work.units == source.units and work.units == candidate.working_model.units
                and np.array_equal(work.local_origin, candidate.working_model.local_origin)
                and _same_transform(work, candidate.working_model)):
            expected = candidate
            break
    if expected is None:
        raise GeometryError('closure is not the independent extraction of the live source selection')
    check('closure source qualification complete')
    # All user callbacks have completed. Final current gate, then registration (which rechecks states).
    current()
    faces = tuple((source_id, work_id) for kind, work_id, source_id in pins[0] if kind == 'face')
    edges = tuple((work_id, source_id) for kind, work_id, source_id in pins[0] if kind == 'edge')
    prior = transfer._LINKS.get(work)
    try:
        transfer._register_source_closure(source, work, faces, edges, source_state, work_state)
    except BaseException:
        # any exception or cancellation after (or during) publication restores the exact prior registry
        # entry (identity, not an equal copy), or removes the new entry when none existed; no callback runs
        if prior is None:
            transfer._LINKS.pop(work, None)
        else:
            transfer._LINKS[work] = prior
        raise
    return closure


def _same_transform(a, b):
    left, right = a.coordinate_transform, b.coordinate_transform
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(left, right))
