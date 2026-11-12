"""Versioned exact transfer for trusted, local computational worker processes.

This is not file interchange or an untrusted-input decoder. Never load a
ModelTransfer supplied by a file, network or another trust domain: its verified
pickle bytes are executable trusted runtime data. Persistent schema codecs and
repair authority are deliberately unchanged.
"""
from __future__ import annotations

import copyreg
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from functools import cached_property
from io import BytesIO
import pickle
from time import perf_counter
from types import MappingProxyType

import numpy as np

from .errors import GeometryError
from .model import GeometryModel

_VERSION = 1


@dataclass(frozen=True)
class ModelTransfer:
    version: int
    model_id: str
    revision: int
    source_checksum: str
    entity_counts: tuple[int, int, int, int]
    payload: bytes


@dataclass(frozen=True)
class _ClassificationReceipt:
    checksum: str
    model_id: str
    revision: int
    entity_counts: tuple[int, int, int, int]


def _issue_classification_receipt(transfer, root_state):
    """Private reuse boundary after this exact root strict topology audit."""
    binding = (transfer.model_id, transfer.revision)
    if (getattr(root_state, '_topology_audit_binding', None) != binding
            or str(root_state.model.model_id) != transfer.model_id
            or root_state.model.revision != transfer.revision
            or (len(root_state.model.vertices), len(root_state.model.edges), len(root_state.model.faces),
                len(root_state.model.members)) != transfer.entity_counts):
        raise GeometryError('Classification transfer requires the same completed root topology audit.')
    return _ClassificationReceipt(transfer.source_checksum, transfer.model_id, transfer.revision, transfer.entity_counts)


def _rebuild_mapping(values):
    return MappingProxyType(values)


def _rebuild_record(record_type, values):
    # Preserve required constructor-derived geometric state without running
    # normalisation again. Only actual cached_property entries are omitted.
    record = object.__new__(record_type)
    for name, value in values:
        object.__setattr__(record, name, value)
    return record


class _RuntimePickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[MappingProxyType] = lambda value: (_rebuild_mapping, (dict(value),))

    def reducer_override(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            record_type = type(value)
            # Explicit owner state protocols rebuild prepared curve state
            # without transporting its invocation/cache authority.
            state_method = record_type.__dict__.get('__getstate__')
            if state_method is not None and getattr(state_method, '__module__', '') != 'dataclasses':
                return NotImplemented
            values = {item.name: getattr(value, item.name) for item in fields(value)}
            extra = dict(getattr(value, '__dict__', {}))
            for base in record_type.__mro__:
                for name in getattr(base, '__slots__', ()):
                    if name not in ('__dict__', '__weakref__') and hasattr(value, name):
                        extra.setdefault(name, getattr(value, name))
            for name, item in sorted(extra.items()):
                descriptor = next((base.__dict__[name] for base in record_type.__mro__ if name in base.__dict__), None)
                if name not in values and not isinstance(descriptor, cached_property):
                    values[name] = item
            return (_rebuild_record, (record_type, tuple(values.items())))
        return NotImplemented


def capture_model_transfer(model: GeometryModel, *, timings=None) -> ModelTransfer:
    """Copy one committed model into immutable, integrity-bound runtime bytes.

    No model edits, feature replay, JSON encoding, hooks or cache authority.
    Restore independently validates topology, structural records and features.
    """
    if not isinstance(model, GeometryModel):
        raise TypeError('capture_model_transfer requires GeometryModel')
    if model._transaction_journal is not None or model._notifying_hooks:
        raise GeometryError('Runtime transfer requires a committed model outside transactions/hooks.')
    started = perf_counter()
    identity, revision = str(model.model_id), model.revision
    counts = (len(model.vertices), len(model.edges), len(model.faces), len(model.members))
    design = model.design_snapshot()
    # These compatibility undo maps duplicate the immutable committed entity
    # records. Owner restore explicitly defaults to those records when absent.
    for name in ('vertex_state', 'edge_state', 'face_state'):
        design['topology'].pop(name, None)
    state = {'version': _VERSION, 'model_id': identity, 'revision': revision, 'entity_counts': counts,
             'design': design, 'tolerance': model.tolerance,
             'units': model.units, 'local_origin': model.local_origin,
             'coordinate_transform': model.coordinate_transform, 'crs_metadata': model.crs_metadata,
             'entity_versions': dict(model._entity_versions),
             'preparation_invalidated': any(hasattr(model, name) for name in (
                 '_intersection_preparation_receipt', '_edge_subcurve_preimages_receipt',
                 '_anyfem_transport_requires_preparation')),
             'extensions': getattr(model, '_serialization_extensions', {})}
    snapshot_done = perf_counter()
    output = BytesIO()
    _RuntimePickler(output, protocol=5).dump(state)
    payload = output.getvalue()
    if str(model.model_id) != identity or model.revision != revision:
        raise GeometryError('Source changed while capturing runtime transfer.')
    transfer = ModelTransfer(_VERSION, identity, revision, sha256(payload).hexdigest(), counts, payload)
    if timings is not None:
        timings.update(snapshot=snapshot_done - started, encode_hash=perf_counter() - snapshot_done,
                       total=perf_counter() - started)
    return transfer


def _freeze_geometry_arrays(value, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _freeze_geometry_arrays(getattr(value, item.name), seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _freeze_geometry_arrays(item, seen)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_geometry_arrays(item, seen)


def restore_model_transfer(transfer: ModelTransfer, *, timings=None) -> GeometryModel:
    """Restore trusted local runtime data into a validated independent model.

    Integrity/version/envelope are checked before trusted pickle decoding.
    This function does not authenticate arbitrary external pickle payloads.
    """
    return _restore_transfer(transfer, timings=timings)


def _restore_classification_transfer(transfer, receipt, *, timings=None):
    if (type(receipt) is not _ClassificationReceipt
            or (receipt.checksum, receipt.model_id, receipt.revision, receipt.entity_counts) !=
            (transfer.source_checksum, transfer.model_id, transfer.revision, transfer.entity_counts)):
        raise GeometryError('Classification transfer receipt does not match this exact source payload.')
    return _restore_transfer(transfer, timings=timings, classification_receipt=receipt)


def _restore_transfer(transfer, *, timings=None, classification_receipt=None):
    started = perf_counter()
    if type(transfer) is not ModelTransfer:
        raise TypeError('restore_model_transfer requires a trusted ModelTransfer')
    if type(transfer.version) is not int or transfer.version != _VERSION:
        raise GeometryError('Unsupported geometry runtime transfer version.')
    if type(transfer.payload) is not bytes or sha256(transfer.payload).hexdigest() != transfer.source_checksum:
        raise GeometryError('Geometry runtime transfer checksum mismatch.')
    try:
        decoded_started = perf_counter()
        state = pickle.loads(transfer.payload)
        decode_done = perf_counter()
        if not isinstance(state, dict) or any(state.get(name) != getattr(transfer, name) for name in
                ('version', 'model_id', 'revision', 'entity_counts')):
            raise GeometryError('Geometry runtime transfer envelope mismatch.')
        if type(transfer.revision) is not int or transfer.revision < 0:
            raise GeometryError('Invalid geometry runtime revision.')
        # Revalidate the public tolerance value, retaining its exact fields.
        tolerance = replace(state['tolerance'])
        model = GeometryModel(model_id=transfer.model_id, tolerance=tolerance)
        model.set_document_settings(units=state['units'], local_origin=state['local_origin'],
            coordinate_transform=state['coordinate_transform'], crs_metadata=state['crs_metadata'])
        # Owner-internal unpublished staging, identical to clone's boundary.
        model._restore_topology_unchecked(state['design']['topology'])
        model.features.restore(state['design']['features'])
        model._serialization_extensions = state['extensions']
        model._entity_versions = dict(state['entity_versions'])
        if state.get('preparation_invalidated', False):
            model._anyfem_transport_requires_preparation = True
        restore_done = perf_counter()
        if classification_receipt is None:
            problems = (*model.validate_topology(), *model._validate_structural())
            if problems:
                raise GeometryError('Invalid geometry runtime topology: ' + '; '.join(problems))
        model.features.validate_persistence(model)
        if (len(model.vertices), len(model.edges), len(model.faces), len(model.members)) != transfer.entity_counts:
            raise GeometryError('Geometry runtime entity count mismatch.')
        # Compatibility restoration deep-copies geometric definitions. Reapply
        # their immutable array boundary rather than exposing writable copies.
        seen = set()
        for edge in model.edges.values():
            _freeze_geometry_arrays(edge.curve, seen)
        for face in model.faces.values():
            _freeze_geometry_arrays(face.surface, seen)
            _freeze_geometry_arrays(face.parameterization, seen)
        model._revision = transfer.revision
        if timings is not None:
            timings.update(hash_check=decoded_started - started, decode=decode_done - decoded_started,
                           restore=restore_done - decode_done, validate_freeze=perf_counter() - restore_done,
                           total=perf_counter() - started)
        return model
    except GeometryError:
        raise
    except Exception as error:
        raise GeometryError(f'Invalid geometry runtime transfer: {type(error).__name__}: {error}') from error
