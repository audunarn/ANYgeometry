"""PP5c bounded constructor-state qualification, NOT a purity or aliasing proof.

RAW NATIVE VIEWS are an ACCEPTED RESIDUAL RISK. Model traversal, owner reference
counts, NumPy ownership/base flags and GC array census bound the alias domain.
Fresh worker constructors, sampled serial verification and CI are the mitigations;
this adapter alone supplies neither the sampling nor the CI acceptance decision.
"""
import gc
import hashlib
import math
import struct
import sys
import weakref
from dataclasses import fields, is_dataclass
import numpy as np
from .surfaces import Cylinder, Plane
from ._planning_qualification import QualificationDecline

_NDARRAY_TYPE = np.ndarray


def _array_equal(a, b):
    return (type(a) is np.ndarray and type(b) is np.ndarray and
            a.dtype == b.dtype and a.shape == b.shape and
            a.flags.c_contiguous == b.flags.c_contiguous and
            not a.dtype.hasobject and not b.dtype.hasobject and
            np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and
            a.tobytes(order='C') == b.tobytes(order='C'))


def _owner_counts(record):
    array = record._circumferential
    if type(array) is not np.ndarray:
        raise QualificationDecline('derived array type')
    base = array.base
    return sys.getrefcount(array), None if base is None else sys.getrefcount(base)


def _plane_normal_count(record):
    array = object.__getattribute__(record, '__dict__')['normal']
    return sys.getrefcount(array)


def _plane_declared_fields(record):
    """Apply the declared-field admission contract before a cache can warm."""
    names = ('origin', 'u_vector', 'v_vector')
    namespace = object.__getattribute__(record, '__dict__')
    for name in names:
        value = namespace[name]
        if (type(value) is not _NDARRAY_TYPE or value.dtype != np.dtype(float)
                or not value.dtype.isnative or value.shape != (3,)
                or not value.flags.c_contiguous or value.flags.writeable
                or not np.all(np.isfinite(value))):
            raise QualificationDecline('unqualified Plane declared field')


def _plane_normal_cache(record, tracked):
    """Validate the existing H cache; never populate or transport that cache."""
    names = ('origin', 'u_vector', 'v_vector')
    namespace = object.__getattribute__(record, '__dict__')
    if type(record) is not Plane or set(namespace) != {*names, 'normal'}:
        raise QualificationDecline('unregistered Plane cache schema')
    _plane_declared_fields(record)
    fresh = Plane(**{name: namespace[name] for name in names})
    fresh.normal
    if any(not _array_equal(namespace[name], vars(fresh)[name]) for name in names):
        raise QualificationDecline('Plane reconstruction changed declared bits')
    if _plane_normal_count(record) != _plane_normal_count(fresh):
        raise QualificationDecline('Plane normal owner-reference alias')
    actual = namespace['normal']
    canonical = vars(fresh)['normal']
    if (type(actual) is not _NDARRAY_TYPE or actual.base is not None
            or not actual.flags.owndata or not actual.flags.c_contiguous
            or not actual.flags.aligned or actual.flags.writeable
            or canonical.base is not None or not canonical.flags.owndata
            or not canonical.flags.c_contiguous or not canonical.flags.aligned
            or canonical.flags.writeable or not _array_equal(actual, canonical)):
        raise QualificationDecline('Plane normal canonical bytes/layout/flags mismatch')
    if weakref.getweakrefcount(actual):
        raise QualificationDecline('Plane normal weak-reference alias')
    for other in tracked:
        if other is actual:
            continue
        try:
            if np.shares_memory(actual, np.ndarray.view(other, type=np.ndarray), max_work=1000):
                raise QualificationDecline('Plane normal GC-census alias')
        except np.exceptions.TooHardError as exc:
            raise QualificationDecline('Plane normal GC alias query exceeded bound') from exc


def _extra(record, path, tracked, expected):
    # This exact registry comes from the PP5b real census: Cylinder only.
    if type(record) is not Cylinder or set(vars(record)) != {f.name for f in fields(Cylinder)} | {'_circumferential'}:
        raise QualificationDecline('non-field model record instance state: unregistered constructor schema')
    # Reject unknown field adapters before the constructor can invoke __array__
    # or __float__. Only the exact passive types produced by Cylinder are admitted.
    for field in fields(Cylinder):
        value = getattr(record, field.name)
        if field.name in ('origin', 'axis', 'radial_direction'):
            valid = (type(value) is np.ndarray and value.dtype == np.dtype(float) and
                     value.shape == (3,) and value.flags.c_contiguous and np.all(np.isfinite(value)))
        else:
            valid = type(value) is float and math.isfinite(value)
        if not valid:
            raise QualificationDecline('unqualified constructor field: '+field.name)
    try:
        fresh = Cylinder(**{f.name: getattr(record, f.name) for f in fields(Cylinder)})
    except Exception as exc:
        raise QualificationDecline('derived constructor recomputation failed') from exc
    # Compare exact counts in the SAME helper against a just-created, unexported
    # constructor owner. CPython 3.14 borrowed locals make a hard-coded count wrong.
    if _owner_counts(record) != _owner_counts(fresh):
        raise QualificationDecline('derived array owner/base-reference alias')
    actual = record._circumferential
    if type(actual) is not np.ndarray or not actual.flags.c_contiguous:
        raise QualificationDecline('derived array type/contiguity')
    if weakref.getweakrefcount(actual) or (actual.base is not None and weakref.getweakrefcount(actual.base)):
        raise QualificationDecline('derived array weak-reference alias')
    if actual.base is None:
        if not actual.flags.owndata:
            raise QualificationDecline('derived array ownership unknown')
    else:
        base = actual.base
        # np.cross returns a full-buffer view on this NumPy runtime. Permit that
        # exact single-owned base pattern, not arbitrary strided/base chains.
        if (type(base) is not np.ndarray or base.base is not None or not base.flags.owndata or
                not base.flags.c_contiguous or base.dtype != actual.dtype or
                base.nbytes != actual.nbytes or
                base.__array_interface__['data'][0] != actual.__array_interface__['data'][0]):
            raise QualificationDecline('derived array base/ownership alias')
    for other in tracked:
        if other is actual or other is actual.base:
            continue
        try:
            if np.shares_memory(actual, np.ndarray.view(other, type=np.ndarray), max_work=1000):
                raise QualificationDecline('derived array GC-census alias')
        except np.exceptions.TooHardError as exc:
            raise QualificationDecline('derived array GC alias query exceeded bound') from exc
    if not _array_equal(actual, fresh._circumferential):
        raise QualificationDecline('derived constructor bytes/dtype/shape/finite mismatch')
    # Repeated normalization must also preserve every original declared bit.
    for f in fields(Cylinder):
        left, right = getattr(record, f.name), getattr(fresh, f.name)
        if type(left) is np.ndarray:
            equal = _array_equal(left, right)
        else:
            equal = (type(left) is float and type(right) is float and math.isfinite(left) and
                     struct.pack('!d',left) == struct.pack('!d',right))
        if not equal:
            raise QualificationDecline('constructor changed declared field: '+f.name)
    row = dict(path=path, record_class='anygeometry.surfaces.Cylinder', attribute='_circumferential',
               dtype=actual.dtype.str, shape=list(actual.shape), c_contiguous=True,
               sha256=hashlib.sha256(actual.tobytes(order='C')).hexdigest(),
               writeable=bool(actual.flags.writeable))
    if expected is not None:
        target = expected.get(path)
        if (type(target) is not dict or set(target) != set(row) or
                type(target.get('writeable')) is not bool or target.get('c_contiguous') is not True or
                type(target.get('shape')) is not list or any(type(n) is not int for n in target['shape']) or
                {k:v for k,v in target.items() if k != 'writeable'} !=
                {k:v for k,v in row.items() if k != 'writeable'}):
            raise QualificationDecline('constructor-state transport mismatch')
        if not target['writeable']:
            actual.flags.writeable = False  # Only a freshly decoded worker record.
        if bool(actual.flags.writeable) != target['writeable']:
            raise QualificationDecline('constructor-state writeability mismatch')
        row['writeable'] = bool(actual.flags.writeable)
    return row


def audit(state, *, restore=None, limit=100000):
    """Audit source read-only; restore=rows only on freshly constructor-decoded records.

    No model/array cache survives this call. No new global mutable store. Unknown
    record extras and warm dataclass caches keep their original refusal semantics.
    """
    if restore is not None:
        if type(restore) is not list or any(type(r) is not dict or type(r.get('path')) is not str for r in restore):
            raise QualificationDecline('constructor-state manifest schema')
        expected = {row['path']: row for row in restore}
        if len(expected) != len(restore):
            raise QualificationDecline('duplicate constructor-state path')
    else:
        expected = None
    # Bind the exact native target before scanning unrelated host objects.
    # CPython's isinstance fallback reads an object's user-defined __class__;
    # issubclass(real_type, exact_type_target) uses native subtype checks instead.
    # Actual ndarray subclasses still enter the alias census below, whose base C
    # view method avoids their __array_function__ overrides.
    if np.ndarray is not _NDARRAY_TYPE or type(_NDARRAY_TYPE) is not type:
        raise QualificationDecline('constructor ndarray type binding changed')
    tracked = [obj for obj in gc.get_objects() if issubclass(type(obj), _NDARRAY_TYPE)]
    pending, seen, rows = [(state, 'model')], set(), []
    while pending:
        item, path = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen) > limit:
            raise QualificationDecline('model record-state walk truncated limit='+str(limit))
        typ = type(item)
        if typ is dict:
            pending.extend((v,path+'['+repr(k)+']') for k,v in item.items())
            pending.extend((k,path+'.key') for k in item)
        elif typ in (tuple,list,set,frozenset):
            pending.extend((v,path+'['+str(i)+']') for i,v in enumerate(item))
        elif not isinstance(item,type) and is_dataclass(item):
            declared = fields(item)
            try:
                namespace = object.__getattribute__(item,'__dict__')
            except AttributeError:
                namespace = {}
            extra = set(namespace)-{f.name for f in declared}
            if typ is Plane and extra:
                _plane_normal_cache(item, tracked)
            elif typ is Plane:
                _plane_declared_fields(item)
            elif extra or typ is Cylinder:
                rows.append(_extra(item,path,tracked,expected))
            for f in declared:
                part = getattr(item,f.name)
                from ._planning_model_state import registered_record_cache
                if registered_record_cache(item, f, part):
                    continue
                if f.metadata.get('definition') is False and part is not None:
                    raise QualificationDecline('warm model record instance cache')
                pending.append((part,path+'.'+f.name))
    rows.sort(key=lambda row:row['path'])
    if expected is not None and set(expected) != {row['path'] for row in rows}:
        raise QualificationDecline('constructor-state coverage mismatch')
    return rows
