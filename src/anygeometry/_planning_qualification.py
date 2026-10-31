"""PP1b engineering controls for a fresh owned worker; serial remains untouched.

Pins and canary references must come from an independently qualified baseline.
Never capture a failing candidate as its own expected reference. Current-file
hashes are engineering provenance, not proof of previously loaded native images.
PP2 binds this gate to owned spawn and the complete task cache registry; PP6 binds
the two-digest CI control. This module itself launches no worker and enables no pool.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import types
import weakref

from .errors import GeometryError


class QualificationDecline(GeometryError):
    """Reject acceleration; the original serial path retains its own authority."""


def _hash(path):
    try:
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    except (OSError, ValueError) as exc:
        raise QualificationDecline('unreadable pinned hash: ' + str(path)) from exc


def pin_module(module):
    if type(module) is not types.ModuleType:
        raise QualificationDecline('unpinned custom module type')
    ns = types.ModuleType.__getattribute__(module, '__dict__')
    path = ns.get('__file__')
    if path:
        path = Path(path).resolve()
        suffix = path.suffix.lower()
        if suffix in ('.pyc', '.pyo'):
            raise QualificationDecline('source-less module is not qualified')
        kind = 'python' if suffix == '.py' else 'native'
        return dict(kind=kind, path=str(path), sha256=_hash(path))
    spec = ns.get('__spec__')
    if spec is not None and spec.origin in ('built-in', 'frozen'):
        return dict(kind=spec.origin, sha256=_hash(sys.executable))
    if spec is not None and spec.origin is None and ns.get('__path__') is not None:
        return dict(kind='namespace', paths=sorted(map(str, ns['__path__'])))
    raise QualificationDecline('module has no qualified origin: ' + str(ns.get('__name__')))


def _type_name(typ):
    module = type.__getattribute__(typ, '__module__')
    name = type.__getattribute__(typ, '__qualname__')
    if type(module) is not str or type(name) is not str:
        raise QualificationDecline('native type has no qualified static name')
    return module + '.' + name


def _input_types():
    # Deliberate adapter list, not discovery of every export in native modules.
    import numpy as np
    from .model import GeometryModel
    from .material_arrangement import MaterialDomain, ArrangementPath
    from .arrangement_geometry import LinePath
    from .surfaces import Plane
    return (type(None), bool, int, float, complex, str, bytes, tuple, list, dict,
            set, frozenset, np.ndarray, np.float64, np.int64, GeometryModel,
            MaterialDomain, ArrangementPath, LinePath, Plane)


def input_type_names():
    return sorted(_type_name(t) for t in _input_types())


def capture_environment(*, generated_modules=(), declined_stores=None):
    """Record the actual loaded closure after declared eager imports.

    Must run at the fresh-worker boundary and again after each task. All loaded
    modules are pinned, including Python dependencies, native modules and BLAS
    library files. Namespace directories and the executable are explicit inputs.
    A new late module changes the manifest and refuses; no wildcard package pin.
    """
    import numpy
    import scipy
    from threadpoolctl import threadpool_info
    # Load metadata/CPU helpers before enumerating the closure.
    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'threadpoolctl')}
    cpu = dict(machine=platform.machine(), processor=platform.processor(),
               identifier=os.environ.get('PROCESSOR_IDENTIFIER', ''), count=os.cpu_count())
    pools = threadpool_info()
    if not pools or any(type(p.get('num_threads')) is not int or p['num_threads'] != 1 for p in pools):
        raise QualificationDecline('BLAS/OpenMP thread count is absent or not one')
    blas = []
    for p in pools:
        if not p.get('filepath'):
            raise QualificationDecline('BLAS native library hash unavailable')
        blas.append({**p, 'sha256': _hash(p['filepath'])})
    modules, files = {}, {}
    generated = frozenset(generated_modules)
    if any(type(n) is not str for n in generated):
        raise QualificationDecline('invalid generated module allowlist')
    allowed_input_names = input_type_names()
    loaded = tuple(sorted(sys.modules.items()))
    for name, module in loaded:
        if module is None:
            raise QualificationDecline('unresolved loaded module: ' + name)
        if name in generated:
            if type(module) is not types.ModuleType:
                raise QualificationDecline('generated module type mismatch')
            namespace = module.__dict__
            if namespace.get('__file__') or namespace.get('__spec__') is not None:
                raise QualificationDecline('generated module origin changed')
            pin = dict(kind='explicit-generated-native',
                       members=sorted([key, type.__getattribute__(type(value), '__name__')]
                                      for key, value in namespace.items()))
        else:
            pin = pin_module(module)
        modules[name] = pin
        if 'path' in pin:
            files[pin['path']] = pin['sha256']
    for p in blas:
        files[str(Path(p['filepath']).resolve())] = p['sha256']
    if not generated.issubset(modules):
        raise QualificationDecline('missing declared generated module')
    providers = {n: p for n, p in modules.items() if p['kind'] == 'native'}
    for name in generated:
        if not providers:
            raise QualificationDecline('generated native module has no pinned providers')
        # Engineering linkage to the complete loaded native extension set. This
        # does not assert which image created a previously loaded Cython object.
        modules[name]['providers'] = providers
    files[str(Path(sys.executable).resolve())] = _hash(sys.executable)
    settings = {k: os.environ.get(k) for k in (
        'ANYGEOMETRY_PLANNING_MODE', 'ANYGEOMETRY_SEAM_POLICY', 'ANYGEOMETRY_PLANNING_WORKERS',
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    if len(sys.modules) != len(loaded) or any(sys.modules.get(n) is not m for n, m in loaded):
        raise QualificationDecline('module closure changed during snapshot')
    return dict(schema='PP1b-v1', modules=modules, files=files, versions=versions,
                python=dict(version=sys.version, executable=str(Path(sys.executable).resolve()),
                            implementation=sys.implementation.name, platform=sys.platform),
                cpu=cpu, blas=sorted(blas, key=lambda x: x['filepath']),
                types=allowed_input_names, settings=settings,
                reset_registry={n: _type_name(type(v)) for n, v in (declined_stores or {}).items()})


def verify_environment(expected, observed):
    try:
        if expected['schema'] != 'PP1b-v1' or observed['schema'] != 'PP1b-v1':
            raise QualificationDecline('unknown pin schema')
        for pin in (expected, observed):
            if not pin['blas'] or any(type(p['num_threads']) is not int or p['num_threads'] != 1
                                      for p in pin['blas']):
                raise QualificationDecline('BLAS/OpenMP thread count must be one')
        if expected['modules'].keys() != observed['modules'].keys():
            raise QualificationDecline('unpinned/missing module')
        if expected['files'] != observed['files'] or expected['modules'] != observed['modules']:
            raise QualificationDecline('code/native hash or module identity mismatch')
        for key in ('cpu', 'blas', 'versions', 'python', 'types', 'settings', 'reset_registry'):
            if expected[key] != observed[key]:
                raise QualificationDecline(key + ' pin mismatch')
    except (KeyError, TypeError, AttributeError) as exc:
        raise QualificationDecline('malformed environment pin') from exc


def check_types(value, pin, *, limit=100000):
    """Typed task-input adapter: opaque objects require a future explicit codec."""
    allowed = frozenset(pin['types'])
    exact_types = _input_types()
    seen, pending = set(), [value]
    while pending:
        item = pending.pop()
        typ = type(item)
        if not any(typ is candidate for candidate in exact_types) or _type_name(typ) not in allowed:
            raise QualificationDecline('unpinned input type: ' + _type_name(typ))
        if typ in (type(None), bool, int, float, complex, str, bytes):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen) > limit:
            raise QualificationDecline('task type walk truncated')
        if typ in (tuple, list, set, frozenset):
            pending.extend(item)
        elif typ is dict:
            pending.extend(item.keys()); pending.extend(item.values())
        else:
            import numpy as np
            if typ is np.ndarray and not item.dtype.hasobject:
                continue
            if typ in (np.float64, np.int64):
                continue
            from .model import GeometryModel
            if typ is GeometryModel:
                from .serialization import to_dict
                pending.append(to_dict(item))
                continue
            from dataclasses import fields, is_dataclass
            if is_dataclass(item):
                for field in fields(item):
                    part = getattr(item, field.name)
                    if field.metadata.get('definition') is False:
                        if part is not None:
                            raise QualificationDecline('warm input instance cache is not admitted')
                    else:
                        pending.append(part)
                continue
            raise QualificationDecline('unsupported pinned type adapter: ' + _type_name(typ))


_CLEAR = {dict: dict.clear, list: list.clear, set: set.clear,
          OrderedDict: OrderedDict.clear, deque: deque.clear,
          weakref.WeakKeyDictionary: weakref.WeakKeyDictionary.clear,
          weakref.WeakValueDictionary: weakref.WeakValueDictionary.clear,
          weakref.WeakSet: weakref.WeakSet.clear}
_LRU = functools._lru_cache_wrapper


def reset_declined(stores, *, owned_worker):
    """All declared declined stores reset, or the task does not run.

    Preflight the entire mapping before any mutation. No metadata dictionary or
    authority cache is inferred to be disposable from its name/type. The caller
    supplies the complete reviewed worker-task registry, and fresh task inputs
    must be constructed after reset. Unknown reset semantics decline.
    """
    if owned_worker is not True or threading.active_count() != 1:
        raise QualificationDecline('reset requires an owned single-thread worker')
    if type(stores) is not dict or any(type(n) is not str for n in stores):
        raise QualificationDecline('invalid reset registry')
    for name, store in stores.items():
        if type(store) not in _CLEAR and type(store) is not _LRU:
            raise QualificationDecline('unsupported reset store: ' + name)
    result = {}
    try:
        for name, store in stores.items():
            if type(store) is _LRU:
                _LRU.cache_clear(store)
                size = _LRU.cache_info(store).currsize
            else:
                _CLEAR[type(store)](store)
                size = len(store)
            if size:
                raise QualificationDecline('reset postcondition failed: ' + name)
            result[name] = size
    except Exception as exc:
        raise QualificationDecline('reset failed; discard owned worker') from exc
    return result


class WorkerGate:
    """Unwired owned-worker lifecycle gate; no result before all controls pass."""
    def __init__(self, expected, *, stores, owned_worker, observe=None, generated_modules=(), ledger=None):
        # Snapshot the expected reference so later caller mutation cannot bless
        # a failing candidate. Injection of observe is for isolated unit tests.
        self.expected = json.loads(json.dumps(expected, allow_nan=False))
        self.stores = dict(stores)
        if self.expected.get('reset_registry') != {n: _type_name(type(v)) for n, v in self.stores.items()}:
            raise QualificationDecline('reset registry does not cover the pinned declined stores')
        self.owned_worker = owned_worker
        self.ledger = ledger
        if ledger is not None and (set(ledger.resets) != set(self.stores) or
                any(ledger.resets[n] is not v for n, v in self.stores.items())):
            raise QualificationDecline('ledger reset binding differs from worker registry')
        self.observe = observe or (lambda: capture_environment(
            generated_modules=generated_modules, declined_stores=self.stores))
        self.ready = False
        self.poisoned = False
        self.reset_counts = {}

    def _before(self, task):
        if self.poisoned:
            raise QualificationDecline('worker gate is poisoned')
        verify_environment(self.expected, self.observe())
        if type(task) is not types.FunctionType:
            raise QualificationDecline('unpinned task callable type')
        module = sys.modules.get(task.__module__)
        if (task.__module__ not in self.expected['modules'] or module is None or
                pin_module(module) != self.expected['modules'][task.__module__]):
            raise QualificationDecline('unpinned task callable module/hash')
        self.reset_counts = (self.ledger.prepare(owned_worker=self.owned_worker) if self.ledger is not None
                             else reset_declined(self.stores, owned_worker=self.owned_worker))

    def start(self, canaries, expected_digests):
        if self.ready or self.poisoned:
            raise QualificationDecline('startup may run only once')
        try:
            if set(canaries) != {'pair', 'face'} or set(expected_digests) != set(canaries):
                raise QualificationDecline('both pair and face canaries are required')
            for name in sorted(canaries):
                task = canaries[name]
                self._before(task)
                actual = task()
                if type(actual) is not str or not actual or actual != expected_digests[name]:
                    raise QualificationDecline('canary mismatch: ' + name)
                verify_environment(self.expected, self.observe())
            self.ready = True
        except BaseException as exc:
            self.poisoned = True
            raise QualificationDecline('startup canary/admission declined: ' + str(exc)) from exc

    def run(self, task, *args, **kwargs):
        if not self.ready or self.poisoned:
            raise QualificationDecline('startup canary not qualified')
        try:
            self._before(task)
            check_types((args, kwargs), self.expected)
            result = task(*args, **kwargs)
            verify_environment(self.expected, self.observe())
            return result
        except BaseException:
            self.poisoned = True
            self.ready = False
            raise


def check_ci_equality(serial, parallel):
    """PP6 hook: compare plan AND applied-model digests under one projection."""
    if (type(serial) is not tuple or type(parallel) is not tuple or
            len(serial) != 2 or len(parallel) != 2 or
            any(type(x) is not str or not x for x in (*serial, *parallel))):
        raise QualificationDecline('CI requires nonempty plan and applied-model digests')
    if serial[0] != parallel[0]:
        raise QualificationDecline('CI plan digest mismatch')
    if serial[1] != parallel[1]:
        raise QualificationDecline('CI applied-model digest mismatch')
