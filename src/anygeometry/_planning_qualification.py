"""PP1g risk-accepted qualification, not a purity proof; serial remains untouched.

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
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import types
import weakref

from .errors import GeometryError

# The opt-in selector chooses host dispatch, not numerical task semantics. A
# captured worker baseline is valid before/after enabling that selector. Actual
# mode, seam, worker setting and native thread variables remain pinned.
NUMERICAL_SETTINGS = (
    'ANYGEOMETRY_PLANNING_MODE', 'ANYGEOMETRY_SEAM_POLICY', 'ANYGEOMETRY_PLANNING_WORKERS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


class QualificationDecline(GeometryError):
    """Reject acceleration; the original serial path retains its own authority."""


def _hash(path):
    try:
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    except (OSError, ValueError) as exc:
        raise QualificationDecline('unreadable pinned hash: ' + str(path)) from exc


def runtime_files():
    """Pin this process's interpreter/runtime images, never another process.

    Windows builtin/frozen code lives in the DLL at sys.dllhandle. Resolve loaded
    VCRuntime/limited-ABI images by handle too; do not guess from the launcher.
    Non-Windows runtime provenance needs a separate reviewed adapter.
    """
    import sys as runtime_sys
    import ctypes
    if runtime_sys.platform != 'win32':
        raise QualificationDecline('interpreter runtime adapter is Windows-only')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_handle = kernel.GetModuleHandleW
    get_handle.argtypes, get_handle.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    get_path = kernel.GetModuleFileNameW
    get_path.argtypes, get_path.restype = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32], ctypes.c_uint32
    handles = [runtime_sys.dllhandle]
    crt = 0
    for name in ('python3.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'):
        handle = get_handle(name)
        if handle:
            handles.append(handle)
            crt += name.startswith('vcruntime')
    if not crt:
        raise QualificationDecline('loaded VCRuntime DLL is missing')
    result = {}
    for handle in handles:
        buffer = ctypes.create_unicode_buffer(32768)
        size = get_path(handle, buffer, len(buffer))
        if not size or size >= len(buffer):
            raise QualificationDecline('runtime DLL path is unresolved/truncated')
        path = str(Path(buffer.value).resolve())
        result[path] = _hash(path)
    return result


def pin_module(module, *, runtime=None):
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
        return dict(kind=spec.origin, runtime_files=runtime_files() if runtime is None else runtime)
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


def capture_environment(*, generated_modules=(), declined_stores=None, trusted_image=False):
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
    runtime = runtime_files()
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
            pin = pin_module(module, runtime=runtime)
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
    files.update(runtime)
    settings = {k: os.environ.get(k) for k in NUMERICAL_SETTINGS}
    if len(sys.modules) != len(loaded) or any(sys.modules.get(n) is not m for n, m in loaded):
        raise QualificationDecline('module closure changed during snapshot')
    result = dict(schema='PP1b-v1', modules=modules, files=files, versions=versions,
                python=dict(version=sys.version, executable=str(Path(sys.executable).resolve()),
                            implementation=sys.implementation.name, platform=sys.platform),
                cpu=cpu, blas=sorted(blas, key=lambda x: x['filepath']),
                types=allowed_input_names, settings=settings,
                reset_registry={n: _type_name(type(v)) for n, v in (declined_stores or {}).items()})
    if trusted_image:
        result['trusted_runtime_image'] = trusted_runtime_record(result)
    return result


def trusted_runtime_record(pin):
    """Risk-accepted qualification, not a purity proof or in-memory-page hash."""
    from ._planning_ledger import trusted_store_identities
    try:
        binaries, runtime = {}, {}
        families = set()
        for name, module in pin['modules'].items():
            runtime.update(module.get('runtime_files', {}))
            if module['kind'] == 'native':
                binaries[module['path']] = module['sha256']
                if name.startswith('numpy.'):
                    families.add('numpy')
                if name.startswith('scipy.'):
                    families.add('scipy')
        if families != {'numpy', 'scipy'} or not runtime:
            raise QualificationDecline('trusted runtime binary families/interpreter images missing')
        binaries.update(runtime)
        executable = pin['python']['executable']
        binaries[executable] = pin['files'][executable]
        for pool in pin['blas']:
            binaries[pool['filepath']] = pool['sha256']
        if any(type(value) is not str or not value for value in
               (pin['versions']['numpy'], pin['versions']['scipy'], pin['python']['version'])):
            raise QualificationDecline('trusted runtime versions missing')
        for path, digest in binaries.items():
            if (type(digest) is not str or len(digest) != 64 or
                    any(char not in '0123456789abcdef' for char in digest) or pin['files'].get(path) != digest):
                raise QualificationDecline('trusted runtime binary hash missing/inconsistent')
        return dict(qualification='RISK-ACCEPTED QUALIFICATION, not a purity proof',
            versions=dict(pin['versions']), python=dict(pin['python']),
            binary_hashes=dict(sorted(binaries.items())),
            stores=trusted_store_identities())
    except (KeyError, TypeError, AttributeError) as exc:
        raise QualificationDecline('malformed trusted runtime image') from exc


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
        if expected.get('trusted_runtime_image') != observed.get('trusted_runtime_image'):
            raise QualificationDecline('trusted runtime image pin mismatch')
        for pin in (expected, observed):
            if 'trusted_runtime_image' in pin and pin['trusted_runtime_image'] != trusted_runtime_record(pin):
                raise QualificationDecline('invalid trusted runtime image record')
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
            raise QualificationDecline('task type walk truncated limit='+str(limit))
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
                state = object.__getattribute__(item, '__dict__')
                # Constructor-defined schema; no runtime instance can extend it.
                pristine = GeometryModel(model_id='00000000-0000-0000-0000-000000000001')
                constructor_keys = set(object.__getattribute__(pristine, '__dict__'))
                if set(state) not in (constructor_keys, constructor_keys | {'_serialization_extensions'}):
                    raise QualificationDecline('non-field model instance state')
                model_extension_record(item,limit=limit)
                for name in ('_arc_cache', '_edge_length_cache', '_trim_loops_cache',
                             '_topology_corner_cache', '_chain_arrays', '_change_hooks'):
                    if type(state[name]) not in (dict, list) or len(state[name]):
                        raise QualificationDecline('warm model instance cache: ' + name)
                if state['_spatial_index'] is not None or state['_transaction_journal'] is not None:
                    raise QualificationDecline('active model instance state')
                _check_model_record_state(state,limit=limit)
                from .serialization import to_dict
                pending.append(to_dict(item))
                continue
            from dataclasses import fields, is_dataclass
            if is_dataclass(item):
                fields_here = fields(item)
                try:
                    state = object.__getattribute__(item, '__dict__')
                except AttributeError:
                    state = {}
                if set(state) - {field.name for field in fields_here}:
                    raise QualificationDecline('non-field input instance state')
                for field in fields_here:
                    part = getattr(item, field.name)
                    if field.metadata.get('definition') is False:
                        if part is not None:
                            raise QualificationDecline('warm input instance cache is not admitted')
                    else:
                        pending.append(part)
                continue
            raise QualificationDecline('unsupported pinned type adapter: ' + _type_name(typ))


def model_extension_record(model, *, limit=100000):
    """PP2 exact decoder-owned JSON state, without stripping or evaluating it.

    Extension containers must form a tree: no cycles/repeated container aliases,
    and no shared container with other declarative model state. JSON scalars are
    exact builtin None/bool/int/finite float/str; objects/subclasses refuse. The
    digest is transport evidence, not a replacement for the original live gate.
    """
    import math
    from dataclasses import fields, is_dataclass
    from .model import GeometryModel
    if type(model) is not GeometryModel:
        raise QualificationDecline('extension adapter requires exact model type')
    state = object.__getattribute__(model, '__dict__')
    if '_serialization_extensions' not in state:
        return dict(present=False, sha256=None)
    extension = state['_serialization_extensions']
    if type(extension) is not dict:
        raise QualificationDecline('extension state must be a plain dict')
    containers = set()
    pending = [(extension, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop(); nodes += 1
        if depth > 40 or nodes > limit:
            raise QualificationDecline('extension schema exceeds bound depth=40 limit='+str(limit))
        typ = type(item)
        if typ in (dict, list):
            if id(item) in containers:
                raise QualificationDecline('extension container alias/cycle')
            containers.add(id(item))
            if typ is dict:
                if any(type(key) is not str for key in item):
                    raise QualificationDecline('extension key must be an exact string')
                pending.extend((part, depth+1) for part in item.values())
            else:
                pending.extend((part, depth+1) for part in item)
        elif typ not in (type(None), bool, int, float, str) or (typ is float and not math.isfinite(item)):
            raise QualificationDecline('unknown/nonfinite extension value')
    pending = [v for k,v in state.items() if k != '_serialization_extensions']
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in containers:
            raise QualificationDecline('extension alias into model state')
        if id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen)>limit:
            raise QualificationDecline('extension alias walk exceeds bound limit='+str(limit))
        typ = type(item)
        if typ is dict:
            pending.extend(item.keys()); pending.extend(item.values())
        elif typ in (tuple, list, set, frozenset):
            pending.extend(item)
        elif not isinstance(item, type) and is_dataclass(item):
            pending.extend(object.__getattribute__(item,f.name) for f in fields(item))
    payload = json.dumps(extension, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return dict(present=True, sha256=hashlib.sha256(payload).hexdigest())


def _check_model_record_state(state, *, limit=100000):
    """Audit original records before serialization can omit instance extras.

    This is an adapter check, not a model transport. The runtime payload boundary
    still admits only JSON data, never a pre-existing GeometryModel instance.
    """
    from ._planning_constructor import audit
    return audit(state,limit=limit)


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
    """Risk-accepted qualification, not a purity proof; no result before controls pass."""
    def __init__(self, expected, *, stores, owned_worker, observe=None, generated_modules=(), ledger=None):
        if ledger is None and observe is None:
            raise QualificationDecline('real worker gate requires a complete live ledger')
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
        if ledger is not None and self.expected.get('live_code_pins', {}) != ledger.live_code_pins():
            raise QualificationDecline('reference live code pins are missing or differ')
        trusted = ledger is not None and ledger.trusted_image
        if trusted and self.expected.get('trusted_runtime_image') != trusted_runtime_record(self.expected):
            raise QualificationDecline('trusted-image ledger lacks its reference record')
        self.observe = observe or (lambda: capture_environment(
            generated_modules=generated_modules, declined_stores=self.stores, trusted_image=trusted))
        self.ready = False
        self.poisoned = False
        self.reset_counts = {}
        self.verification_seconds = []

    def _verify(self):
        started = time.perf_counter()
        verify_environment(self.expected, self.observe())
        if self.ledger is not None and self.expected.get('live_code_pins', {}) != self.ledger.live_code_pins():
            raise QualificationDecline('reference live code pins changed')
        self.verification_seconds.append(time.perf_counter() - started)

    def _after(self):
        self._verify()
        if self.ledger is not None:
            self.ledger.validate()

    def _before(self, task):
        if self.poisoned:
            raise QualificationDecline('worker gate is poisoned')
        self._verify()
        if type(task) is not types.FunctionType:
            raise QualificationDecline('unpinned task callable type')
        module = sys.modules.get(task.__module__)
        if (task.__module__ not in self.expected['modules'] or module is None or
                pin_module(module) != self.expected['modules'][task.__module__]):
            raise QualificationDecline('unpinned task callable module/hash')
        if ('.' in task.__qualname__ or module.__dict__.get(task.__qualname__) is not task
                or task.__globals__ is not module.__dict__ or task.__closure__ is not None):
            raise QualificationDecline('task binding/globals are not the pinned top-level function')
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
                self._after()
            self.ready = True
        except BaseException as exc:
            self.poisoned = True
            raise QualificationDecline('startup canary/admission declined: ' + str(exc)) from exc

    def run(self, task, *args, **kwargs):
        # Retained no-input entry point. Live instances can contain pre-reset
        # state; arguments must cross run_json instead.
        if args or kwargs:
            self.ready, self.poisoned = False, True
            raise QualificationDecline('live inputs forbidden; use checksum-verified payload')
        return self.run_json(task, b'{"args":[],"kwargs":{}}',
                             hashlib.sha256(b'{"args":[],"kwargs":{}}').hexdigest())

    def run_json(self, task, payload, sha256):
        if not self.ready or self.poisoned:
            raise QualificationDecline('startup canary not qualified')
        try:
            self._before(task)
            args, kwargs = decode_task_payload(payload, sha256)
            check_types((args, kwargs), self.expected)
            result = task(*args, **kwargs)
            self._after()
            return result
        except BaseException:
            self.poisoned = True
            self.ready = False
            raise


def decode_task_payload(payload, sha256, *, limit=4_000_000):
    """Narrow JSON values only; PP2 owns any future model/array transport codec."""
    if (type(payload) is not bytes or len(payload) > limit or type(sha256) is not str
            or hashlib.sha256(payload).hexdigest() != sha256):
        raise QualificationDecline('payload type/size/checksum mismatch limit='+str(limit))
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise QualificationDecline('duplicate payload key')
            result[key] = value
        return result
    def constant(value):
        raise QualificationDecline('nonfinite JSON constant')
    try:
        record = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant,
                            parse_float=lambda value: float(value) if math.isfinite(float(value)) else constant(value))
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise QualificationDecline('invalid task payload') from exc
    if (type(record) is not dict or set(record) != {'args', 'kwargs'} or
            type(record['args']) is not list or type(record['kwargs']) is not dict):
        raise QualificationDecline('invalid task payload schema')
    return tuple(record['args']), record['kwargs']


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
