"""PP1g risk-accepted qualification, not a purity proof; strict mode retained.

Definition metadata is protected, not cleared. Reviewed numerical cache roots
are disposable only inside a fresh owned worker, before fresh task input creation.
Unknown stores and active contexts decline before any mutation. This policy is
bound to the pinned source/runtime and finite canaries, not a universal proof.
"""
import contextvars
import dataclasses
import enum
import functools
import gc
import hashlib
import marshal
import struct
import sys
import time
import types
import typing
import weakref

from ._planning_purity import inventory
from ._planning_qualification import QualificationDecline, reset_declined

_NUMERICAL_ROOTS = frozenset((
    '_GAUSS_LEGENDRE', '_BRANCH_CONSTANTS', '_PI_CACHE', '_W_FACTORS', '_ISO_CACHE',
    '_SECTION_CANONICAL', '_ISSUED_CONTEXTS', '_SECTIONS', '_NETS', '_ABSOLUTE',
    '_SCALED', '_MEMO', '_FIRST_CHOICE', '_PATCHES', '_BRANCH_POLYNOMIALS'))
_FIXED_ROOTS = (
    'anygeometry.backend._TRUE',
    'anygeometry.batch_intersections.SEAM_POLICY_VERSIONS',
    'anygeometry.batch_intersections._CARRIER_FIELDS', 'anygeometry.batch_intersections._CARRIER_ROWS',
    'anygeometry.identity._ENTITY_KIND_ORDER', 'anygeometry.nurbs._CURVE_FIELDS',
    'anygeometry.nurbs._SURFACE_FIELDS', 'anygeometry.nurbs._ON_SURFACE_FIELDS',
    'anygeometry.nurbs_surface_intersection._topology._PLANE',
    'anygeometry.prepared_mixed_sheet_network._SURFACE_FAMILIES',
    'anygeometry.quadric_algebra._ONE_PLUS_X2', 'anygeometry.serialization._RECORD_VERSIONS',
    'anygeometry.serialization._CURRENT_REQUIRED_FIELDS', 'anygeometry.serialization._CURRENT_OPTIONAL_FIELDS',
    'anygeometry.serialization._V3_REQUIRED_FIELDS', 'anygeometry.serialization._RULED_FIELDS',
    'anygeometry.serialization._QUADRIC_FIELDS', 'anygeometry.serialization._ELLIPTIC_QUADRIC_FIELDS',
    'anygeometry.serialization._QUADRIC_CURVE_FIELDS', 'anygeometry.serialization._BEZIER_CURVE_FIELDS',
    'anygeometry.support_curves._RECORD_FIELDS', 'anygeometry.support_curves._BRANCH_FIELDS',
    'anygeometry.support_curves._PLANE_FIELDS', 'anygeometry.surfaces._LSTSQ_GUFUNC',
    'anygeometry._planning_qualification._CLEAR',
)
_CONTAINER_CONSTANT_ROOTS = (
    'anygeometry.quadric_algebra._ONE_PLUS_X2',
    'anygeometry.batch_intersections._CARRIER_FIELDS',
)
# Exact observed runtime stores; never erase implementation bindings to admit a task.
_RUNTIME_EMPTY_PATHS = (
    'anygeometry.backend._RUNTIME_PYTHON_GLOBALS',
    'anygeometry.backend._RUNTIME_PYTHON_GLOBALS.__builtins__',
    'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[0][1].__dict__',
    'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[12][1]._finfo_cache',
    'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[14][1].__dict__',
    'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[18][1].__dict__',
    'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[19][1].__dict__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[3][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[3][0].__all__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[3][0].array_function_dispatch.keywords',
    'anygeometry.backend._RUNTIME_POINT_EDGES[33][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[33][0].__all__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[33][0].array_function_dispatch.keywords',
    'anygeometry.backend._RUNTIME_POINT_EDGES[33][0]._block_check_depths_match.__defaults__[0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[52][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[52][0].matmul.__dict__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[52][0].multiply.__dict__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[52][0].__all__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[52][0].einsum_symbols_set',
    'anygeometry.backend._RUNTIME_POINT_EDGES[61][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[61][0].__all__',
    'anygeometry.backend._RUNTIME_POINT_EDGES[61][0].POSSIBLE_FLAGS',
    'anygeometry.nurbs_surface_intersection._topology._PLANE.__dict__',
    'anygeometry.surfaces._LSTSQ_GUFUNC.__dict__',
)
_ACTIONS = frozenset(('reset', 'covered-by-reset', 'pinned-definition', 'require-live-binding', 'trusted-runtime-image', 'require-empty', 'decline'))


# Exact reviewed constant stores. Their contents, not just root identities, are
# checked before/after tasks. No path-prefix or arbitrary opaque-object adapter.
_LIVE_CONTENT_PATHS = tuple(path for path in _RUNTIME_EMPTY_PATHS if path.endswith((
    '.__all__', '.keywords', '.einsum_symbols_set', '.POSSIBLE_FLAGS',
    '._PLANE.__dict__')) or (path.endswith('.__dict__') and
    ('_RUNTIME_TRUSTED_NUMPY' in path or '.matmul.' in path or '.multiply.' in path)))
_LIVE_NAMESPACE_PATH = 'anygeometry.backend._RUNTIME_PYTHON_GLOBALS'
_DECLINED_RUNTIME_PATHS = (
    'anygeometry.backend._RUNTIME_PYTHON_GLOBALS.__builtins__',
    'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[12][1]._finfo_cache',
    'anygeometry.backend._RUNTIME_POINT_EDGES[3][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[33][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[52][0]',
    'anygeometry.backend._RUNTIME_POINT_EDGES[61][0]',
)


# Owner-accepted trusted runtime image, pinned by versions and binary hashes,
# not purity-proven. Exact canonical inventory paths, no prefix admission.
_TRUSTED_IMAGE_STORES = _DECLINED_RUNTIME_PATHS + (
    'anygeometry.exact_curves._BranchConstants.__annotate_func__.cell[0]',)


def _registered_namespace(value):
    if type(value) is not dict or type(value.get('__name__')) is not str:
        return None
    module = sys.modules.get(value['__name__'])
    return module if type(module) is types.ModuleType and vars(module) is value else None


def live_namespace_snapshot(value):
    """Binding-only attestation, plus actual function/code identities and hashes.

    This does not certify the transitive state of imported modules, builtins,
    loaders or specs. Those remain independently pinned/declined by the ledger
    and worker environment. Code bytes are reference measurements, not a proof
    that an in-memory function was compiled from the source file.
    """
    from ._planning_qualification import pin_module
    module = _registered_namespace(value)
    if module is None or any(type(name) is not str for name in value):
        raise QualificationDecline('live namespace has no exact registered module')
    source_pin = pin_module(module)
    bindings, pins = [], {}
    for name, item in sorted(value.items()):
        binding = [name, id(item), id(type(item))]
        if type(item) is types.FunctionType:
            if item.__globals__ is not value or item.__module__ != module.__name__:
                raise QualificationDecline('foreign live function globals/module: ' + name)
            code_hash = hashlib.sha256(marshal.dumps(item.__code__)).hexdigest()
            binding.extend((id(item.__code__), id(item.__globals__), code_hash))
            pins[name] = dict(module=module.__name__, qualname=item.__qualname__,
                              sha256=code_hash, source=source_pin)
        bindings.append(tuple(binding))
    if not pins:
        raise QualificationDecline('live namespace has no pinned Python implementation')
    return content_digest((id(module), bindings)), pins


def check_coverage(report, rows):
    expected = {r.path: r.kind for r in report.stores}
    if report.truncated or set(rows) != set(expected):
        raise QualificationDecline('ledger coverage is incomplete/truncated')
    for path, row in rows.items():
        if row['kind'] != expected[path] or row['action'] not in _ACTIONS:
            raise QualificationDecline('ledger coverage/type decision mismatch: ' + path)


def _metadata_parts(item):
    """Explicit static metadata schemas only; never a general instance adapter."""
    import numpy as np
    from .transactions import _MISSING
    from .tolerance import TolerancePolicy
    from .audit import AuditMetrics
    from .predicates import IntersectionQualificationPolicy
    from .structural import FrozenMetadata, ParameterRange, SheetTopologyPolicy
    from fractions import Fraction
    typ = type(item)
    sentinels = (('dataclasses.MISSING', dataclasses.MISSING),
                 ('numpy._NoValue', np._NoValue), ('transactions._MISSING', _MISSING),
                 ('dataclasses._FIELD', dataclasses._FIELD),
                 ('dataclasses._FIELD_CLASSVAR', dataclasses._FIELD_CLASSVAR),
                 ('dataclasses._FIELD_INITVAR', dataclasses._FIELD_INITVAR))
    for name, sentinel in sentinels:
        if item is sentinel:
            try:
                state = object.__getattribute__(item, '__dict__')
            except AttributeError:
                state = None
            return (name, state)
    if typ is dataclasses.Field or typ is dataclasses._DataclassParams:
        return tuple((name, types.MemberDescriptorType.__get__(
            type.__getattribute__(typ, '__dict__')[name], item, typ)) for name in typ.__slots__)
    if typ is types.MappingProxyType:
        backing = gc.get_referents(item)
        if len(backing) != 1 or type(backing[0]) is not dict:
            raise QualificationDecline('metadata mappingproxy backing is not a plain dictionary')
        return (backing[0],)
    if typ is types.GenericAlias:
        return (item.__origin__, item.__args__, item.__parameters__)
    if typ is types.UnionType:
        return (item.__args__,)
    if typ is property:
        return (item.fget, item.fset, item.fdel, item.__doc__)
    if typ is Fraction:
        return (object.__getattribute__(item, '_numerator'),
                object.__getattribute__(item, '_denominator'))
    if any(typ is candidate for candidate in (type(typing.List[int]), type(typing.List),
            type(typing.Callable[[int], str]), type(typing.Annotated[int, 'metadata']))):
        return (object.__getattribute__(item, '__dict__'),)
    if any(base is enum.Enum for base in type.__getattribute__(typ, '__mro__')):
        return (typ, object.__getattribute__(item, '__dict__'))
    if any(typ is candidate for candidate in (TolerancePolicy, AuditMetrics,
            IntersectionQualificationPolicy, FrozenMetadata, ParameterRange, SheetTopologyPolicy)):
        return tuple((field.name, object.__getattribute__(item, field.name))
                     for field in dataclasses.fields(typ))
    return None


def content_digest(value):
    """Bounded, typed content encoding. Unsupported mutable/native leaves refuse.

    Symbol identities are process-local references, not a bytecode attestation.
    Their implementation provenance remains the file pin (F7 remains open).
    """
    seen, chunks, held = {}, [], []

    def emit(raw):
        chunks.append(len(raw).to_bytes(8, 'big') + raw)

    def visit(item, depth=0):
        if depth > 40 or len(chunks) > 100000:
            raise QualificationDecline('protected digest exceeds bound')
        typ = type(item)
        emit((type.__getattribute__(typ, '__module__') + '.' +
              type.__getattribute__(typ, '__qualname__')).encode())
        if item is None or item is Ellipsis or item is NotImplemented:
            return
        if typ in (str, bytes, int, bool):
            emit(item if typ is bytes else str(item).encode('utf-8', 'surrogatepass'))
        elif typ in (float, complex):
            emit(struct.pack('>d', item) if typ is float else struct.pack('>dd', item.real, item.imag))
        elif (any(base is type for base in type.__getattribute__(typ, '__mro__')) or
              typ in (types.FunctionType, types.BuiltinFunctionType,
                      types.MethodDescriptorType, types.WrapperDescriptorType,
                      types.GetSetDescriptorType, types.MemberDescriptorType)):
            emit(str(id(item)).encode())
            if typ is types.FunctionType:
                emit(str(id(item.__code__)).encode())
                emit(str(id(item.__globals__)).encode())
        elif typ in (dict, list, tuple, set, frozenset):
            if id(item) in seen:
                emit(b'ref' + str(seen[id(item)]).encode())
                return
            seen[id(item)] = len(seen)
            emit(str(len(item)).encode())
            if typ is dict:
                for key, part in dict.items(item):
                    visit(key, depth + 1); visit(part, depth + 1)
            elif typ in (set, frozenset):
                # Separate bounded encodings avoid relying on hash iteration order.
                for digest in sorted(content_digest(part) for part in item):
                    emit(digest.encode())
            else:
                for part in item:
                    visit(part, depth + 1)
        else:
            parts = _metadata_parts(item)
            if parts is None:
                raise QualificationDecline('unsupported protected content: ' + type.__getattribute__(typ, '__name__'))
            if id(item) in seen:
                emit(b'ref' + str(seen[id(item)]).encode())
                return
            seen[id(item)] = len(seen)
            # The structural tuples are temporary. Retain them until encoding
            # ends so CPython cannot recycle an ID already in the alias table.
            held.append(parts)
            visit(parts, depth + 1)
    visit(value)
    return hashlib.sha256(b''.join(chunks)).hexdigest()


class Ledger:
    def __init__(self, report, objects, rows, modules, *, trusted_image=False):
        check_coverage(report, rows)
        self.report, self.objects, self.rows = report, objects, rows
        self.modules = modules
        self.trusted_image = trusted_image
        self.registered = {m.__name__: m for m in modules if sys.modules.get(m.__name__) is m}
        self.resets = {p: objects[p] for p, row in rows.items() if row['action'] == 'reset'}
        self.costs = []
        self.census_costs = []
        self.digests = {}
        self.live_digests = {}
        self.live_references = {}
        for path, row in rows.items():
            if row['action'] == 'require-live-binding' and path == _LIVE_NAMESPACE_PATH:
                try:
                    value = objects[path]
                    # ID equality is meaningful only while every attested object
                    # remains alive, including code replaced on a live function.
                    self.live_references[path] = (_registered_namespace(value),
                        tuple(value.items()), tuple(item.__code__ for item in value.values()
                                                   if type(item) is types.FunctionType))
                    self.live_digests[path] = live_namespace_snapshot(value)[0]
                except QualificationDecline as exc:
                    row.update(action='decline', reason=str(exc))
            elif row['action'] in ('pinned-definition', 'require-live-binding'):
                try:
                    self.digests[path] = content_digest(objects[path])
                except QualificationDecline as exc:
                    row.update(action='decline', reason=str(exc))

    def live_code_pins(self):
        return {path: live_namespace_snapshot(self.objects[path])[1]
                for path in self.live_digests}

    def validate(self):
        started = time.perf_counter()
        if any(sys.modules.get(name) is not module for name, module in self.registered.items()):
            raise QualificationDecline('ledger module binding changed')
        objects = {}
        report = inventory(self.modules, store_objects=objects)
        if report.truncated:
            raise QualificationDecline('live census truncated')
        roots = tuple(self.resets)
        def beneath_reset(path):
            return any(path.startswith(root + '.') or path.startswith(root + '[') for root in roots)
        for alias, target in report.aliases:
            if beneath_reset(target) and not beneath_reset(alias):
                raise QualificationDecline('reset descendant has external alias: ' + alias)
        before = {path for path in self.objects if not beneath_reset(path)}
        after = {path for path in objects if not beneath_reset(path)}
        if before != after:
            raise QualificationDecline('live store set changed: ' + str(sorted(before ^ after)[:3]))
        for path in before:
            if objects[path] is not self.objects[path] or type(objects[path]) is not type(self.objects[path]):
                raise QualificationDecline('live store rebound: ' + path)
        for path, expected in self.digests.items():
            if content_digest(objects[path]) != expected:
                raise QualificationDecline('protected store content changed: ' + path)
        for path, expected in self.live_digests.items():
            if live_namespace_snapshot(objects[path])[0] != expected:
                raise QualificationDecline('live binding/code changed: ' + path)
        self.census_costs.append(time.perf_counter() - started)
        return objects

    def prepare(self, *, owned_worker):
        check_coverage(self.report, self.rows)
        objects = self.validate()
        # Refusal checks precede all mutations, including independent caches.
        for path, row in self.rows.items():
            if row['action'] == 'decline':
                raise QualificationDecline('declined ledger store: ' + path)
            if row['action'] == 'require-empty':
                value = objects[path]
                if type(value) is contextvars.ContextVar:
                    try:
                        value = value.get()
                    except LookupError:
                        value = None
                    if path == 'anygeometry.arrangement_geometry._JUNCTION_BOUNDS' and value is False:
                        # Declared default False, guarded by a restoring context
                        # manager: inactive switch, not an active work/budget token.
                        continue
                    if value is not None and not (type(value) in (tuple, dict, list, set) and len(value) == 0):
                        raise QualificationDecline('active context declines: ' + path)
                elif len(value):
                    raise QualificationDecline('active recursion/default state declines: ' + path)
        start = time.perf_counter()
        counts = reset_declined({p: objects[p] for p in self.resets}, owned_worker=owned_worker)
        self.costs.append(time.perf_counter() - start)
        return counts

    def record(self):
        return dict(stores=list(self.rows.values()), aliases=list(self.report.aliases),
                    unknowns=list(self.report.unknowns), truncated=self.report.truncated,
                    reset_seconds=list(self.costs), census_seconds=list(self.census_costs),
                    live_code_pins=self.live_code_pins())


def build_ledger(report, objects, *, modules=None, trusted_image=False):
    if type(trusted_image) is not bool:
        raise QualificationDecline('trusted-image selection must be explicit bool')
    if set(objects) != {r.path for r in report.stores}:
        raise QualificationDecline('ledger handle coverage mismatch')
    if modules is None:
        names = set()
        for path in objects:
            matches = [name for name, module in sys.modules.items()
                       if type(module) is types.ModuleType and path.startswith(name + '.')]
            if matches:
                names.add(max(matches, key=len))
        modules = tuple(sys.modules[name] for name in sorted(names))
    if not modules:
        raise QualificationDecline('ledger requires explicit live module roots')
    definitions, fresh = {}, {}
    census = inventory(modules, store_objects=fresh, definition_objects=definitions)
    if census.truncated or set(fresh) != set(objects) or any(fresh[p] is not v for p, v in objects.items()):
        raise QualificationDecline('ledger creation live coverage mismatch')
    # Two source-reviewed constant container trees. Record actual container IDs,
    # not a path prefix; callables/classes and their defaults are not traversed.
    constant_ids = set()
    def constant_containers(value):
        if type(value) not in (dict, list, tuple, set, frozenset) or id(value) in constant_ids:
            return
        constant_ids.add(id(value))
        items = tuple(value.keys()) + tuple(value.values()) if type(value) is dict else value
        for item in items:
            constant_containers(item)
    for path in _CONTAINER_CONSTANT_ROOTS:
        if path in objects:
            constant_containers(objects[path])
    rows = {}
    reset_roots = []
    # First choose concrete cache roots. Nested weak-store implementation dicts
    # are covered by the owner's clear operation, never independently cleared.
    for row in report.stores:
        path, value = row.path, objects[row.path]
        parts = path.split('.')
        kind = type(value)
        if trusted_image and path in _TRUSTED_IMAGE_STORES and kind is dict:
            action = 'trusted-runtime-image'
        elif kind in (functools._lru_cache_wrapper, weakref.WeakKeyDictionary,
                      weakref.WeakValueDictionary, weakref.WeakSet):
            action = 'reset'
        elif (any(path == m.__name__ + '.' + parts[-1] for m in modules)
              and parts[-1] in _NUMERICAL_ROOTS and kind in (dict, list, set)):
            action = 'reset'
        elif path == _LIVE_NAMESPACE_PATH and _registered_namespace(value) is not None:
            action = 'require-live-binding'
        elif path in _LIVE_CONTENT_PATHS:
            action = 'require-live-binding'
        elif path in _DECLINED_RUNTIME_PATHS and len(value):
            action = 'decline'
        elif (path in _RUNTIME_EMPTY_PATHS or kind is contextvars.ContextVar or
              '.__repr__.cell[' in path or '.deepcopy.__defaults__' in path):
            action = 'require-empty'
        elif id(value) in definitions or id(value) in constant_ids or path in _FIXED_ROOTS:
            action = 'pinned-definition'
        else:
            action = 'decline'
        if action == 'reset':
            reset_roots.append(path)
        rows[path] = dict(path=path, kind=row.kind, action=action,
                         reason=('trusted runtime image, pinned by versions and binary hashes, not purity-proven'
                                 if action == 'trusted-runtime-image' else
                                 'PP1f explicit opaque runtime decline; transitive mutable state not qualified'
                                 if path in _DECLINED_RUNTIME_PATHS and action == 'decline' else
                                 'PP1f exact binding/content guard; prior reset/static/empty policy retained'))
    for path, row in rows.items():
        for parent in reset_roots:
            if path != parent and (path.startswith(parent + '.') or path.startswith(parent + '[')):
                row.update(action='covered-by-reset', parent=parent)
                break
    return Ledger(report, objects, rows, modules, trusted_image=trusted_image)


def loaded_geometry_ledger(*, trusted_image=False):
    modules = tuple(m for name, m in sorted(sys.modules.items())
                    if name == 'anygeometry' or name.startswith('anygeometry.'))
    objects = {}
    report = inventory(modules, store_objects=objects)
    return build_ledger(report, objects, modules=modules, trusted_image=trusted_image)
