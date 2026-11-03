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
import json
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


# Owner-accepted opaque runtime risk with exact namespace/owner checks. The owned
# annotation dictionary additionally has a structural digest and source provenance.
_TRUSTED_IMAGE_STORES = _DECLINED_RUNTIME_PATHS + (
    'anygeometry.exact_curves._BranchConstants.__annotate_func__.cell[0]',)


def _trusted_store_values():
    from . import backend, exact_curves
    return (backend._RUNTIME_PYTHON_GLOBALS['__builtins__'],
            backend._RUNTIME_TRUSTED_NUMPY[12][1]._finfo_cache,
            *(backend._RUNTIME_POINT_EDGES[index][0] for index in (3, 33, 52, 61)),
            exact_curves._BranchConstants.__annotate_func__.__closure__[0].cell_contents)


def trusted_store_identity(path, value):
    """Exact live namespace/owner binding, not a path-plus-dict assertion."""
    if path not in _TRUSTED_IMAGE_STORES or type(value) is not dict:
        raise QualificationDecline('unknown trusted store identity')
    index = _TRUSTED_IMAGE_STORES.index(path)
    if value is not _trusted_store_values()[index]:
        raise QualificationDecline('trusted store owner binding changed: ' + path)
    namespaces = ('builtins', 'numpy.finfo._finfo_cache', 'numpy._core.fromnumeric',
                  'numpy._core.shape_base', 'numpy._core.einsumfunc', 'numpy._core._asarray',
                  'anygeometry.exact_curves._BranchConstants annotations')
    namespace = namespaces[index]
    if index in (0, 2, 3, 4, 5):
        module = _registered_namespace(value)
        if module is None or module.__name__ != namespace:
            raise QualificationDecline('trusted namespace identity mismatch: ' + path)
    elif index == 1:
        import numpy as np
        from . import backend
        if backend._RUNTIME_TRUSTED_NUMPY[12][1] is not np.finfo or value is not np.finfo._finfo_cache:
            raise QualificationDecline('trusted finfo owner mismatch')
    basis = ('source-pinned owned annotation dictionary with structural boundary digest; not purity-proven'
             if index == 6 else
             'exact runtime owner/namespace plus source/runtime binary provenance; contents not purity-proven')
    return dict(path=path, namespace=namespace, basis=basis,
                content_check='structural' if index == 6 else 'risk-accepted opaque contents')


def trusted_store_identities():
    return [trusted_store_identity(path, value)
            for path, value in zip(_TRUSTED_IMAGE_STORES, _trusted_store_values())]


def trusted_annotation_digest(value):
    """Structural check only; never evaluates a ForwardRef or traverses globals.

    Protect keys, exact reference class, reference identity, expression/flags and
    referenced opaque object identities. Underlying globals/native contents are
    still risk-accepted, not proved transitively immutable.
    """
    import annotationlib
    if type(value) is not dict:
        raise QualificationDecline('annotation store is not an exact dict')
    rows = []
    for key, item in value.items():
        if type(key) is not str or type(item) is not annotationlib.ForwardRef:
            raise QualificationDecline('annotation store schema changed')
        values = []
        for name in ('__arg__', '__forward_is_argument__', '__forward_is_class__', '__forward_module__'):
            part = object.__getattribute__(item, name)
            if type(part) not in (str, bool, type(None)):
                raise QualificationDecline('annotation reference scalar changed')
            values.append(part)
        opaque = []
        for name in ('__globals__', '__extra_names__', '__code__', '__ast_node__', '__cell__', '__owner__',
                     '__stringifier_dict__'):
            try:
                part = object.__getattribute__(item, name)
            except AttributeError:
                opaque.append((name, 'absent'))
            else:
                opaque.append((name, id(part)))
        rows.append((key, id(item), tuple(values), tuple(opaque)))
    return content_digest(tuple(rows))


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
                 ('dataclasses._HAS_DEFAULT_FACTORY', dataclasses._HAS_DEFAULT_FACTORY),
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


def runtime_infrastructure():
    """Verify only the exact owner-load provenance handle; never a store waiver."""
    owner=sys.modules.get('anygeometry.model')
    handle=None if owner is None else vars(owner).get('_RUNTIME_OPERATION_OWNER')
    if type(handle) is not tuple or len(handle)!=4:
        raise QualificationDecline('missing owner-load runtime infrastructure')
    module,verifier,code,seal=handle
    if (type(module) is not types.ModuleType or type(verifier) is not types.FunctionType or
            verifier.__code__ is not code or type(seal) is not tuple or
            sys.modules.get('anygeometry._planning_runtime_anchor') is not module or
            vars(module).get('OWNER') is not handle or vars(module).get('verify_owner') is not verifier):
        raise QualificationDecline('runtime infrastructure binding changed')
    verifier(owner)
    from ._planning_qualification import pin_module
    native,encoder,rows,complete=vars(module)['JSON_ANCHOR']
    return handle,dict(anchor=pin_module(module),json_native=pin_module(native),json_encoder=pin_module(encoder))


def _census(modules, objects, *, omit_cache_payloads=False, definitions=None):
    """Keep the existing bound while treating registered H stores as opaque.

    LRU wrappers still expose their wrapped callable/defaults to the ordinary
    walker. Their C cache entries were never traversed by inventory. This is a
    content exemption only for the existing reset registry, not new eligibility.
    """
    definitions = {} if definitions is None else definitions
    infrastructure,_=runtime_infrastructure()
    modules=tuple(module for module in modules if module is not infrastructure[0])
    def infrastructure_boundary(value,path):
        return ((path=='anygeometry.model._RUNTIME_OPERATION_OWNER' and value is infrastructure) or
                (path=='anygeometry._planning_runtime_anchor' and value is infrastructure[0]))
    omitted = {}
    roots = {vars(module)['__name__'] + '.' + name: value
             for module in modules for name, value in vars(module).items()
             if name in _NUMERICAL_ROOTS and type(value) in (dict, list, set)}

    def boundary(value, path):
        if (roots.get(path) is value or type(value) in (
                weakref.WeakKeyDictionary, weakref.WeakValueDictionary, weakref.WeakSet)):
            omitted[path] = value
            return True
        return False

    report = inventory(modules, store_objects=objects, definition_objects=definitions,
                       store_boundary=boundary if omit_cache_payloads else None,
                       infrastructure_boundary=infrastructure_boundary)
    if any(id(value) in definitions for value in omitted.values()):
        raise QualificationDecline('registered cache aliases protected definition')
    protected = (*_FIXED_ROOTS, *_LIVE_CONTENT_PATHS, _LIVE_NAMESPACE_PATH, *_TRUSTED_IMAGE_STORES,
                 *(path for paths in definitions.values() for path in paths))
    if any(target in omitted and any(alias == path or alias.startswith(path + '.') or
                                     alias.startswith(path + '[') for path in protected)
           for alias, target in report.aliases):
        raise QualificationDecline('registered cache aliases protected root')
    return report, omitted


class Ledger:
    def __init__(self, report, objects, rows, modules, *, trusted_image=False,
                 omit_cache_payloads=False):
        check_coverage(report, rows)
        self.report, self.objects, self.rows = report, objects, rows
        self.modules = modules
        self.infrastructure,self.infrastructure_pin=runtime_infrastructure()
        self.trusted_image = trusted_image
        self.omit_cache_payloads = omit_cache_payloads
        self.registered = {m.__name__: m for m in modules if sys.modules.get(m.__name__) is m}
        self.resets = {p: objects[p] for p, row in rows.items() if row['action'] == 'reset'}
        self.costs = []
        self.census_costs = []
        self.digests = {}
        self.live_digests = {}
        self.live_references = {}
        self.trusted_identities = {}
        self.trusted_digests = {}
        for path, row in rows.items():
            if row['action'] == 'trusted-runtime-image':
                identity = trusted_store_identity(path, objects[path])
                self.trusted_identities[path] = identity
                row.update(identity=identity, reason=identity['basis'])
                if path == _TRUSTED_IMAGE_STORES[-1]:
                    self.trusted_digests[path] = trusted_annotation_digest(objects[path])
                    # Hold references: process-local identity cannot be recycled.
                    self.live_references[path] = tuple(objects[path].items())
            elif row['action'] == 'require-live-binding' and path == _LIVE_NAMESPACE_PATH:
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
        infrastructure,pin=runtime_infrastructure()
        if infrastructure is not self.infrastructure or pin!=self.infrastructure_pin:
            raise QualificationDecline('runtime infrastructure source/binding changed')
        if any(sys.modules.get(name) is not module for name, module in self.registered.items()):
            raise QualificationDecline('ledger module binding changed')
        objects = {}
        report, omitted = _census(self.modules, objects, omit_cache_payloads=self.omit_cache_payloads)
        if report.truncated:
            raise QualificationDecline('live census truncated')
        if any(self.rows.get(path, {}).get('action') != 'reset' for path in omitted):
            raise QualificationDecline('host cache registry decision changed')
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
                raise QualificationDecline('live store binding rebound: ' + path)
        for path, expected in self.digests.items():
            if content_digest(objects[path]) != expected:
                raise QualificationDecline('protected store content changed: ' + path)
        for path, expected in self.live_digests.items():
            if live_namespace_snapshot(objects[path])[0] != expected:
                raise QualificationDecline('live binding/code changed: ' + path)
        for path, expected in self.trusted_identities.items():
            if trusted_store_identity(path, objects[path]) != expected:
                raise QualificationDecline('trusted namespace changed: ' + path)
        for path, expected in self.trusted_digests.items():
            if trusted_annotation_digest(objects[path]) != expected:
                raise QualificationDecline('trusted annotation content changed: ' + path)
        self.census_costs.append(time.perf_counter() - started)
        return objects

    def prepare(self, *, owned_worker):
        check_coverage(self.report, self.rows)
        objects = self.validate()
        return self._reset_checked_handles(objects, owned_worker=owned_worker)

    def reset_for_batch(self, *, owned_worker):
        """Risk-accepted batch reset, not purity-proven; no census/hash here.

        Handles were validated at the preceding boundary. Rebindings detected at
        the next boundary invalidate every staged result in that assignment.
        """
        check_coverage(self.report, self.rows)
        return self._reset_checked_handles(self.objects, owned_worker=owned_worker)

    def _reset_checked_handles(self, objects, *, owned_worker):
        # Refusal checks precede all mutations, including independent caches.
        self._check_live_state(objects)
        start = time.perf_counter()
        counts = reset_declined({p: objects[p] for p in self.resets}, owned_worker=owned_worker)
        self.costs.append(time.perf_counter() - start)
        return counts

    def validate_host(self):
        """Admit existing H store state without clearing or replacing any store."""
        check_coverage(self.report, self.rows)
        objects = self.validate()
        self._check_live_state(objects)
        return objects

    def _check_live_state(self, objects):
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

    def record(self):
        return dict(stores=list(self.rows.values()), aliases=list(self.report.aliases),
                    unknowns=list(self.report.unknowns), truncated=self.report.truncated,
                    reset_seconds=list(self.costs), census_seconds=list(self.census_costs),
                    live_code_pins=self.live_code_pins())


def build_ledger(report, objects, *, modules=None, trusted_image=False, omit_cache_payloads=False):
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
    census, omitted = _census(modules, fresh, definitions=definitions,
                             omit_cache_payloads=omit_cache_payloads)
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
                         reason=('exact trusted namespace/owner check pending; not purity-proven'
                                 if action == 'trusted-runtime-image' else
                                 'PP1f explicit opaque runtime decline; transitive mutable state not qualified'
                                 if path in _DECLINED_RUNTIME_PATHS and action == 'decline' else
                                 'PP1f exact binding/content guard; prior reset/static/empty policy retained'))
    for path, row in rows.items():
        for parent in reset_roots:
            if path != parent and (path.startswith(parent + '.') or path.startswith(parent + '[')):
                row.update(action='covered-by-reset', parent=parent)
                break
    if any(rows[path]['action'] != 'reset' for path in omitted):
        raise QualificationDecline('host cache omission is not a registered store')
    return Ledger(report, objects, rows, modules, trusted_image=trusted_image,
                  omit_cache_payloads=omit_cache_payloads)


def loaded_geometry_ledger(*, trusted_image=False, modules=None, omit_cache_payloads=False):
    if type(omit_cache_payloads) is not bool:
        raise QualificationDecline('host census selection must be explicit bool')
    if modules is None:
        if omit_cache_payloads:
            raise QualificationDecline('host census requires explicit numerical module roots')
        modules = tuple(m for name, m in sorted(sys.modules.items())
                        if name == 'anygeometry' or name.startswith('anygeometry.'))
    if type(modules) is not tuple or not modules or any(type(m) is not types.ModuleType for m in modules):
        raise QualificationDecline('invalid explicit numerical module roots')
    objects = {}
    report, _ = _census(modules, objects, omit_cache_payloads=omit_cache_payloads)
    return build_ledger(report, objects, modules=modules, trusted_image=trusted_image,
                        omit_cache_payloads=omit_cache_payloads)

class PackageWorkerLedger:
    """Production package closure with explicit owned-worker reset authority.

    The diagnostic interpreter ledger above remains unchanged. This route uses
    the common/constructor graph and pinned external interfaces, never the
    diagnostic external-internals anchor.
    """
    trusted_image=False

    def __init__(self,model,*,resource_pin=None,role=None):
        from ._planning_manifest import LoadedManifest
        from ._planning_qualification import verify_resource_file_pin,verify_worker_resource_absence
        if role not in (None,'planner','worker') or role is not None and resource_pin is None:
            raise QualificationDecline('package resource role binding')
        self.role=role
        self.resource_pin=None if resource_pin is None else json.loads(json.dumps(resource_pin,allow_nan=False))
        self._resource_binding=self.resource_pin
        self._resource_token=json.dumps(self.resource_pin,sort_keys=True,allow_nan=False)
        if resource_pin is not None:verify_resource_file_pin(self.resource_pin)
        if role=='worker':verify_worker_resource_absence()
        self.manifest=LoadedManifest(model=model,include_host=False)
        self.costs=[];self.census_costs=[]
        self.store_bindings=dict(self.manifest.callables.registered_stores)
        self.objects={path:row[2] for path,row in self.store_bindings.items()}
        self.resets={path:value for path,value in self.objects.items() if type(value) is not contextvars.ContextVar}
        self.rows={path:dict(path=path,kind=type(value).__module__+'.'+type(value).__name__,
            action='require-empty' if type(value) is contextvars.ContextVar else 'reset') for path,value in self.objects.items()}
        self._references=tuple(value for part in (self.manifest.callables,*self.manifest.roles.values()) for value in part.strong_references)
        self._contexts=tuple({id(value):value for value in self._references if type(value) is contextvars.ContextVar}.values())
        self._check_live_state()

    def live_code_pins(self):
        return dict(common=self.manifest.callables.digest,constructor=self.manifest.roles['constructor'].digest)

    def _check_live_state(self):
        from ._planning_manifest import context_state,cancellation_registry
        from ._planning_qualification import verify_worker_resource_absence
        if self.role=='worker':verify_worker_resource_absence()
        if self.resource_pin is not self._resource_binding or json.dumps(self.resource_pin,sort_keys=True,allow_nan=False)!=self._resource_token:
            raise QualificationDecline('package resource pin changed')
        cancellation_registry(worker=True)
        for variable in self._contexts:
            context_state(variable,variable.name,role='worker')
        for path,(owner,name,value,kind) in self.store_bindings.items():
            if type(value) is not kind or (owner is not None and (sys.modules.get(owner.__name__) is not owner or vars(owner).get(name) is not value)):
                raise QualificationDecline('registered worker store rebound: '+path)
        # Cache payloads are disposable, but cannot own an alias to protected
        # mutable definitions. Inspect data referents only, never execute hooks.
        from ._planning_manifest import _is_type
        leaves=(str,bytes,int,float,bool,complex,type(None),type(Ellipsis),type(NotImplemented),
                types.ModuleType,types.FunctionType,types.CodeType,types.BuiltinFunctionType)
        protected={id(value) for value in self._references if type(value) not in leaves and not _is_type(value)}-set(map(id,self.resets.values()))
        pending=[]
        for path,value in self.resets.items():
            if type(value) is functools._lru_cache_wrapper:continue # native LRU entries expose no Python traversal API
            if type(value) in (weakref.WeakKeyDictionary,weakref.WeakValueDictionary):children=value.values()
            elif type(value) is weakref.WeakSet:children=value
            else:children=gc.get_referents(value)
            pending.extend((path+'.payload',item) for item in children)
        seen=set()
        while pending:
            path,value=pending.pop();identity=id(value)
            if identity in seen:continue
            seen.add(identity)
            if len(seen)>80000:raise QualificationDecline('worker cache alias walk bound exceeded')
            # CPython shares this immutable leaf across numerical empty results
            # and definition metadata. It has no descendants or mutable storage.
            if type(value) is tuple and len(value)==0:continue
            if identity in protected:
                bindings=[row[0] for part in (self.manifest.callables,*self.manifest.roles.values())
                    for row in part._identities if len(row)>1 and row[1]==identity]
                kind=type.__getattribute__(type(value),'__module__')+'.'+type.__getattribute__(type(value),'__qualname__')
                error=QualificationDecline('registered cache descendant aliases protected definition: '+path+' -> '+kind+' / '+str(bindings[:3]))
                error.alias_details=dict(cache_path=path,type=kind,bindings=bindings,visited=len(seen),
                    tuple_length=len(value) if type(value) is tuple else None)
                raise error
            if type(value) in leaves or type(value) is weakref.ReferenceType or _is_type(value):continue
            pending.extend((path+'.referent'+str(index),item) for index,item in enumerate(gc.get_referents(value)) if not _is_type(item))

    def validate(self):
        started=time.perf_counter()
        if self.resource_pin is not None:
            from ._planning_qualification import verify_resource_file_pin
            verify_resource_file_pin(self.resource_pin)
        self._check_live_state();self.manifest.verify()
        self.census_costs.append(time.perf_counter()-started)
        return self.objects

    def reset_for_batch(self,*,owned_worker):
        self._check_live_state()
        started=time.perf_counter()
        counts=reset_declined(self.resets,owned_worker=owned_worker)
        self.costs.append(time.perf_counter()-started)
        return counts

    def prepare(self,*,owned_worker):
        self.validate()
        return self.reset_for_batch(owned_worker=owned_worker)

    def record(self):
        return dict(stores=list(self.rows.values()),reset_seconds=list(self.costs),
            census_seconds=list(self.census_costs),live_code_pins=self.live_code_pins(),
            nodes=self.manifest.nodes,policy='package-worker-v1')


def package_worker_ledger(model,*,resource_pin=None,role=None):
    return PackageWorkerLedger(model,resource_pin=resource_pin,role=role)
