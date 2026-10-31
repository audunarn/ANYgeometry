"""Opt-in PP1 prerequisites, never imported by the serial planning path.

Inventory is deliberately conservative: discovery is not a purity proof. Every
store is declined until its key, value, alias and observable-work contract is
qualified. Nothing here clears stores or transports authority. ``python_token``
attests supported Python objects only, NOT native images or numerical runtime.
``ImportFreeze`` is a scoped import guard, NOT a sandbox for arbitrary Python or
native code. A caller must qualify its complete callable closure separately.
These primitives do not enable a pool or admit the numerical planning closure.
"""
from __future__ import annotations

import builtins
from collections import OrderedDict, deque
import contextvars
from dataclasses import dataclass
import dis
import functools
import hashlib
import importlib
import struct
import sys
import threading
import types
import typing
import weakref

from .errors import GeometryError


class PurityDecline(GeometryError):
    """An acceleration prerequisite is unproved; no tentative result is valid."""


@dataclass(frozen=True)
class Store:
    path: str
    kind: str
    decision: str
    reason: str


@dataclass(frozen=True)
class Inventory:
    stores: tuple[Store, ...]
    unknowns: tuple[tuple[str, str], ...]
    aliases: tuple[tuple[str, str], ...]
    truncated: bool


_ATOMS = (type(None), bool, int, str, bytes, float, complex, range,
          type(Ellipsis), type(NotImplemented))
_STORES = (dict, list, set, OrderedDict, deque, weakref.WeakKeyDictionary,
           weakref.WeakValueDictionary, weakref.WeakSet, contextvars.ContextVar,
           threading.local, functools._lru_cache_wrapper)
_METADATA = frozenset(('__builtins__', '__loader__', '__spec__', '__cached__',
                       '__doc__', '__file__', '__package__', '__name__', '__path__'))


def _namespace(module):
    if type(module) is not types.ModuleType:
        raise PurityDecline('unsupported module object')
    return types.ModuleType.__getattribute__(module, '__dict__')


def inventory(modules, *, max_nodes=80000, max_depth=24, store_objects=None, definition_objects=None):
    """Bounded actual-type walk; no user descriptors, iterators or repr calls.

    Module metadata is excluded from numerical stores, not from loader safety.
    Unknown native/descriptor state is explicitly reported. Aliases are retained.
    Empty and warmed stores receive the same decision; a read hit is not missed.
    """
    if type(modules) is not tuple or max_nodes < 1 or max_depth < 1:
        raise PurityDecline('invalid inventory bounds/modules')
    if store_objects is not None and (type(store_objects) is not dict or store_objects):
        raise PurityDecline('store handle output must be an empty dictionary')
    if definition_objects is not None and (type(definition_objects) is not dict or definition_objects):
        raise PurityDecline('definition output must be an empty dictionary')
    namespaces = [(m, _namespace(m)) for m in modules]
    included = {id(m) for m, _ in namespaces}
    stores, unknowns, aliases, seen = [], [], [], {}
    count = 0
    truncated = False

    def definition(value, path):
        # Register the metadata object itself, never all descendants of a path.
        if definition_objects is not None and type(value) in _STORES:
            definition_objects.setdefault(id(value), []).append(path)

    def unknown(path, reason):
        unknowns.append((path, reason))

    def namespace(mapping, path, depth, metadata=False):
        for name, value in tuple(dict.items(mapping)):
            if type(name) is not str:
                unknown(path, 'non-string namespace key')
                continue
            if metadata and name in _METADATA:
                continue
            visit(value, path + '.' + name, depth + 1)
            if truncated:
                break

    def visit(value, path, depth):
        nonlocal count, truncated
        count += 1
        if count > max_nodes or depth > max_depth:
            truncated = True
            return
        typ = type(value)
        if typ in _ATOMS:
            return
        identity = id(value)
        if identity in seen:
            aliases.append((path, seen[identity]))
            return
        seen[identity] = path
        if typ in _STORES:
            kind = type.__getattribute__(typ, '__name__')
            reason = ('value-keyed LRU: typed-bit keys, returned aliasing and work semantics unproved'
                      if typ is functools._lru_cache_wrapper else
                      'state/key/value/authority and observable-work semantics unproved')
            stores.append(Store(path, kind, 'declined', reason))
            if store_objects is not None:
                store_objects[path] = value
        if typ in (dict, OrderedDict):
            items = dict.items(value) if typ is dict else OrderedDict.items(value)
            for i, (key, item) in enumerate(items):
                visit(key, path + f'.key[{i}]', depth + 1)
                visit(item, path + ('.' + key if type(key) is str else f'.value[{i}]'), depth + 1)
                if truncated:
                    break
        elif typ in (tuple, list, set, frozenset, deque):
            for i, item in enumerate(value):
                visit(item, path + f'[{i}]', depth + 1)
                if truncated:
                    break
        elif typ in (weakref.WeakKeyDictionary, weakref.WeakValueDictionary, weakref.WeakSet):
            namespace(object.__getattribute__(value, '__dict__'), path + '.__dict__', depth)
        elif typ is types.FunctionType:
            namespace(value.__dict__, path + '.__dict__', depth)
            visit(value.__defaults__, path + '.__defaults__', depth + 1)
            if value.__kwdefaults__:
                definition(value.__kwdefaults__, path + '.__kwdefaults__')
                visit(value.__kwdefaults__, path + '.__kwdefaults__', depth + 1)
            if value.__closure__:
                for i, cell in enumerate(value.__closure__):
                    try:
                        visit(cell.cell_contents, path + f'.cell[{i}]', depth + 1)
                    except ValueError:
                        unknown(path, 'empty closure cell')
        elif typ is functools._lru_cache_wrapper:
            namespace(value.__dict__, path + '.__dict__', depth)
        elif typ is types.MethodType:
            visit(value.__func__, path + '.__func__', depth + 1)
            visit(value.__self__, path + '.__self__', depth + 1)
        elif typ is types.ModuleType:
            if identity not in included:
                unknown(path, 'module outside supplied closure')
        elif typ is type:
            # A native mappingproxy over an actual type dictionary, not an
            # arbitrary MappingProxyType whose backing mapping may run code.
            class_ns = type.__getattribute__(value, '__dict__')
            annotate = class_ns.get('__annotate_func__')
            if type(annotate) is types.FunctionType and annotate.__closure__:
                eager = typing._make_eager_annotate
                eager_code = any(part is annotate.__code__ for part in eager.__code__.co_consts)
                for key, cell in zip(annotate.__code__.co_freevars, annotate.__closure__):
                    if (eager_code and annotate.__globals__ is vars(typing)
                            and key in ('checked_types', 'types')):
                        try:
                            body = cell.cell_contents
                        except ValueError:
                            continue
                        if type(body) is dict and set(body) == set(class_ns.get('_fields', ())):
                            definition(body, path + '.__annotate_func__.' + key)
                    if key != '__classdict__':
                        continue
                    try:
                        body = cell.cell_contents
                    except ValueError:
                        continue
                    if (type(body) is dict and body.get('__annotate_func__') is annotate
                            and body.get('__module__') == class_ns.get('__module__')
                            and ((body.get('__dataclass_fields__') is not None and
                                  body['__dataclass_fields__'] is class_ns.get('__dataclass_fields__'))
                                 or all(key in class_ns and part is class_ns[key]
                                        for key, part in body.items()))):
                        definition(body, path + '.__annotate_func__.__classdict__')
            for name, item in class_ns.items():
                if name not in ('__dict__', '__weakref__', '__doc__', '__module__'):
                    if name in ('__annotations__', '__annotations_cache__', '__dataclass_fields__',
                                '__conditional_annotations__', '_field_defaults'):
                        definition(item, path + '.' + name)
                    visit(item, path + '.' + name, depth + 1)
                    if truncated:
                        break
            unknown(path, 'class construction/metaclass semantics unqualified')
        elif typ in (staticmethod, classmethod):
            visit(value.__func__, path + '.__func__', depth + 1)
        elif typ is contextvars.ContextVar:
            visit(value.get(None), path + '.current', depth + 1)
            unknown(path, 'other contexts/default and work semantics unqualified')
        else:
            # Only concrete native slot/dict descriptors are read. Do not call
            # getattr(value, ...) or a user-supplied __dict__ property.
            for base in type.__getattribute__(typ, '__mro__'):
                for name, descriptor in type.__getattribute__(base, '__dict__').items():
                    if type(descriptor) is types.MemberDescriptorType:
                        try:
                            visit(descriptor.__get__(value, typ), path + '.' + name, depth + 1)
                        except AttributeError:
                            pass
                    elif name == '__dict__':
                        # Native heap types may supply arbitrary getsets too.
                        # Do not execute one to manufacture a complete walk.
                        unknown(path + '.__dict__', 'instance dictionary access unqualified')
                    if truncated:
                        break
                if truncated:
                    break
            unknown(path, 'opaque instance/descriptor/native state')

    for m, mapping in namespaces:
        name = mapping.get('__name__')
        if type(name) is not str:
            raise PurityDecline('invalid module name')
        for key in ('__all__', '__annotations__', '__conditional_annotations__'):
            if key in mapping:
                definition(mapping[key], name + '.' + key)
        namespace(mapping, name, 0, metadata=True)
        if truncated:
            break
    return Inventory(tuple(stores), tuple(unknowns), tuple(aliases), truncated)


def require_pure(report):
    """No discovery result is silently upgraded to fixed or task-scoped."""
    if type(report) is not Inventory or report.truncated:
        raise PurityDecline('truncated or invalid registry')
    if report.stores:
        raise PurityDecline('declined store: ' + report.stores[0].path)
    if report.unknowns:
        raise PurityDecline('unknown state: ' + report.unknowns[0][0])


def python_token(modules, *, max_nodes=80000):
    """Hash actual supported Python objects, never current source-file bytes.

    Unsupported/native bindings refuse instead of receiving a wildcard token.
    Tokens are comparable across processes but are NOT full runtime admission.
    Typed encoding distinguishes +/-0, int/float and nested code/constants.
    """
    if type(modules) is not tuple:
        raise PurityDecline('unsupported closure')
    by_name = {}
    for m in modules:
        ns = _namespace(m)
        name = ns.get('__name__')
        if type(name) is not str or name in by_name:
            raise PurityDecline('invalid/duplicate closure module')
        by_name[name] = m
    digest = hashlib.sha256()
    seen = {}
    count = 0

    def emit(raw):
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)

    def encode(value, depth=0):
        nonlocal count
        count += 1
        if count > max_nodes or depth > 100:
            raise PurityDecline('truncated code identity')
        typ = type(value)
        emit(type.__getattribute__(typ, '__name__').encode())
        if value is None or value is Ellipsis or value is NotImplemented:
            return
        if typ in (str, bytes, int, bool):
            emit(value if typ is bytes else str(value).encode('utf-8', 'surrogatepass'))
            return
        if typ is float:
            emit(struct.pack('>d', value))
            return
        if typ is complex:
            emit(struct.pack('>dd', value.real, value.imag))
            return
        if typ is tuple:
            emit(str(len(value)).encode())
            for item in value:
                encode(item, depth + 1)
            return
        identity = id(value)
        if identity in seen:
            emit(b'ref'); emit(str(seen[identity]).encode())
            return
        seen[identity] = len(seen)
        if typ is types.CodeType:
            encode((value.co_code, value.co_exceptiontable, value.co_flags,
                    value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount,
                    value.co_nlocals, value.co_stacksize, value.co_names, value.co_varnames,
                    value.co_freevars, value.co_cellvars, value.co_consts,
                    value.co_name, value.co_qualname, value.co_firstlineno, value.co_linetable), depth + 1)
        elif typ is types.FunctionType:
            if value.__dict__:
                raise PurityDecline('unsupported decorated/function state')
            encode(value.__code__, depth + 1)
            encode(value.__defaults__, depth + 1)
            encode(tuple(sorted((value.__kwdefaults__ or {}).items())), depth + 1)
            for cell in value.__closure__ or ():
                try:
                    encode(cell.cell_contents, depth + 1)
                except ValueError as exc:
                    raise PurityDecline('unsupported empty closure') from exc
            def instructions(code):
                yield from dis.get_instructions(code)
                for constant in code.co_consts:
                    if type(constant) is types.CodeType:
                        yield from instructions(constant)
            ops = tuple(instructions(value.__code__))
            names = sorted({i.argval for i in ops if i.opname in ('LOAD_GLOBAL', 'LOAD_NAME')})
            for name in names:
                encode(name, depth + 1)
                if name in value.__globals__:
                    encode(value.__globals__[name], depth + 1)
                elif name in value.__builtins__:
                    encode(value.__builtins__[name], depth + 1)
                else:
                    raise PurityDecline('unsupported unresolved global: ' + name)
            if any(i.opname == 'IMPORT_NAME' for i in ops):
                raise PurityDecline('native importer runtime attestation unavailable')
        elif typ is types.ModuleType:
            ns = _namespace(value)
            name = ns.get('__name__')
            if by_name.get(name) is not value:
                raise PurityDecline('module outside code closure')
            encode(name, depth + 1)
            if any(type(key) is not str for key in ns):
                raise PurityDecline('unsupported module namespace key')
            for key in sorted(ns):
                if key not in _METADATA:
                    encode(key, depth + 1)
                    encode(ns[key], depth + 1)
        else:
            raise PurityDecline('unsupported/native code identity: ' + type.__getattribute__(typ, '__name__'))

    encode(('PP1-python-only-v1', sys.implementation.name, tuple(sys.version_info)), 0)
    for name in sorted(by_name):
        encode(by_name[name])
    return digest.hexdigest()


def require_matching_python(host_token, modules):
    if type(host_token) is not str or python_token(modules) != host_token:
        raise PurityDecline('code-skew: host and planner loaded Python differ')


def eager_closure(roots, *, allowed_names):
    """Import a finite, explicit static Python closure before task execution.

    Walk loaded function bytecode, including nested functions and local imports.
    Dynamic import/exec capability and native module bodies decline. This is an
    overapproximation/discovery prerequisite, not proof of arbitrary dispatch.
    """
    allowed = frozenset(allowed_names)
    if any(type(n) is not str for n in allowed) or len(allowed) > 10000:
        raise PurityDecline('invalid import closure')
    pending = list(roots)
    loaded = {}
    forbidden = {'__import__', 'eval', 'exec', 'compile', 'import_module', 'reload',
                 'exec_module', 'load_module', 'get_code', 'sys', 'importlib',
                 '__loader__', '__builtins__', '__globals__', '__code__'}
    seen = set()

    def code_imports(code, package):
        if id(code) in seen:
            return
        seen.add(id(code))
        if len(seen) > 80000:
            raise PurityDecline('truncated import closure')
        if forbidden.intersection(code.co_names):
            raise PurityDecline('dynamic import/code capability in closure')
        instructions = tuple(dis.get_instructions(code))
        for index, ins in enumerate(instructions):
            if ins.opname != 'IMPORT_NAME':
                continue
            if index < 2 or any(i.opname not in ('LOAD_CONST', 'LOAD_SMALL_INT')
                                for i in instructions[index-2:index]):
                raise PurityDecline('unknown import operands')
            level, fromlist = (i.argval for i in instructions[index-2:index])
            name = importlib.util.resolve_name('.' * level + ins.argval, package) if level else ins.argval
            pending.append(name)
            if fromlist:
                for item in fromlist:
                    if item == '*':
                        raise PurityDecline('unknown star import closure')
                    child = name + '.' + item
                    if child in allowed:
                        pending.append(child)
        for const in code.co_consts:
            if type(const) is types.CodeType:
                code_imports(const, package)

    while pending:
        name = pending.pop()
        if name in loaded:
            continue
        if name not in allowed:
            raise PurityDecline('import outside declared closure: ' + name)
        m = importlib.import_module(name)
        ns = _namespace(m)
        spec = ns.get('__spec__')
        # Native identity cannot be recovered from a current path/version.
        if spec is not None and (spec.origin in ('built-in', 'frozen') or
                                 not str(spec.origin).endswith('.py')):
            raise PurityDecline('native module import closure unqualified: ' + name)
        loaded[name] = m
        package = ns.get('__package__') or name.rpartition('.')[0]
        for value in tuple(ns.values()):
            if type(value) is types.FunctionType:
                code_imports(value.__code__, package)
            elif type(value) is types.ModuleType:
                pending.append(_namespace(value)['__name__'])
            elif type(value) is type:
                raise PurityDecline('class dispatch import closure unqualified')
    return tuple(loaded[name] for name in sorted(loaded))


class ImportFreeze:
    """Scoped guard for an already qualified, quiescent owned process.

    All new imports and reloads fail with a sticky violation. Cached imports
    require the exact allowed module object. End-of-task membership/hook checks
    detect replacement/preinsertion even if the task caught an import error.
    Direct retained loader aliases/native execution need closure qualification;
    this context manager alone must never be used as an admission certificate.
    """
    def __init__(self, modules):
        self.allowed = {}
        for m in modules:
            if m is None:
                continue
            ns = _namespace(m)
            name = ns.get('__name__')
            if type(name) is not str:
                raise PurityDecline('invalid import module')
            self.allowed[name] = m
        self.violation = None
        self.entered = False

    def _decline(self, reason):
        self.violation = self.violation or reason
        raise PurityDecline(self.violation)

    def _membership(self):
        if (len(sys.modules) != len(self.before) or
                any(sys.modules.get(k) is not v for k, v in self.before.items())):
            self._decline('module membership changed during import freeze')

    def _allowed(self, name):
        if name not in self.allowed or sys.modules.get(name) is not self.allowed[name]:
            self._decline('import outside frozen module closure: ' + name)

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        self._membership()
        absolute = (importlib.util.resolve_name('.' * level + name, (globals or {}).get('__package__'))
                    if level else name)
        self._allowed(absolute)
        # Prefixes also have to be attested, because __import__('a.b') returns a.
        parts = absolute.split('.')
        for i in range(1, len(parts)):
            self._allowed('.'.join(parts[:i]))
        if fromlist:
            ns = _namespace(self.allowed[absolute])
            for item in fromlist:
                if item == '*' or item not in ns:
                    self._decline('unqualified from-import')
                if type(ns[item]) is types.ModuleType:
                    self._allowed(_namespace(ns[item])['__name__'])
        result = self.original_import(name, globals, locals, fromlist, level)
        self._membership()
        return result

    def _import_module(self, name, package=None):
        self._membership()
        absolute = importlib.util.resolve_name(name, package) if name.startswith('.') else name
        self._allowed(absolute)
        return self.allowed[absolute]

    def _reload(self, module):
        self._decline('reload forbidden during import freeze')

    def find_spec(self, fullname, path=None, target=None):
        self._decline('new import forbidden during import freeze: ' + fullname)

    def __enter__(self):
        if self.entered or threading.active_count() != 1:
            raise PurityDecline('import freeze requires a single-thread owned epoch')
        self.entered = True
        self.before = dict(sys.modules)
        for name, module in self.allowed.items():
            if self.before.get(name) is not module:
                self._decline('module not loaded before import freeze')
        self.original_import = builtins.__import__
        self.original_import_module = importlib.import_module
        self.original_reload = importlib.reload
        self.original_meta = sys.meta_path
        self.meta = [self, *sys.meta_path]
        self.guarded_import = self._import
        self.guarded_import_module = self._import_module
        self.guarded_reload = self._reload
        builtins.__import__ = self.guarded_import
        importlib.import_module = self.guarded_import_module
        importlib.reload = self.guarded_reload
        sys.meta_path = self.meta
        return self

    def __exit__(self, typ, value, traceback):
        try:
            self._membership()
            if (builtins.__import__ is not self.guarded_import or
                    importlib.import_module is not self.guarded_import_module or
                    importlib.reload is not self.guarded_reload or sys.meta_path is not self.meta or
                    len(self.meta) != len(self.original_meta) + 1 or
                    any(a is not b for a, b in zip(self.meta[1:], self.original_meta))):
                self._decline('import guard changed during task')
            if self.violation:
                raise PurityDecline(self.violation)
        finally:
            builtins.__import__ = self.original_import
            importlib.import_module = self.original_import_module
            importlib.reload = self.original_reload
            sys.meta_path = self.original_meta
