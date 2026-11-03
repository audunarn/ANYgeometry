"""Passive codec observations for one declared CPython build; no lookup or priming.

The returned objects are observations, not proof that a codec is safe to call.
Host admission must also bind this adapter's source and mutable Python state.
"""
if '_ANCHOR' in globals():
    raise RuntimeError('codec inspector reload requires a fresh process')
import ctypes
from ctypes import wintypes as W
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES, ModuleSpec
from pathlib import Path
import sys
from types import MappingProxyType
from collections import namedtuple

_NAME = 'anygeometry._planning_runtime_inspector'
_EXPORT_NAMES = ('snapshot', 'matches', 'identity')
_MODULE_TYPE = type(sys)
_BUILTIN_TYPE = type(len)

class InspectionUnavailable(RuntimeError):
    """Missing, unsupported or changed passive runtime inspector."""

def _decline(reason):
    raise InspectionUnavailable('passive codec inspector ' + reason)
def _native_image_offsets(exports, path):
    """Read C entry provenance without invoking any inspector export.

    Exact CPython builtins retain their owning module. FROM_ADDRESS with
    UNCHANGED_REFCOUNT (0x6) borrows the image handle; no module ref is acquired.
    Only image-relative offsets enter the cross-process identity.
    """
    entry = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(
        ('PyCFunction_GetFunction', ctypes.pythonapi))
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    containing = kernel.GetModuleHandleExW
    containing.argtypes = [W.DWORD, ctypes.c_void_p, ctypes.POINTER(W.HMODULE)]
    containing.restype = W.BOOL
    filename = kernel.GetModuleFileNameW
    filename.argtypes = [W.HMODULE, W.LPWSTR, W.DWORD]
    filename.restype = W.DWORD
    offsets = []
    for export in exports:
        address = entry(export)
        image = W.HMODULE()
        if not address or not containing(0x6, ctypes.c_void_p(address), ctypes.byref(image)) or not image.value:
            _decline('native entry image lookup failed')
        buffer = ctypes.create_unicode_buffer(32768)
        length = filename(image, buffer, len(buffer))
        if not 0 < length < len(buffer) or Path(buffer.value).resolve() != path:
            _decline('native entry is outside the bound helper image')
        offset = address - image.value
        if offset < 0:
            _decline('native entry offset is invalid')
        offsets.append(offset)
    return tuple(offsets)


def _snapshot(module):
    # Modules and loader metadata ARE mutable. Inspect exact types before any
    # attribute dispatch, then bind/recheck their live identities and metadata.
    if type(module) is not _MODULE_TYPE:
        _decline('module type differs from the runtime inspector')
    namespace = vars(module)
    if (type(namespace.get('__name__')) is not str or namespace['__name__'] != _NAME
            or sys.modules.get(_NAME) is not module
            or type(namespace.get('__package__')) is not str or namespace['__package__'] != 'anygeometry'):
        _decline('module ownership differs')
    package = sys.modules.get('anygeometry')
    if type(package) is not _MODULE_TYPE or vars(package).get('_planning_runtime_inspector') is not module:
        _decline('package export differs')
    spec, loader = namespace.get('__spec__'), namespace.get('__loader__')
    if type(spec) is not ModuleSpec or type(loader) is not ExtensionFileLoader:
        _decline('extension loader provenance differs')
    if (spec.loader is not loader or type(spec.name) is not str or spec.name != _NAME
            or type(loader.name) is not str or loader.name != _NAME):
        _decline('extension loader binding differs')
    origin = namespace.get('__file__')
    if type(origin) is not str or type(spec.origin) is not str or type(loader.path) is not str:
        _decline('extension origin is invalid')
    path = Path(origin).resolve()
    if (path.parent != Path(__file__).resolve().parent
            or path.name not in tuple('_planning_runtime_inspector' + suffix for suffix in EXTENSION_SUFFIXES)
            or Path(spec.origin).resolve() != path or Path(loader.path).resolve() != path):
        _decline('extension image provenance differs')
    if any(type(name) is not str for name in namespace):
        _decline('module namespace key is invalid')
    public = {name for name in namespace if not name.startswith('__')}
    if public != set(_EXPORT_NAMES):
        _decline('export schema differs')
    exports = tuple(namespace[name] for name in _EXPORT_NAMES)
    for name, export in zip(_EXPORT_NAMES, exports):
        if (type(export) is not _BUILTIN_TYPE or export.__name__ != name
                or type(export.__module__) is not str or export.__module__ != _NAME
                or export.__self__ is not module):
            _decline('live export differs: ' + name)
    # No export has been invoked before all metadata and native-image checks.
    offsets = _native_image_offsets(exports, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (module, type(module), spec, type(spec), loader, type(loader),
            exports, tuple(type(value) for value in exports), offsets, str(path), digest,
            tuple(sorted(namespace)))


def _capture():
    if sys.platform != 'win32' or sys.implementation.name != 'cpython':
        return None, 'requires Windows CPython'
    try:
        return _snapshot(importlib.import_module(_NAME)), None
    except (ImportError, OSError, InspectionUnavailable) as error:
        return None, str(error)

_ANCHOR, _FAILURE = _capture()

def _validated():
    if _ANCHOR is None:
        _decline(_FAILURE)
    current = _snapshot(_ANCHOR[0])
    for index in range(6):
        if current[index] is not _ANCHOR[index]:
            _decline('module or loader identity changed')
    if any(a is not b for a,b in zip(current[6],_ANCHOR[6])):
        _decline('native export identity changed')
    if current[7:] != _ANCHOR[7:]:
        _decline('native provenance changed')
    return _ANCHOR

def identity():
    # Bootstrap imports are outside the captured workload interval. Once an
    # anchor exists, use only its retained native export to refuse hooks before
    # file hashing/ctypes provenance can emit an audit event.
    if _ANCHOR is None:
        _decline(_FAILURE)
    _ANCHOR[6][2]()
    anchor = _validated()
    result = anchor[6][2]()
    if (type(result) is not dict or result.get('schema') != 2
            or result.get('audit_hooks_absent') is not True
            or result.get('abi') != 'cp314-cp314-win_amd64'
            or result.get('runtime_sha256') != '7ffe3ab11342db03cc18b026f27485c0f74f5bf10f239f1f6573394900596394'
            or result.get('headers_sha256') != '70647ed05bf469f9d5fd21cb6fe3733c14c245c04c340d947428f72ce0442d72'
            or result.get('profile_sha256') != '23754f3aa260f172b0f0344e86fc98b3ef63940a59129b8751764d3f13e9c579'):
        _decline('native profile differs')
    _validated()
    wire = dict(result, module=_NAME, binary_sha256=anchor[10],
                exports=[dict(name=name,image_offset=offset)
                         for name,offset in zip(_EXPORT_NAMES,anchor[8])])
    # Construct the portable record before the last allocation-aware native
    # absence recheck, so wrapper allocation cannot publish stale absence.
    anchor[6][2]()
    return wire

class Observation(namedtuple('_Observation', 'search_path codecs error_handlers profile token')):
    __slots__ = ()
    def matches(self):
        if _ANCHOR is None:
            _decline(_FAILURE)
        if not _ANCHOR[6][1](self.token):
            return False
        anchor = _validated()
        result = anchor[6][1](self.token)
        _validated()
        # Provenance validation allocates and may run finalizers. The last
        # registry check must follow it; a prior observed drift stays false.
        return anchor[6][1](self.token) and result

def snapshot():
    profile = identity()
    anchor = _validated()
    token,path,codecs,errors = anchor[6][0]()
    value = Observation(path,codecs,errors,MappingProxyType(profile),token)
    _validated()
    if not anchor[6][1](token):
        _decline('registry changed while wrapping observation')
    return value
