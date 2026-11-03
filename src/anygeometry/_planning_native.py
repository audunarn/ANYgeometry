"""Private stable-ABI resource helper loader and source-independent binary identity.

No numerical backend is enabled here. Missing/unsupported helpers refuse before
any Job, file or process acquisition. The identity is an admission input, never
an exemption from the caller's module/source binding rules.
"""
if '_OWNER_ANCHOR' in globals():
    raise RuntimeError('planning native owner reload requires a fresh process')

import ctypes
from ctypes import wintypes as W
import hashlib
import importlib
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES, ModuleSpec
import os
from pathlib import Path
import sys
import sysconfig

_NAME = 'anygeometry._planning_native_core'
_EXPORT_NAMES = ('new_owner', 'reserve_child', 'job_create', 'state', 'close',
                 'query', 'configure', 'terminate', 'assign', 'observe', 'create',
                 'member', 'resume', 'wait', 'exit_code', 'inject_fault', 'identity')
_MODULE_TYPE = type(sys)
_BUILTIN_TYPE = type(len)


def _decline(reason):
    from ._planning_qualification import QualificationDecline
    raise QualificationDecline('planning resource helper ' + reason)


def _native_image_offsets(exports, path):
    """Read C entry provenance without invoking any resource-helper export.

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
        _decline('module type differs from the resource extension')
    namespace = vars(module)
    if (type(namespace.get('__name__')) is not str or namespace['__name__'] != _NAME
            or sys.modules.get(_NAME) is not module
            or type(namespace.get('__package__')) is not str or namespace['__package__'] != 'anygeometry'):
        _decline('module ownership differs')
    package = sys.modules.get('anygeometry')
    if type(package) is not _MODULE_TYPE or vars(package).get('_planning_native_core') is not module:
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
            or path.name not in tuple('_planning_native_core' + suffix for suffix in EXTENSION_SUFFIXES)
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


def _capture_owner():
    if os.name != 'nt' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        return None, 'requires Windows CPython with GIL'
    try:
        return _snapshot(importlib.import_module(_NAME)), None
    except Exception as error:
        # Preserve helper-OFF imports without accepting a later re-baseline.
        return None, 'unavailable at owner load: ' + type(error).__name__ + ': ' + str(error)


_OWNER_ANCHOR, _OWNER_FAILURE = _capture_owner()


def _validated():
    try:
        module = importlib.import_module(_NAME)
    except (ImportError, OSError, AttributeError) as error:
        _decline('unavailable: ' + str(error))
    if _OWNER_ANCHOR is None:
        _decline(_OWNER_FAILURE)
    live = _snapshot(module)
    anchored = _OWNER_ANCHOR
    if (any(live[index] is not anchored[index] for index in range(6))
            or any(value is not expected for value, expected in zip(live[6], anchored[6]))
            or any(value is not expected for value, expected in zip(live[7], anchored[7]))
            or live[8:] != anchored[8:]):
        _decline('owner-load binding changed')
    return anchored


def load():
    if os.name != 'nt' or sysconfig.get_config_var('Py_GIL_DISABLED'):
        _decline('requires Windows CPython with GIL')
    try:
        anchor = _validated()
        info = anchor[6][-1]()
    except (ImportError, OSError, AttributeError) as error:
        _decline('unavailable: ' + str(error))
    if (type(info) is not dict or set(info) != {'abi', 'resource_abi', 'build'}
            or info.get('abi') != 'cp311-abi3' or type(info.get('resource_abi')) is not int
            or info['resource_abi'] != 1 or type(info.get('build')) is not str):
        _decline('ABI mismatch')
    return anchor[0]


def identity():
    load()
    anchor = _OWNER_ANCHOR
    # Invoke only the strongly retained, validated native identity callable.
    return dict(anchor[6][-1](), module=_NAME, binary_sha256=anchor[10],
                exports_schema=1, module_type='builtins.module',
                export_type='builtins.builtin_function_or_method',
                exports=[dict(name=name, image_offset=offset)
                         for name, offset in zip(_EXPORT_NAMES, anchor[8])])


class _Function:
    """ctypes-shaped test seam; resource calls execute inside the native owner."""
    def __init__(self, function): self.function = function
    def __call__(self, *args): return self.function(*args)


def _number(handle):
    return (handle.value if hasattr(handle, 'value') else handle) or 0


class _SlotHandle(tuple):
    """Immutable integer-like handle with its native ownership slot attached.

    The slot travels with one call, including through the single-argument fault
    seam. Interruptions and reentrant cleanup cannot publish ambient routing.
    """
    __slots__ = ()

    def __new__(cls, value, index, kind):
        return tuple.__new__(cls, (int(value), index, kind))

    @property
    def value(self): return self[0]
    @property
    def slot(self): return self[1:]
    def __int__(self): return self.value
    def __index__(self): return self.value
    def __eq__(self, other):
        if isinstance(other, _SlotHandle): return tuple.__eq__(self, other)
        return self.value == other
    def __ne__(self, other): return not self == other
    def __hash__(self): return hash(self.value)


class NativeAPI:
    def __init__(self):
        self.native = load()
        # Existing owners retain exact native cleanup operations even if a live
        # module edge subsequently drifts. Manifest verification detects that
        # drift; cleanup must never invoke a newly substituted export.
        self._exports = _OWNER_ANCHOR[6]
        # Registered before the first acquiring call; an empty capsule owns no OS resources.
        self.owner = self._export('new_owner')()
        for name in ('CreateJobObjectW', 'SetInformationJobObject', 'QueryInformationJobObject',
                     'TerminateJobObject', 'CloseHandle', 'CreateProcessW', 'IsProcessInJob',
                     'ResumeThread', 'AssignProcessToJobObject', 'OpenProcess'):
            setattr(self, name, _Function(getattr(self, '_'+name)))
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.GetSystemDirectoryW = kernel.GetSystemDirectoryW

    def _call(self, function, *args, failure=0):
        try:
            result = function(self.owner, *args)
        except OSError as error:
            ctypes.set_last_error(error.winerror)
            return failure
        return 1 if result is None else result

    def _export(self, name): return self._exports[_EXPORT_NAMES.index(name)]

    def _CreateJobObjectW(self, *_): return self._call(self._export('job_create'))
    def _CloseHandle(self, handle):
        slot = getattr(handle, 'slot', None)
        if slot is None:
            slot = (-1, 0) if _number(handle) == self.state()['job'] else (-2, 0)
        return self._call(self._export('close'), _number(handle), *slot)
    def _TerminateJobObject(self, handle, code):
        if _number(handle) != self.state()['job']: ctypes.set_last_error(6); return 0
        return self._call(self._export('terminate'))
    def _SetInformationJobObject(self, handle, kind, value, size):
        if _number(handle) != self.state()['job']: ctypes.set_last_error(6); return 0
        return self._call(self._export('configure'), kind, ctypes.string_at(value, size))
    def _QueryInformationJobObject(self, handle, kind, value, size, returned):
        if _number(handle) != self.state()['job']: ctypes.set_last_error(6); return 0
        result = self._call(self._export('query'), kind, size)
        if result == 0: return 0
        ctypes.memmove(value, result, len(result))
        return 1
    def _AssignProcessToJobObject(self, handle, process):
        if _number(handle) != self.state()['job']: ctypes.set_last_error(6); return 0
        return self._call(self._export('assign'), _number(process))
    def _OpenProcess(self, access, inherit, pid):
        if access != 0x1000 or inherit: ctypes.set_last_error(5); return 0
        result = self._call(self._export('observe'), pid)
        return _SlotHandle(*result, 5) if result else 0
    def _CreateProcessW(self, executable, command, index, stdout, stderr, environment, directory, info):
        result = self._call(self._export('create'), index, executable, command, environment, str(directory), str(stdout), str(stderr))
        if result:
            from ._planning_windows_launch import ProcessInfo
            snapshot = self.state(index)
            record = ctypes.cast(info, ctypes.POINTER(ProcessInfo)).contents
            record.process, record.thread = snapshot['process'], snapshot['thread']
            record.pid, record.tid = snapshot['pid'], snapshot['tid']
        return result
    def _IsProcessInJob(self, process, job, member):
        result = self._call(self._export('member'), _number(process), _number(job), failure=None)
        if result is None: return 0
        ctypes.cast(member, ctypes.POINTER(W.BOOL))[0] = result
        return 1
    def _ResumeThread(self, thread):
        slot = getattr(thread, 'slot', None)
        index = slot[0] if slot is not None and slot[1] == 1 else -1
        return self._call(self._export('resume'), _number(thread), index, failure=0xffffffff)
    def state(self, index=-1): return self._export('state')(self.owner, index)
    def reserve(self): return self._export('reserve_child')(self.owner)
    def wait(self, index, milliseconds): return self._export('wait')(self.owner, index, milliseconds)
    def close_child(self, index, kind):
        state = self.state(index)
        handle = state['process'] if kind == 0 else state['thread'] if kind == 1 else state['io'][kind-2]
        if not handle: return 1
        return self.CloseHandle(_SlotHandle(handle, index, kind))
    def resume_child(self, index):
        return self.ResumeThread(_SlotHandle(self.state(index)['thread'], index, 1))
    def inject_fault(self, operation, code=5, *, after=False):
        self._export('inject_fault')(self.owner, operation, code, after)
