"""Explicit H + P/W admission for the experimental planner.

H remains outside the owned Job. Its declared peak is checked against the
current-process lifetime private-commit peak; it is not an allocation limit.
The separate P/W cap is enforced and read back by OwnedTree. No defaults here
grant a machine-wide budget, a free-RAM fraction, or whole-model authority.
"""
import ctypes
import hashlib
from pathlib import Path
import sys
from dataclasses import dataclass

import psutil

from ._planning_executor import PlannerMemoryExceeded
from ._planning_qualification import QualificationDecline


_SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
_WIN_DLL_TYPE = ctypes.WinDLL if sys.platform=='win32' else None


class _MemoryStatus(ctypes.Structure):
    _fields_=[('dwLength',ctypes.c_uint32),('dwMemoryLoad',ctypes.c_uint32),
        ('ullTotalPhys',ctypes.c_uint64),('ullAvailPhys',ctypes.c_uint64),
        ('ullTotalPageFile',ctypes.c_uint64),('ullAvailPageFile',ctypes.c_uint64),
        ('ullTotalVirtual',ctypes.c_uint64),('ullAvailVirtual',ctypes.c_uint64),
        ('ullAvailExtendedVirtual',ctypes.c_uint64)]


_AvailablePrototype=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.POINTER(_MemoryStatus),use_last_error=True) if sys.platform=='win32' else None
_ResolvePrototype=ctypes.WINFUNCTYPE(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_char_p,use_last_error=True) if sys.platform=='win32' else None
_ContainingPrototype=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),use_last_error=True) if sys.platform=='win32' else None
_FilenamePrototype=ctypes.WINFUNCTYPE(ctypes.c_uint32,ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32,use_last_error=True) if sys.platform=='win32' else None
_DirectoryPrototype=ctypes.WINFUNCTYPE(ctypes.c_uint32,ctypes.c_wchar_p,ctypes.c_uint32,use_last_error=True) if sys.platform=='win32' else None


def _prototype(prototype,result,arguments):
    allowed={'_argtypes_','_restype_','_flags_','__module__','__doc__','__dict__','__weakref__',
        '__firstlineno__','__static_attributes__','__annotate__','__annotate_func__','__annotations__'}
    if (type(prototype) is not type(ctypes._CFuncPtr) or
            prototype.__mro__!=(prototype,*ctypes._CFuncPtr.__mro__) or
            set(vars(prototype))-allowed or prototype._argtypes_!=arguments or
            prototype._restype_ is not result or prototype._flags_!=ctypes._FUNCFLAG_USE_LASTERROR):
        raise QualificationDecline('available memory fixed signature changed')
    return prototype


def _memory_kernel():
    if ctypes.WinDLL is not _WIN_DLL_TYPE or type(ctypes.windll) is not ctypes.LibraryLoader or ctypes.windll._dlltype is not _WIN_DLL_TYPE:
        raise QualificationDecline('available memory provider changed')
    # LibraryLoader owns one cached load reference. Never call WinDLL(name) per
    # poll, and never mutate signatures on its cached exported function objects.
    kernel=ctypes.windll.kernel32
    if type(kernel) is not _WIN_DLL_TYPE or _WIN_DLL_TYPE.__getattribute__ is not object.__getattribute__:
        raise QualificationDecline('available memory library type/read interface')
    if type(vars(kernel).get('_handle')) is not int or vars(kernel)['_handle']<=0:
        raise QualificationDecline('available memory library handle')
    return kernel


def _memory_layout():
    fields=[('dwLength',ctypes.c_uint32),('dwMemoryLoad',ctypes.c_uint32),
        ('ullTotalPhys',ctypes.c_uint64),('ullAvailPhys',ctypes.c_uint64),
        ('ullTotalPageFile',ctypes.c_uint64),('ullAvailPageFile',ctypes.c_uint64),
        ('ullTotalVirtual',ctypes.c_uint64),('ullAvailVirtual',ctypes.c_uint64),
        ('ullAvailExtendedVirtual',ctypes.c_uint64)]
    if (sys.platform!='win32' or type(_MemoryStatus) is not type(ctypes.Structure) or
            _MemoryStatus.__mro__!=(_MemoryStatus,*ctypes.Structure.__mro__) or
            _MemoryStatus.__module__!=__name__ or _MemoryStatus.__name__!='_MemoryStatus' or
            _MemoryStatus._fields_!=fields or ctypes.sizeof(_MemoryStatus)!=64):
        raise QualificationDecline('available memory ABI/schema changed')
    allowed={name for name,_ in fields}|{'_fields_','__module__','__doc__','__dict__','__weakref__',
        '__firstlineno__','__static_attributes__','__annotations__','__annotate__','__annotate_func__'}
    if set(vars(_MemoryStatus))-allowed:
        raise QualificationDecline('available memory structure has unknown hooks')
    offsets=[getattr(_MemoryStatus,name).offset for name,_ in fields]
    if offsets!=[0,4,8,16,24,32,40,48,56]:
        raise QualificationDecline('available memory field offsets changed')
    return offsets


def _memory_binding():
    _memory_layout()
    kernel=_memory_kernel()
    query=_prototype(_AvailablePrototype,ctypes.c_int,(ctypes.POINTER(_MemoryStatus),))(('GlobalMemoryStatusEx',kernel))
    resolve=_prototype(_ResolvePrototype,ctypes.c_void_p,(ctypes.c_void_p,ctypes.c_char_p))(('GetProcAddress',kernel))
    # Borrow the CData pointer storage while query is strongly local. cast()
    # can add a self-reference to its _objects map and delay its retirement.
    address=ctypes.c_void_p.from_buffer(query).value
    if not address or resolve(kernel._handle,b'GlobalMemoryStatusEx')!=address:
        raise QualificationDecline('available memory export differs from provider')
    return kernel,query,address


def _available_bytes():
    # Fresh function/record instances and observation on every original poll;
    # fixed signature types and the provider's existing library cache are reused.
    kernel,query,_address=_memory_binding()
    row=_MemoryStatus();row.dwLength=ctypes.sizeof(row)
    if not query(ctypes.byref(row)):raise ctypes.WinError(ctypes.get_last_error())
    if row.dwLength!=64 or not 0<=row.dwMemoryLoad<=100 or not 0<=row.ullAvailPhys<=row.ullTotalPhys or row.ullAvailExtendedVirtual!=0:
        raise QualificationDecline('available memory native result schema')
    return row.ullAvailPhys


def available_memory_identity():
    """Finite H-only external API identity; no ownership-helper import."""
    offsets=_memory_layout();kernel,query,address=_memory_binding()
    containing=_prototype(_ContainingPrototype,ctypes.c_int,(ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)))(('GetModuleHandleExW',kernel))
    filename=_prototype(_FilenamePrototype,ctypes.c_uint32,(ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint32))(('GetModuleFileNameW',kernel))
    system_directory=_prototype(_DirectoryPrototype,ctypes.c_uint32,(ctypes.c_wchar_p,ctypes.c_uint32))(('GetSystemDirectoryW',kernel))
    buffer=ctypes.create_unicode_buffer(32768)
    length=system_directory(buffer,len(buffer))
    if not 0<length<len(buffer):raise QualificationDecline('available memory system directory lookup failed')
    directory=Path(buffer.value).resolve()
    image=ctypes.c_void_p()
    if not containing(0x6,ctypes.c_void_p(address),ctypes.byref(image)) or not image.value:
        raise QualificationDecline('available memory forwarded image lookup failed')
    paths=[]
    for handle in (kernel._handle,image.value):
        length=filename(handle,buffer,len(buffer))
        if not 0<length<len(buffer):raise QualificationDecline('available memory image filename failed')
        path=Path(buffer.value).resolve()
        if path.parent!=directory or path.name.lower() not in ('kernel32.dll','kernelbase.dll'):
            raise QualificationDecline('available memory export outside finite system provider')
        paths.append(path)
    if paths[0].name.lower()!='kernel32.dll' or address<image.value:
        raise QualificationDecline('available memory provider image mismatch')
    return dict(abi='MEMORYSTATUSEX-v1',size=64,offsets=offsets,export='GlobalMemoryStatusEx',
        provider_path=str(paths[0]),provider_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        image_path=str(paths[1]),image_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),image_offset=address-image.value)


def _integer(value, name, *, minimum=1):
    if type(value) is not int or not minimum <= value <= _SIZE_MAX:
        raise QualificationDecline('invalid memory ' + name)
    return value


def observe_memory():
    """Observe only H and aggregate available RAM, without a process census."""
    if sys.platform != 'win32':
        raise QualificationDecline('Windows private-commit accounting unavailable')
    info = psutil.Process().memory_info()
    return dict(host_peak_bytes=info.peak_pagefile,
                host_private_bytes=info.private,
                available_bytes=_available_bytes())


def _sample():
    value = observe_memory()
    if type(value) is not dict or set(value) != {
            'host_peak_bytes', 'host_private_bytes', 'available_bytes'}:
        raise QualificationDecline('unknown native memory accounting')
    peak = _integer(value['host_peak_bytes'], 'host peak')
    private = _integer(value['host_private_bytes'], 'host private', minimum=0)
    available = _integer(value['available_bytes'], 'available RAM', minimum=0)
    if private > peak:
        raise QualificationDecline('host private commit exceeds reported lifetime peak')
    return peak, private, available


@dataclass(frozen=True)
class MemoryBudget:
    host_peak_estimate_bytes: int
    tree_limit_bytes: int
    total_limit_bytes: int
    free_fraction: tuple

    def __post_init__(self):
        _integer(self.host_peak_estimate_bytes, 'host estimate')
        _integer(self.tree_limit_bytes, 'tree cap')
        _integer(self.total_limit_bytes, 'aggregate cap')
        fraction = self.free_fraction
        if (type(fraction) is not tuple or len(fraction) != 2 or
                any(type(v) is not int or not 0 < v <= _SIZE_MAX for v in fraction) or
                fraction[0] > fraction[1]):
            raise QualificationDecline('invalid memory free-RAM fraction')

    def begin(self):
        # Revalidate and snapshot primitive values even if a frozen object was
        # deliberately modified with object.__setattr__ before this call.
        self.__post_init__()
        host, tree, total = (self.host_peak_estimate_bytes, self.tree_limit_bytes,
                             self.total_limit_bytes)
        numerator, denominator = self.free_fraction
        try:
            peak, private, available = _sample()
        except QualificationDecline:
            raise
        except Exception as exc:
            raise QualificationDecline('native memory accounting unavailable: ' + type(exc).__name__) from exc
        if peak > host:
            raise QualificationDecline('observed host peak exceeds declared estimate')
        # Exact ceil(5*(H+PW)/4), no floating-point allowance or overflow.
        required = (5 * (host + tree) + 3) // 4
        effective = min(total, (available * numerator) // denominator)
        if required > effective:
            raise QualificationDecline('host plus tree and 25 percent headroom exceed memory admission budget')
        return MemoryLease(host, tree, total, (numerator, denominator),
                           peak, private, available, required, effective)


class MemoryLease:
    """One admitted plan; policy values never read again from caller controls."""
    def __init__(self, host, tree, total, fraction, peak, private, available, required, effective):
        self._policy = (host, tree, total, fraction, available, required, effective)
        self._peak = peak
        self._private = private
        self._samples = 1
        self._terminal_reason = None
        self._initial = (peak,private)
        self._phases = [dict(phase='admission',host_peak_bytes=peak,host_private_bytes=private,job=None)]

    @property
    def tree_limit_bytes(self):
        return self._policy[1]

    def check(self):
        if self._terminal_reason is not None:
            raise PlannerMemoryExceeded(self._terminal_reason)
        try:
            peak, private, _available = _sample()
        except Exception as exc:
            self._terminal_reason = 'admitted host memory accounting lost: ' + type(exc).__name__
            raise PlannerMemoryExceeded(self._terminal_reason) from exc
        self._samples += 1
        previous = self._peak
        self._peak = max(previous, peak)
        self._private = private
        if peak < previous:
            self._terminal_reason = 'host lifetime private-commit peak decreased'
        elif peak > self._policy[0]:
            self._terminal_reason = 'host private-commit peak exceeded admitted estimate'
        if self._terminal_reason is not None:
            raise PlannerMemoryExceeded(self._terminal_reason)

    def report(self):
        host, tree, total, fraction, available, required, effective = self._policy
        return dict(scope='planning-and-CI', host_accounting='Windows lifetime peak private commit',
                    host_peak_estimate_bytes=host, tree_limit_bytes=tree,
                    total_limit_bytes=total, free_fraction=fraction,
                    initial_available_bytes=available,
                    initial_host_peak_bytes=self._initial[0],initial_host_private_bytes=self._initial[1],
                    required_with_headroom_bytes=required, effective_limit_bytes=effective,
                    observed_host_peak_bytes=self._peak, observed_host_private_bytes=self._private,
                    samples=self._samples, terminal_reason=self._terminal_reason,
                    host_allocation_enforced=False,phases=list(self._phases),
                    exact_isolated_startup_peak='not measured: workers may begin tasks before all admissions are observed')

    def observe_phase(self,phase,job=None):
        if phase not in ('capture','pair-returned','face-returned','before-retirement','retired'):
            raise PlannerMemoryExceeded('unknown memory observation phase')
        self.check()
        if any(row['phase']==phase for row in self._phases):
            raise PlannerMemoryExceeded('duplicate memory observation phase')
        observation=None
        if job is not None:
            if (type(job) is not dict or set(job)!={'requested_bytes','limit_bytes','flags','peak_job_bytes','peak_process_bytes'} or
                    any(type(value) is not int for value in job.values()) or
                    job['requested_bytes']!=self.tree_limit_bytes or job['limit_bytes']!=self.tree_limit_bytes or
                    job['flags']!=0x2200 or not 0<=job['peak_job_bytes']<=self.tree_limit_bytes):
                raise PlannerMemoryExceeded('whole Job memory observation differs from admitted cap')
            observation=dict(job)
            observation.update(startup_peak_upper_bound_bytes=job['peak_job_bytes'],
                scope='complete owned P/W Job lifetime through observation; includes task work and earlier stages',
                enforcement_verified=True,isolated_startup_peak_measured=False)
        self._phases.append(dict(phase=phase,host_peak_bytes=self._peak,host_private_bytes=self._private,job=observation))
