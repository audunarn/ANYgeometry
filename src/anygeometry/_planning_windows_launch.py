"""Documented Windows creation transaction for the owned planner tree.

Job-list assignment is part of CreateProcessW; the retained primary thread is
resumed only after membership is verified. Unsupported attributes refuse. This
qualifies explicit runtime behavior, not every possible creator-death interleave.
"""
import ctypes
from ctypes import wintypes as W
import math
import msvcrt
import os
import subprocess

from ._planning_qualification import QualificationDecline


JOB_LIST = 0x2000d
HANDLE_LIST = 0x20002
CREATE_FLAGS = 0x80000 | 0x8000000 | 0x400 | 4


class StartupInfo(ctypes.Structure):
    _fields_ = [('cb', W.DWORD), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR),
                ('title', W.LPWSTR)] + [(name, W.DWORD) for name in
                ('x', 'y', 'x_size', 'y_size', 'x_chars', 'y_chars', 'fill', 'flags')] + [
                ('show', W.WORD), ('reserved_size', W.WORD), ('reserved_bytes', ctypes.c_void_p),
                ('stdin', W.HANDLE), ('stdout', W.HANDLE), ('stderr', W.HANDLE)]


class StartupInfoEx(ctypes.Structure):
    _fields_ = [('startup', StartupInfo), ('attributes', ctypes.c_void_p)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [('process', W.HANDLE), ('thread', W.HANDLE), ('pid', W.DWORD), ('tid', W.DWORD)]


def _error(operation):
    error = ctypes.WinError(ctypes.get_last_error())
    error.job_operation = operation
    return error


def _bind(api):
    void = ctypes.c_void_p
    for name, arguments, result in (
        ('GetCurrentProcess', [], W.HANDLE),
        ('DuplicateHandle', [W.HANDLE, W.HANDLE, W.HANDLE, ctypes.POINTER(W.HANDLE), W.DWORD, W.BOOL, W.DWORD], W.BOOL),
        ('InitializeProcThreadAttributeList', [void, W.DWORD, W.DWORD, ctypes.POINTER(ctypes.c_size_t)], W.BOOL),
        ('UpdateProcThreadAttribute', [void, W.DWORD, ctypes.c_size_t, void, ctypes.c_size_t, void, void], W.BOOL),
        ('DeleteProcThreadAttributeList', [void], None),
        ('CreateProcessW', [W.LPCWSTR, W.LPWSTR, void, void, W.BOOL, W.DWORD, void, W.LPCWSTR, void, ctypes.POINTER(ProcessInfo)], W.BOOL),
        ('IsProcessInJob', [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)], W.BOOL),
        ('ResumeThread', [W.HANDLE], W.DWORD),
        ('WaitForSingleObject', [W.HANDLE, W.DWORD], W.DWORD),
        ('GetExitCodeProcess', [W.HANDLE, ctypes.POINTER(W.DWORD)], W.BOOL),
        ('CloseHandle', [W.HANDLE], W.BOOL),
    ):
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result


class WindowsChild:
    """Pre-registered owner of native output handles, including failed creation."""
    def __init__(self, api, command):
        _bind(api)
        self.api, self.args = api, list(command)
        # Allocate before creation; the kernel writes into retained storage.
        self.info = ProcessInfo()
        self.returncode = None
        self._duplicates = []

    @property
    def pid(self):
        return int(self.info.pid) if self.info.pid else None

    @property
    def _handle(self):
        return self.info.process

    def _duplicate(self, handle):
        target = W.HANDLE()
        current = self.api.GetCurrentProcess()
        if not self.api.DuplicateHandle(current, handle, current, ctypes.byref(target), 0, True, 2):
            raise _error('DuplicateHandle')
        self._duplicates.append(target.value)
        return target.value

    def _close_duplicates(self):
        errors = []
        for handle in tuple(self._duplicates):
            if self.api.CloseHandle(handle):
                self._duplicates.remove(handle)
            else:
                errors.append(_error('CloseHandle inherited I/O duplicate'))
        if errors:
            raise errors[0]

    def create(self, job, directory, environment, stdout, stderr):
        attributes = None
        initialized = False
        try:
            with open(os.devnull, 'rb') as null:
                inherited = (W.HANDLE * 3)(self._duplicate(msvcrt.get_osfhandle(null.fileno())),
                                           self._duplicate(msvcrt.get_osfhandle(stdout.fileno())),
                                           self._duplicate(msvcrt.get_osfhandle(stderr.fileno())))
            size = ctypes.c_size_t()
            ctypes.set_last_error(0)
            sized = self.api.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
            if sized or ctypes.get_last_error() != 122 or not 0 < size.value <= 1024**2:
                raise _error('InitializeProcThreadAttributeList size')
            attributes = ctypes.create_string_buffer(size.value)
            if not self.api.InitializeProcThreadAttributeList(attributes, 2, 0, ctypes.byref(size)):
                raise _error('InitializeProcThreadAttributeList')
            initialized = True
            jobs = (W.HANDLE * 1)(job)
            for name, value in ((JOB_LIST, jobs), (HANDLE_LIST, inherited)):
                if not self.api.UpdateProcThreadAttribute(attributes, 0, name, value, ctypes.sizeof(value), None, None):
                    raise _error('UpdateProcThreadAttribute ' + hex(name))
            startup = StartupInfoEx()
            startup.startup.cb = ctypes.sizeof(startup)
            startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = inherited
            startup.attributes = ctypes.cast(attributes, ctypes.c_void_p)
            # An explicit executable plus a writable command line avoids ambiguous
            # Windows executable parsing. The child receives an exact Unicode block.
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(self.args))
            if any(type(k) is not str or type(v) is not str or '\0' in k + v or '=' in k
                   for k, v in environment.items()):
                raise QualificationDecline('unsupported owned process environment')
            block = ctypes.create_unicode_buffer('\0'.join(k + '=' + v for k, v in
                                                 sorted(environment.items(), key=lambda pair: pair[0].upper())) + '\0\0')
            if not self.api.CreateProcessW(self.args[0], command, None, None, True, CREATE_FLAGS,
                                          block, str(directory), ctypes.byref(startup), ctypes.byref(self.info)):
                raise _error('CreateProcessW with owned Job list')
            if not self.info.process or not self.info.thread or not self.info.pid or not self.info.tid:
                raise QualificationDecline('incomplete owned process creation record')
        finally:
            try:
                if initialized:
                    self.api.DeleteProcThreadAttributeList(attributes)
            finally:
                self._close_duplicates()

    def resume(self, job):
        member = W.BOOL()
        if not self.api.IsProcessInJob(self.info.process, job, ctypes.byref(member)):
            raise _error('IsProcessInJob before resume')
        if not member.value:
            raise QualificationDecline('created process is outside owned Job')
        previous = self.api.ResumeThread(self.info.thread)
        if previous == 0xffffffff:
            raise _error('ResumeThread')
        if previous != 1:
            raise QualificationDecline('unexpected primary-thread suspend count')
        if not self.api.CloseHandle(self.info.thread):
            raise _error('CloseHandle primary thread')
        self.info.thread = None

    def _exited(self):
        code = W.DWORD()
        if not self.api.GetExitCodeProcess(self.info.process, ctypes.byref(code)):
            raise _error('GetExitCodeProcess')
        self.returncode = int(code.value)
        return self.returncode

    def poll(self):
        if self.returncode is not None or not self.info.process:
            return self.returncode
        result = self.api.WaitForSingleObject(self.info.process, 0)
        if result == 0:
            return self._exited()
        if result == 258:
            return None
        raise _error('WaitForSingleObject poll')

    def wait(self, timeout=None):
        if self.returncode is not None or not self.info.process:
            return self.returncode
        milliseconds = 0xffffffff if timeout is None else max(0, min(0xfffffffe, math.ceil(timeout * 1000)))
        result = self.api.WaitForSingleObject(self.info.process, milliseconds)
        if result == 0:
            return self._exited()
        if result == 258:
            raise subprocess.TimeoutExpired(self.args, timeout)
        raise _error('WaitForSingleObject wait')

    def close_handles(self):
        errors = []
        try:
            self._close_duplicates()
        except OSError as error:
            errors.append(error)
        for name in ('thread', 'process'):
            handle = getattr(self.info, name)
            if handle:
                if self.api.CloseHandle(handle):
                    setattr(self.info, name, None)
                else:
                    errors.append(_error('CloseHandle child ' + name))
        if errors:
            raise errors[0]
