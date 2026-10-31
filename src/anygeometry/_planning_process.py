"""Windows-only owned process containment for the internal PP2 skeleton."""
import ctypes
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from ._planning_qualification import QualificationDecline


class OwnedTree:
    def __init__(self, directory):
        if os.name != 'nt':
            raise QualificationDecline('owned planner requires Windows Job containment')
        self.directory = Path(directory)
        self.children = []
        self.commands = []
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        for name, args, result in (
            ('CreateJobObjectW', [ctypes.c_void_p, W.LPCWSTR], W.HANDLE),
            ('AssignProcessToJobObject', [W.HANDLE, W.HANDLE], W.BOOL),
            ('SetInformationJobObject', [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD], W.BOOL),
            ('QueryInformationJobObject', [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p], W.BOOL),
            ('TerminateJobObject', [W.HANDLE, W.UINT], W.BOOL),
            ('CloseHandle', [W.HANDLE], W.BOOL)):
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError()
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', W.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                        ('active_limit', W.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', W.DWORD), ('scheduling', W.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('ro', 'wo', 'oo', 'rb', 'wb', 'ob')]
        class Limits(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('pm', ctypes.c_size_t),
                        ('jm', ctypes.c_size_t), ('ppm', ctypes.c_size_t), ('pjm', ctypes.c_size_t)]
        limits = Limits()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway permission.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.api.CloseHandle(self.handle)
            self.handle = None
            raise ctypes.WinError()

    def spawn(self, role, directory, timeout):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).with_name('_planning_child.py').resolve()
        command = [sys.executable, '-B', '-u', str(script), role, str(directory), str(timeout)]
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', TEMP=str(directory), TMP=str(directory))
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        # Explicit numerical mode is not overridden. Unsupported options are not
        # silently converted into the canary diagnostic contract.
        with (directory/'stdout.txt').open('wb') as out, (directory/'stderr.txt').open('wb') as err:
            child = subprocess.Popen(command, cwd=directory, env=env, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW | 4)
        self.children.append(child)
        self.commands.append(command)
        try:
            if not self.api.AssignProcessToJobObject(self.handle, W.HANDLE(int(child._handle))):
                raise ctypes.WinError()
            native = ctypes.WinDLL('ntdll')
            native.NtResumeProcess.argtypes = [W.HANDLE]
            native.NtResumeProcess.restype = W.LONG
            if native.NtResumeProcess(W.HANDLE(int(child._handle))) != 0:
                raise RuntimeError('owned process resume failed')
        except BaseException:
            # Our suspended child may not yet belong to the job.
            if child.poll() is None:
                if psutil.Process(child.pid).cmdline() != command:
                    raise RuntimeError('owned suspended command changed')
                child.kill()
                child.wait(timeout=5)
            raise
        return child

    def wait(self, child, deadline):
        try:
            code = child.wait(timeout=max(.001, deadline-time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise QualificationDecline('owned process deadline') from exc
        if code != 0:
            raise QualificationDecline('owned process exit: ' + str(code))

    def _pids(self):
        buffer = ctypes.create_string_buffer(8 + 64*ctypes.sizeof(ctypes.c_void_p))
        if not self.api.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None):
            raise ctypes.WinError()
        count = ctypes.c_uint32.from_buffer(buffer, 4).value
        if count > 64:
            raise RuntimeError('owned process list overflow')
        return tuple(ctypes.c_size_t.from_buffer(buffer, 8+i*ctypes.sizeof(ctypes.c_void_p)).value
                     for i in range(count))

    def _owned_console_command(self, command):
        # Windows can add this hidden console helper to our Job during startup.
        # Membership was obtained from THIS Job, not a machine process scan.
        # Accept only the measured system helper command, never an image-name kill.
        if len(command) != 2 or command[1] != '0x4':
            return False
        self.api.GetSystemDirectoryW.argtypes = [W.LPWSTR, W.UINT]
        self.api.GetSystemDirectoryW.restype = W.UINT
        buffer = ctypes.create_unicode_buffer(32768)
        count = self.api.GetSystemDirectoryW(buffer, len(buffer))
        if not 0 < count < len(buffer):
            raise ctypes.WinError()
        executable = Path(command[0].removeprefix('\\??\\')).resolve()
        return executable == (Path(buffer.value)/'conhost.exe').resolve()

    def close(self):
        if self.handle is None:
            return
        checked = []
        try:
            for pid in self._pids():
                try:
                    command = psutil.Process(pid).cmdline()
                except psutil.NoSuchProcess:
                    continue
                checked.append(dict(pid=pid, command=command))
                (self.directory/'cleanup-preflight.json').write_text(
                    json.dumps(checked, indent=2), encoding='utf-8')
                if self._owned_console_command(command):
                    checked[-1]['kind'] = 'owned-system-console'
                    continue
                # Only Job members are inspected. Nested workers use the same
                # exact executable/script and task-owned descendant directories.
                script = str(Path(__file__).with_name('_planning_child.py').resolve())
                if (len(command) != 7 or command[:4] != [sys.executable, '-B', '-u', script]
                        or command[4] not in ('planner', 'worker')
                        or not Path(command[5]).resolve().is_relative_to(self.directory.resolve())):
                    raise RuntimeError('unexpected owned process command; see cleanup-preflight.json')
            if not self.api.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError()
            end = time.monotonic()+5
            while self._pids() and time.monotonic() < end:
                time.sleep(.02)
            if self._pids():
                raise RuntimeError('owned processes remain')
            for child in self.children:
                child.wait(timeout=1)
            report = dict(verified=True, active_processes=0, checked=checked, commands=self.commands)
            (self.directory/'cleanup.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            self.api.CloseHandle(self.handle)
            self.handle = None
