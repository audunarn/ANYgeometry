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
if os.name == 'nt':
    # P and W must see the same launch closure before admission capture.
    from ._planning_windows_launch import WindowsChild


def validate_memory_limit(value):
    if value is not None and (type(value) is not int or not 0<value<2**(8*ctypes.sizeof(ctypes.c_size_t))):
        raise QualificationDecline('invalid owned-tree committed-memory limit')


def _windows_error(operation):
    # WinDLL(use_last_error=True) swaps a ctypes thread-local copy. WinError()
    # without this explicit code reads the native slot after it was restored.
    code=ctypes.get_last_error()
    error=ctypes.WinError(code)
    error.job_operation=operation
    return error


def _memory_failure(error,requested):
    return dict(operation=getattr(error,'job_operation',None),
                winerror=getattr(error,'winerror',None),requested_bytes=requested)


class OwnedTree:
    def __init__(self, directory, *, memory_limit_bytes=None):
        validate_memory_limit(memory_limit_bytes)
        self.memory_limit_bytes=memory_limit_bytes
        if os.name != 'nt':
            raise QualificationDecline('owned planner requires Windows Job containment')
        self.directory = Path(directory)
        if memory_limit_bytes is not None and not self.directory.is_dir():
            raise QualificationDecline('capped owned-tree directory must already exist')
        self._memory_observation=None
        self.children = []
        self.commands = []
        self._handle_error_sequence = 0
        self._retiring = False
        from ._planning_native import NativeAPI
        self.api = NativeAPI()
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
            raise _windows_error('CreateJobObjectW')
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
        self._limits_type=Limits
        limits = Limits()
        limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway permission.
        if memory_limit_bytes is not None:
            limits.basic.flags|=0x200  # JOB_MEMORY: sum of committed memory, not RSS.
            limits.jm=memory_limit_bytes
        try:
            if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise _windows_error('SetInformationJobObject')
            if memory_limit_bytes is not None:
                report=self.memory_snapshot()
                (self.directory/'memory-limit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        except BaseException as exc:
            failure=_memory_failure(exc,memory_limit_bytes)
            closed=bool(self.api.CloseHandle(self.handle))
            close_error=None if closed else _windows_error('CloseHandle')
            if closed:self.handle = None
            if memory_limit_bytes is not None and closed:
                # No child can have been created before constructor admission.
                report=dict(verified=True,active_processes=0,checked=[],commands=[],
                            memory_error=type(exc).__name__+': '+str(exc),started_children=0,
                            memory_observation=self._memory_observation,memory_failure=failure)
                try:(self.directory/'cleanup.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                except OSError:pass  # Preserve the original failed admission.
            if not closed:raise close_error from exc
            raise

    def memory_snapshot(self):
        self._memory_observation=None
        if self.memory_limit_bytes is None:return None
        if self.handle is None:raise QualificationDecline('owned-tree memory enforcement unavailable')
        limits=self._limits_type()
        if not self.api.QueryInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits),None):
            raise _windows_error('QueryInformationJobObject')
        report=dict(requested_bytes=self.memory_limit_bytes,limit_bytes=int(limits.jm),flags=int(limits.basic.flags),
                    peak_job_bytes=int(limits.pjm),peak_process_bytes=int(limits.ppm))
        self._memory_observation=report
        if limits.basic.flags!=0x2200 or limits.jm!=self.memory_limit_bytes or limits.pjm>self.memory_limit_bytes:
            raise QualificationDecline('owned-tree memory enforcement differs from request: '+str(report))
        return report

    def spawn(self, role, directory, timeout):
        if self._retiring or not self.handle:
            raise QualificationDecline('owned Job is retiring or closed')
        self.memory_snapshot()
        if role not in ('planner','worker','session-planner','session-worker'):
            raise QualificationDecline('unsupported owned process role')
        directory = Path(directory).resolve()
        if not directory.is_relative_to(self.directory.resolve()):
            raise QualificationDecline('owned process directory escapes tree')
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
        child=WindowsChild(self.api,command)
        # Register preallocated native output storage before CreateProcessW can
        # write any process/thread handle, including interrupted creation.
        self.children.append(child)
        self.commands.append(command)
        try:
            child.create(self.handle,directory,env,directory/'stdout.txt',directory/'stderr.txt')
            child.resume(self.handle)
        except BaseException:
            # Job-list assignment is part of native creation, never a later
            # Popen assignment window. Retire the entire tentative tree on error.
            try:self.close()
            except Exception as cleanup:
                from ._planning_executor import PlannerCleanupError
                raise PlannerCleanupError('owned launch rollback failed') from cleanup
            raise
        return child

    @property
    def handle(self):
        # Native storage is authoritative even when a signal interrupted the
        # Python caller immediately after acquisition or close returned.
        return self.api.state()['job'] or None

    @handle.setter
    def handle(self, value):
        if value != (self.api.state()['job'] or None):
            raise RuntimeError('cannot overwrite native Job ownership')

    def wait(self, child, deadline, *, poll=None):
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                from ._planning_executor import PlannerDeadline
                raise PlannerDeadline('owned process deadline')
            if poll is not None:poll()
            remaining=deadline-time.monotonic()
            if remaining<=0:
                from ._planning_executor import PlannerDeadline
                raise PlannerDeadline('owned process deadline')
            try:
                code = child.wait(timeout=min(.05,remaining) if poll is not None else remaining)
                break
            except subprocess.TimeoutExpired as exc:
                if poll is not None:continue
                from ._planning_executor import PlannerDeadline
                raise PlannerDeadline('owned process deadline') from exc
        if poll is not None:poll()
        if code != 0:
            raise QualificationDecline('owned process exit: ' + str(code))

    def _pids(self):
        buffer = ctypes.create_string_buffer(8 + 64*ctypes.sizeof(ctypes.c_void_p))
        if not self.api.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None):
            raise _windows_error('QueryInformationJobObject')
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

    def _close_child_handles(self):
        errors=[]
        for child in self.children:
            if hasattr(child,'close_handles'):
                try:child.close_handles()
                except Exception as error:errors.append(error)
        return errors

    def _raise_handle_errors(self,errors):
        if not errors:return
        self._handle_error_sequence+=1
        failure=dict(verified=False,job_handle_retained=self.handle is not None,
            attempt=self._handle_error_sequence,
            errors=[dict(operation=getattr(error,'job_operation',None),
                         winerror=getattr(error,'winerror',None),error=str(error)) for error in errors])
        payload=json.dumps(failure,indent=2)
        try:
            # Exclusive creation preserves every failure, including retries.
            history=self.directory/f'handle-cleanup-error-{self._handle_error_sequence:06d}.json'
            with history.open('x',encoding='utf-8') as stream:stream.write(payload)
            try:
                with (self.directory/'handle-cleanup-error.json').open('x',encoding='utf-8') as stream:
                    stream.write(payload)
            except FileExistsError:
                pass  # The first failure remains the canonical terminal record.
        except OSError as artifact_error:raise errors[0] from artifact_error
        raise errors[0]

    def close(self):
        self._retiring = True
        if self.handle is None:
            self._raise_handle_errors(self._close_child_handles())
            return
        checked = []
        try:
            memory=None;memory_error=None;self._memory_observation=None
            try:memory=self.memory_snapshot()
            except BaseException as exc:memory_error=exc
            if self.memory_limit_bytes is not None:
                # Keep unverified accounting even if later containment cleanup
                # fails. This record never claims zero remaining processes.
                observation=dict(phase='before_cleanup',memory_observation=self._memory_observation,
                    memory_error=None if memory_error is None else type(memory_error).__name__+': '+str(memory_error),
                    memory_failure=None if memory_error is None else _memory_failure(memory_error,self.memory_limit_bytes))
                try:(self.directory/'memory-retirement.json').write_text(json.dumps(observation,indent=2),encoding='utf-8')
                except BaseException as exc:
                    if memory_error is None:memory_error=exc
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
                        or command[4] not in ('planner', 'worker', 'session-planner', 'session-worker')
                        or not Path(command[5]).resolve().is_relative_to(self.directory.resolve())):
                    raise RuntimeError('unexpected owned process command; see cleanup-preflight.json')
            if not self.api.TerminateJobObject(self.handle, 1):
                raise _windows_error('TerminateJobObject')
            end = time.monotonic()+5
            while self._pids() and time.monotonic() < end:
                time.sleep(.02)
            if self._pids():
                raise RuntimeError('owned processes remain')
            for child in self.children:
                child.wait(timeout=1)
            if self.memory_limit_bytes is not None:
                # The pre-cleanup sample is evidence, not the final verdict:
                # live children can commit between that sample and retirement.
                # Query again after verified zero while the Job handle is live.
                memory=None
                try:memory=self.memory_snapshot()
                except BaseException as exc:
                    if memory_error is None:memory_error=exc
            report = dict(verified=True, active_processes=0, checked=checked, commands=self.commands)
            if memory is not None:report['memory']=memory
            if memory_error is not None:report['memory_error']=type(memory_error).__name__+': '+str(memory_error)
            if memory_error is not None:report['memory_failure']=_memory_failure(memory_error,self.memory_limit_bytes)
            if self.memory_limit_bytes is not None:
                report['memory_observation']=self._memory_observation
                report['memory_phase']='after_cleanup'
            (self.directory/'cleanup.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            if memory_error is not None:raise memory_error
        finally:
            handle_errors=self._close_child_handles()
            if self.api.CloseHandle(self.handle):
                self.handle = None
            else:handle_errors.append(_windows_error('CloseHandle owned Job'))
            self._raise_handle_errors(handle_errors)
