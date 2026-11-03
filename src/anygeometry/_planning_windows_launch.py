"""Windows child adapter; all handle ownership transitions execute natively.

The ctypes-shaped API is a fault/query seam, not a raw acquiring WinDLL. Owners
are registered before acquisition and authoritative slots live in the helper.
"""
import ctypes
from ctypes import wintypes as W
import math
import subprocess
from ._planning_qualification import QualificationDecline

class ProcessInfo(ctypes.Structure):
    _fields_ = [('process', W.HANDLE), ('thread', W.HANDLE), ('pid', W.DWORD), ('tid', W.DWORD)]

def _error(operation):
    error = ctypes.WinError(ctypes.get_last_error())
    error.job_operation = operation
    return error

def _bind(api):
    # Compatibility seam for existing probes. NativeAPI has no raw owning calls.
    from ._planning_native import NativeAPI
    if not isinstance(api, NativeAPI):
        raise QualificationDecline('native resource owner required')

class WindowsChild:
    def __init__(self, api, command):
        _bind(api)
        self.api, self.args = api, list(command)
        self.index = api.reserve()
        self.returncode = None

    @property
    def info(self):
        state = self.api.state(self.index)
        return ProcessInfo(state['process'], state['thread'], state['pid'], state['tid'])
    @property
    def pid(self): return self.info.pid or None
    @property
    def _handle(self): return self.info.process
    @property
    def _duplicates(self):
        # Historical name retained for test/report compatibility: these now are
        # CreateFileW-owned stdio slots, not CRT/DuplicateHandle copies.
        return [value for value in self.api.state(self.index)['io'] if value]

    def _close_duplicates(self):
        errors=[]
        for kind in range(2,5):
            if not self.api.close_child(self.index,kind): errors.append(_error('CloseHandle inherited I/O'))
        if errors: raise errors[0]

    def create(self, job, directory, environment, stdout, stderr):
        if not job or job != self.api.state()['job']:
            raise QualificationDecline('owned Job is closed before creation')
        if any(type(k) is not str or type(v) is not str or not k or '=' in k or '\0' in k or '\0' in v
               for k,v in environment.items()):
            raise QualificationDecline('invalid owned process environment')
        if not self.args or any(type(value) is not str or '\0' in value for value in self.args):
            raise QualificationDecline('invalid owned process command')
        block='\0'.join(k+'='+v for k,v in sorted(environment.items(), key=lambda item:item[0].upper()))+'\0\0'
        record=ProcessInfo()
        try:
            if not self.api.CreateProcessW(self.args[0], subprocess.list2cmdline(self.args), self.index,
                                           stdout, stderr, block, directory, ctypes.byref(record)):
                raise _error('CreateProcessW native transaction')
            if not all((record.process,record.thread,record.pid,record.tid)):
                raise QualificationDecline('incomplete owned process creation record')
        finally:
            self._close_duplicates()

    def resume(self, job):
        if not job:
            raise QualificationDecline('owned Job is closed before resume')
        member=W.BOOL()
        if not self.api.IsProcessInJob(self.info.process,job,ctypes.byref(member)):
            raise _error('IsProcessInJob before resume')
        if not member.value: raise QualificationDecline('created process is outside owned Job')
        previous=self.api.resume_child(self.index)
        if previous==0xffffffff: raise _error('ResumeThread')
        if previous!=1: raise QualificationDecline('unexpected primary-thread suspend count')
        if not self.api.close_child(self.index,1): raise _error('CloseHandle primary thread')

    def poll(self):
        if self.returncode is not None or not self.info.process: return self.returncode
        result,code=self.api.wait(self.index,0)
        if result==0: self.returncode=code
        elif result!=258: raise OSError('unexpected native wait result')
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None or not self.info.process: return self.returncode
        milliseconds=0xffffffff if timeout is None else max(0,min(0xfffffffe,math.ceil(timeout*1000)))
        result,code=self.api.wait(self.index,milliseconds)
        if result==258: raise subprocess.TimeoutExpired(self.args,timeout)
        if result!=0: raise OSError('unexpected native wait result')
        self.returncode=code
        return code

    def close_handles(self):
        errors=[]
        try: self._close_duplicates()
        except OSError as error: errors.append(error)
        for name,kind in (('thread',1),('process',0)):
            if not self.api.close_child(self.index,kind): errors.append(_error('CloseHandle child '+name))
        if errors: raise errors[0]
