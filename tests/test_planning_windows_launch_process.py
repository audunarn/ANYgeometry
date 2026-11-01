"""Fake-host death and handle isolation; every observed PID is our own child."""
import ctypes
from ctypes import wintypes as W
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from anygeometry import _planning_process as process


def test_only_declared_stdio_handles_cross_into_fake_child(tmp_path,monkeypatch):
    fake=tmp_path/'fake source';fake.mkdir()
    script=fake/'_planning_child.py'
    script.write_text('''import ctypes,json,os,sys
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetFinalPathNameByHandleW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong]
k.GetFinalPathNameByHandleW.restype=ctypes.c_ulong
b=ctypes.create_unicode_buffer(32768)
invalid=None
try:n=k.GetFinalPathNameByHandleW(int(os.environ['W0_UNLISTED_HANDLE']),b,len(b),0)
except OSError as error:
 if error.winerror not in (6,-1073741816):raise
 invalid=error.winerror;n=0
Path(sys.argv[2],'inherited.json').write_text(json.dumps({'path':b.value if n else None,'invalid_handle':invalid,'unicode':os.environ['W0_UNICODE']}))
print('owned stdout',flush=True)
print('owned stderr',file=sys.stderr,flush=True)
''')
    monkeypatch.setattr(process,'__file__',str(fake/'_planning_process.py'))
    tree=process.OwnedTree(tmp_path);borrowed=tmp_path/'unlisted unique file.txt'
    with borrowed.open('wb') as f:
        handle=msvcrt.get_osfhandle(f.fileno());os.set_handle_inheritable(handle,True)
        monkeypatch.setenv('W0_UNLISTED_HANDLE',str(handle));monkeypatch.setenv('W0_UNICODE','snow-\u2603')
        try:
            child=tree.spawn('worker',tmp_path/'child space',10)
            assert child.wait(timeout=10)==0
            report=json.loads((tmp_path/'child space/inherited.json').read_text())
            assert report['unicode']=='snow-\u2603'
            assert report['path'] is None or Path(report['path'].removeprefix('\\\\?\\')).resolve()!=borrowed.resolve()
            assert (tmp_path/'child space/stdout.txt').read_text().strip()=='owned stdout'
            assert (tmp_path/'child space/stderr.txt').read_text().strip()=='owned stderr'
        finally:tree.close()
    assert json.loads((tmp_path/'cleanup.json').read_text())['active_processes']==0


@pytest.mark.parametrize('phase',['after-native-create','before-resume','after-resume'])
def test_fake_creator_death_retires_its_owned_child(tmp_path,phase):
    fake=tmp_path/'fake';fake.mkdir()
    (fake/'_planning_child.py').write_text('import time\ntime.sleep(30)\n')
    host=tmp_path/'fake_host.py'
    host.write_text('''import ctypes,json,os,sys
from pathlib import Path
from ctypes import wintypes as W
sys.path.insert(0,sys.argv[1])
from anygeometry import _planning_process as process
folder=Path(sys.argv[2]);phase=sys.argv[3]
process.__file__=str(folder/'fake/_planning_process.py')
tree=process.OwnedTree(folder)
def record():
 child=tree.children[0]
 (folder/'owned-child.json').write_text(json.dumps({'pid':child.pid,'command':tree.commands[0]}))
if phase=='after-native-create':
 from anygeometry._planning_windows_launch import _bind
 _bind(tree.api);create=tree.api.CreateProcessW
 def interrupted(*a):
  result=create(*a)
  if result:record();os._exit(41)
  return result
 tree.api.CreateProcessW=interrupted
elif phase=='before-resume':
 def interrupted(*a):record();os._exit(42)
 tree.api.ResumeThread=interrupted
child=tree.spawn('worker',folder/'child',10)
record();os._exit(43)
''')
    source=str(Path(process.__file__).resolve().parents[1])
    command=[sys.executable,'-B',str(host),source,str(tmp_path),phase]
    started=time.perf_counter();execution=dict(command=command,cwd=str(tmp_path),wait_cap_seconds=15,phase=phase)
    with (tmp_path/'host-stdout.txt').open('wb') as out,(tmp_path/'host-stderr.txt').open('wb') as err:
        child=subprocess.Popen(command,cwd=tmp_path,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        execution['owned_pid']=child.pid
        try:code=child.wait(timeout=15);execution.update(status='exited',exit_code=code)
        except subprocess.TimeoutExpired:
            observed=psutil.Process(child.pid).cmdline();execution.update(status='timeout',observed_command=observed)
            assert observed==command
            child.kill();child.wait(timeout=2);raise
        finally:
            execution['process_wall_seconds']=time.perf_counter()-started
            (tmp_path/'host-execution.json').write_text(json.dumps(execution,indent=2))
    assert code=={'after-native-create':41,'before-resume':42,'after-resume':43}[phase]
    record=json.loads((tmp_path/'owned-child.json').read_text())
    # PID came only from this fake host's pre-registered native creation record.
    def retired():
        (tmp_path/'owned-retirement.json').write_text(json.dumps(dict(verified=True,active_processes=0,record=record)))
    try:owned=psutil.Process(record['pid'])
    except psutil.NoSuchProcess:retired();return
    try:observed=owned.cmdline()
    except psutil.NoSuchProcess:retired();return
    assert observed==record['command']
    try:owned.wait(timeout=5)
    except psutil.TimeoutExpired:pytest.fail('owned child survived fake creator death')
    retired()
