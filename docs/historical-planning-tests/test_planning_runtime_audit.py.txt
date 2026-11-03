"""Actual hooks only in disposable subprocesses; no removal or callback probing."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

CHILD=r'''
import sys,json
import _audit_fixture as fixture
from anygeometry import _planning_codec_registry as adapter
from anygeometry import _planning_runtime_inspector as native
mode=sys.argv[1]
calls=[]
token=native.snapshot()[0]
value=adapter.snapshot()
if mode=='allocation_python':
    def wire_zip(names,values):
        if values is adapter._ANCHOR[8]:
            sys.addaudithook(lambda *args:calls.append(args[0]))
        return zip(names,values)
    adapter.zip=wire_zip
    try: adapter.identity()
    except RuntimeError as error: assert 'audit' in str(error)
    else: raise AssertionError('wire construction published stale absence')
    assert not calls,calls
    print(json.dumps(dict(mode=mode,callbacks=0,refused=True)))
    sys.exit(0)
if mode.endswith('python'):
    sys.addaudithook(lambda *args:calls.append(args[0]))
else:
    fixture.install()
fixture.reset();calls.clear()
if mode.startswith('post'):
    assert native.matches(token) is False,'native matches accepted new hook'
    assert value.matches() is False,'adapter matches accepted new hook'
else:
    for function in (native.snapshot,native.identity,adapter.snapshot,adapter.identity):
        try: function()
        except RuntimeError as error:
            assert 'audit' in str(error),str(error)
        else: raise AssertionError('hook presence was accepted')
assert not calls,calls
assert fixture.count()==0,'inspector emitted an audit event'
print(json.dumps(dict(mode=mode,callbacks=0,refused=True)))
'''

@pytest.mark.parametrize('mode',['initial_python','post_python','initial_native','post_native','allocation_python'])
def test_actual_audit_hooks_refuse_without_invocation(mode,tmp_path):
    import anygeometry
    import _audit_fixture
    paths=[str(Path(anygeometry.__file__).resolve().parent.parent),str(Path(_audit_fixture.__file__).resolve().parent)]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(paths))
    result=subprocess.run([sys.executable,'-B','-c',CHILD,mode],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=15)
    (tmp_path/'stdout.txt').write_text(result.stdout)
    (tmp_path/'stderr.txt').write_text(result.stderr)
    assert result.returncode==0,result.stderr

def test_absence_and_audit_layout_are_bound():
    from anygeometry._planning_codec_registry import snapshot
    value=snapshot()
    assert value.profile['audit_hooks_absent'] is True
    assert value.profile['schema']==2
    assert value.profile['interpreter_audit_offset']>value.profile['interpreter_codecs_offset']
    assert value.profile['runtime_audit_head_offset']>value.profile['runtime_audit_mutex_offset']
    assert value.matches()
