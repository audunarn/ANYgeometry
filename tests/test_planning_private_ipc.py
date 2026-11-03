"""Private file IO sharing policy: real files, exact packets, bounded errors."""
import json,os,threading,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from anygeometry import _planning_executor as ipc
from anygeometry._planning_qualification import QualificationDecline


def denied(code=None,number=13):
    error=PermissionError(number,'original private IO denial')
    if code is not None:error.winerror=code
    return error


def clocked(monkeypatch,platform='nt'):
    clock=[0.];sleeps=[]
    monkeypatch.setattr(ipc,'os',SimpleNamespace(name=platform,getpid=os.getpid))
    monkeypatch.setattr(ipc.time,'monotonic',lambda:clock[0])
    def sleep(delay):sleeps.append(delay);clock[0]+=delay
    monkeypatch.setattr(ipc.time,'sleep',sleep)
    return clock,sleeps


@pytest.mark.parametrize('operation',['stat','read','write','replace'])
@pytest.mark.parametrize('code',[None,5,32,33])
def test_actual_packet_io_transient_denial_preserves_wire(tmp_path,monkeypatch,operation,code):
    path=tmp_path/'command-1.json';packet=ipc.pack({'x':[1,'value']},'command')
    ipc.write_packet(path,packet);expected=path.read_bytes()
    clock,sleeps=clocked(monkeypatch);calls=[];original=getattr(Path,{'read':'read_text','write':'write_text'}.get(operation,operation))
    def blocked(self,*args,**kwargs):
        if self in (path,path.with_suffix('.json.tmp')):
            calls.append(1)
            if len(calls)==1:raise denied(code)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,{'read':'read_text','write':'write_text'}.get(operation,operation),blocked)
    if operation in ('write','replace'):ipc.write_packet(path,packet)
    else:assert ipc.read_packet(path,'command')=={'x':[1,'value']}
    assert len(calls)==2 and sleeps==[.002] and clock[0]<=.05
    assert path.read_bytes()==expected


@pytest.mark.parametrize('platform,error',[('posix',denied()),('nt',denied(87)),('nt',denied(None,1)),('nt',OSError('unrelated'))])
def test_nonmatching_packet_io_denial_is_not_retried(tmp_path,monkeypatch,platform,error):
    path=tmp_path/'packet.json';ipc.write_packet(path,ipc.pack({},'kind'))
    _,sleeps=clocked(monkeypatch,platform);calls=[]
    def blocked(*args,**kwargs):calls.append(1);raise error
    monkeypatch.setattr(Path,'read_text',blocked)
    with pytest.raises(type(error)) as caught:ipc.read_packet(path,'kind')
    assert caught.value is error and calls==[1] and sleeps==[]


def test_permanent_write_denial_retains_last_error_temp_and_shared_budget(tmp_path,monkeypatch):
    path=tmp_path/'packet.json';clock,sleeps=clocked(monkeypatch);errors=[]
    def blocked(*args):
        error=denied();errors.append(error);raise error
    monkeypatch.setattr(Path,'replace',blocked)
    with pytest.raises(PermissionError) as caught:ipc.write_packet(path,ipc.pack({},'kind'))
    assert caught.value is errors[-1] and 1<=len(errors)<=8
    assert sum(sleeps)<=.05 and clock[0]<=.05
    assert path.with_suffix('.json.tmp').is_file() and not path.exists()


@pytest.mark.parametrize('kind',['JSON','checksum','schema','size'])
def test_corrupt_packet_validation_is_not_retried(tmp_path,monkeypatch,kind):
    path=tmp_path/'packet.json';packet=ipc.pack({},'kind')
    if kind=='checksum':packet['sha256']='0'*64
    if kind=='schema':packet['schema']='wrong'
    text='{' if kind=='JSON' else ' '*100003 if kind=='size' else json.dumps(packet)
    path.write_text(text);_,sleeps=clocked(monkeypatch)
    with pytest.raises(QualificationDecline):ipc.read_packet(path,'kind',limit=1 if kind=='size' else 4000000)
    assert sleeps==[]


def test_real_reader_holds_destination_until_first_replace_denial(tmp_path,monkeypatch):
    path=tmp_path/'reply.json';ipc.write_packet(path,ipc.pack({'old':1},'kind'))
    original=Path.replace;failures=[];seen=threading.Event()
    def observed(self,*args,**kwargs):
        try:return original(self,*args,**kwargs)
        except PermissionError as error:
            failures.append(error);seen.set();raise
    monkeypatch.setattr(Path,'replace',observed)
    with path.open('rb') as reader:
        if os.name=='nt':
            def release():seen.wait(.2);time.sleep(.004);reader.close()
            thread=threading.Thread(target=release);thread.start()
            try:ipc.write_packet(path,ipc.pack({'new':2},'kind'))
            finally:seen.set();thread.join()
            assert failures and not thread.is_alive()
        else:
            ipc.write_packet(path,ipc.pack({'new':2},'kind'))
            assert failures==[]
    assert ipc.read_packet(path,'kind')=={'new':2}
