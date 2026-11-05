"""Synthetic protocol records, NOT actual memory/cleanup observations."""
import json
from pathlib import Path

import pytest
from anygeometry import _planning_executor as executor
from anygeometry._planning_qualification import QualificationDecline
import _planning_platform_contract as contract
from test_planning_memory_process import _controlled_terminal512


def _records(tmp_path,monkeypatch,reuse,stages):
    error,control,source,document,request=_controlled_terminal512(tmp_path,monkeypatch,reuse)
    folder=control.directory/('session' if reuse else 'pair')
    if not reuse and stages[0]=='face':
        folder.rename(control.directory/'face');folder=control.directory/'face'
    _,session=executor.read_packet(folder/'session.json','session',with_digest=True)
    previous=session;digests=[]
    for sequence,stage in enumerate(stages):
        body=dict(session=session,sequence=sequence,previous=previous,stage=stage,
                  tasks=[] if stage=='finish' else [dict(kind=stage)])
        packet=executor.pack(body,'session-command')
        executor.write_packet(folder/f'command-{sequence}.json',packet)
        previous=packet['sha256'];digests.append(previous)
    result=executor.read_packet(folder/'result.json','result')
    result['request_sha256']=previous
    executor.write_packet(folder/'result.json',executor.pack(result,'result'))
    return error,control,source,document,request,folder,session,digests


def _assert(records,reuse):
    error,control,source,document,request,*_=records
    contract.assert_unsupported512_terminal(error,reuse=reuse,control=control,source=source,
        document=document,calls=['candidate'],request=request)


@pytest.mark.parametrize('reuse,stages',[(True,('pair','face')),(True,('pair','finish')),
    (True,('pair','face','finish')),(False,('pair','finish')),(False,('face','finish'))])
def test_valid_later_decline_chain_keeps_exact_latest_binding(tmp_path,monkeypatch,reuse,stages,request):
    records=_records(tmp_path,monkeypatch,reuse,stages)
    _assert(records,reuse)
    summary=json.loads((records[1].directory/'unsupported512-terminal.json').read_text())
    assert summary['failures'][0]['command_sha256']==records[-1][-1]
    assert summary['parallel_qualified'] is False
    request.node.user_properties.append(('execution_scope','synthetic-resource-protocol-validation'))


@pytest.mark.parametrize('reuse,stages',[(True,('pair',)),(False,('pair',)),(False,('face',))])
def test_original_command_zero_decline_remains_valid(tmp_path,monkeypatch,reuse,stages):
    _assert(_records(tmp_path,monkeypatch,reuse,stages),reuse)


@pytest.mark.parametrize('fault',('stale','result-session','command-session','previous','sequence',
    'missing','none','repeat','reorder','empty-data','finish-not-last','nonreuse-face',
    'extra-index','alias','wrong-case','tamper','folder','symlink','bounds'))
def test_chain_never_accepts_stale_ambiguous_or_unbound_records(tmp_path,monkeypatch,fault):
    reuse=fault!='nonreuse-face'
    stages=(('pair','face') if fault=='empty-data' else ('pair','face','finish')) if reuse else ('pair','face')
    records=_records(tmp_path,monkeypatch,reuse,stages)
    _,control,_,_,_,folder,session,digests=records
    target=folder/'command-1.json'
    if fault in ('stale','result-session'):
        body=executor.read_packet(folder/'result.json','result')
        body['request_sha256']=digests[0] if fault=='stale' else digests[-1]
        if fault=='result-session':body['session_sha256']='0'*64
        executor.write_packet(folder/'result.json',executor.pack(body,'result'))
    elif fault in ('command-session','previous','sequence','repeat','reorder','empty-data','finish-not-last'):
        body=executor.read_packet(target,'session-command')
        if fault=='command-session':body['session']='0'*64
        elif fault=='previous':body['previous']=session
        elif fault=='sequence':body['sequence']=0
        elif fault=='empty-data':body['tasks']=[]
        else:
            body['stage']='pair' if fault=='repeat' else 'finish'
            body['tasks']=[] if body['stage']=='finish' else [dict(kind='pair')]
            if fault=='reorder':
                zero=executor.read_packet(folder/'command-0.json','session-command');zero['stage']='face';zero['tasks']=[dict(kind='face')]
                executor.write_packet(folder/'command-0.json',executor.pack(zero,'session-command'))
        executor.write_packet(target,executor.pack(body,'session-command'))
        # Keep all later envelopes/result internally bound to the changed bytes;
        # the intended metadata fault, not an incidental stale final digest, fails.
        previous=executor.read_packet(target,'session-command',with_digest=True)[1]
        for i in range(2,len(stages)):
            later=executor.read_packet(folder/f'command-{i}.json','session-command')
            later['previous']=previous
            packet=executor.pack(later,'session-command');previous=packet['sha256']
            executor.write_packet(folder/f'command-{i}.json',packet)
        result=executor.read_packet(folder/'result.json','result');result['request_sha256']=previous
        executor.write_packet(folder/'result.json',executor.pack(result,'result'))
    elif fault=='missing':target.unlink()
    elif fault=='none':
        for i in range(3):(folder/f'command-{i}.json').unlink()
    elif fault in ('extra-index','alias','wrong-case'):
        name={'extra-index':'command-3.json','alias':'command-01.json','wrong-case':'COMMAND-7.json'}[fault]
        (folder/name).write_bytes((folder/'command-0.json').read_bytes())
    elif fault=='tamper':
        packet=json.loads(target.read_text());packet['sha256']='0'*64;target.write_text(json.dumps(packet))
    elif fault=='folder':folder.rename(control.directory/'pair')
    elif fault=='symlink':
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda path:path==target or original(path))
    elif fault=='bounds':target.write_bytes(b'x'*8_100_001)
    with pytest.raises((AssertionError,QualificationDecline,FileNotFoundError)):
        _assert(records,reuse)


@pytest.mark.parametrize('fault',('valid','file-bound','total-bound'))
def test_validated_copy_includes_only_bounded_optional_protocol_files(tmp_path,monkeypatch,fault):
    records=_records(tmp_path,monkeypatch,True,('pair','face','finish'))
    _assert(records,True)
    _,control,_,_,request,folder,*_=records
    summary=json.loads((control.directory/'unsupported512-terminal.json').read_text())
    root=tmp_path/'checkout';(root/'tests').mkdir(parents=True)
    monkeypatch.setattr(contract,'__file__',str(root/'tests/_planning_platform_contract.py'))
    monkeypatch.setenv('CI','true')
    (folder/'plan.json').write_bytes(b'opaque synthetic plan')
    if fault=='file-bound':(folder/'command-1.json').write_bytes(b'x'*8_100_001)
    elif fault=='total-bound':
        for name in ('cleanup.json','workers-cleanup.json','memory-limit.json','session.json','plan.json'):
            (folder/name).write_bytes(b'x'*6_500_000)
    if fault!='valid':
        with pytest.raises(AssertionError):contract._retain_unsupported512_evidence(summary,[folder],request)
    else:
        destination=contract._retain_unsupported512_evidence(summary,[folder],request)
        kept=json.loads((destination/'summary.json').read_text())
        assert len(kept['retained_files'])==9
        assert {p.name for p in (destination/'session').iterdir()}=={
            'cleanup.json','workers-cleanup.json','memory-limit.json','session.json',
            'command-0.json','command-1.json','command-2.json','result.json','plan.json'}
