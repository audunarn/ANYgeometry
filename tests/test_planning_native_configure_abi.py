"""Private bytes-payload parser ABI; resource-only, no numerical qualification."""
import os
from pathlib import Path
import json
import pytest
from _planning_contract_helpers import require_qualified_planning_lane
from anygeometry._planning_qualification import QualificationDecline


def resource_native(request):
    from anygeometry import _planning_native as adapter
    if os.name!='nt':
        reason='planning resource helper requires Windows CPython with GIL'
        require_qualified_planning_lane(reason)
        with pytest.raises(QualificationDecline,match='^planning resource helper requires Windows CPython with GIL$') as caught:
            adapter.load()
        assert type(caught.value) is QualificationDecline
        request.node.user_properties.extend((('execution_scope','initial-native-helper-refusal'),
            ('original_test_body_executed',False),('native_parser_qualification',False)))
        return None
    native=adapter.load()
    request.node.user_properties.extend((('execution_scope','native-configure-ABI-assertions'),
                                         ('original_test_body_executed',True)))
    return native


def payload_case(kind):
    calls=[]
    class Bytes(bytes):
        def __bytes__(self):calls.append('bytes');raise AssertionError('Python bytes hook invoked')
        def __len__(self):calls.append('len');raise AssertionError('Python length hook invoked')
        def __getitem__(self,key):calls.append('getitem');raise AssertionError('Python item hook invoked')
    class Foreign:
        def __bytes__(self):calls.append('foreign bytes');raise AssertionError('foreign conversion invoked')
    if kind=='bytes':value=b''
    elif kind=='subclass':value=Bytes(b'')
    elif kind=='bytearray':value=bytearray()
    elif kind=='memoryview':value=memoryview(b'')
    elif kind=='str':value=''
    elif kind=='none':value=None
    elif kind=='object':value=Foreign()
    elif kind=='ctypes':
        import ctypes
        value=ctypes.create_string_buffer(b'')
    else:raise AssertionError(kind)
    return value,calls


@pytest.mark.parametrize('kind',('bytes','subclass'))
def test_configure_accepts_bytes_without_python_hooks_and_preserves_empty_owner(request,kind):
    native=resource_native(request)
    if native is None:return
    owner=native.new_owner();before=native.state(owner);payload,calls=payload_case(kind)
    # No Job exists: OSError6 proves parser acceptance without native mutation.
    with pytest.raises(OSError) as caught:native.configure(owner,9,payload)
    assert caught.value.winerror==6 and caught.value.job_operation=='SetInformationJobObject'
    assert native.state(owner)==before and calls==[]


@pytest.mark.parametrize('kind',('bytearray','memoryview','str','none','object','ctypes'))
def test_configure_rejects_nonbytes_without_hooks_or_owner_mutation(request,kind):
    native=resource_native(request)
    if native is None:return
    owner=native.new_owner();before=native.state(owner);payload,calls=payload_case(kind)
    with pytest.raises(TypeError):native.configure(owner,9,payload)
    assert native.state(owner)==before and calls==[]


def test_configure_keeps_payload_type_refusal_before_foreign_capsule(request):
    native=resource_native(request)
    if native is None:return
    with pytest.raises(TypeError):native.configure(object(),9,bytearray())
    with pytest.raises(ValueError):native.configure(object(),9,b'')


def test_actual_configure_readback_and_owner_retirement_are_valid(tmp_path,request):
    native=resource_native(request)
    if native is None:return
    from anygeometry._planning_process import OwnedTree
    cap=128*1024**2;tree=OwnedTree(tmp_path,memory_limit_bytes=cap)
    try:
        report=tree.memory_snapshot()
        assert report['limit_bytes']==cap and report['flags']==0x2200
        assert 0<=report['peak_job_bytes']<=cap
    finally:tree.close()
    assert tree.handle is None
    final=json.loads((tmp_path/'cleanup.json').read_text())
    assert final['verified'] and final['active_processes']==0
    assert final['memory']['limit_bytes']==cap


def test_public_adapter_copies_ctypes_to_the_private_bytes_contract(tmp_path,request):
    native=resource_native(request)
    if native is None:return
    import ctypes
    from anygeometry._planning_process import OwnedTree
    cap=128*1024**2;tree=OwnedTree(tmp_path,memory_limit_bytes=cap)
    try:
        value=tree._limits_type();value.basic.flags=0x2200;value.jm=cap
        with pytest.raises(TypeError):native.configure(tree.api.owner,9,value)
        assert tree.api.SetInformationJobObject(tree.handle,9,ctypes.byref(value),ctypes.sizeof(value))
        value.basic.flags=0;value.jm=cap*2
        actual=tree.memory_snapshot()
        assert actual['flags']==0x2200 and actual['limit_bytes']==cap
    finally:tree.close()
    assert tree.handle is None
    final=json.loads((tmp_path/'cleanup.json').read_text())
    assert final['verified'] and final['active_processes']==0
