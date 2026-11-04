"""Metadata refusal attribution only; never legacy-runtime qualification."""
import collections.abc
import importlib.machinery
import sys
import types

import pytest

from anygeometry import _planning_code as code
from anygeometry._planning_qualification import QualificationDecline


PATH='anygeometry._planning_qualification._CLEAR.value'
REASON='unsupported frozen collections.abc implementation alias'


def test_actual_infrastructure_keeps_alias_unsupported(request):
    implementation=sys.modules['_collections_abc']
    value=collections.abc.MutableMapping.clear
    alias=(value.__globals__ is vars(implementation) and
           value.__globals__ is not vars(sys.modules['collections.abc']))
    if alias:
        with pytest.raises(QualificationDecline,match='^'+REASON+'$'):
            code.InfrastructurePin('host')
        request.node.user_properties.extend((('execution_scope','initial-alias-refusal'),
            ('postcapture_qualification',False)))
    else:
        # The qualified runtime does not have the divergent implementation alias.
        captured=code.InfrastructurePin('host')
        assert captured.wire['role']=='host'
        request.node.user_properties.append(('execution_scope','metadata-infrastructure-capture'))


def test_canonical_interface_is_precisely_attributed():
    value=collections.abc.MutableMapping.clear
    implementation=sys.modules['_collections_abc']
    expected=(value.__globals__ is vars(implementation) and
              value.__globals__ is not vars(sys.modules['collections.abc']))
    assert code._unsupported_frozen_collections_alias(PATH,value) is expected


@pytest.mark.parametrize('fault',('path','body','globals','defaults','kwdefaults','closure',
    'public_module','implementation_module','class','member','importer'))
def test_noncanonical_interface_is_not_attributed(monkeypatch,fault):
    value=collections.abc.MutableMapping.clear
    path=PATH
    if fault=='path':path+='other'
    elif fault=='body':
        value=types.FunctionType((lambda self:None).__code__,value.__globals__,'clear')
    elif fault=='globals':value=types.FunctionType(value.__code__,dict(value.__globals__),'clear')
    elif fault in ('defaults','kwdefaults'):
        value=types.FunctionType(value.__code__,value.__globals__,'clear')
        if fault=='defaults':value.__defaults__=(None,)
        else:value.__kwdefaults__={'other':None}
    elif fault=='closure':
        def outer():
            marker=None
            def clear(self):return marker
            return clear
        value=outer()
    elif fault in ('public_module','implementation_module'):
        name='collections.abc' if fault=='public_module' else '_collections_abc'
        fake=types.ModuleType(name);vars(fake).update(vars(sys.modules[name]))
        monkeypatch.setitem(sys.modules,name,fake)
    elif fault=='class':monkeypatch.setattr(collections.abc,'MutableMapping',type('Foreign',(),{}))
    elif fault=='member':
        monkeypatch.setattr(collections.abc.MutableMapping,'clear',lambda self:None)
    else:monkeypatch.setattr(importlib.machinery.FrozenImporter,'get_code',classmethod(lambda *args:None))
    if fault in ('body','globals','defaults','kwdefaults','closure'):
        value.__module__='collections.abc';value.__qualname__='MutableMapping.clear'
        monkeypatch.setattr(collections.abc.MutableMapping,'clear',value)
    assert code._unsupported_frozen_collections_alias(path,value) is False


def test_attribution_does_not_execute_clear(monkeypatch):
    called=[]
    def clear(self):called.append(self)
    monkeypatch.setattr(collections.abc.MutableMapping,'clear',clear)
    assert code._unsupported_frozen_collections_alias(PATH,clear) is False
    assert called==[]


def test_untrusted_metadata_does_not_invoke_equality(monkeypatch):
    called=[]
    class Foreign:
        def __eq__(self,other):called.append(other);raise AssertionError('metadata hook')
    value=collections.abc.MutableMapping.clear
    monkeypatch.setattr(value,'__module__',Foreign())
    assert code._unsupported_frozen_collections_alias(PATH,value) is False
    assert called==[]


def test_actual_copied_globals_reset_retains_generic_refusal(monkeypatch):
    from anygeometry import _planning_qualification as qualification
    original=collections.abc.MutableMapping.clear
    value=types.FunctionType(original.__code__,dict(original.__globals__),'clear')
    value.__module__='collections.abc';value.__qualname__='MutableMapping.clear'
    for key,member in qualification._CLEAR.items():
        if member is original:monkeypatch.setitem(qualification._CLEAR,key,value)
    monkeypatch.setattr(collections.abc.MutableMapping,'clear',value)
    with pytest.raises(QualificationDecline,match='^infrastructure function global owner$'):
        code.InfrastructurePin('host')
