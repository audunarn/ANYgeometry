"""Source-evidence lifetime does not discard executable or mutation authority."""
import ast
import types
import pytest
from _planning_numpy_profile_contract import numpy_profile_test
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_qualification import QualificationDecline


def source_fixture(tmp_path):
    path=tmp_path/'evidence.py'
    path.write_text('class Outer:\n class Inner:\n  pass\ndef build():\n class Local:\n  pass\n return Local\n')
    owner=types.ModuleType('evidence_fixture');owner.__file__=str(path)
    return owner,path


def test_code_only_source_never_materializes_ast(tmp_path,monkeypatch):
    owner,path=source_fixture(tmp_path);evidence=manifest._ImmutableEvidence()
    with monkeypatch.context() as patch:
        patch.setattr(ast,'parse',lambda *a,**k:pytest.fail('unrequested AST materialized'))
        digest,tree,nodes,codes=evidence.source(owner,str(path),syntax=False)
        assert tree is None and nodes=={} and 'Outer.Inner' in codes
        assert evidence.source(owner,str(path),syntax=False)[3] is codes
        assert evidence.release_parsing()==dict(source_trees=0,ast_node_visits=0)


def test_explicit_syntax_keeps_same_compiled_code_and_bodies(tmp_path):
    owner,path=source_fixture(tmp_path);evidence=manifest._ImmutableEvidence()
    first=evidence.source(owner,str(path),syntax=False)
    syntax=evidence.source(owner,str(path),syntax=True)
    assert syntax[0]==first[0] and syntax[3] is first[3]
    assert type(syntax[1]) is ast.Module
    assert ast.dump(syntax[1])==ast.dump(ast.parse(path.read_bytes()))
    ids=tuple(map(id,first[3].values()));evidence.release_parsing()
    assert evidence.source(owner,str(path),syntax=True)[3] is first[3]
    assert tuple(map(id,first[3].values()))==ids


def test_class_declaration_index_preserves_nested_and_local_owners(tmp_path):
    owner,path=source_fixture(tmp_path);evidence=manifest._ImmutableEvidence()
    names=evidence.class_declarations(owner,str(path))
    assert names==frozenset(('Outer','Outer.Inner','build.<locals>.Local'))
    assert all(tree is None and nodes=={} for _,tree,nodes,_ in evidence.compiled.values())
    assert evidence.class_declarations(owner,str(path)) is names


@pytest.mark.parametrize('consumer',['code','syntax','classes'])
def test_source_drift_refuses_before_any_reparse(tmp_path,monkeypatch,consumer):
    owner,path=source_fixture(tmp_path);evidence=manifest._ImmutableEvidence()
    evidence.class_declarations(owner,str(path));evidence.begin()
    path.write_text(path.read_text()+'\nclass Changed: pass\n')
    monkeypatch.setattr(ast,'parse',lambda *a,**k:pytest.fail('changed source reparsed'))
    with pytest.raises(QualificationDecline,match='immutable source evidence changed'):
        if consumer=='classes':evidence.class_declarations(owner,str(path))
        else:evidence.source(owner,str(path),syntax=consumer=='syntax')


def test_slot_class_index_cannot_bless_unregistered_schema(monkeypatch):
    from anygeometry import nurbs
    evidence=manifest._ImmutableEvidence();cls=nurbs._PreparedCurveState
    names=evidence.class_declarations(nurbs,nurbs.__file__)
    assert cls.__qualname__ in names
    original=vars(cls)['__slots__']
    monkeypatch.setattr(cls,'__slots__',(*original,'missing'))
    with pytest.raises(QualificationDecline,match='slot metadata native descriptor'):
        manifest._prepare_slot_metadata(cls,evidence,initial=True)


def test_syntax_demand_is_exact_boolean(tmp_path):
    owner,path=source_fixture(tmp_path)
    with pytest.raises(TypeError,match='syntax'):
        manifest._ImmutableEvidence().source(owner,str(path),syntax=1)


@numpy_profile_test
def test_demanded_and_eager_syntax_preserve_complete_role_graph(monkeypatch):
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model
    model,_=_fixture();freeze_model(model)
    demanded=manifest.LoadedManifest(model=model,include_host=False)
    original=manifest._ImmutableEvidence.source
    def eager(self,owner,filename,*,syntax=True):
        return original(self,owner,filename,syntax=True)
    def repeated_binding(self,path,namespace,name):
        self.add(path,lambda:(name in namespace,self.atom(namespace.get(name))),
                 [namespace,namespace.get(name)],early=True)
    with monkeypatch.context() as patch:
        patch.setattr(manifest._ImmutableEvidence,'source',eager)
        patch.setattr(manifest._RetainedChecks,'binding',repeated_binding)
        previous=manifest.LoadedManifest(model=model,include_host=False)
    assert demanded.nodes==previous.nodes and demanded.wire==previous.wire
    for current,prior in ((demanded.callables,previous.callables),
                          (demanded.roles['constructor'],previous.roles['constructor'])):
        assert current._reference_tokens==prior._reference_tokens
        assert current._identities==prior._identities
    assert demanded.parsing_release['source_trees']<previous.parsing_release['source_trees']
    assert len(demanded.callables._retained_checks.early)<len(previous.callables._retained_checks.early)
    assert len(demanded.callables._retained_checks.references)==len(previous.callables._retained_checks.references)
    demanded.verify();demanded.verify(full=True);previous.verify()


def test_repeated_same_namespace_binding_retains_one_original_check():
    checks=manifest._RetainedChecks();value=object();namespace={'value':value}
    for index in range(100):checks.binding('caller'+str(index)+'.value',namespace,'value')
    assert len(checks.early)==1
    assert any(item is namespace for item in checks.references)
    assert any(item is value for item in checks.references)
    checks.verify();namespace['value']=object()
    with pytest.raises(QualificationDecline):checks.verify()


def test_repeated_binding_never_replaces_original_admission_value():
    checks=manifest._RetainedChecks();namespace={'value':object()}
    checks.binding('first.value',namespace,'value');namespace['value']=object()
    with pytest.raises(QualificationDecline):checks.binding('second.value',namespace,'value')
    with pytest.raises(QualificationDecline):checks.verify()


def test_distinct_namespace_and_key_alias_bindings_remain_separate():
    checks=manifest._RetainedChecks();value=object()
    first={'left':value,'right':value};second={'left':value}
    for namespace,key in ((first,'left'),(first,'right'),(second,'left')):
        checks.binding('root.'+key,namespace,key)
    assert len(checks.early)==3
    del second['left']
    with pytest.raises(QualificationDecline):checks.verify()


def test_retained_binding_rejects_hooked_key_and_namespace_passively():
    calls=[]
    class Key(str):
        def __hash__(self):calls.append('hash');raise AssertionError('key hook')
        def __eq__(self,other):calls.append('eq');raise AssertionError('key hook')
    class Namespace(dict):
        def get(self,*args):calls.append('get');raise AssertionError('namespace hook')
    checks=manifest._RetainedChecks()
    for namespace,key in (({},Key('x')),(Namespace(),'x'),(types.MappingProxyType(Namespace()),'x')):
        with pytest.raises(QualificationDecline):checks.binding('root',namespace,key)
    assert calls==[]


@pytest.mark.parametrize('proxy',[False,True])
def test_registered_module_type_key_retains_original_binding_without_hooks(proxy):
    calls=[]
    class Collision:
        def __hash__(self):return hash(types.ModuleType)
        def __eq__(self,other):calls.append('eq');raise AssertionError('foreign key equality')
    namespace={Collision():object()};value=object()
    # Install the legitimate key before the foreign collision to avoid invoking
    # foreign behavior in test setup. The missing-key case uses collision only.
    checks=manifest._RetainedChecks()
    checks.binding('missing module key',types.MappingProxyType(namespace) if proxy else namespace,types.ModuleType)
    checks.verify()
    namespace={types.ModuleType:value};checks=manifest._RetainedChecks()
    view=types.MappingProxyType(namespace) if proxy else namespace
    checks.binding('first module key',view,types.ModuleType)
    checks.binding('second module key',view,types.ModuleType)
    assert len(checks.early)==1 and any(item is value for item in checks.references)
    checks.verify();namespace[types.ModuleType]=object()
    with pytest.raises(QualificationDecline):checks.binding('third module key',view,types.ModuleType)
    with pytest.raises(QualificationDecline):checks.verify()
    assert calls==[]


def test_unknown_native_like_namespace_key_refuses_without_hash():
    calls=[]
    class Meta(type):
        def __hash__(cls):calls.append('hash');raise AssertionError('class key hook')
    class Foreign(metaclass=Meta):pass
    with pytest.raises(QualificationDecline):manifest._RetainedChecks().binding('foreign',{},Foreign)
    assert calls==[]
