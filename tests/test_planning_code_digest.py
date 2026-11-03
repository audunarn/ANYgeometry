"""Code fingerprints describe fields, independently of marshal ref bookkeeping."""
import types
import pytest
from anygeometry._planning_qualification import QualificationDecline


def digest(code):
    from anygeometry._planning_code import code_digest
    return code_digest(code)


def test_code_fingerprint_ignores_reference_count_changes():
    def kernel(value):return value.selected()+3.25
    before=digest(kernel.__code__)
    references=[kernel.__code__.co_consts,kernel.__code__.co_names,*kernel.__code__.co_consts]
    assert digest(kernel.__code__)==before
    assert references


@pytest.mark.parametrize('field,value',[('co_consts',(None,5)),('co_names',('different',)),
    ('co_filename','different.py'),('co_firstlineno',123),('co_exceptiontable',b'\x00'),('co_flags',0)])
def test_code_fingerprint_retains_changed_fields(field,value):
    def kernel(value):return value.selected()
    assert digest(kernel.__code__.replace(**{field:value}))!=digest(kernel.__code__)


def test_unknown_code_constant_refuses_without_user_hooks():
    class Foreign:
        def __repr__(self):raise AssertionError('repr must not run')
    code=(lambda:None).__code__.replace(co_consts=(None,Foreign()))
    with pytest.raises(QualificationDecline):digest(code)


def test_nested_code_and_frozen_constants_are_canonical():
    def kernel():
        return lambda item:item in {'a','b'}
    assert digest(kernel.__code__)==digest(kernel.__code__.replace())


def test_nested_code_constant_change_is_retained():
    def kernel():return lambda:17.25
    index,nested=next((i,value) for i,value in enumerate(kernel.__code__.co_consts) if type(value) is types.CodeType)
    child=nested.replace(co_consts=tuple(18.25 if type(value) is float and value==17.25 else value for value in nested.co_consts))
    changed=kernel.__code__.replace(co_consts=tuple(child if i==index else value for i,value in enumerate(kernel.__code__.co_consts)))
    assert digest(changed)!=digest(kernel.__code__)


def test_compiler_slice_constants_are_exact_and_closed():
    def kernel(value):return value[1:]
    assert any(type(value) is slice for value in kernel.__code__.co_consts)
    before=digest(kernel.__code__)
    changed=kernel.__code__.replace(co_consts=tuple(slice(2,None,None) if type(value) is slice else value
        for value in kernel.__code__.co_consts))
    assert digest(changed)!=before
    unknown=kernel.__code__.replace(co_consts=(slice(object(),None,None),))
    with pytest.raises(QualificationDecline):digest(unknown)


@pytest.mark.parametrize('change',['nested_constant','flags'])
def test_code_field_mutation_is_terminal(change):
    from anygeometry._planning_manifest import capture_callable_manifest
    from anygeometry._planning_executor import PlannerSourceChanged
    def kernel():return lambda:17.25
    manifest=capture_callable_manifest((kernel,))
    if change=='flags':kernel.__code__=kernel.__code__.replace(co_flags=kernel.__code__.co_flags ^ 0x10)
    else:
        index,nested=next((i,value) for i,value in enumerate(kernel.__code__.co_consts) if type(value) is types.CodeType)
        child=nested.replace(co_consts=tuple(18.25 if type(value) is float and value==17.25 else value for value in nested.co_consts))
        kernel.__code__=kernel.__code__.replace(co_consts=tuple(child if i==index else value for i,value in enumerate(kernel.__code__.co_consts)))
    with pytest.raises(PlannerSourceChanged):manifest.verify()
