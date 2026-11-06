"""Explicit initial-refusal classification for runtime-bound manifest tests.

A refusal result never represents successful capture or postcapture coverage.
"""
import sys
import os
import ast
import hashlib
from pathlib import Path
import types
import pytest
from anygeometry import _planning_manifest as manifest
from anygeometry import _planning_qualification as qualification
from anygeometry._planning_qualification import QualificationDecline


_RUNTIME_IMAGE_REASON='interpreter runtime adapter is Windows-only'


def _runtime_only_refusals(error):
    """Inspect every real refusal, including rows omitted from the exception text."""
    if (type(error) is not QualificationDecline or error.__cause__ is not None
            or error.__context__ is not None or type(error.args) is not tuple
            or len(error.args)!=1 or type(error.args[0]) is not str):
        return None
    if 'refusals' not in vars(error):
        return (_RUNTIME_IMAGE_REASON,) if error.args[0]==_RUNTIME_IMAGE_REASON else None
    rows=vars(error)['refusals']
    if type(rows) is not list or not rows or any(type(row) is not str for row in rows):
        return None
    if any(not row.partition(': ')[0] or row.partition(': ')[2]!=_RUNTIME_IMAGE_REASON for row in rows):
        return None
    if error.args[0]!='callable closure refused: '+str(rows[:5]):
        return None
    return tuple(rows)


def _runtime_guard_evidence():
    """Attest the actual source-defined OS guard, never a replacement provider."""
    if sys.platform=='win32':return None
    assert qualification.is_package_module(qualification)
    assert sys.modules.get(qualification.__name__) is qualification
    function=vars(qualification).get('runtime_files')
    assert type(function) is types.FunctionType and function.__globals__ is vars(qualification)
    assert function.__module__==qualification.__name__ and function.__qualname__=='runtime_files'
    assert function.__defaults__ is None and function.__kwdefaults__ is None and function.__closure__ is None
    assert vars(qualification).get('QualificationDecline') is QualificationDecline
    path=Path(qualification.__file__).resolve()
    assert path==Path(manifest.__file__).resolve().parent/'_planning_qualification.py'
    assert Path(function.__code__.co_filename).resolve()==path
    raw=path.read_bytes();tree=ast.parse(raw)
    declaration=next(node for node in tree.body if type(node) is ast.FunctionDef and node.name=='runtime_files')
    expected=ast.parse("import sys as runtime_sys\nimport ctypes\nif runtime_sys.platform != 'win32':\n    raise QualificationDecline('interpreter runtime adapter is Windows-only')\n")
    assert [ast.dump(node,include_attributes=False) for node in declaration.body[1:4]]==[
        ast.dump(node,include_attributes=False) for node in expected.body]
    compiled=compile(raw,str(path),'exec',dont_inherit=True)
    code=next(part for part in compiled.co_consts if type(part) is types.CodeType and part.co_qualname=='runtime_files')
    digest=manifest.code_digest(code)
    assert manifest.code_digest(function.__code__)==digest,'runtime image guard differs from source'
    with pytest.raises(QualificationDecline) as caught:function()
    assert type(caught.value) is QualificationDecline and caught.value.args==(_RUNTIME_IMAGE_REASON,)
    return dict(provider=str(path),provider_sha256=hashlib.sha256(raw).hexdigest(),code_sha256=digest)


def classify_initial_runtime_refusal(error,*,request=None):
    """Only this exact initial image boundary; never postcapture/mutation proof."""
    rows=_runtime_only_refusals(error)
    if rows is None:return False
    require_qualified_planning_lane(_RUNTIME_IMAGE_REASON)
    evidence=_runtime_guard_evidence()
    if evidence is None:return False
    if request is not None:
        request.node.user_properties.extend((('execution_scope','initial-runtime-image-refusal'),
            ('runtime_refusal',_RUNTIME_IMAGE_REASON),('runtime_refusal_count',len(rows)),
            ('runtime_guard_source_sha256',evidence['provider_sha256']),
            ('original_test_body_executed',False),('postcapture_qualification',False)))
    return True


def _capture_with_runtime_image_boundary(roots,request,label,kwargs):
    try:
        captured=manifest.capture_callable_manifest(roots,**kwargs)
    except QualificationDecline as error:
        if classify_initial_runtime_refusal(error,request=request):return None
        raise
    if request is not None:
        request.node.user_properties.append(('execution_scope',label))
    return captured


def require_qualified_planning_lane(reason):
    """A mandatory positive lane must never count an initial refusal as success."""
    if reason is not None and os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS') == '1':
        pytest.fail('qualified planning test lane requires original assertions: ' + reason)


def dispatcher_boundary_refusal():
    try:
        from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
    except ModuleNotFoundError as error:
        if error.name not in {'numpy._core', 'numpy._core._multiarray_umath'}:
            raise
        return 'NumPy array dispatcher interface unavailable'
    return None


def runtime_boundary_refusal():
    reason = dispatcher_boundary_refusal()
    if reason is not None:
        return reason
    if sys.platform != 'win32':
        return 'interpreter runtime adapter is Windows-only'
    return None


def capture_with_runtime_boundary(roots, *, request=None, **kwargs):
    # Adapter availability alone does not imply that a pure root requires it.
    reason = dispatcher_boundary_refusal()
    require_qualified_planning_lane(reason)
    if reason is not None:
        if request is not None:
            request.node.user_properties.extend((
                ('execution_scope', 'initial-runtime-refusal'),
                ('runtime_refusal', reason), ('original_test_body_executed',False),
                ('postcapture_qualification', False)))
        with pytest.raises(QualificationDecline) as caught:
            manifest.capture_callable_manifest(roots,**kwargs)
        assert type(caught.value) is QualificationDecline
        assert str(caught.value) == reason
        return None
    return _capture_with_runtime_image_boundary(roots,request,'original-runtime-assertions',kwargs)


def capture_with_dispatcher_boundary(roots, *, request=None, **kwargs):
    """Preserve pure roots; classify only actual dispatcher/runtime refusals."""
    reason = dispatcher_boundary_refusal()
    require_qualified_planning_lane(reason)
    if reason is not None:
        if request is not None:
            request.node.user_properties.extend((
                ('execution_scope', 'initial-dispatcher-refusal'),
                ('runtime_refusal', reason), ('original_test_body_executed',False),
                ('postcapture_qualification', False)))
        with pytest.raises(QualificationDecline) as caught:
            manifest.capture_callable_manifest(roots, **kwargs)
        assert type(caught.value) is QualificationDecline
        assert str(caught.value) == reason
        return None
    return _capture_with_runtime_image_boundary(roots,request,'original-dispatcher-assertions',kwargs)


def classify_initial_dispatcher_refusal(request):
    """Before a specific negative, attest only a missing dispatcher with a minimal root."""
    reason = dispatcher_boundary_refusal()
    require_qualified_planning_lane(reason)
    if reason is None:
        return False
    request.node.user_properties.extend((
        ('execution_scope', 'initial-dispatcher-refusal'),
        ('runtime_refusal', reason), ('original_test_body_executed',False),
        ('postcapture_qualification', False)))
    with pytest.raises(QualificationDecline) as caught:
        manifest.capture_callable_manifest((None,))
    assert type(caught.value) is QualificationDecline
    assert str(caught.value) == reason
    return True
