"""Classify initial admission to the immutable NumPy context image profile.

The positive lane is Windows CPython 3.14.2 / NumPy 2.4.6 / SciPy 1.17.1.
The production source/native SHA pins remain the admission authority. Other
images exercise only the exact initial typed refusal and unchanged anchored
NumPy settings; they provide no postcapture, worker, or native qualification.
This classifies the selected test's declared positive profile. It does not prove
that a legacy runtime or its distinct admission path is itself unsupported.
"""
from functools import wraps
import inspect
import sys

import numpy as np
import pytest
from _planning_contract_helpers import require_qualified_planning_lane


PROFILE_CONTRACT = 'numpy-context-immutable-images-v1'


def _initial_profile_refusal():
    from anygeometry import _planning_numpy_context as context
    from anygeometry._planning_qualification import is_package_module, pin_module
    modules = tuple(sys.modules.get(name) for name in (
        'numpy._core._ufunc_config', 'numpy._core._multiarray_umath', 'numpy._core.umath'))
    if any(not is_package_module(module) for module in modules):
        return 'NumPy context owner modules unavailable'
    source, native = (pin_module(module, package_only=True) for module in modules[:2])
    if (source.get('kind') != 'python' or source.get('sha256') != context._SOURCE_SHA256 or
            native.get('kind') != 'native' or native.get('sha256') != context._NATIVE_SHA256):
        return 'NumPy context source/native image outside profile'
    return None


def _anchored_context():
    # Verify existing owner-load bindings and function/code edges before calling
    # getters. Unsupported images never authorize invoking unknown replacements.
    from anygeometry._planning_ledger import runtime_infrastructure
    handle, _ = runtime_infrastructure()
    getters = {row[1]: row[3] for row in vars(handle[0])['ANCHOR']
               if row[0] == 'numpy' and row[1] in ('geterr', 'geterrcall', 'getbufsize')}
    assert set(getters) == {'geterr', 'geterrcall', 'getbufsize'}
    assert all(vars(np).get(name) is getter for name, getter in getters.items())
    return getters['geterr'](), getters['geterrcall'](), getters['getbufsize']()


def numpy_profile_test(function):
    """Run the original positive body or a labelled exact initial refusal."""
    signature = inspect.signature(function)
    has_request = 'request' in signature.parameters
    parameters = list(signature.parameters.values())
    if not has_request:
        parameters.append(inspect.Parameter('request', inspect.Parameter.KEYWORD_ONLY))

    @wraps(function)
    def invoke(*args, **kwargs):
        request = kwargs['request'] if has_request else kwargs.pop('request')
        request.node.user_properties.append(('numpy_profile_contract', PROFILE_CONTRACT))
        reason = _initial_profile_refusal()
        require_qualified_planning_lane(reason)
        if reason is None:
            request.node.user_properties.append(('execution_scope', 'original-profile-assertions'))
            request.node.user_properties.append(('original_test_body_executed', True))
            return function(*args, **kwargs)

        request.node.user_properties.extend((
            ('execution_scope', 'initial-profile-refusal'),
            ('numpy_profile_refusal', reason),
            ('postcapture_qualification', False),
            ('original_test_body_executed', False),
            ('numerical_worker_execution', False),
            ('legacy_runtime_admission', 'not-evaluated')))
        before = _anchored_context()
        from anygeometry._planning_numpy_context import capture_numpy_context
        from anygeometry._planning_qualification import QualificationDecline
        with pytest.raises(QualificationDecline) as caught:
            capture_numpy_context(package_only=True)
        assert type(caught.value) is QualificationDecline
        assert str(caught.value) == reason
        after = _anchored_context()
        assert after[0] == before[0] and after[1] is before[1] and after[2] == before[2]

    invoke.__signature__ = signature.replace(parameters=parameters)
    return invoke
