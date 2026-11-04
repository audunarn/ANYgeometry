"""Test-only actual initial native-helper boundary; never postcapture coverage."""
from functools import wraps
import inspect
import os
import pytest
from anygeometry import _planning_native as _adapter
from anygeometry._planning_qualification import QualificationDecline
from _planning_contract_helpers import require_qualified_planning_lane

REASON = 'planning resource helper requires Windows CPython with GIL'


def native_resource_test(function):
    signature = inspect.signature(function)
    has_request = 'request' in signature.parameters
    parameters = list(signature.parameters.values())
    if not has_request:
        parameters.append(inspect.Parameter('request', inspect.Parameter.KEYWORD_ONLY))

    @wraps(function)
    def invoke(*args, **kwargs):
        request = kwargs['request'] if has_request else kwargs.pop('request')
        if os.name != 'nt':
            require_qualified_planning_lane(REASON)
            with pytest.raises(QualificationDecline) as caught:
                _adapter.load()
            assert type(caught.value) is QualificationDecline
            assert str(caught.value) == REASON
            request.node.user_properties.extend((
                ('execution_scope', 'initial-native-helper-refusal'),
                ('runtime_refusal', REASON), ('original_test_body_executed', False),
                ('postcapture_qualification', False), ('native_export_qualification', False)))
            return
        request.node.user_properties.append(('execution_scope', 'original-native-export-assertions'))
        result = function(*args, **kwargs)
        if not any(name == 'original_test_body_executed' for name, _value in request.node.user_properties):
            request.node.user_properties.append(('original_test_body_executed', True))
        return result

    invoke.__signature__ = signature.replace(parameters=parameters)
    return invoke
