"""Explicit initial-refusal classification for runtime-bound manifest tests.

A refusal result never represents successful capture or postcapture coverage.
"""
import sys
import os
import pytest
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_qualification import QualificationDecline


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


def capture_with_runtime_boundary(roots, *, request=None):
    reason = runtime_boundary_refusal()
    require_qualified_planning_lane(reason)
    if reason is not None:
        if request is not None:
            request.node.user_properties.extend((
                ('execution_scope', 'initial-runtime-refusal'),
                ('runtime_refusal', reason), ('postcapture_qualification', False)))
        with pytest.raises(QualificationDecline) as caught:
            manifest.capture_callable_manifest(roots)
        assert type(caught.value) is QualificationDecline
        assert str(caught.value) == reason
        return None
    if request is not None:
        request.node.user_properties.append(('execution_scope', 'original-runtime-assertions'))
    return manifest.capture_callable_manifest(roots)


def capture_with_dispatcher_boundary(roots, *, request=None, **kwargs):
    """Classify only the missing dispatcher; preserve pure-root POSIX capture."""
    reason = dispatcher_boundary_refusal()
    require_qualified_planning_lane(reason)
    if reason is not None:
        if request is not None:
            request.node.user_properties.extend((
                ('execution_scope', 'initial-dispatcher-refusal'),
                ('runtime_refusal', reason), ('postcapture_qualification', False)))
        with pytest.raises(QualificationDecline) as caught:
            manifest.capture_callable_manifest(roots, **kwargs)
        assert type(caught.value) is QualificationDecline
        assert str(caught.value) == reason
        return None
    if request is not None:
        request.node.user_properties.append(('execution_scope', 'original-dispatcher-assertions'))
    return manifest.capture_callable_manifest(roots, **kwargs)


def classify_initial_dispatcher_refusal(request):
    """Before a specific negative, attest only a missing dispatcher with a minimal root."""
    reason = dispatcher_boundary_refusal()
    require_qualified_planning_lane(reason)
    if reason is None:
        return False
    request.node.user_properties.extend((
        ('execution_scope', 'initial-dispatcher-refusal'),
        ('runtime_refusal', reason), ('postcapture_qualification', False)))
    with pytest.raises(QualificationDecline) as caught:
        manifest.capture_callable_manifest((None,))
    assert type(caught.value) is QualificationDecline
    assert str(caught.value) == reason
    return True
