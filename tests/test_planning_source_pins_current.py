"""Every planning source pin is the sha256 of the committed source bytes, for every role.

A stale pin makes the qualified planner decline (serial fallback) without failing any
ordinary result, so it is checked directly here rather than inferred from other tests.
"""
import pytest

from anygeometry import _planning_manifest as manifest


@pytest.mark.parametrize('role', ['common', 'host', 'constructor'])
def test_operation_registry_source_pins_match_the_source_bytes(role):
    manifest.verify_operation_registry(role)
