"""Explicit internal forcing for tests whose contract is the general 4D route.

This is test-only monkeypatching, not a public qualification/admission option.
New rational structural positives are exercised without this fixture.
"""
from importlib import import_module
import pytest


@pytest.fixture(autouse=True)
def force_generic_for_generic_regressions(request, monkeypatch):
    filename = request.node.path.name
    force = (filename in {"test_generic.py", "test_exact_preparation_reuse.py", "test_restriction_and_batch.py"}
             or filename == "test_fast_path.py" and request.node.name.startswith("test_generic_")
             or filename == "test_foundation.py" and (request.node.name.startswith("test_intersecting_pair_")
                or request.node.name.startswith("test_issued_records_")))
    if force:
        module = import_module("anygeometry.nurbs_surface_intersection.query")
        monkeypatch.setattr(module,"derive_structural",lambda *a:(None,"explicit internal generic-route regression"))
