"""R5 generated edges derive authority from a live registered owner."""
import types
import pytest
from _planning_legacy_contract import capture_with_legacy_dataclass_boundary
from _planning_contract_helpers import (capture_with_runtime_boundary,
    classify_initial_dispatcher_refusal, classify_initial_runtime_refusal)
from anygeometry._planning_manifest import capture_callable_manifest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.tolerance import TolerancePolicy


def test_registered_generated_slots_and_repr_closure_are_attested(request):
    captured=capture_with_legacy_dataclass_boundary((TolerancePolicy,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    captured.verify()


def test_module_spelling_does_not_register_a_class(request):
    if classify_initial_dispatcher_refusal(request):return # Exact initial refusal is not registration coverage.
    class Foreign:
        pass
    Foreign.__module__='anygeometry.tolerance'
    with pytest.raises(QualificationDecline) as caught:
        capture_callable_manifest((Foreign,))
    if classify_initial_runtime_refusal(caught.value,request=request):return
    assert 'unregistered' in str(caught.value)


def test_foreign_generated_closure_class_refuses(monkeypatch,request):
    if classify_initial_dispatcher_refusal(request):return
    original=vars(TolerancePolicy)['__setattr__']
    class Foreign:
        pass
    Foreign.__module__=TolerancePolicy.__module__
    # Dataclass providers may rename generated locals between patch releases.
    # Select the actual owning/pre-slots class, never a spelling-only no-op.
    original_cells=original.__closure__ or ()
    owners=tuple(cell for cell in original_cells
        if isinstance(cell.cell_contents,type)
        and cell.cell_contents.__module__==TolerancePolicy.__module__
        and cell.cell_contents.__qualname__==TolerancePolicy.__qualname__
        and vars(cell.cell_contents).get('__dataclass_fields__') is vars(TolerancePolicy)['__dataclass_fields__'])
    assert len(owners)==1, (original.__code__.co_freevars,
        tuple(type(cell.cell_contents).__name__ for cell in original_cells))
    owner=owners[0]
    cells=tuple(types.CellType(Foreign) if cell is owner else cell for cell in original_cells)
    assert sum(cell.cell_contents is Foreign for cell in cells)==1
    assert owner.cell_contents is not Foreign
    forged=types.FunctionType(original.__code__,original.__globals__,original.__name__,
        original.__defaults__,cells)
    forged.__module__=original.__module__
    forged.__qualname__=original.__qualname__
    monkeypatch.setattr(TolerancePolicy,'__setattr__',forged)
    with pytest.raises(QualificationDecline) as caught:
        capture_callable_manifest((TolerancePolicy,))
    if classify_initial_runtime_refusal(caught.value,request=request):return


def test_forged_provider_origin_refuses(monkeypatch,request):
    if classify_initial_dispatcher_refusal(request):return
    original=vars(TolerancePolicy)['__repr__']
    forged=types.FunctionType(original.__code__.replace(co_filename='<forged-provider>'),
        original.__globals__,original.__name__,original.__defaults__,original.__closure__)
    forged.__module__=original.__module__
    forged.__qualname__=original.__qualname__
    forged.__dict__.update(vars(original))
    monkeypatch.setattr(TolerancePolicy,'__repr__',forged)
    with pytest.raises(QualificationDecline) as caught:
        capture_callable_manifest((TolerancePolicy,))
    if classify_initial_runtime_refusal(caught.value,request=request):return
