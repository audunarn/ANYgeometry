"""Same-call validated Arc identity, never detached/historical branch admission."""
from dataclasses import replace
import json

import pytest
from anygeometry import GeometryError, support_curves
from anygeometry import nurbs_runtime_binding as runtime
from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
from test_structural_fan_bounds import bound
from test_structural_fan_cache_lifetime import saturated_sections


def test_repeated_validation_preserves_exact_actual_arc_tuple(saturated_sections):
    args,_,arcs=saturated_sections;curve=args[1].curve
    for _ in range(3):
        assert runtime.validate_section_runtime(curve) is arcs
        assert curve._arcs() is arcs


def test_repeated_structural_then_public_caller_keep_exact_result(saturated_sections,tmp_path):
    args,kwargs,_=saturated_sections
    first=bound(args,kwargs)
    assert first is not None
    (tmp_path/'first-bound.json').write_text(json.dumps(dict(
        numerator=first[0].numerator,denominator=first[0].denominator,examined=first[1])))
    assert bound(args,kwargs)==first
    result=certify_fan_subtriangle(*args,**kwargs)
    assert result==first and 0<=result[0]<=kwargs['surface_error']


@pytest.mark.parametrize('fault',('bank','context','key','conflict'))
def test_validation_never_republishes_into_changed_bank_key_or_conflict(saturated_sections,monkeypatch,fault,request):
    args,_,arcs=saturated_sections;curve=args[1].curve
    bank=support_curves._RUNTIME_CACHE_BANK.get();key=(curve.plane_record,curve.surface,curve.path)
    original=runtime._section_runtime;old_path=curve.path
    def observed(value):
        result=original(value)
        if value is curve:
            if fault=='bank':bank['sections']={}
            elif fault=='context':support_curves._RUNTIME_CACHE_BANK.set({})
            elif fault=='key':object.__setattr__(curve,'path',old_path+1)
            else:bank['sections'][key]=tuple(replace(arc) for arc in arcs)
        return result
    monkeypatch.setattr(runtime,'_section_runtime',observed)
    try:
        with pytest.raises(GeometryError,match='runtime cache'):
            runtime.validate_section_runtime(curve)
    finally:object.__setattr__(curve,'path',old_path)
    request.node.user_properties.append(('execution_scope','fault-injected-cache-component'))


def test_failed_literal_validation_does_not_republish_poisoned_arcs(saturated_sections):
    args,_,arcs=saturated_sections;curve=args[1].curve
    cache=support_curves._RUNTIME_CACHE_BANK.get()['sections'];key=(curve.plane_record,curve.surface,curve.path)
    old=arcs[0].s0
    try:
        object.__setattr__(arcs[0],'s0',old+1)
        with pytest.raises(GeometryError,match='runtime cache'):
            runtime.validate_section_runtime(curve)
        assert key not in cache
    finally:object.__setattr__(arcs[0],'s0',old)
