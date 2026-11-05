"""Retain admitted owners for publication only; persistent ancestry stays weak."""
import gc
from weakref import ref, ReferenceType

import pytest
from anygeometry import GeometryError, clone_prepared_geometry, to_dict
from anygeometry import prepared_material_source_transfer as transfer
from test_prepared_material_curvilinear import square_cells


def _staged_clone():
    source,_,_,_=square_cells()
    intermediate=clone_prepared_geometry(source,new_preparation_epoch=True)
    prepared=clone_prepared_geometry(intermediate,new_preparation_epoch=True)
    candidate=prepared.clone(preserve_identity=True)
    stage=transfer._prepare_source_lineage(prepared,candidate,())
    assert any(owner() is intermediate for owner,*_ in stage[1])
    return source,intermediate,prepared,candidate,stage


def test_public_repeated_cuts_survive_gc_at_publication(monkeypatch):
    from test_prepared_material_source_transfer import (
        test_repeated_public_axis_cuts_compose_original_edge_intervals_through_genuine_clone as original_gate)
    original=transfer._publish_source_lineage
    published=[]
    def collect_then_publish(model,stage):
        routes=() if stage is None else stage[1]
        gc.collect()
        assert routes and all(owner() is not None for owner,*_ in routes)
        result=original(model,stage)
        published.append(len(routes))
        return result
    monkeypatch.setattr(transfer,'_publish_source_lineage',collect_then_publish)
    original_gate()  # every original public-cut/edge-interval assertion executes
    assert len(published)==2 and published[-1]>=2


def test_staged_owner_lifetime_is_operation_scoped():
    source,intermediate,prepared,candidate,stage=_staged_clone()
    weak=ref(intermediate);before=to_dict(prepared)
    del intermediate
    gc.collect()
    transfer._publish_source_lineage(prepared,stage)
    assert weak() is not None and to_dict(prepared)==before
    assert all(type(row[0]) is ReferenceType for row in transfer._LINKS[prepared])
    assert any(row[0] is weak for row in transfer._LINKS[prepared])
    del stage
    gc.collect()
    assert weak() is None  # persistent model routes did not become strong owners
    assert source is not None and candidate is not None


def test_ancestor_dead_before_staging_is_not_resurrected():
    source,intermediate,prepared,candidate,stage=_staged_clone()
    weak=ref(intermediate)
    del intermediate,stage
    gc.collect()
    assert weak() is None
    later=transfer._prepare_source_lineage(prepared,candidate,())
    assert all(owner is not weak for owner,*_ in later[1])
    transfer._publish_source_lineage(prepared,later)
    assert source is not None


@pytest.mark.parametrize('fault',('source','published','link-identity','dead-route'))
def test_admitted_route_changes_still_refuse_without_lineage_publication(fault):
    source,intermediate,prepared,candidate,stage=_staged_clone()
    old=transfer._LINKS[prepared]
    if fault=='source':intermediate.add_point(12.,13.,14.)
    elif fault=='published':prepared.add_point(12.,13.,14.)
    elif fault=='link-identity':transfer._LINKS[prepared]=tuple(list(old))
    else:
        victim=source.clone();weak=ref(victim)
        row=stage[1][0];forged=(weak,*row[1:])
        stage=(stage[0],(forged,*stage[1][1:]),*stage[2:])
        del victim;gc.collect();assert weak() is None
    before=to_dict(prepared);actual=transfer._LINKS[prepared]
    with pytest.raises(GeometryError,match='source lineage .*changed'):
        transfer._publish_source_lineage(prepared,stage)
    assert to_dict(prepared)==before and transfer._LINKS[prepared] is actual
    transfer._restore_source_lineage(prepared,stage)
    assert transfer._LINKS[prepared] is old
    assert source is not None and intermediate is not None and candidate is not None
