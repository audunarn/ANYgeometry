"""Raw sweep charts provide UV footprints, never replacement world boundaries."""
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsCurve, CurveOnSurface
from anygeometry.nurbs_sweep_chart import source_sweep_chart_model, has_sweep_chart
from anygeometry.local_approximation_field import (
    query_local_approximation_field, bind_local_approximation_field,
    prepare_local_boundary_limit_query, prepare_local_cell_limit_query,
)


def raw_allocation():
    source=owner.GeometryModel()
    curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.,0.),(0.,1.,1.),(0.,2.,0.)))
    ends=tuple(source.add_point(*point) for point in curve.evaluate((0.,1.)))
    edge=source.add_curve(*ends,curve)
    face=source.extrude((edge,),(3.,0.,0.))[0]
    field=query_local_approximation_field(source,(face,),relative_accuracy=.01,
        scale_error=.02,policy_floor=1e-8)
    prepared=owner.clone_prepared_geometry(source,new_preparation_epoch=True)
    regions=owner.query_material_surface_regions(prepared,(face,))
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    return source,prepared,face,regions,allocation


def test_raw_sweep_real_allocation_uses_owner_uv_without_source_change():
    source,prepared,face,regions,allocation=raw_allocation()
    before=owner.to_dict(source),owner.to_dict(prepared)
    assert not has_sweep_chart(prepared.faces[face])
    view=source_sweep_chart_model(prepared,face)
    assert view is not None
    boundary=prepare_local_boundary_limit_query(prepared,allocation)
    cell=prepare_local_cell_limit_query(prepared,allocation,_boundary=True)
    from anygeometry.prepared_material_curvilinear import _uv_interval
    for use in prepared.faces[face].loop:
        curve=view.edges[use.edge].curve
        assert type(curve) is CurveOnSurface
        box=_uv_interval(view.support,curve,Q(1,4),Q(3,4))
        value,records=boundary(use.edge,Q(1,4),Q(3,4))
        assert value==cell(face,(box,))[0] and Q(0)<value<=Q(.02)
        assert records[0][0]==face
    assert (owner.to_dict(source),owner.to_dict(prepared))==before


def test_raw_sweep_unsupported_view_keeps_real_refusal(monkeypatch):
    _,prepared,face,_,allocation=raw_allocation()
    import anygeometry.nurbs_sweep_chart as charts
    boundary=prepare_local_boundary_limit_query(prepared,allocation)
    # Isolate unsupported lookup after genuine allocation entry validation.
    monkeypatch.setattr(charts,'source_sweep_chart_model',lambda *args:None)
    with pytest.raises(owner.GeometryError,match='no declared chart enclosure'):
        boundary(prepared.faces[face].loop[0].edge,Q(0),Q(1))


def test_raw_sweep_allocation_entry_rejects_late_source_mutation():
    source,prepared,_,_,allocation=raw_allocation()
    source.add_point(9.,9.,9.)
    with pytest.raises(owner.GeometryError):
        prepare_local_boundary_limit_query(prepared,allocation)


def test_literal_surface_boundary_path_is_unchanged():
    from test_local_approximation_field import fixture
    from anygeometry.arrangement_geometry import freeze_edge
    from anygeometry.prepared_material_curvilinear import _uv_interval
    _,prepared,face,_,regions,allocation=fixture()
    assert source_sweep_chart_model(prepared,face) is None
    support=regions.regions[0].sources[0].support
    boundary=prepare_local_boundary_limit_query(prepared,allocation)
    cell=prepare_local_cell_limit_query(prepared,allocation,_boundary=True)
    for use in prepared.faces[face].loop:
        box=_uv_interval(support,freeze_edge(prepared,use.edge),Q(0),Q(1))
        assert boundary(use.edge,Q(0),Q(1))[0]==cell(face,(box,))[0]
