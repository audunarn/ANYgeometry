"""True curved trim field, complete interval budgets and finite branch stars."""
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface,NurbsCurve,CurveOnSurface
from anygeometry._local_curve_stars import certify_curve_stars
from anygeometry._local_curvature import LocalApproximationNotCertified
from anygeometry.local_approximation_field import query_local_approximation_field,bind_local_approximation_field


def curved_loop_model():
    model=owner.GeometryModel()
    support=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))
    points=((.75,.5),(.5,.75),(.25,.5),(.5,.25))
    corners=((.75,.75),(.25,.75),(.25,.25),(.75,.25))
    curves=tuple(CurveOnSurface(support,NurbsCurve(2,(0.,0.,0.,1.,1.,1.),
        (points[i],corners[i],points[(i+1)%4]))) for i in range(4))
    vertices=tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges=tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve) for i,curve in enumerate(curves))
    face=model.add_face(edges,surface=support);model.add_sheet((face,))
    return model,face,edges,curves


def test_finite_curved_stars_cover_closed_loop_and_keep_same_curve_complement(monkeypatch):
    _,_,_,curves=curved_loop_model()
    import anygeometry._local_clearance_bvh as bvh
    original=bvh.build_clearance_bvh;ledgers=[]
    def observed(records,**kwargs):
        records=tuple(records);ledgers.append(tuple(record.key for record in records))
        return original(records,**kwargs)
    monkeypatch.setattr(bvh,'build_clearance_bvh',observed)
    stars=certify_curve_stars((curves,))
    assert stars and all(star.clearance_lower>0 and star.curvature_upper>0 for star in stars)
    for index in range(4):
        intervals=sorted(star.inner for star in stars if star.curve_index==index)
        assert intervals[0][0]==0 and intervals[-1][1]==1
        assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
        assert any(any(key[0]==index for key in ledger) for ledger in ledgers)
    assert all(sum(hi-lo for _,lo,hi in star.outer_domains)<1 for star in stars)


def test_curved_field_sampling_and_chords_record_interval_budget_and_mutation():
    source,face,edges,_=curved_loop_model();before=owner.to_dict(source)
    field=query_local_approximation_field(source,(face,),relative_accuracy=.01,scale_error=.02,policy_floor=1e-8)
    assert field.boundary_tiles and all(row[2]<field.scale_error for row in field.boundary_tiles)
    prepared=owner.clone_prepared_geometry(source,new_preparation_epoch=True)
    regions=owner.query_material_surface_regions(prepared,(face,))
    allocation=bind_local_approximation_field(source,prepared,field,regions)
    sampling=owner.query_prepared_material_boundary_sampling(prepared,regions,
        tuple((prepared.handle('edge',edge),(Q(0),Q(1))) for edge in edges),
        chord_error=.02,local_approximation=allocation)
    chords=sampling.chords
    assert chords.chord_error==.02 and chords.local_field_fingerprint==allocation.fingerprint
    assert len(chords.interval_errors)==len(chords.intervals)==len(chords.local_interval_footprints)
    assert len({limit for _,_,_,limit in chords.interval_errors})>1
    for interval,record in zip(chords.intervals,chords.interval_errors):
        assert record[:3]==(interval.edge.id,*interval.parameter_range)
        assert Q(interval.world_error)<=record[3]<=Q(.02)
    assert owner.to_dict(source)==before
    object.__setattr__(chords,'interval_errors',())
    with pytest.raises(owner.GeometryError):
        owner.validate_prepared_material_boundary_sampling_binding(prepared,regions,sampling)


def test_curved_star_budget_cancellation_and_exact_anchor_refusal():
    from anygeometry.predicates import IntersectionQualificationPolicy
    _,_,_,curves=curved_loop_model()
    with pytest.raises(owner.GeometryError):
        certify_curve_stars((curves,),policy=IntersectionQualificationPolicy(max_boxes_per_pair=2))
    sentinel=RuntimeError('caller cancellation')
    def cancel():raise sentinel
    with pytest.raises(RuntimeError) as caught:certify_curve_stars((curves,),check=cancel)
    assert caught.value is sentinel
    with pytest.raises(LocalApproximationNotCertified,match='anchor'):
        certify_curve_stars(((curves[0],curves[2],curves[1],curves[3]),))


def test_hole_branches_remain_nonlocal_competitors():
    from dataclasses import replace
    _,_,_,curves=curved_loop_model()
    inner=tuple(replace(curve,pcurve=replace(curve.pcurve,
        controls=tuple(tuple(.25+.5*x for x in point) for point in curve.pcurve.controls)))
        for curve in curves)
    stars=certify_curve_stars((curves,inner))
    assert {star.curve_index for star in stars}==set(range(8))
    assert all(0<star.clearance_lower<Q(1,4) for star in stars)
    assert all(all(index<4 for index,lo,hi in star.outer_domains)
        for star in stars if star.curve_index<4)


def test_source_publication_preserves_local_interval_records():
    from dataclasses import replace
    from test_local_approximation_field import fixture
    from test_prepared_material_curvilinear import square_cells
    source,prepared,face,_,regions,allocation=fixture()
    _,_,_,template=square_cells()
    cells=tuple(replace(cell,face=prepared.handle('face',face)) for cell in template)
    chords=owner.query_prepared_material_boundary_chords(prepared,regions,
        tuple((prepared.handle('edge',use.edge),0.,1.) for use in prepared.faces[face].loop),
        chord_error=.01,local_approximation=allocation)
    receipt=owner.validate_prepared_material_curvilinear_cells(prepared,regions,chords,cells,
        surface_error=.01,local_approximation=allocation)
    transfer=owner.query_prepared_material_source_transfer(source,prepared,(receipt,))
    assert transfer.local_boundary_errors
    assert all(row[3]<=Q(.001) and row[4]==0 for row in transfer.local_boundary_errors)
    object.__setattr__(transfer,'local_boundary_errors',())
    with pytest.raises(owner.GeometryError):
        owner.validate_prepared_material_source_transfer_binding(source,prepared,transfer,cell_receipts=(receipt,))


def test_local_c0_knot_stations_are_explicit_and_nonbinary_stations_refuse():
    from anygeometry.local_approximation_field import local_boundary_corner_parameters
    corner=NurbsCurve(1,(0.,0.,.25,1.,1.),((0.,0.,0.),(.5,.25,0.),(1.,0.,0.)))
    assert local_boundary_corner_parameters(corner)==(Q(1,4),)
    from dataclasses import replace
    # Interior source knot 1/4 mapped into domain [0,3/4] is exactly 1/3.
    with pytest.raises(owner.GeometryError,match='binary64'):
        local_boundary_corner_parameters(replace(corner,domain=(0.,.75)))
