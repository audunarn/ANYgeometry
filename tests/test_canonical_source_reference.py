"""Fresh canonical chart owners retain exact identity and actual-world error."""
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.curves import Straight
from anygeometry.nurbs import NurbsCurve, CurveOnSurface
from anygeometry.nurbs_sweep_chart import has_sweep_chart
from anygeometry.source_cell_complex import _support_owner
import anygeometry.prepared_material_curvilinear as boundary


def raw_sweep(*,rounded=False):
    model=owner.GeometryModel()
    curve=(NurbsCurve(1,(0.,0.,1.,1.),((0.,.1,0.),(0.,.3,0.)),domain=(.25,.75))
        if rounded else NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.,0.),(0.,1.,1.),(0.,2.,0.))))
    ends=tuple(model.add_point(*point) for point in curve.evaluate((0.,1.)))
    edge=model.add_curve(*ends,curve)
    face=model.extrude((edge,),(3.,0.,0.))[0]
    regions=owner.query_material_surface_regions(model,(face,))
    return model,face,regions


def test_raw_reference_has_genuine_canonical_owner_and_vertex_identity():
    model,face,regions=raw_sweep()
    before=owner.to_dict(model)
    assert not has_sweep_chart(model.faces[face])
    references=boundary._sampling_references(model,regions)
    support,reference=references[face]
    assert reference.owner is not model and reference.face.surface==support
    for use in model.faces[face].loop:
        edge=reference.owner.edges[use.edge]
        assert type(edge.curve) is CurveOnSurface
        for t,v in ((Q(0),edge.start),(Q(1),edge.end)):
            node=reference.node(use.edge,t,v)
            assert node.identity==reference.vertices[v].identity
    assert owner.to_dict(model)==before


def test_raw_chord_occurrences_use_owner_uv_and_preserve_world_samples():
    model,face,regions=raw_sweep()
    before=owner.to_dict(model)
    # A full bowed profile is not a .1 chord; exact quarter intervals are.
    intervals=tuple((model.handle('edge',use.edge),i/4,(i+1)/4)
                    for use in model.faces[face].loop for i in range(4))
    chords=owner.query_prepared_material_boundary_chords(model,regions,intervals,chord_error=.1)
    assert len(chords.intervals)==16
    for interval in chords.intervals:
        expected=model.sample_edge(interval.edge.id,interval.parameter_range)
        assert interval.xyz==tuple(tuple(float(x) for x in row) for row in expected)
        assert interval.occurrences
    assert owner.to_dict(model)==before


def test_world_affine_connector_keeps_nonzero_actual_endpoint_residual():
    # Polynomial affine support makes the sufficient derivative test exact.
    # Trimmed stored .1/.3 controls still have nonzero endpoint rounding error.
    model,face,regions=raw_sweep(rounded=True)
    before=owner.to_dict(model)
    references=boundary._sampling_references(model,regions)
    support,reference=references[face]
    edge=next(model.edges[use.edge] for use in model.faces[face].loop
              if type(model.edges[use.edge].curve) is Straight)
    ideal=reference.owner.edges[edge.id].curve
    ends=tuple(tuple(float(x) for x in model.vertex_position(v)) for v in (edge.start,edge.end))
    residuals=tuple(tuple(Q(x)-y for x,y in zip(end,ideal.exact_point(t))) for end,t in zip(ends,(0,1)))
    assert residuals[0]==residuals[1] and any(residuals[0])
    assert boundary._cell_reference_is_world_affine(model,references,edge.id,Q(0),Q(1),lambda:None)
    bound,_=boundary._cell_reference_world_chord(model,references,edge.id,Q(0),Q(1),ends,.001,lambda:None)
    squared=sum(x*x for x in residuals[0])
    assert Q(bound)>0 and Q(bound)**2>=squared
    assert owner.to_dict(model)==before


def test_unsupported_source_view_keeps_original_owner(monkeypatch):
    model,face,_=raw_sweep()
    import anygeometry.nurbs_sweep_chart as charts
    monkeypatch.setattr(charts,'source_sweep_chart_model',lambda *args:None)
    assert _support_owner(model,face) is model


def test_literal_nurbs_owner_is_not_adapted():
    from test_local_approximation_field import fixture
    _,prepared,face,_,_,_=fixture()
    assert _support_owner(prepared,face) is prepared
