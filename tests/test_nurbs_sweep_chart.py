"""Exact source correspondence survives cuts without rewriting world edges."""
from dataclasses import replace
import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, Plane, plan_plane_split, apply_plane_split
from anygeometry.entities import OrientedEdge
from anygeometry.nurbs import NurbsCurve, NurbsSurface
from anygeometry.surfaces import CoonsSurface
from anygeometry.serialization import to_dict, from_dict
from anygeometry.nurbs_sweep_chart import chart_model, has_sweep_chart, RECIPE
from anygeometry.nurbs_sweep_chart import install_sweep_chart


def sweep():
    model = GeometryModel()
    profile = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((0., 0., 0.), (0., .5, 1.), (0., 1., 0.)))
    opposite = replace(profile, controls=tuple((x+5., y, z) for x, y, z in profile.controls))
    a, b, c, d = model.add_points(((0., 0., 0.), (0., 1., 0.), (5., 1., 0.), (5., 0., 0.)))
    bottom = model.add_curve(a, b, profile)
    right = model.add_line(b, c)
    top = model.add_curve(d, c, opposite)
    left = model.add_line(d, a)
    face = model.add_face_from_loop(tuple(OrientedEdge(edge, forward) for edge, forward in
        ((bottom, True), (right, True), (top, False), (left, True))), (0, 1, 2, 3), surface=CoonsSurface())
    return model, face, (bottom, right, top, left)


def test_public_translation_cut_keeps_marker_and_reconstructs_exact_chart():
    model, parent, edges = sweep()
    original = to_dict(model)
    assert model.faces[parent].parameterization is None
    assert not has_sweep_chart(model.faces[parent])
    plane = Plane((2.5, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    plan = plan_plane_split(model, (parent,), plane)
    assert to_dict(model) == original
    result = apply_plane_split(model, plan, policy='imprint')
    assert len(model.faces) == 2 and len(result.trace_edges) == 1
    assert len(model.faces_using_edge(result.trace_edges[0])) == 2
    for candidate in (model, from_dict(to_dict(model))):
        for identifier, face in candidate.faces.items():
            assert isinstance(face.surface, CoonsSurface)
            assert isinstance(face.parameterization, NurbsSurface)
            owner = chart_model(candidate, identifier)
            assert np.array_equal(candidate.face_support_point(identifier, .3, .7),
                                  candidate.face_point(identifier, .3, .7))
            for loop in (face.loop, *face.holes):
                for use in loop:
                    uv = candidate.face_boundary_uv(identifier, use.edge, (0., .5, 1.))
                    mapped = face.parameterization.evaluate_many(uv)
                    assert np.allclose(mapped, candidate.sample_edge(use.edge, (0., .5, 1.)), atol=1e-12, rtol=0.)
                    assert owner.edges[use.edge].curve is not None
        assert not candidate.validate_topology()


def test_sweep_recipe_does_not_override_a_changed_tensor():
    model, parent, _ = sweep()
    plan = plan_plane_split(model, (parent,), Plane((2.5, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    apply_plane_split(model, plan, policy='imprint')
    child = next(iter(model.faces))
    face = model.faces[child]
    assert RECIPE in face.metadata
    original = face.parameterization
    changed = replace(original, controls=tuple(tuple((x, y, z+.1) for x, y, z in row) for row in original.controls))
    object.__setattr__(face, 'parameterization', changed)
    with pytest.raises(GeometryError, match='tensor differs'):
        chart_model(model, child)


def cropped_rational_sweep():
    model = GeometryModel()
    profile = NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        ((0., 0., 0.), (0., .5, 1.), (0., 1., 0.)), (1., 2., 1.), (.25, .75))
    opposite = replace(profile, controls=tuple((x+5., y, z) for x, y, z in profile.controls))
    a, b, d, c = model.add_points(tuple(tuple(curve.evaluate(t))
        for curve in (profile, opposite) for t in (0., 1.)))
    bottom = model.add_curve(a, b, profile)
    right = model.add_line(b, c)
    top = model.add_curve(d, c, opposite)
    left = model.add_line(d, a)
    face = model.add_face_from_loop(tuple(OrientedEdge(e, f) for e, f in
        ((bottom, True), (right, True), (top, False), (left, True))), (0, 1, 2, 3), surface=CoonsSurface())
    return model, face, profile, (left, right)


def test_cropped_rational_connector_transport_preserves_source_and_survives_reload():
    model, face, profile, connectors = cropped_rational_sweep()
    before = to_dict(model)
    assert any(float(x) != x for x in profile.exact_point(0.))
    plane = Plane((2.5, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    plan = plan_plane_split(model, [face], plane)
    assert to_dict(model) == before
    result = apply_plane_split(model, plan, policy='imprint')
    assert len(result.trace_edges) == 1 and len(model.faces) == 2
    for candidate in (model, from_dict(to_dict(model))):
        for child in candidate.faces:
            view = chart_model(candidate, child)
            for use in candidate.faces[child].loop:
                chart = view.edges[use.edge].curve
                xyz = candidate.sample_edge(use.edge, (0., .5, 1.))
                assert np.max(np.abs(chart.evaluate(np.array((0., .5, 1.)))-xyz)) < 1e-12
        assert not candidate.validate_topology()


def test_sweep_connector_declared_intervals_refuse_gap_and_changed_source():
    model, face, _profile, _connectors = cropped_rational_sweep()
    with model.transaction():
        install_sweep_chart(model, face)
    current = model.faces[face]
    recipe = dict(current.metadata[RECIPE])
    records = list(recipe['connectors'])
    source, u, base_id, tip_id, base, tip, uses = records[0]
    records[0] = (source, u, base_id, tip_id, base, tip, ((source, 0., .5),))
    recipe['connectors'] = tuple(records)
    model._faces[face] = replace(current, metadata={RECIPE: recipe})
    with pytest.raises(GeometryError, match='omit source material'):
        chart_model(model, face)


def test_weighted_constant_section_has_exact_transverse_rectangle():
    from fractions import Fraction as Q
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry.regular_source_reference import _view
    model, face, _profile, _connectors = cropped_rational_sweep()
    with model.transaction():
        install_sweep_chart(model, face)
    surface = model.faces[face].parameterization
    curve = PlaneNurbsCurve.from_plane(Plane((2.5, 0., 0.), (0., 1., 0.), (0., 0., 1.)), surface)
    arc = curve._arcs()[0]
    view = _view(arc)
    view.curve = curve
    end = max(arc.s0, arc.s1)
    _s, x = view.rectangle(end-Q(1, 2**40), end, parent=view.x_box)
    assert x == (Q(1, 2), Q(1, 2))
