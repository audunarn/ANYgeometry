"""Projected-boundary correspondence is a diagnostic, never a domain certificate."""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from anygeometry import (GeometryError, GeometryModel, Cylinder, apply_intersections,
    clone_prepared_geometry, from_dict, plan_intersections, to_dict)
from anygeometry.curves import Arc
from anygeometry.entities import OrientedEdge
from anygeometry.cylinder_boundary_correspondence import (
    _arc_finiteness, _arc_root, _compare_straight, _frame, _quadratic_min, _sqrt_ceiling,
    _straight_root, _straight_root_evidence, _tile_spans,
    query_prepared_cylinder_boundary_correspondence as query,
    validate_prepared_cylinder_boundary_correspondence_binding as validate)
from anygeometry.edge_subcurve_preimages import (
    EdgeSubcurvePreimage, PolynomialEdgeAncestor, PolynomialEdgeDefinition)


def prepare(model):
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    return model


def panel(first_angle, *, spline_first=False):
    """Authored quarter-turn cylinder panel between first_angle and +90 degrees."""
    import math
    model = GeometryModel()
    radius = 1.0
    angles = (first_angle, first_angle + math.pi / 2)
    points = {}
    for index, angle in enumerate(angles):
        for level in (0.0, 1.0):
            points[(index, level)] = model.add_point(
                radius * math.cos(angle), radius * math.sin(angle), level)
    via_bottom = model.add_point(radius * math.cos(first_angle + math.pi / 4),
                                 radius * math.sin(first_angle + math.pi / 4), 0.0)
    via_top = model.add_point(radius * math.cos(first_angle + math.pi / 4),
                              radius * math.sin(first_angle + math.pi / 4), 1.0)
    a, b = points[(0, 0.0)], points[(0, 1.0)]
    c, d = points[(1, 1.0)], points[(1, 0.0)]
    if spline_first:
        control = model.add_point(radius * math.cos(first_angle),
                                  radius * math.sin(first_angle), 0.5)
        first = model.add_spline(a, (control,), b)
    else:
        first = model.add_line(a, b)
    arc_top = model.add_arc(b, via_top, c)
    third = model.add_line(c, d)
    arc_bottom = model.add_arc(d, via_bottom, a)
    face = model.add_face_from_loop((OrientedEdge(first, True), OrientedEdge(arc_top, True),
                                     OrientedEdge(third, True), OrientedEdge(arc_bottom, True)),
                                    surface=Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                                     1.0, 1.0, first_angle, math.pi / 2))
    return prepare(model), face


def unpack(value):
    return F(*value)


def test_unsplit_projected_boundary_is_exact_and_owner_bound():
    model, face = panel(0.0)
    committed = to_dict(model)
    result = query(model, face)
    assert result.outcome == 'exact'
    assert result.refusals == ()
    assert result.descendants == (face,)
    assert [root.family for root in result.boundary_roots] == ['straight', 'arc'] * 2
    assert all(root.outcome == 'exact' for root in result.boundary_roots)
    assert all(len(root.children) == 1 for root in result.boundary_roots)
    straights = [root for root in result.boundary_roots if root.family == 'straight']
    assert all(root.constant_angle and root.finiteness_certified for root in straights)
    arcs = [root for root in result.boundary_roots if root.family == 'arc']
    # Unsplit arcs: exact native definition identity PLUS the exact
    # whole-circle finiteness certificate; never exact without finiteness.
    assert all(root.finiteness_certified is True for root in arcs)
    assert all(root.native_identity_certified is True for root in arcs)
    assert all(root.arc_finiteness is not None for root in arcs)
    validate(model, result)
    assert to_dict(model) == committed
    assert query(clone_prepared_geometry(model), face) == result
    # Boundary-only diagnostic: no material, partition, reference or mesh flags.
    for forbidden in ('material_coverage_qualified', 'partition_coverage_qualified',
                      'reference_complete', 'meshing_permitted'):
        assert not hasattr(result, forbidden)
    with pytest.raises(Exception):
        result.outcome = 'refused'


def test_generated_rounded_twelve_panel_definitions_are_exact_without_snapping():
    from anygeometry.generators.structural import cylinder
    model = cylinder(1.0, 1.0)
    prepare(model)
    face = min(model.faces)
    result = query(model, face)
    assert result.outcome == 'exact'
    assert len(result.boundary_roots) == 4
    straights = [root for root in result.boundary_roots if root.family == 'straight']
    arcs = [root for root in result.boundary_roots if root.family == 'arc']
    assert len(straights) == 2 and len(arcs) == 2
    # The rounded float-trig rulings are genuinely constant-angle straights:
    # certified from the actual stored positions, not from nominal 2*pi*i/12.
    assert all(root.constant_angle for root in straights)
    assert all(unpack(root.lift_cross) == 0 and unpack(root.lift_dot) > 0
               for root in straights)
    assert all(unpack(root.radial_min_squared) > 0 for root in straights)
    # The rounded float-trig arcs are certified finite by the exact
    # whole-circle plane/line/circle certificate on their actual stored
    # positions, not by nominal circle substitution.
    assert all(root.finiteness_certified is True for root in arcs)
    assert all(root.arc_finiteness.case == 'line_meets_plane' for root in arcs)
    validate(model, result)


def test_actual_nonconstant_projected_straight_counterexample():
    # Straight (1, eps*t, t) projects to (atan(eps*t), t): the adjudicated
    # counterexample to any constant start-angle carrier substitution.  The
    # tiny off-surface sag stays far inside the model's unchanged tolerances.
    eps = F(1, 2**20)
    model = GeometryModel()
    a = model.add_point(1.0, 0.0, 0.0)
    b = model.add_point(1.0, float(eps), 1.0)
    c = model.add_point(1.0, 0.0, 1.0)
    e1 = model.add_line(a, b)
    e2 = model.add_line(b, c)
    e3 = model.add_line(c, a)
    face = model.add_face_from_loop((OrientedEdge(e1, True), OrientedEdge(e2, True),
                                     OrientedEdge(e3, True)),
                                    surface=Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                                     1.0, 1.0, 0.0, 3.141592653589793))
    prepare(model)
    result = query(model, face)
    assert result.outcome == 'exact'
    root = result.boundary_roots[0]
    assert root.family == 'straight'
    assert root.constant_angle is False
    assert unpack(root.lift_cross) == eps
    assert unpack(root.lift_dot) == 1
    x0, y0, x1, y1 = (unpack(value) for value in root.endpoint_radial)
    assert (x0, y0, x1, y1) == (1, 0, 1, eps)
    assert root.seam_crossed is False
    # Nonzero lift cross certifies theta(1) != theta(0): the projected curve
    # is not the constant carrier, exactly, with no trig evaluation.
    assert unpack(root.radial_min_squared) == 1


def test_on_seam_ruling_records_seam_identity():
    # A ruling exactly on the negative radial ray projects to the constant
    # principal value pi; the seam flag records that branch identity.  The
    # ruling vertices are exact, not float-trig approximations of pi.
    import math
    model = GeometryModel()
    a = model.add_point(-1.0, 0.0, 0.0)
    b = model.add_point(-1.0, 0.0, 1.0)
    c = model.add_point(0.0, -1.0, 1.0)
    d = model.add_point(0.0, -1.0, 0.0)
    via_bottom = model.add_point(math.cos(5 * math.pi / 4), math.sin(5 * math.pi / 4), 0.0)
    via_top = model.add_point(math.cos(5 * math.pi / 4), math.sin(5 * math.pi / 4), 1.0)
    first = model.add_line(a, b)
    arc_top = model.add_arc(b, via_top, c)
    third = model.add_line(c, d)
    arc_bottom = model.add_arc(d, via_bottom, a)
    face = model.add_face_from_loop((OrientedEdge(first, True), OrientedEdge(arc_top, True),
                                     OrientedEdge(third, True), OrientedEdge(arc_bottom, True)),
                                    surface=Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                                     1.0, 1.0, math.pi, math.pi / 2))
    prepare(model)
    result = query(model, face)
    assert result.outcome == 'exact'
    root = result.boundary_roots[0]
    assert root.family == 'straight'
    assert root.seam_crossed is True
    assert root.constant_angle is True
    assert unpack(root.lift_cross) == 0 and unpack(root.lift_dot) == 1
    assert tuple(unpack(value) for value in root.endpoint_radial) == (-1, 0, -1, 0)
    assert unpack(root.radial_min_squared) == 1
    validate(model, result)


def test_arc_finiteness_certificate_is_exact_analytic_proof():
    frame = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 1],
                    'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                    'start_angle': 0.0, 'sweep_angle': 1.0})
    # The adjudicated counterexample: the unchanged circle through
    # (2,0,0),(1,1,0),(1,-1,0) contains the radial origin — its circumcenter
    # is (1,0,0) with r^2 = 1 and it is TANGENT to the radial-null line (the
    # frame axis) at (0,0,0), separation exactly 0.  The whole-circle
    # certificate must refuse: no exact projected outcome exists for it.
    tangent = ((F(2), F(0), F(0)), (F(1), F(1), F(0)), (F(1), F(-1), F(0)))
    certified, evidence, reason = _arc_finiteness(frame, tangent)
    assert certified is False
    assert reason == 'arc finiteness uncertified: circle tangent to radial-null line'
    assert evidence.case == 'line_meets_plane'
    assert tuple(unpack(value) for value in evidence.center) == (1, 0, 0)
    assert unpack(evidence.radius_squared) == 1
    assert unpack(evidence.separation_squared) == 0
    # A cross-section circle: the axis meets its plane at the circumcenter,
    # strictly inside the disk and NOT on the circle — avoided, certified.
    cross_section = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(-1), F(0), F(0)))
    certified, evidence, reason = _arc_finiteness(frame, cross_section)
    assert certified is True and reason is None
    assert evidence.case == 'line_meets_plane'
    assert unpack(evidence.separation_squared) == -1
    # A vertical circle whose plane contains the axis and whose center lies
    # ON the axis: the line crosses the circle — refused, exactly.
    crossing = ((F(1), F(0), F(0)), (F(1), F(0), F(1)), (F(-1), F(0), F(0)))
    certified, evidence, reason = _arc_finiteness(frame, crossing)
    assert certified is False
    assert reason == 'arc finiteness uncertified: circle meets radial-null line'
    assert evidence.case == 'line_in_plane'
    assert tuple(unpack(value) for value in evidence.center) == (0, 0, F(1, 2))
    assert unpack(evidence.radius_squared) == F(5, 4)
    assert unpack(evidence.separation_squared) == F(-5, 4)
    # A circle in a plane containing the axis direction but offset from it:
    # the line is parallel to the plane and disjoint — structurally avoided.
    parallel = ((F(1), F(4), F(0)), (F(1), F(5), F(1)), (F(1), F(6), F(0)))
    certified, evidence, reason = _arc_finiteness(frame, parallel)
    assert certified is True and reason is None
    assert evidence.case == 'line_parallel_to_plane'
    assert evidence.separation_squared is None
    # A circle in a plane containing the line but far from it: certified by
    # exact line/circle distance.
    apart = ((F(4), F(0), F(0)), (F(5), F(0), F(1)), (F(6), F(0), F(0)))
    certified, evidence, reason = _arc_finiteness(frame, apart)
    assert certified is True and reason is None
    assert evidence.case == 'line_in_plane'
    assert unpack(evidence.separation_squared) == 24
    # Degenerate inputs refuse rather than certifying.
    certified, evidence, reason = _arc_finiteness(frame,
        ((F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(2), F(0), F(0))))
    assert certified is False and evidence is None
    assert reason == 'arc finiteness uncertified: collinear arc points'
    degenerate = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [1, 0, 0],
                         'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                         'start_angle': 0.0, 'sweep_angle': 1.0})
    certified, evidence, reason = _arc_finiteness(degenerate,
        ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(-1), F(0), F(0))))
    assert certified is False and evidence is None
    assert reason == 'arc finiteness uncertified: degenerate authored radial frame'


def test_public_arc_radial_zero_fixture_is_typed_model_refusal():
    # An arc whose whole circle meets the radial-null line cannot be legally
    # authored on a cylinder face: any arc consistent with the surface is a
    # cross-section circle, and the radial-null line (through the authored
    # origin along the frame axis) meets each cross-section plane exactly at
    # the circle's center, never ON the circle, so the exact whole-circle
    # certificate always certifies legally authored arcs.  The attempted
    # fixture below (top arc (1,0,0) -> via (1,0,1) -> (-1,0,0), whose circle
    # meets the frame axis) necessarily bulges off the surface and is refused
    # by typed model validation at authoring time — not by the producer.
    # No fake public producer acceptance is invented; the analytic
    # counterexample circles (tangent to / meeting the radial-null line)
    # are proved refused exactly in
    # test_arc_finiteness_certificate_is_exact_analytic_proof, and the
    # public producer's arc refusal path is exercised by
    # test_missing_arc_split_ancestry_refuses_instead_of_inferring.
    import math
    model = GeometryModel()
    a = model.add_point(1.0, 0.0, 0.0)
    b = model.add_point(1.0, 0.0, 1.0)
    c = model.add_point(-1.0, 0.0, 0.0)
    d = model.add_point(0.0, -1.0, 0.0)
    via_mid = model.add_point(math.cos(5 * math.pi / 4), math.sin(5 * math.pi / 4), 0.0)
    via_last = model.add_point(math.cos(7 * math.pi / 4), math.sin(7 * math.pi / 4), 0.0)
    arc_top = model.add_arc(a, b, c)
    arc_mid = model.add_arc(c, via_mid, d)
    arc_last = model.add_arc(d, via_last, a)
    with pytest.raises(GeometryError, match='inconsistent with its explicit surface'):
        model.add_face_from_loop((OrientedEdge(arc_top, True),
                                  OrientedEdge(arc_mid, True),
                                  OrientedEdge(arc_last, True)),
                                 surface=Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                                  1.0, 2.0, math.pi, math.pi))


def test_recorded_arc_split_enclosure_does_not_grant_exact_correspondence():
    from anygeometry.generators.structural import cylinder
    model = cylinder(1.0, 1.0, circumferential_segments=3)
    model.add_plate(model.add_points(((2.0, 0.5, -0.5), (2.0, 0.5, 1.5),
                                      (-2.0, 0.5, 1.5), (-2.0, 0.5, -0.5))))
    prepare(model)
    from anygeometry.prepared_face_preimages import query_prepared_face_preimages
    faces = query_prepared_face_preimages(model)
    split = next(authored for authored, descendants in faces.face_descendants
                 if len(descendants) > 1)
    result = query(model, split)
    assert result.outcome == 'refused'
    arcs = [root for root in result.boundary_roots if root.family == 'arc']
    assert arcs and all(root.outcome == 'refused' for root in arcs)
    assert all(root.reason == 'arc split enclosed within tolerance, not exact' for root in arcs)
    assert all(root.children and all(not child.parameter_mapping_qualified
                                    for child in root.children) for root in arcs)
    assert result.unaccounted_boundary_edges
    assert any('incomplete descendant boundary scope' in refusal
               for refusal in result.refusals)
    # Authenticated straight ancestry still certifies its own roots.
    straights = [root for root in result.boundary_roots if root.family == 'straight']
    assert straights and all(root.outcome == 'exact' for root in straights)
    # Descendant exterior inventory: paired interior incidences between
    # descendants are recorded as opposite orientations on distinct faces.
    for edge, uses in result.interior_incidence:
        assert len(uses) == 2
        assert uses[0][0] != uses[1][0] and uses[0][1] != uses[1][1]
        assert all(face in result.descendants for face, _ in uses)


def test_unsupported_family_refuses_explicitly():
    # A collinear spline is geometrically the ruling, but the authored native
    # family is spline: the certificate refuses rather than re-typing it.
    model, face = panel(0.0, spline_first=True)
    result = query(model, face)
    assert result.outcome == 'refused'
    assert result.boundary_roots[0].family == 'spline'
    assert result.boundary_roots[0].reason == 'unsupported family'
    assert all(root.outcome == 'exact'
               for root in result.boundary_roots[1:])


def test_reversed_use_is_recorded_as_reflection():
    model = GeometryModel()
    import math
    a = model.add_point(1.0, 0.0, 0.0)
    b = model.add_point(1.0, 0.0, 1.0)
    c = model.add_point(0.0, 1.0, 1.0)
    d = model.add_point(0.0, 1.0, 0.0)
    via_bottom = model.add_point(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0)
    via_top = model.add_point(math.cos(math.pi / 4), math.sin(math.pi / 4), 1.0)
    first = model.add_line(a, b)
    arc_top = model.add_arc(b, via_top, c)
    third = model.add_line(d, c)  # Created backwards; traversed reversed below.
    arc_bottom = model.add_arc(d, via_bottom, a)
    face = model.add_face_from_loop((OrientedEdge(first, True), OrientedEdge(arc_top, True),
                                     OrientedEdge(third, False), OrientedEdge(arc_bottom, True)),
                                    surface=Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                                     1.0, 1.0, 0.0, math.pi / 2))
    prepare(model)
    result = query(model, face)
    assert result.outcome == 'exact'
    root = next(root for root in result.boundary_roots if root.family == 'straight'
                and root.children[0].use_forward is False)
    assert root.forward is False
    assert root.children[0].interval == ((0, 1), (1, 1))
    validate(model, result)


def test_projection_uses_the_authored_rational_frame():
    model, face = panel(0.0)
    result = query(model, face)
    import json
    from anygeometry.prepared_face_preimages import query_prepared_authored_face_definition
    payload = json.loads(query_prepared_authored_face_definition(model, face).definition_json)
    surface = payload['face']['surface']
    assert result.frame.origin == tuple((F(float(x)).numerator, F(float(x)).denominator)
                                        for x in surface['origin'])
    assert result.frame.axis == tuple((F(float(x)).numerator, F(float(x)).denominator)
                                      for x in surface['axis'])
    assert result.frame.radial == tuple((F(float(x)).numerator, F(float(x)).denominator)
                                        for x in surface['radial_direction'])
    # The serialized Cylinder stores NO circumferential coefficient: the
    # projection's rational cross is recomputed from the serialized axis and
    # radial for this projection only; it is not a stored native raw-frame
    # coefficient and carries no evaluate/local_uv/atlas semantics.
    assert 'circumferential' not in surface
    axis = [F(float(x)) for x in surface['axis']]
    radial = [F(float(x)) for x in surface['radial_direction']]
    cross = [axis[1] * radial[2] - axis[2] * radial[1],
             axis[2] * radial[0] - axis[0] * radial[2],
             axis[0] * radial[1] - axis[1] * radial[0]]
    assert result.frame.rational_cross == tuple((value.numerator, value.denominator)
                                                for value in cross)
    # Hand-computed radial coordinates of the first ruling endpoint.
    root = result.boundary_roots[0]
    x0, y0, _, _ = (unpack(value) for value in root.endpoint_radial)
    assert (x0, y0) == (1, 0)
    # Effective coefficient semantics: a skewed authored frame (non-unit axis)
    # is used exactly as serialized, with no normalization — the cross and
    # the radial y coordinate both double.
    skewed = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 2],
                     'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                     'start_angle': 0.0, 'sweep_angle': 1.0})
    assert skewed.rational_cross == ((0, 1), (2, 1), (0, 1))
    evidence = _straight_root_evidence(skewed, ((F(0), F(3), F(0)), (F(0), F(3), F(1))))
    x0, y0, x1, y1 = (unpack(value) for value in evidence['endpoint_radial'])
    assert (x0, y0, x1, y1) == (0, 6, 0, 6)


def test_cancellation_check_cancels_without_model_edits():
    model, face = panel(0.0)
    original = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        query(model, face, cancellation_check=lambda _tag: True)
    assert to_dict(model) == original


def test_mutating_callback_cannot_gain_acceptance(monkeypatch):
    import anygeometry.cylinder_boundary_correspondence as module
    model, face = panel(0.0)
    original = to_dict(model)
    last = False
    done = False
    edge_validation = module.validate_prepared_edge_subcurve_preimages_binding

    def final_validation(*args, **kwargs):
        nonlocal last
        last = True
        return edge_validation(*args, **kwargs)

    monkeypatch.setattr(module, 'validate_prepared_edge_subcurve_preimages_binding',
                        final_validation)

    def callback(_tag):
        nonlocal done
        if last and not done:
            done = True
            del model._prepared_face_preimages_receipt
        return False

    with pytest.raises(GeometryError):
        query(model, face, cancellation_check=callback)
    assert done
    assert to_dict(model) == original


def test_stale_revision_and_malformed_binding_refuse():
    model, face = panel(0.0)
    result = query(model, face)
    with pytest.raises(GeometryError, match='stale'):
        query(model, face, expected_revision=model.revision + 5)
    with pytest.raises(GeometryError, match='definition binding'):
        validate(model, replace(result, outcome='refused'))
    with pytest.raises(GeometryError):
        validate(model, 'not a result')
    model.add_point(8, 9, 10)
    with pytest.raises(GeometryError, match='stale'):
        validate(model, result)
    # Unprepared loads carry no local proof and never trust caller arrays.
    with pytest.raises(GeometryError):
        query(from_dict(to_dict(panel(0.0)[0])), face)


def test_engine_certifies_discrepancy_with_exact_rational_bounds():
    frame = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 1],
                    'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                    'start_angle': 0.0, 'sweep_angle': 1.0})
    source = ((F(1), F(0), F(0)), (F(1), F(0), F(1)))
    # Identical child: exact whole-curve parameter correspondence.
    outcome, evidence = _compare_straight(frame, source, (F(0), F(1)), source)
    assert outcome == 'exact' and evidence == {}
    # Purely axial displacement: exact sup |delta a| = 1/2, no angular claim.
    moved = ((F(1), F(0), F(1, 2)), (F(1), F(0), F(3, 2)))
    outcome, evidence = _compare_straight(frame, source, (F(0), F(1)), moved)
    assert outcome == 'certified_discrepancy'
    assert unpack(evidence['axial_error_bound']) == F(1, 2)
    assert evidence['angular_witness'] is None
    assert evidence['opposite_directions'] is False
    # Quarter-turn displacement: exact rational angular lower bound 1.
    turned = ((F(0), F(1), F(0)), (F(0), F(1), F(1)))
    outcome, evidence = _compare_straight(frame, source, (F(0), F(1)), turned)
    assert outcome == 'certified_discrepancy'
    assert unpack(evidence['angular_witness']) == 0
    assert unpack(evidence['angular_error_lower_bound']) == 1
    assert evidence['opposite_directions'] is False
    # Diametral displacement: opposite directions certify |delta theta| = pi.
    flipped = ((F(-1), F(0), F(0)), (F(-1), F(0), F(1)))
    outcome, evidence = _compare_straight(frame, source, (F(0), F(1)), flipped)
    assert outcome == 'certified_discrepancy'
    assert evidence['opposite_directions'] is True
    assert unpack(evidence['angular_error_lower_bound']) == 3


def test_engine_records_seam_crossing_and_radial_zero_exactly():
    frame = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 1],
                    'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                    'start_angle': 0.0, 'sweep_angle': 1.0})
    # A straight chord crossing the negative radial ray: the continuous lift
    # change is in (pi/2, pi), decided by exact cross/dot, never by atan2.
    crossing = ((F(-3), F(4), F(0)), (F(-3), F(-4), F(1)))
    evidence = _straight_root_evidence(frame, crossing)
    assert evidence['seam_crossed'] is True
    assert unpack(evidence['lift_cross']) == 24 and unpack(evidence['lift_dot']) == -7
    assert evidence['constant_angle'] is False
    assert unpack(evidence['radial_min_squared']) == 9
    # A straight chord through the frame axis: the projection is not finite.
    through_axis = ((F(1), F(0), F(0)), (F(-1), F(0), F(0)))
    evidence = _straight_root_evidence(frame, through_axis)
    assert unpack(evidence['radial_min_squared']) == 0
    assert evidence['constant_angle'] is False


def test_engine_tiling_and_rational_helpers():
    assert _tile_spans(((1, F(0), F(1, 2)), (2, F(1, 2), F(1)))) is None
    assert _tile_spans(((1, F(0), F(1, 2)),)) == 'incomplete'
    assert _tile_spans(((1, F(0), F(1, 2)), (2, F(1, 4), F(1)))) == 'conflict'
    assert _tile_spans(((1, F(0), F(0)),)) == 'conflict'
    assert _quadratic_min(1, -1, 1) == F(3, 4)
    assert _quadratic_min(4, 0, 1) == 1
    assert _quadratic_min(-1, 0, 1) == 0
    assert _quadratic_min(0, -2, 1) == -1
    q = _sqrt_ceiling(F(2))
    assert q * q >= 2 and (q - 1) * (q - 1) < 2
    assert _sqrt_ceiling(F(1, 4)) == F(1, 2)


def _synthetic_record(model, edge_id, interval, controls):
    packed = tuple(tuple((value.numerator, value.denominator) for value in point)
                   for point in controls)
    definition = PolynomialEdgeDefinition(edge_id, 1, 2, packed, 'synthetic')
    ancestor = PolynomialEdgeAncestor(model.model_id, model.revision, 'synthetic', definition)
    return EdgeSubcurvePreimage(edge_id, ancestor,
        tuple((value.numerator, value.denominator) for value in interval), definition,
        (), ((0, 1), (0, 1), (0, 1)), (0, 1), None)


def test_straight_ancestry_gaps_and_off_boundary_descendants_refuse():
    model, face = panel(0.0)
    frame = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 1],
                    'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                    'start_angle': 0.0, 'sweep_angle': 1.0})
    controls = {1: ((F(1), F(0), F(0)), (F(1), F(0), F(1)))}
    # A descendant that is not on the current boundary refuses.
    record = _synthetic_record(model, 20, (F(0), F(1)), controls[1])
    result = _straight_root(frame, controls, {1: [record]}, {}, 1, 0, True)
    assert result.outcome == 'refused' and result.reason == 'descendant not on current boundary'
    # A tiling gap in the authenticated coverage refuses.
    first = _synthetic_record(model, 21, (F(0), F(1, 3)), controls[1])
    second = _synthetic_record(model, 22, (F(1, 2), F(1)), controls[1])
    result = _straight_root(frame, controls, {1: [first, second]},
                            {21: (face, True), 22: (face, True)}, 1, 0, True)
    assert result.outcome == 'refused' and result.reason == 'ancestry incomplete'
    # Missing ancestry refuses.
    result = _straight_root(frame, controls, {}, {1: (face, True)}, 1, 0, True)
    assert result.outcome == 'refused' and result.reason == 'ancestry unavailable'
    # A radial zero on the authored straight refuses the projection.
    zero_controls = {1: ((F(1), F(0), F(0)), (F(-1), F(0), F(0)))}
    record = _synthetic_record(model, 23, (F(0), F(1)), zero_controls[1])
    result = _straight_root(frame, zero_controls, {1: [record]}, {23: (face, True)}, 1, 0, True)
    assert result.outcome == 'refused' and result.reason == 'radial zero on authored straight'


def test_directed_use_reversal_refuses_oriented_boundary():
    # Native parameter identity alone is not oriented boundary equality: the
    # child's directed source interval sign combined with its current EdgeUse
    # direction must equal the original authored EdgeUse direction.  The
    # tiling check drops direction and cannot prove orientation.
    model, face = panel(0.0)
    frame = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 1],
                    'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                    'start_angle': 0.0, 'sweep_angle': 1.0})
    controls = {1: ((F(1), F(0), F(0)), (F(1), F(0), F(1)))}
    # Authored use is backward, but the child follows the source interval
    # forward with a forward current use: the current boundary traverses the
    # source opposite to the authored loop.  Reversal refuses.
    record = _synthetic_record(model, 25, (F(0), F(1)), controls[1])
    result = _straight_root(frame, controls, {1: [record]}, {25: (face, True)}, 1, 0, False)
    assert result.outcome == 'refused'
    assert result.reason == 'descendant orientation reversed'
    # The consistent reflected traversal — reversed interval AND reversed
    # use against a backward authored use — is the oriented match, not a
    # reversal, and stays exact.
    reversed_controls = ((F(1), F(0), F(1)), (F(1), F(0), F(0)))
    reflected = _synthetic_record(model, 26, (F(1), F(0)), reversed_controls)
    result = _straight_root(frame, controls, {1: [reflected]}, {26: (face, True)}, 1, 0, False)
    assert result.outcome == 'exact' and result.reason is None


def test_arc_directed_use_reversal_refuses():
    # An unsplit arc must keep the authored use direction: same edge and
    # vertices with a reversed current use refuses, while the identically
    # anchored native identity stays certified as separate evidence.
    frame = _frame({'type': 'cylinder', 'origin': [0, 0, 0], 'axis': [0, 0, 1],
                    'radial_direction': [1, 0, 0], 'radius': 1.0, 'height': 1.0,
                    'start_angle': 0.0, 'sweep_angle': 1.0})
    positions = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(-1), F(0), F(0)))
    arc_definitions = {7: (1, 5, 2, positions)}
    current_edges = {7: ('arc', 1, 5, 2, positions)}
    result = _arc_root(frame, current_edges, arc_definitions, {7: (3, True)}, 7, 0, False)
    assert result.outcome == 'refused'
    assert result.reason == 'descendant orientation reversed'
    assert result.native_identity_certified is True
    assert result.finiteness_certified is None
    assert result.children == ()
    # Same directed use as authored: exact, with the whole-circle finiteness
    # certificate on the identically anchored native identity.
    result = _arc_root(frame, current_edges, arc_definitions, {7: (3, False)}, 7, 0, False)
    assert result.outcome == 'exact'
    assert result.finiteness_certified is True and result.native_identity_certified is True


def test_temporary_live_mutation_restored_by_callback_returns_unpolluted_entry_result(monkeypatch):
    # A callback temporarily moves an arc via vertex after the entry snapshot
    # and a later stage restores it before final owner revalidation.  The
    # derivation must use the detached entry snapshot only: the result is
    # exactly the unpolluted entry result, never a contaminated certificate.
    import numpy as np
    import anygeometry.cylinder_boundary_correspondence as module
    model, face = panel(0.0)
    clean = query(model, face)
    committed = to_dict(model)
    arc_edge = next(edge for edge in model.edges
                    if isinstance(model.edges[edge].curve, Arc))
    via = model.edges[arc_edge].curve.via_vertex
    original_position = model.vertices[via].position
    stage = {}
    real_edge_query = module.query_prepared_edge_subcurve_preimages

    def edge_query_wrapper(*args, **kwargs):
        binding = real_edge_query(*args, **kwargs)
        stage['edges_done'] = True
        return binding

    monkeypatch.setattr(module, 'query_prepared_edge_subcurve_preimages', edge_query_wrapper)
    real_validation = module.validate_prepared_edge_subcurve_preimages_binding

    def final_validation(*args, **kwargs):
        if stage.get('polluted'):
            object.__setattr__(model.vertices[via], 'position', original_position)
            stage['polluted'] = False
        return real_validation(*args, **kwargs)

    monkeypatch.setattr(module, 'validate_prepared_edge_subcurve_preimages_binding',
                        final_validation)

    def callback(_tag):
        if stage.get('edges_done') and 'polluted' not in stage:
            object.__setattr__(model.vertices[via], 'position', np.array([9.0, 9.0, 9.0]))
            stage['polluted'] = True
        return False

    result = query(model, face, cancellation_check=callback)
    assert stage == {'edges_done': True, 'polluted': False}
    assert result == clean
    assert to_dict(model) == committed
    validate(model, result)
    assert to_dict(model) == committed


def test_temporary_owner_record_controls_mutation_returns_unpolluted_entry_result(monkeypatch):
    # The owner edge-preimage records are frozen dataclasses but remain
    # mutable through object.__setattr__ and are shared with the owner
    # registry.  A callback temporarily rewrites one authenticated straight
    # ancestry record's CURRENT controls after the query has consumed the
    # actual owner receipt and a later stage restores them before final
    # owner revalidation.  The derivation must use the detached entry
    # snapshot only: the result is exactly the unpolluted entry result,
    # never a contaminated certified_discrepancy built from the temporarily
    # mutated live record, and the returned proof shares no record object
    # with the owner registry.
    import anygeometry.cylinder_boundary_correspondence as module
    from anygeometry.definition_binding import definition_checksum
    model, face = panel(0.0)
    clean = query(model, face)
    committed = to_dict(model)
    # Actual owner query receipt: the authenticated straight ancestry record.
    binding = module.query_prepared_edge_subcurve_preimages(model)
    straight_roots = {root.root_edge_id for root in clean.boundary_roots
                      if root.family == 'straight'}
    record = next(row for row in binding.records
                  if row.ancestor.definition.edge_id in straight_roots)
    original_controls = record.current_definition.controls
    # Displaced current segment (0, 1, 0) -> (0, 1, 1): a derivation reading
    # this live record would certify a quarter-turn angular discrepancy.
    displaced = (((0, 1), (1, 1), (0, 1)), ((0, 1), (1, 1), (1, 1)))
    stage = {}
    real_edge_query = module.query_prepared_edge_subcurve_preimages

    def edge_query_wrapper(*args, **kwargs):
        owner = real_edge_query(*args, **kwargs)
        stage['edges_done'] = True
        return owner

    monkeypatch.setattr(module, 'query_prepared_edge_subcurve_preimages', edge_query_wrapper)
    real_validation = module.validate_prepared_edge_subcurve_preimages_binding

    def final_validation(*args, **kwargs):
        if stage.get('polluted'):
            object.__setattr__(record.current_definition, 'controls', original_controls)
            stage['polluted'] = False
        return real_validation(*args, **kwargs)

    monkeypatch.setattr(module, 'validate_prepared_edge_subcurve_preimages_binding',
                        final_validation)

    def callback(_tag):
        if stage.get('edges_done') and 'polluted' not in stage:
            object.__setattr__(record.current_definition, 'controls', displaced)
            stage['polluted'] = True
        return False

    result = query(model, face, cancellation_check=callback)
    assert stage == {'edges_done': True, 'polluted': False}
    assert result == clean
    assert definition_checksum(result) == definition_checksum(clean)
    assert to_dict(model) == committed
    live_records = {id(row) for row in (*binding.records, *binding.alias_records)}
    assert all(id(row) not in live_records
               for row in (*result.edge_preimages.records, *result.edge_preimages.alias_records))
    validate(model, result)
    assert to_dict(model) == committed


def test_unrestored_live_mutation_is_typed_refusal(monkeypatch):
    # The same temporary mutation without the restore is a typed owner
    # refusal, never a certificate built from polluted live input.
    import numpy as np
    import anygeometry.cylinder_boundary_correspondence as module
    model, face = panel(0.0)
    committed = to_dict(model)
    arc_edge = next(edge for edge in model.edges
                    if isinstance(model.edges[edge].curve, Arc))
    via = model.edges[arc_edge].curve.via_vertex
    original_position = model.vertices[via].position
    stage = {}
    real_edge_query = module.query_prepared_edge_subcurve_preimages

    def edge_query_wrapper(*args, **kwargs):
        binding = real_edge_query(*args, **kwargs)
        stage['edges_done'] = True
        return binding

    monkeypatch.setattr(module, 'query_prepared_edge_subcurve_preimages', edge_query_wrapper)

    def callback(_tag):
        if stage.get('edges_done') and 'polluted' not in stage:
            object.__setattr__(model.vertices[via], 'position', np.array([9.0, 9.0, 9.0]))
            stage['polluted'] = True
        return False

    with pytest.raises(GeometryError):
        query(model, face, cancellation_check=callback)
    assert stage.get('polluted') is True
    # Restore outside the query: the model itself is unchanged by the probe.
    object.__setattr__(model.vertices[via], 'position', original_position)
    assert to_dict(model) == committed


def test_validator_refuses_forgery_repaired_during_callback():
    # The validator pins the request target/revision and the input checksum
    # at entry, before any callback: an initially forged result repaired
    # in place during the rederivation callback still refuses.
    model, face = panel(0.0)
    result = query(model, face)
    committed = to_dict(model)
    object.__setattr__(result, 'outcome', 'refused')
    repaired = []

    def repair_callback(_tag):
        if not repaired:
            repaired.append(True)
            object.__setattr__(result, 'outcome', 'exact')
        return False

    with pytest.raises(GeometryError, match='input changed during validation'):
        validate(model, result, cancellation_check=repair_callback)
    assert repaired
    assert to_dict(model) == committed
    # Mutating a legitimate input in place during callbacks is refused by the
    # same entry-pinned input checksum, even though the owner rederivation
    # itself is clean.
    legitimate = query(model, face)
    mutated = []

    def mutate_callback(_tag):
        if not mutated:
            mutated.append(True)
            object.__setattr__(legitimate, 'unaccounted_boundary_edges', (999,))
        return False

    with pytest.raises(GeometryError, match='input changed during validation'):
        validate(model, legitimate, cancellation_check=mutate_callback)
    assert mutated
    assert to_dict(model) == committed
