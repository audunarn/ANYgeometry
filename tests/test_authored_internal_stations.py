"""Independent exact x=3 line oracle; no native or mesh qualification."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import (GeometryError, GeometryModel, Plane, to_dict, apply_intersections,
                         plan_intersections, query_prepared_authored_boundary_correspondence,
    query_prepared_authored_internal_stations as query,
    validate_prepared_authored_internal_station_coordinates as validate,
)
from examples.authored_child_coverage_handoff import build


def fixture():
    model, correspondence, left, right = build()
    edge, incidence = correspondence.interior_incidence[0]
    assert edge == 25 and (left, right) == (3, 4)
    return model, correspondence, edge, incidence


def unpack(points):
    return tuple(tuple(F(*value) for value in point) for point in points)


def xyz(receipt):
    return np.array([[float(value) for value in point] for point in unpack(receipt.current_points)],
                    dtype=float).reshape((-1, 3))


def planar_source(*, reverse=False, sheet=False, reverse_edge=False):
    model = GeometryModel()
    if reverse_edge:
        from anygeometry.entities import OrientedEdge
        vertices = model.add_points(((0,0,0),(3,0,0),(4,0,0),(4,4,0),(3,4,0),(0,4,0)))
        edges = model.add_polyline(vertices,close=True)
        root = model.add_face_from_loop(tuple(OrientedEdge(edge,True) for edge in edges),(0,2,3,5),
                                        surface=Plane((0,0,0),(1,0,0),(0,1,0)))
        model.add_line(vertices[4],vertices[1])
    else:
        root = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    model.set_face_surface(root, Plane((0,4,0),(1,0,0),(0,-1,0)) if reverse else
                           Plane((0,0,0),(1,0,0),(0,1,0)))
    if sheet: model.add_sheet((root,), name='paired incidence source')
    model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    apply_intersections(model, plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding = query_prepared_authored_boundary_correspondence(model, root)
    return model, binding, binding.interior_incidence[0][0]


def test_exact_portable_line_order_duplicates_and_rational_non_binary64_parameters():
    model, binding, edge, incidence = fixture()
    before = to_dict(model)
    caches = dict(model._edge_length_cache)
    parameters = (F(1,3), 1, 0, F(1,3), F(1,10))
    receipt = query(model, binding, edge, parameters)
    first, last = unpack(receipt.endpoint_points)
    expected = tuple((F(3), (1-p)*first[1]+p*last[1], F(0)) for p in parameters)
    assert unpack(receipt.current_points) == expected == unpack(receipt.authored_points)
    assert unpack(receipt.authored_uv) == tuple(point[:2] for point in expected)
    assert receipt.parameters == tuple((F(p).numerator, F(p).denominator) for p in parameters)
    assert receipt.endpoint_ids == (model.edges[edge].start, model.edges[edge].end)
    assert receipt.child_incidence == incidence
    assert {forward for _, forward in incidence} == {False, True}
    assert type(receipt.decomposition_seam_tag) is bool
    validate(model, receipt, xyz(receipt))
    assert to_dict(model) == before and model._edge_length_cache == caches
    with pytest.raises(FrozenInstanceError):
        receipt.edge_id = -1


def test_reverse_source_order_preserves_stored_direction_and_incidence():
    model, binding, edge, incidence = fixture()
    forward = query(model, binding, edge, (0, F(1,3), 1))
    reverse = query(model, binding, edge, (1, F(1,3), 0))
    assert reverse.endpoint_ids == forward.endpoint_ids
    assert reverse.current_points == forward.current_points[::-1]
    assert reverse.child_incidence == incidence
    assert reverse.authored_uv == forward.authored_uv[::-1]


def test_reversed_plane_orientation_preserves_authentic_stored_edge_direction():
    model, binding, edge = planar_source(reverse=True)
    receipt = query(model, binding, edge, (0,F(1,3),1))
    first, last = unpack(receipt.endpoint_points)
    assert first[0] == last[0] == 3 and {first[1],last[1]} == {0,4}
    expected = tuple((F(3),(1-p)*first[1]+p*last[1],F(0)) for p in (0,F(1,3),1))
    assert unpack(receipt.current_points) == expected
    assert unpack(receipt.authored_uv) == tuple((point[0],4-point[1]) for point in expected)
    validate(model,receipt,xyz(receipt))


def test_preexisting_reversed_line_preserves_authentic_stored_direction():
    model,binding,edge,_=fixture()
    normal=query(model,binding,edge,(0,F(1,3),1))
    model,binding,edge=planar_source(reverse_edge=True)
    reversed_line=query(model,binding,edge,(0,F(1,3),1))
    assert reversed_line.endpoint_points==normal.endpoint_points[::-1]
    first,last=unpack(reversed_line.endpoint_points)
    assert unpack(reversed_line.current_points)==tuple(tuple((1-p)*a+p*b for a,b in zip(first,last))
                                                       for p in (0,F(1,3),1))
    assert unpack(reversed_line.authored_uv)==tuple(row[:2] for row in unpack(reversed_line.current_points))
    validate(model,reversed_line,xyz(reversed_line))


def test_edge_identifier_is_normalized_once_and_coercion_edits_refuse():
    model,binding,edge,_=fixture()
    calls=[]
    class Identifier(int):
        def __int__(self):
            calls.append(1)
            return edge
    assert query(model,binding,Identifier(edge),(.5,)).edge_id==edge
    assert calls==[1]
    class ChangingIdentifier(int):
        def __int__(self):
            model.add_point(80,80,80)
            return edge
    with pytest.raises(GeometryError):query(model,binding,ChangingIdentifier(edge),(.5,))


def test_parameter_iteration_cannot_change_the_authenticated_source():
    model,binding,edge,_=fixture()
    def values():
        model.add_point(81,81,81)
        yield F(1,3)
    with pytest.raises(GeometryError):query(model,binding,edge,values())


def test_structural_occurrence_ids_and_orientations_are_retained():
    model, binding, edge = planar_source(sheet=True)
    receipt = query(model,binding,edge,(0,1))
    assert receipt.occurrences
    for coedge, face_use, sheet, face, direction, face_direction in receipt.occurrences:
        row, use = model.coedges[coedge], model.face_uses[face_use]
        assert row.edge_id == edge and row.face_use_id == face_use
        assert (sheet,face,direction,face_direction) == (
            use.sheet_id,use.face_id,row.orientation.value,use.orientation.value)
        assert face in {child for child,_ in receipt.child_incidence}


@pytest.mark.parametrize('value', [np.float16(.5), np.float32(.3), np.float64(.7), np.int64(1)])
def test_numpy_parameters_keep_their_exact_value(value):
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, np.int64(edge), (value,))
    exact = F(int(value)) if isinstance(value, np.integer) else F(*value.as_integer_ratio())
    assert receipt.parameters == ((exact.numerator, exact.denominator),)


@pytest.mark.parametrize('parameters', [(True,), (np.bool_(True),), ('0.5',), (complex(.5),),
                                        (float('nan'),), (float('inf'),), (-.1,), (1.1,), None])
def test_invalid_parameters_refuse(parameters):
    model, binding, edge, _ = fixture()
    with pytest.raises(GeometryError): query(model, binding, edge, parameters)


@pytest.mark.parametrize('edge_id', [True, np.bool_(True), 1., '25', 0, -1, 999])
def test_invalid_or_unpaired_edge_refuses(edge_id):
    model, binding, _, _ = fixture()
    with pytest.raises(GeometryError): query(model, binding, edge_id, (F(1,3),))


@pytest.mark.parametrize('kind', ['revision', 'vertex', 'edge', 'wrong-model'])
def test_stale_raw_changes_and_wrong_model_refuse(kind):
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, (0,.5,1))
    if kind == 'revision': model.add_point(99,99,99)
    if kind == 'vertex':
        key = model.edges[edge].start
        model._vertices[key] = replace(model.vertices[key], position=(3,0,1e-12))
    if kind == 'edge': model._edges[edge] = replace(model.edges[edge], start=model.edges[edge].end)
    if kind == 'wrong-model': model = GeometryModel()
    with pytest.raises(GeometryError): query(model, binding, edge, (0,.5,1))
    with pytest.raises(GeometryError): validate(model, receipt, xyz(receipt))


@pytest.mark.parametrize('field,value', [('edge_id',999), ('parameters',((1,2),)),
    ('endpoint_ids',(1,2)), ('decomposition_seam_tag',True), ('tolerance',(1,1)),
    ('authored_points',(((3,1),(2,1),(1,1)),)), ('current_points',(((3,1),(2,1),(1,1)),)),
    ('child_incidence',()), ('occurrences',((1,2,3,4,1,1),))])
def test_forged_receipt_fields_refuse(field,value):
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, (F(1,3),))
    # Pick the opposite tag value so this always changes the definition.
    if field == 'decomposition_seam_tag': value = not receipt.decomposition_seam_tag
    with pytest.raises(GeometryError): validate(model, replace(receipt, **{field:value}), xyz(receipt))


@pytest.mark.parametrize('parameters', [((1,0),), ((1,),), ((object(),1),)])
def test_malformed_packed_receipt_parameters_are_typed_refusals(parameters):
    model,binding,edge,_=fixture()
    receipt=query(model,binding,edge,(.5,))
    with pytest.raises(GeometryError):
        validate(model,replace(receipt,parameters=parameters),xyz(receipt))


@pytest.mark.parametrize('bad', [np.zeros((1,2)), [[3,float('nan'),0]], [[3,2,1j]], [[3,2,1e-4]]])
def test_coordinate_shape_finite_real_and_existing_tolerance(bad):
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, (.5,))
    with pytest.raises(GeometryError): validate(model, receipt, bad)
    validate(model, receipt, xyz(receipt))


def test_empty_stations_have_an_explicit_empty_coordinate_shape():
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, ())
    validate(model, receipt, np.empty((0,3)))
    assert receipt.current_points == receipt.authored_uv == ()


def test_cancellation_and_callback_exception_identity_without_mutation():
    model, binding, edge, _ = fixture()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        query(model, binding, edge, (.5,), cancellation_check=lambda _:True)
    receipt = query(model, binding, edge, (.5,))
    error = RuntimeError('caller cancellation exception')
    def raise_error(_): raise error
    with pytest.raises(RuntimeError) as caught:
        validate(model, receipt, xyz(receipt), cancellation_check=raise_error)
    assert caught.value is error and to_dict(model) == before


def test_transient_endpoint_edit_cannot_replace_detached_source(monkeypatch):
    import anygeometry.authored_internal_stations as owner
    model, binding, edge, _ = fixture()
    before = to_dict(model)
    expected = query(model, binding, edge, (F(1,3),))
    vertex = model.edges[edge].start
    original = model.vertices[vertex]
    real = owner._original_domain
    stage = {'reconstruct':False, 'changed':False}
    def reconstruct(definition,check):
        stage['reconstruct'] = True
        result = real(definition,check)
        stage['reconstruct'] = False
        return result
    def callback(_):
        if stage['reconstruct']:
            stage['changed'] = True
            model._vertices[vertex] = replace(original, position=(3,0,1))
        else: model._vertices[vertex] = original
        return False
    monkeypatch.setattr(owner,'_original_domain',reconstruct)
    try:
        assert query(model, binding, edge, (F(1,3),), cancellation_check=callback) == expected
        assert stage['changed'] and to_dict(model) == before
    finally: model._vertices[vertex] = original


def test_late_callback_edits_and_coordinate_receipt_mutation_refuse():
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, (.5,))
    def edit(_):
        model._edges[edge] = replace(model.edges[edge], start=model.edges[edge].end)
        return False
    with pytest.raises(GeometryError): query(model, binding, edge, (.5,), cancellation_check=edit)
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, (.5,))
    def forge(_):
        object.__setattr__(receipt,'current_points',(((3,1),(2,1),(1,1)),))
        return False
    with pytest.raises(GeometryError): validate(model, receipt, xyz(receipt), cancellation_check=forge)


def test_input_coercion_and_caller_coordinate_edits_are_guarded():
    model, binding, edge, _ = fixture()
    receipt = query(model, binding, edge, (.5,))
    coordinates = xyz(receipt)
    def callback(_):
        coordinates[:] = 99
        return False
    validate(model, receipt, coordinates, cancellation_check=callback)
    class Input:
        def __array__(self,dtype=None,copy=None):
            model.add_point(70,70,70)
            return np.array([[3,2,0]],dtype=dtype)
    with pytest.raises(GeometryError): validate(model, receipt, Input())


def test_portable_handoff_reports_only_station_evidence():
    from examples.authored_internal_stations_handoff import verify
    result = verify()
    assert result['internal_edge'] == 25 and result['children'] == (3,4)
    assert result['accepted_mesh'] is result['subdivision_permission'] is False


def test_public_paired_curved_edge_refuses_without_straight_substitution():
    from anygeometry.curves import Straight
    model=GeometryModel()
    root=model.add_plate(model.add_points(((0,-2,1),(3,-2,1),(3,2,1),(0,2,1))))
    model.set_face_surface(root,Plane((0,0,1),(1,0,0),(0,1,0)))
    controls=model.add_points(((0,0,0),(1,1,0),(2,1,0),(3,0,0)))
    spline=model.add_spline(controls[0],controls[1:-1],controls[-1])
    model.extrude((spline,),(0,0,2))
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding=query_prepared_authored_boundary_correspondence(model,root)
    curved=[edge for edge,_ in binding.interior_incidence if type(model.edges[edge].curve) is not Straight]
    assert curved
    before=to_dict(model)
    with pytest.raises(GeometryError,match='literal Straight'):
        query(model,binding,curved[0],(F(1,3),))
    assert to_dict(model)==before


def test_nonorthogonal_tilted_plane_has_exact_original_uv():
    model=GeometryModel()
    origin=np.array((.5,.25,-1.)); u=np.array((1.,0.,1.)); v=np.array((1.,1.,0.))
    normal=np.array((-1.,1.,1.))
    point=lambda a,b: origin+a*u+b*v
    root=model.add_plate(model.add_points(tuple(point(a,b) for a,b in ((0,0),(4,0),(4,4),(0,4)))))
    model.set_face_surface(root,Plane(origin,u,v))
    model.add_plate(model.add_points((point(3,-1)-normal,point(3,5)-normal,
                                     point(3,5)+normal,point(3,-1)+normal)))
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding=query_prepared_authored_boundary_correspondence(model,root)
    edge=binding.interior_incidence[0][0]
    receipt=query(model,binding,edge,(0,F(1,3),1))
    # The existing arrangement stores a last-bit displaced line. This query
    # maps its literal direction, not the ideal intersection at u=3. Independent
    # inversion of this explicit frame is u=z+1 and v=y-1/4.
    expected_uv=tuple((row[2]+1,row[1]-F(1,4)) for row in unpack(receipt.current_points))
    assert unpack(receipt.authored_uv) == expected_uv
    assert all(abs(row[0]-3) <= F(float(np.finfo(float).eps))*8 for row in expected_uv)
    for (a,b), xyz_row in zip(unpack(receipt.authored_uv),unpack(receipt.authored_points)):
        assert xyz_row == tuple(F(float(o))+a*F(float(x))+b*F(float(y)) for o,x,y in zip(origin,u,v))
    validate(model,receipt,xyz(receipt))


def test_coordinate_tolerance_closed_boundary_is_exact():
    model,binding,edge,_=fixture()
    receipt=query(model,binding,edge,(.5,))
    tolerance=float(F(*receipt.tolerance))
    coordinates=xyz(receipt)
    assert coordinates[0,2] == 0.
    coordinates[0,2]=tolerance
    validate(model,receipt,coordinates)
    coordinates[0,2]=np.nextafter(tolerance,np.inf)
    with pytest.raises(GeometryError,match='coordinate error'):
        validate(model,receipt,coordinates)


def test_exact_plane_lift_refuses_a_sub_tolerance_offset(monkeypatch):
    import anygeometry.authored_internal_stations as owner
    model,binding,edge,_=fixture()
    original=owner._original_domain
    def offset_definition(definition, check):
        domain=original(definition,check)
        # A bound original snapshot is normally immutable. Inject only this
        # exact-lift discriminator; stale-receipt tests cover authentication.
        return replace(domain,support=Plane((0,0,1e-12),(1,0,0),(0,1,0)))
    assert 1e-12 < model.tolerance.effective_length(model.edge_length(edge))
    monkeypatch.setattr(owner,'_original_domain',offset_definition)
    with pytest.raises(GeometryError,match='trim is not exactly on support'):
        query(model,binding,edge,(.5,))
