"""Stored polynomial support identities; no trimmed-material promotion.

Authored source-only in this assignment. These tests have not been executed.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import json

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, to_dict
from anygeometry.extrusions import BezierDirectrix
from anygeometry.surfaces import CoonsSurface, ExtrudedSurface
from anygeometry.native_support_snapshots import capture_native_supports
from anygeometry.polynomial_extrusion_support import (
    COONS_POLYNOMIAL, BEZIER_EXTRUSION, prove_polynomial_extrusion_support as prove,
)
from anygeometry.native_material_reference_scope import (
    query_prepared_native_material_reference_scope as query,
    validate_prepared_native_material_reference_scope_binding as validate,
)

B = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
D = (.25, 0., 1.5)


def translate(points, vector):
    return tuple(tuple(F(x)+F(y) for x, y in zip(point, vector)) for point in points)


def coons(bottom=B, direction=D, reverse_storage=False):
    top = translate(bottom, direction)
    oriented = (tuple(bottom), (bottom[-1], top[-1]), top[::-1], (top[0], bottom[0]))
    return ((0, 1, 2, 3), tuple((i+1, not reverse_storage,
        'spline' if i in (0, 2) else 'straight',
        tuple(tuple(point) for point in (points[::-1] if reverse_storage else points)))
        for i, points in enumerate(oriented)))


def extrusion(controls=B, direction=D, u=(0., 1.), v=(0., 1.)):
    controls = tuple(tuple(point) for point in controls)
    return controls, controls, tuple(direction), tuple(direction), tuple(u), tuple(v)


def unpack_map(result):
    return tuple(tuple(F(*x) for x in row) for row in result['child_to_original_uv'])


@pytest.mark.parametrize('reverse_storage', (False, True))
def test_exact_cubic_coons_identity_uses_oriented_coefficients(reverse_storage):
    source = coons(reverse_storage=reverse_storage)
    before = deepcopy(source)
    result = prove(COONS_POLYNOMIAL, source, BEZIER_EXTRUSION, extrusion())
    assert unpack_map(result) == ((0, 1), (0, 1))
    assert result['orientation_sign'] == 1
    assert result['support_correspondence_qualified']
    assert not any(result[key] for key in ('trimmed_material_qualified',
        'reference_mapping_qualified', 'floating_evaluation_preservation_qualified', 'meshing_permitted'))
    assert source == before


@pytest.mark.parametrize('u,v,sign', [((.25,.75),(.125,.875),1), ((.75,.25),(.125,.875),-1),
                                     ((.75,.25),(.875,.125),1)])
def test_cropped_and_reversed_parameter_ranges_have_exact_signed_maps(u, v, sign):
    result = prove(COONS_POLYNOMIAL, coons(), BEZIER_EXTRUSION, extrusion(u=u, v=v))
    assert unpack_map(result) == ((F(u[0]), F(u[1])-F(u[0])), (F(v[0]), F(v[1])-F(v[0])))
    assert result['orientation_sign'] == sign
    assert F(*result['parameter_jacobian_determinant']) == (F(u[1])-F(u[0]))*(F(v[1])-F(v[0]))


def test_reversed_directrix_and_negative_ruling_scale_are_distinguished():
    top = translate(B, D)
    child = extrusion(top[::-1], tuple(-F(x) for x in D))
    result = prove(COONS_POLYNOMIAL, coons(), BEZIER_EXTRUSION, child)
    assert unpack_map(result) == ((1, -1), (1, -1))
    assert result['orientation_sign'] == 1
    # Independently compose the exact polynomial coefficients, no owner evaluator.
    for u, v in ((F(0), F(0)), (F(1,3), F(2,7)), (F(1), F(1))):
        t, s = 1-u, 1-v
        original = (3*t+s/4, 6*t-15*t*t+10*t**3, 3*s/2)
        from math import comb
        current = tuple(sum(F(child[0][i][axis])*comb(3,i)*u**i*(1-u)**(3-i)
                            for i in range(4))-v*F(D[axis]) for axis in range(3))
        assert current == original


def test_exact_degree_elevation_and_world_affine_coefficients():
    elevated = (B[0], *tuple(tuple(F(i,4)*F(B[i-1][a])+(1-F(i,4))*F(B[i][a])
                           for a in range(3)) for i in range(1,4)), B[-1])
    assert unpack_map(prove(COONS_POLYNOMIAL, coons(), BEZIER_EXTRUSION, extrusion(elevated))) == ((0,1),(0,1))
    # Exact affine transform with reflection/shear; transformed definitions,
    # rather than transform provenance, must independently satisfy identities.
    def affine(p):
        x,y,z = map(F,p)
        return -2*x+y/2+8, y+z/4-4, 2*z
    points = tuple(affine(p) for p in B)
    q0, qd = affine((0,0,0)), affine(D)
    direction = tuple(x-y for x,y in zip(qd,q0))
    result = prove(COONS_POLYNOMIAL, coons(points,direction), BEZIER_EXTRUSION, extrusion(points,direction))
    assert result['orientation_sign'] == 1


@pytest.mark.parametrize('defect', ('top', 'connector', 'curve-connector', 'corners', 'piecewise'))
def test_coons_tolerance_only_and_different_chart_definitions_refuse(defect):
    corners, sides = coons()
    sides = list(sides)
    if defect in ('top', 'connector'):
        index = 2 if defect == 'top' else 1
        edge, forward, kind, controls = sides[index]
        controls = list(controls); point = list(controls[0]); point[0] = F(point[0])+F(1,2**45)
        controls[0] = tuple(point); sides[index] = (edge, forward, kind, tuple(controls))
    elif defect == 'curve-connector':
        edge, forward, _, controls = sides[1]
        sides[1] = (edge, forward, 'spline', controls)
    elif defect == 'corners': corners = (0,1,3,4)
    else: sides.append(sides[-1])
    with pytest.raises(GeometryError):
        prove(COONS_POLYNOMIAL, (corners,tuple(sides)), BEZIER_EXTRUSION, extrusion())


@pytest.mark.parametrize('defect', ('raw-controls', 'raw-vector', 'rounded-control', 'outside-u', 'outside-v', 'zero-range', 'nonfinite', 'legacy'))
def test_child_or_legacy_unsupported_definitions_refuse(defect):
    data = list(extrusion())
    if defect in ('raw-controls', 'rounded-control'):
        points = list(data[1]); points[1] = (F(1)+F(1,2**45),F(2),F(0)); data[1] = tuple(points)
        if defect == 'rounded-control': data[0] = data[1]
    elif defect == 'raw-vector': data[3] = (F(1,4),F(0),F(3,2)+F(1,2**45))
    elif defect == 'outside-u': data[4] = (0.,1.01)
    elif defect == 'outside-v': data[5] = (-.01,1.)
    elif defect == 'zero-range': data[5] = (0.,0.)
    elif defect == 'nonfinite': data[5] = (0.,float('inf'))
    if defect == 'legacy':
        with pytest.raises(GeometryError, match='unavailable'):
            prove('unsupported', (), BEZIER_EXTRUSION, tuple(data))
    else:
        with pytest.raises(GeometryError):
            prove(COONS_POLYNOMIAL, coons(), BEZIER_EXTRUSION, tuple(data))


def test_rank_and_injectivity_are_proved_not_assumed():
    with pytest.raises(GeometryError, match='zero ruling'):
        prove(COONS_POLYNOMIAL, coons(direction=(0,0,0)), BEZIER_EXTRUSION, extrusion())
    # Retracing projected profile: endpoints agree, so injectivity is impossible.
    retraced = ((0,0,0),(1,1,0),(1,-1,0),(0,0,0))
    with pytest.raises(GeometryError, match='unproved'):
        prove(COONS_POLYNOMIAL, coons(retraced), BEZIER_EXTRUSION, extrusion(retraced))


def test_charged_work_callback_failure_identity():
    sentinel = GeometryError('owned callback sentinel')
    calls = []
    def charge():
        calls.append(1)
        if len(calls) == 5: raise sentinel
    with pytest.raises(GeometryError) as caught:
        prove(COONS_POLYNOMIAL, coons(), BEZIER_EXTRUSION, extrusion(), charge=charge)
    assert caught.value is sentinel and len(calls) == 5


def bare_wall():
    model = GeometryModel()
    points = model.add_points(B)
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    face, = model.extrude((edge,), D)
    return model, face


def test_prospective_capture_detaches_coons_and_raw_extrusion_coefficients():
    model, face = bare_wall()
    snapshot = capture_native_supports(model)
    assert snapshot[0][1] == COONS_POLYNOMIAL
    assert snapshot[0][2][0] == (0,1,2,3)
    assert tuple(row[0] for row in snapshot[0][2][1]) == tuple(use.edge for use in model.faces[face].loop)
    assert unpack_map(prove(snapshot[0][1], snapshot[0][2], BEZIER_EXTRUSION, extrusion())) == ((0,1),(0,1))
    support = ExtrudedSurface(BezierDirectrix(B), D)
    model._faces[face] = replace(model.faces[face], surface=support)
    original = to_dict(model)
    captured = capture_native_supports(model)
    raw = support.directrix._array.copy(); raw[1,0] += 2**-40
    object.__setattr__(support.directrix, '_array', raw)
    changed = capture_native_supports(model)
    assert to_dict(model) == original and captured != changed
    with pytest.raises(GeometryError, match='public/raw'):
        prove(COONS_POLYNOMIAL, snapshot[0][2], changed[0][1], changed[0][2])
    assert captured[0][2][0] == captured[0][2][1]


def test_public_native_query_exposes_support_only_and_preserves_refusals():
    from test_authored_boundary_correspondence import fixture
    model, root, original = fixture(cubic=True)
    before = to_dict(model)
    result = query(model, [root])
    row, = result.material_rows
    assert row.classification == 'support_only', row.refusal
    assert row.physical_support_qualified and not row.document_material_qualified
    evidence = json.loads(row.domain_evidence_json)
    assert {child for child,_ in evidence['polynomial_support_correspondence']} == set(row.current_face_ids)
    assert not result.document_material_qualified
    assert not result.geometry_native_reference_maps_qualified
    assert not result.meshing_permitted and not result.external_reference_scope_qualified
    validate(model, result)
    assert to_dict(model) == before
    assert next(f for f in original['faces'] if f['id'] == root)['surface'] == {'type':'coons'}


def test_public_binding_rejects_raw_same_revision_mutation_and_tamper():
    from test_authored_boundary_correspondence import fixture
    model, root, _ = fixture(cubic=True)
    receipt = query(model, [root])
    before = to_dict(model)
    child = receipt.material_rows[0].current_face_ids[0]
    support = model.faces[child].surface
    assert type(support) is ExtrudedSurface
    raw = support._vector.copy()
    object.__setattr__(support, '_vector', raw+np.array((0.,0.,2**-40)))
    assert to_dict(model) == before
    with pytest.raises(GeometryError): validate(model, receipt)
    object.__setattr__(support, '_vector', raw)
    validate(model, receipt)
    forged = replace(receipt, document_material_qualified=True)
    with pytest.raises(GeometryError): validate(model, forged)


def test_public_query_late_callback_cannot_publish_changed_support():
    from test_authored_boundary_correspondence import fixture
    model, root, _ = fixture(cubic=True)
    before = to_dict(model)
    child = next(face for face in model.faces.values() if type(face.surface) is ExtrudedSurface)
    raw = child.surface._vector.copy()
    changed = []
    def callback(phase):
        if phase == 'native material/reference scope final check':
            object.__setattr__(child.surface, '_vector', raw+np.array((0.,0.,2**-40)))
            changed.append(True)
        return False
    try:
        with pytest.raises(GeometryError): query(model,[root],cancellation_check=callback)
        assert changed and to_dict(model) == before
    finally:
        object.__setattr__(child.surface, '_vector', raw)


def test_public_query_cancellation_and_error_identity():
    from test_authored_boundary_correspondence import fixture
    model, root, _ = fixture(cubic=True)
    sentinel = GeometryError('late support callback sentinel')
    def callback(phase):
        if phase == 'native material/reference scope final check': raise sentinel
        return False
    with pytest.raises(GeometryError) as caught: query(model,[root],cancellation_check=callback)
    assert caught.value is sentinel
    with pytest.raises(GeometryError, match='cancelled'):
        query(model,[root],cancellation_check=lambda phase: True)


def test_aggregate_proof_budget_is_not_renewed_for_polynomial_support():
    from anygeometry.cylinder_charts import _Proof, _Refusal
    from anygeometry.native_arc_parameter_maps import NativeArcParameterMapPolicy
    proof = _Proof(NativeArcParameterMapPolicy(4), None)
    with pytest.raises(_Refusal, match='budget'):
        prove(COONS_POLYNOMIAL, coons(), BEZIER_EXTRUSION, extrusion(), charge=proof.charge)
    assert proof.counts['interval_operations'] == 4


def test_explicit_sampled_coons_is_captured_but_not_reinterpreted_as_bernstein():
    model, face = bare_wall()
    top = translate(B,D)
    support = CoonsSurface(np.asarray(B), np.asarray((B[-1],top[-1]),dtype=float),
                           np.asarray(top,dtype=float), np.asarray((B[0],top[0]),dtype=float))
    model._faces[face] = replace(model.faces[face], surface=support)
    row, = capture_native_supports(model)
    assert row[1] == 'coons_sampled'
    with pytest.raises(GeometryError, match='unsupported'):
        prove(row[1],row[2],BEZIER_EXTRUSION,extrusion())
