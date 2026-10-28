"""Owner resolution signal from a certified fixed-FE error lower witness."""
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
import json

import pytest

from anygeometry import (GeometryError, SourceCellRefinementRequired, CurvilinearMaterialCell,
    validate_prepared_material_curvilinear_cells, validate_prepared_material_curvilinear_cells_binding)
from anygeometry.nurbs import nurbs_from_record
from anygeometry.regular_source_reference import SourceNode, SourceSide
from anygeometry._source_cell_point_error import fan_point_error_lower_squared
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work
from anygeometry.serialization import to_dict
from test_source_cell_certification import cropped_paraboloid_triangles
from test_prepared_material_curvilinear import square_cells
from test_structural_fan_bounds import unpack


def captured84():
    data = json.loads((Path(__file__).parent/'data'/'principal_straight_fan84.json').read_text())
    assert data['side']['curve'] is None and not data['side']['pieces']
    support = nurbs_from_record(unpack(data['support']['record']))
    nodes = tuple(SourceNode(*unpack(value)) for value in data['nodes'])
    row = data['side']
    side = SourceSide(nodes[row['start']], nodes[row['end']])
    args = support, side, unpack(data['centre_uv']), unpack(data['xyz_side']), unpack(data['centre_xyz'])
    return args, unpack(data['kwargs']['surface_error'])


def test_actual_fan84_has_strict_certified_error_lower_witness():
    args, limit = captured84()
    lower = fan_point_error_lower_squared(*args, check=lambda: None)
    assert lower > limit*limit
    support, side, centre, xyz, target = args
    # Independent exact source and fixed-FE values at the two selected points.
    uv = tuple((a[0]+b[0])/2 for a, b in zip(side.start.box, side.end.box))
    mid = tuple((Q(a)+Q(b))/2 for a, b in zip(*xyz))
    exact = max(sum((Q(x)-Q(y))**2 for x, y in zip(support.exact_point(*p), fe))
                for p, fe in ((uv, mid), (centre, target)))
    assert 0 < lower <= exact


@pytest.mark.parametrize('quad', (False, True))
def test_genuine_public_whole_cells_raise_typed_resolution_signal(quad):
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.)
    if quad:
        uv = cells[0].uv+cells[1].uv[-1:]
        xyz = cells[0].xyz+cells[1].xyz[-1:]
        cells = (CurvilinearMaterialCell(cells[0].face, (0, 1, 2, 3), uv, xyz,
                                       tuple((i, i, True) for i in range(4))),)
    before = to_dict(model)
    with pytest.raises(SourceCellRefinementRequired) as caught:
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    error = caught.value
    assert isinstance(error, GeometryError)
    assert error.face_id == cells[0].face.id and error.node_ids in tuple(c.node_ids for c in cells)
    assert type(error.error_limit) is Q and type(error.lower_error_squared) is Q
    assert error.error_limit == Q(.0025) and error.lower_error_squared > error.error_limit**2
    assert error.requested_error == Q(.0025)
    assert to_dict(model) == before


@pytest.mark.parametrize('quad', (False, True))
def test_nonviolating_public_cells_keep_receipt_semantics(quad):
    model, regions, chords, cells = square_cells(quad)
    result = validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    validate_prepared_material_curvilinear_cells_binding(model, regions, result)


def test_unknown_lower_bound_family_has_no_resolution_authority():
    args, _ = captured84()
    assert fan_point_error_lower_squared(args[0], object(), *args[2:], check=lambda: None) is None


def test_node_precision_refusal_precedes_coarse_cell_resolution_signal():
    model, regions, chords, old = cropped_paraboloid_triangles(.25, .75, 1.)
    uv = old[0].uv+old[1].uv[-1:]
    xyz = old[0].xyz+old[1].xyz[-1:]
    # Four genuine triangles, complete boundary and shared interior node;
    # only the interior node's literal target violates geometry precision.
    cells = tuple(CurvilinearMaterialCell(old[0].face, (i, (i+1)%4, 4),
        (uv[i], uv[(i+1)%4], (.5, .5)), (xyz[i], xyz[(i+1)%4], (0., 0., .01)),
        ((0, i, True),)) for i in range(4))
    with pytest.raises(GeometryError, match='node target') as caught:
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    assert not isinstance(caught.value, SourceCellRefinementRequired)


def test_transport_hard_refusal_precedes_any_point_witness(monkeypatch):
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.)
    monkeypatch.setattr('anygeometry.source_cell_complex.parameter_transport_bound', lambda side: Q(1))
    def forbidden(*args, **kwargs): raise AssertionError('precision/transport gate was skipped')
    monkeypatch.setattr('anygeometry._source_cell_point_error.fan_point_error_lower_squared', forbidden)
    with pytest.raises(GeometryError, match='transport') as caught:
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    assert not isinstance(caught.value, SourceCellRefinementRequired)


@pytest.mark.parametrize('kind', ('stale_chord', 'bad_connectivity', 'cancel', 'callback_mutation'))
def test_existing_hard_refusals_never_become_resolution_requests(kind):
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.)
    callback = None
    old = chords.chord_error
    if kind == 'stale_chord': object.__setattr__(chords, 'chord_error', old+1.)
    if kind == 'bad_connectivity': cells = (replace(cells[0], node_ids=(0, 0, 2)), cells[1])
    if kind == 'cancel': callback = lambda phase: phase == 'curvilinear source-cell qualification'
    if kind == 'callback_mutation':
        def callback(phase):
            if phase == 'curvilinear source-cell qualification': object.__setattr__(chords, 'chord_error', old+1.)
    try:
        with pytest.raises(GeometryError) as caught:
            validate_prepared_material_curvilinear_cells(model, regions, chords, cells,
                surface_error=.0025, cancellation_check=callback)
        assert not isinstance(caught.value, SourceCellRefinementRequired)
    finally: object.__setattr__(chords, 'chord_error', old)


def test_unknown_general_certificate_error_is_not_reclassified(monkeypatch):
    model, regions, chords, cells = square_cells(False)
    original = GeometryError('indeterminate geometry certificate')
    def fail(*args, **kwargs): raise original
    monkeypatch.setattr('anygeometry.source_cell_complex.certify_fan_subtriangle', fail)
    with pytest.raises(GeometryError) as caught:
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    assert caught.value is original


def test_native_prepared_poison_is_hard_refusal_not_resolution():
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.)
    curve = next(iter(model.edges.values())).curve.pcurve
    original = curve._prepared
    other = replace(curve, weights=(1., 2.))
    object.__setattr__(curve, '_prepared', other._prepared)
    try:
        with pytest.raises(GeometryError) as caught:
            validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
        assert not isinstance(caught.value, SourceCellRefinementRequired)
    finally: object.__setattr__(curve, '_prepared', original)


def test_incomplete_material_coverage_is_hard_refusal_not_resolution():
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.)
    with pytest.raises(GeometryError) as caught:
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells[:1], surface_error=.0025)
    assert not isinstance(caught.value, SourceCellRefinementRequired)


def test_lower_point_work_and_check_exceptions_propagate():
    args, _ = captured84()
    token = CURRENT_WORK.set(Work(0, 12))
    try:
        with pytest.raises(GeometryError): fan_point_error_lower_squared(*args, check=lambda: None)
    finally: CURRENT_WORK.reset(token)
    original = GeometryError('cancelled lower-point check')
    def fail(): raise original
    with pytest.raises(GeometryError) as caught: fan_point_error_lower_squared(*args, check=fail)
    assert caught.value is original


def test_public_exception_requires_strict_fraction_lower_bound():
    error = SourceCellRefinementRequired(7, (1, 2, 3), Q(1, 4), Q(1, 8))
    assert error.error_limit == Q(1, 4) and error.lower_error_squared == Q(1, 8)
    assert error.requested_error == error.error_limit
    with pytest.raises(GeometryError): SourceCellRefinementRequired(7, (1, 2, 3), Q(1, 4), Q(1, 16))


def test_positive_transport_signal_retains_full_requested_error(monkeypatch):
    model, regions, chords, cells = cropped_paraboloid_triangles(.25, .75, 1.)
    transport = Q(1, 10**12)
    assert 0 < transport <= Q(regions.regions[0].sources[0].world_tolerance)
    monkeypatch.setattr('anygeometry.source_cell_complex.parameter_transport_bound', lambda side: transport)
    with pytest.raises(SourceCellRefinementRequired) as caught:
        validate_prepared_material_curvilinear_cells(model, regions, chords, cells, surface_error=.0025)
    error = caught.value
    # Independent allocation identity: no rounding or tolerance replacement.
    assert error.requested_error == Q(.0025)
    assert error.error_limit == Q(.0025)-transport > 0
    assert error.requested_error-error.error_limit == transport
    assert error.lower_error_squared > error.error_limit**2


def test_refinement_context_cannot_claim_request_smaller_than_allocation():
    with pytest.raises(GeometryError):
        SourceCellRefinementRequired(7, (1, 2, 3), Q(1, 4), Q(1, 8), requested_error=Q(1, 8))
    with pytest.raises(GeometryError):
        SourceCellRefinementRequired(7, (1, 2, 3), Q(1, 4), Q(1, 8), requested_error=float('inf'))
