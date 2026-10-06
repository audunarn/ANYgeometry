"""Independent source/oracle adversaries. Authored but NOT run in this slice.

No preparation or meshing is invoked by these tests. A future controlled owner
comparison run is still required; pure interval tests do not qualify that path.
"""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import pytest


@pytest.fixture
def oracle(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/general_intersections'))
    import mixed_bay_oracle
    return mixed_bay_oracle


def test_cubic_power_coefficients_are_independent_expansion(oracle):
    assert oracle._bernstein_power(oracle.WALL_CONTROLS) == (
        (F(0), F(3)), (F(0), F(6), F(-15), F(10)), (F(0),))


def test_raw_nonorthogonal_basis_is_not_an_orthonormal_shortcut(oracle):
    columns = ((F(2), F(0), F(0)), (F(1), F(3), F(0)), (F(1), F(2), F(4)))
    inverse = oracle._inverse_columns(columns)
    for i, row in enumerate(inverse):
        assert tuple(oracle._dot(row, column) for column in columns) == tuple(F(i == j) for j in range(3))
    # Exact carrier point with rational unit radial coordinates 3/5,4/5.
    q = (F(3, 5), F(4, 5), F(7, 11))
    point = tuple(sum(columns[j][i]*q[j] for j in range(3)) for i in range(3))
    assert tuple(oracle._dot(row, point) for row in inverse) == q
    assert q[0]**2+q[1]**2 == 1
    assert oracle._dot(columns[0], point) != q[0]
    with pytest.raises(oracle.OracleRefusal, match='singular'):
        oracle._inverse_columns((columns[0], columns[0], columns[2]))


def test_section_coefficients_equal_raw_inverse_substitution(oracle):
    columns = ((F(2), F(0), F(0)), (F(1), F(3), F(0)), (F(1), F(2), F(4)))
    carrier = oracle.Carrier(3, (F(1), F(-2), F(3)), columns,
        oracle._inverse_columns(columns), F(0), F(1))
    base = oracle._bernstein_power(oracle.WALL_CONTROLS)
    (a, b, c), (z, k) = oracle._section_coefficients(carrier, base, oracle.WALL_VECTOR)
    for u, v in ((F(0), F(0)), (F(1, 3), F(2, 7)), (F(1), F(1))):
        # Direct independent cubic formula, not a curve evaluator or sampler
        # used to assert completeness. These are arithmetic discriminators.
        p = (3*u+v/4, 6*u-15*u*u+10*u**3, 3*v/2)
        q = tuple(sum(row[j]*(p[j]-carrier.origin[j]) for j in range(3))
                  for row in carrier.inverse_rows)
        assert oracle._value(a, u)*v*v+oracle._value(b, u)*v+oracle._value(c, u) == q[0]**2+q[1]**2-1
        assert oracle._value(z, u)+k*v == q[2]


def test_resultant_keeps_exact_shared_root_and_degenerate_case(oracle):
    # s^2-t=0 and s-1/2=0 meet exactly at t=1/4.
    coefficients = ((F(1),), (F(0),), (F(0), F(-1)))
    polynomial = oracle._linear_boundary_resultant(coefficients, (F(-1, 2),), F(1))
    assert polynomial == (F(1, 4), F(-1))
    assert oracle._value(polynomial, F(1, 4)) == 0
    zero = oracle._linear_boundary_resultant(coefficients, (F(0),), F(0))
    event = oracle._event('test', (1, 2), 'coincident', zero, (F(0), F(1)), None)
    assert event.coincidence_unresolved and not event.roots


def test_sturm_events_retain_tangency_endpoints_and_distinct_near_roots(oracle):
    # t*(t-1/2)^2*(t-1) includes a tangency, not a sign-change oracle.
    p = oracle._mul(oracle._mul((F(0), F(1)), (F(-1), F(1))),
                    oracle._mul((F(-1, 2), F(1)), (F(-1, 2), F(1))))
    event = oracle._event('test', (1, 2), 'roots', p, (F(0), F(1)), None)
    assert len(event.roots) == 3
    assert event.roots[0] == (F(0), F(0))
    assert event.roots[-1] == (F(1), F(1))
    assert event.roots[1][0] <= F(1, 2) <= event.roots[1][1]
    p = oracle._mul((F(-1, 2), F(1)), (-(F(1, 2)+F(1, 2**48)), F(1)))
    near = oracle._event('test', (1, 2), 'near', p, (F(0), F(1)), None)
    assert len(near.roots) == 2


def requirement(oracle):
    return oracle.RequiredInterval('test', 4, (1, 2),
        ((F(0), F(0), F(0)), (F(1), F(0), F(0))), 10, 11)


def observation(oracle, interval, edge=20, owners=(1, 2)):
    return oracle.IntervalObservation(4, edge, owners, tuple(map(F, interval)), F(0))


def test_reversed_exact_interval_tiling_remains_partial(oracle):
    rows = (observation(oracle, (F(1, 3), 0)), observation(oracle, (1, F(1, 3)), 21))
    result = oracle.compare_interval_observations((requirement(oracle),), rows)
    assert result.status == 'partial' and result.covered_source_edges == (4,)
    assert not result.whole_joint_completeness and not result.material_coverage
    with pytest.raises(oracle.OracleRefusal, match='cannot qualify'):
        result.require_complete()


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'gap', 'overlap', 'wrong-owner', 'wrong-source', 'outside'])
def test_interval_omission_duplicate_gap_and_wrong_ownership_refuse(oracle, case):
    rows = [observation(oracle, (0, 1))]
    if case == 'missing': rows = []
    elif case == 'duplicate': rows *= 2
    elif case == 'gap': rows = [observation(oracle, (0, F(1, 2))), observation(oracle, (F(1, 2)+F(1, 2**60), 1), 21)]
    elif case == 'overlap': rows = [observation(oracle, (0, F(1, 2))), observation(oracle, (F(1, 2)-F(1, 2**60), 1), 21)]
    elif case == 'wrong-owner': rows = [observation(oracle, (0, 1), owners=(1, 3))]
    elif case == 'wrong-source': rows = [replace(rows[0], source_edge_id=9)]
    elif case == 'outside': rows = [observation(oracle, (0, 2))]
    with pytest.raises(oracle.OracleMismatch):
        oracle.compare_interval_observations((requirement(oracle),), rows)


def test_empty_or_duplicate_expectations_do_not_prove_anything(oracle):
    with pytest.raises(oracle.OracleRefusal):
        oracle.compare_interval_observations((), ())
    row = requirement(oracle)
    with pytest.raises(oracle.OracleRefusal):
        oracle.compare_interval_observations((row, row), ())


def test_fixed_authored_fixture_inventory_never_claims_clipped_branch_completeness(oracle):
    from large_connected_fixtures import connected_mixed
    from anygeometry import to_dict
    fixture = connected_mixed(10, start_bay=0)
    before = to_dict(fixture.model)
    result = oracle.build_authored_inventory(fixture)
    assert len(result.required_intervals) == 9
    assert sum(row.family == 'floor/bottom-cubic' for row in result.required_intervals) == 1
    assert sum(row.family == 'authored-pipe-seam' for row in result.required_intervals) == 8
    assert len(result.carrier_sections) == 16 and len(result.literal_seam_events) == 16
    assert all(row.qualified_material_intervals is None for row in result.carrier_sections)
    assert result.pending and not result.whole_joint_completeness
    assert to_dict(fixture.model) == before


def test_inventory_cancel_identity_and_translated_fixture_refusal(oracle):
    from large_connected_fixtures import connected_mixed
    from anygeometry import to_dict
    fixture = connected_mixed(10)
    before = to_dict(fixture.model)
    sentinel = RuntimeError('cancel sentinel')
    def fail(): raise sentinel
    with pytest.raises(RuntimeError) as caught:
        oracle.build_authored_inventory(fixture, cancellation_check=fail)
    assert caught.value is sentinel and to_dict(fixture.model) == before
    with pytest.raises(oracle.OracleRefusal):
        oracle.build_authored_inventory(connected_mixed(10, start_bay=60))


def test_tampered_expected_inventory_refuses_before_prepared_lookup(oracle, monkeypatch):
    from large_connected_fixtures import connected_mixed
    fixture = connected_mixed(10)
    inventory = oracle.build_authored_inventory(fixture)
    forged = replace(inventory, required_intervals=inventory.required_intervals[:-1])
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject source expectations before prepared query')
    monkeypatch.setattr(oracle, 'query_prepared_edge_subcurve_preimages', forbidden)
    with pytest.raises(oracle.OracleRefusal, match='altered'):
        oracle.compare_prepared(forged, fixture.model, fixture.model)


def _comparison_routing_fixture(oracle, monkeypatch, *, bottom_on_floor):
    """Stub only public proof acquisition, to discriminate ownership routing.

    The manufactured document is not a prepared geometry fixture and supplies no
    owner acceptance evidence. Real query validators require a separate run.
    """
    from copy import deepcopy
    from types import SimpleNamespace as NS
    from anygeometry import EntityRef
    from large_connected_fixtures import connected_mixed
    fixture = connected_mixed(10)
    authored = fixture.model
    inventory = oracle.build_authored_inventory(fixture)
    original_to_dict = oracle.to_dict
    document = deepcopy(original_to_dict(authored))
    bottom = inventory.required_intervals[0]
    wall = next(f['id'] for f in document['faces'] if f['surface'] == {'type': 'coons'})
    floor = next(i for i in bottom.owners if i != wall)
    if bottom_on_floor:
        next(f for f in document['faces'] if f['id'] == floor)['loop'].append([bottom.source_edge_id, True])
    prepared = NS(model_id=authored.model_id, history={floor: wall})
    prepared.resolve_ref = lambda ref: (EntityRef('face', prepared.history.get(ref.id, ref.id)),)
    face_binding = NS(authored_model_id=authored.model_id, authored_revision=authored.revision,
        authored_checksum=inventory.source_checksum,
        face_descendants=tuple((f['id'], (f['id'],)) for f in document['faces']))
    rows = []
    pack = lambda value: (value.numerator, value.denominator)
    for row in inventory.required_intervals:
        definition = NS(edge_id=row.source_edge_id, start=row.start_vertex, end=row.end_vertex,
            controls=tuple(tuple(pack(x) for x in point) for point in row.controls))
        ancestor = NS(model_id=authored.model_id, revision=authored.revision,
            source_checksum=inventory.source_checksum, definition=definition)
        rows.append(NS(edge_id=row.source_edge_id, ancestor=ancestor,
                       interval=((0, 1), (1, 1)), squared_distance_bound=(0, 1)))
    monkeypatch.setattr(oracle, 'to_dict', lambda model: deepcopy(document) if model is prepared else original_to_dict(model))
    monkeypatch.setattr(oracle, 'query_prepared_edge_subcurve_preimages', lambda model: NS(records=tuple(rows), alias_records=()))
    monkeypatch.setattr(oracle, 'query_prepared_face_preimages', lambda model: face_binding)
    validated = []
    def validate_faces(model, binding):
        assert model is prepared and binding == face_binding
        validated.append(deepcopy(prepared.history))
    def validate_edges(model, binding, *, cancellation_check=None):
        oracle._check(cancellation_check)
    monkeypatch.setattr(oracle, 'validate_prepared_face_preimages_binding', validate_faces)
    monkeypatch.setattr(oracle, 'validate_prepared_edge_subcurve_preimages_binding', validate_edges)
    return inventory, authored, prepared, validated, floor, wall


def test_retargeted_history_cannot_supply_missing_common_owner(oracle, monkeypatch):
    inventory, authored, prepared, _, _, _ = _comparison_routing_fixture(
        oracle, monkeypatch, bottom_on_floor=False)
    # Raw resolve_ref maps both owners to the wall, falsely supplying the whole
    # bottom edge. The authenticated face map retains the floor and refuses it.
    with pytest.raises(oracle.OracleMismatch, match='missing required'):
        oracle.compare_prepared(inventory, authored, prepared)


def test_callback_history_edits_cannot_retarget_pinned_face_ownership(oracle, monkeypatch):
    inventory, authored, prepared, validated, floor, wall = _comparison_routing_fixture(
        oracle, monkeypatch, bottom_on_floor=True)
    def mutate_untrusted_history():
        prepared.history[wall] = floor
        return False
    result = oracle.compare_prepared(inventory, authored, prepared,
                                     cancellation_check=mutate_untrusted_history)
    assert len(result.covered_source_edges) == 9
    assert validated and validated[-1][wall] == floor
    assert not result.whole_joint_completeness
