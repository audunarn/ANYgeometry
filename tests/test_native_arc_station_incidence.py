"""Finite supplied stations against captured native Arc harmonics; no substitutes."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from importlib.metadata import version
import math
import os
import sys
from pathlib import Path

import pytest

from anygeometry import (GeometryError, GeometryModel, plan_intersections, apply_intersections,
    query_prepared_native_arc_parameter_maps,
    validate_prepared_native_arc_parameter_maps_binding as validate_maps,
    NativeArcStation, NativeArcStationIncidenceError,
    query_prepared_native_arc_station_incidence as query,
    validate_prepared_native_arc_station_incidence_binding as validate)
from anygeometry.entities import OrientedEdge
from anygeometry.native_arc_parameter_maps import _digest
from anygeometry.native_arc_station_incidence import NativeArcStationIncidencePolicy as ProofPolicy
from anygeometry.native_arc_station_incidence import _issued
from anygeometry.edge_subcurve_preimages import _record_edge_subcurve_unification, _edge_ancestry_definition
import anygeometry
import test_edge_subcurve_preimages as owner


def test_effective_source_runtime():
    assert Path(anygeometry.__file__).resolve() == Path(__file__).resolve().parents[1] / 'src/anygeometry/__init__.py'
    assert all(os.environ[x] == '1' for x in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    if os.environ.get('NATIVE_ARC_STATION_RUNTIME_EVIDENCE'):
        Path(os.environ['NATIVE_ARC_STATION_RUNTIME_EVIDENCE']).write_text(__import__('json').dumps({
            'executable': sys.executable, 'python_version': sys.version,
            'source_origin': str(Path(anygeometry.__file__).resolve()),
            'numpy_version': version('numpy'), 'pytest_version': version('pytest'),
            'threads': {x: os.environ[x] for x in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
        }, indent=2))


_CACHE = {}


def nice_fixture():
    """Public prepared model whose captured arc frame is exactly on the proof grid."""
    model = GeometryModel()
    a, b, c = model.add_points(((8., 0., 0.), (9., 1., 0.), (10., 0., 0.)))
    arc = model.add_arc(a, b, c)
    e = model.add_point(9., -1., 0.)
    face = model.add_face_from_loop((OrientedEdge(arc, True),
        OrientedEdge(model.add_line(c, e), True), OrientedEdge(model.add_line(e, a), True)))
    plan = plan_intersections(model, (face,), policy='connect')
    apply_intersections(model, plan, policy='connect')
    return model, arc


def reversed_alias_fixture():
    """Public prepared model with a reversed alias occurrence re-sealed onto split children."""
    if 'reversed' not in _CACHE:
        model, _, _ = owner.fixture()
        start, via, end = model.add_points(((8., 0., 0.), (9., 1., 0.), (10., 0., 0.)))
        canonical = model.add_arc(start, via, end)
        duplicate = model.add_arc(end, via, start)
        plan = plan_intersections(model, tuple(model.faces), policy='connect')
        draft = owner.capture(model, allow_seed=True)
        candidate = model.clone(preserve_identity=True)
        _record_edge_subcurve_unification(draft, canonical, (duplicate,), model=candidate)
        before = _edge_ancestry_definition(candidate, canonical)
        _, children = candidate.split_edge(canonical, .37)
        owner.record(draft, canonical, .37, children, 1e-9, model=candidate, parent_definition=before)
        owner.commit(model, candidate, draft, plan)
        _CACHE['reversed'] = (model, canonical, duplicate, children)
    return _CACHE['reversed']


def cylinder_fixture():
    """Actual public prepared fixture: structural cylinder split by a plate."""
    if 'cylinder' not in _CACHE:
        from anygeometry.generators.structural import cylinder
        model = cylinder(1., 1., circumferential_segments=3)
        roots = frozenset(model.edges)
        model.add_plate(model.add_points(((2., .5, -.5), (2., .5, 1.5),
                                          (-2., .5, 1.5), (-2., .5, -.5))))
        plan = plan_intersections(model, tuple(model.faces), policy='connect')
        apply_intersections(model, plan, policy='connect')
        _CACHE['cylinder'] = (model, roots)
    return _CACHE['cylinder']


def harmonic_point(frame, s):
    w = float(frame.sweep) * float(s)
    return tuple(float(x + y*math.cos(w) + z*math.sin(w))
        for x, y, z in zip(frame.center, frame.cosine, frame.sine))


def test_exact_simple_identity_map():
    model, arc = nice_fixture()
    maps = query_prepared_native_arc_parameter_maps(model)
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    binding = query(model, [station], maps=maps)
    record = binding.records[0]
    assert record.classification == 'exact'
    assert (record.current_bound, record.ancestor_bound) == (F(0), F(0))
    assert record.edge_id == arc and record.s == F(0) and record.point == (F(8), F(0), F(0))
    assert record.ancestor_edge_id == arc
    assert (record.ancestor_model_id, record.ancestor_revision) == (model.model_id, model.revision)
    assert record.ancestor_source_checksum == maps.source_checksum
    assert record.interval == (F(0), F(1))
    assert record.current_frame == maps.records[0].current_frame
    assert record.map_classification == 'exact'
    assert binding.stations == (station,)
    assert binding.model_id == model.model_id and binding.revision == model.revision
    assert binding.source_checksum == maps.source_checksum
    assert binding.ancestry_digest == maps.ancestry_digest
    assert binding.map_receipt_digest == _digest(maps)
    assert binding.maps is maps
    assert binding.tolerance == F(model.tolerance.length)
    assert binding.work_counts
    validate(model, binding)
    validate_maps(model, binding.maps)


def test_reversed_alias_selection_and_ambiguity():
    model, canonical, duplicate, children = reversed_alias_fixture()
    child = children[0]
    ambiguous = NativeArcStation(child, F(0), (8., 0., 0.))
    with pytest.raises(GeometryError, match='ambiguous'):
        query(model, [ambiguous])
    reversed_binding = query(model, [NativeArcStation(child, F(0), (8., 0., 0.),
        ancestor_edge_id=duplicate)])
    record = reversed_binding.records[0]
    assert record.ancestor_edge_id == duplicate
    assert record.interval[0] > record.interval[1]
    assert record.classification == 'bounded'
    assert F(0) < record.current_bound <= binding_tolerance(model)
    assert F(0) < record.ancestor_bound <= binding_tolerance(model)
    validate(model, reversed_binding)
    original = query(model, [NativeArcStation(child, F(0), (8., 0., 0.),
        ancestor_edge_id=canonical)])
    assert original.records[0].ancestor_edge_id == canonical
    assert original.records[0].interval[0] < original.records[0].interval[1]
    with pytest.raises(GeometryError, match='unavailable'):
        query(model, [NativeArcStation(child, F(0), (8., 0., 0.), ancestor_edge_id=99999)])


def binding_tolerance(model):
    return F(model.tolerance.length)


def test_split_map_batch_on_public_prepared_fixture():
    model, roots = cylinder_fixture()
    maps = query_prepared_native_arc_parameter_maps(model)
    stations = []
    for row in maps.records:
        stations.append(NativeArcStation(row.edge_id, F(0), harmonic_point(row.current_frame, 0)))
        stations.append(NativeArcStation(row.edge_id, F(1, 2), harmonic_point(row.current_frame, F(1, 2))))
    binding = query(model, stations)
    assert binding.stations == tuple(stations)
    assert len(binding.records) == len(stations)
    # Exact endpoints may have zero certified residual on either platform.
    # Keep the API's exact/bounded distinction tied to both independent bounds.
    assert all(record.classification == (
        'exact' if record.current_bound == record.ancestor_bound == 0 else 'bounded')
        for record in binding.records)
    assert any(record.classification == 'bounded' and record.edge_id not in roots
        for record in binding.records)
    assert all(0 <= record.current_bound <= binding.tolerance for record in binding.records)
    assert all(record.ancestor_bound is not None and 0 <= record.ancestor_bound <= binding.tolerance
        for record in binding.records)
    assert [record.edge_id for record in binding.records] == [s.edge_id for s in stations]
    assert any(record.edge_id not in roots for record in binding.records)
    assert binding.map_receipt_digest == _digest(maps)
    validate(model, binding)
    again = query(model, stations, maps=maps)
    assert again.records == binding.records


def test_direct_ancestor_versus_double_tolerance():
    model, arc = nice_fixture()
    tolerance = binding_tolerance(model)
    # Each bound separately receives one full tolerance; the two are never
    # added: both bounds sit at 0.75 tolerance (sum 1.5 tolerance) and pass.
    inside = query(model, [NativeArcStation(arc, F(0), (8. + float(3)*float(tolerance)/4, 0., 0.))])
    record = inside.records[0]
    assert record.classification == 'bounded'
    assert record.current_bound <= tolerance and record.ancestor_bound <= tolerance
    assert record.current_bound + record.ancestor_bound > tolerance
    # A bound at 1.5 tolerance is within the sum of two full tolerances and
    # still refuses: tolerances cannot be added.
    with pytest.raises(GeometryError, match='station distance exceeds'):
        query(model, [NativeArcStation(arc, F(0), (8. + float(3)*float(tolerance)/2, 0., 0.))])
    # Direct ancestor counterexample: the point lies exactly on the captured
    # current harmonic (bound zero) yet the direct ancestor bound alone refuses.
    split_model, roots = cylinder_fixture()
    maps = query_prepared_native_arc_parameter_maps(split_model)
    batch = query(split_model, [NativeArcStation(row.edge_id, F(0),
        harmonic_point(row.current_frame, 0))
        for row in maps.records if row.edge_id not in roots])
    exact = next(record for record in batch.records if record.current_bound == F(0))
    assert exact.ancestor_bound > F(0)
    row = next(r for r in maps.records if r.edge_id == exact.edge_id)
    with pytest.raises(GeometryError, match='ancestor distance exceeds'):
        query(split_model, [NativeArcStation(row.edge_id, F(0),
            harmonic_point(row.current_frame, 0))], tolerance=exact.ancestor_bound/2)
    with pytest.raises(GeometryError, match='ancestor distance exceeds'):
        query(split_model, [NativeArcStation(row.edge_id, F(0),
            harmonic_point(row.current_frame, 0))], tolerance=F(1, 2**40))


def test_offcurve_station_and_coordinate_substitution_refused():
    model, arc = nice_fixture()
    # Equality of saved and mesh XYZ coordinates is never proof: an off-curve
    # supplied point must refuse even if it equals some stored coordinate.
    with pytest.raises(GeometryError, match='station distance exceeds'):
        query(model, [NativeArcStation(arc, F(0), (100., 100., 100.))])
    with pytest.raises(GeometryError, match='station distance exceeds'):
        query(model, [NativeArcStation(arc, F(1, 2), (8., 0., 0.))])


def test_invalid_input_refused():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    for stations in (None, 1, 'x', (s for s in (station,)), [arc]):
        with pytest.raises(GeometryError):
            query(model, stations)
    for edge in (0, -1, True, 'x', 1.0):
        with pytest.raises(GeometryError, match='positive integer'):
            NativeArcStation(edge, F(0), (8., 0., 0.))
    for s in (-1, 1.5, float('nan'), float('inf'), -float('inf')):
        with pytest.raises(GeometryError, match='finite value in'):
            NativeArcStation(arc, s, (8., 0., 0.))
    for point in ((8., 0.), (8., 0., 0., 0.), (8., float('nan'), 0.), (8., 0., float('inf')), 8.):
        with pytest.raises(GeometryError, match='three finite coordinates'):
            NativeArcStation(arc, F(0), point)
    with pytest.raises(GeometryError, match='positive integer'):
        NativeArcStation(arc, F(0), (8., 0., 0.), ancestor_edge_id=0)
    with pytest.raises(GeometryError, match='no looser'):
        query(model, [station], tolerance=2*float(model.tolerance.length))
    for bad in (0., -1., float('nan'), float('inf')):
        with pytest.raises(GeometryError, match='tolerance'):
            query(model, [station], tolerance=bad)
    with pytest.raises(GeometryError, match='proof policy'):
        query(model, [station], policy=object())
    with pytest.raises(GeometryError, match='prepared native Arc parameter map'):
        query(model, [station], maps=object())
    with pytest.raises(GeometryError, match='unavailable'):
        query(model, [NativeArcStation(99999, F(0), (8., 0., 0.))])


def test_unqualified_map_refused(monkeypatch):
    model, arc = nice_fixture()
    maps = query_prepared_native_arc_parameter_maps(model)
    doctored = replace(maps, records=(replace(maps.records[0], classification='refused'),))
    import anygeometry.native_arc_station_incidence as station_module
    monkeypatch.setattr(station_module, 'deepcopy',
        lambda value: doctored if value is maps else deepcopy(value))
    with pytest.raises(GeometryError, match='unqualified'):
        query(model, [NativeArcStation(arc, F(0), (8., 0., 0.))], maps=maps)


def test_copy_forgery_wrong_model_and_mutation_refused():
    model, arc = nice_fixture()
    binding = query(model, [NativeArcStation(arc, F(0), (8., 0., 0.))])
    validate(model, binding)
    with pytest.raises(GeometryError, match='not issued'):
        validate(model, deepcopy(binding))
    with pytest.raises(GeometryError, match='not issued'):
        validate(model, replace(binding))
    other = GeometryModel()
    with pytest.raises(GeometryError, match='not issued'):
        validate(other, binding)
    with pytest.raises(GeometryError, match='not issued'):
        query(other, [NativeArcStation(arc, F(0), (8., 0., 0.))], maps=binding.maps)
    records = binding.records
    object.__setattr__(binding, 'records', ())
    with pytest.raises(GeometryError, match='changed'):
        validate(model, binding)
    with pytest.raises(GeometryError, match='changed'):
        validate(model, binding, cancellation_check=lambda _: object.__setattr__(binding, 'records', records))
    object.__setattr__(binding, 'records', records)
    validate(model, binding)


def test_cancellation_is_atomic_and_marker_propagates():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    binding = query(model, [station])
    with pytest.raises(GeometryError, match='cancelled'):
        query(model, [station], cancellation_check=lambda _: True)
    with pytest.raises(GeometryError, match='cancelled'):
        validate(model, binding, cancellation_check=lambda _: True)
    marker = NativeArcStationIncidenceError('caller marker')
    def stop(_):
        raise marker
    with pytest.raises(NativeArcStationIncidenceError) as caught:
        query(model, [station], cancellation_check=stop)
    assert caught.value is marker
    with pytest.raises(NativeArcStationIncidenceError) as caught:
        validate(model, binding, cancellation_check=stop)
    assert caught.value is marker


def test_budget_exhaustion_has_no_partial_receipt():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(1, 2), harmonic_point(
        query_prepared_native_arc_parameter_maps(model).records[0].current_frame, F(1, 2)))
    with pytest.raises(GeometryError, match='budget'):
        query(model, [station], policy=ProofPolicy(max_interval_operations=1))
    assert not _issued.get(model)
    with pytest.raises(GeometryError, match='budget'):
        query(model, [station, station], policy=ProofPolicy(max_interval_operations=8))
    assert not _issued.get(model)


def test_mutation_during_proof_and_readonly_nonmutation():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    expected = query(model, [station])
    receipt = model._edge_subcurve_preimages_receipt
    rows = receipt[0].arc_records
    def callback(_):
        object.__setattr__(receipt[0], 'arc_records', ())
        object.__setattr__(receipt[0], 'arc_records', rows)
        return False
    actual = query(model, [station], cancellation_check=callback)
    assert actual.records == expected.records
    validate(model, actual, cancellation_check=callback)
    def forge(_):
        changed = replace(receipt[0], arc_records=())
        model._edge_subcurve_preimages_receipt = (changed, receipt[1], receipt[2])
        return False
    with pytest.raises(GeometryError, match='changed'):
        query(model, [station], cancellation_check=forge)
    model._edge_subcurve_preimages_receipt = receipt
    # The public query is read-only: nothing about the model moves.
    revision, checksum = model.revision, receipt[1]
    maps = query_prepared_native_arc_parameter_maps(model)
    maps_digest = _digest(maps)
    query(model, [station], maps=maps)
    validate(model, expected)
    assert model.revision == revision and model._edge_subcurve_preimages_receipt[1] == checksum
    assert model._edge_subcurve_preimages_receipt[0] is receipt[0]
    assert _digest(maps) == maps_digest


def test_stale_maps_and_stale_receipt_refused():
    model, arc = nice_fixture()
    maps = query_prepared_native_arc_parameter_maps(model)
    binding = query(model, [NativeArcStation(arc, F(0), (8., 0., 0.))], maps=maps)
    validate(model, binding)
    model.add_point(11., 11., 11.)
    with pytest.raises(GeometryError, match='stale|preparation'):
        query(model, [NativeArcStation(arc, F(0), (8., 0., 0.))], maps=maps)
    with pytest.raises(GeometryError, match='stale|preparation'):
        validate(model, binding)

def test_station_request_frozen_against_callback_mutation():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    maps = query_prepared_native_arc_parameter_maps(model)
    # Final-check adversary: mutate the caller station during the final
    # callback.  The proof certified the frozen snapshot; the changed original
    # request must refuse atomically instead of issuing inconsistent evidence.
    def final_forge(phase):
        if 'final' in phase:
            object.__setattr__(station, 'point', (F(100), F(100), F(100)))
        return False
    with pytest.raises(GeometryError, match='request changed'):
        query(model, [station], maps=maps, cancellation_check=final_forge)
    object.__setattr__(station, 'point', (F(8), F(0), F(0)))
    # Mid-arithmetic adversary: mutate s/XYZ/edge during the station proof
    # callbacks; only the detached snapshot is certified and the mutated
    # original request refuses after the final callback.
    def mid_forge(phase):
        object.__setattr__(station, 's', F(1, 2))
        object.__setattr__(station, 'point', (F(100), F(100), F(100)))
        object.__setattr__(station, 'edge_id', 99999)
        return False
    with pytest.raises(GeometryError, match='request changed'):
        query(model, [station], maps=maps, cancellation_check=mid_forge)
    object.__setattr__(station, 's', F(0))
    object.__setattr__(station, 'point', (F(8), F(0), F(0)))
    object.__setattr__(station, 'edge_id', arc)
    # A clean run stores the detached frozen snapshot, never caller objects.
    binding = query(model, [station], maps=maps)
    assert binding.stations == (station,)
    assert binding.stations[0] is not station
    record = binding.records[0]
    assert (record.s, record.point) == (F(0), (F(8), F(0), F(0)))
    validate(model, binding)
    # Later caller mutation cannot alter the detached receipt or its records.
    object.__setattr__(station, 'point', (F(100), F(100), F(100)))
    assert binding.stations[0].point == (F(8), F(0), F(0))
    assert binding.records[0].point == (F(8), F(0), F(0))
    validate(model, binding)


def test_map_query_work_is_charged_to_the_caller_budget():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    maps = query_prepared_native_arc_parameter_maps(model)
    map_ops = dict(maps.work_counts)['interval_operations']
    # A preexisting issued map receipt is historical: supplying it costs only
    # the station arithmetic within the same caller allowance.
    solo = query(model, [station], maps=maps)
    solo_ops = dict(solo.work_counts)['interval_operations']
    assert solo_ops > map_ops
    # The allowance that exactly covers the station proof alone cannot also
    # absorb the NEW map-query arithmetic: one aggregate caller policy, no
    # second default 200000 budget.  The refusal issues no partial receipt.
    before = len(_issued.get(model, {}))
    with pytest.raises(GeometryError, match='budget'):
        query(model, [station], policy=ProofPolicy(max_interval_operations=solo_ops))
    assert len(_issued.get(model, {})) == before
    # A maps=None batch charges the map query's actual work into the same
    # proof: the aggregate count is exactly map work plus station work.
    aggregate = query(model, [station],
        policy=ProofPolicy(max_interval_operations=map_ops + solo_ops))
    assert dict(aggregate.work_counts)['interval_operations'] == map_ops + solo_ops
    validate(model, aggregate)
    # A tiny allowance refuses inside the map query itself.
    with pytest.raises(GeometryError, match='budget'):
        query(model, [station], policy=ProofPolicy(max_interval_operations=1))


def test_current_only_mode_certifies_new_edge_without_ancestor():
    model, roots = cylinder_fixture()
    maps = query_prepared_native_arc_parameter_maps(model)
    row = next(row for row in maps.records if row.edge_id not in roots)
    station = NativeArcStation(row.edge_id, F(0), harmonic_point(row.current_frame, 0))
    # Current-only mode needs no map receipt and no ancestor: the binding is
    # the authenticated prepared current Arc definition of the named edge.
    binding = query(model, [station], preserve_ancestor=False)
    record = binding.records[0]
    assert record.scope == 'current_only'
    assert record.current_frame == row.current_frame
    assert record.current_bound <= binding.tolerance
    assert record.ancestor_bound is None and record.ancestor_frame is None
    assert record.interval is None and record.ancestor_edge_id is None
    assert record.ancestor_model_id is None and record.ancestor_revision is None
    assert record.ancestor_source_checksum is None
    assert binding.maps is None and binding.map_receipt_digest is None
    assert binding.preserve_ancestor is False
    validate(model, binding)
    # A supplied historical map receipt is guarded and reused without
    # re-execution; its classification does not affect the current-only claim.
    with_maps = query(model, [station], maps=maps, preserve_ancestor=False)
    assert with_maps.records[0].scope == 'current_only'
    assert with_maps.records[0].current_bound == record.current_bound
    assert with_maps.records[0].map_classification == row.classification
    assert with_maps.map_receipt_digest == _digest(maps)
    validate(model, with_maps)
    # Ancestor mode keeps its stronger contract for the same edge.
    ancestor_binding = query(model, [station])
    assert ancestor_binding.records[0].scope == 'ancestor'
    assert ancestor_binding.records[0].current_bound == record.current_bound
    assert ancestor_binding.records[0].ancestor_bound is not None


def test_current_only_accepts_unqualified_map_without_ancestor_claim(monkeypatch):
    model, arc = nice_fixture()
    maps = query_prepared_native_arc_parameter_maps(model)
    doctored = replace(maps, records=(replace(maps.records[0], classification='refused'),))
    import anygeometry.native_arc_station_incidence as station_module
    monkeypatch.setattr(station_module, 'deepcopy',
        lambda value: doctored if value is maps else deepcopy(value))
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    # Ancestor preservation still refuses the unqualified map.
    with pytest.raises(GeometryError, match='unqualified'):
        query(model, [station], maps=maps)
    # Current-only mode certifies the captured current harmonic anyway; the
    # record honestly reports the observed map classification.
    binding = query(model, [station], maps=maps, preserve_ancestor=False)
    record = binding.records[0]
    assert record.scope == 'current_only'
    assert record.map_classification == 'refused'
    assert record.classification == 'exact' and record.current_bound == F(0)
    validate(model, binding)


def test_current_only_input_rules_and_refusals():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    binding = query(model, [station], preserve_ancestor=False)
    assert binding.records[0].classification == 'exact'
    validate(model, binding)
    # Forged/copied current-only receipts are refused.
    with pytest.raises(GeometryError, match='not issued'):
        validate(model, replace(binding))
    # Malformed mode and contradictory ancestor selection are refused.
    with pytest.raises(GeometryError, match='preservation'):
        query(model, [station], preserve_ancestor=1)
    with pytest.raises(GeometryError, match='ancestor'):
        query(model, [NativeArcStation(arc, F(0), (8., 0., 0.),
            ancestor_edge_id=arc)], preserve_ancestor=False)
    # Unavailable edges, off-curve points and exhausted budgets refuse.
    with pytest.raises(GeometryError, match='unavailable'):
        query(model, [NativeArcStation(99999, F(0), (8., 0., 0.))], preserve_ancestor=False)
    with pytest.raises(GeometryError, match='station distance exceeds'):
        query(model, [NativeArcStation(arc, F(0), (100., 100., 100.))], preserve_ancestor=False)
    with pytest.raises(GeometryError, match='budget'):
        query(model, [station], preserve_ancestor=False,
            policy=ProofPolicy(max_interval_operations=1))
    # Stale current-only receipts refuse after the model changes.
    model.add_point(11., 11., 11.)
    with pytest.raises(GeometryError, match='stale|preparation'):
        validate(model, binding)


def test_current_only_definition_capture_and_mutation_refused():
    model, arc = nice_fixture()
    station = NativeArcStation(arc, F(0), (8., 0., 0.))
    receipt = model._edge_subcurve_preimages_receipt
    def forge(_):
        changed = replace(receipt[0], arc_records=())
        model._edge_subcurve_preimages_receipt = (changed, receipt[1], receipt[2])
        return False
    # The actual current Arc definitions are captured before the first
    # callback; a callback that forges them must refuse, atomically.
    with pytest.raises(GeometryError, match='changed'):
        query(model, [station], preserve_ancestor=False, cancellation_check=forge)
    model._edge_subcurve_preimages_receipt = receipt
    binding = query(model, [station], preserve_ancestor=False)
    with pytest.raises(GeometryError, match='changed'):
        validate(model, binding, cancellation_check=forge)
    model._edge_subcurve_preimages_receipt = receipt
    validate(model, binding)


@pytest.mark.parametrize('current_only', (False, True))
@pytest.mark.parametrize('edit', ('clear', 'reverse', 'append', 'replace'))
def test_original_request_container_mutation_refuses(current_only, edit):
    model, arc = nice_fixture()
    stations = [NativeArcStation(arc, 0, (8., 0., 0.)),
                NativeArcStation(arc, 1, (10., 0., 0.))]
    before = len(_issued.get(model, {}))
    def mutate(phase):
        if phase == 'native Arc station incidence final check':
            if edit == 'clear':
                stations.clear()
            elif edit == 'reverse':
                stations.reverse()
            elif edit == 'append':
                stations.append(stations[0])
            else:
                stations[0] = replace(stations[0])
        return False
    with pytest.raises(GeometryError, match='request changed'):
        query(model, stations, preserve_ancestor=not current_only,
              cancellation_check=mutate)
    assert len(_issued.get(model, {})) == before


def test_generated_plate_pipe_joint_arcs_current_only():
    from anygeometry.generators.structural import cylinder
    from anygeometry.curves import Arc
    from anygeometry.edge_subcurve_preimages import query_prepared_edge_subcurve_preimages
    model = cylinder(1., 1., circumferential_segments=4)
    model.add_plate(model.add_points(((-2., -2., .5), (2., -2., .5),
                                      (2., 2., .5), (-2., 2., .5))))
    apply_intersections(model, plan_intersections(model, tuple(model.faces), policy='connect'),
                        policy='connect')
    ancestry = query_prepared_edge_subcurve_preimages(model)
    inherited = {row.edge_id for row in ancestry.arc_records + ancestry.arc_alias_records}
    joints = [edge for edge in model.edges.values() if isinstance(edge.curve, Arc)
              and all(abs(model.vertices[v].position[2] - .5) < 1e-12
                      for v in (edge.start, edge.end, edge.curve.via_vertex))]
    assert len(joints) == 4
    assert not ({edge.id for edge in joints} & inherited)
    stations = [NativeArcStation(edge.id, 0, model.vertices[edge.start].position)
                for edge in joints]
    binding = query(model, stations, preserve_ancestor=False)
    assert len(binding.records) == 4 and len(binding.current_definitions) == 4
    assert all(row.scope == 'current_only' and row.ancestor_edge_id is None
               and row.current_bound <= binding.tolerance for row in binding.records)
    validate(model, binding)
    with pytest.raises(GeometryError, match='ancestry unavailable'):
        query(model, stations)
