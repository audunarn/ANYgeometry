"""Shared-boundary alias provenance: capture, per-root selection, and refusals.

Small fixtures only; no mesher, mixed-model or large-strip execution.
"""
from dataclasses import replace
from fractions import Fraction as F

import pytest

from anygeometry import (GeometryError, GeometryModel, apply_intersections,
    plan_intersections, to_dict)
from anygeometry.authored_boundary_correspondence import (
    query_prepared_authored_boundary_correspondence as correspondence,
    validate_prepared_authored_boundary_correspondence_binding as validate_correspondence)
from anygeometry.authored_boundary_stations import (
    query_prepared_authored_boundary_stations as stations,
    validate_prepared_authored_boundary_station_coordinates as validate_stations)
from anygeometry.authored_constraint_scope import query_prepared_authored_constraint_scope
from anygeometry.definition_binding import definition_checksum
from anygeometry.edge_subcurve_preimages import (
    _capture_edge_subcurve_preimages as capture,
    _edge_subcurve_definition as definition,
    _finalize_edge_subcurve_preimages as finalize,
    _pack,
    _record_edge_subcurve_unification as unify,
    _rebind_edge_subcurve_incidence as rebind,
    _reseal_occurrence,
    _residual,
    _retained_incidence,
    _seal,
    _unpack,
    query_prepared_edge_subcurve_preimages as query,
    validate_prepared_edge_subcurve_preimages_binding as validate)
from anygeometry.batch_intersections import clone_prepared_geometry


def authored_pair(cutter=False):
    """Two adjacent plates sharing one exact boundary; optional crossing cutter."""
    model = GeometryModel()
    first = model.add_plate(model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    second = model.add_plate(model.add_points(((1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0))))
    positions = {}
    for face in (first, second):
        for use in model.faces[face].loop:
            entity = model.edges[use.edge]
            ends = tuple(sorted((tuple(model.vertex_position(entity.start)),
                                 tuple(model.vertex_position(entity.end)))))
            positions.setdefault(ends, []).append(use.edge)
    roots = tuple(next(rows for rows in positions.values() if len(rows) == 2))
    if cutter:
        model.add_plate(model.add_points(((0.5, 0.5, -1), (1.5, 0.5, -1),
                                          (1.5, 0.5, 1), (0.5, 0.5, 1))))
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    return model, first, second, plan, roots


def adjacent(cutter=False):
    model, first, second, plan, roots = authored_pair(cutter)
    apply_intersections(model, plan, policy='connect')
    return model, first, second, plan, roots


def crossing():
    model = GeometryModel()
    for points in (((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
                   ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))):
        model.add_plate(model.add_points(points))
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    return model


def shared_edge(model):
    uses = {}
    for face in model.faces.values():
        for use in face.loop:
            uses.setdefault(use.edge, []).append(face.id)
    return next(edge for edge, rows in sorted(uses.items()) if len(rows) == 2)


def test_unified_shared_boundary_seals_both_roots_occurrences():
    model, first, second, plan, roots = adjacent()
    binding = query(model)
    shared = shared_edge(model)
    assert roots[0] != roots[1]
    assert len(binding.alias_records) == 1
    alias = binding.alias_records[0]
    assert alias.edge_id == shared
    assert alias.ancestor.definition.edge_id == roots[1]
    assert alias.interval == ((1, 1), (0, 1))  # exact reversed re-seal
    assert alias.tolerance is None and alias.squared_distance_bound == (0, 1)
    primary = next(row for row in binding.records if row.edge_id == shared)
    assert primary.ancestor.definition.edge_id == roots[0]
    assert len({row.edge_id for row in binding.records}) == len(binding.records)
    validate(model, binding)
    definition_checksum(binding)


def test_old_primary_ancestry_behavior_without_unification():
    model = crossing()
    binding = query(model)
    assert binding.alias_records == ()
    assert binding.records and binding.unavailable_edge_ids
    with pytest.raises(GeometryError, match='unavailable'):
        query(model, edge_ids=binding.unavailable_edge_ids)
    validate(model, binding)


def test_correspondence_selects_the_requested_root_occurrence():
    model, first, second, plan, roots = adjacent()
    shared = shared_edge(model)
    first_result = correspondence(model, first)
    second_result = correspondence(model, second)
    exterior = lambda result: {edge for loop in result.exterior_loops
                               for _, _, edges in loop for edge in edges}
    assert shared in exterior(first_result) and shared in exterior(second_result)
    validate_correspondence(model, first_result)
    validate_correspondence(model, second_result)


def test_stations_use_the_requested_root_alias_interval_and_orientation():
    model, first, second, plan, roots = adjacent()
    shared = shared_edge(model)
    first_result = correspondence(model, first)
    second_result = correspondence(model, second)
    values = (F(0), F(1, 3), F(1))
    first_stations = stations(model, first_result, shared, values)
    second_stations = stations(model, second_result, shared, values)
    assert first_stations.authored_parameters == ((0, 1), (1, 3), (1, 1))
    assert second_stations.authored_parameters == ((1, 1), (2, 3), (0, 1))
    assert first_stations.authored_points == second_stations.authored_points
    assert first_stations.current_polynomial_points == second_stations.current_polynomial_points
    assert first_stations.authored_uv != second_stations.authored_uv
    for result in (first_stations, second_stations):
        coordinates = [[float(F(*point)) for point in row]
                        for row in result.current_polynomial_points]
        validate_stations(model, result, coordinates)


def test_full_public_scope_query_on_all_authored_roots():
    model, first, second, plan, roots = adjacent()
    scope = query_prepared_authored_constraint_scope(model, (first, second))
    assert scope.selected_root_ids == (first, second)
    assert scope.outside_root_ids == ()
    trace = next(row for row in scope.inventory['traces']
                 if row['current_edge_id'] == shared_edge(model))
    assert trace['adjacent_authored_root_ids'] == [first, second]


def test_subsequent_cuts_split_carried_aliases():
    model, first, second, plan, roots = adjacent(cutter=True)
    binding = query(model)
    primary = {row.edge_id: row for row in binding.records
               if row.ancestor.definition.edge_id == roots[0]}
    aliases = {row.edge_id: row for row in binding.alias_records
               if row.ancestor.definition.edge_id == roots[1]}
    assert len(primary) == 2 and set(primary) == set(aliases)
    segments = sorted((tuple(F(*x) for x in primary[edge].interval) for edge in primary),
                      key=lambda interval: interval[0])
    assert segments[0][0] == 0 and segments[0][1] == segments[1][0] and segments[1][1] == 1
    reversed_segments = sorted((tuple(F(*x) for x in aliases[edge].interval) for edge in aliases),
                               key=lambda interval: -interval[0])
    assert reversed_segments[0][0] == 1 and reversed_segments[0][1] == reversed_segments[1][0] \
        and reversed_segments[1][1] == 0
    first_result = correspondence(model, first)
    second_result = correspondence(model, second)
    children = tuple(sorted(primary))
    first_row = next(row for loop in first_result.exterior_loops for row in loop
                    if row[0] == roots[0])
    second_row = next(row for loop in second_result.exterior_loops for row in loop
                     if row[0] == roots[1])
    assert tuple(first_row[2]) == children
    assert tuple(second_row[2]) == tuple(reversed(children))
    validate_correspondence(model, first_result)
    validate_correspondence(model, second_result)
    for child in children:
        first_stations = stations(model, first_result, child, (F(1, 2),))
        second_stations = stations(model, second_result, child, (F(1, 2),))
        assert first_stations.authored_points == second_stations.authored_points
        validate_stations(model, first_stations,
                          [[float(F(*point)) for point in row]
                           for row in first_stations.current_polynomial_points])
        validate_stations(model, second_stations,
                          [[float(F(*point)) for point in row]
                           for row in second_stations.current_polynomial_points])


def test_repreparation_carries_aliases_idempotently():
    model, first, second, plan, roots = adjacent()
    binding = query(model)
    assert apply_intersections(model, plan, policy='connect').reused
    assert query(model) == binding
    fresh = plan_intersections(model, tuple(model.faces), policy='connect')
    assert apply_intersections(model, fresh, policy='connect').reused
    assert query(model) == binding
    assert query(model).alias_records == binding.alias_records


def test_stale_receipt_refuses_queries_and_validation():
    model, first, second, plan, roots = adjacent()
    binding = query(model)
    vertex = model.edges[shared_edge(model)].start
    model.move_point(vertex, 1., .5, 0.)
    with pytest.raises(GeometryError, match='provenance'):
        query(model)
    with pytest.raises(GeometryError, match='provenance'):
        query(model, expected_revision=binding.revision)
    with pytest.raises(GeometryError, match='provenance'):
        validate(model, binding)


def test_forged_alias_bindings_refuse():
    model, first, second, plan, roots = adjacent()
    binding = query(model)
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate(model, replace(binding, alias_records=()))
    forged = replace(binding.alias_records[0], interval=((0, 1), (1, 1)))
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate(model, replace(binding, alias_records=(forged,)))
    with pytest.raises(GeometryError, match='needs a PreparedEdgeSubcurvePreimages binding'):
        validate(model, binding.alias_records[0])


def test_unification_refusal_rolls_back_the_whole_batch(monkeypatch):
    import anygeometry.edge_subcurve_preimages as provenance
    model, first, second, plan, roots = authored_pair()
    original = to_dict(model)
    def refuse(*args, **kwargs):
        raise GeometryError('unification refused')
    monkeypatch.setattr(provenance, '_record_edge_subcurve_unification', refuse)
    with pytest.raises(GeometryError, match='unification refused'):
        apply_intersections(model, plan, policy='connect')
    assert to_dict(model) == original
    assert not hasattr(model, '_edge_subcurve_preimages_receipt')


def merge_shared_vertices(candidate, draft, canonical, duplicate):
    """Producer sequence: merge each coincident duplicate endpoint into the
    canonical edge's vertex, re-binding incidence before the canonical reuse."""
    from anygeometry.intersections import _merge_vertex
    canonical_entity = candidate.edges[canonical]
    duplicate_entity = candidate.edges[duplicate]
    ends = (canonical_entity.start, canonical_entity.end)
    prior = {edge: definition(candidate, edge) for edge in
             set(candidate.edges_using_vertex(duplicate_entity.start))
             | set(candidate.edges_using_vertex(duplicate_entity.end))}
    for old in (duplicate_entity.start, duplicate_entity.end):
        new = next(vertex for vertex in ends
                   if tuple(candidate.vertex_position(old)) == tuple(candidate.vertex_position(vertex)))
        _merge_vertex(candidate, old, new)
    rebind(draft, prior, model=candidate)


def test_unification_hook_seals_only_consistent_definitions():
    model, first, second, plan, roots = authored_pair()
    canonical, duplicate = roots
    draft = capture(model, allow_seed=True)
    candidate = model.clone(preserve_identity=True)
    with candidate.transaction():
        merge_shared_vertices(candidate, draft, canonical, duplicate)
    unify(draft, canonical, (duplicate,), model=candidate)
    rows = draft.aliases.get(canonical)
    assert rows and len(rows) == 1
    assert rows[0].ancestor.definition.edge_id == duplicate
    assert rows[0].interval == ((1, 1), (0, 1))
    assert rows[0].current_definition == definition(candidate, canonical)
    unify(draft, canonical, (duplicate,), model=candidate)
    assert len(draft.aliases[canonical]) == 1
    with pytest.raises(GeometryError, match='another model'):
        unify(draft, canonical, (duplicate,), model=GeometryModel())
    # A changed duplicate definition (no recorded split) disappears.
    changed = capture(model, allow_seed=True)
    changed_candidate = model.clone(preserve_identity=True)
    changed.records[duplicate] = replace(changed.records[duplicate],
        current_definition=replace(changed.records[duplicate].current_definition, checksum='forged'))
    unify(changed, canonical, (duplicate,), model=changed_candidate)
    assert canonical not in changed.aliases
    # A changed canonical definition refuses.
    broken = capture(model, allow_seed=True)
    broken_candidate = model.clone(preserve_identity=True)
    broken.records[canonical] = replace(broken.records[canonical],
        current_definition=replace(broken.records[canonical].current_definition, checksum='forged'))
    with pytest.raises(GeometryError, match='canonical definition changed'):
        unify(broken, canonical, (duplicate,), model=broken_candidate)
    # An untracked canonical still receives sealed occurrences; unknown
    # duplicates create no ancestry.
    untracked = capture(model, allow_seed=True)
    untracked_candidate = model.clone(preserve_identity=True)
    with untracked_candidate.transaction():
        merge_shared_vertices(untracked_candidate, untracked, canonical, duplicate)
    del untracked.records[canonical]
    unify(untracked, canonical, (duplicate,), model=untracked_candidate)
    assert untracked.aliases[canonical]
    start, end = untracked_candidate.add_points(((3, 0, 0), (3, 1, 0)))
    new = untracked_candidate.add_line(start, end)
    before = {edge: list(rows) for edge, rows in untracked.aliases.items()}
    unify(untracked, new, (canonical,), model=untracked_candidate)
    assert {edge: list(rows) for edge, rows in untracked.aliases.items()} == before


def test_stations_refuse_unqualified_or_non_exterior_edges():
    model, first, second, plan, roots = adjacent()
    result = correspondence(model, second)
    with pytest.raises(GeometryError, match='qualified exterior'):
        stations(model, result, roots[1], (F(0),))
    cut_model, cut_first, cut_second, cut_plan, cut_roots = adjacent(cutter=True)
    cut_result = correspondence(cut_model, cut_first)
    interior = next(edge for edge, _ in cut_result.interior_incidence)
    with pytest.raises(GeometryError, match='qualified exterior'):
        stations(cut_model, cut_result, interior, (F(0),))
    with pytest.raises(GeometryError, match='exterior edge ID'):
        stations(cut_model, cut_result, 1.5, (F(0),))


def test_reseal_orientation_follows_endpoint_ids_not_tolerance():
    """A coarse tolerance must not admit the forward direction when the
    endpoint IDs order the exact reverse (T >= L admits the wrong seal)."""
    model = GeometryModel()
    origin, near, far, tip = model.add_points(((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)))
    ancestor_edge = model.add_line(origin, tip)
    duplicate = model.add_line(near, far)  # exact restriction over [1/3, 2/3]
    canonical = model.add_line(far, near)  # exact reverse: swapped endpoint IDs
    draft = capture(model, allow_seed=True)
    source = _seal(draft.records[ancestor_edge].ancestor,
                   ((1, 3), (2, 3)), definition(model, duplicate), (1, 1))
    draft.records[duplicate] = source
    # The wrong forward direction stays admissible within the SAME tolerance.
    _, _, forward = _residual(source.ancestor, ((1, 3), (2, 3)), definition(model, canonical))
    assert 0 < forward <= _unpack(source.tolerance)**2
    unify(draft, canonical, (duplicate,), model=model)
    rows = draft.aliases[canonical]
    assert len(rows) == 1
    assert rows[0].interval == ((2, 3), (1, 3))  # topology-ordered reverse
    assert rows[0].squared_distance_bound == (0, 1)  # exact reverse, zero residual
    assert rows[0].tolerance == (1, 1)
    assert rows[0].current_definition == definition(model, canonical)


def test_reseal_refuses_unmatched_degenerate_and_residual_mismatch():
    model = GeometryModel()
    near, far = model.add_points(((1, 0, 0), (2, 0, 0)))
    duplicate = model.add_line(near, far)
    canonical = model.add_line(far, near)
    stranger, bow = model.add_points(((2, 1, 0), (1.5, 5, 0)))
    mismatch = model.add_line(far, stranger)
    bowed = model.add_spline(near, (bow,), far)
    draft = capture(model, allow_seed=True)
    source = draft.records[duplicate]
    current = definition(model, canonical)
    # Degenerate endpoint identity on either side cannot orient the interval.
    degenerate = replace(source.current_definition, start=near, end=near)
    assert _reseal_occurrence(replace(source, current_definition=degenerate), current) is None
    assert _reseal_occurrence(source, replace(current, start=far, end=far)) is None
    # Mismatched endpoint identity remains unauthenticated through the producer.
    unify(draft, mismatch, (duplicate,), model=model)
    assert mismatch not in draft.aliases
    # Matching ordered endpoints but nonlinear controls: the selected forward
    # interval fails the exact whole-interval residual despite matching endpoints.
    unify(draft, bowed, (duplicate,), model=model)
    assert bowed not in draft.aliases
    # The exact reverse still seals through the same draft.
    unify(draft, canonical, (duplicate,), model=model)
    assert draft.aliases[canonical][0].interval == ((1, 1), (0, 1))


def test_qualified_copy_carries_alias_records():
    model, first, second, plan, roots = adjacent()
    binding = query(model)
    assert binding.alias_records
    made = clone_prepared_geometry(model)
    copied = query(made)
    assert copied == binding
    assert copied.alias_records == binding.alias_records
    validate(made, copied)


def test_alias_incidence_rebind_keeps_exact_and_drops_nonlinear():
    model, first, second, plan, roots = adjacent()
    binding = query(model)
    shared = shared_edge(model)
    alias = next(row for row in binding.alias_records if row.edge_id == shared)
    draft = capture(model)
    prior = definition(model, shared)
    rebind(draft, {shared: prior}, model=model)
    assert draft.aliases[shared] == [alias]
    # An unsupported nonlinear change of the same endpoints becomes unavailable.
    entity = model.edges[shared]
    position = model.vertex_position(entity.start)
    arc = model.add_point(float(position[0]), float(position[1]) + 5., float(position[2]))
    spline = model.add_spline(entity.start, (arc,), entity.end)
    assert _retained_incidence(alias, prior, definition(model, spline)) is None


def test_capture_and_finalize_refuse_callback_mutation_and_edits():
    model, first, second, plan, roots = authored_pair()
    vertex = min(model.vertices)
    def mutate(reason):
        model.move_point(vertex, .25, .25, 0.)
        return False
    with pytest.raises(GeometryError, match='source changed during capture'):
        capture(model, allow_seed=True, cancellation_check=mutate)
    with pytest.raises(GeometryError, match='cancelled'):
        capture(model, allow_seed=True, cancellation_check=lambda reason: True)
    draft = capture(model, allow_seed=True)
    model.move_point(vertex, .5, .25, 0.)
    with pytest.raises(GeometryError, match='changed before finalize'):
        finalize(model, draft)


def test_optional_split_enclosure_failure_drops_shared_occurrences(monkeypatch):
    import anygeometry.edge_subcurve_preimages as owner

    from anygeometry.edge_attachment_remapping import split_edge_attachments
    model, first, second, _, _ = adjacent()
    parent = shared_edge(model)
    assert any(row.edge_id == parent for row in query(model).alias_records)
    old_correspondences = (correspondence(model, first), correspondence(model, second))
    draft = capture(model)
    prior = definition(model, parent)
    candidate = clone_prepared_geometry(model)
    error = owner._SubcurveEnclosureUnavailable('injected optional enclosure failure')

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(owner, '_seal', unavailable)
    with candidate.transaction():
        _, children = split_edge_attachments(candidate, parent, .5)
        with pytest.raises(owner._SubcurveEnclosureUnavailable) as caught:
            owner._record_edge_subcurve_split(draft, parent, .5, children, 1e-8,
                model=candidate, parent_definition=prior)
        assert caught.value is draft.enclosure_failure is error
        # Exercise the exact fail-closed cleanup used by the batch catch. This
        # focused producer test does not manufacture a complete batch receipt.
        owner._drop_edge_subcurve_records(draft, (parent, *children))
        assert all(edge not in draft.records and edge not in draft.aliases
                   for edge in (parent, *children))
    binding = finalize(candidate, draft).binding
    assert parent not in candidate.edges
    assert set(children) <= set(binding.unavailable_edge_ids)
    assert not any(row.edge_id in (parent, *children) for row in binding.alias_records)
    for root in (first, second):
        with pytest.raises(GeometryError):
            correspondence(candidate, root)
    for old in old_correspondences:
        for edge in children:
            with pytest.raises(GeometryError):
                stations(candidate, old, edge, (.5,))
