"""Focused tests for the incremental deterministic AABB broad phase."""

from __future__ import annotations

import math
import random

import pytest

from anygeometry import GeometryModel
from anygeometry.spatial import AABB, AABBTree, IndexUpdateKind


def box(x0: float, x1: float, *, y: float = 0.0) -> AABB:
    return AABB((x0, y, 0.0), (x1, y + 1.0, 1.0))


def test_aabb_construction_is_closed_finite_and_conservative() -> None:
    bounds = AABB.from_points(
        ((2.0, -1.0, 4.0), (-3.0, 5.0, 0.0)), margin=(0.5, 1.0, 2.0)
    )
    assert bounds.minimum == (-3.5, -2.0, -2.0)
    assert bounds.maximum == (2.5, 6.0, 6.0)
    assert bounds.contains(AABB((-3.0, -1.0, 0.0), (2.0, 5.0, 4.0)))
    assert bounds.intersects(AABB((2.5, 6.0, 6.0), (3.0, 7.0, 7.0)))
    assert bounds.intersection(AABB((2.5, 6.0, 6.0), (3.0, 7.0, 7.0))) == AABB(
        (2.5, 6.0, 6.0), (2.5, 6.0, 6.0)
    )

    circular_bound = AABB.from_sphere((10.0, 20.0, 30.0), 2.5)
    assert circular_bound.contains(
        AABB.from_points(((10.0, 22.5, 30.0), (7.5, 20.0, 30.0)))
    )
    with pytest.raises(ValueError, match="finite"):
        AABB((0.0, 0.0, 0.0), (math.inf, 1.0, 1.0))
    with pytest.raises(ValueError, match="minimum"):
        AABB((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="at least one"):
        AABB.from_points(())


def test_incremental_insert_remove_update_and_sorted_queries() -> None:
    tree = AABBTree(fat_margin=0.25)
    tree.insert(("face", 9), box(2.0, 4.0))
    tree.insert(("edge", 7), box(1.0, 3.0))
    tree.insert(("vertex", 3), AABB.around_point((2.0, 0.5, 0.5)))
    tree.insert(("edge", 1), box(20.0, 21.0))
    tree.validate()

    result = tree.query(box(1.5, 2.5))
    assert result.keys == (("edge", 7), ("face", 9), ("vertex", 3))
    assert result.diagnostics.candidate_count == 3
    assert result.diagnostics.leaf_tests < len(tree) + 1

    # Movement inside a private fat leaf changes exact query results without a
    # structural reinsertion or false returned overlap.
    update = tree.update(("vertex", 3), AABB.around_point((2.1, 0.5, 0.5)))
    assert update.kind is IndexUpdateKind.UPDATE
    assert not update.structural_change
    assert tree.query(AABB.around_point((2.0, 0.5, 0.5))).keys == (
        ("edge", 7),
        ("face", 9),
    )

    moved = tree.update(
        ("vertex", 3), AABB.around_point((30.0, 30.0, 30.0))
    )
    assert moved.structural_change
    assert tree.diagnostics.reinsertions == 1

    # Shrinking inside a formerly large exact box also refits the tree instead
    # of retaining a pathological broad leaf indefinitely.
    shrink_tree = AABBTree(((("face", 1), box(-100.0, 100.0)),))
    shrunk = shrink_tree.update(("face", 1), box(-1.0, 1.0))
    assert shrunk.structural_change
    assert shrink_tree.fat_bounds(("face", 1)) == box(-1.0, 1.0)
    removed = tree.remove(("face", 9))
    assert removed.kind is IndexUpdateKind.REMOVE
    assert tree.discard(("face", 9)) is None
    tree.validate()
    assert tree.keys == (("edge", 1), ("edge", 7), ("vertex", 3))


def test_changed_region_query_unions_regions_and_can_exclude_changed_keys() -> None:
    tree = AABBTree(
        (
            (("edge", 1), box(0.0, 2.0)),
            (("edge", 2), box(1.0, 3.0)),
            (("face", 1), box(9.0, 11.0)),
            (("face", 2), box(10.0, 12.0)),
            (("vertex", 99), AABB.around_point((100.0, 100.0, 100.0))),
        )
    )
    result = tree.query_changed_regions(
        (box(1.5, 1.6), box(10.5, 10.6)),
        changed_keys=(("edge", 1), ("face", 1)),
        include_changed=False,
    )
    assert result.keys == (("edge", 2), ("face", 2))
    assert result.diagnostics.region_count == 2
    assert result.diagnostics.candidate_count == 2
    assert result.diagnostics.raw_candidate_hits == 2


def test_net_zero_transaction_reconciles_a_provisionally_materialized_tree() -> None:
    geometry = GeometryModel()

    with geometry.transaction():
        provisional = geometry.add_point(9.0, 9.0, 9.0)
        assert geometry.spatial_candidates(
            (8.5, 8.5, 8.5), (9.5, 9.5, 9.5)
        ) == (("vertex", provisional),)
        geometry.remove_vertex(provisional, record=False)
        assert geometry.spatial_candidates(
            (8.5, 8.5, 8.5), (9.5, 9.5, 9.5)
        ) == ()

    assert provisional not in geometry.vertices
    assert geometry.revision == 0
    assert geometry.spatial_candidates(
        (8.5, 8.5, 8.5), (9.5, 9.5, 9.5)
    ) == ()
    assert ("vertex", provisional) in geometry.last_change_set.spatial_updates
    assert geometry.add_point(1.0, 0.0, 0.0) == provisional + 1


def test_overlap_pairs_are_unique_sorted_and_insertion_order_independent() -> None:
    items = [
        (("edge", 4), box(0.0, 2.0)),
        (("edge", 2), box(1.0, 3.0)),
        (("face", 7), box(2.5, 4.0)),
        (("vertex", 1), AABB.around_point((100.0, 100.0, 100.0))),
    ]
    expected = (
        (("edge", 2), ("edge", 4)),
        (("edge", 2), ("face", 7)),
    )
    forward = AABBTree(items)
    backward = AABBTree(reversed(items))
    assert forward.overlap_pairs().pairs == expected
    assert backward.overlap_pairs().pairs == expected

    changed = forward.overlap_pairs(changed_keys=(("edge", 4),))
    assert changed.pairs == ((("edge", 2), ("edge", 4)),)
    cross_kind = forward.overlap_pairs(
        pair_filter=lambda left, right: left[0] != right[0]
    )
    assert cross_kind.pairs == ((("edge", 2), ("face", 7)),)


def test_bulk_constructor_is_deterministic_and_avoids_incremental_refits() -> None:
    items = [
        (("face", 9), box(10.0, 11.0, y=5.0)),
        (("edge", 4), box(0.0, 2.0)),
        (("edge", 2), box(1.0, 3.0)),
        (("vertex", 1), AABB.around_point((100.0, 100.0, 100.0))),
        (("face", 7), box(2.5, 4.0)),
    ]
    forward = AABBTree(items, fat_margin=0.1, relative_margin=0.01)
    backward = AABBTree(reversed(items), fat_margin=0.1, relative_margin=0.01)

    forward.validate()
    backward.validate()
    assert forward.items == backward.items == tuple(sorted(items))
    assert forward.query(box(1.5, 2.6)) == backward.query(box(1.5, 2.6))
    assert forward.overlap_pairs() == backward.overlap_pairs()
    assert forward.diagnostics.insertions == len(items)
    assert forward.diagnostics.refit_steps == 0
    assert forward.diagnostics.rotations == 0


def test_random_mutation_sequence_preserves_tree_invariants() -> None:
    generator = random.Random(7391)
    tree = AABBTree(fat_margin=0.05, relative_margin=0.02)
    active: dict[tuple[str, int], AABB] = {}
    for identifier in range(200):
        start = generator.uniform(-500.0, 500.0)
        bounds = box(start, start + generator.uniform(0.01, 4.0), y=identifier * 3.0)
        key = ("edge", identifier)
        active[key] = bounds
        tree.insert(key, bounds)
    for identifier in range(0, 200, 3):
        key = ("edge", identifier)
        moved = box(identifier * 7.0, identifier * 7.0 + 0.5, y=-identifier * 2.0)
        active[key] = moved
        tree.update(key, moved)
    for identifier in range(0, 200, 5):
        key = ("edge", identifier)
        active.pop(key)
        tree.remove(key)
    tree.validate()
    assert tree.items == tuple(sorted(active.items()))


def test_pair_generation_matches_small_brute_force_oracle() -> None:
    generator = random.Random(881)
    items = []
    for identifier in range(80):
        x = generator.uniform(-20.0, 20.0)
        y = generator.uniform(-20.0, 20.0)
        size = generator.uniform(0.0, 4.0)
        items.append(
            (
                ("edge", identifier),
                AABB((x, y, -1.0), (x + size, y + size, 1.0)),
            )
        )
    tree = AABBTree(items)
    expected = tuple(
        (left_key, right_key)
        for index, (left_key, left_box) in enumerate(items)
        for right_key, right_box in items[index + 1 :]
        if left_box.intersects(right_box)
    )
    assert tree.overlap_pairs().pairs == tuple(sorted(expected))


def test_large_ordered_grid_build_remains_balanced() -> None:
    tree = AABBTree(
        (
            (kind, identifier),
            AABB.around_point((float(identifier % 121), float(identifier // 121), 0.0)),
        )
        for identifier, kind in (
            (identifier, ("vertex", "edge", "face")[identifier % 3])
            for identifier in range(12_000)
        )
    )

    tree.validate()
    assert tree.diagnostics.height <= 2 * math.ceil(math.log2(len(tree)))
    assert tree.diagnostics.insertions == len(tree)
    assert tree.diagnostics.refit_steps == 0
    assert tree.diagnostics.rotations == 0


@pytest.mark.parametrize("count", (256, 1024))
def test_sparse_pair_generation_work_is_materially_subquadratic(count: int) -> None:
    # This is a work-count scaling test, not a machine-speed assertion.  Boxes
    # are deliberately separated so each query descends to only its own leaf.
    tree = AABBTree(
        (("edge", index), box(4.0 * index, 4.0 * index + 1.0))
        for index in range(count)
    )
    tree.validate()
    result = tree.overlap_pairs()
    naive_pair_count = count * (count - 1) // 2
    assert result.pairs == ()
    assert result.diagnostics.leaf_tests == count
    assert result.diagnostics.leaf_tests * 20 < naive_pair_count
    assert tree.diagnostics.height <= 2 * math.ceil(math.log2(count))
