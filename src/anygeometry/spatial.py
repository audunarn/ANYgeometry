"""Deterministic incremental three-dimensional AABB broad phase.

The index in this module deliberately knows nothing about ``GeometryModel``.
Kernel integration supplies compact ``(kind, integer_id)`` keys and
conservative entity bounds.  Keeping that boundary explicit makes spatial
updates journal-friendly and allows transactions to stage or roll them back
without the index reaching into model storage.

The implementation is a height-balanced dynamic AABB tree.  Point and region
queries are normally ``O(log n + k)`` and sparse all-pair candidate generation
is normally ``O(n log n + k)`` rather than beginning with all ``n*(n-1)/2``
pairs.  Results are sorted by entity key, independent of traversal shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import TypeAlias

__all__ = [
    "AABB",
    "AABBTree",
    "IndexDiagnostics",
    "IndexUpdate",
    "IndexUpdateKind",
    "PairQueryResult",
    "QueryDiagnostics",
    "SpatialKey",
    "SpatialQueryResult",
]


SpatialKey: TypeAlias = tuple[str, int]
Vector3: TypeAlias = tuple[float, float, float]


def _vector3(value: Sequence[Real], *, name: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three components")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} components must be finite")
    return result  # type: ignore[return-value]


def _margin3(value: Real | Sequence[Real]) -> Vector3:
    if isinstance(value, Real):
        result = (float(value),) * 3
    else:
        result = _vector3(value, name="margin")
    if any(component < 0.0 for component in result):
        raise ValueError("margin components must be non-negative")
    return result


def _key(value: SpatialKey) -> SpatialKey:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("a spatial key must be a (kind, integer_id) tuple")
    kind, identifier = value
    if not isinstance(kind, str) or not kind:
        raise TypeError("a spatial key kind must be a non-empty string")
    if isinstance(identifier, bool) or not isinstance(identifier, int):
        raise TypeError("a spatial key identifier must be an integer")
    if identifier < 0:
        raise ValueError("a spatial key identifier must be non-negative")
    return kind, identifier


@dataclass(frozen=True, slots=True)
class AABB:
    """A closed, finite, axis-aligned three-dimensional bounding box.

    Bounds are inclusive: boxes touching at a point, edge, or face overlap in
    the broad phase.  Callers constructing bounds for curves or surfaces must
    include all analytic extrema.  ``expanded`` can add a computational
    margin but is never used to shrink supplied conservative bounds.
    """

    minimum: Vector3
    maximum: Vector3

    def __post_init__(self) -> None:
        minimum = _vector3(self.minimum, name="minimum")
        maximum = _vector3(self.maximum, name="maximum")
        if any(lo > hi for lo, hi in zip(minimum, maximum, strict=True)):
            raise ValueError("AABB minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_points(
        cls,
        points: Iterable[Sequence[Real]],
        *,
        margin: Real | Sequence[Real] = 0.0,
    ) -> "AABB":
        """Return conservative bounds for a non-empty collection of points."""

        iterator = iter(points)
        try:
            first = _vector3(next(iterator), name="point")
        except StopIteration as exc:
            raise ValueError("at least one point is required") from exc
        lo = list(first)
        hi = list(first)
        for raw_point in iterator:
            point = _vector3(raw_point, name="point")
            for axis in range(3):
                lo[axis] = min(lo[axis], point[axis])
                hi[axis] = max(hi[axis], point[axis])
        return cls(tuple(lo), tuple(hi)).expanded(margin)

    @classmethod
    def around_point(
        cls,
        point: Sequence[Real],
        *,
        margin: Real | Sequence[Real] = 0.0,
    ) -> "AABB":
        position = _vector3(point, name="point")
        return cls(position, position).expanded(margin)

    @classmethod
    def from_sphere(
        cls,
        center: Sequence[Real],
        radius: Real,
        *,
        margin: Real | Sequence[Real] = 0.0,
    ) -> "AABB":
        """Return exact AABB bounds for a sphere (and thus a circular curve)."""

        position = _vector3(center, name="center")
        value = float(radius)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("radius must be finite and non-negative")
        bounds = cls(
            tuple(component - value for component in position),
            tuple(component + value for component in position),
        )
        return bounds.expanded(margin)

    @classmethod
    def union_all(cls, boxes: Iterable["AABB"]) -> "AABB":
        iterator = iter(boxes)
        try:
            result = next(iterator)
        except StopIteration as exc:
            raise ValueError("at least one AABB is required") from exc
        if not isinstance(result, AABB):
            raise TypeError("boxes must contain AABB values")
        for box in iterator:
            if not isinstance(box, AABB):
                raise TypeError("boxes must contain AABB values")
            result = result.union(box)
        return result

    @property
    def extents(self) -> Vector3:
        return tuple(
            hi - lo for lo, hi in zip(self.minimum, self.maximum, strict=True)
        )  # type: ignore[return-value]

    @property
    def center(self) -> Vector3:
        return tuple(
            0.5 * (lo + hi)
            for lo, hi in zip(self.minimum, self.maximum, strict=True)
        )  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        dx, dy, dz = self.extents
        return dx * dy * dz

    @property
    def surface_area(self) -> float:
        """Surface-area heuristic cost, valid for degenerate boxes too."""

        dx, dy, dz = self.extents
        return 2.0 * (dx * dy + dy * dz + dz * dx)

    @property
    def heuristic_cost(self) -> float:
        """A non-negative tree cost that also distinguishes lines and points."""

        dx, dy, dz = self.extents
        return self.surface_area + dx + dy + dz

    def expanded(self, margin: Real | Sequence[Real]) -> "AABB":
        amount = _margin3(margin)
        return AABB(
            tuple(lo - delta for lo, delta in zip(self.minimum, amount, strict=True)),
            tuple(hi + delta for hi, delta in zip(self.maximum, amount, strict=True)),
        )

    def union(self, other: "AABB") -> "AABB":
        if not isinstance(other, AABB):
            return NotImplemented
        return AABB(
            tuple(
                min(left, right)
                for left, right in zip(self.minimum, other.minimum, strict=True)
            ),
            tuple(
                max(left, right)
                for left, right in zip(self.maximum, other.maximum, strict=True)
            ),
        )

    def intersects(self, other: "AABB", *, margin: Real = 0.0) -> bool:
        """Return whether two closed boxes overlap within an optional margin."""

        if not isinstance(other, AABB):
            return False
        tolerance = float(margin)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("intersection margin must be finite and non-negative")
        return all(
            left_lo <= right_hi + tolerance
            and right_lo <= left_hi + tolerance
            for left_lo, left_hi, right_lo, right_hi in zip(
                self.minimum,
                self.maximum,
                other.minimum,
                other.maximum,
                strict=True,
            )
        )

    overlaps = intersects

    def contains(self, other: "AABB") -> bool:
        if not isinstance(other, AABB):
            return False
        return all(
            outer_lo <= inner_lo and inner_hi <= outer_hi
            for outer_lo, outer_hi, inner_lo, inner_hi in zip(
                self.minimum,
                self.maximum,
                other.minimum,
                other.maximum,
                strict=True,
            )
        )

    def intersection(self, other: "AABB") -> "AABB | None":
        if not self.intersects(other):
            return None
        return AABB(
            tuple(
                max(left, right)
                for left, right in zip(self.minimum, other.minimum, strict=True)
            ),
            tuple(
                min(left, right)
                for left, right in zip(self.maximum, other.maximum, strict=True)
            ),
        )


@dataclass(frozen=True, slots=True)
class QueryDiagnostics:
    """Work counters for one broad-phase query or pair-generation pass."""

    region_count: int = 0
    node_visits: int = 0
    branch_visits: int = 0
    leaf_tests: int = 0
    raw_candidate_hits: int = 0
    candidate_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "region_count",
            "node_visits",
            "branch_visits",
            "leaf_tests",
            "raw_candidate_hits",
            "candidate_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def combined(self, other: "QueryDiagnostics") -> "QueryDiagnostics":
        """Add work counts; callers should replace de-duplicated candidates."""

        return QueryDiagnostics(
            region_count=self.region_count + other.region_count,
            node_visits=self.node_visits + other.node_visits,
            branch_visits=self.branch_visits + other.branch_visits,
            leaf_tests=self.leaf_tests + other.leaf_tests,
            raw_candidate_hits=self.raw_candidate_hits + other.raw_candidate_hits,
            candidate_count=self.candidate_count + other.candidate_count,
        )


@dataclass(frozen=True, slots=True)
class SpatialQueryResult:
    keys: tuple[SpatialKey, ...]
    diagnostics: QueryDiagnostics


@dataclass(frozen=True, slots=True)
class PairQueryResult:
    pairs: tuple[tuple[SpatialKey, SpatialKey], ...]
    diagnostics: QueryDiagnostics


class IndexUpdateKind(str, Enum):
    INSERT = "insert"
    REMOVE = "remove"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class IndexUpdate:
    """The journal-friendly outcome of one index mutation."""

    kind: IndexUpdateKind
    key: SpatialKey
    previous: AABB | None
    current: AABB | None
    structural_change: bool
    refit_steps: int


@dataclass(frozen=True, slots=True)
class IndexDiagnostics:
    """Cumulative mutation and current-shape diagnostics."""

    size: int
    height: int
    branch_count: int
    insertions: int
    removals: int
    updates: int
    reinsertions: int
    rotations: int
    refit_steps: int


@dataclass(slots=True)
class _Node:
    aabb: AABB
    parent: int | None
    left: int | None
    right: int | None
    height: int
    key: SpatialKey | None
    minimum_key: SpatialKey

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class AABBTree:
    """Incremental height-balanced AABB tree keyed by geometry entity.

    ``fat_margin`` and ``relative_margin`` affect only private tree bounds.
    Exact conservative bounds are retained separately and are always checked
    before a key is returned.  Consequently tree padding can reduce update
    churn without introducing false returned overlaps.
    """

    def __init__(
        self,
        items: Iterable[tuple[SpatialKey, AABB]] = (),
        *,
        fat_margin: Real = 0.0,
        relative_margin: Real = 0.0,
    ) -> None:
        absolute = float(fat_margin)
        relative = float(relative_margin)
        if not math.isfinite(absolute) or absolute < 0.0:
            raise ValueError("fat_margin must be finite and non-negative")
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError("relative_margin must be finite and non-negative")
        self._fat_margin = absolute
        self._relative_margin = relative
        self._root: int | None = None
        self._nodes: dict[int, _Node] = {}
        self._leaves: dict[SpatialKey, int] = {}
        self._boxes: dict[SpatialKey, AABB] = {}
        self._next_node = 0
        self._insertions = 0
        self._removals = 0
        self._updates = 0
        self._reinsertions = 0
        self._rotations = 0
        self._refit_steps = 0
        for key, bounds in items:
            self.insert(key, bounds)

    def __len__(self) -> int:
        return len(self._boxes)

    def __contains__(self, key: object) -> bool:
        return key in self._boxes

    @property
    def keys(self) -> tuple[SpatialKey, ...]:
        return tuple(sorted(self._boxes))

    @property
    def items(self) -> tuple[tuple[SpatialKey, AABB], ...]:
        return tuple((key, self._boxes[key]) for key in sorted(self._boxes))

    @property
    def diagnostics(self) -> IndexDiagnostics:
        root_height = -1 if self._root is None else self._nodes[self._root].height
        return IndexDiagnostics(
            size=len(self),
            height=root_height,
            branch_count=max(0, len(self._nodes) - len(self._leaves)),
            insertions=self._insertions,
            removals=self._removals,
            updates=self._updates,
            reinsertions=self._reinsertions,
            rotations=self._rotations,
            refit_steps=self._refit_steps,
        )

    def bounds(self, key: SpatialKey) -> AABB:
        return self._boxes[_key(key)]

    def fat_bounds(self, key: SpatialKey) -> AABB:
        normalized = _key(key)
        return self._nodes[self._leaves[normalized]].aabb

    def insert(self, key: SpatialKey, bounds: AABB) -> IndexUpdate:
        normalized = _key(key)
        if not isinstance(bounds, AABB):
            raise TypeError("bounds must be an AABB")
        if normalized in self._boxes:
            raise KeyError(f"spatial key already exists: {normalized!r}")
        leaf = self._allocate_node(
            _Node(
                aabb=self._fatten(bounds),
                parent=None,
                left=None,
                right=None,
                height=0,
                key=normalized,
                minimum_key=normalized,
            )
        )
        self._leaves[normalized] = leaf
        self._boxes[normalized] = bounds
        steps = self._insert_leaf(leaf)
        self._insertions += 1
        self._refit_steps += steps
        return IndexUpdate(
            IndexUpdateKind.INSERT,
            normalized,
            None,
            bounds,
            True,
            steps,
        )

    def remove(self, key: SpatialKey) -> IndexUpdate:
        normalized = _key(key)
        if normalized not in self._leaves:
            raise KeyError(normalized)
        previous = self._boxes.pop(normalized)
        leaf = self._leaves.pop(normalized)
        steps = self._detach_leaf(leaf)
        del self._nodes[leaf]
        self._removals += 1
        self._refit_steps += steps
        return IndexUpdate(
            IndexUpdateKind.REMOVE,
            normalized,
            previous,
            None,
            True,
            steps,
        )

    def discard(self, key: SpatialKey) -> IndexUpdate | None:
        normalized = _key(key)
        if normalized not in self._leaves:
            return None
        return self.remove(normalized)

    def update(self, key: SpatialKey, bounds: AABB) -> IndexUpdate:
        normalized = _key(key)
        if not isinstance(bounds, AABB):
            raise TypeError("bounds must be an AABB")
        if normalized not in self._leaves:
            raise KeyError(normalized)
        previous = self._boxes[normalized]
        leaf = self._leaves[normalized]
        new_fat = self._fatten(bounds)
        self._boxes[normalized] = bounds
        self._updates += 1
        # The private fat leaf exists specifically to absorb small exact-box
        # movements.  A second, looser containment test prevents a once-large
        # leaf from remaining huge forever after its exact box shrinks.
        current_fat = self._nodes[leaf].aabb
        slack = 4.0 * self._padding_for(bounds)
        reasonable_limit = new_fat.expanded(slack)
        if current_fat.contains(bounds) and reasonable_limit.contains(current_fat):
            return IndexUpdate(
                IndexUpdateKind.UPDATE,
                normalized,
                previous,
                bounds,
                False,
                0,
            )

        steps = self._detach_leaf(leaf)
        node = self._nodes[leaf]
        node.aabb = new_fat
        node.parent = None
        steps += self._insert_leaf(leaf)
        self._reinsertions += 1
        self._refit_steps += steps
        return IndexUpdate(
            IndexUpdateKind.UPDATE,
            normalized,
            previous,
            bounds,
            True,
            steps,
        )

    def upsert(self, key: SpatialKey, bounds: AABB) -> IndexUpdate:
        normalized = _key(key)
        if normalized in self._leaves:
            return self.update(normalized, bounds)
        return self.insert(normalized, bounds)

    def query(
        self,
        bounds: AABB,
        *,
        exclude: Iterable[SpatialKey] = (),
        kinds: Iterable[str] | None = None,
    ) -> SpatialQueryResult:
        """Return all exact stored boxes overlapping ``bounds``, sorted."""

        if not isinstance(bounds, AABB):
            raise TypeError("bounds must be an AABB")
        excluded = frozenset(_key(key) for key in exclude)
        accepted_kinds = None if kinds is None else frozenset(kinds)
        if accepted_kinds is not None and any(
            not isinstance(kind, str) or not kind for kind in accepted_kinds
        ):
            raise TypeError("kinds must contain non-empty strings")
        keys, diagnostics = self._query_one(bounds, excluded, accepted_kinds)
        return SpatialQueryResult(tuple(sorted(keys)), diagnostics)

    def query_regions(
        self,
        regions: Iterable[AABB],
        *,
        changed_keys: Iterable[SpatialKey] = (),
        include_changed: bool = True,
        kinds: Iterable[str] | None = None,
    ) -> SpatialQueryResult:
        """Query the union of changed/affected regions without a global scan.

        ``regions`` may include old bounds of removed entities as well as new
        bounds.  Set ``include_changed=False`` to return only neighbouring
        candidates; this is useful for incremental validation closures.
        """

        changed = frozenset(_key(key) for key in changed_keys)
        excluded = frozenset() if include_changed else changed
        accepted_kinds = None if kinds is None else frozenset(kinds)
        if accepted_kinds is not None and any(
            not isinstance(kind, str) or not kind for kind in accepted_kinds
        ):
            raise TypeError("kinds must contain non-empty strings")
        found: set[SpatialKey] = set()
        totals = QueryDiagnostics()
        for region in regions:
            if not isinstance(region, AABB):
                raise TypeError("regions must contain AABB values")
            keys, diagnostics = self._query_one(region, excluded, accepted_kinds)
            found.update(keys)
            totals = totals.combined(diagnostics)
        final = tuple(sorted(found))
        totals = QueryDiagnostics(
            region_count=totals.region_count,
            node_visits=totals.node_visits,
            branch_visits=totals.branch_visits,
            leaf_tests=totals.leaf_tests,
            raw_candidate_hits=totals.raw_candidate_hits,
            candidate_count=len(final),
        )
        return SpatialQueryResult(final, totals)

    query_changed_regions = query_regions

    def overlap_pairs(
        self,
        *,
        changed_keys: Iterable[SpatialKey] | None = None,
        pair_filter: Callable[[SpatialKey, SpatialKey], bool] | None = None,
    ) -> PairQueryResult:
        """Generate actual AABB-overlap pairs through tree queries.

        When ``changed_keys`` is provided, only pairs incident to one of those
        active keys are generated.  This supports incremental audits while old
        bounds of removals can be handled separately with ``query_regions``.
        """

        if pair_filter is not None and not callable(pair_filter):
            raise TypeError("pair_filter must be callable")
        seeds = (
            self.keys
            if changed_keys is None
            else tuple(sorted({_key(key) for key in changed_keys}))
        )
        missing = tuple(key for key in seeds if key not in self._boxes)
        if missing:
            raise KeyError(f"spatial keys do not exist: {missing!r}")
        pairs: set[tuple[SpatialKey, SpatialKey]] = set()
        totals = QueryDiagnostics()
        for seed in seeds:
            nearby, diagnostics = self._query_one(
                self._boxes[seed], frozenset(), None
            )
            totals = totals.combined(diagnostics)
            for other in nearby:
                if other == seed:
                    continue
                pair = (seed, other) if seed < other else (other, seed)
                if pair_filter is None or pair_filter(*pair):
                    pairs.add(pair)
        ordered = tuple(sorted(pairs))
        totals = QueryDiagnostics(
            region_count=totals.region_count,
            node_visits=totals.node_visits,
            branch_visits=totals.branch_visits,
            leaf_tests=totals.leaf_tests,
            raw_candidate_hits=totals.raw_candidate_hits,
            candidate_count=len(ordered),
        )
        return PairQueryResult(ordered, totals)

    def validate(self) -> None:
        """Raise ``AssertionError`` if internal tree invariants are broken."""

        if self._root is None:
            assert not self._nodes
            assert not self._leaves
            assert not self._boxes
            return
        assert self._root in self._nodes
        assert self._nodes[self._root].parent is None
        seen: set[int] = set()

        def visit(node_id: int) -> tuple[int, AABB, SpatialKey, int]:
            assert node_id not in seen
            seen.add(node_id)
            node = self._nodes[node_id]
            if node.is_leaf:
                assert node.right is None
                assert node.height == 0
                assert node.key is not None
                assert node.minimum_key == node.key
                assert self._leaves[node.key] == node_id
                assert node.aabb.contains(self._boxes[node.key])
                return 0, node.aabb, node.key, 1
            assert node.key is None
            assert node.left is not None and node.right is not None
            left = self._nodes[node.left]
            right = self._nodes[node.right]
            assert left.parent == node_id
            assert right.parent == node_id
            left_height, left_box, left_key, left_count = visit(node.left)
            right_height, right_box, right_key, right_count = visit(node.right)
            expected_height = 1 + max(left_height, right_height)
            assert node.height == expected_height
            assert abs(left_height - right_height) <= 1
            assert node.aabb == left_box.union(right_box)
            assert node.minimum_key == min(left_key, right_key)
            return (
                expected_height,
                node.aabb,
                node.minimum_key,
                left_count + right_count,
            )

        _, _, _, leaf_count = visit(self._root)
        assert seen == set(self._nodes)
        assert leaf_count == len(self._leaves) == len(self._boxes)
        assert set(self._leaves) == set(self._boxes)

    def _fatten(self, bounds: AABB) -> AABB:
        return bounds.expanded(self._padding_for(bounds))

    def _padding_for(self, bounds: AABB) -> float:
        largest_extent = max(bounds.extents)
        return self._fat_margin + self._relative_margin * largest_extent

    def _allocate_node(self, node: _Node) -> int:
        identifier = self._next_node
        self._next_node += 1
        self._nodes[identifier] = node
        return identifier

    def _insert_leaf(self, leaf: int) -> int:
        if self._root is None:
            self._root = leaf
            self._nodes[leaf].parent = None
            return 0

        sibling = self._choose_sibling(self._nodes[leaf].aabb)
        old_parent = self._nodes[sibling].parent
        parent_box = self._nodes[sibling].aabb.union(self._nodes[leaf].aabb)
        left, right = (
            (sibling, leaf)
            if self._nodes[sibling].minimum_key
            <= self._nodes[leaf].minimum_key
            else (leaf, sibling)
        )
        parent = self._allocate_node(
            _Node(
                aabb=parent_box,
                parent=old_parent,
                left=left,
                right=right,
                height=self._nodes[sibling].height + 1,
                key=None,
                minimum_key=min(
                    self._nodes[sibling].minimum_key,
                    self._nodes[leaf].minimum_key,
                ),
            )
        )
        self._nodes[left].parent = parent
        self._nodes[right].parent = parent
        if old_parent is None:
            self._root = parent
        else:
            ancestor = self._nodes[old_parent]
            if ancestor.left == sibling:
                ancestor.left = parent
            else:
                assert ancestor.right == sibling
                ancestor.right = parent
        return self._refit_from(parent)

    def _detach_leaf(self, leaf: int) -> int:
        if leaf == self._root:
            self._root = None
            self._nodes[leaf].parent = None
            return 0
        node = self._nodes[leaf]
        parent = node.parent
        assert parent is not None
        parent_node = self._nodes[parent]
        sibling = (
            parent_node.right if parent_node.left == leaf else parent_node.left
        )
        assert sibling is not None
        grandparent = parent_node.parent
        if grandparent is None:
            self._root = sibling
            self._nodes[sibling].parent = None
            steps = 0
        else:
            grandparent_node = self._nodes[grandparent]
            if grandparent_node.left == parent:
                grandparent_node.left = sibling
            else:
                assert grandparent_node.right == parent
                grandparent_node.right = sibling
            self._nodes[sibling].parent = grandparent
            steps = self._refit_from(grandparent)
        del self._nodes[parent]
        node.parent = None
        return steps

    def _choose_sibling(self, leaf_box: AABB) -> int:
        assert self._root is not None
        index = self._root
        while not self._nodes[index].is_leaf:
            node = self._nodes[index]
            assert node.left is not None and node.right is not None
            combined_cost = node.aabb.union(leaf_box).heuristic_cost
            inheritance = 2.0 * (combined_cost - node.aabb.heuristic_cost)
            parent_here = 2.0 * combined_cost

            def descend_cost(child_id: int) -> float:
                child = self._nodes[child_id]
                joined = child.aabb.union(leaf_box).heuristic_cost
                if child.is_leaf:
                    return joined + inheritance
                return joined - child.aabb.heuristic_cost + inheritance

            left_cost = descend_cost(node.left)
            right_cost = descend_cost(node.right)
            # A strict height-balanced tree must attach a new leaf beside an
            # existing leaf. Attaching beside an arbitrarily tall internal
            # branch can create an imbalance larger than one rotation can
            # repair. Continue the SAH descent even when a new parent at this
            # level has the lower area cost.
            if left_cost < right_cost:
                index = node.left
            elif right_cost < left_cost:
                index = node.right
            else:
                left_key = self._nodes[node.left].minimum_key
                right_key = self._nodes[node.right].minimum_key
                index = node.left if left_key <= right_key else node.right
        return index

    def _sync(self, node_id: int) -> None:
        node = self._nodes[node_id]
        if node.is_leaf:
            node.height = 0
            assert node.key is not None
            node.minimum_key = node.key
            return
        assert node.left is not None and node.right is not None
        left = self._nodes[node.left]
        right = self._nodes[node.right]
        node.height = 1 + max(left.height, right.height)
        node.aabb = left.aabb.union(right.aabb)
        node.minimum_key = min(left.minimum_key, right.minimum_key)

    def _refit_from(self, start: int | None) -> int:
        steps = 0
        index = start
        while index is not None:
            # A newly attached/detached child changes this node before its
            # cached height is consulted by the balancing decision.
            self._sync(index)
            index = self._balance(index)
            self._sync(index)
            steps += 1
            index = self._nodes[index].parent
        return steps

    def _balance(self, node_id: int) -> int:
        node = self._nodes[node_id]
        if node.is_leaf or node.height < 2:
            return node_id
        assert node.left is not None and node.right is not None
        left_id = node.left
        right_id = node.right
        left = self._nodes[left_id]
        right = self._nodes[right_id]
        balance = right.height - left.height

        if balance > 1:
            assert right.left is not None and right.right is not None
            first_id = right.left
            second_id = right.right
            first = self._nodes[first_id]
            second = self._nodes[second_id]
            old_parent = node.parent
            right.left = node_id
            right.parent = old_parent
            node.parent = right_id
            self._replace_child(old_parent, node_id, right_id)
            choose_first = first.height > second.height or (
                first.height == second.height
                and first.minimum_key <= second.minimum_key
            )
            if choose_first:
                right.right = first_id
                node.right = second_id
                second.parent = node_id
            else:
                right.right = second_id
                node.right = first_id
                first.parent = node_id
            self._sync(node_id)
            self._sync(right_id)
            self._rotations += 1
            return right_id

        if balance < -1:
            assert left.left is not None and left.right is not None
            first_id = left.left
            second_id = left.right
            first = self._nodes[first_id]
            second = self._nodes[second_id]
            old_parent = node.parent
            left.left = node_id
            left.parent = old_parent
            node.parent = left_id
            self._replace_child(old_parent, node_id, left_id)
            choose_first = first.height > second.height or (
                first.height == second.height
                and first.minimum_key <= second.minimum_key
            )
            if choose_first:
                left.right = first_id
                node.left = second_id
                second.parent = node_id
            else:
                left.right = second_id
                node.left = first_id
                first.parent = node_id
            self._sync(node_id)
            self._sync(left_id)
            self._rotations += 1
            return left_id

        return node_id

    def _replace_child(
        self, parent: int | None, old_child: int, new_child: int
    ) -> None:
        if parent is None:
            self._root = new_child
            return
        node = self._nodes[parent]
        if node.left == old_child:
            node.left = new_child
        else:
            assert node.right == old_child
            node.right = new_child

    def _query_one(
        self,
        bounds: AABB,
        excluded: frozenset[SpatialKey],
        accepted_kinds: frozenset[str] | None,
    ) -> tuple[set[SpatialKey], QueryDiagnostics]:
        if self._root is None:
            return set(), QueryDiagnostics(region_count=1)
        stack = [self._root]
        found: set[SpatialKey] = set()
        node_visits = 0
        branch_visits = 0
        leaf_tests = 0
        raw_hits = 0
        while stack:
            node_id = stack.pop()
            node = self._nodes[node_id]
            node_visits += 1
            if not node.aabb.intersects(bounds):
                continue
            if node.is_leaf:
                leaf_tests += 1
                assert node.key is not None
                key = node.key
                if (
                    key not in excluded
                    and (accepted_kinds is None or key[0] in accepted_kinds)
                    and self._boxes[key].intersects(bounds)
                ):
                    found.add(key)
                    raw_hits += 1
                continue
            branch_visits += 1
            assert node.left is not None and node.right is not None
            children = (node.left, node.right)
            first, second = sorted(
                children, key=lambda child: self._nodes[child].minimum_key
            )
            stack.append(second)
            stack.append(first)
        return found, QueryDiagnostics(
            region_count=1,
            node_visits=node_visits,
            branch_visits=branch_visits,
            leaf_tests=leaf_tests,
            raw_candidate_hits=raw_hits,
            candidate_count=len(found),
        )
