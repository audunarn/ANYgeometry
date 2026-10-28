"""Deeply immutable fast-path records (tuples of Fractions, floats and ints only)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = ["CoverWitness", "AtlasReport", "StructuralCertificate", "RationalProjectionCertificate", "FrozenRoot", "FrozenBox", "ArcRecord", "ComponentRecord", "digest_of"]


@dataclass(frozen=True, slots=True)
class StructuralCertificate:
    """Exact certificate that world equality is the scalar equation ``field(u, v) = 0`` in native knots.

    ``rows`` are two independent vectors orthogonal to ``direction``; ``affine`` holds per projection row
    ``(a, b, c)`` with ``row . A_ij = a + b Greville_u[i] + c Greville_v[j]`` verified for ALL controls;
    ``determinant`` = b1 c2 - b2 c1 != 0.  ``field`` is the Bernstein/B-spline coefficient grid of
    ``G = (B - A) . D / (D . D)``.  ``domain`` is the native rectangle common to both selected ranges
    (``None`` when the rectangles do not meet).
    """

    direction: tuple
    rows: tuple
    affine: tuple
    determinant: object
    greville_u: tuple
    greville_v: tuple
    field: tuple
    domain: object


@dataclass(frozen=True, slots=True)
class RationalProjectionCertificate:
    """Exact separable rational projection and positive-denominator scalar numerator.

    Normalized tensor weights factor, projected controls are origin+U*x_i+V*y_j,
    and every rational univariate span has the same strict derivative sign.
    Hence native correspondence is u=s,v=t on the entire active knot rectangle.
    """
    direction: tuple
    rows: tuple
    projection_origin: tuple
    projection_basis: tuple
    coordinates: tuple
    weight_factors: tuple
    derivative_proofs: tuple
    greville_u: tuple
    greville_v: tuple
    field: tuple
    domain: object


@dataclass(frozen=True, slots=True)
class FrozenRoot:
    axis: int
    value: object
    lo: object
    hi: object
    poly: tuple
    edge: tuple
    patch: int
    ident: int


@dataclass(frozen=True, slots=True)
class FrozenBox:
    patch: int
    u_range: tuple
    v_range: tuple
    coefficients: tuple
    kind: str


@dataclass(frozen=True, slots=True)
class ArcRecord:
    box_index: int
    first_root: int
    second_root: int
    slicing_axis: int               # 0: the arc is a graph over u, 1: over v (box kind "v" / "u")
    native_box: tuple               # ((u0,u1),(v0,v1)) Fractions enclosing the whole arc
    first_chart: tuple              # outward float chart enclosure on the first surface
    second_chart: tuple
    world_lo: tuple                 # outward float world enclosure (intersection of both originals' bounds)
    world_hi: tuple


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    index: int
    closed: bool
    root_ids: tuple                 # certified endpoint identities in path order (closed: first repeated last)
    arcs: tuple                     # ArcRecord in path order
    first_chart: tuple
    second_chart: tuple
    world_lo: tuple
    world_hi: tuple


@dataclass(frozen=True, slots=True)
class CoverWitness:
    """A coverage cell whose zeros are certified to lie on ONE regular graph (local proof only).

    ``spans`` = ((iu, iv), (is, it)) span indices; ``cell`` = four Fraction pairs in span-local [0,1]
    coordinates; ``free_coordinate``/``region`` = the padded neighborhood of the strict Krawczyk inclusion
    (region = four pairs: free interval at ``free_coordinate``, unknown boxes elsewhere).  It is NOT a curve:
    Section roots and component arcs refer to its graph by witness index.
    """

    spans: tuple
    cell: tuple
    free_coordinate: int
    region: tuple


@dataclass(frozen=True, slots=True)
class AtlasReport:
    """Accounting of the cover and attempted continuation; components require both to complete."""

    excluded_by_hull: int
    excluded_by_f: int
    covered: int
    unresolved: int
    frontier_remaining: int         # nodes left on the stack when a budget stopped the cover
    complete_local_cover: bool      # every cell excluded or covered by a local witness (NO stitching implied)
    stitching_implemented: bool
    witnesses: tuple                # CoverWitness records (bounded by max_trace_segments)
    unresolved_sample: tuple        # first unresolved (box_a, box_b) pairs, bounded
    reason: str
    roots: tuple = ()
    components: tuple = ()
    excluded_by_k: int = 0
    reductions: tuple = ()
    coverage_uses: tuple = ()
    covered_cells: int = 0


@dataclass(frozen=True, slots=True)
class CoverageUse:
    """Candidate's entire residual box is contained in this prior graph proof region."""
    spans: tuple
    cell: tuple
    witness: int


@dataclass(frozen=True, slots=True)
class CellReduction:
    """Every zero of original lies in residual by the displayed exact K operator.

    Coordinates are span-local, free Phi equals the original free interval,
    and k_image lists the three unknown projections in coordinate order.
    """
    spans: tuple
    original: tuple
    residual: tuple
    free_coordinate: int
    k_image: tuple


@dataclass(frozen=True, slots=True)
class SectionRoot:
    """Unique root of F=0 on a fixed coordinate section of a graph witness."""

    witness: int
    free_coordinate: int
    value: object
    region: tuple


@dataclass(frozen=True, slots=True)
class GenericArc:
    witness: int
    first_root: int
    second_root: int
    direction: int
    first_chart: tuple
    second_chart: tuple
    world_lo: tuple
    world_hi: tuple


def digest_of(*parts) -> str:
    """Integrity digest over deterministic reprs (detects altered fields; not a secret)."""
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
