"""Immutable source-definition snapshots, limits and result records.

The only dependency on shared code is read-only: ``NurbsSurface`` (public
definition fields), ``TolerancePolicy`` and ``GeometryError``.  Qualification
limits are read by attribute from an ``IntersectionQualificationPolicy``
(predicates.py) with the documented current defaults; no receipt or
``plane_nurbs.CURRENT_WORK`` authority is used.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field, fields
from enum import Enum

from ..errors import GeometryError
from ..nurbs import NurbsSurface

__all__ = [
    "NurbsSurfaceIntersectionCancelled",
    "NurbsSurfaceIntersectionIntegrityError",
    "IntersectionStatus",
    "IntersectionStats",
    "WorkCounters",
    "IntersectionResult",
    "SourceDefinition",
    "Limits",
]

_FIELDS = ("degree_u", "degree_v", "knots_u", "knots_v", "controls", "weights", "u_range", "v_range")


def isolate_callback(callback):
    """Wrap a user callback so it always runs in a fresh copy of the context captured NOW (public entry).

    The copy never contains a private engine worker, and context mutations made by the callback (for example
    resetting ``plane_nurbs.CURRENT_WORK``) are discarded.  Exceptions and return values pass through.
    """
    if callback is None:
        return None
    base = contextvars.copy_context()

    def call():
        return base.copy().run(callback)

    return call


def _freeze(value):
    """Recursively copy lists/tuples into tuples; other values must already be immutable scalars."""
    kind = type(value)
    # Native constructor fields dominate live source checks. Still traverse
    # EVERY container so raw nested mutations remain visible; avoid generic
    # isinstance dispatch for the validated native representation.
    if value is None or kind is int or kind is float or kind is str:
        return value
    if kind is tuple:
        return tuple(_freeze(item) for item in value)
    if kind is list:
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (int, float, str)):
        return value
    raise GeometryError("NURBS surface definition holds an unsupported value")


class NurbsSurfaceIntersectionCancelled(GeometryError):
    """The cancellation callback returned a truthy value; no result is issued."""


class NurbsSurfaceIntersectionIntegrityError(GeometryError):
    """A live input no longer matches the definition captured at query start."""


class NurbsSurfaceIntersectionRefused(GeometryError):
    """An evaluation could not meet its representability or work bound."""


class IntersectionStatus(str, Enum):
    DISJOINT = "disjoint"            # complete exclusion certificate, no unaccounted frontier
    UNCLASSIFIED = "unclassified"    # intersecting/unproven pair or exhausted resources
    CERTIFIED = "certified"          # complete cover plus component accounting; inspect algorithm field


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    """Complete literal public definition of one surface (all containers are immutable tuples)."""

    values: tuple

    @staticmethod
    def capture(surface: object) -> "SourceDefinition":
        """Deep-frozen canonical literal definition: validated by the public constructor, no aliases."""
        if type(surface) is not NurbsSurface:
            raise GeometryError("NURBS surface intersection needs NurbsSurface inputs")
        try:
            frozen = tuple(_freeze(getattr(surface, name)) for name in _FIELDS)
        except AttributeError as error:
            raise GeometryError("NURBS surface definition is incomplete") from error
        rebuilt = NurbsSurface(*frozen)          # re-validates; its containers are fresh tuples
        return SourceDefinition(tuple(getattr(rebuilt, name) for name in _FIELDS))

    def matches(self, surface: object) -> bool:
        """Complete live public definition (re-frozen, so in-place list mutation is seen) equals the snapshot."""
        if type(surface) is not NurbsSurface:
            return False
        try:
            return tuple(_freeze(getattr(surface, name)) for name in _FIELDS) == self.values
        except (AttributeError, GeometryError):
            return False

    def rebuild(self) -> NurbsSurface:
        """Trusted value reconstructed (and re-validated by the public constructor) from the snapshot."""
        return NurbsSurface(*self.values)


@dataclass(frozen=True, slots=True)
class Limits:
    max_boxes_per_pair: int = 200000
    max_subdivision_depth: int = 32
    max_components: int = 4096
    max_trace_segments: int = 65536
    max_newton_iterations: int = 16
    interval_mode: str = "filtered"

    @staticmethod
    def read(qualification: object) -> "Limits":
        if qualification is None:
            return Limits()
        values = {}
        for name in Limits.__dataclass_fields__:
            default = Limits.__dataclass_fields__[name].default
            value = getattr(qualification, name, default)
            if name=="interval_mode":
                if value not in ("filtered","exact"):
                    raise GeometryError("qualification interval_mode must be filtered or exact")
                values[name]=value
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GeometryError(f"qualification {name} must be a positive integer")
            values[name] = value
        return Limits(**values)


@dataclass(frozen=True, slots=True)
class IntersectionStats:
    """Immutable issued statistics (a snapshot of ``WorkCounters``)."""

    boxes_visited: int = 0
    excluded_pairs: int = 0
    splits: int = 0
    hull_evaluations: int = 0
    hull_cache_hits: int = 0
    cells_prepared: int = 0
    peak_stack: int = 0
    max_depth: int = 0
    cancellation_checks: int = 0
    integrity_checks: int = 0
    prepare_seconds: float = 0.0
    prune_seconds: float = 0.0       # generic node hull retrieval/separation + F exclusion; excludes initial prep
    total_seconds: float = 0.0
    structural_seconds: float = 0.0
    topology_seconds: float = 0.0
    engine_box_charges: int = 0
    engine_subdivisions: int = 0
    roots: int = 0
    boxes: int = 0
    arcs: int = 0
    components: int = 0
    excluded_by_f: int = 0
    certificate_attempts: int = 0
    certificates: int = 0
    unresolved_cells: int = 0
    pieces_prepared: int = 0
    piece_reuses: int = 0
    exclusion_seconds: float = 0.0   # homogeneous F exclusion SUBSET of prune_seconds; do not sum both
    certification_seconds: float = 0.0
    section_attempts: int = 0
    excluded_by_k: int = 0
    contractions: int = 0
    float_attempts: int = 0
    exact_escalations: int = 0
    filtered_seconds: float = 0.0
    exact_seconds: float = 0.0
    cell_cache_peak: int = 0
    broadphase_checks: int = 0
    reused_graph_cells: int = 0
    covered_cells: int = 0
    coverage_lookup_checks: int = 0
    exact_preparations_reused: int = 0


@dataclass(slots=True)
class WorkCounters:
    """Mutable internal counters; never embedded in a result (use ``freeze``)."""

    boxes_visited: int = 0
    excluded_pairs: int = 0
    splits: int = 0
    hull_evaluations: int = 0
    hull_cache_hits: int = 0
    cells_prepared: int = 0
    peak_stack: int = 0
    max_depth: int = 0
    cancellation_checks: int = 0
    integrity_checks: int = 0
    prepare_seconds: float = 0.0
    prune_seconds: float = 0.0       # same stages as the issued IntersectionStats
    total_seconds: float = 0.0
    structural_seconds: float = 0.0
    topology_seconds: float = 0.0
    engine_box_charges: int = 0
    engine_subdivisions: int = 0
    roots: int = 0
    boxes: int = 0
    arcs: int = 0
    components: int = 0
    excluded_by_f: int = 0
    certificate_attempts: int = 0
    certificates: int = 0
    unresolved_cells: int = 0
    pieces_prepared: int = 0
    piece_reuses: int = 0
    exclusion_seconds: float = 0.0   # F exclusion subset, not additional pruning time
    certification_seconds: float = 0.0
    section_attempts: int = 0
    excluded_by_k: int = 0
    contractions: int = 0
    float_attempts: int = 0
    exact_escalations: int = 0
    filtered_seconds: float = 0.0
    exact_seconds: float = 0.0
    cell_cache_peak: int = 0
    broadphase_checks: int = 0
    reused_graph_cells: int = 0
    covered_cells: int = 0
    coverage_lookup_checks: int = 0
    exact_preparations_reused: int = 0

    def freeze(self) -> IntersectionStats:
        return IntersectionStats(**{item.name: getattr(self, item.name) for item in fields(self)})


@dataclass(frozen=True, slots=True)
class IntersectionResult:
    """Kernel-specific immutable result.

    ``components`` is non-empty only for CERTIFIED results. ``algorithm``
    distinguishes scalar specialization and generic section-root continuation.
    ``cover_complete`` is True only after the corresponding support proof completes.
    """

    status: IntersectionStatus
    reason: str
    stats: IntersectionStats
    first_source: SourceDefinition
    second_source: SourceDefinition
    components: tuple = ()
    cover_complete: bool = False
    unresolved: tuple = field(default=())   # ((first chart box, second chart box),) of the refused pair
    algorithm: str = "generic-cover"        # or "shared-affine-projection-scalar"
    structural: object = None               # StructuralCertificate of the fast path
    roots: tuple = ()                       # frozen endpoint identities (index == root id)
    boxes: tuple = ()                       # frozen regular scalar boxes
    descriptor_digest: str = ""             # integrity digest of structural/roots/boxes/components
    atlas: object = None                    # AtlasReport of the generic continuing cover and continuation
