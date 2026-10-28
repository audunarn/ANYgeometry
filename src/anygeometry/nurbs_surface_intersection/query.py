"""Experimental NURBS x NURBS certified intersection query.

Uses an exact scalar specialization or a complete generic product cover with
oriented section-root continuation to issue ordered open/closed components.
DISJOINT requires complete exclusion; unresolved regularity, incidence or work
limits return UNCLASSIFIED rather than an empty or partial successful result.
"""
from __future__ import annotations

import dataclasses
import math
import time
from fractions import Fraction

from ..errors import GeometryError
from ..tolerance import TolerancePolicy
from ._atlas import run_atlas
from ..plane_nurbs import PlaneNurbsNotCertified
from ._intervals import float_up
from ._records import digest_of, RationalProjectionCertificate
from ._structural import derive_structural
from ._topology import Enclosures, freeze_section, run_scalar_section
from ._source import (IntersectionResult, IntersectionStatus, Limits, NurbsSurfaceIntersectionCancelled,
                      NurbsSurfaceIntersectionIntegrityError, SourceDefinition, WorkCounters, isolate_callback)

__all__ = ["query_nurbs_surface_intersection"]


def _sqrt_up(value: Fraction) -> Fraction:
    """Exact rational upper bound of sqrt(value) (value >= 0)."""
    if value == 0:
        return Fraction(0)
    shift = 64
    root = math.isqrt((value.numerator << (2 * shift)) // value.denominator) + 1
    return Fraction(root, 1 << shift)


def _extent(prepared: PreparedSurface) -> float:
    lower, upper = prepared.hull(((0.0, 1.0), (0.0, 1.0)))
    squared = sum(((b - a) ** 2 for a, b in zip(lower, upper)), Fraction(0))
    made = float_up(_sqrt_up(squared))
    if not math.isfinite(made):
        raise GeometryError("surface hull extent is not representable in float64; query refused")
    return made


ALGORITHM = "shared-affine-projection-scalar"


def _fast_path(first, second, source_a, source_b, certificate, tolerances, limits, stats, tick, recheck, started):
    """Complete scalar topology for the exactly certified shared-affine-projection subset."""

    algorithm = ("shared-separable-rational-projection-scalar"
                 if type(certificate) is RationalProjectionCertificate else ALGORITHM)
    def issue(status, reason, **extra):
        recheck()                              # source recheck immediately before issuance
        tick()                                 # final cancellation poll
        stats.total_seconds = time.perf_counter() - started
        return IntersectionResult(status, reason, stats.freeze(), source_a, source_b, algorithm=algorithm,
                                  structural=certificate, **extra)

    if certificate.domain is None:
        return issue(IntersectionStatus.DISJOINT,
                     "selected native rectangles do not meet and the exact chart correspondence is u=s, v=t",
                     cover_complete=True)
    topology_at = time.perf_counter()
    try:
        section = run_scalar_section(source_a, certificate, limits, tick, stats)
        if section.coincident:
            raise PlaneNurbsNotCertified("the scalar field vanishes identically (coincident surfaces)")
        enclosures = Enclosures(source_a.rebuild(), source_b.rebuild())
        roots, boxes, components = freeze_section(section, enclosures, limits.max_components,
                                                  limits.max_trace_segments)
    except PlaneNurbsNotCertified as error:
        stats.topology_seconds = time.perf_counter() - topology_at
        return issue(IntersectionStatus.UNCLASSIFIED,
                     f"scalar topology refused: {error}; complete component accounting is unavailable")
    stats.topology_seconds = time.perf_counter() - topology_at
    stats.roots, stats.boxes = len(roots), len(boxes)
    stats.arcs = sum(len(c.arcs) for c in components)
    stats.components = len(components)
    if len(components) > limits.max_components or stats.arcs > limits.max_trace_segments:
        return issue(IntersectionStatus.UNCLASSIFIED, "component or trace-segment limit exceeded")
    digest = digest_of(certificate, roots, boxes, components)
    if not components:
        return issue(IntersectionStatus.DISJOINT, "the scalar field has a strict sign on every box",
                     cover_complete=True, roots=roots, boxes=boxes, descriptor_digest=digest)
    return issue(IntersectionStatus.CERTIFIED,
                  "complete scalar root/arc/component accounting for the certified shared projection subset",
                 components=components, cover_complete=True, roots=roots, boxes=boxes, descriptor_digest=digest)


def query_nurbs_surface_intersection(first, second, *, tolerances, qualification=None, cancellation_check=None):
    started = time.perf_counter()
    if not isinstance(tolerances, TolerancePolicy):
        raise GeometryError("tolerances must be a TolerancePolicy")
    if cancellation_check is not None and not callable(cancellation_check):
        raise GeometryError("cancellation_check must be callable")
    # values that affect the proof are bound once here; the callback runs in an isolated entry-context copy
    tolerances = TolerancePolicy(**{f.name: getattr(tolerances, f.name) for f in dataclasses.fields(tolerances)})
    call = isolate_callback(cancellation_check)
    limits = Limits.read(qualification)
    source_a, source_b = SourceDefinition.capture(first), SourceDefinition.capture(second)
    stats = WorkCounters()

    def recheck():
        stats.integrity_checks += 1
        if not (source_a.matches(first) and source_b.matches(second)):
            raise NurbsSurfaceIntersectionIntegrityError("an input surface changed during the query")

    def tick():
        if call is not None:
            stats.cancellation_checks += 1
            if call():         # truthy cancels; exceptions propagate unchanged
                raise NurbsSurfaceIntersectionCancelled("NURBS surface intersection cancelled")
            recheck()                          # callback boundary: live inputs must be unchanged

    tick()                                     # before any expensive exact work
    from ._dependencies import verify_dependencies
    verify_dependencies()
    structural_at = time.perf_counter()
    certificate, structural_reason = derive_structural(source_a, source_b, tick)
    stats.structural_seconds = time.perf_counter() - structural_at
    if certificate is not None:
        return _fast_path(first, second, source_a, source_b, certificate, tolerances, limits, stats, tick,
                          recheck, started)
    # Generic route: complete support cover followed by certified section-root continuation.
    report = run_atlas(source_a, source_b, tolerances, limits, stats, tick)
    recheck()                                  # before issuance
    tick()
    stats.total_seconds = time.perf_counter() - started
    if report.complete_local_cover and report.components:
        digest = digest_of(report.witnesses,report.roots,(),report.components)
        return IntersectionResult(IntersectionStatus.CERTIFIED,report.reason,stats.freeze(),source_a,source_b,
                                  report.components,True,(),algorithm="generic-atlas",roots=report.roots,
                                  descriptor_digest=digest,atlas=report)
    if report.complete_local_cover and report.covered == 0:
        return IntersectionResult(IntersectionStatus.DISJOINT, "every pair node excluded by exact certificates",
                                  stats.freeze(), source_a, source_b, (), True, (), atlas=report)
    return IntersectionResult(IntersectionStatus.UNCLASSIFIED, report.reason, stats.freeze(), source_a, source_b,
                              (), False, (), atlas=report)
