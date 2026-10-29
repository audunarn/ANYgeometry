"""Certified continuation evaluation of one arc of a fast-path result.

Validation (correctness baseline, NOT optimized): the complete canonical scalar topology (roots, boxes,
components) is re-derived from the live, verified source definitions with the same pure engine and compared,
record by record, with the supplied result; digests are only a secondary integrity aid.  The time is reported
as ``validation_seconds``.

Parameterization: the station ``t`` is the affine combination ``s(t) = (1 - t) s_a + t s_b`` of the ACTUAL
slicing coordinates of the two endpoint roots.  Independent copies of the frozen isolating intervals are
refined; ``s(t)`` is enclosed by interval arithmetic from them (never replaced by interval centres) and the
dependent coordinate of the unique graph point is enclosed by interval Newton in the box's local Bernstein
form: existence by strict opposite signs of ``G`` on the two dependent edges over the whole ``s`` interval,
uniqueness/contraction by the strict sign of ``G_x`` (division by an interval excluding zero).  Station 0/1
enclose the endpoint root itself.  The reported precision target is the Euclidean diameter of the outward world
enclosure, ``<= min(tolerances.length, tolerances.surface_residual)`` decided exactly on the float endpoints;
if the target cannot be reached within ``4 * max_newton_iterations`` refinements (including representability
limits of the outward float box) ``NurbsSurfaceIntersectionRefused`` is raised.  The witness is approximate.
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from fractions import Fraction

from ..errors import GeometryError
from ..plane_nurbs import PlaneNurbsNotCertified
from ..tolerance import TolerancePolicy
from ._bernstein import derivative_grid, grid_ranges, subbox
from ._intervals import Interval, diameter_upper
from ._records import digest_of
from ._source import (IntersectionStatus, Limits, NurbsSurfaceIntersectionCancelled,
                      NurbsSurfaceIntersectionIntegrityError, NurbsSurfaceIntersectionRefused, WorkCounters,
                      isolate_callback)
from ._structural import derive_structural
from ._topology import Enclosures, freeze_section, run_scalar_section, world_box

__all__ = ["ArcEvaluation", "evaluate_arc", "result_digest"]


def evaluate_arcs(result, requests, *, first, second, tolerances,
                  qualification=None, cancellation_check=None):
    """Evaluate (component, arc, station) requests after ONE canonical validation.

    Validation and prepared original surfaces are local to this invocation;
    no reusable receipt, registry or digest alone supplies proof authority.
    Every point and final batch issuance check callbacks and live definitions.
    """
    if getattr(result,"status",None) is not IntersectionStatus.CERTIFIED:
        raise NurbsSurfaceIntersectionIntegrityError("arc evaluation requires a CERTIFIED result")
    if getattr(result,"algorithm",None)=="generic-atlas":
        from ._generic_evaluate import evaluate_generic_batch
        evaluator = evaluate_generic_batch
    else:
        from ._structural_evaluate import evaluate_structural_batch
        evaluator = evaluate_structural_batch
    return evaluator(result,requests,first=first,second=second,tolerances=tolerances,
                     qualification=qualification,cancellation_check=cancellation_check)


@dataclass(frozen=True, slots=True)
class ArcEvaluation:
    component_index: int
    arc_index: int
    station: Fraction
    native_box: tuple               # ((u0,u1),(v0,v1)) Fractions, certified to contain the point
    first_chart: tuple
    second_chart: tuple
    world_lo: tuple
    world_hi: tuple
    witness: tuple                  # approximate float point inside the world enclosure (not exact geometry)
    witness_is_exact: bool
    world_diameter: float           # outward Euclidean diameter bound of the world enclosure
    target: float
    bisections: int
    validation_seconds: float
    seconds: float


def result_digest(structural, roots, boxes, components) -> str:
    return digest_of(structural, roots, boxes, components)


def _horner(poly, t):
    total = Fraction(0)
    for c in reversed(poly):
        total = total * t + c
    return total


class _RootCopy:
    """Refinable copy of a frozen root (the frozen record is never touched)."""

    def __init__(self, root):
        self.root = root
        self.lo, self.hi = root.lo, root.hi
        a, b = root.edge
        if not (a < b and a <= self.lo <= self.hi <= b and root.axis in (0, 1) and root.poly):
            raise NurbsSurfaceIntersectionIntegrityError("malformed root descriptor")
        t_lo, t_hi = (self.lo - a) / (b - a), (self.hi - a) / (b - a)
        if self.lo == self.hi:
            if _horner(root.poly, t_lo) != 0:
                raise NurbsSurfaceIntersectionIntegrityError("exact root does not vanish on its edge polynomial")
        elif not _horner(root.poly, t_lo) * _horner(root.poly, t_hi) < 0:
            raise NurbsSurfaceIntersectionIntegrityError("root interval is not an isolating sign change")

    @property
    def exact(self):
        return self.lo == self.hi

    def refine(self):
        if self.exact:
            return
        a, b = self.root.edge
        t_lo, t_hi = (self.lo - a) / (b - a), (self.hi - a) / (b - a)
        middle = (t_lo + t_hi) / 2
        value = _horner(self.root.poly, middle)
        if value == 0:
            self.lo = self.hi = a + middle * (b - a)
        elif (value > 0) == (_horner(self.root.poly, t_lo) > 0):
            self.lo = a + middle * (b - a)
        else:
            self.hi = a + middle * (b - a)

    def projection(self, slicing) -> Interval:
        if self.root.axis == slicing:
            return Interval(self.root.value)
        return Interval(self.lo, self.hi)

    def native(self):
        if self.root.axis == 0:
            return ((self.root.value, self.root.value), (self.lo, self.hi))
        return ((self.lo, self.hi), (self.root.value, self.root.value))


def _diameter_squared(low, high) -> Fraction | None:
    if not all(math.isfinite(x) for x in (*low, *high)):
        return None
    return sum(((Fraction(h) - Fraction(l)) ** 2 for l, h in zip(low, high)), Fraction(0))


def _snap(interval: Interval, bits: int = 96) -> Interval:
    scale = 1 << bits
    lo = Fraction(math.floor(interval.lo * scale), scale)
    hi = Fraction(-math.floor(-interval.hi * scale), scale)
    return Interval(lo, hi)


def _grid(box):
    """Scalar Bernstein grid as 1-tuples indexed [slicing][dependent] (box-local parameters)."""
    c = box.coefficients
    if box.kind == "v":
        return tuple(tuple((x,) for x in row) for row in c)
    return tuple(tuple((c[i][j],) for i in range(len(c))) for j in range(len(c[0])))


def _enclose_dependent(grid, d_grid, sigma: Interval):
    """Local dependent interval of the unique graph over every local ``sigma``, or ``None`` if unproven."""
    low = grid_ranges(subbox(grid, sigma.lo, sigma.hi, 0, 0))[0]
    high = grid_ranges(subbox(grid, sigma.lo, sigma.hi, 1, 1))[0]
    if not ((low.hi < 0 and high.lo > 0) or (low.lo > 0 and high.hi < 0)):
        return None
    x = Interval(0, 1)
    for _ in range(12):
        m = x.mid
        g_m = grid_ranges(subbox(grid, sigma.lo, sigma.hi, m, m))[0]
        g_x = grid_ranges(subbox(d_grid, sigma.lo, sigma.hi, x.lo, x.hi))[0]
        if not g_x.excludes_zero():
            return None
        reciprocal = Interval(1 / g_x.hi, 1 / g_x.lo)
        newton = Interval(m) - g_m * reciprocal
        if not x.intersects(newton):
            raise NurbsSurfaceIntersectionIntegrityError("interval Newton produced an empty enclosure")
        new = _snap(Interval(max(x.lo, newton.lo), min(x.hi, newton.hi)))
        new = Interval(max(Fraction(0), new.lo), min(Fraction(1), new.hi))
        if new == x:
            break
        x = new
    return x


def evaluate_arc(result, component_index, arc_index, station, *, first, second, tolerances,
                 qualification=None, cancellation_check=None) -> ArcEvaluation:
    """Evaluate one source-bound arc with a fresh canonical validation."""
    return evaluate_arcs(result,((component_index,arc_index,station),),first=first,second=second,
                         tolerances=tolerances,qualification=qualification,
                         cancellation_check=cancellation_check)[0]
