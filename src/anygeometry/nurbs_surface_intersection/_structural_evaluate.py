"""Canonical-once evaluation of exact structural scalar intersections."""
from fractions import Fraction
import dataclasses
import math
import time

from ..errors import GeometryError
from ..tolerance import TolerancePolicy
from ._evaluate import ArcEvaluation, _RootCopy, _grid, _enclose_dependent, _diameter_squared
from ._bernstein import derivative_grid
from ._intervals import Interval, diameter_upper
from ._topology import Enclosures, world_box
from ._source import (IntersectionResult, IntersectionStatus, SourceDefinition, Limits, isolate_callback,
    NurbsSurfaceIntersectionIntegrityError, NurbsSurfaceIntersectionCancelled, NurbsSurfaceIntersectionRefused)

ALGORITHMS = ("shared-affine-projection-scalar", "shared-separable-rational-projection-scalar")


def evaluate_structural_batch(result, requests, *, first, second, tolerances,
                              qualification=None, cancellation_check=None):
    from .query import query_nurbs_surface_intersection
    from ._dependencies import verify_dependencies
    verify_dependencies()
    started = time.perf_counter()
    if not isinstance(tolerances, TolerancePolicy):
        raise GeometryError("tolerances must be a TolerancePolicy")
    if cancellation_check is not None and not callable(cancellation_check):
        raise GeometryError("cancellation_check must be callable")
    if (type(result) is not IntersectionResult or type(result.first_source) is not SourceDefinition
            or type(result.second_source) is not SourceDefinition):
        raise NurbsSurfaceIntersectionIntegrityError("structural evaluation requires canonical result/source types")
    source_a,source_b = SourceDefinition.capture(first),SourceDefinition.capture(second)
    if result.first_source != source_a or result.second_source != source_b:
        raise NurbsSurfaceIntersectionIntegrityError("structural result differs from actual source definitions")
    supplied = (result.status,result.algorithm,result.structural,result.roots,result.boxes,
                result.components,result.descriptor_digest)
    tolerances = TolerancePolicy(**{f.name:getattr(tolerances,f.name) for f in dataclasses.fields(tolerances)})
    limits = Limits.read(qualification)
    callback = isolate_callback(cancellation_check)
    def live():
        if not source_a.matches(first) or not source_b.matches(second):
            raise NurbsSurfaceIntersectionIntegrityError("an input surface differs from the captured structural source")
    def tick():
        if callback is not None and callback():
            raise NurbsSurfaceIntersectionCancelled("structural arc evaluation cancelled")
        live()
    tick()
    canonical = query_nurbs_surface_intersection(first,second,tolerances=tolerances,
                                               qualification=limits,cancellation_check=tick)
    trusted = (canonical.status,canonical.algorithm,canonical.structural,canonical.roots,canonical.boxes,
               canonical.components,canonical.descriptor_digest)
    if (canonical.status is not IntersectionStatus.CERTIFIED or canonical.algorithm not in ALGORITHMS
            or supplied != trusted):
        raise NurbsSurfaceIntersectionIntegrityError("structural descriptors differ from the canonical re-derivation")
    # No caller result/source/descriptor reads beyond this comparison. Iterator
    # and callback mutation cannot replace any invocation-local proof authority.
    return _evaluate_validated_structural(canonical,requests,tolerances=tolerances,limits=limits,
        tick=tick,live=live,started=started,source_a=source_a,source_b=source_b)


def _evaluate_validated_structural(canonical, requests, *, tolerances, limits, tick, live, started, source_a, source_b):
    """Internal invocation-local continuation; canonical must be freshly rederived."""
    roots,boxes,components = canonical.roots,canonical.boxes,canonical.components
    validation_seconds = time.perf_counter()-started
    surface_a,surface_b = source_a.rebuild(),source_b.rebuild()
    enclosures = Enclosures(surface_a,surface_b)
    cap = 4*limits.max_newton_iterations
    target = min(tolerances.length,tolerances.surface_residual)
    target_squared = Fraction(target)**2
    root_cache = {}
    def root_copy(index):
        if index not in root_cache:
            root_cache[index] = _RootCopy(roots[index])
        return root_cache[index]
    def evaluate_one(component_index,arc_index,station):
        try:
            station = Fraction(station)
        except (TypeError,ValueError,OverflowError) as error:
            raise GeometryError("station must be a finite real number") from error
        if not 0 <= station <= 1:
            raise GeometryError("station must lie in [0,1]")
        try:
            arc = components[component_index].arcs[arc_index]
            if component_index < 0 or arc_index < 0:
                raise IndexError
        except (IndexError, TypeError) as error:
            raise GeometryError("no such component or arc") from error
        box = boxes[arc.box_index]
        copies = (root_copy(arc.first_root), root_copy(arc.second_root))
        steps = 0
    
        def reached(low, high) -> bool:
            squared = _diameter_squared(low, high)
            return squared is not None and squared <= target_squared
    
        def refuse(message):
            raise NurbsSurfaceIntersectionRefused(message)
    
        if station in (0, 1):
            root = copies[0 if station == 0 else 1]
            while True:
                native = root.native()
                charts, low, high = world_box(enclosures, native)
                if reached(low, high):
                    break
                if steps >= cap or root.exact:
                    refuse("endpoint enclosure cannot reach the length/residual target (work or representability)")
                tick()
                root.refine()
                steps += 1
        else:
            slicing = arc.slicing_axis
            grid = _grid(box)
            d_grid = derivative_grid(grid, 1)
            s_range, x_range = (box.u_range, box.v_range) if box.kind == "v" else (box.v_range, box.u_range)
            previous = None
            while True:
                ia, ib = (c.projection(slicing) for c in copies)
                s_int = ia.scale(1 - station) + ib.scale(station)          # actual endpoints, interval arithmetic
                lo, hi = max(s_int.lo, s_range[0]), min(s_int.hi, s_range[1])
                native = None
                if lo <= hi:
                    width = s_range[1] - s_range[0]
                    sigma = Interval((lo - s_range[0]) / width, (hi - s_range[0]) / width)
                    x = _enclose_dependent(grid, d_grid, sigma)
                    if x is not None:
                        x_lo = x_range[0] + x.lo * (x_range[1] - x_range[0])
                        x_hi = x_range[0] + x.hi * (x_range[1] - x_range[0])
                        native = ((lo, hi), (x_lo, x_hi)) if box.kind == "v" else ((x_lo, x_hi), (lo, hi))
                        charts, low, high = world_box(enclosures, native)
                        if reached(low, high):
                            break
                state = (native, copies[0].lo, copies[0].hi, copies[1].lo, copies[1].hi)
                if steps >= cap or state == previous:
                    refuse("the point enclosure cannot reach the length/residual target within the work bound "
                           "(or the station is too close to an endpoint to separate)")
                previous = state
                tick()
                for c in copies:
                    c.refine()
                steps += 1
        chart_a = tuple((a + b) / 2 for a, b in charts[0])
        witness = tuple(float(v) for v in surface_a.evaluate(*chart_a))
        witness = tuple(min(max(w, l), h) for w, l, h in zip(witness, low, high))
        squared = _diameter_squared(low, high)
        diameter = diameter_upper(squared) if squared is not None else math.inf
        live()
        tick()                                  # the tail ends with a callback boundary (cancel, then live check)
        live()                                  # also without a callback; nothing below calls user code
        return ArcEvaluation(component_index, arc_index, station, native, charts[0], charts[1], low, high, witness,
                             False, diameter, target, steps, validation_seconds, time.perf_counter() - started)
    output=[]
    for request in requests:
        tick()
        if len(output) >= limits.max_trace_segments:
            raise NurbsSurfaceIntersectionRefused("evaluation batch point limit exhausted")
        output.append(evaluate_one(*request))
    tick()
    live()
    return tuple(output)
