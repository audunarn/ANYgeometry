"""Bounded conservative UV parameter-region matching through public enclosures.

``match_curve_parameter_regions`` finds a cover of parameters whose curve point
could lie within ``uv_tolerance`` of ANY point of the target box. Each returned
component also has one representative certified within tolerance of EVERY point
of that box. This is neither a unique inverse nor recovery of a historical domain.

The curve protocol is a whole ``domain == (0, 1)``, ``subcurve(a,b).chart_bounds()``
returning ``(low[2], high[2])``, and ``chart_enclosure(t)`` returning two coordinate
intervals. Generic protocol point intervals use normalized UV. PlaneNurbsCurve's
public point enclosure instead uses knot coordinates; this module transports it
to normalized UV exactly using its public surface ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real, Rational
from contextvars import ContextVar
from contextlib import contextmanager

from .errors import GeometryError

__all__ = [
    'CurveParameterMatchError', 'CurveParameterMatchInconclusive',
    'ParameterRegion', 'ParameterCandidate', 'ParameterRegionMatch',
    'match_curve_parameter_regions',
    'bounded_section_work',
]

_ACTIVE_WORK = ContextVar('anygeometry_parameter_matching_work',default=None)


def _cooperative_tick():
    """Matching-only hook; ordinary owner/qualification execution is unchanged."""
    work=_ACTIVE_WORK.get()
    if work is not None:work.charge()


class CurveParameterMatchError(GeometryError):
    """No partial result is returned; ``reason`` identifies the refusal."""
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class CurveParameterMatchInconclusive(CurveParameterMatchError):
    """Possible regions remain but no representative is certified in one."""
    def __init__(self, message: str):
        super().__init__('inconclusive', message)


@dataclass(frozen=True, slots=True)
class ParameterRegion:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ParameterCandidate:
    parameter: float
    chart_box: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    residual_squared: Fraction


@dataclass(frozen=True, slots=True)
class ParameterRegionMatch:
    regions: tuple[ParameterRegion, ...]
    candidates: tuple[ParameterCandidate, ...]
    work_used: int


def _failure(reason, message):
    raise CurveParameterMatchError(reason, message)


def _number(value, reason='invalid_input'):
    if isinstance(value, bool) or not isinstance(value, Real):
        _failure(reason, 'coordinates and tolerances must be finite real numbers')
    if isinstance(value, Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    if not math.isfinite(value):
        _failure(reason, 'coordinates and tolerances must be finite real numbers')
    try:
        numerator, denominator = value.as_integer_ratio()
    except (AttributeError, TypeError, ValueError, OverflowError):
        _failure(reason, 'real coordinates must provide an exact integer ratio')
    return Fraction(int(numerator), int(denominator))


def _box(value, reason='invalid_input'):
    try:
        lower, upper = value
        if len(lower) != 2 or len(upper) != 2:
            _failure(reason, 'UV boxes need two lower and two upper coordinates')
        lower = tuple(_number(v, reason) for v in lower)
        upper = tuple(_number(v, reason) for v in upper)
    except (TypeError, ValueError) as error:
        if isinstance(error, CurveParameterMatchError):
            raise
        _failure(reason, 'UV boxes need two lower and two upper coordinates')
    if any(a > b for a, b in zip(lower, upper)):
        _failure(reason, 'UV box bounds are reversed')
    return lower, upper


def _distances(first, second):
    """Exact minimum and maximum squared distances between two closed boxes."""
    low, high = first
    other_low, other_high = second
    minimum = sum(max(a-d, c-b, Fraction(0))**2
                  for a, b, c, d in zip(low, high, other_low, other_high))
    maximum = sum(max(abs(a-d), abs(b-c))**2
                  for a, b, c, d in zip(low, high, other_low, other_high))
    return minimum, maximum


class _Budget:
    def __init__(self, limit, checkpoint, max_depth=48):
        self.limit, self.checkpoint, self.used = limit, checkpoint, 0
        self.max_depth=max_depth
        self.callback_error = None
    @property
    def work_used(self):return self.used

    def charge(self):
        if self.used >= self.limit:
            _failure('budget_exhausted', 'parameter matching exceeded max_boxes')
        self.used += 1
        if self.checkpoint is not None:
            try:
                self.checkpoint()
            except BaseException as error:
                self.callback_error = error
                raise

    def query(self, callback):
        """Type owner refusals without translating the caller's cancellation."""
        try:
            return callback()
        except CurveParameterMatchError:
            raise
        except GeometryError as error:
            if error is self.callback_error:
                raise
            raise CurveParameterMatchError('enclosure_refused', f'public owner enclosure refused: {error}') from error


class _SubBudget(_Budget):
    def __init__(self,parent,limit,max_depth):
        self.parent,self.start,self.limit=parent,parent.used,limit
        self.max_depth=min(parent.max_depth,max_depth)
    @property
    def used(self):return self.parent.used-self.start
    @property
    def callback_error(self):return self.parent.callback_error
    def charge(self):
        if self.used>=self.limit:_failure('budget_exhausted','parameter matching exceeded max_boxes')
        self.parent.charge()


class _Bridge:
    def __init__(self,work):self.work=work;self.subdivisions=0
    @property
    def max_depth(self):return self.work.max_depth
    def take(self):self.work.charge();return True


@contextmanager
def _activate(work):
    from .plane_nurbs import CURRENT_WORK
    current=CURRENT_WORK.get();parent=_ACTIVE_WORK.get()
    if current is not None and (not isinstance(current,_Bridge) or current.work is not parent):
        _failure('nested_context','foreign owner work context cannot be overridden')
    active_token=_ACTIVE_WORK.set(work);section_token=CURRENT_WORK.set(_Bridge(work))
    try:yield
    finally:
        CURRENT_WORK.reset(section_token);_ACTIVE_WORK.reset(active_token)


@contextmanager
def bounded_section_work(*,max_work=10000,max_depth=48,checkpoint=None):
    """Cooperatively bound section construction AND public enclosure queries.

    Yields a live ledger with ``work_used``. Counts matching records, spatial
    boxes and scoped exact refinement/arc steps, each with a checkpoint. Existing
    qualification contexts and explicit nested contexts refuse rather than loosen
    budgets. A matcher inside this context omits its own checkpoint and uses a
    subbudget of this same ledger. Tokens always reset, including cancellation.
    This is not a wall-clock or rational-arithmetic bit-complexity guarantee.
    Foreign generic protocols must themselves cooperate for inner cancellation.
    """
    if isinstance(max_work,bool) or not isinstance(max_work,int) or max_work<=0:
        _failure('invalid_input','max_work must be a positive integer')
    if isinstance(max_depth,bool) or not isinstance(max_depth,int) or max_depth<0:
        _failure('invalid_input','max_depth must be a nonnegative integer')
    if checkpoint is not None and not callable(checkpoint):_failure('invalid_input','checkpoint must be callable')
    if _ACTIVE_WORK.get() is not None:_failure('nested_context','explicit nested work contexts are unsupported')
    work=_Budget(max_work,checkpoint,max_depth)
    with _activate(work):yield work


def _point_box(master, parameter):
    intervals = master.chart_enclosure(parameter)
    try:
        (a, b), (c, d) = intervals
    except (TypeError, ValueError):
        _failure('invalid_enclosure', 'point enclosure must contain two intervals')
    box = _box(((a, c), (b, d)), 'invalid_enclosure')
    from .support_curves import PlaneNurbsCurve
    if isinstance(master, PlaneNurbsCurve):
        ranges = (master.surface.u_range, master.surface.v_range)
        bounds = []
        for low, high, span in zip(*box, ranges):
            first, last = map(Fraction, span)
            if first == last:
                _failure('invalid_enclosure', 'section chart range has zero span')
            ends=((low-first)/(last-first), (high-first)/(last-first))
            bounds.append((min(ends),max(ends)))
        box = (tuple(p[0] for p in bounds), tuple(p[1] for p in bounds))
    return box


def match_curve_parameter_regions(master, target_box, uv_tolerance, *,
                                  max_boxes=10000, max_depth=48,
                                  param_resolution=1e-12, checkpoint=None):
    """Return immutable complete POSSIBLE-region cover and certified candidates.

    Only an exact rational minimum-distance proof rejects an interval. An exact
    maximum-distance proof admits a whole interval; otherwise subdivision runs
    until ``param_resolution``. Adjacent possible/admitted intervals merge, but
    the ends 0 and 1 never merge across the periodic seam. Finite resolution may
    conservatively join true regions separated by a gap below that resolution.

    ``max_boxes`` bounds enclosure queries AND retained work/result records:
    each query and each newly stored record is charged before it happens. Thus
    queries <= work_used <= max_boxes, with bounded memory. Checkpoint runs for
    every charged unit; any exception it raises propagates unchanged. Exhaustion,
    float subdivision stagnation, or unrepresented possible components never
    return partial success. Owner GeometryError query refusals become typed
    enclosure_refused errors; caller checkpoint exceptions propagate unchanged.
    """
    try:
        domain = tuple(master.domain)
    except (AttributeError, TypeError):
        _failure('invalid_input', 'matching requires a whole curve domain (0, 1)')
    if domain != (0, 1):
        _failure('invalid_input', 'matching requires a whole curve domain (0, 1)')
    if not all(callable(getattr(master, name, None)) for name in ('subcurve', 'chart_enclosure')):
        _failure('invalid_input', 'curve lacks the public enclosure protocol')
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or max_boxes <= 0:
        _failure('invalid_input', 'max_boxes must be a positive integer')
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        _failure('invalid_input', 'max_depth must be a nonnegative integer')
    tolerance, resolution = _number(uv_tolerance), _number(param_resolution)
    if tolerance <= 0 or resolution <= 0:
        _failure('invalid_input', 'UV tolerance and parameter resolution must be positive')
    if checkpoint is not None and not callable(checkpoint):
        _failure('invalid_input', 'checkpoint must be callable')
    target = _box(target_box)
    squared = tolerance**2
    parent=_ACTIVE_WORK.get()
    if parent is not None and checkpoint is not None:
        _failure('nested_context','a nested matcher uses the enclosing checkpoint')
    work = _Budget(max_boxes, checkpoint,max_depth) if parent is None else _SubBudget(parent,max_boxes,max_depth)
    with _activate(work):
        return _match_regions(master,target,squared,resolution,work.max_depth,work)


def _match_regions(master,target,squared,resolution,max_depth,work):
    work.charge()
    pending = [(0., 1., 0)]
    leaves = []
    while pending:
        lower, upper, depth = pending.pop()
        work.charge()
        box = work.query(lambda: _box(master.subcurve(lower, upper).chart_bounds(), 'invalid_enclosure'))
        minimum, maximum = _distances(box, target)
        if minimum > squared:
            continue
        inside = maximum <= squared
        if inside or Fraction(upper)-Fraction(lower) <= resolution:
            work.charge()
            leaves.append((lower, upper, inside))
            continue
        if depth >= max_depth:
            _failure('depth_exhausted', 'possible interval exceeded max_depth')
        middle = lower+(upper-lower)/2
        if not lower < middle < upper:
            _failure('precision_exhausted', 'parameter subdivision has no float progress')
        work.charge()
        pending.append((middle, upper, depth+1))
        work.charge()
        pending.append((lower, middle, depth+1))
    # Left-first traversal is already sorted; no unbounded secondary sort/search.
    components = []
    for lower, upper, inside in leaves:
        seed = lower+(upper-lower)/2 if inside else None
        work.charge()
        if components and lower <= components[-1][1]:
            a, b, previous = components[-1]
            components[-1] = (a, max(b, upper), previous if previous is not None else seed)
        else:
            components.append((lower, upper, seed))
    regions, candidates = [], []
    for lower, upper, seed in components:
        proposals = dict.fromkeys(p for p in (seed, lower+(upper-lower)/2, lower, upper) if p is not None)
        candidate = None
        for parameter in proposals:
            work.charge()
            box = work.query(lambda: _point_box(master, parameter))
            residual = _distances(box, target)[1]
            if residual <= squared:
                candidate = ParameterCandidate(parameter, box, residual)
                break
        if candidate is None:
            raise CurveParameterMatchInconclusive('possible parameter component has no certified representative')
        work.charge()
        regions.append(ParameterRegion(lower, upper))
        work.charge()
        candidates.append(candidate)
    work.charge()
    return ParameterRegionMatch(tuple(regions), tuple(candidates), work.used)
