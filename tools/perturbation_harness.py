"""Perturbation harness for the robust-predicate programme (slice R0, instrument 2).

    python tools/perturbation_harness.py [--case NAME ...] [--op NAME] [--variant NAME ...]
                                         [--sites SITE,...] [--seed N] [--heavy]
                                         [--json PATH] [--markdown PATH]
    python tools/perturbation_harness.py --merge a.json b.json --json all.json --markdown all.md

For every case of ``tools/adversarial_corpus.py`` (the adversarial cases plus the
G0 workloads) and every variant, the harness runs the case's engine operations
on a freshly built model and records a float-free outcome: the topology digest
of the resulting document (as ``benchmarks/generality/run.py split_document``),
the ``IntersectionResult`` kind of queries, point-membership decisions, or the
raised error's type and (number-masked) message. Each outcome is compared with
the unperturbed run; reversible operations are also run with reversed operand
order and compared with the unperturbed forward run.

Variants

* ``none``: no wrapping at all; a repeat of the reference run (determinism).
* ``outputs/<pattern>``: every chart inversion (``local_uv`` of all built-in
  supports, ``closest_uv``) and every derived ``freeze_edge`` product (the
  ``EllipticArc`` built from an arc's computed frame) is moved by one ulp.
* ``outputs+evaluations/<pattern>``: in addition every surface/curve
  evaluation and derivative at the capability boundary (``evaluate``,
  ``derivative`` of LinePath / BezierPath / EllipticArc, surface ``evaluate``,
  ``_evaluate_surface_many``, ``_surface_derivatives_many``).
* ``route/numpy``, ``route/scalar``, ``route/fma``: the same formulas through
  an alternative arithmetic route. ``numpy`` forces the product's vectorized
  NumPy formula also where the product takes a scalar fast path
  (``LinePath.evaluate``); ``scalar`` uses Python ``math`` scalars
  (``math.atan2`` instead of ``np.arctan2``, closed-form Cramer on the normal
  equations instead of ``lstsq``, ``math.cos``/``math.sin``); ``fma`` is the
  scalar route with fused multiply-add (``math.fma`` on Python >= 3.13, an exact rational
  equivalent before) in dot
  products, determinants and affine combinations. Route sites:
  ``ROUTE_SITES`` (plane/cylinder/cone chart inversion, plane/cylinder
  evaluation, line and ellipse evaluation).
* ``reverse``: unperturbed, operands reversed (reversible operations only).

Patterns: ``+1`` (every float one ulp up), ``-1`` (one ulp down), ``random``
(per float a seeded random sign). Perturbation applies at the outermost
wrapped call only (a wrapped call made inside another wrapped call is passed
through), so a consumer sees exactly one ulp per value. Non-finite values are
left unchanged.

When a combined variant flips, the harness re-runs the operation with each
site that was actually called enabled alone, and lists the sites that flip on
their own together with their R.1 inventory ids (attribution).

The harness wraps engine functions in this process only, by attribute
assignment, and restores them afterwards; product code is never changed.
It is a falsification instrument: a flip is a defect with a reproducer; no flip
proves nothing about inputs or patterns that were not run.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import hashlib
import json
import math
from fractions import Fraction
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import anygeometry  # noqa: E402
from anygeometry import arrangement_geometry, exact_curves, surfaces  # noqa: E402

PATTERNS = ("+1", "-1", "random")
MODES = ("outputs", "outputs+evaluations")
ROUTES = ("numpy", "scalar", "fma")


# =========================================================================== ulp nudging

class Perturber:
    """Deterministic one-ulp perturbation of every float in a value."""

    def __init__(self, pattern: str, seed: int):
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern {pattern!r}")
        self.pattern = pattern
        self.rng = np.random.default_rng(seed)

    def _directions(self, size):
        if self.pattern == "+1":
            return np.full(size, np.inf)
        if self.pattern == "-1":
            return np.full(size, -np.inf)
        return np.where(self.rng.integers(0, 2, size) == 1, np.inf, -np.inf)

    def nudge(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype != np.float64 or value.size == 0:
                return value
            target = self._directions(value.size).reshape(value.shape)
            moved = np.nextafter(value, target)
            return np.where(np.isfinite(value), moved, value)
        if isinstance(value, (float, np.floating)) and not isinstance(value, bool):
            if not math.isfinite(value):
                return value
            moved = math.nextafter(float(value), float(self._directions(1)[0]))
            return np.float64(moved) if isinstance(value, np.float64) else moved
        if isinstance(value, tuple):
            return tuple(self.nudge(item) for item in value)
        if isinstance(value, list):
            return [self.nudge(item) for item in value]
        return value


def ulp_distance(a: float, b: float) -> int:
    """Number of representable doubles between ``a`` and ``b`` (same sign scale)."""
    ia = int(np.float64(a).view(np.int64))
    ib = int(np.float64(b).view(np.int64))
    # Map the sign-magnitude ordering onto a monotone integer line.
    ia = ia if ia >= 0 else -(ia & 0x7FFFFFFFFFFFFFFF)
    ib = ib if ib >= 0 else -(ib & 0x7FFFFFFFFFFFFFFF)
    return abs(ia - ib)


# =========================================================================== sites

@dataclasses.dataclass(frozen=True)
class Site:
    name: str
    group: str            # "output" | "evaluation" | "stored"
    owner: object         # class or module defining the attribute
    attribute: str
    inventory: tuple[str, ...]
    kind: str = "outputs"  # "outputs": perturb return value; "freeze": perturb a frozen definition
    freeze_type: type | None = None


_C = surfaces
_A = arrangement_geometry
_E = exact_curves

SITES: dict[str, Site] = {site.name: site for site in (
    # --- chart inversion (outputs)
    Site("Plane.local_uv", "output", _C.Plane, "local_uv", ("A1", "D1", "D4", "E4")),
    Site("Cylinder.local_uv", "output", _C.Cylinder, "local_uv", ("A2", "A3", "A4", "D1", "D7")),
    Site("Cone.local_uv", "output", _C.Cone, "local_uv", ("A2", "A4", "F2", "F3")),
    Site("ExtrudedSurface.local_uv", "output", _C.ExtrudedSurface, "local_uv", ("A6", "E1")),
    Site("ExtrudedSurface.local_uv_many", "output", _C.ExtrudedSurface, "local_uv_many", ("A6", "E1")),
    Site("RuledSurface.local_uv", "output", _C.RuledSurface, "local_uv", ("A6",)),
    Site("CoonsSurface.local_uv", "output", _C.CoonsSurface, "local_uv", ("A6",)),
    Site("closest_uv", "output", _C, "closest_uv", ("A6",)),
    # --- freeze_edge products: the arc definition is computed from three vertices
    Site("freeze_edge.arc", "output", _A, "freeze_edge", ("B2", "B11", "C1", "C4"), kind="freeze",
         freeze_type=_E.EllipticArc),
    # --- evaluations and derivatives at the capability boundary
    Site("Plane.evaluate", "evaluation", _C.Plane, "evaluate", ("A1", "D1", "D6")),
    Site("Cylinder.evaluate", "evaluation", _C.Cylinder, "evaluate", ("A2", "E2")),
    Site("Cone.evaluate", "evaluation", _C.Cone, "evaluate", ("A2", "F2")),
    Site("ExtrudedSurface.evaluate", "evaluation", _C.ExtrudedSurface, "evaluate", ("E1",)),
    Site("_evaluate_surface_many", "evaluation", _C, "_evaluate_surface_many", ("A6", "E5")),
    Site("_surface_derivatives_many", "evaluation", _C, "_surface_derivatives_many", ("A6", "D2", "E5")),
    Site("LinePath.evaluate", "evaluation", _A.LinePath, "evaluate", ("B1", "B3", "B11", "C1", "D1", "D6")),
    Site("LinePath.derivative", "evaluation", _A.LinePath, "derivative", ("D2", "D6")),
    Site("BezierPath.evaluate", "evaluation", _A.BezierPath, "evaluate", ("B10", "B11", "C1")),
    Site("BezierPath.derivative", "evaluation", _A.BezierPath, "derivative", ("D2", "D6")),
    Site("EllipticArc.evaluate", "evaluation", _E.EllipticArc, "evaluate", ("B2", "B5", "B11", "C1", "C3")),
    Site("EllipticArc.derivative", "evaluation", _E.EllipticArc, "derivative", ("D2", "D6")),
    # --- stored data (input perturbation; opt-in through --sites only)
    Site("freeze_edge.line", "stored", _A, "freeze_edge", ("B1", "C1"), kind="freeze",
         freeze_type=_A.LinePath),
    Site("freeze_edge.spline", "stored", _A, "freeze_edge", ("B10", "C1"), kind="freeze",
         freeze_type=_A.BezierPath),
)}


def sites_for_mode(mode: str) -> list[str]:
    groups = {"outputs": {"output"}, "outputs+evaluations": {"output", "evaluation"}}[mode]
    return [name for name, site in SITES.items() if site.group in groups]


# =========================================================================== patching

def _anygeometry_modules():
    return [module for name, module in sorted(sys.modules.items())
            if module is not None and (name == "anygeometry" or name.startswith("anygeometry."))]


class Patcher:
    """Attribute replacement with exact restore. Module-level functions are
    replaced in every ``anygeometry`` module that bound the same object."""

    def __init__(self):
        self._saved: list[tuple[object, str, object]] = []

    def replace(self, owner, attribute, factory):
        if isinstance(owner, type):
            original = owner.__dict__[attribute]
            self._saved.append((owner, attribute, original))
            setattr(owner, attribute, factory(original))
            return
        original = getattr(owner, attribute)
        replacement = factory(original)
        for module in _anygeometry_modules():
            if module.__dict__.get(attribute) is original:
                self._saved.append((module, attribute, original))
                setattr(module, attribute, replacement)

    def restore(self):
        while self._saved:
            owner, attribute, original = self._saved.pop()
            setattr(owner, attribute, original)


def clear_caches():
    """Empty every functools cache in anygeometry so no run sees another's values."""
    for module in _anygeometry_modules():
        for value in list(vars(module).values()):
            clear = getattr(value, "cache_clear", None)
            if callable(clear) and isinstance(value, functools._lru_cache_wrapper):
                clear()


class Perturbation:
    """Wrap the selected sites; count calls; perturb at the outermost call only."""

    def __init__(self, sites, pattern, seed):
        self.sites = list(sites)
        self.perturber = Perturber(pattern, seed)
        self.calls = collections.Counter()
        self.depth = 0
        self.patcher = Patcher()

    def _wrap_outputs(self, name, original):
        harness = self

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            harness.calls[name] += 1
            harness.depth += 1
            try:
                value = original(*args, **kwargs)
            finally:
                harness.depth -= 1
            return value if harness.depth else harness.perturber.nudge(value)
        return wrapped

    def _wrap_freeze(self, names, original):
        harness = self
        kinds = {SITES[name].freeze_type: name for name in names}

        @functools.wraps(original)
        def wrapped(model, edge_id):
            value = original(model, edge_id)
            name = kinds.get(type(value))
            if name is None:
                return value
            harness.calls[name] += 1
            if harness.depth:
                return value
            return perturb_definition(value, harness.perturber)
        return wrapped

    def __enter__(self):
        freeze = [name for name in self.sites if SITES[name].kind == "freeze"]
        for name in self.sites:
            site = SITES[name]
            if site.kind == "outputs":
                self.patcher.replace(site.owner, site.attribute,
                                     lambda original, name=name: self._wrap_outputs(name, original))
        if freeze:
            self.patcher.replace(_A, "freeze_edge", lambda original: self._wrap_freeze(freeze, original))
        return self

    def __exit__(self, *exc):
        self.patcher.restore()
        return False


def perturb_definition(curve, perturber):
    """One-ulp perturbation of a frozen curve definition (a new, valid record)."""
    if isinstance(curve, _E.EllipticArc):
        return dataclasses.replace(curve, center=perturber.nudge(tuple(curve.center)),
                                   u_vector=perturber.nudge(tuple(curve.u_vector)),
                                   v_vector=perturber.nudge(tuple(curve.v_vector)))
    if isinstance(curve, _A.LinePath):
        return _A.LinePath(perturber.nudge(tuple(curve.start)), perturber.nudge(tuple(curve.end)))
    if isinstance(curve, _A.BezierPath):
        return _A.BezierPath(tuple(perturber.nudge(tuple(map(float, point))) for point in curve.controls))
    return curve


# =========================================================================== routes

def _dot_plain(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _exact_fma(x, y, z):
    """``x*y + z`` with one rounding (``math.fma`` before Python 3.13): the exact rational
    result rounds correctly to the nearest float (OverflowError on overflow, as ``math.fma``);
    special values follow ``math.fma``: a finite product meets a non-finite ``z`` exactly, and an
    invalid operation raises ValueError."""
    if math.isfinite(x) and math.isfinite(y):
        if not math.isfinite(z):
            return z
        exact = Fraction(x) * Fraction(y) + Fraction(z)
        return math.copysign(0., x * y + z) if exact == 0 else float(exact)
    result = x * y + z
    if math.isnan(result) and not any(math.isnan(value) for value in (x, y, z)):
        raise ValueError("invalid operation in fma")
    return result


_fma = getattr(math, "fma", _exact_fma)


def _dot_fma(a, b):
    return _fma(float(a[0]), float(b[0]),
                    _fma(float(a[1]), float(b[1]), float(a[2]) * float(b[2])))


def _det2_plain(a, b, c, d):
    return a * d - b * c


def _det2_fma(a, d, b, c):
    # a*d - b*c with one rounding for the first product (Kahan-free fma form)
    return _fma(a, d, -(b * c))


def _numpy_route_functions():
    """The product's vectorized NumPy formulas also for scalar calls (the
    product's own scalar fast path of LinePath.evaluate is bypassed)."""
    def line_evaluate(original):
        def evaluate(self, parameters):
            t = np.asarray(parameters, dtype=float)
            return np.asarray(self.start) + t[..., None] * (np.asarray(self.end) - self.start)
        return evaluate
    return {"LinePath.evaluate": line_evaluate}


def _route_functions(fma: bool):
    dot = _dot_fma if fma else _dot_plain

    def det2(a, b, c, d):  # | a b ; c d |
        return _det2_fma(a, d, b, c) if fma else _det2_plain(a, b, c, d)

    def lerp(s, t, e):
        return _fma(t, e - s, s) if fma else s + t * (e - s)

    def plane_local_uv(original):
        def local_uv(self, point):
            offset = surfaces._vector3(point, "point") - self.origin
            u, v = self.u_vector, self.v_vector
            # Normal equations solved in closed form (Cramer): the exact
            # least-squares solution in real arithmetic.
            g11, g12, g22 = dot(u, u), dot(u, v), dot(v, v)
            r1, r2 = dot(u, offset), dot(v, offset)
            determinant = det2(g11, g12, g12, g22)
            return det2(r1, g12, r2, g22) / determinant, det2(g11, r1, g12, r2) / determinant
        return local_uv

    def angular_local_uv(original):
        def local_uv(self, point):
            offset = surfaces._vector3(point, "point") - self.origin
            axis = self.axis
            axial = dot(offset, axis)
            radial = [float(offset[k]) - axial * float(axis[k]) for k in range(3)]
            angle = math.atan2(dot(radial, self._circumferential), dot(radial, self.radial_direction))
            delta = surfaces._angle_on_sweep(angle, self.start_angle, self.sweep_angle)
            return delta / self.sweep_angle, axial / self.height
        return local_uv

    def plane_evaluate(original):
        def evaluate(self, u, v):
            u, v = float(u), float(v)
            o, a, b = self.origin, self.u_vector, self.v_vector
            if fma:
                return np.array([_fma(v, float(b[k]), _fma(u, float(a[k]), float(o[k])))
                                 for k in range(3)])
            return np.array([float(o[k]) + u * float(a[k]) + v * float(b[k]) for k in range(3)])
        return evaluate

    def cylinder_evaluate(original):
        def evaluate(self, u, v):
            angle = self.start_angle + float(u) * self.sweep_angle
            c, s = math.cos(angle), math.sin(angle)
            r, w, axis, o = self.radial_direction, self._circumferential, self.axis, self.origin
            h = float(v) * self.height
            out = []
            for k in range(3):
                radial = (_fma(c, float(r[k]), s * float(w[k])) if fma
                          else c * float(r[k]) + s * float(w[k]))
                out.append(float(o[k]) + self.radius * radial + h * float(axis[k]))
            return np.array(out)
        return evaluate

    def _per_parameter(scalar):
        def evaluate_many(parameters):
            t = np.asarray(parameters, dtype=float)
            flat = [scalar(float(x)) for x in t.reshape(-1)]
            return np.asarray(flat, dtype=float).reshape(t.shape + (3,))
        return evaluate_many

    def line_evaluate(original):
        def evaluate(self, parameters):
            start = tuple(map(float, self.start))
            end = tuple(map(float, self.end))
            return _per_parameter(lambda t: [lerp(start[k], t, end[k]) for k in range(3)])(parameters)
        return evaluate

    def ellipse_evaluate(original):
        def evaluate(self, parameters):
            centre, u, v = self.center, self.u_vector, self.v_vector
            start, sweep = float(self.start_angle), float(self.sweep_angle)
            values = exact_curves._parameters(parameters)

            def at(t):
                angle = start + sweep * t
                c, s = math.cos(angle), math.sin(angle)
                if fma:
                    return [_fma(s, float(v[k]), _fma(c, float(u[k]), float(centre[k])))
                            for k in range(3)]
                return [float(centre[k]) + c * float(u[k]) + s * float(v[k]) for k in range(3)]
            return _per_parameter(at)(values)
        return evaluate

    return {
        "Plane.local_uv": plane_local_uv,
        "Cylinder.local_uv": angular_local_uv,
        "Cone.local_uv": angular_local_uv,
        "Plane.evaluate": plane_evaluate,
        "Cylinder.evaluate": cylinder_evaluate,
        "LinePath.evaluate": line_evaluate,
        "EllipticArc.evaluate": ellipse_evaluate,
    }


ROUTE_SITES = tuple(_route_functions(False))


class RouteSwitch:
    """Replace the selected sites by an alternative arithmetic route."""

    def __init__(self, route, sites=None):
        if route not in ROUTES:
            raise ValueError(f"unknown route {route!r}")
        self.functions = (_numpy_route_functions() if route == "numpy"
                          else _route_functions(route == "fma"))
        self.sites = [name for name in (sites or self.functions) if name in self.functions]
        self.calls = collections.Counter()
        self.patcher = Patcher()

    def __enter__(self):
        for name in self.sites:
            site = SITES[name]
            factory = self.functions[name]

            def counted(original, name=name, factory=factory):
                route = factory(original)

                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    self.calls[name] += 1
                    return route(*args, **kwargs)
                return wrapped
            self.patcher.replace(site.owner, site.attribute, counted)
        return self

    def __exit__(self, *exc):
        self.patcher.restore()
        return False


# =========================================================================== running

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def mask_numbers(text: str) -> str:
    return _NUMBER.sub("#", text)


_VERBATIM_KEYS = {"digest", "topology_digest"}


def canonical(outcome) -> str:
    def strip(value):
        if isinstance(value, bool) or value is None or isinstance(value, int):
            return value
        if isinstance(value, float):
            return "<f>"
        if isinstance(value, str):
            return mask_numbers(value)
        if isinstance(value, dict):
            # Digests are compared verbatim: masking their digits would hide topology changes.
            return {str(key): (value[key] if str(key) in _VERBATIM_KEYS else strip(value[key]))
                    for key in sorted(value, key=str)}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return mask_numbers(str(value))
    return json.dumps(strip(outcome), sort_keys=True)


def seed_for(seed: int, *parts) -> int:
    return zlib.crc32("|".join(map(str, (seed, *parts))).encode()) & 0x7FFFFFFF


def run_operation(case, operation, *, reverse=False, context_manager=None):
    """Build the case afresh, run one operation, return (outcome, calls)."""
    try:
        model, context = case.build()
    except Exception as error:  # noqa: BLE001 - a refused build is an outcome
        return {"build_error": type(error).__name__, "message": mask_numbers(str(error))[:160]}, {}
    clear_caches()
    manager = context_manager or _Null()
    with np.errstate(all="ignore"):
        try:
            with manager:
                outcome = operation.run(model, context, reverse)
        except Exception as error:  # noqa: BLE001
            outcome = {"error": type(error).__name__, "message": mask_numbers(str(error))[:160]}
    clear_caches()
    return outcome, dict(getattr(manager, "calls", {}))


class _Null:
    calls = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def variant_names(include_routes=True):
    names = ["none"]
    names += [f"{mode}/{pattern}" for mode in MODES for pattern in PATTERNS]
    if include_routes:
        names += [f"route/{route}" for route in ROUTES]
    return names


def make_manager(variant, *, seed, case, operation, sites=None):
    if variant == "none":
        return None
    family, _, detail = variant.partition("/")
    if family == "route":
        return RouteSwitch(detail, sites)
    if family in MODES:
        chosen = sites if sites is not None else sites_for_mode(family)
        return Perturbation(chosen, detail, seed_for(seed, case, operation, variant))
    raise ValueError(f"unknown variant {variant!r}")


def compare(reference, outcome):
    return canonical(reference) != canonical(outcome)


def attribute(case, operation, variant, reference, calls, seed):
    """Re-run a flipped variant with each called site alone."""
    rows = []
    for name in sorted(calls):
        manager = make_manager(variant, seed=seed, case=case.name, operation=operation.name, sites=[name])
        outcome, _ = run_operation(case, operation, context_manager=manager)
        if compare(reference, outcome):
            rows.append({"site": name, "inventory": list(SITES[name].inventory), "outcome": outcome})
    return rows


def reproducer(case, operation, variant, site=None):
    command = (f"python tools/perturbation_harness.py --case \"{case}\" --op \"{operation}\" "
               f"--variant \"{variant}\"")
    if site:
        command += f" --sites \"{site}\""
    return command


def run_case(case, *, variants, seed, attribute_flips=True, op_filter=None, sites=None, log=print):
    rows = {}
    for operation in case.operations:
        if op_filter and operation.name != op_filter:
            continue
        start = time.perf_counter()
        reference, _ = run_operation(case, operation)
        result = {"reference": reference, "variants": {}, "reversal": None}
        for variant in variants:
            if variant == "reverse":
                continue
            manager = make_manager(variant, seed=seed, case=case.name, operation=operation.name, sites=sites)
            outcome, calls = run_operation(case, operation, context_manager=manager)
            flipped = compare(reference, outcome)
            entry = {"flipped": flipped, "calls": calls}
            if flipped:
                entry["outcome"] = outcome
                entry["reproducer"] = reproducer(case.name, operation.name, variant,
                                                 ",".join(sites) if sites else None)
                if attribute_flips and variant != "none" and len(calls) > 1:
                    entry["attribution"] = attribute(case, operation, variant, reference, calls, seed)
                elif calls:
                    entry["attribution"] = [{"site": name, "inventory": list(SITES[name].inventory)}
                                            for name in calls]
            result["variants"][variant] = entry
            log(f"  {case.name:36s} {operation.name:22s} {variant:28s} "
                f"{'FLIP' if flipped else 'same'}"
                + (f"  sites={[row['site'] for row in entry.get('attribution', [])]}" if flipped else ""))
        if operation.reversible and "reverse" in variants:
            outcome, _ = run_operation(case, operation, reverse=True)
            flipped = compare(reference, outcome)
            result["reversal"] = {"flipped": flipped, "outcome": outcome if flipped else None,
                                  "reproducer": reproducer(case.name, operation.name, "reverse")
                                  if flipped else None}
            log(f"  {case.name:36s} {operation.name:22s} {'reverse':28s} {'FLIP' if flipped else 'same'}")
        result["seconds"] = round(time.perf_counter() - start, 2)
        rows[operation.name] = result
    return rows


# =========================================================================== reports

def summarize(report):
    flips = []
    for case_name, case in report["cases"].items():
        for op_name, op in case["operations"].items():
            for variant, entry in op["variants"].items():
                if entry["flipped"]:
                    flips.append({"case": case_name, "operation": op_name, "variant": variant,
                                  "sites": [row["site"] for row in entry.get("attribution", [])],
                                  "inventory": sorted({i for row in entry.get("attribution", [])
                                                       for i in row["inventory"]}),
                                  "reference": op["reference"], "outcome": entry["outcome"],
                                  "reproducer": entry["reproducer"]})
            if op.get("reversal") and op["reversal"]["flipped"]:
                flips.append({"case": case_name, "operation": op_name, "variant": "reverse", "sites": [],
                              "inventory": [], "reference": op["reference"],
                              "outcome": op["reversal"]["outcome"],
                              "reproducer": op["reversal"]["reproducer"]})
    return flips


def changed_fields(reference, outcome, path=""):
    """Paths (with before -> after) at which two outcomes differ."""
    if isinstance(reference, dict) and isinstance(outcome, dict):
        rows = []
        for key in sorted(set(reference) | set(outcome), key=str):
            rows += changed_fields(reference.get(key), outcome.get(key), f"{path}/{key}")
        return rows
    if isinstance(reference, list) and isinstance(outcome, list) and len(reference) == len(outcome):
        rows = []
        for index, (left, right) in enumerate(zip(reference, outcome)):
            rows += changed_fields(left, right, f"{path}[{index}]")
        return rows
    if canonical(reference) == canonical(outcome):
        return []
    return [f"{path or '/'}: {json.dumps(reference, default=str)[:60]} -> {json.dumps(outcome, default=str)[:60]}"]


def _short(outcome):
    text = json.dumps(outcome, default=str, sort_keys=True)
    return text if len(text) <= 110 else text[:107] + "..."


def markdown(report) -> str:
    flips = summarize(report)
    lines = ["# Perturbation harness report", "",
             f"- anygeometry: `{report['meta']['anygeometry']}`",
             f"- python {report['meta']['python']}, numpy {report['meta']['numpy']}, seed {report['meta']['seed']}",
             f"- variants: {', '.join(report['meta']['variants'])}",
             f"- cases: {len(report['cases'])}, operations: "
             f"{sum(len(case['operations']) for case in report['cases'].values())}, "
             f"flips: {len(flips)}, wall: {report['meta']['seconds']} s", ""]
    # per-case overview
    lines += ["## Per case", "", "| Case | Operation | Flipping variants | Reversal |", "|---|---|---|---|"]
    for case_name, case in report["cases"].items():
        for op_name, op in case["operations"].items():
            flipped = [variant for variant, entry in op["variants"].items() if entry["flipped"]]
            reversal = op.get("reversal")
            rev = "n/a" if reversal is None else ("FLIP" if reversal["flipped"] else "same")
            lines.append(f"| {case_name} | {op_name} | {', '.join(flipped) or '-'} | {rev} |")
    # per site
    per_site = collections.defaultdict(list)
    for flip in flips:
        for site in flip["sites"] or ["(not attributed)"]:
            per_site[site].append(f"{flip['case']} {flip['operation']} {flip['variant']}")
    lines += ["", "## Per perturbation site", "", "| Site | Inventory | Flips |", "|---|---|---|"]
    for site in sorted(per_site):
        inventory = ", ".join(SITES[site].inventory) if site in SITES else ""
        lines.append(f"| {site} | {inventory} | {len(per_site[site])}: " + "; ".join(per_site[site][:6])
                     + (" ..." if len(per_site[site]) > 6 else "") + " |")
    lines += ["", "## Flips", ""]
    for flip in flips:
        lines += [f"### {flip['case']} / {flip['operation']} / {flip['variant']}", "",
                  f"- sites: {', '.join(flip['sites']) or '-'} (inventory {', '.join(flip['inventory']) or '-'})",
                  "- changed: " + "; ".join(f"`{row}`" for row in
                                           changed_fields(flip["reference"], flip["outcome"])[:4]),
                  f"- reference: `{_short(flip['reference'])}`",
                  f"- perturbed: `{_short(flip['outcome'])}`",
                  f"- reproducer: `{flip['reproducer']}`", ""]
    coverage = collections.Counter()
    for case in report["cases"].values():
        for op in case["operations"].values():
            for entry in op["variants"].values():
                for name in entry["calls"]:
                    coverage[name] += 1
    lines += ["## Site coverage (operation runs in which the site was called)", "",
              "| Site | Runs |", "|---|---|"]
    for name in SITES:
        lines.append(f"| {name} | {coverage.get(name, 0)} |")
    return "\n".join(lines) + "\n"


def merge(paths):
    merged = None
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if merged is None:
            merged = report
            merged["meta"]["parts"] = [str(path)]
            continue
        merged["cases"].update(report["cases"])
        merged["meta"]["seconds"] = round(merged["meta"]["seconds"] + report["meta"]["seconds"], 1)
        merged["meta"]["parts"].append(str(path))
    return merged


def main(argv=None) -> int:
    import adversarial_corpus as corpus  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", help="case name (repeatable); prefix match with a trailing *")
    parser.add_argument("--group", choices=("adversarial", "g0"))
    parser.add_argument("--op", help="run only this operation")
    parser.add_argument("--variant", action="append", help="variant (repeatable), e.g. outputs/+1, route/fma, reverse")
    parser.add_argument("--sites", help="comma-separated site names (restricts the perturbed/route sites)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heavy", action="store_true", help="include heavy G0 workloads (grid32)")
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.merge:
        report = merge(args.merge)
        if args.json:
            args.json.write_text(json.dumps(report, default=str), encoding="utf-8")
        if args.markdown:
            args.markdown.write_text(markdown(report), encoding="utf-8")
        return 0
    if args.list:
        for name, case in corpus.CASES.items():
            print(f"{name:40s} {case.group:12s} {', '.join(op.name for op in case.operations)}")
        for name, site in SITES.items():
            print(f"site {name:32s} {site.group:10s} {', '.join(site.inventory)}")
        return 0

    names = list(corpus.CASES) if args.heavy else corpus.default_cases()
    if args.group:
        names = [name for name in names if corpus.CASES[name].group == args.group]
    if args.case:
        selected = []
        for pattern in args.case:
            if pattern.endswith("*"):
                selected += [name for name in corpus.CASES if name.startswith(pattern[:-1])]
            elif pattern in corpus.CASES:
                selected.append(pattern)
            else:
                parser.error(f"unknown case {pattern!r}")
        names = list(dict.fromkeys(selected))
    variants = args.variant or [*variant_names(), "reverse"]
    sites = args.sites.split(",") if args.sites else None
    for name in sites or ():
        if name not in SITES:
            parser.error(f"unknown site {name!r}")

    started = time.perf_counter()
    report = {"meta": {"anygeometry": anygeometry.__file__, "python": sys.version.split()[0],
                       "numpy": np.__version__, "seed": args.seed, "variants": variants,
                       "sites": sites, "fma": hasattr(math, "fma")}, "cases": {}}
    for name in names:
        case = corpus.CASES[name]
        print(f"{name} ({case.note})", flush=True)
        report["cases"][name] = {"group": case.group, "note": case.note, "inventory": list(case.inventory),
                                 "operations": run_case(case, variants=variants, seed=args.seed,
                                                        attribute_flips=not args.no_attribution,
                                                        op_filter=args.op, sites=sites,
                                                        log=lambda text: print(text, flush=True))}
    report["meta"]["seconds"] = round(time.perf_counter() - started, 1)
    flips = summarize(report)
    print(f"{len(flips)} flips in {report['meta']['seconds']} s")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
