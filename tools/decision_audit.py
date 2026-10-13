"""R0 structural audit: enumerate rounding-sensitive decision sites (heuristic).

Slice R0 of the robust-predicate programme (GENERALITY_ARCHITECTURE.md R.4,
instrument 1). An AST scan (stdlib ``ast`` only) of the R-certified modules
that lists every site of the kinds

  (a) ``tolerance``  a comparison involving a float expression and a
      tolerance/eps-like quantity (``tol``, ``eps``, ``angular``, small float
      constants, ``np.finfo(...).eps``), and ``isclose``-style calls;
  (b) ``ordering``   ordering/selection without a threshold: ``sorted``/
      ``min``/``max``/``.sort``/``argmin``/``argmax``/``argsort``/``np.clip``
      on float data, ``round``/``floor``/``ceil``/``int()``/``%``/``//`` on
      floats, ``==``/``!=``/``in`` on floats, sign branches (``x > 0``),
      comparisons of two float expressions, ``np.sign``, truthiness of floats;
  (c) ``provenance`` constructors of curve/surface/arrangement definitions
      (``LinePath``, ``EllipticArc``, ``BezierPath``, ``Plane``, ``Cylinder``,
      ``Cone`` and friends, ``add_point``) whose arguments derive from an
      ``evaluate``/``derivative``/``local_uv``/``uv``/``closest_uv``/witness/
      ``position``/least-squares result;
  (d) ``library``    geometric predicates of a library (shapely ``covers``,
      ``within``, ``intersects``, ...) on float data. R.4 states that library
      decisions cannot be proven by this audit; their call sites are listed.

The scan is HEURISTIC and says so: floatness and derivation are inferred from
names, calls and a per-function assignment data flow, not from types. It can
report false positives and miss sites (see docs/agent_plans/R0_DECISION_AUDIT.md).

A site is *certified* when one of its source lines (or the line above it)
carries ``# coordinate-only: <reason>`` or its statement calls a function of
``exact_predicates`` listed in ``CERTIFIED_EXACT_PREDICATES``. Each site is
matched against ``INVENTORY`` (R.1 ids plus the section-10 open-item sites,
ids ``X1``-``X4``); unmatched sites are reported as ``unassigned``.

Usage::

    python tools/decision_audit.py [--root REPO] [--json OUT] [--markdown OUT] [--fail]
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("src") / "anygeometry"

# --------------------------------------------------------------------------
# Scope: module -> None (whole module) or the qualified function names scanned
# (nested functions of a listed function are included).
# --------------------------------------------------------------------------
SCOPE: dict[str, tuple[str, ...] | None] = {
    # R.4 module list.
    "arrangement_geometry": None,
    "material_arrangement": None,
    "batch_intersections": None,
    "member_arrangements": None,
    "attachment_remapping": None,
    "trimmed_charts": None,
    "analytic_supports": None,
    "intersections": None,
    "predicates": None,
    "curved_intersections": None,
    # R.4: construction routines the modules call in model.py; A1/A6/A9 and
    # freeze_edge inputs (arc frames, spline points).
    "model": ("GeometryModel._detect_corners", "GeometryModel.face_local_uv",
              "GeometryModel.face_local_uv_many", "GeometryModel.face_support_local_uv",
              "GeometryModel._arc_frame", "GeometryModel._spline_points"),
    # Inventory additions: chart inversions and angle wraps of A1/A2/A4/A6.
    "surfaces": ("Plane.local_uv", "Cylinder.local_uv", "Cone.local_uv",
                 "_angle_on_sweep", "closest_uv"),
    # A2/A4/F2 functions.
    "quadric_supports": ("_inside_patch", "_inside_patch_many", "_merge", "charts_support",
                         "_apex_contacts", "_plane_through_apex", "_coaxial_rings"),
    # Section-10 open item X1: three-point arc frames behind freeze_edge.
    "curves": ("arc_frame", "_wrapped_angle"),
    # Section-10 open item X3: tolerances derived from extents.
    "tolerance": ("TolerancePolicy.effective_length", "TolerancePolicy.effective_coincidence",
                  "TolerancePolicy.effective_healing", "TolerancePolicy.effective_area",
                  "TolerancePolicy.effective_parameter"),
}

# exact_predicates functions whose calls certify a site (none exist yet).
CERTIFIED_EXACT_PREDICATES: frozenset[str] = frozenset()
COORDINATE_ONLY = re.compile(r"#\s*coordinate-only:\s*\S")


@dataclass(frozen=True)
class InventoryEntry:
    """One R.1 entry located in the code.

    ``function`` is a regular expression matched in full against the site's
    qualified function name (or a prefix of it ending at a nested function).
    ``region`` (start, end) are regular expressions: the site must lie between
    the first line of the function matching ``start`` and the first later line
    matching ``end`` (inclusive). ``text`` must be found in the site's source.
    """
    id: str
    module: str
    function: str
    region: tuple[str, str] | None = None
    text: str | None = None


def _e(id, module, function, region=None, text=None):
    return InventoryEntry(id, module, function, region, text)


# R.1 entries (line numbers of the architecture are of 9ff7ed0; the table
# locates them by function and source anchors so it survives line drift).
INVENTORY: tuple[InventoryEntry, ...] = (
    # Group A: chart inversion feeding decisions.
    _e("A1", "surfaces", r"Plane\.local_uv"),
    _e("A1", "intersections", r"clip_line_to_face", text=r"covers|local_uv"),
    _e("A1", "curved_intersections", r"qualified_curve_face", text=r"local_uv|middle_uv"),
    _e("A1", "predicates", r"_plane_parameters|_snap_unit_parameter"),
    _e("A1", "material_arrangement", r"MaterialDomain\.uv"),
    _e("A1", "material_arrangement", r"MaterialDomain\.area_loop.*"),
    _e("A1", "attachment_remapping", r"remap_face_attachments", text=r"uv|local_uv"),
    _e("A1", "model", r"GeometryModel\.face_local_uv(_many)?"),
    _e("A2", "surfaces", r"(Cylinder|Cone)\.local_uv"),
    _e("A2", "quadric_supports", r"_inside_patch(_many)?"),
    _e("A2", "analytic_supports", r"_inside"),
    _e("A2", "predicates", r"qualified_line_cylinder\.patch_parameters"),
    _e("A3", "material_arrangement", r"MaterialDomain\.(_extruded_uv|uv)", text=r"round|period"),
    _e("A4", "surfaces", r"_angle_on_sweep"),
    _e("A4", "quadric_supports", r"_inside_patch(_many)?", text=r"raw|angle|tau|period"),
    _e("A5", "batch_intersections", r"_rectangular_cylinder"),
    _e("A6", "surfaces", r"closest_uv"),
    _e("A6", "curved_intersections", r"_newton_curve_pair|_support_projection|_refine_surface_pair"),
    _e("A6", "model", r"GeometryModel\.face_support_local_uv"),
    _e("A7", "batch_intersections", r"_pair_paths", text=r"witness|direction|LinePath"),
    _e("A7", "predicates", r"qualified_plane_plane", text=r"direction|solve|witness|LinePath"),
    _e("A8", "batch_intersections", r"_child_support", text=r"corner|norm|Plane\("),
    _e("A9", "model", r"GeometryModel\._detect_corners"),
    # Group B: point/parameter membership on curves.
    _e("B1", "arrangement_geometry", r"_point_parameters", region=(r"isinstance\(curve, LinePath\)", r"return \(\)")),
    _e("B2", "arrangement_geometry", r"_point_parameters", region=(r"isinstance\(curve, EllipticArc\)", r"return tuple\(roots\)|isinstance\(curve, BezierPath\)")),
    _e("B2", "arrangement_geometry", r"ellipse_curve_junctions", region=(r"result = \[\]", r"return")),
    _e("B3", "arrangement_geometry", r"point_parameters"),
    _e("B4", "arrangement_geometry", r"plane_roots"),
    _e("B5", "arrangement_geometry", r"_angle_parameter"),
    _e("B5", "material_arrangement", r"MaterialDomain\.horizontal_roots", text=r"endpoint|finfo|norm"),
    _e("B5", "material_arrangement", r"MaterialDomain\.contains_loop", text=r"8\*np\.finfo"),
    _e("B6", "material_arrangement", r"_parameters"),
    _e("B6", "material_arrangement", r"arrange_material", text=r"_parameters\("),
    _e("B7", "batch_intersections", r"_apply_intersections_in_place",
       region=(r"for fraction in sorted\(values\)", r"len\(stations\) != 1|unique")),
    _e("B8", "arrangement_geometry", r"line_curve_junctions|ellipse_curve_junctions"),
    _e("B9", "member_arrangements", r"_support_roots", text=r"root\.witness|32\*np\.finfo"),
    _e("B10", "arrangement_geometry", r"_point_parameters", region=(r"isinstance\(curve, BezierPath\)", r"raise GeometryError")),
    _e("B11", "batch_intersections", r"_apply_intersections_in_place\.edge", text=r"evaluate\(0\.\)"),
    _e("B11", "material_arrangement", r"arrange_material",
       region=(r"curve = path\.curve\.subcurve\(a, b\)", r"start, end = vertex\(curve\.evaluate")),
    _e("B11", "member_arrangements", r"plan_member_arrangements", text=r"evaluate\(parameter\)|position"),
    # Group C: vertex identity and duplicate edges.
    _e("C1", "material_arrangement", r"_VertexLookup\..*|arrange_material\.vertex"),
    _e("C2", "batch_intersections", r"_apply_intersections_in_place\.(vertex|nearby|cell|grid_for)"),
    _e("C2", "batch_intersections", r"_apply_intersections_in_place", text=r"math\.floor\(float\(value\)/size\)"),
    _e("C3", "material_arrangement", r"arrange_material", region=(r"duplicate = None", r"if duplicate is not None")),
    _e("C3", "batch_intersections", r"_coincident"),
    _e("C4", "batch_intersections", r"_apply_intersections_in_place\.edge",
       region=(r"isinstance\(curve, EllipticArc\) and \(", r"add_arc|add_curve")),
    # Group D: containment, ordering, area on material domains.
    _e("D1", "material_arrangement", r"MaterialDomain\.contains_loop"),
    _e("D2", "material_arrangement", r"arrange_material", region=(r"outgoing = \[\[\] for _ in vertices\]", r"seen, cycles = set\(\)")),
    _e("D2", "material_arrangement", r"MaterialDomain\.(tangent|curvature)"),
    _e("D3", "material_arrangement", r"arrange_material", text=r"area|original|retained"),
    _e("D3", "trimmed_charts", r"validate_trimmed_surface_charts_binding", text=r"area"),
    _e("D4", "material_arrangement", r"MaterialDomain\.from_model"),
    _e("D5", "material_arrangement", r"MaterialDomain\.left_probe"),
    _e("D5", "material_arrangement", r"arrange_material", region=(r"middle = domain\.uv\(curve, \.5\)", r"holes\[owner\]\.append")),
    _e("D6", "material_arrangement", r"arrange_material", region=(r"endpoints = \[np\.asarray\(point\.position\)", r"endpoints\.append\(point\)")),
    _e("D7", "material_arrangement", r"arrange_material", region=(r"if endpoints:", r"splits = \[\[0\., 1\.\]")),
    _e("D6", "material_arrangement", r"_add_exact_planar_seams|_same_ray"),
    _e("D7", "material_arrangement", r"_add_exact_planar_seams", text=r"Segment\(|_clip_line_exact"),
    _e("D8", "material_arrangement", r"arrange_material", region=(r"splits = \[\[0\., 1\.\]", r"splits\[first\]\.append\(Fraction")),
    _e("D8", "member_arrangements", r"plan_member_arrangements", text=r"evaluate\(parameter\)"),
    # Group E: support/coincidence certificates, query-route classification.
    _e("E1", "member_arrangements", r"_support_roots", text=r"linspace|samples|points"),
    _e("E2", "member_arrangements", r"_support_roots"),
    _e("E3", "curved_intersections", r"_curve_exactly_on_support"),
    # _point_in_polygon is the float ray parity _inside_face calls.
    _e("E4", "curved_intersections", r"_support_trim_loops|_inside_face|_point_in_polygon"),
    _e("E4", "curved_intersections", r"qualified_curve_face", text=r"_inside_face"),
    _e("E4", "intersections", r"clip_line_to_face", text=r"covers|Polygon"),
    _e("E5", "curved_intersections", r"qualified_curve_face"),
    _e("E5", "predicates", r"qualified_(line_line|line_plane|plane_plane|segment_segment)", text=r"sine|denominator"),
    _e("E6", "analytic_supports", r"_ordered|plane_cylinder_support|cylinder_cylinder_support|_parallel"),
    _e("E7", "member_arrangements", r"plan_member_arrangements", text=r"hi1 < lo2|hi2 < lo1"),
    _e("E7", "material_arrangement", r"arrange_material", text=r"hi1 < lo2|hi2 < lo1"),
    _e("E8", "attachment_remapping", r"_box_rejects"),
    _e("E9", "batch_intersections", r"_coincident_cylinder_rectangles"),
    # Group F: deferred families.
    _e("F1", "arrangement_geometry", r"_merge_touching"),
    _e("F2", "quadric_supports", r"_merge|charts_support|_apex_contacts|_plane_through_apex|_coaxial_rings"),
    _e("F3", "material_arrangement", r"MaterialDomain\.at_apex"),
    _e("F4", "batch_intersections", r"_coincident"),
    # Section-10 accepted open item "inventory sites" (no R.1 id yet).
    # X1 three-point arc frames (freeze_edge -> model._arc_frame -> curves.arc_frame).
    _e("X1", "curves", r"arc_frame|_wrapped_angle"),
    _e("X1", "model", r"GeometryModel\._arc_frame"),
    _e("X1", "arrangement_geometry", r"freeze_edge"),
    # X2 broad-phase pruning in batch_intersections.
    _e("X2", "batch_intersections", r"_synchronize_boundary_events", text=r"lo-|hi\+|world_tolerance"),
    _e("X2", "batch_intersections", r"_plan_intersections", text=r"hi1 < lo2|hi2 < lo1|min\(axis=0\)"),
    # X3 tolerances derived from extents (batch_intersections -> tolerance.py).
    _e("X3", "batch_intersections", r"_plan_intersections", text=r"effective_length|norm\(hi|ptp"),
    _e("X3", "tolerance", r"TolerancePolicy\..*"),
    # X4 positive-area overlap probe in _coplanar_traces.
    _e("X4", "batch_intersections", r"_coplanar_traces", region=(r"for cell in arranged\.cells", r"positive-area overlap")),
)

# --------------------------------------------------------------------------
# Heuristic vocabularies.
# --------------------------------------------------------------------------
_TOL_WORD = re.compile(r"(?i)(^|_)(tol|tols|rtol|atol|eps|epsilon|tiny|angular|touch|threshold|margin|slack)(_|$)|tolerance")
SMALL_CONSTANT = 1e-3            # a float literal 0 < |c| <= this is tolerance-like
FLOAT_CALLS = {
    "evaluate", "derivative", "local_uv", "local_uv_many", "uv", "closest_uv", "norm", "sqrt",
    "atan2", "arctan2", "cos", "sin", "tan", "acos", "arccos", "asin", "arcsin", "hypot", "exp",
    "log", "dot", "cross", "float", "lstsq", "solve", "pinv", "area_loop", "tangent", "curvature",
    "point_parameters", "plane_roots", "curve_junctions", "bounds", "vertex_position", "linspace",
    "effective_length", "effective_coincidence", "effective_area", "left_probe", "ptp",
    "oriented_end_tangent", "oriented_start_tangent", "_angle_on_sweep", "_wrapped_angle",
    "trigonometric_roots", "isolate_real_roots", "nextafter", "_non_negative_finite",
    "_positive_finite",
}
FLOAT_ATTRS = {
    "witness", "witnesses", "start_angle", "sweep_angle", "center", "position", "radius",
    "u_vector", "v_vector", "origin", "normal", "direction", "controls", "sweep", "e1", "e2",
    "pi", "tau", "material_area", "length", "area", "relative_length", "relative_area",
}
INT_CALLS = {"len", "int", "range", "enumerate", "id", "hash", "str", "bool", "count", "index",
             "matrix_rank", "flatnonzero", "nonzero"}
# Calls whose result is float when an argument is (containers, numpy wrappers,
# elementwise functions). Any other call with float arguments is NOT assumed to
# return a float (it may return ids, kinds, records).
PASSTHROUGH_CALLS = {"asarray", "array", "abs", "fabs", "sum", "tuple", "list", "min", "max", "amin",
                     "amax", "clip", "mean", "sorted", "reversed", "column_stack", "stack", "vstack",
                     "hstack", "concatenate", "copy", "astype", "item", "squeeze", "ravel", "reshape",
                     "next", "minimum", "maximum", "einsum", "matmul", "inner", "outer", "square",
                     "power", "subtract", "add", "multiply", "divide", "prod", "cumsum", "diff",
                     "fsum", "nan_to_num", "float64", "float32", "isfinite"}
FLOAT_ARG = re.compile(r"^(point|points|world|uv|uvs|parameter|parameters|t|u|v|s|angle|angles|value|"
                       r"values|position|tangent|normal|offset|direction|fraction|area|length|radius|"
                       r"center|lo|hi|lower|upper|start_angle|sweep|extent|witness|deviation)$")
DERIVED_CALLS = {"evaluate", "derivative", "local_uv", "local_uv_many", "uv", "closest_uv",
                 "lstsq", "solve", "pinv", "tangent", "left_probe"}
DERIVED_ATTRS = {"witness", "witnesses", "position"}
CONSTRUCTORS = {"LinePath", "EllipticArc", "BezierPath", "Plane", "Cylinder", "Cone", "Sphere",
                "Torus", "ExtrudedSurface", "CylinderIntersectionCurve", "QuadricIntersectionCurve",
                "BezierQuadricCurve", "ArrangementPoint", "ArcFrame", "add_point"}
# Selection of an element/order by float keys.
SELECT_CALLS = {"sorted", "sort", "argmin", "argmax", "argsort", "nanargmin", "nanargmax",
                "nsmallest", "nlargest"}
# min/max/clip used as a value (clamps, bounding boxes); reported separately.
MINMAX_CALLS = {"min", "max", "amin", "amax", "nanmin", "nanmax", "clip"}
ROUND_CALLS = {"round", "floor", "ceil", "trunc", "rint", "fix"}
ISCLOSE_CALLS = {"isclose", "allclose"}
LIBRARY_PREDICATES = {"covers", "covered_by", "within", "intersects", "touches", "crosses",
                      "overlaps", "disjoint", "relate", "equals", "contains_properly"}


def _name(node):
    """Last identifier of a Name/Attribute/Call function, or ''."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_tol_constant(node):
    return (isinstance(node, ast.Constant) and isinstance(node.value, float)
            and 0 < abs(node.value) <= SMALL_CONSTANT)


@dataclass
class _Names:
    floats: set = field(default_factory=set)
    derived: set = field(default_factory=set)
    tolerances: set = field(default_factory=set)


class _Analyzer:
    def __init__(self, names: _Names):
        self.n = names

    def tol_like(self, node) -> bool:
        """A tolerance-like quantity occurs in ``node``'s value. Arguments of
        calls are not searched (``f(x, tolerance=t)`` is not a threshold),
        except for value-passing calls such as ``abs``/``float``/``max``."""
        if isinstance(node, (ast.Name, ast.Attribute)):
            ident = _name(node)
            if _TOL_WORD.search(ident) or ident in self.n.tolerances:
                return True
            return isinstance(node, ast.Attribute) and self.tol_like(node.value) and not isinstance(node.value, ast.Call)
        if _is_tol_constant(node):
            return True
        if isinstance(node, ast.Call):
            if _name(node) in PASSTHROUGH_CALLS or _name(node) in {"float", "nextafter"}:
                return any(self.tol_like(arg) for arg in node.args)
            return False
        if isinstance(node, ast.Lambda):
            return False
        return any(self.tol_like(child) for child in ast.iter_child_nodes(node))

    def float_like(self, node) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Constant):
            return isinstance(node.value, float)
        if isinstance(node, ast.Name):
            return node.id in self.n.floats or bool(_TOL_WORD.search(node.id)) or node.id in self.n.tolerances
        if isinstance(node, ast.Attribute):
            return (node.attr in FLOAT_ATTRS or bool(_TOL_WORD.search(node.attr))
                    or (node.attr == "eps"))
        if isinstance(node, ast.Call):
            name = _name(node)
            if name in INT_CALLS:
                return False
            if name in FLOAT_CALLS:
                return True
            if name not in PASSTHROUGH_CALLS:
                return False
            return any(self.float_like(arg) for arg in node.args) or (
                isinstance(node.func, ast.Attribute) and self.float_like(node.func.value))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.MatMult)):
                return True
            return self.float_like(node.left) or self.float_like(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.float_like(node.operand)
        if isinstance(node, ast.Subscript):
            return self.float_like(node.value)
        if isinstance(node, ast.IfExp):
            return self.float_like(node.body) or self.float_like(node.orelse)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(self.float_like(item) for item in node.elts)
        if isinstance(node, ast.Starred):
            return self.float_like(node.value)
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            return self.float_like(node.elt) or any(self.float_like(g.iter) for g in node.generators)
        if isinstance(node, ast.NamedExpr):
            return self.float_like(node.value)
        return False

    def derived_from(self, node) -> str | None:
        """The first derived source (call/attribute/name) inside ``node``."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and _name(sub) in DERIVED_CALLS:
                return _name(sub) + "()"
            if isinstance(sub, ast.Attribute) and sub.attr in DERIVED_ATTRS:
                return "." + sub.attr
            if isinstance(sub, ast.Name) and sub.id in self.n.derived:
                return sub.id
        return None


def _targets(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            yield from _targets(item)
    elif isinstance(target, ast.Starred):
        yield from _targets(target.value)


def _own_nodes(function):
    """Nodes of ``function``'s body, not descending into nested defs/classes."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _infer_names(function, inherited: _Names) -> _Names:
    names = _Names(set(inherited.floats), set(inherited.derived), set(inherited.tolerances))
    if not isinstance(function, ast.Module):
        args = function.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if FLOAT_ARG.match(arg.arg):
                names.floats.add(arg.arg)
    analyzer = _Analyzer(names)
    nodes = list(_own_nodes(function))
    for _ in range(3):  # small fixed point over assignments
        for node in nodes:
            pairs = []
            if isinstance(node, ast.Assign):
                pairs = [(t, node.value) for t in node.targets]
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value is not None:
                pairs = [(node.target, node.value)]
            elif isinstance(node, (ast.For, ast.comprehension)):
                pairs = [(node.target, node.iter)]
            elif isinstance(node, ast.NamedExpr):
                pairs = [(node.target, node.value)]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr in {"append", "extend", "add", "update"}
                  and isinstance(node.func.value, ast.Name)):
                # Records (tuples) appended to a list are mixed; only scalars count.
                pairs = [(node.func.value, arg) for arg in node.args if not isinstance(arg, ast.Tuple)]
            for target, value in pairs:
                if (isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, ast.Tuple)
                        and len(target.elts) == len(value.elts)):
                    pairs.extend(zip(target.elts, value.elts))  # elementwise
                    continue
                is_float = analyzer.float_like(value)
                if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Name, ast.Subscript)):
                    is_float = False  # unpacking mixed records: floatness unknown
                derived = analyzer.derived_from(value) is not None
                small = _is_tol_constant(value)
                # A bound shifted by a tolerance is itself a threshold.
                shifted = (not isinstance(function, ast.Module)
                           and isinstance(value, ast.BinOp) and analyzer.tol_like(value)
                           or (isinstance(value, ast.Call) and _name(value) == "nextafter"
                               and analyzer.tol_like(value)))
                for name in _targets(target):
                    if shifted:
                        names.tolerances.add(name)
                    if is_float:
                        names.floats.add(name)
                    if derived:
                        names.derived.add(name)
                    if small and isinstance(function, ast.Module):
                        names.tolerances.add(name)
    return names


@dataclass
class Site:
    file: str
    line: int
    end_line: int
    column: int
    function: str
    kind: str          # tolerance | ordering | provenance | library
    subkind: str
    text: str
    inventory_ids: list = field(default_factory=list)
    inventory_id: str = "unassigned"
    certified: bool = False
    certification: str = ""
    detail: str = ""


def _ops_ordering(compare):
    return any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in compare.ops)


def _ops_equality(compare):
    return any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in compare.ops)


def _only_tolerances(compare, analyzer):
    """A check on a tolerance value itself (e.g. ``tolerance <= 0``): not a site."""
    for sub in ast.walk(compare):
        if isinstance(sub, ast.Name) and not analyzer.tol_like(sub):
            if not isinstance(getattr(sub, "ctx", None), ast.Load):
                continue
            return False
        if isinstance(sub, ast.Call) and _name(sub) not in {"float", "abs", "finfo"}:
            if not analyzer.tol_like(sub.func):
                return False
    return True


def _is_zero(node):
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value == 0


_ID_ATTR = re.compile(r"(?i)(^|_)(id|ids|name|names|index|key|kind|label|handle)$")


def _identifier_key(key: ast.Lambda) -> bool:
    """``key=lambda x: x``, ``x.id``, ``str(x)``, ``len(x)``: an ordering by identity."""
    body = key.body
    parts = body.elts if isinstance(body, ast.Tuple) else [body]
    def plain(part):
        if isinstance(part, ast.Name):
            return True
        if isinstance(part, ast.Attribute):
            return bool(_ID_ATTR.search(part.attr))
        if isinstance(part, ast.Call):
            return _name(part) in {"str", "len", "id", "repr", "int"}
        return False
    return all(plain(part) for part in parts)


def _classify(node, analyzer) -> tuple[str, str, str] | None:
    """(kind, subkind, detail) of a candidate node, or None."""
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        if any((isinstance(o, ast.Constant) and (o.value is None or isinstance(o.value, (str, bytes))))
               or (isinstance(o, ast.Tuple) and not o.elts) for o in operands):
            return None
        if not (_ops_ordering(node) or _ops_equality(node)):
            return None
        if _ops_equality(node) and not _ops_ordering(node):
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                if analyzer.float_like(node.left):
                    return ("ordering", "float-equality", "membership test on a float")
                return None
            if any(analyzer.float_like(o) for o in operands):
                if analyzer.tol_like(node) and not _only_tolerances(node, analyzer):
                    return ("tolerance", "tolerance-compare", "")
                return ("ordering", "float-equality", "")
            return None
        if analyzer.tol_like(node):
            if _only_tolerances(node, analyzer):
                return None
            return ("tolerance", "tolerance-compare", "")
        if any(analyzer.float_like(o) for o in operands):
            if any(_is_zero(o) for o in operands):
                return ("ordering", "sign-branch", "")
            return ("ordering", "float-compare", "")
        return None
    if isinstance(node, ast.Call):
        name = _name(node)
        if name in ISCLOSE_CALLS:
            return ("tolerance", "isclose-call", "")
        if name in SELECT_CALLS or name in MINMAX_CALLS:
            if name == "clip" and _name(getattr(node.func, "value", None)) not in {"np", "numpy"}:
                return None
            key = next((k.value for k in node.keywords if k.arg == "key"), None)
            target = node.func.value if (name == "sort" and isinstance(node.func, ast.Attribute)) else None
            floaty = any(analyzer.float_like(a) for a in node.args) or (target is not None and analyzer.float_like(target))
            if isinstance(key, ast.Lambda):
                # Any key into records counts unless it is plainly an identifier.
                floaty = floaty or not _identifier_key(key)
            if name in {"argmin", "argmax", "argsort", "nanargmin", "nanargmax"}:
                floaty = True
            if not floaty:
                return None
            if name in SELECT_CALLS or key is not None:
                return ("ordering", "select-by-float-key", name)
            return ("ordering", "min-max-value", name)
        if name in ROUND_CALLS:
            return ("ordering", "round-floor", "")
        if name == "int" and node.args and analyzer.float_like(node.args[0]):
            return ("ordering", "round-floor", "int() of a float")
        if name in {"sign", "copysign"}:
            return ("ordering", "sign-branch", name)
        if name in LIBRARY_PREDICATES:
            return ("library", "library-predicate", name)
        if name in CONSTRUCTORS:
            source = None
            for arg in (*node.args, *(k.value for k in node.keywords)):
                source = source or analyzer.derived_from(arg)
            if source is not None:
                return ("provenance", "derived-construction", f"{name} from {source}")
            return None
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.FloorDiv)):
        if analyzer.float_like(node.left) or analyzer.float_like(node.right):
            return ("ordering", "round-floor", "modulo/floor-division of a float")
        return None
    if isinstance(node, (ast.If, ast.While, ast.IfExp)):
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
        scalar = (isinstance(test, ast.BinOp) or (isinstance(test, ast.Call) and _name(test) in
                  {"float", "abs", "norm", "sqrt", "fabs"}))
        if scalar and analyzer.float_like(test):
            return ("ordering", "sign-branch", "truthiness of a float")
    return None


def _functions(tree):
    """(qualname, node, parent qualname) for every function, outer first."""
    out = []

    def walk(node, prefix, parent):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + child.name
                out.append((qual, child, parent))
                walk(child, qual + ".", qual)
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".", parent)
            else:
                walk(child, prefix, parent)
    walk(tree, "", "<module>")
    return out


def _in_scope(qualname, wanted):
    if wanted is None:
        return True
    return any(qualname == w or qualname.startswith(w + ".") for w in wanted)


def _certified_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("exact_predicates"):
            names.update(a.asname or a.name for a in node.names if a.name in CERTIFIED_EXACT_PREDICATES)
    return names


def scan_source(source: str, file: str = "<snippet>", functions=None) -> list[Site]:
    """All decision sites of ``source`` (optionally only in ``functions``)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    certified_calls = _certified_names(tree)
    module_names = _infer_names(tree, _Names())
    names_by_function = {"<module>": module_names}
    units = [("<module>", tree)] if functions is None else []
    for qual, node, parent in _functions(tree):
        names_by_function[qual] = _infer_names(node, names_by_function.get(parent, module_names))
        if _in_scope(qual, functions):
            units.append((qual, node))
    statements = {}
    sites = []
    for qual, unit in units:
        analyzer = _Analyzer(names_by_function[qual])
        for node in _own_nodes(unit):
            if isinstance(node, ast.stmt):
                for sub in ast.walk(node):
                    statements[id(sub)] = node  # innermost statement wins (DFS order)
            found = _classify(node, analyzer)
            if found is None:
                continue
            kind, subkind, detail = found
            line, end = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
            if isinstance(node, (ast.If, ast.While)):
                end = node.test.end_lineno
            text = ast.get_source_segment(source, node.test if isinstance(node, (ast.If, ast.While)) else node) or ""
            text = " ".join(text.split())
            if len(text) > 240:
                text = text[:237] + "..."
            site = Site(file, line, end, node.col_offset, qual, kind, subkind, text, detail=detail)
            span = lines[max(0, line - 2):end]
            if any(COORDINATE_ONLY.search(item) for item in span):
                site.certified, site.certification = True, "coordinate-only"
            statement = statements.get(id(node))
            if statement is not None and certified_calls | CERTIFIED_EXACT_PREDICATES:
                for sub in ast.walk(statement):
                    if isinstance(sub, ast.Call) and (
                            (isinstance(sub.func, ast.Name) and sub.func.id in certified_calls)
                            or (isinstance(sub.func, ast.Attribute) and sub.func.attr in CERTIFIED_EXACT_PREDICATES
                                and _name(sub.func.value) == "exact_predicates")):
                        site.certified, site.certification = True, "exact_predicates"
            sites.append(site)
    sites.sort(key=lambda s: (s.line, s.column, s.kind))
    return sites


def _function_spans(tree):
    return {qual: (node.lineno, node.end_lineno) for qual, node, _ in _functions(tree)}


def assign_inventory(sites, module, source, inventory=INVENTORY):
    """Fill ``inventory_ids``/``inventory_id`` of ``sites`` from ``inventory``."""
    lines = source.splitlines()
    spans = _function_spans(ast.parse(source))
    for site in sites:
        site.inventory_ids = []
        full_line = " ".join(lines[site.line - 1:site.end_line])
        for entry in inventory:
            if entry.module != module or entry.id in site.inventory_ids:
                continue
            owner = next((q for q in _prefixes(site.function) if re.fullmatch(entry.function, q)), None)
            if owner is None:
                continue
            if entry.text is not None and not (re.search(entry.text, site.text) or re.search(entry.text, full_line)):
                continue
            if entry.region is not None:
                start, end = spans.get(owner, (1, len(lines)))
                first = next((i for i in range(start, end + 1) if re.search(entry.region[0], lines[i - 1])), None)
                if first is None:
                    continue
                last = next((i for i in range(first, end + 1) if re.search(entry.region[1], lines[i - 1])), end)
                if not first <= site.line <= last:
                    continue
            site.inventory_ids.append(entry.id)
        site.inventory_id = site.inventory_ids[0] if site.inventory_ids else "unassigned"
    return sites


def _prefixes(qualname):
    """``a.b.c`` -> ``a.b.c``, ``a.b``, ``a`` (a site in a nested function also
    belongs to the enclosing function)."""
    parts = qualname.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def run_audit(root: Path = ROOT, scope=SCOPE) -> dict:
    """Scan ``scope`` under ``root`` and return the JSON-serialisable report."""
    root = Path(root)
    sites, missing = [], []
    for module, functions in scope.items():
        path = root / PACKAGE / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        found = scan_source(source, f"{PACKAGE.as_posix()}/{module}.py", functions)
        if functions is not None:
            present = {q for q, _n, _p in _functions(ast.parse(source))}
            missing.extend(f"{module}.{f}" for f in functions if f not in present)
        sites.extend(assign_inventory(found, module, source))
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    uncertified = [s for s in sites if not s.certified]
    assigned_ids = {i for s in sites for i in s.inventory_ids}
    return {
        "tool": "tools/decision_audit.py",
        "commit": commit,
        "heuristic": True,
        "scope": {m: (list(f) if f is not None else "whole module") for m, f in scope.items()},
        "missing_scope_functions": missing,
        "totals": {
            "sites": len(sites),
            "uncertified": len(uncertified),
            "certified": len(sites) - len(uncertified),
            "unassigned": sum(1 for s in sites if s.inventory_id == "unassigned"),
        },
        "by_kind": dict(Counter(s.kind for s in sites)),
        "by_subkind": dict(Counter(f"{s.kind}/{s.subkind}" for s in sites)),
        "by_module_kind": {m: dict(Counter(s.kind for s in sites if s.file.endswith(f"/{m}.py")))
                           for m in scope},
        "by_inventory_id": dict(sorted(Counter(i for s in sites for i in s.inventory_ids).items(),
                                       key=lambda kv: _id_key(kv[0]))),
        "inventory_ids_without_sites": sorted({e.id for e in INVENTORY} - assigned_ids, key=_id_key),
        "sites": [asdict(s) for s in sites],
    }


def _id_key(identifier):
    match = re.match(r"([A-Z]+)(\d+)", identifier)
    return (match.group(1), int(match.group(2))) if match else (identifier, 0)


def markdown_summary(report: dict, unassigned_limit: int | None = None) -> str:
    out = [f"# R0 decision-site audit (commit {report['commit'][:10]})", "",
           "Heuristic AST scan (tools/decision_audit.py); reporting mode.", ""]
    t = report["totals"]
    out += [f"- sites: {t['sites']} (uncertified {t['uncertified']}, certified {t['certified']})",
            f"- unassigned to an inventory entry: {t['unassigned']}", ""]
    kinds = ["tolerance", "ordering", "provenance", "library"]
    unassigned_by_module = Counter(Path(s["file"]).stem for s in report["sites"]
                                   if s["inventory_id"] == "unassigned")
    out += ["| module | " + " | ".join(kinds) + " | total | unassigned |", "|---|" + "---:|" * (len(kinds) + 2)]
    for module, counts in report["by_module_kind"].items():
        out.append(f"| {module} | " + " | ".join(str(counts.get(k, 0)) for k in kinds)
                   + f" | {sum(counts.values())} | {unassigned_by_module.get(module, 0)} |")
    out.append("| **total** | " + " | ".join(str(report["by_kind"].get(k, 0)) for k in kinds)
               + f" | {t['sites']} | {t['unassigned']} |")
    out += ["", "Subkinds: " + ", ".join(f"{k} {v}" for k, v in sorted(report["by_subkind"].items())), "",
            "Sites per inventory id: " + ", ".join(f"{k} {v}" for k, v in report["by_inventory_id"].items()), ""]
    if report["inventory_ids_without_sites"]:
        out += ["Inventory ids with no detected site: " + ", ".join(report["inventory_ids_without_sites"]), ""]
    if report["missing_scope_functions"]:
        out += ["Scope functions not found: " + ", ".join(report["missing_scope_functions"]), ""]
    unassigned = [s for s in report["sites"] if s["inventory_id"] == "unassigned"]
    by_function = Counter((Path(s["file"]).name, s["function"]) for s in unassigned)
    out += [f"## Unassigned sites ({len(unassigned)})", "",
            "Candidates for the inventory (R.1). Per function:", "",
            "| module | function | sites |", "|---|---|---:|"]
    out += [f"| {m} | {f} | {n} |" for (m, f), n in sorted(by_function.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1]))]
    out += ["", "All unassigned sites:", "",
            "| file:line | function | kind/subkind | source |", "|---|---|---|---|"]
    for site in unassigned[:unassigned_limit]:
        text = site["text"].replace("|", "\\|")
        out.append(f"| {Path(site['file']).name}:{site['line']} | {site['function']} | "
                   f"{site['kind']}/{site['subkind']} | `{text}` |")
    if unassigned_limit is not None and len(unassigned) > unassigned_limit:
        out.append(f"| ... | {len(unassigned) - unassigned_limit} more in the JSON report | | |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path, help="write the JSON report here")
    parser.add_argument("--markdown", type=Path, help="write the Markdown summary here")
    parser.add_argument("--fail", action="store_true", help="exit 1 when an uncertified site exists")
    args = parser.parse_args(argv)
    report = run_audit(args.root)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    summary = markdown_summary(report)
    if args.markdown:
        args.markdown.write_text(summary, encoding="utf-8")
    t = report["totals"]
    print(f"decision sites: {t['sites']} uncertified: {t['uncertified']} unassigned: {t['unassigned']} "
          f"by kind: {report['by_kind']}")
    return 1 if args.fail and t["uncertified"] else 0


if __name__ == "__main__":
    sys.exit(main())
