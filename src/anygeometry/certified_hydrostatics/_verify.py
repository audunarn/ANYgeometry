"""Independent verifier of certified hydrostatics results (design section 6, slice CH1c).

Written from the design specification only: nothing here uses the snapshot, admission, section, form or query
code of the provider. ``verify_certified_hydrostatics`` re-derives every claim in exact rational arithmetic from
the model and the face list and compares it with the complete result object.

Steps (design section 6): (0) the bound query equals the caller's query; (1) the proof is parsed (bounded, metered),
recomputed from the result object and compared with ``expected``; (2) the snapshot, admission, model identity and
artifact descriptor are rebuilt from the model and the running environment; (3) V, J, contacts, occupancies,
cut points, segments, loops, topology and the waterplane quantities are recomputed; (4) exact equality and the
form rules; (5) completeness. Any discrepancy raises ``VERIFICATION_FAILED`` with ``error.sub_reason``.

Structural rules this module keeps (and a source-inspection test enforces):
* Trusted values only. Every value that comes from outside (the query, the result object, ``expected``, the face
  list, the policy) is read once at entry and rebuilt from exact built-in types (``type(x) is int`` / ``float`` /
  ``str`` / ``bool`` / ``bytes`` / ``Fraction``; never ``isinstance``), so a subclass that overrides ``__eq__``,
  ``__iter__`` or a property cannot influence a decision. After entry only trusted values flow on.
* Every exact value is a ``_Q`` whose arithmetic and ordering products are checked against the bit cap as they are
  formed; the dunders the verifier does not use raise.
* Every loop, comprehension, generator and data builtin over input-sized data ticks once per element (``_each``),
  so the work between two checkpoints does not grow with the data.
* Structured comparisons of parsed JSON use ``_same`` (type exact: ``true`` is not ``1``).
* The only operations allowed to run as one span larger than a unit are the two listed in design amendment 6b: the
  canonical encoding of the model document for ``document_checksum`` (charged one unit per model entity first) and
  one UTF-8 decode of the proof (bounded by ``max_proof_bytes``). After a refusal no outside data are read.
* The returned ``VerifiedHydrostatics`` holds only checked or recomputed values; it forwards nothing.

Conventions used here (the specification leaves them open):
* ``expected`` is a mapping with ``artifact_files`` (sequence of ``(relative path, sha256 hex)``; an
  ``ImplementationIdentity`` or its dict under ``artifact`` is also accepted), ``convention_version``, ``units``
  and ``policy`` (a policy object or its dict).
* the snapshot digest is SHA-256 of the canonical JSON (design 6a) of the document built in
  ``_snapshot_document``.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata as _metadata
import json
import math
import operator
import os
import platform
import re
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Mapping

from ..curves import Straight
from ..surfaces import Plane
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._forms import QuantityResult
from ._query import HalfSpaceQuery
from ._result import CertifiedHydrostaticsResult, Contacts, SectionResult, Topology, TopologyLoop
from ._work import Meter, WorkBudget

__all__ = ["VerifiedHydrostatics", "verify_certified_hydrostatics"]

C = RefusalCategory
_CHUNK = 65536
_WINDOW = 4096          # characters scanned or encoded per checkpoint
_QUANTITIES = ("A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")
_MOMENTS = ("Jx", "Jy", "Jz")
_SUB_REASONS = ("artifact", "model", "face set", "snapshot", "convention", "policy", "query", "request", "value",
                "enclosure", "accuracy flag", "state", "topology")


# ------------------------------------------------------------------ refusals


def _fail(sub: str, message: str, cause: BaseException | None = None) -> CertifiedHydrostaticsRefusal:
    assert sub in _SUB_REASONS
    error = CertifiedHydrostaticsRefusal(C.VERIFICATION_FAILED, f"[{sub}] {message}")
    error.sub_reason = sub
    if cause is not None:
        error.__cause__ = cause
    return error


def _refuse(category: RefusalCategory, message: str) -> CertifiedHydrostaticsRefusal:
    error = CertifiedHydrostaticsRefusal(category, message)
    error.sub_reason = None
    return error


_RESOURCE = (C.BUDGET_EXHAUSTED, C.DEADLINE_EXCEEDED, C.CANCELLED, C.BIT_LENGTH_EXCEEDED)


# ------------------------------------------------------------------ the running verification: checkpoints and cap

_ACTIVE = threading.local()


def _fit(x):
    """Check an exact value against the bit-length cap of the running verification.

    Fast path: the budget's high-water mark is compared directly; the budget itself decides (and raises) whenever the
    value exceeds the largest length seen so far."""
    budget = getattr(_ACTIVE, "budget", None)
    if budget is not None:
        if isinstance(x, Fraction):
            bits = max(x.numerator.bit_length(), x.denominator.bit_length())
        else:
            bits = int(x).bit_length()
        if bits > budget._bits or (budget.max_bits is not None and bits > budget.max_bits):
            budget.check_value(x)
    return x


def _tick(kind: str = "element") -> None:
    """One checkpoint (budget, deadline, cancellation) for one processed element."""
    gauge = getattr(_ACTIVE, "gauge", None)
    if gauge is not None:
        gauge.unit(kind)


def _each(items, kind: str = "element"):
    """Iterate ``items`` with one checkpoint before each element is handed out."""
    for item in items:
        _tick(kind)
        yield item


def _charge_text(length: int, kind: str = "string_bulk") -> None:
    """Design 6b: one linear copy, compare, escape or encode of a single string scalar (bounded by the proof's byte
    limit) is one bulk operation, charged one unit per 64 KiB before it runs."""
    for _ in range(length // _CHUNK + 1):
        _tick(kind)


def _joined(parts, separator: str = "") -> str:
    pieces = []
    length = 0
    for piece in _each(parts, "join_piece"):
        pieces.append(piece)
        length += len(piece) + len(separator)
    _charge_text(length)                 # the copy into one string, charged before it runs (6b)
    return separator.join(_each(pieces, "join_copy"))


def _tsorted(items, key=None):
    """``sorted`` with one checkpoint per element copied and per comparison (the sort itself is C code, so the
    comparator ticks)."""
    gauge = getattr(_ACTIVE, "gauge", None)

    def compare(a, b):
        if gauge is not None:
            gauge.unit("sort_comparison")
        ka, kb = (a, b) if key is None else (key(a), key(b))
        return (ka > kb) - (ka < kb)

    return sorted([x for x in _each(items, "sort_element")], key=functools.cmp_to_key(compare))


class _Q(Fraction):
    """A rational whose every arithmetic result and every ordering product is checked against the bit cap as it is
    formed. All exact values of the verifier are ``_Q``; mixed operations with plain Fractions or ints stay guarded
    because ``_Q`` is a subclass (its reflected operators run first). Operations the verifier does not use raise."""

    __slots__ = ()


def _wrap(result):
    if isinstance(result, Fraction):
        _fit(result)
        return _Q(result.numerator, result.denominator)
    return result


def _guarded(name):
    base = getattr(Fraction, name)

    def operation(self, *args):
        return _wrap(base(self, *args))

    operation.__name__ = name
    return operation


def _ordered(name, compare):
    base = getattr(Fraction, name)

    def operation(self, other):
        if isinstance(other, Fraction):
            other_num, other_den = other.numerator, other.denominator
        elif isinstance(other, int):
            other_num, other_den = other, 1
        else:
            return base(self, other)
        left, right = self.numerator * other_den, other_num * self.denominator
        _fit(left)
        _fit(right)
        return compare(left, right)

    operation.__name__ = name
    return operation


def _forbidden(name):
    def operation(self, *args, **kwargs):
        raise TypeError(f"{name} is not available on guarded rationals")

    operation.__name__ = name
    return operation


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
              "__neg__", "__pos__", "__abs__"):
    setattr(_Q, _name, _guarded(_name))
for _name, _op in (("__lt__", operator.lt), ("__le__", operator.le), ("__gt__", operator.gt), ("__ge__", operator.ge)):
    setattr(_Q, _name, _ordered(_name, _op))
for _name in ("__floordiv__", "__rfloordiv__", "__mod__", "__rmod__", "__divmod__", "__rdivmod__", "__pow__",
              "__rpow__", "__round__", "__trunc__", "__floor__", "__ceil__", "__int__", "__float__", "__complex__",
              "limit_denominator"):
    setattr(_Q, _name, _forbidden(_name))

Z = _Q(0)
ONE = _Q(1)
HALF = _Q(1, 2)


def _q(value) -> "_Q":
    """A guarded rational from an int, float or Fraction (checked against the cap)."""
    return _fit(_Q(value))


def _pair(x) -> list:
    _fit(x)
    return [x.numerator, x.denominator]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _argmax_abs(v) -> int:
    best = 0
    for i in (1, 2):
        if abs(v[i]) > abs(v[best]):
            best = i
    return best


def _sign(x) -> int:
    return (x > 0) - (x < 0)


# ------------------------------------------------------------------ integers in chunks, canonical JSON

_INT_CHUNK = 300


def _int_from_digits(digits: str) -> int:
    """Convert a digit string in chunks of at most 300 digits (independent of Python's global digit limit); the
    partial value is checked against the bit cap after every chunk."""
    value = 0
    for i in range(0, len(digits), _INT_CHUNK):
        _tick("integer_chunk")
        chunk = digits[i:i + _INT_CHUNK]
        value = value * 10 ** len(chunk) + int(chunk)
        _fit(value)
    return value


def _int_to_digits(value: int) -> str:
    negative = value < 0
    value = abs(value)
    parts = []
    base = 10 ** _INT_CHUNK
    while value >= base:
        _tick("integer_chunk")
        value, rest = divmod(value, base)
        parts.append(f"{rest:0{_INT_CHUNK}d}")
    _tick("integer_chunk")
    return ("-" if negative else "") + str(value) + _joined(_reversed_list(parts))


def _reversed_list(items):
    """The items in reverse order (one checkpoint per element)."""
    out = []
    for item in _each(items, "reverse"):
        out.append(item)
    out.reverse()
    return out


def _json_string_pieces(text: str):
    """``json.dumps(text, ensure_ascii=False)`` as pieces: the opening quote, the escaped text window by window (at most
    ``_WINDOW`` characters per checkpoint), the closing quote. Escaping is per character, so the concatenation equals
    the whole-string result exactly (a window boundary never falls inside an escape sequence)."""
    yield '"'
    for k in range(0, len(text), _WINDOW):
        _tick("escape_window")
        yield json.dumps(text[k:k + _WINDOW], ensure_ascii=False)[1:-1]
    yield '"'


def _encode_pieces(node, depth):
    """Canonical JSON text of ``node`` (design 6a) as a lazy stream of pieces; one checkpoint per element, per
    window of a long string and per chunk of a long integer."""
    _tick("serialise_element")
    if depth > 64:
        raise _refuse(C.MALFORMED_PROOF, "nesting too deep")
    if node is None:
        yield "null"
    elif node is True:
        yield "true"
    elif node is False:
        yield "false"
    elif type(node) is int:
        yield _int_to_digits(node)
    elif type(node) is str:
        yield from _json_string_pieces(node)
    elif type(node) in (list, tuple):
        yield "["
        first = True
        for item in _each(node, "serialise_element"):
            if not first:
                yield ","
            first = False
            yield from _encode_pieces(item, depth + 1)
        yield "]"
    elif type(node) is dict:
        yield "{"
        first = True
        for key in _tsorted(node):
            if not first:
                yield ","
            first = False
            yield from _json_string_pieces(key)
            yield ":"
            yield from _encode_pieces(node[key], depth + 1)
        yield "}"
    else:
        raise _refuse(C.MALFORMED_PROOF, f"cannot serialise {type(node).__name__}")


def _encode_windows(document):
    """The canonical UTF-8 encoding of ``document`` as a lazy stream of byte windows of at most ``_WINDOW``
    characters each (one checkpoint per window); nothing is joined."""
    try:
        for piece in _each(_encode_pieces(document, 0), "encode_piece"):
            for k in range(0, len(piece), _WINDOW):
                _tick("encode_window")
                yield piece[k:k + _WINDOW].encode("utf-8")
    except UnicodeEncodeError as error:
        raise _refuse(C.MALFORMED_PROOF, "the document is not valid UTF-8 text") from error


def _sha256_windows(windows, kind: str) -> str:
    h = hashlib.sha256()
    for window in windows:
        _tick(kind)
        h.update(window)
    return h.hexdigest()


def _equals_windows(windows, data: bytes) -> bool:
    """Whether the stream of byte windows spells exactly ``data``; compared window by window with one checkpoint per
    window, stopping at the first difference."""
    position = 0
    for window in windows:
        _tick("compare_element")
        end = position + len(window)
        if data[position:end] != window:
            return False
        position = end
    return position == len(data)


def _same(a, b) -> bool:
    """Type-exact structural equality of JSON-like values (``True`` is not ``1``, a list is not a tuple)."""
    _tick("compare_element")
    if type(a) in (list, tuple) and type(b) in (list, tuple):
        pass
    elif type(a) is not type(b):
        return False
    if type(a) is dict:
        if len(a) != len(b):
            return False
        for key in _each(a, "compare_element"):
            if key not in b or not _same(a[key], b[key]):
                return False
        return True
    if type(a) in (list, tuple):
        if len(a) != len(b):
            return False
        for x, y in _each(zip(a, b), "compare_element"):
            if not _same(x, y):
                return False
        return True
    if type(a) is str and type(b) is str:
        _charge_text(min(len(a), len(b)))
    return a == b


def _fraction_of_float(value) -> Fraction:
    x = float(value)
    if not math.isfinite(x):
        raise _refuse(C.NONFINITE_INPUT, "a stored coordinate is not finite")
    return _q(x)


# ------------------------------------------------------------------ polynomials in epsilon (simulation of simplicity)
# These operate on coefficient tuples of at most three terms: constant-size iteration.


def _padd(p, q):
    n = max(len(p), len(q))
    return tuple((p[i] if i < len(p) else Z) + (q[i] if i < len(q) else Z) for i in range(n))


def _psub(p, q):
    n = max(len(p), len(q))
    return tuple((p[i] if i < len(p) else Z) - (q[i] if i < len(q) else Z) for i in range(n))


def _pmul(p, q):
    out = [Z] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        if x:
            for j, y in enumerate(q):
                out[i + j] = out[i + j] + x * y
    return tuple(out)


def _psign(p) -> int:
    for c in p:
        if c:
            return 1 if c > 0 else -1
    return 0


# ------------------------------------------------------------------ entry: trusted copies of every outside value


def _seq(x, sub: str, what: str):
    """A list or tuple of exactly these built-in types (never a subclass with its own iteration)."""
    if type(x) is not tuple and type(x) is not list:
        raise _fail(sub, f"{what} is not a list or tuple")
    return x


def _rat(x, sub: str, what: str, *, integers: bool = True) -> "_Q":
    """A guarded rational from an exact ``Fraction`` (or exact ``int``); anything else, subclasses included, is
    refused. The value is rebuilt from its integer parts."""
    t = type(x)
    if t is int and integers:
        return _q(x)
    if t is Fraction:
        n, d = x.numerator, x.denominator
        if type(n) is int and type(d) is int and d > 0:
            return _fit(_Q(n, d))
    raise _fail(sub, f"{what} is not an exact rational")


def _int(x, sub: str, what: str) -> int:
    if type(x) is not int:
        raise _fail(sub, f"{what} is not an integer")
    return x


def _str(x, sub: str, what: str) -> str:
    if type(x) is not str:
        raise _fail(sub, f"{what} is not a string")
    return x


def _ints(items, sub: str, what: str) -> tuple:
    out = []
    for x in _each(_seq(items, sub, what), "ingest"):
        out.append(_int(x, sub, what))
    return tuple(_each(out, "ingest"))


def _vec3(items, sub: str, what: str) -> tuple:
    items = _seq(items, sub, what)
    if len(items) != 3:
        raise _fail(sub, f"{what} does not have three components")
    return (_rat(items[0], sub, what), _rat(items[1], sub, what), _rat(items[2], sub, what))


class _Req:
    """A request (half-space, basis, accuracies, forms) rebuilt from exact built-in types."""

    __slots__ = ("normal", "offset", "origin", "e1", "e2", "accuracy", "accuracy_pairs", "forms", "want_binary64",
                 "want_decimal")

    def accuracy_of(self, name: str):
        for key, value in _each(self.accuracy, "ingest"):
            if key == name:
                return value
        if name in _MOMENTS:
            for key, value in _each(self.accuracy, "ingest"):
                if key == "J":
                    return value
        return None


def _ingest_query(q) -> _Req:
    r = _Req()
    r.normal = _vec3(q.normal, "query", "the normal")
    r.offset = _rat(q.offset, "query", "the offset")
    r.origin = _vec3(q.waterplane_origin, "query", "the waterplane origin")
    axes = _seq(q.waterplane_axes, "query", "the waterplane axes")
    if len(axes) != 2:
        raise _fail("query", "the waterplane axes are not two vectors")
    r.e1, r.e2 = _vec3(axes[0], "query", "an axis"), _vec3(axes[1], "query", "an axis")
    pairs = []
    for item in _each(_seq(q.absolute_accuracy, "query", "the accuracies"), "ingest"):
        item = _seq(item, "query", "an accuracy entry")
        if len(item) != 2:
            raise _fail("query", "an accuracy entry is not a pair")
        pairs.append((_str(item[0], "query", "an accuracy name"), _rat(item[1], "query", "an accuracy")))
    r.accuracy = tuple(_each(pairs, "ingest"))
    # plain integer pairs, prepared while ingesting, so a refusal can report them without further work
    r.accuracy_pairs = tuple((name, (v.numerator, v.denominator)) for name, v in _each(r.accuracy, "ingest"))
    forms = []
    for form in _each(_seq(q.forms, "query", "the forms"), "ingest"):
        forms.append(_str(form, "query", "a form"))
    r.forms = tuple(_each(forms, "ingest"))
    return r


def _req_dict(r: _Req) -> dict:
    return {"absolute_accuracy": {name: _pair(v) for name, v in _each(r.accuracy)},
            "forms": [f for f in _each(r.forms)],
            "normal": [_pair(x) for x in _each(r.normal)], "offset": _pair(r.offset),
            "waterplane_axes": [[_pair(x) for x in _each(r.e1)], [_pair(x) for x in _each(r.e2)]],
            "waterplane_origin": [_pair(x) for x in _each(r.origin)]}


class _QR:
    """One quantity of the result."""

    __slots__ = ("name", "numerator", "norm_squared", "binary64", "decimal", "accuracy", "accuracy_met")


class _Sec:
    """One side of the result."""

    __slots__ = ("side", "occupancy", "section", "quantities", "topology")


class _Res:
    """The result object, rebuilt from exact built-in types."""

    __slots__ = ("query", "occupancy", "generic", "contacts", "volume", "moments", "sections", "ops", "units", "proof")


def _ingest_quantity(item) -> _QR:
    """Read every attribute of the quantity exactly once and keep only validated exact built-in values."""
    out = _QR()
    name, numerator, norm_squared = item.name, item.numerator, item.norm_squared
    accuracy, flag, binary, decimal = item.accuracy, item.accuracy_met, item.binary64, item.decimal
    out.name = _str(name, "request", "a quantity name")
    out.numerator = None if numerator is None else _rat(numerator, "value", "an exact numerator", integers=False)
    out.norm_squared = None if norm_squared is None else _rat(norm_squared, "value", "an exact |n|^2", integers=False)
    out.accuracy = None if accuracy is None else _rat(accuracy, "request", "a requested accuracy", integers=False)
    if flag is not None and type(flag) is not bool:
        raise _fail("accuracy flag", "an accuracy flag is not a boolean")
    out.accuracy_met = flag
    if binary is None:
        out.binary64 = None
    else:
        binary = _seq(binary, "enclosure", "a binary64 enclosure")
        if len(binary) != 2 or type(binary[0]) is not float or type(binary[1]) is not float:
            raise _fail("enclosure", "binary64 bounds are not doubles")
        out.binary64 = (binary[0], binary[1])
    if decimal is None:
        out.decimal = None
    else:
        decimal = _seq(decimal, "enclosure", "a decimal enclosure")
        if len(decimal) != 3:
            raise _fail("enclosure", "a decimal enclosure is not a triple")
        out.decimal = (_str(decimal[0], "enclosure", "a decimal bound"), _str(decimal[1], "enclosure", "a decimal bound"),
                       _rat(decimal[2], "enclosure", "a decimal width", integers=False))
    return out


def _ingest_work(work):
    if type(work) is not dict:
        raise _fail("value", "the reported work counts are not a mapping")
    ops, units = work.get("operations"), work.get("units")
    if type(ops) is not int or ops < 0 or type(units) is not dict:
        raise _fail("value", "the reported work counts are malformed")
    total = 0
    copy = {}
    for key in _each(units, "ingest"):
        value = units[key]
        if type(key) is not str or type(value) is not int or value < 0:
            raise _fail("value", "the reported work counts are not non-negative integers")
        _fit(value)
        total += value
        copy[key] = value
    if ops != total:
        raise _fail("value", "the reported operations differ from the sum of the unit counts")
    return _fit(ops), copy


def _ingest_section(section) -> _Sec:
    out = _Sec()
    out.side = _str(section.side, "state", "a side name")
    out.occupancy = _str(section.occupancy, "state", "a side occupancy")
    out.section = _str(section.section, "state", "a section flag")
    quantities = section.quantities
    if quantities is None:
        out.quantities = None
    else:
        out.quantities = tuple(_ingest_quantity(x) for x in _each(_seq(quantities, "request", "the quantities"), "ingest"))
    topology = section.topology
    if topology is None:
        out.topology = None
    else:
        loops, components = [], []
        for loop in _each(_seq(topology.loops, "topology", "the loops"), "ingest"):
            loops.append((_ints(loop.identity, "topology", "a loop identity"), _str(loop.role, "topology", "a loop role")))
        for component in _each(_seq(topology.components, "topology", "the components"), "ingest"):
            component = _seq(component, "topology", "a component")
            if len(component) != 2:
                raise _fail("topology", "a component is not an (outer, holes) pair")
            holes = []
            for hole in _each(_seq(component[1], "topology", "the holes"), "ingest"):
                holes.append(_ints(hole, "topology", "a hole identity"))
            components.append((_ints(component[0], "topology", "an outer identity"), tuple(_each(holes, "ingest"))))
        out.topology = (tuple(_each(loops, "ingest")), tuple(_each(components, "ingest")))
    return out


def _ingest_result(result) -> _Res:
    """Read every field of the result once. The query is ingested separately (step 0)."""
    out = _Res()
    out.occupancy = _str(result.occupancy, "state", "the occupancy")
    generic = result.generic
    if type(generic) is not bool:
        raise _fail("value", "the degeneracy flag is not a boolean")
    out.generic = generic
    contacts = result.contacts
    out.contacts = (_ints(contacts.vertices, "value", "a contact vertex"), _ints(contacts.edges, "value", "a contact edge"),
                    _ints(contacts.faces, "value", "a contact face"))
    out.volume = _ingest_quantity(result.volume)
    out.moments = tuple(_ingest_quantity(x) for x in _each(_seq(result.moments, "request", "the moments"), "ingest"))
    out.sections = tuple(_ingest_section(x) for x in _each(_seq(result.sections, "state", "the sections"), "ingest"))
    out.ops, out.units = _ingest_work(result.work)
    proof = result.proof
    if type(proof) is not bytes:
        raise _refuse(C.MALFORMED_PROOF, "the proof is not bytes")
    out.proof = proof
    return out


def _ingest_faces(faces) -> tuple:
    out = []
    for f in _each(faces, "ingest"):
        if type(f) is not int:
            try:
                f = int.__index__(operator.index(f))
            except TypeError as error:
                raise _fail("face set", "a face id is not an integer", error) from error
        out.append(f)
    return tuple(_each(out, "ingest"))


def _policy_dict(policy) -> dict:
    if isinstance(policy, Mapping):
        source = policy
        get = lambda k: source[k]
    else:
        get = lambda k: getattr(policy, k)
    version, digits, size = get("convention_version"), get("max_integer_digits"), get("max_proof_bytes")
    if type(version) is not str or type(digits) is not int or type(size) is not int:
        raise ValueError("the expected policy has malformed fields")
    return {"convention_version": version, "max_integer_digits": digits, "max_proof_bytes": size}


def _sorted_pairs(files, *, expected: bool = False) -> list:
    rows = []
    for pair in _each(files, "file_row"):
        if type(pair) not in (tuple, list) or len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not str:
            if expected:
                raise ValueError("an expected file entry is not a (path, hash) pair of strings")
            raise _refuse(C.MALFORMED_PROOF, "a file entry is not a (path, hash) pair of strings")
        rows.append([pair[0], pair[1]])
    return _tsorted(rows)


def _expected_identity(expected) -> dict:
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a mapping with artifact_files, convention_version, units and policy")
    if "artifact_files" in expected:
        files = expected["artifact_files"]
    elif "artifact" in expected:
        art = expected["artifact"]
        files = art["files"] if isinstance(art, Mapping) else art.files
    else:
        raise ValueError("expected lacks the artifact file hashes")
    for key in ("convention_version", "units", "policy"):
        if key not in expected:
            raise ValueError(f"expected lacks {key}")
    version, units = expected["convention_version"], expected["units"]
    if type(version) is not str or type(units) is not str:
        raise ValueError("the expected convention version and units must be strings")
    return {"files": _sorted_pairs(files, expected=True), "convention_version": version, "units": units,
            "policy": _policy_dict(expected["policy"])}


# ------------------------------------------------------------------ the exact snapshot (own implementation)


class _Face:
    __slots__ = ("id", "loops", "support", "normal", "w", "axis", "loops2d", "vids", "eids", "box")


class _Snapshot:
    def __init__(self):
        self.faces: list[_Face] = []
        self.pos: dict[int, tuple] = {}
        self.edges: dict[int, tuple[int, int]] = {}


def _point_of(vector) -> tuple:
    return (_fraction_of_float(vector[0]), _fraction_of_float(vector[1]), _fraction_of_float(vector[2]))


def _read_snapshot(model, faces) -> _Snapshot:
    snap = _Snapshot()
    for fid in _each(faces, "snapshot_read"):
        if fid not in model.faces:
            raise _refuse(C.INVALID_LOOP, f"face {fid} is not in the model")
        face = model.faces[fid]
        surface = face.surface
        if not isinstance(surface, Plane):
            raise _refuse(C.UNSUPPORTED_SURFACE, f"face {fid} is not supported by a Plane")
        f = _Face()
        f.id = fid
        f.support = (_point_of(surface.origin), _point_of(surface.u_vector), _point_of(surface.v_vector))
        f.loops = []
        for loop in _each([face.loop, *face.holes], "snapshot_read"):
            items = []
            for oriented in _each(loop, "snapshot_read"):
                edge = model.edges[oriented.edge]
                if not isinstance(edge.curve, Straight):
                    raise _refuse(C.UNSUPPORTED_SURFACE, f"edge {edge.id} is not straight")
                start, end = (edge.start, edge.end) if oriented.forward else (edge.end, edge.start)
                for vid in (edge.start, edge.end):
                    if vid not in snap.pos:
                        snap.pos[vid] = _point_of(model.vertex_position(vid))
                if snap.edges.setdefault(edge.id, (edge.start, edge.end)) != (edge.start, edge.end):
                    raise _refuse(C.INVALID_LOOP, "inconsistent edge record")
                items.append((edge.id, bool(oriented.forward), start, end))
            if len(items) < 3:
                raise _refuse(C.INVALID_LOOP, f"a loop of face {fid} has fewer than three edges")
            for i, item in _each(enumerate(items), "snapshot_close"):
                if item[3] != items[(i + 1) % len(items)][2]:
                    raise _refuse(C.NOT_CLOSED, f"a loop of face {fid} is not a closed chain")
            f.loops.append(items)
        snap.faces.append(f)
    return snap


def _pair3(vec) -> list:
    return [_pair(vec[0]), _pair(vec[1]), _pair(vec[2])]


def _snapshot_document(snap: _Snapshot) -> dict:
    faces = []
    for f in _each(snap.faces, "snapshot_document"):
        loops = []
        for loop in _each(f.loops, "snapshot_document"):
            row = []
            for eid, fwd, start, _end in _each(loop, "snapshot_document"):
                row.append([eid, fwd, start])
            loops.append(row)
        support = [_pair3(f.support[0]), _pair3(f.support[1]), _pair3(f.support[2])]
        faces.append({"id": f.id, "support": support, "loops": loops})
    uses: dict[int, list] = {}
    vertex_faces: dict[int, set] = {}
    vertex_edges: dict[int, set] = {}
    for f in _each(snap.faces, "snapshot_document"):
        for loop in _each(f.loops, "snapshot_document"):
            for eid, fwd, start, end in _each(loop, "snapshot_document"):
                uses.setdefault(eid, []).append([f.id, fwd])
                for vid in (start, end):
                    vertex_faces.setdefault(vid, set()).add(f.id)
                    vertex_edges.setdefault(vid, set()).add(eid)
    vertices, edges, incidence = [], [], []
    for vid in _each(_tsorted(vertex_faces), "snapshot_document"):
        p = snap.pos[vid]
        vertices.append([vid, [_pair(p[0]), _pair(p[1]), _pair(p[2])]])
        incidence.append([vid, _tsorted(vertex_faces[vid]), _tsorted(vertex_edges[vid])])
    for eid in _each(_tsorted(uses), "snapshot_document"):
        edges.append([eid, snap.edges[eid][0], snap.edges[eid][1], _tsorted(uses[eid])])
    return {"faces": faces, "vertices": vertices, "edges": edges, "vertex_incidence": incidence}


def _sha256_chunks(data: bytes, kind: str) -> str:
    h = hashlib.sha256()
    for i in range(0, max(len(data), 1), _CHUNK):
        _tick(kind)
        h.update(data[i:i + _CHUNK])
    return h.hexdigest()


# ------------------------------------------------------------------ admission (section 2), own implementation


def _project(point, axis):
    return (point[(axis + 1) % 3], point[(axis + 2) % 3])


def _orient2(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment2(p, q, r) -> bool:
    """p on the closed segment q-r (collinearity assumed checked by the caller)."""
    return min(q[0], r[0]) <= p[0] <= max(q[0], r[0]) and min(q[1], r[1]) <= p[1] <= max(q[1], r[1])


def _segments_meet(p1, p2, q1, q2) -> bool:
    d1, d2 = _orient2(p1, p2, q1), _orient2(p1, p2, q2)
    d3, d4 = _orient2(q1, q2, p1), _orient2(q1, q2, p2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return ((d1 == 0 and _on_segment2(q1, p1, p2)) or (d2 == 0 and _on_segment2(q2, p1, p2))
            or (d3 == 0 and _on_segment2(p1, q1, q2)) or (d4 == 0 and _on_segment2(p2, q1, q2)))


def _locate2(p, loops) -> str:
    """'in' / 'on' / 'out' of point ``p`` against polygon-with-holes ``loops`` (lists of 2D points)."""
    for loop in _each(loops, "locate_loop"):
        m = len(loop)
        for i in range(m):
            _tick("locate_edge")
            q, r = loop[i], loop[(i + 1) % m]
            if _orient2(q, r, p) == 0 and _on_segment2(p, q, r):
                return "on"
    inside = False
    for loop in _each(loops, "locate_loop"):
        m = len(loop)
        for i in range(m):
            _tick("locate_edge")
            q, r = loop[i], loop[(i + 1) % m]
            q_le, r_le = q[1] <= p[1], r[1] <= p[1]
            if q_le == r_le:
                continue
            det = _orient2(q, r, p)
            if (q_le and det > 0) or ((not q_le) and det < 0):
                inside = not inside
    return "in" if inside else "out"


def _newell(points):
    nx = ny = nz = Z
    m = len(points)
    for i in range(m):
        _tick("newell_vertex")
        p, q = points[i], points[(i + 1) % m]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    return (nx / 2, ny / 2, nz / 2)


def _prepare_faces(snap: _Snapshot) -> None:
    for f in _each(snap.faces, "prepare_face"):
        o, u, v = f.support
        support_normal = _cross(u, v)
        if support_normal == (0, 0, 0):
            raise _refuse(C.INVALID_LOOP, f"face {f.id} has a degenerate support plane")
        f.loops2d = []
        pts_loops = []
        for loop in _each(f.loops, "prepare_loop"):
            pts = []
            for _e, _f, start, _end in _each(loop, "coplanarity_test"):
                p = snap.pos[start]
                if _dot(_sub3(p, o), support_normal) != 0:
                    raise _refuse(C.NOT_EXACTLY_PLANAR, f"face {f.id} has a vertex off its support plane")
                pts.append(p)
            pts_loops.append(pts)
        normals = [_newell(p) for p in _each(pts_loops, "prepare_loop")]
        outer = normals[0]
        if outer == (0, 0, 0):
            raise _refuse(C.INVALID_LOOP, f"face {f.id} has zero area")
        f.normal = outer
        f.w = _dot(outer, pts_loops[0][0])
        f.axis = _argmax_abs(outer)
        for k, pts in _each(enumerate(pts_loops), "prepare_loop"):
            if normals[k] == (0, 0, 0):
                raise _refuse(C.INVALID_LOOP, f"a loop of face {f.id} has zero area")
            f.loops2d.append([_project(p, f.axis) for p in _each(pts, "project_vertex")])
            if k and _dot(normals[k], outer) >= 0:
                raise _refuse(C.INVALID_HOLES, f"a hole of face {f.id} is not wound against its outer loop")
        f.vids, f.eids = set(), set()
        for loop in _each(f.loops, "face_index"):
            for eid, _f, start, _end in _each(loop, "face_index"):
                f.vids.add(start)
                f.eids.add(eid)


def _check_loops(snap: _Snapshot) -> None:
    for f in _each(snap.faces, "check_face"):
        for k, loop2 in _each(enumerate(f.loops2d), "check_loop"):
            m = len(loop2)
            for i in range(m):
                _tick("loop_edge_length")
                if loop2[i] == loop2[(i + 1) % m]:
                    raise _refuse(C.INVALID_LOOP, f"face {f.id} has a zero-length edge")
            for i in range(m):
                for j in range(i + 1, m):
                    _tick("loop_pair_test")
                    p1, p2 = loop2[i], loop2[(i + 1) % m]
                    q1, q2 = loop2[j], loop2[(j + 1) % m]
                    if j == i + 1 or (i == 0 and j == m - 1):
                        # consecutive edges: they must meet only at the shared vertex
                        if j == i + 1:
                            x, s, y = p1, p2, q2
                        else:
                            x, s, y = q1, q2, p2
                        if _orient2(x, s, y) == 0 and ((x[0] - s[0]) * (y[0] - s[0]) + (x[1] - s[1]) * (y[1] - s[1])) > 0:
                            raise _refuse(C.INVALID_LOOP, f"face {f.id} has a zero-angle spike")
                    elif _segments_meet(p1, p2, q1, q2):
                        raise _refuse(C.INVALID_LOOP, f"a loop of face {f.id} is not simple")
        # holes: strictly inside the outer loop, pairwise disjoint
        outer = f.loops2d[0]
        for k in range(1, len(f.loops2d)):
            hole = f.loops2d[k]
            for i in range(len(hole)):
                for j in range(len(outer)):
                    _tick("hole_pair_test")
                    if _segments_meet(hole[i], hole[(i + 1) % len(hole)], outer[j], outer[(j + 1) % len(outer)]):
                        raise _refuse(C.INVALID_HOLES, f"a hole of face {f.id} touches the outer loop")
            if _locate2(hole[0], [outer]) != "in":
                raise _refuse(C.INVALID_HOLES, f"a hole of face {f.id} is not inside the outer loop")
            for k2 in range(1, k):
                other = f.loops2d[k2]
                for i in range(len(hole)):
                    for j in range(len(other)):
                        _tick("hole_pair_test")
                        if _segments_meet(hole[i], hole[(i + 1) % len(hole)], other[j], other[(j + 1) % len(other)]):
                            raise _refuse(C.INVALID_HOLES, f"holes of face {f.id} touch")
                if _locate2(hole[0], [other]) != "out" or _locate2(other[0], [hole]) != "out":
                    raise _refuse(C.INVALID_HOLES, f"holes of face {f.id} are nested")


def _check_combinatorics(snap: _Snapshot) -> None:
    uses: dict[int, list] = {}
    corners: dict[int, list] = {}
    for f in _each(snap.faces, "edge_use"):
        for loop in _each(f.loops, "edge_use"):
            for i, (eid, fwd, start, _end) in _each(enumerate(loop), "edge_use"):
                uses.setdefault(eid, []).append((f.id, fwd))
                corners.setdefault(start, []).append((loop[i - 1][0], eid))
    parent = {f.id: f.id for f in _each(snap.faces, "union_find")}

    def find(x):
        while parent[x] != x:
            _tick("union_find")
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for eid, items in _each(uses.items(), "edge_use_check"):
        if len(items) != 2:
            raise _refuse(C.NOT_CLOSED if len(items) < 2 else C.NON_MANIFOLD, f"edge {eid} is used {len(items)} times")
        (fa, da), (fb, db) = items
        if da == db:
            raise _refuse(C.INVERTED_ORIENTATION, f"edge {eid} is used twice in the same direction")
        if fa == fb:
            raise _refuse(C.NON_MANIFOLD, f"edge {eid} is used twice by one face")
        parent[find(fa)] = find(fb)
    roots = set()
    for f in _each(snap.faces, "union_find"):
        roots.add(find(f.id))
    if len(roots) != 1:
        raise _refuse(C.NON_MANIFOLD, "the faces do not form one connected shell")
    for vid, cs in _each(corners.items(), "vertex_link"):
        adjacency: dict[int, list] = {}
        for a, b in _each(cs, "vertex_link"):
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        for v in _each(adjacency.values(), "vertex_link"):
            if len(v) != 2:
                raise _refuse(C.NON_MANIFOLD, f"the link of vertex {vid} is not a cycle")
        seen, stack = set(), [next(iter(adjacency))]
        while stack:
            _tick("vertex_link")
            e = stack.pop()
            if e not in seen:
                seen.add(e)
                for nxt in _each(adjacency[e], "vertex_link"):
                    stack.append(nxt)
        if len(seen) != len(adjacency):
            raise _refuse(C.NON_MANIFOLD, f"the link of vertex {vid} is not a single cycle")


def _merge_intervals(intervals):
    out = []
    for lo, hi in _each(_tsorted(intervals), "interval_merge"):
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _intersect_intervals(a, b):
    out = []
    for lo1, hi1 in _each(a, "interval_intersect"):
        for lo2, hi2 in _each(b, "interval_intersect"):
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if lo <= hi:
                out.append((lo, hi))
    return _merge_intervals(out)


def _seg_intersection2(p1, p2, q1, q2):
    """Intersection of two closed 2D segments: None, (P, P) for a point or (P, Q) for a collinear overlap."""
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (q2[0] - q1[0], q2[1] - q1[1])
    w = (q1[0] - p1[0], q1[1] - p1[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if den != 0:
        t = (w[0] * d2[1] - w[1] * d2[0]) / den
        u = (w[0] * d1[1] - w[1] * d1[0]) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            pt = (p1[0] + t * d1[0], p1[1] + t * d1[1])
            return pt, pt
        return None
    if w[0] * d1[1] - w[1] * d1[0] != 0:
        return None
    dd = d1[0] * d1[0] + d1[1] * d1[1]
    t1 = (w[0] * d1[0] + w[1] * d1[1]) / dd
    t2 = ((q2[0] - p1[0]) * d1[0] + (q2[1] - p1[1]) * d1[1]) / dd
    lo, hi = max(Z, min(t1, t2)), min(ONE, max(t1, t2))
    if lo > hi:
        return None
    return ((p1[0] + lo * d1[0], p1[1] + lo * d1[1]), (p1[0] + hi * d1[0], p1[1] + hi * d1[1]))


def _param_on(x, a, b):
    d = (b[0] - a[0], b[1] - a[1])
    return ((x[0] - a[0]) * d[0] + (x[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])


def _covered(piece, shared_pts, shared_segs) -> bool:
    x, y = piece
    if x == y:
        if x in shared_pts:
            return True
        for a, b in _each(shared_segs, "cover_test"):
            if _orient2(a, b, x) == 0 and _on_segment2(x, a, b):
                return True
        return False
    intervals = []
    for a, b in _each(shared_segs, "cover_test"):
        if _orient2(a, b, x) == 0 and _orient2(a, b, y) == 0:
            ta, tb = _param_on(a, x, y), _param_on(b, x, y)
            intervals.append((min(ta, tb), max(ta, tb)))
    for lo, hi in _each(_merge_intervals(intervals), "cover_test"):
        if lo <= 0 and hi >= 1:
            return True
    return False


def _edges2(face: _Face, snap: _Snapshot):
    """All edges of a face as (eid, forward, p2d, q2d)."""
    out = []
    for loop in _each(face.loops, "edge_project"):
        for eid, fwd, start, end in _each(loop, "edge_project"):
            out.append((eid, fwd, _project(snap.pos[start], face.axis), _project(snap.pos[end], face.axis)))
    return out


def _coplanar_pair(snap, f, g, shared_edges, shared_vids) -> None:
    shared_pts = set()
    for v in _each(shared_vids, "shared_cell"):
        shared_pts.add(_project(snap.pos[v], f.axis))
    shared_segs = []
    for e in _each(shared_edges, "shared_cell"):
        s, t = snap.edges[e]
        shared_segs.append((_project(snap.pos[s], f.axis), _project(snap.pos[t], f.axis)))
    ef, eg = _edges2(f, snap), _edges2(g, snap)
    sigma_f, sigma_g = _sign(f.normal[f.axis]), _sign(g.normal[f.axis])
    splits_f = {i: [Z, ONE] for i in _each(range(len(ef)), "split_init")}
    splits_g = {j: [Z, ONE] for j in _each(range(len(eg)), "split_init")}
    for i, (_a, _fa, p1, p2) in _each(enumerate(ef), "edge_pair_test"):
        for j, (_b, _fb, q1, q2) in _each(enumerate(eg), "edge_pair_test"):
            piece = _seg_intersection2(p1, p2, q1, q2)
            if piece is None:
                continue
            if not _covered(piece, shared_pts, shared_segs):
                raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} meet outside their shared cells")
            for x in _each((piece[0],) if piece[0] == piece[1] else (piece[0], piece[1]), "split_point"):
                splits_f[i].append(_param_on(x, p1, p2))
                splits_g[j].append(_param_on(x, q1, q2))
            if piece[0] != piece[1]:
                d1 = (p2[0] - p1[0], p2[1] - p1[1])
                d2 = (q2[0] - q1[0], q2[1] - q1[1])
                left1 = (-d1[1] * sigma_f, d1[0] * sigma_f)
                left2 = (-d2[1] * sigma_g, d2[0] * sigma_g)
                if left1[0] * left2[0] + left1[1] * left2[1] > 0:
                    raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} overlap")
    for edges, splits, other in ((ef, splits_f, g.loops2d), (eg, splits_g, f.loops2d)):
        for idx, (_e, _fw, p1, p2) in _each(enumerate(edges), "subpiece_test"):
            ts = _tsorted(set(_each(splits[idx], "split_collect")))
            for t0, t1 in _each(zip(ts, ts[1:]), "subpiece_test"):
                tm = (t0 + t1) / 2
                mid = (p1[0] + tm * (p2[0] - p1[0]), p1[1] + tm * (p2[1] - p1[1]))
                if _locate2(mid, other) == "in":
                    raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} overlap")


def _line_intervals(face: _Face, snap, other_normal, other_w, direction):
    """Closed intervals (in the coordinate s = direction . r) of ``face`` intersected with the other plane."""
    critical = []
    intervals = []
    for loop in _each(face.loops, "line_edge_test"):
        for eid, _fw, start, end in _each(loop, "line_edge_test"):
            q, r = snap.pos[start], snap.pos[end]
            gq, gr = _dot(other_normal, q) - other_w, _dot(other_normal, r) - other_w
            if gq == 0 and gr == 0:
                sq, sr = _dot(direction, q), _dot(direction, r)
                intervals.append((min(sq, sr), max(sq, sr)))
                critical.append((sq, q))
                critical.append((sr, r))
            elif gq == 0:
                critical.append((_dot(direction, q), q))
            elif gr == 0:
                critical.append((_dot(direction, r), r))
            elif (gq > 0) != (gr > 0):
                tau = gq / (gq - gr)
                point = _add3(q, _scale3(_sub3(r, q), tau))
                critical.append((_dot(direction, point), point))
    critical = _tsorted(critical, key=lambda item: item[0])
    for s, _p in _each(critical, "critical_point"):
        intervals.append((s, s))
    for (s0, p0), (s1, p1) in _each(zip(critical, critical[1:]), "line_interval_test"):
        if s0 == s1:
            continue
        mid = _scale3(_add3(p0, p1), HALF)
        if _locate2(_project(mid, face.axis), face.loops2d) != "out":
            intervals.append((s0, s1))
    return _merge_intervals(intervals)


def _noncoplanar_pair(snap, f, g, shared_edges, shared_vids) -> None:
    direction = _cross(f.normal, g.normal)
    i_f = _line_intervals(f, snap, g.normal, g.w, direction)
    i_g = _line_intervals(g, snap, f.normal, f.w, direction)
    touching = _intersect_intervals(i_f, i_g)
    shared = []
    for e in _each(shared_edges, "shared_cell"):
        s, t = snap.edges[e]
        a, b = _dot(direction, snap.pos[s]), _dot(direction, snap.pos[t])
        shared.append((min(a, b), max(a, b)))
    for v in _each(shared_vids, "shared_cell"):
        a = _dot(direction, snap.pos[v])
        shared.append((a, a))
    merged_shared = _merge_intervals(shared)
    if not _same_intervals(touching, merged_shared):
        raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} meet outside their shared cells")


def _same_intervals(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in _each(zip(a, b), "interval_compare"):
        if x != y:
            return False
    return True


def _check_embedding(snap: _Snapshot) -> None:
    scale = 1
    for p in _each(snap.pos.values(), "box_scale"):
        for i in range(3):
            scale = max(scale, p[i].denominator)
    for f in _each(snap.faces, "box_build"):
        lows, highs = [None] * 3, [None] * 3
        for loop in _each(f.loops, "box_build"):
            for _e, _f, start, _end in _each(loop, "box_build"):
                p = snap.pos[start]
                for i in range(3):
                    # every coordinate is a double: its denominator divides the largest one, so this is exact
                    c = _fit(p[i].numerator * (scale // p[i].denominator))
                    lows[i] = c if lows[i] is None or c < lows[i] else lows[i]
                    highs[i] = c if highs[i] is None or c > highs[i] else highs[i]
        f.box = ((lows[0], highs[0]), (lows[1], highs[1]), (lows[2], highs[2]))
    order = _tsorted(range(len(snap.faces)), key=lambda i: snap.faces[i].box[0][0])
    for pos, i in _each(enumerate(order), "box_sweep"):
        f = snap.faces[i]
        j_pos = pos + 1
        while j_pos < len(order) and snap.faces[order[j_pos]].box[0][0] <= f.box[0][1]:
            _tick("box_test")
            g = snap.faces[order[j_pos]]
            j_pos += 1
            if g.box[1][0] > f.box[1][1] or f.box[1][0] > g.box[1][1] or g.box[2][0] > f.box[2][1] or f.box[2][0] > g.box[2][1]:
                continue
            shared_edges = f.eids & g.eids
            shared_vids = f.vids & g.vids
            direction = _cross(f.normal, g.normal)
            if direction == (0, 0, 0):
                if _dot(f.normal, snap.pos[next(iter(g.vids))]) != f.w:
                    if shared_vids:
                        raise _refuse(C.SELF_INTERSECTING, "parallel faces share a vertex")
                    continue
                _coplanar_pair(snap, f, g, shared_edges, shared_vids)
            else:
                _noncoplanar_pair(snap, f, g, shared_edges, shared_vids)


def _admit(snap: _Snapshot) -> Fraction:
    _prepare_faces(snap)
    _check_loops(snap)
    _check_combinatorics(snap)
    _check_embedding(snap)
    volume = Z
    for f in _each(snap.faces, "admit_face"):
        g = Z
        for loop in _each(f.loops, "admit_loop"):
            pts = [snap.pos[start] for _e, _fw, start, _end in _each(loop, "admit_vertex")]
            g += _chain_G(pts, f.axis)[0]
        volume += f.w / f.normal[f.axis] * g / 3
    if volume <= 0:
        raise _refuse(C.INVERTED_ORIENTATION, "the solid has non-positive volume")
    return _fit(volume)


# ------------------------------------------------------------------ Green / Mirtich integrals


def _pmul_list(p, q):
    out = [Z] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] = out[i + j] + x * y
    return out


def _edge_G(a0, b0, a1, b1):
    """Green's-theorem edge contributions to the integrals of 1, a, b, a^2, ab, b^2 (CCW positive)."""
    db = b1 - b0
    if db == 0:
        return (Z, Z, Z, Z, Z, Z)
    da = a1 - a0
    A1, B1 = [a0, da], [b0, db]
    A2 = _pmul_list(A1, A1)
    B2 = _pmul_list(B1, B1)

    def integral(poly):
        return sum((c / (k + 1) for k, c in enumerate(poly)), Z)

    ab = _pmul_list(A1, B1)
    return (db * integral(A1),
            db * integral(A2) / 2,
            db * integral(ab),
            db * integral(_pmul_list(A2, A1)) / 3,
            db * integral(_pmul_list(A2, B1)) / 2,
            db * integral(_pmul_list(A1, B2)))


def _chain_G(points, axis):
    """Sum of ``_edge_G`` over a closed chain of 3D points projected on the cyclic axes of ``axis``."""
    total = [Z] * 6
    m = len(points)
    for i in range(m):
        _tick("green_edge")
        p, q = points[i], points[(i + 1) % m]
        e = _edge_G(p[(axis + 1) % 3], p[(axis + 2) % 3], q[(axis + 1) % 3], q[(axis + 2) % 3])
        for k in range(6):
            total[k] += e[k]
    return total


def _segments_G(segments, axes):
    """Sum over directed segments ((P, Q) 3D points) projected on the coordinate indices ``axes`` = (a, b)."""
    a, b = axes
    total = [Z] * 6
    for p, q in _each(segments, "green_edge"):
        e = _edge_G(p[a], p[b], q[a], q[b])
        for k in range(6):
            total[k] += e[k]
    return total


def _face_contribution(G, normal, w, axis):
    """(V, Jx, Jy, Jz) contributions of one planar piece with chain integrals G (cyclic projection)."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    nc = normal[axis]
    g00, g10, g01, g20, g11, g02 = G
    a0, a1, a2 = w / nc, -normal[a] / nc, -normal[b] / nc
    quad = {axis: a0 * a0 * g00 + 2 * a0 * a1 * g10 + 2 * a0 * a2 * g01 + a1 * a1 * g20 + 2 * a1 * a2 * g11
            + a2 * a2 * g02,
            a: g20, b: g02}
    v = w / nc * g00 / 3
    j = [normal[i] / nc * quad[i] / 2 for i in range(3)]
    return v, j


# ------------------------------------------------------------------ section recomputation


@dataclass
class _Side:
    occupancy: str
    section: str
    quantities: dict | None
    topology: dict | None
    chain: list = field(default_factory=list)


def _clip_loop(points, h):
    """Sutherland-Hodgman against h < 0 (the ``below`` convention) -> list of 3D points; one checkpoint per vertex."""
    out = []
    m = len(points)
    for i in range(m):
        _tick("clip_vertex")
        p, q = points[i], points[(i + 1) % m]
        hp, hq = h[i], h[(i + 1) % m]
        p_in, q_in = hp < 0, hq < 0
        if p_in:
            out.append(p)
        if p_in != q_in:
            t = hp / (hp - hq)
            out.append(_add3(p, _scale3(_sub3(q, p), t)))
    return out


class _Plane:
    """The request, exactly (trusted guarded rationals from the ingested query)."""

    def __init__(self, req: _Req):
        self.n, self.d, self.o = req.normal, req.offset, req.origin
        self.e1, self.e2 = req.e1, req.e2
        self.nn = _dot(self.n, self.n)
        self.k = _argmax_abs(self.n)
        self.ab = tuple(i for i in range(3) if i != self.k)
        self.orient = (-1 if self.k % 2 else 1) * _sign(self.n[self.k])


def _affine_vertex(c0, c1):
    return ((c0[0], c1[0]), (c0[1], c1[1]), (c0[2], c1[2]))


def _section_side(snap: _Snapshot, plane: _Plane, side: str, heights: dict) -> _Side:
    inside = (lambda h: h < 0) if side == "below" else (lambda h: h <= 0)
    tau = 1 if side == "below" else -1
    crossing = {}
    for eid, (s, e) in _each(snap.edges.items(), "edge_crossing"):
        hs, he = heights[s], heights[e]
        if inside(hs) != inside(he):
            ps, pe = snap.pos[s], snap.pos[e]
            nn = he - hs
            t0 = -hs / nn
            delta = _sub3(pe, ps)
            c0 = _add3(ps, _scale3(delta, t0))
            c1 = _scale3(delta, -tau / nn)
            crossing[eid] = _affine_vertex(c0, c1)
    any_in = any_out = False
    for h in _each(heights.values(), "side_class"):
        if inside(h):
            any_in = True
        else:
            any_out = True
    occupancy = "PARTIAL" if any_in and any_out else ("FULL" if any_in else "EMPTY")
    if not crossing:
        return _Side(occupancy, "ABSENT", None, None)
    segments = []
    for f in _each(snap.faces, "section_face"):
        direction = _cross(plane.n, f.normal)
        cuts = []
        for loop in _each(f.loops, "face_cut_scan"):
            for eid, _fw, _s, _e in _each(loop, "face_cut_scan"):
                if eid in crossing:
                    cuts.append(eid)
        if not cuts:
            continue
        if len(cuts) % 2 or direction == (0, 0, 0):
            raise _refuse(C.SYMBOLIC_DEGENERACY, f"face {f.id} has an odd or impossible number of cut points")

        def key(eid, direction=direction):
            point = crossing[eid]
            return (direction[0] * point[0][0] + direction[1] * point[1][0] + direction[2] * point[2][0],
                    direction[0] * point[0][1] + direction[1] * point[1][1] + direction[2] * point[2][1])

        cuts = _tsorted(cuts, key=key)
        keys = [key(eid) for eid in _each(cuts, "cut_key")]
        for k0, k1 in _each(zip(keys, keys[1:]), "cut_key"):
            if k0 == k1:
                raise _refuse(C.SYMBOLIC_DEGENERACY, "two distinct cut points coincide on a cut line")
        axis = f.axis
        a_i, b_i = (axis + 1) % 3, (axis + 2) % 3
        loops_const = [[((p[0],), (p[1],)) for p in _each(loop, "loop_constant")]
                       for loop in _each(f.loops2d, "loop_constant")]
        for e0, e1 in _each(zip(cuts, cuts[1:]), "interval_classification"):
            p0, p1 = crossing[e0], crossing[e1]
            mid = tuple(((p0[i][0] + p1[i][0]) / 2, (p0[i][1] + p1[i][1]) / 2) for i in range(3))
            if _inside_sym(mid[a_i], mid[b_i], loops_const):
                segments.append((e0, e1))
    # loops
    nxt, indeg = {}, {}
    for a, b in _each(segments, "loop_link"):
        if a in nxt or b in indeg:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "the cut-point graph is not 2-regular")
        nxt[a] = b
        indeg[b] = a
    if set(_each(nxt, "loop_link")) != set(_each(indeg, "loop_link")) or set(_each(nxt, "loop_link")) != set(
            _each(crossing, "loop_link")):
        raise _refuse(C.SYMBOLIC_DEGENERACY, "the cut-point graph has open chains")
    loops, seen = [], set()
    for start in _each(_tsorted(nxt), "loop_walk"):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            _tick("loop_walk")
            seen.add(cur)
            loop.append(cur)
            cur = nxt[cur]
        loops.append(loop)
    # waterplane quantities from the limit chain
    a_i, b_i = plane.ab
    chain = []
    for x, y in _each(segments, "chain_segment"):
        chain.append((tuple(crossing[x][i][0] for i in range(3)), tuple(crossing[y][i][0] for i in range(3))))
    g00, g10, g01, g20, g11, g02 = _segments_G(chain, (a_i, b_i))
    n, k = plane.n, plane.k
    u_c = _basis_coeffs(plane, plane.e1)
    v_c = _basis_coeffs(plane, plane.e2)

    def combine(p, q):
        (p0, p1, p2), (q0, q1, q2) = p, q
        return (p0 * q0 * g00 + (p0 * q1 + p1 * q0) * g10 + (p0 * q2 + p2 * q0) * g01 + p1 * q1 * g20
                + (p1 * q2 + p2 * q1) * g11 + p2 * q2 * g02)

    one = (ONE, Z, Z)
    raw = {"A": g00, "Mu": combine(u_c, one), "Mv": combine(v_c, one), "Iuu": combine(u_c, u_c),
           "Ivv": combine(v_c, v_c), "Iuv": combine(u_c, v_c)}
    factor = plane.nn * plane.orient / abs(n[k])
    quantities = {name: (raw[name] * factor, plane.nn) for name in _each(_QUANTITIES, "quantity")}
    # topology
    roles, polys = [], []
    for loop in _each(loops, "loop_area"):
        area = (Z,)
        m = len(loop)
        for i in range(m):
            _tick("loop_area")
            p, q = crossing[loop[i]], crossing[loop[(i + 1) % m]]
            area = _padd(area, _psub(_pmul(p[a_i], q[b_i]), _pmul(q[a_i], p[b_i])))
        s = _psign(area) * plane.orient
        if s == 0:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "a loop has zero symbolic area")
        roles.append("outer" if s > 0 else "hole")
        polys.append([(crossing[e][a_i], crossing[e][b_i]) for e in _each(loop, "loop_polygon")])
    outers = [i for i, r in _each(enumerate(roles), "nesting") if r == "outer"]
    holes = [i for i, r in _each(enumerate(roles), "nesting") if r == "hole"]
    holders: dict[int, list] = {i: [] for i in _each(outers, "nesting")}
    for h in _each(holes, "nesting"):
        probe = polys[h][0]
        cands = []
        for o in _each(outers, "nesting"):
            if _inside_sym(probe[0], probe[1], [polys[o]]):
                cands.append(o)
        if not cands:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "a hole loop lies in no outer loop")
        best = []
        for o in _each(cands, "nesting"):
            innermost = True
            for o2 in _each(cands, "nesting"):
                if o2 != o and not _inside_sym(polys[o][0][0], polys[o][0][1], [polys[o2]]):
                    innermost = False
                    break
            if innermost:
                best.append(o)
        if len(best) != 1:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "the nesting of a hole loop is ambiguous")
        holders[best[0]].append(h)

    def ident(i):
        return tuple(_tsorted(loops[i]))

    topology = {"loops": _tsorted([(ident(i), roles[i]) for i in _each(range(len(loops)), "topology_element")]),
                "components": _tsorted([(ident(o), tuple(_tsorted([ident(h) for h in _each(holders[o], "topology_element")])))
                                        for o in _each(outers, "topology_element")])}
    return _Side(occupancy, "PRESENT", quantities, topology, chain)


def _basis_coeffs(plane: _Plane, e):
    """Coefficients (b0, b1, b2) of u = e.(r - o) as an affine function of the projection coordinates."""
    n, k = plane.n, plane.k
    a, b = plane.ab
    b0 = e[k] * plane.d / n[k] - _dot(e, plane.o)
    b1 = e[a] - e[k] * n[a] / n[k]
    b2 = e[b] - e[k] * n[b] / n[k]
    return (b0, b1, b2)


def _inside_sym(pa, pb, loops) -> bool:
    inside = False
    for loop in _each(loops, "point_in_polygon_loop"):
        m = len(loop)
        for i in range(m):
            _tick("point_in_polygon_edge")
            qa, qb = loop[i]
            ra, rb = loop[(i + 1) % m]
            q_le = _psign(_psub(pb, qb)) >= 0
            r_le = _psign(_psub(pb, rb)) >= 0
            if q_le == r_le:
                continue
            det = _psub(_pmul(_psub(ra, qa), _psub(pb, qb)), _pmul(_psub(rb, qb), _psub(pa, qa)))
            s = _psign(det)
            if s == 0:
                raise _refuse(C.SYMBOLIC_DEGENERACY, "a test point lies on a loop edge")
            if (q_le and s > 0) or ((not q_le) and s < 0):
                inside = not inside
    return inside


def _recompute(snap: _Snapshot, plane: _Plane, total_volume) -> dict:
    heights = {}
    for vid, p in _each(snap.pos.items(), "vertex_height"):
        heights[vid] = _dot(plane.n, p) - plane.d
    on_plane, nonneg, nonpos = [], True, True
    for v, h in _each(heights.items(), "vertex_class"):
        if h == 0:
            on_plane.append(v)
        nonneg = nonneg and h >= 0
        nonpos = nonpos and h <= 0
    on_plane = _tsorted(on_plane)
    generic = not on_plane
    contacts_edges = []
    for e, (s, t) in _each(snap.edges.items(), "contact_edge"):
        if heights[s] == 0 and heights[t] == 0:
            contacts_edges.append(e)
    contacts_edges = _tsorted(contacts_edges)
    contacts_faces = []
    for f in _each(snap.faces, "contact_face"):
        flat = True
        for v in _each(f.vids, "contact_face"):
            if heights[v] != 0:
                flat = False
                break
        if flat:
            contacts_faces.append(f.id)
    contacts_faces = _tsorted(contacts_faces)
    occupancy = "EMPTY" if nonneg else ("FULL" if nonpos else "PARTIAL")
    # V and J: clipped faces (below convention) plus the cap
    volume = Z
    moments = [Z, Z, Z]
    for f in _each(snap.faces, "clip_face"):
        g = [Z] * 6
        for loop in _each(f.loops, "clip_loop"):
            pts = [snap.pos[start] for _e, _fw, start, _end in _each(loop, "clip_gather")]
            h = [heights[start] for _e, _fw, start, _end in _each(loop, "clip_gather")]
            clipped = _clip_loop(pts, h)
            if len(clipped) >= 3:
                part = _chain_G(clipped, f.axis)
                g = [g[k] + part[k] for k in range(6)]
        v, j = _face_contribution(g, f.normal, f.w, f.axis)
        volume += v
        for i in range(3):
            moments[i] += j[i]
    below = _section_side(snap, plane, "below", heights)
    sides = {}
    if generic:
        sides["plane"] = below
    else:
        sides["below"] = below
        sides["above"] = _section_side(snap, plane, "above", heights)
    if below.section == "PRESENT":
        k = plane.k
        g = _segments_G(below.chain, ((k + 1) % 3, (k + 2) % 3))
        v, j = _face_contribution(g, plane.n, plane.d, k)
        volume += v
        for i in range(3):
            moments[i] += j[i]
    if occupancy == "EMPTY" and (volume != 0 or moments[0] != 0 or moments[1] != 0 or moments[2] != 0):
        raise _refuse(C.SYMBOLIC_DEGENERACY, "an empty half-space has non-zero volume")
    if occupancy == "FULL" and volume != total_volume:
        raise _refuse(C.SYMBOLIC_DEGENERACY, "a full half-space does not give the solid volume")
    return {"occupancy": occupancy, "generic": generic,
            "contacts": {"vertices": on_plane, "edges": contacts_edges, "faces": contacts_faces},
            "V": volume, "J": moments, "sides": sides}


# ------------------------------------------------------------------ comparisons of forms


def _cmp_exact(r, num, ns) -> int:
    """sign(r - x) with x = num (ns None) or num / sqrt(ns); all arguments are guarded rationals."""
    if ns is None:
        return _sign(r - num)
    num_root, den_root = math.isqrt(ns.numerator), math.isqrt(ns.denominator)
    if num_root * num_root == ns.numerator and den_root * den_root == ns.denominator:
        return _sign(r - num / _Q(num_root, den_root))
    # compare r * sqrt(ns) with num
    if r >= 0 and num < 0:
        return 1
    if r < 0 and num >= 0:
        return -1
    lhs, rhs = r * r * ns, num * num
    if r >= 0:
        return _sign(lhs - rhs)
    return -_sign(lhs - rhs)


_DECIMAL_CHARS = re.compile(r"[0-9.]*")


def _decimal_parts(text):
    """Split a decimal bound of the grammar `-?digits(.digits)?` into (negative, whole, frac), scanning it in windows
    of at most 4096 characters (one checkpoint each); ``None`` when the grammar does not hold."""
    negative = text.startswith("-")
    start = 1 if negative else 0
    dot = -1
    for k in range(start, max(len(text), start + 1), _WINDOW):
        _tick("decimal_window")
        window = text[k:k + _WINDOW]
        if not _DECIMAL_CHARS.fullmatch(window):
            return None
        at = window.find(".")
        if at >= 0:
            if dot >= 0 or window.find(".", at + 1) >= 0:
                return None
            dot = k + at
    if len(text) == start:
        return None
    if dot < 0:
        return negative, text[start:], ""
    if dot == start or dot == len(text) - 1:
        return None
    return negative, text[start:dot], text[dot + 1:]


_LOG2_10_LOW = 3321928094          # 3.321928094 < log2(10) = 3.32192809488...; exact integer arithmetic, no floats
_LOG2_10_SCALE = 10 ** 9


def _pow10_bits_lower(k: int) -> int:
    """A lower bound of ``(10 ** k).bit_length()`` (exact for every k up to 10**9 or so; never above the true value)."""
    return k * _LOG2_10_LOW // _LOG2_10_SCALE + 1


def _significant_digits(whole: str, frac: str) -> int:
    """The number of digits of ``whole + frac`` after its leading zeros, found window by window (one checkpoint each)."""
    skipped = 0
    for part in (whole, frac):
        for k in range(0, len(part), _WINDOW):
            _tick("decimal_window")
            window = part[k:k + _WINDOW]
            stripped = len(window.lstrip("0"))
            skipped += len(window) - stripped
            if stripped:
                return len(whole) + len(frac) - skipped
    return 0


def _require_decimal_fits(whole: str, frac: str) -> None:
    """Refuse ``BIT_LENGTH_EXCEEDED`` before any integer is built when the denominator ``10 ** len(frac)`` or the
    numerator (its digits without leading zeros) is certain to exceed the active bit cap."""
    budget = getattr(_ACTIVE, "budget", None)
    cap = None if budget is None else budget.max_bits
    if cap is None:
        return
    digits = _significant_digits(whole, frac)     # a number of k digits (leading one nonzero) is >= 10 ** (k - 1)
    for name, bits in (("denominator", _pow10_bits_lower(len(frac))),
                       ("numerator", _pow10_bits_lower(digits - 1) if digits else 0)):
        if bits > cap:
            raise _refuse(C.BIT_LENGTH_EXCEEDED,
                          f"a decimal bound's {name} needs at least {bits} bits, the cap is {cap}")


def _decimal_fraction(text):
    """A decimal bound (a trusted ``str``); the grammar is `-?digits(.digits)?`. The digit limit of the policy applies
    to integer tokens of the proof only, not to decimal strings (their length is bounded by the proof's byte limit)."""
    _charge_text(2 * len(text))          # the split into whole and fraction digits and their concatenation (6b)
    parts = _decimal_parts(text)
    if parts is None:
        raise _fail("enclosure", "a decimal bound is not a plain decimal string")
    negative, whole, frac = parts
    _require_decimal_fits(whole, frac)
    numerator = _int_from_digits(whole + frac)
    denominator = _fit(10 ** len(frac))
    value = _fit(_Q(numerator, denominator))
    return -value if negative else value


def _check_quantity(item: _QR, name, num, ns, req: _Req, exact_required: bool) -> None:
    _tick("compare_element")
    if item.name != name:
        raise _fail("request", f"quantity {name} is missing or misnamed")
    if exact_required:
        if item.numerator is None:
            raise _fail("request", f"{name}: exact form missing")
        if item.numerator != num or item.norm_squared != ns:
            raise _fail("value", f"{name}: exact value differs from the recomputed one")
    else:
        if item.numerator is not None or item.norm_squared is not None:
            raise _fail("request", f"{name}: unrequested exact form present")
    if (item.binary64 is not None) != req.want_binary64:
        raise _fail("request", f"{name}: binary64 form against the request")
    if (item.decimal is not None) != req.want_decimal:
        raise _fail("request", f"{name}: decimal form against the request")
    acc = req.accuracy_of(name)
    if item.accuracy != acc:
        raise _fail("request", f"{name}: bound accuracy differs from the request")
    met_ok = True
    if item.binary64 is not None:
        lo, hi = item.binary64
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise _fail("enclosure", f"{name}: binary64 bounds are not finite doubles")
        flo, fhi = _q(lo), _q(hi)
        if lo == hi:
            if _cmp_exact(flo, num, ns) != 0:
                raise _fail("enclosure", f"{name}: a point enclosure differs from the exact value")
        else:
            if math.nextafter(lo, math.inf) != hi:
                raise _fail("enclosure", f"{name}: binary64 endpoints are not adjacent doubles")
            if not (_cmp_exact(flo, num, ns) < 0 < _cmp_exact(fhi, num, ns)):
                raise _fail("enclosure", f"{name}: the binary64 enclosure does not contain the exact value")
            if acc is not None and fhi - flo > acc:
                met_ok = False
    if item.decimal is not None:
        lo_s, hi_s, width = item.decimal
        lo, hi = _decimal_fraction(lo_s), _decimal_fraction(hi_s)
        requested = acc if acc is not None else _Q(1, 10 ** 12)
        if width != requested:
            raise _fail("enclosure", f"{name}: decimal width differs from the requested width")
        if hi - lo > requested or lo > hi:
            raise _fail("enclosure", f"{name}: the decimal enclosure is wider than requested")
        if not (_cmp_exact(lo, num, ns) <= 0 <= _cmp_exact(hi, num, ns)):
            raise _fail("enclosure", f"{name}: the decimal enclosure does not contain the exact value")
    flag = (True if met_ok else False) if acc is not None else None
    if item.accuracy_met is not flag or flag is False:
        raise _fail("accuracy flag", f"{name}: accuracy flag differs from the recomputed one")


# ------------------------------------------------------------------ the claim (built from the trusted copy)


def _claim_quantity(item: _QR) -> dict:
    _tick("claim_element")
    out: dict = {"accuracy": None if item.accuracy is None else _pair(item.accuracy),
                 "accuracy_met": item.accuracy_met}
    if item.numerator is not None:
        out["exact"] = {"numerator": _pair(item.numerator),
                        "norm_squared": None if item.norm_squared is None else _pair(item.norm_squared)}
    if item.binary64 is not None:
        out["binary64"] = {"lo": float.hex(item.binary64[0]), "hi": float.hex(item.binary64[1])}
    if item.decimal is not None:
        out["decimal"] = {"lo": item.decimal[0], "hi": item.decimal[1], "width": _pair(item.decimal[2])}
    return out


def _claim_section(section: _Sec) -> dict:
    topology = None
    if section.topology is not None:
        loops, components = [], []
        for identity, role in _each(section.topology[0], "claim_element"):
            loops.append({"identity": [x for x in _each(identity, "claim_element")], "role": role})
        for outer, holes in _each(section.topology[1], "claim_element"):
            components.append({"outer": [x for x in _each(outer, "claim_element")],
                               "holes": [[x for x in _each(h, "claim_element")] for h in _each(holes, "claim_element")]})
        topology = {"loops": loops, "components": components}
    quantities = None
    if section.quantities is not None:
        quantities = {}
        for q in _each(section.quantities, "claim_element"):
            quantities[q.name] = _claim_quantity(q)
    return {"occupancy": section.occupancy, "section": section.section, "quantities": quantities,
            "topology": topology}


def _claim_of(res: _Res) -> dict:
    return {"occupancy": res.occupancy, "generic": res.generic,
            "contacts": {"vertices": [x for x in _each(res.contacts[0], "claim_element")],
                         "edges": [x for x in _each(res.contacts[1], "claim_element")],
                         "faces": [x for x in _each(res.contacts[2], "claim_element")]},
            "volume": _claim_quantity(res.volume),
            "moments": {m.name: _claim_quantity(m) for m in _each(res.moments, "claim_element")},
            "sections": {s.side: _claim_section(s) for s in _each(res.sections, "claim_element")},
            "work": {"operations": res.ops, "units": {k: v for k, v in _each(res.units.items(), "claim_element")}}}


# ------------------------------------------------------------------ the proof document


@functools.lru_cache(maxsize=16)
def _number_pattern(limit: int):
    """An integer token pattern that consumes at most ``limit + 1`` digits (the scan never runs to the token's end)."""
    return re.compile("-?(?:0|[1-9][0-9]{0,%d})" % limit)


_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')
_HEX4 = re.compile(r"[0-9a-fA-F]{4}")
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _parse_proof(proof: bytes, policy: dict) -> dict:
    """Bounded, metered parser of the canonical JSON subset (no whitespace, no floats, integers only).

    Every value costs one checkpoint; integer tokens are checked against the digit limit before conversion and
    converted in chunks of at most 300 digits; strings are scanned in windows of at most 4096 characters."""
    if len(proof) > policy["max_proof_bytes"]:
        raise _refuse(C.MALFORMED_PROOF, "the proof exceeds the byte limit")
    limit = policy["max_integer_digits"]
    try:
        text = proof.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _refuse(C.MALFORMED_PROOF, "the proof is not UTF-8") from error
    n = len(text)
    # a token can never be longer than the text, so the scan bound is the smaller of the two; the regex engine's
    # repetition limit (2**32 - 1) caps it, which can only refuse (never accept) a token over 2**31 - 1 digits
    scan_limit = min(limit, n, 2**31 - 1)

    def bad(message):
        return _refuse(C.MALFORMED_PROOF, message)

    def string(i):
        """Scan a string in windows of at most 4096 characters, one checkpoint per window or escape."""
        if i >= n or text[i] != '"':
            raise bad("a string was expected")
        pieces, k = [], i + 1
        while True:
            _tick("string_piece")
            window_end = min(k + 4096, n)
            found = _STRING_SPECIAL.search(text, k, window_end)
            if found is None:
                if window_end >= n:
                    raise bad("an unterminated string")
                pieces.append(text[k:window_end])
                k = window_end
                continue
            pieces.append(text[k:found.start()])
            ch = found.group()
            if ch == '"':
                return _joined(pieces), found.end()
            if ch != "\\":
                raise bad("a control character inside a string")
            esc = text[found.end():found.end() + 1]
            if esc in _ESCAPES:
                pieces.append(_ESCAPES[esc])
                k = found.end() + 1
            elif esc == "u":
                hexa = text[found.end() + 1:found.end() + 5]
                if not _HEX4.fullmatch(hexa):
                    raise bad("an invalid unicode escape")
                code = int(hexa, 16)
                k = found.end() + 5
                if 0xD800 <= code < 0xDC00 and text.startswith("\\u", k):
                    low = text[k + 2:k + 6]
                    if _HEX4.fullmatch(low) and 0xDC00 <= int(low, 16) < 0xE000:
                        code = 0x10000 + ((code - 0xD800) << 10) + (int(low, 16) - 0xDC00)
                        k += 6
                pieces.append(chr(code))
            else:
                raise bad("an invalid escape")

    def value(i, depth):
        _tick("proof_element")
        if depth > 32:
            raise bad("the proof is nested too deeply")
        if i >= n:
            raise bad("unexpected end of the proof")
        c = text[i]
        if c == "{":
            out = {}
            i += 1
            if text.startswith("}", i):
                return out, i + 1
            while True:
                _tick("proof_element")
                key, i = string(i)
                if key in out:
                    raise bad("duplicate key")
                if not text.startswith(":", i):
                    raise bad("a colon was expected")
                out[key], i = value(i + 1, depth + 1)
                if text.startswith(",", i):
                    i += 1
                    continue
                if text.startswith("}", i):
                    return out, i + 1
                raise bad("a comma or closing brace was expected")
        if c == "[":
            out = []
            i += 1
            if text.startswith("]", i):
                return out, i + 1
            while True:
                _tick("proof_element")
                item, i = value(i, depth + 1)
                out.append(item)
                if text.startswith(",", i):
                    i += 1
                    continue
                if text.startswith("]", i):
                    return out, i + 1
                raise bad("a comma or closing bracket was expected")
        if c == '"':
            return string(i)
        if text.startswith("true", i):
            return True, i + 4
        if text.startswith("false", i):
            return False, i + 5
        if text.startswith("null", i):
            return None, i + 4
        match = _number_pattern(scan_limit).match(text, i)
        if match is None:
            raise bad("an unexpected character")
        token = match.group(0)
        end = match.end()
        digits = token.lstrip("-")
        if len(digits) > limit or (end < n and text[end] in "0123456789"):
            raise bad("an integer token exceeds the digit limit")
        if end < n and text[end] in ".eE+-":
            raise bad("non-integer numbers are not allowed")
        number = _int_from_digits(digits)
        return (-number if token.startswith("-") else number), end

    try:
        document, end = value(0, 0)
    except RecursionError as error:
        raise bad("the proof is nested too deeply") from error
    if end != n:
        raise bad("trailing characters after the proof")
    return _check_shape(document)


def _is_pair(x) -> bool:
    return type(x) is list and len(x) == 2 and type(x[0]) is str and type(x[1]) is str


def _check_shape(document) -> dict:
    """Every top-level section has its expected container type and the artifact fields their expected types."""
    sections = ("artifact", "model", "convention", "preparation_policy", "query", "result")
    if type(document) is not dict or {"artifact", "model", "convention", "preparation_policy", "query",
                                      "result"} != set(_each(document, "shape_check")):
        raise _refuse(C.MALFORMED_PROOF, "the proof has the wrong top-level structure")
    for key in ("artifact", "model", "convention", "preparation_policy", "query", "result"):
        if type(document[key]) is not dict:
            raise _refuse(C.MALFORMED_PROOF, f"proof section {key} is not an object")
    model = document["model"]
    if set(_each(model, "shape_check")) != {"model_id", "revision", "document_checksum", "face_ids", "snapshot_digest"}:
        raise _refuse(C.MALFORMED_PROOF, "the proof's model section has the wrong fields")
    if type(model["face_ids"]) is not list:
        raise _refuse(C.MALFORMED_PROOF, "the proof's face ids are not a list")
    artifact = document["artifact"]
    if set(_each(artifact, "shape_check")) != {"distribution", "distribution_version", "package_version", "files",
                                               "dependencies", "python"}:
        raise _refuse(C.MALFORMED_PROOF, "the proof's artifact section has the wrong fields")
    if type(artifact["files"]) is not list:
        raise _refuse(C.MALFORMED_PROOF, "the proof's file list is malformed")
    for x in _each(artifact["files"], "shape_check"):
        if not _is_pair(x):
            raise _refuse(C.MALFORMED_PROOF, "the proof's file list is malformed")
    deps = artifact["dependencies"]
    if type(deps) is not list:
        raise _refuse(C.MALFORMED_PROOF, "the proof's dependency list is malformed")
    for x in _each(deps, "shape_check"):
        if not (type(x) is list and len(x) == 2 and type(x[0]) is str and (x[1] is None or type(x[1]) is str)):
            raise _refuse(C.MALFORMED_PROOF, "the proof's dependency list is malformed")
    if set(_each(document["preparation_policy"], "shape_check")) != {"convention_version", "max_integer_digits",
                                                                      "max_proof_bytes"}:
        raise _refuse(C.MALFORMED_PROOF, "the proof's policy section has the wrong fields")
    return document


_CONVENTION = {
    "body_frame": "model coordinates are the body frame; no rotation, no normalisation",
    "half_space": "n.r <= d with exact rational n, d (unnormalised)",
    "area_element": "true area element dA of the plane; waterplane quantities are numerator/|n|",
    "moment_order": ["A", "Mu", "Mv", "Iuu", "Ivv", "Iuv"],
    "limits": "below: n.r = d - eps, above: n.r = d + eps, eps -> 0+; V and J use the below convention",
    "loop_identity": "sorted ids of the crossed shell edges",
}


def _consistent_read(model) -> None:
    if getattr(model, "_transaction_journal", None) is not None:
        raise _refuse(C.CONCURRENT_MUTATION, "a model transaction is open")


def _first_difference(a, b, path=()):
    _tick("diff_node")
    ta, tb = type(a), type(b)
    if ta is dict and tb is dict:
        for key in _tsorted(set(_each(a, "diff_node")) | set(_each(b, "diff_node"))):
            if key not in a or key not in b:
                return path + (key,), "missing"
            found = _first_difference(a[key], b[key], path + (key,))
            if found:
                return found
        return None
    if ta in (list, tuple) and tb in (list, tuple):
        if len(a) != len(b):
            return path, "length"
        for i in _each(range(len(a)), "diff_node"):
            found = _first_difference(a[i], b[i], path + (i,))
            if found:
                return found
        return None
    if ta is str and tb is str:
        _charge_text(min(len(a), len(b)))
    if ta is not tb or a != b:
        return path, "value"
    return None


def _classify(path, kind) -> str:
    """Sub-reason of a difference between the proof's result claim and the claim of the result object."""
    if not path:
        return "value"
    head = path[0]
    if head in ("occupancy", "generic"):
        return "state"
    if head in ("contacts", "work"):
        return "value"
    if head == "sections":
        if len(path) <= 2:
            return "state"
        rest = path[2:]
        if rest[0] in ("occupancy", "section"):
            return "state"
        if rest[0] == "topology":
            return "topology"
        if rest[0] == "quantities":
            if len(rest) <= 2:
                return "state" if kind == "value" else "request"
            return _classify_quantity(rest[2:], kind)
        return "state"
    if head in ("volume", "moments"):
        rest = path[1:] if head == "volume" else path[2:]
        if head == "moments" and len(path) < 2:
            return "request"
        return _classify_quantity(rest, kind)
    return "value"


def _classify_quantity(rest, kind) -> str:
    if not rest:
        return "request"
    key = rest[0]
    if key == "exact":
        return "request" if kind == "missing" and len(rest) == 1 else "value"
    if key in ("binary64", "decimal"):
        return "request" if kind == "missing" and len(rest) == 1 else "enclosure"
    if key == "accuracy_met":
        return "accuracy flag"
    if key == "accuracy":
        return "request"
    return "request" if kind == "missing" else "value"


# ------------------------------------------------------------------ environment (step 2)


def _declared_dependencies(package: str) -> list:
    """`project.dependencies` of the pyproject.toml next to a source checkout (``[]`` when there is none)."""
    import tomllib
    path = os.path.join(os.path.dirname(os.path.dirname(package)), "pyproject.toml")
    try:
        with open(path, "rb") as handle:
            declared = tomllib.load(handle).get("project", {}).get("dependencies", [])
    except (OSError, ValueError):
        return []
    if type(declared) is not list:
        return []
    return [x for x in _each(declared, "dependency") if type(x) is str]


def _environment() -> dict:
    from .. import __version__ as package_version      # relative: the package must not import itself by name
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the anygeometry package directory
    rows = []
    for root, dirs, names in os.walk(package):
        dirs[:] = [d for d in _each(dirs, "package_entry") if d != "__pycache__"]
        for name in _each(names, "package_entry"):
            rows.append(os.path.relpath(os.path.join(root, name), package).replace(os.sep, "/"))
    rows = _tsorted(rows)
    files = []
    for rel in _each(rows, "package_file"):
        h = hashlib.sha256()
        with open(os.path.join(package, *rel.split("/")), "rb") as handle:
            while True:
                _tick("package_hash_chunk")
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        files.append([rel, h.hexdigest()])
    try:
        dist = _metadata.distribution("anygeometry")
        dist_version = dist.version
        requires = dist.requires or []
    except _metadata.PackageNotFoundError:
        # not installed (a source checkout): the runtime dependencies are the project's declared dependencies
        dist_version, requires = None, _declared_dependencies(package)
    deps = []
    for requirement in _each(requires, "dependency"):
        if "extra ==" in requirement.replace("'", '"').replace("extra==", "extra =="):
            continue
        name = re.match(r"[A-Za-z0-9_.\-]+", requirement.strip())
        if name is None:
            continue
        try:
            version = _metadata.version(name.group(0))
        except _metadata.PackageNotFoundError:
            version = None
        deps.append([name.group(0), version])
    return {"distribution": "anygeometry", "distribution_version": dist_version,
            "package_version": package_version, "files": files,
            "dependencies": _tsorted(deps, key=lambda x: x[0]),
            "python": f"{platform.python_implementation()} {platform.python_version()}"}


# every record collection the library encoder walks (serialization._serialized_model_state), plus the feature history
_MODEL_COLLECTIONS = ("vertices", "edges", "faces", "parts", "sheets", "face_uses", "coedges", "members",
                      "member_edge_uses", "attachments", "junctions", "groups", "tags", "features")


def _document_checksum(model) -> str:
    """Hash of the canonical model document. The library serialisation and the JSON encoding of the whole document are
    one bulk operation (design 6b): it is charged one unit per model entity before it starts and is followed
    immediately by a checkpoint."""
    from ..serialization import to_dict
    for name in _MODEL_COLLECTIONS:
        try:
            count = len(getattr(model, name))
        except (AttributeError, TypeError):
            count = 0
        for _ in _each(range(count), "document_entity"):
            pass
    document = to_dict(model)
    payload = {k: v for k, v in _each(document.items(), "document_entry") if k != "checksum"}
    # one library call over the whole document: its hashing below is metered per 64 KiB
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")
    _tick("document_encoded")
    own = _sha256_chunks(data, "document_hash_chunk")
    if own != document["checksum"]["value"]:
        raise _fail("model", "the document checksum could not be reproduced independently")
    return own


# ------------------------------------------------------------------ result object


@dataclass(frozen=True)
class VerifiedHydrostatics:
    """A result whose every claim was recomputed and found equal (design section 6).

    Nothing here is forwarded to, or references, the objects that were supplied: ``result`` and ``query`` are rebuilt
    from the checked copies, and ``exact`` holds the recomputed exact values (plain Fractions): ``V``, ``J``,
    ``occupancy``, ``generic``, ``contacts`` and per side the ``(numerator, norm_squared)`` pairs of the waterplane
    quantities and the topology. ``volume``, ``moments``, ``sections``, ``occupancy``, ``generic`` and ``contacts`` read
    the rebuilt ``result``."""

    result: Any
    query: Any
    face_ids: tuple
    model_id: str
    revision: int
    document_checksum: str
    snapshot_digest: str
    artifact_files_sha256: str
    proof_sha256: str
    solid_volume: Fraction
    exact: dict = field(compare=False)
    work: dict = field(compare=False)

    @property
    def volume(self):
        return self.result.volume

    @property
    def moments(self):
        return self.result.moments

    @property
    def sections(self):
        return self.result.sections

    @property
    def occupancy(self):
        return self.result.occupancy

    @property
    def generic(self):
        return self.result.generic

    @property
    def contacts(self):
        return self.result.contacts


def _plain(q) -> Fraction:
    return Fraction(q.numerator, q.denominator)


def _plain3(v) -> tuple:
    return (_plain(v[0]), _plain(v[1]), _plain(v[2]))


def _plain_or_none(q):
    return None if q is None else _plain(q)


def _rebuild_quantity(item: _QR) -> QuantityResult:
    decimal = None if item.decimal is None else (item.decimal[0], item.decimal[1], _plain(item.decimal[2]))
    return QuantityResult(item.name, _plain_or_none(item.numerator), _plain_or_none(item.norm_squared), item.binary64,
                          decimal, _plain_or_none(item.accuracy), item.accuracy_met)


def _rebuild_section(section: _Sec) -> SectionResult:
    quantities = None
    if section.quantities is not None:
        quantities = tuple(_rebuild_quantity(q) for q in _each(section.quantities, "rebuild"))
    topology = None
    if section.topology is not None:
        topology = Topology(
            loops=tuple(TopologyLoop(identity, role) for identity, role in _each(section.topology[0], "rebuild")),
            components=tuple((outer, holes) for outer, holes in _each(section.topology[1], "rebuild")))
    return SectionResult(section.side, section.occupancy, section.section, quantities, topology)


def _rebuild_query(r: _Req) -> HalfSpaceQuery:
    return HalfSpaceQuery(
        normal=_plain3(r.normal), offset=_plain(r.offset), waterplane_origin=_plain3(r.origin),
        waterplane_axes=(_plain3(r.e1), _plain3(r.e2)),
        absolute_accuracy=tuple((name, _plain(v)) for name, v in _each(r.accuracy, "rebuild")),
        forms=tuple(_each(r.forms, "rebuild")))


def _rebuild_result(res: _Res, r: _Req) -> CertifiedHydrostaticsResult:
    return CertifiedHydrostaticsResult(
        query=_rebuild_query(r), occupancy=res.occupancy, generic=res.generic,
        contacts=Contacts(res.contacts[0], res.contacts[1], res.contacts[2]),
        volume=_rebuild_quantity(res.volume), moments=tuple(_rebuild_quantity(m) for m in _each(res.moments, "rebuild")),
        sections=tuple(_rebuild_section(x) for x in _each(res.sections, "rebuild")),
        work={"operations": res.ops, "units": {k: v for k, v in _each(res.units.items(), "rebuild")}},
        proof=res.proof)


# ------------------------------------------------------------------ main entry


class _Gauge:
    """One operation per unit: every primitive unit is a checkpoint (budget, deadline, cancellation)."""

    def __init__(self, meter: Meter):
        self.meter = meter
        self.budget = meter.budget

    def unit(self, kind: str, count: int = 1) -> None:
        self.meter.unit(kind, 1)

    def fit(self, value):
        return self.meter.fit(value)


def _refusal_context(r) -> dict | None:
    """Constant-size context of a refusal from the already ingested request (``None`` before it was ingested)."""
    if r is None:
        return None

    def pairs(vector):
        return [[vector[0].numerator, vector[0].denominator], [vector[1].numerator, vector[1].denominator],
                [vector[2].numerator, vector[2].denominator]]

    return {"normal": pairs(r.normal), "offset": [r.offset.numerator, r.offset.denominator],
            "waterplane_origin": pairs(r.origin), "waterplane_axes": [pairs(r.e1), pairs(r.e2)],
            "absolute_accuracy": getattr(r, "accuracy_pairs", None), "forms": getattr(r, "forms", None)}


def verify_certified_hydrostatics(model, faces, query, result, expected, *, budget: WorkBudget | None = None,
                                  cancellation_check: Callable[[], bool] | None = None) -> VerifiedHydrostatics:
    """Verify ``result`` for ``query`` against ``model`` / ``faces`` and the consumer's ``expected`` identities.

    Raises ``CertifiedHydrostaticsRefusal`` (``VERIFICATION_FAILED`` with ``sub_reason``, ``MALFORMED_PROOF``, or a
    resource category) and never returns a partially verified object. Every value that comes from the caller is
    rebuilt from exact built-in types before it is used (see the module docstring)."""
    meter = _Gauge(Meter(budget, cancellation_check))
    previous = (getattr(_ACTIVE, "gauge", None), getattr(_ACTIVE, "budget", None))
    _ACTIVE.gauge = meter
    _ACTIVE.budget = meter.budget
    holder: dict = {}
    try:
        exp = _expected_identity(expected)      # a malformed ``expected`` is the caller's error: ValueError/TypeError
        try:
            return _verify(model, faces, query, result, exp, holder)
        except (AttributeError, KeyError, TypeError, IndexError, ValueError) as error:
            if isinstance(error, CertifiedHydrostaticsRefusal):
                raise
            raise _refuse(C.MALFORMED_PROOF, f"malformed proof or result structure: {error!r}") from error
    except CertifiedHydrostaticsRefusal as error:
        _ACTIVE.gauge = None            # the error context is not metered work and reads no outside data
        _ACTIVE.budget = None
        error.complete(phase="verify", request=_refusal_context(holder.get("request")),
                       work=meter.budget.counts())
        raise
    finally:
        _ACTIVE.gauge, _ACTIVE.budget = previous


def _verify(model, faces, query, result, exp, holder) -> VerifiedHydrostatics:
    faces = _ingest_faces(faces)
    # (0) the bound query is the caller's query
    mine_req = _ingest_query(query)
    holder["request"] = mine_req
    try:
        bound_req = _ingest_query(result.query)
    except CertifiedHydrostaticsRefusal:
        raise
    except Exception as error:
        raise _fail("query", "the result carries no readable query", error) from error
    mine = _req_dict(mine_req)
    if not _same(_req_dict(bound_req), mine):
        raise _fail("query", "the query bound in the result differs from the supplied query")
    plane = _Plane(mine_req)
    if plane.n == (0, 0, 0) or _dot(_cross(plane.e1, plane.e2), plane.n) == 0:
        raise _refuse(C.INVALID_QUERY, "the half-space or waterplane basis is invalid")
    for form in _each(mine_req.forms, "form_check"):
        if form not in ("exact", "binary64", "decimal"):
            raise _refuse(C.INVALID_QUERY, "unknown form")
    mine_req.want_binary64, mine_req.want_decimal, want_exact = False, False, False
    for form in _each(mine_req.forms, "form_check"):
        mine_req.want_binary64 = mine_req.want_binary64 or form == "binary64"
        mine_req.want_decimal = mine_req.want_decimal or form == "decimal"
        want_exact = want_exact or form == "exact"

    # (1) proof
    res = _ingest_result(result)
    proof = _parse_proof(res.proof, exp["policy"])
    if not _same(proof["query"], mine):
        raise _fail("query", "the proof's query differs from the supplied query")
    claim = _claim_of(res)
    difference = _first_difference(proof["result"], claim)
    if difference:
        path, kind = difference
        raise _fail(_classify(path, kind), f"the proof's claim differs from the result object at {path!r}")
    if not _equals_windows(_encode_windows(proof), res.proof):
        raise _fail("value", "the proof is not in canonical form")
    if not _same(proof["preparation_policy"], exp["policy"]):
        raise _fail("policy", "the proof's preparation policy differs from the expected policy")
    if not _same(proof["convention"], {**_CONVENTION, "convention_version": exp["convention_version"],
                                        "units": exp["units"]}):
        raise _fail("convention", "the convention block differs from the specified one")
    if exp["policy"]["convention_version"] != exp["convention_version"]:
        raise _fail("convention", "the expected policy names another convention version")
    artifact = proof["artifact"]
    if not _same(_sorted_pairs(artifact["files"]), exp["files"]):
        raise _fail("artifact", "the proof's file hashes differ from the expected artifact identity")

    # (2) snapshot, admission, identities
    face_list = [x for x in _each(faces, "face_list")]
    if not _same(face_list, proof["model"]["face_ids"]):
        raise _fail("face set", "the supplied face list differs from the proof's face set")
    _consistent_read(model)
    revision_before = model.revision
    try:
        snap = _read_snapshot(model, faces)
    except CertifiedHydrostaticsRefusal as error:
        if error.category in _RESOURCE:
            raise
        raise _fail("snapshot", f"the model can no longer be read as the prepared snapshot ({error.detail})",
                    error) from error
    if model.revision != revision_before:
        raise _refuse(C.CONCURRENT_MUTATION, "the model changed during verification")
    _consistent_read(model)
    document = _snapshot_document(snap)
    digest = _sha256_windows(_encode_windows(document), "snapshot_hash_chunk")
    pm = proof["model"]
    if not _same(digest, pm["snapshot_digest"]):
        raise _fail("snapshot", "the rebuilt snapshot digest differs from the proof")
    total_volume = _admit(snap)
    model_id = str(model.model_id)
    checksum = _document_checksum(model)
    revision = model.revision
    if not (_same(pm["model_id"], model_id) and _same(pm["revision"], revision)
            and _same(pm["document_checksum"], checksum)):
        raise _fail("model", "the model identity differs from the proof's")
    env = _environment()
    for key in _each(env, "artifact_check"):
        value = env[key]
        if key == "dependencies":
            if not _same(_tsorted(artifact["dependencies"], key=lambda x: x[0]), value):
                raise _fail("artifact", "the claimed dependency versions differ from the running environment")
        elif key == "files":
            if not _same(_sorted_pairs(artifact["files"]), _sorted_pairs(value)):
                raise _fail("artifact", "the package file hashes differ from the running environment")
        elif not _same(artifact[key], value):
            raise _fail("artifact", f"the claimed {key} differs from the running environment")

    # (3) recompute
    truth = _recompute(snap, plane, total_volume)

    # (4)/(5) compare, with completeness
    if res.generic is not truth["generic"]:
        raise _fail("value", "the degeneracy flag differs")
    if (not _same(_tsorted(res.contacts[0]), truth["contacts"]["vertices"])
            or not _same(_tsorted(res.contacts[1]), truth["contacts"]["edges"])
            or not _same(_tsorted(res.contacts[2]), truth["contacts"]["faces"])):
        raise _fail("value", "the contact lists differ")
    if res.occupancy != truth["occupancy"]:
        raise _fail("state", "the exact half-space occupancy differs")
    _check_quantity(res.volume, "V", truth["V"], None, mine_req, True)
    if len(res.moments) != 3:
        raise _fail("request", "the first volume moments are incomplete")
    for item, name, value in _each(zip(res.moments, _MOMENTS, truth["J"]), "compare_element"):
        _check_quantity(item, name, value, None, mine_req, True)
    names = [s.side for s in _each(res.sections, "compare_element")]
    if not _same(names, [k for k in _each(truth["sides"], "compare_element")]):
        raise _fail("state", "the set of reported sides differs")
    for section in _each(res.sections, "compare_element"):
        side = truth["sides"][section.side]
        if section.occupancy != side.occupancy or section.section != side.section:
            raise _fail("state", f"side {section.side}: occupancy or section flag differs")
        if side.section == "ABSENT":
            if section.quantities is not None:
                raise _fail("state", f"side {section.side}: quantities reported for an absent section")
            if section.topology is not None:
                raise _fail("topology", f"side {section.side}: topology reported for an absent section")
            continue
        if section.quantities is None:
            raise _fail("state", f"side {section.side}: quantities missing for a present section")
        if len(section.quantities) != len(_QUANTITIES):
            raise _fail("request", f"side {section.side}: quantity inventory differs")
        for item, name in _each(zip(section.quantities, _QUANTITIES), "compare_element"):
            num, ns = side.quantities[name]
            _check_quantity(item, name, num, ns, mine_req, want_exact)
        if section.topology is None:
            raise _fail("topology", f"side {section.side}: topology missing for a present section")
        got_loops = _tsorted(section.topology[0])
        got_components = _tsorted([(outer, tuple(_tsorted(holes))) for outer, holes in
                                   _each(section.topology[1], "compare_element")])
        if not _same(got_loops, side.topology["loops"]) or not _same(got_components, side.topology["components"]):
            raise _fail("topology", f"side {section.side}: loops or components differ")

    exact = {"V": _plain(truth["V"]), "J": tuple(_plain(x) for x in _each(truth["J"], "compare_element")),
             "occupancy": truth["occupancy"], "generic": truth["generic"],
             "contacts": {k: tuple(_each(v, "compare_element")) for k, v in _each(truth["contacts"].items(), "compare_element")},
             "sides": {name: {"occupancy": side.occupancy, "section": side.section,
                              "quantities": None if side.quantities is None else {
                                  q: (_plain(v[0]), _plain(v[1])) for q, v in _each(side.quantities.items(), "compare_element")},
                              "topology": side.topology}
                       for name, side in _each(truth["sides"].items(), "compare_element")}}
    return VerifiedHydrostatics(
        result=_rebuild_result(res, mine_req), query=_rebuild_query(mine_req), face_ids=faces, model_id=model_id,
        revision=revision, document_checksum=checksum, snapshot_digest=digest,
        artifact_files_sha256=_sha256_windows(_encode_windows(exp["files"]), "files_hash_chunk"),
        proof_sha256=_sha256_chunks(res.proof, "proof_hash_chunk"), solid_volume=_plain(total_volume), exact=exact,
        work=_ACTIVE.gauge.budget.counts())
