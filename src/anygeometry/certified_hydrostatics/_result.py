"""The query result object and its canonical proof (design section 6, producer side).

Canonical proof schema (JSON, sorted keys, no whitespace, UTF-8; rationals are ``[numerator, denominator]``
integer pairs in lowest terms with a positive denominator; doubles are ``float.hex()`` strings; decimals are
strings)::

    {
      "artifact":   <ImplementationIdentity.as_dict()>,
      "model":      {"model_id", "revision", "document_checksum", "face_ids", "snapshot_digest"},
      "convention": {"convention_version", "units", "body_frame", "half_space", "area_element", "moment_order",
                     "limits", "loop_identity"},
      "preparation_policy": <CertifiedHydrostaticsPolicy.as_dict()>,
      "query":      <HalfSpaceQuery.as_dict()>,
      "result":     <CertifiedHydrostaticsResult.claim()>
    }

    claim() = {
      "occupancy": "EMPTY" | "FULL" | "PARTIAL",          # exact half-space; governs V and J
      "generic":   bool,                                  # no shell vertex on the plane
      "contacts":  {"vertices": [ids], "edges": [ids], "faces": [ids]},   # sorted
      "volume":    QUANTITY,                              # V
      "moments":   {"Jx": QUANTITY, "Jy": QUANTITY, "Jz": QUANTITY},
      "sections":  {"plane": SECTION}                     # generic plane
                 | {"below": SECTION, "above": SECTION},  # degenerate plane
      "work":      {"operations": int, "units": {kind: int}}   # this query's computation, before serialisation
    }
    SECTION  = {"occupancy": "EMPTY"|"FULL"|"PARTIAL", "section": "PRESENT"|"ABSENT",
                "quantities": {"A","Mu","Mv","Iuu","Ivv","Iuv": QUANTITY} | None,
                "topology": {"loops": [{"identity": [edge ids], "role": "outer"|"hole"}, ...],   # by identity
                             "components": [{"outer": [edge ids], "holes": [[edge ids], ...]}, ...]} | None}
    QUANTITY = {"exact": {"numerator": PAIR, "norm_squared": PAIR | null} | absent,   # value = numerator, or
                                                                  # numerator / sqrt(norm_squared)
                "binary64": {"lo": hex, "hi": hex} | absent,
                "decimal":  {"lo": str, "hi": str, "width": PAIR} | absent,
                "accuracy": PAIR | null, "accuracy_met": true | null}

``exact`` is always present for ``V`` and ``Jx/Jy/Jz`` (``norm_squared`` null) and present for the waterplane
quantities exactly when ``"exact"`` is among the requested forms; ``binary64``/``decimal`` exactly when requested.
``accuracy`` is the requested accuracy (null when none) and ``accuracy_met`` is true when one was requested (a
request that cannot be met is refused). The ``decimal`` width is the requested accuracy, or 10^-12 when the
decimal form is requested for a quantity without one."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction

from ._forms import QuantityResult
from ._query import HalfSpaceQuery

__all__ = ["CertifiedHydrostaticsResult", "SectionResult", "Topology", "TopologyLoop", "Contacts",
           "claim_quantity", "proof_document"]


def _pair(x: Fraction) -> list:
    return [x.numerator, x.denominator]


@dataclass(frozen=True)
class Contacts:
    vertices: tuple
    edges: tuple
    faces: tuple


@dataclass(frozen=True)
class TopologyLoop:
    identity: tuple      # sorted crossed-edge ids
    role: str            # "outer" | "hole"


@dataclass(frozen=True)
class Topology:
    loops: tuple         # TopologyLoop sorted by identity
    components: tuple    # (outer identity, (hole identity, ...)) sorted

    @property
    def outer_count(self) -> int:
        return sum(1 for loop in self.loops if loop.role == "outer")

    @property
    def hole_count(self) -> int:
        return sum(1 for loop in self.loops if loop.role == "hole")


@dataclass(frozen=True)
class SectionResult:
    """One side of the plane: ``plane`` (generic), or ``below`` / ``above`` (degenerate)."""

    side: str
    occupancy: str
    section: str                      # "PRESENT" | "ABSENT"
    quantities: tuple | None          # QuantityResult in order A, Mu, Mv, Iuu, Ivv, Iuv; None when ABSENT
    topology: Topology | None

    def quantity(self, name: str) -> QuantityResult:
        for item in self.quantities or ():
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class CertifiedHydrostaticsResult:
    query: HalfSpaceQuery
    occupancy: str
    generic: bool
    contacts: Contacts
    volume: QuantityResult
    moments: tuple                    # QuantityResult Jx, Jy, Jz
    sections: tuple                   # SectionResult: ("plane",) or ("below", "above")
    work: dict = field(compare=False)
    proof: bytes = field(default=b"", repr=False)

    def section(self, side: str) -> SectionResult:
        for item in self.sections:
            if item.side == side:
                return item
        raise KeyError(side)

    @property
    def degenerate(self) -> bool:
        return not self.generic

    def claim(self) -> dict:
        """The complete claim as the canonical JSON-able document (schema in the module docstring)."""
        return {"occupancy": self.occupancy, "generic": self.generic,
                "contacts": {"vertices": list(self.contacts.vertices), "edges": list(self.contacts.edges),
                             "faces": list(self.contacts.faces)},
                "volume": claim_quantity(self.volume),
                "moments": {item.name: claim_quantity(item) for item in self.moments},
                "sections": {item.side: _claim_section(item) for item in self.sections},
                "work": {"operations": self.work["operations"], "units": dict(self.work["units"])}}

    def proof_document(self) -> dict:
        return json.loads(self.proof.decode("utf-8"))


def claim_quantity(item: QuantityResult) -> dict:
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


def _claim_section(item: SectionResult) -> dict:
    topology = None
    if item.topology is not None:
        topology = {"loops": [{"identity": list(loop.identity), "role": loop.role} for loop in item.topology.loops],
                    "components": [{"outer": list(outer), "holes": [list(h) for h in holes]}
                                   for outer, holes in item.topology.components]}
    quantities = None if item.quantities is None else {q.name: claim_quantity(q) for q in item.quantities}
    return {"occupancy": item.occupancy, "section": item.section, "quantities": quantities, "topology": topology}


def proof_document(prepared, query: HalfSpaceQuery, claim: dict) -> dict:
    """The canonical proof document of ``claim`` for ``query`` against ``prepared`` (see the module docstring)."""
    identity = prepared.proof_identity()
    convention = {
        "convention_version": prepared.policy.convention_version, "units": prepared.units,
        "body_frame": "model coordinates are the body frame; no rotation, no normalisation",
        "half_space": "n.r <= d with exact rational n, d (unnormalised)",
        "area_element": "true area element dA of the plane; waterplane quantities are numerator/|n|",
        "moment_order": ["A", "Mu", "Mv", "Iuu", "Ivv", "Iuv"],
        "limits": "below: n.r = d - eps, above: n.r = d + eps, eps -> 0+; V and J use the below convention",
        "loop_identity": "sorted ids of the crossed shell edges"}
    return {"artifact": identity["artifact"], "model": identity["model"], "convention": convention,
            "preparation_policy": prepared.policy.as_dict(), "query": query.as_dict(), "result": claim}


def write_canonical(document, meter=None, *, max_digits: int, max_bytes: int, uncharged=()) -> bytes:
    """Canonical JSON bytes (sorted keys, no spaces, UTF-8), written element by element.

    With ``meter`` every value (scalar, array or object) and every object key reserves one
    ``serialize_element`` unit before it is written, so budget, deadline and cancellation are polled per
    element. ``uncharged`` is a path of keys whose subtree is written without charge (the ``work`` field of
    the proof, whose own elements are not part of the work it reports). Integers are written without the
    interpreter's int/str digit limit; one over ``max_digits`` digits, or output over ``max_bytes``,
    refuses ``RESULT_OUT_OF_RANGE``."""
    from ._errors import RefusalCategory, refuse
    from ._arith import int_to_str
    limit = 10 ** max_digits
    parts: list = []
    size = 0

    def emit(text: str) -> None:
        nonlocal size
        parts.append(text)
        size += len(text)
        if size > max_bytes:
            raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "the proof exceeds the policy's byte limit")

    def walk(value, path) -> None:
        charged = meter is not None and path[:len(uncharged)] != tuple(uncharged)
        if charged:
            meter.unit("serialize_element")
        if isinstance(value, dict):
            emit("{")
            for index, key in enumerate(sorted(value)):
                if charged:
                    meter.unit("serialize_element")
                emit(("," if index else "") + json.dumps(key, ensure_ascii=False) + ":")
                walk(value[key], path + (key,))
            emit("}")
        elif isinstance(value, (list, tuple)):
            emit("[")
            for index, item in enumerate(value):
                emit("," if index else "")
                walk(item, path)
            emit("]")
        elif value is None or isinstance(value, bool):
            emit("null" if value is None else ("true" if value else "false"))
        elif isinstance(value, int):
            if abs(value) >= limit:
                raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "an integer of the proof exceeds the digit limit")
            emit(int_to_str(value))
        elif isinstance(value, str):
            emit(json.dumps(value, ensure_ascii=False))
        else:
            raise TypeError(f"cannot serialise {type(value).__name__}")

    walk(document, ())
    return "".join(parts).encode("utf-8")
