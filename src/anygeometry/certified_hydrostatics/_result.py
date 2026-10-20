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

    def claim(self, meter=None, *, with_work: bool = True) -> dict:
        """The complete claim as the canonical JSON-able document (schema in the module docstring).

        With ``meter`` every element copied out of the result (contact ids, topology ids, quantities) reserves
        a ``claim_element`` unit first; ``with_work=False`` omits the ``work`` field (the engine appends it last)."""
        tick = _ticker(meter)
        out = {"occupancy": self.occupancy, "generic": self.generic,
               "contacts": {"vertices": _copy(self.contacts.vertices, tick), "edges": _copy(self.contacts.edges, tick),
                            "faces": _copy(self.contacts.faces, tick)},
               "volume": claim_quantity(self.volume, tick),
               "moments": {item.name: claim_quantity(item, tick) for item in self.moments},
               "sections": {item.side: _claim_section(item, tick) for item in self.sections}}
        if with_work:
            out["work"] = {"operations": self.work["operations"], "units": dict(self.work["units"])}
        return out

    def proof_document(self) -> dict:
        return json.loads(self.proof.decode("utf-8"))


def _ticker(meter):
    if meter is None:
        return lambda: None
    return lambda: meter.unit("claim_element")


def _copy(values, tick) -> list:
    out = []
    for value in values:
        tick()
        out.append(value)
    return out


def claim_quantity(item: QuantityResult, tick=lambda: None) -> dict:
    tick()
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


def _claim_section(item: SectionResult, tick=lambda: None) -> dict:
    tick()
    topology = None
    if item.topology is not None:
        topology = {"loops": [], "components": []}
        for loop in item.topology.loops:
            tick()
            topology["loops"].append({"identity": _copy(loop.identity, tick), "role": loop.role})
        for outer, holes in item.topology.components:
            tick()
            topology["components"].append({"outer": _copy(outer, tick), "holes": [_copy(h, tick) for h in holes]})
    quantities = None if item.quantities is None else {q.name: claim_quantity(q, tick) for q in item.quantities}
    return {"occupancy": item.occupancy, "section": item.section, "quantities": quantities, "topology": topology}


def proof_document(prepared, query: HalfSpaceQuery, claim: dict, meter=None) -> dict:
    """The canonical proof document of ``claim`` for ``query`` against ``prepared`` (see the module docstring).

    With ``meter`` the per-element copies of the artifact and model identity reserve ``claim_element`` units."""
    tick = _ticker(meter)
    impl = prepared.implementation
    artifact = {"distribution": impl.distribution, "distribution_version": impl.distribution_version,
                "package_version": impl.package_version, "files": [], "dependencies": [], "python": impl.python}
    for path, digest in impl.files:
        tick()
        artifact["files"].append([path, digest])
    for name, version in impl.dependencies:
        tick()
        artifact["dependencies"].append([name, version])
    model = {"model_id": prepared.model_id, "revision": prepared.revision,
             "document_checksum": prepared.document_checksum, "face_ids": _copy(prepared.face_ids, tick),
             "snapshot_digest": prepared.snapshot_digest}
    convention = {
        "convention_version": prepared.policy.convention_version, "units": prepared.units,
        "body_frame": "model coordinates are the body frame; no rotation, no normalisation",
        "half_space": "n.r <= d with exact rational n, d (unnormalised)",
        "area_element": "true area element dA of the plane; waterplane quantities are numerator/|n|",
        "moment_order": ["A", "Mu", "Mv", "Iuu", "Ivv", "Iuv"],
        "limits": "below: n.r = d - eps, above: n.r = d + eps, eps -> 0+; V and J use the below convention",
        "loop_identity": "sorted ids of the crossed shell edges"}
    return {"artifact": artifact, "model": model, "convention": convention,
            "preparation_policy": prepared.policy.as_dict(), "query": query.as_dict(), "result": claim}


def write_canonical(document, meter=None, *, max_digits: int, max_bytes: int, late=None) -> bytes:
    """Canonical JSON bytes (sorted keys, no spaces, UTF-8), written element by element.

    With ``meter`` every value (scalar, array or object) and every object key reserves one
    ``serialize_element`` unit before it is written, so budget, deadline and cancellation are polled per
    element; without a meter nothing is charged (the default charges everything that is written).
    ``late = (path, key, produce)`` appends ``key: produce()`` as the last key of the object at ``path``
    (``key`` must sort after the others), after every charged element; that value is written without charge:
    the bytes of the proof's ``work`` field are the only part not counted in the work it reports.
    Integers are written without the interpreter's int/str digit limit; one over ``max_digits`` digits, or
    output over ``max_bytes`` *encoded bytes*, refuses ``RESULT_OUT_OF_RANGE``."""
    from ._errors import RefusalCategory, refuse
    from ._arith import digits_exceed, int_to_str
    parts: list = []
    size = 0
    blocks = 0                 # 64 KiB blocks of output charged for the final join and encoding (6b bulk)
    metering = meter is not None

    def emit(text: str) -> None:
        nonlocal size, blocks
        parts.append(text)
        size += len(text) if text.isascii() else len(text.encode("utf-8"))
        if size > max_bytes:
            raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "the proof exceeds the policy's byte limit")
        if metering and size // 65536 + 1 > blocks:
            meter.unit("serialize_bulk", size // 65536 + 1 - blocks)
            blocks = size // 65536 + 1

    def walk(value, path, charged) -> None:
        if charged and meter is not None:
            meter.unit("serialize_element")
        if isinstance(value, dict):
            emit("{")
            keys = sorted(value)
            for index, key in enumerate(keys):
                if charged and meter is not None:
                    meter.unit("serialize_element")
                emit(("," if index else "") + json.dumps(key, ensure_ascii=False) + ":")
                walk(value[key], path + (key,), charged)
            if late is not None and path == tuple(late[0]):
                nonlocal metering
                metering = False       # the work field is written after the count it reports (not charged)
                key = late[1]
                assert not keys or keys[-1] < key
                emit(("," if keys else "") + json.dumps(key, ensure_ascii=False) + ":")
                walk(late[2](), path + (key,), False)
            emit("}")
        elif isinstance(value, (list, tuple)):
            emit("[")
            for index, item in enumerate(value):
                emit("," if index else "")
                walk(item, path, charged)
            emit("]")
        elif value is None or isinstance(value, bool):
            emit("null" if value is None else ("true" if value else "false"))
        elif isinstance(value, int):
            if digits_exceed(value, max_digits):
                raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "an integer of the proof exceeds the digit limit")
            emit(int_to_str(value))
        elif isinstance(value, str):
            # design 6b: escaping one string scalar is one bulk operation, bounded by the byte limit and charged
            # one unit per 64 KiB before it runs
            if len(value) > max_bytes:
                raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "a string of the proof exceeds the byte limit")
            if charged and meter is not None:      # the escape and the UTF-8 size measurement of a non-ASCII string
                meter.unit("serialize_bulk", (1 if value.isascii() else 2) * (len(value) // 65536 + 1))
            emit(json.dumps(value, ensure_ascii=False))
        else:
            raise TypeError(f"cannot serialise {type(value).__name__}")

    walk(document, (), True)
    data = "".join(parts).encode("utf-8")     # its cost was charged per 64 KiB block as the output grew (6b bulk)
    if meter is not None:
        meter.unit("serialize_bulk", 0)        # the checkpoint after the bulk step (polls; charges nothing)
    return data
