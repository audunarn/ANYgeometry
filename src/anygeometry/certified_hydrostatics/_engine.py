"""``query_certified_hydrostatics``: the certified half-space query (design sections 3-6, producer side)."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Callable

from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory, refuse
from ._forms import build_quantity
from ._prepare import PreparedCertifiedHydrostatics
from ._query import HalfSpaceQuery, WATERPLANE_NAMES
from ._result import (CertifiedHydrostaticsResult, Contacts, SectionResult, Topology, TopologyLoop, proof_document,
                      serialize)
from ._sections import (build_section, cap_sums, clipped_sums, contacts, exact_occupancy, vertex_heights,
                        waterplane_sums)
from ._work import Meter, WorkBudget

__all__ = ["query_certified_hydrostatics"]


def _count_and_audit(value, max_digits: int) -> int:
    """Number of scalar/array elements of a JSON document; refuses integer tokens over the digit limit."""
    if isinstance(value, dict):
        return len(value) + sum(_count_and_audit(v, max_digits) for v in value.values())
    if isinstance(value, (list, tuple)):
        return len(value) + sum(_count_and_audit(v, max_digits) for v in value)
    if isinstance(value, int) and not isinstance(value, bool) and abs(value).bit_length() > max_digits * 3.32:
        if len(str(abs(value))) > max_digits:
            raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "an integer of the proof exceeds the digit limit")
    return 1


def _delta(after: dict, before: dict) -> dict:
    units = {kind: count - before["units"].get(kind, 0) for kind, count in after["units"].items()
             if count != before["units"].get(kind, 0)}
    return {"operations": after["operations"] - before["operations"], "units": units}


def _side(name: str, section, query: HalfSpaceQuery, meter: Meter) -> SectionResult:
    if not section.points:
        return SectionResult(name, section.occupancy, "ABSENT", None, None)
    sums = waterplane_sums(section, query, meter)
    quantities = tuple(build_quantity(key, sums[key], query.normal_squared, query.forms, query.accuracy(key), meter)
                       for key in WATERPLANE_NAMES)
    meter.unit("topology_element", len(section.loops) + len(section.components))
    topology = Topology(tuple(TopologyLoop(loop.identity, loop.role) for loop in section.loops), section.components)
    return SectionResult(name, section.occupancy, "PRESENT", quantities, topology)


def _evaluate(prepared: PreparedCertifiedHydrostatics, query: HalfSpaceQuery, meter: Meter,
              before: dict) -> CertifiedHydrostaticsResult:
    snapshot = prepared.snapshot
    heights = vertex_heights(snapshot, query, meter)
    vertices, edges, faces = contacts(snapshot, heights, meter)
    generic = not vertices
    occupancy = exact_occupancy(heights)
    below = build_section(snapshot, query, heights, +1, meter)
    above = None if generic else build_section(snapshot, query, heights, -1, meter)

    total6, moment24 = clipped_sums(snapshot, heights, meter)
    cap6, cap24 = cap_sums(below, meter)
    volume = meter.fit((total6 + cap6) / 6)
    moments = [meter.fit((moment24[k] + cap24[k]) / 24) for k in range(3)]
    volume_quantity = build_quantity("V", volume, None, query.forms, query.accuracy("V"), meter, always_exact=True)
    moment_quantities = tuple(build_quantity(name, value, None, query.forms, query.accuracy(name), meter,
                                             always_exact=True)
                              for name, value in zip(("Jx", "Jy", "Jz"), moments))
    if generic:
        sections = (_side("plane", below, query, meter),)
    else:
        sections = (_side("below", below, query, meter), _side("above", above, query, meter))
    work = _delta(meter.budget.counts(), before)
    result = CertifiedHydrostaticsResult(query, occupancy, generic, Contacts(vertices, edges, faces), volume_quantity,
                                         moment_quantities, sections, work)
    document = proof_document(prepared, query, result.claim())
    meter.unit("serialize_element", _count_and_audit(document, prepared.policy.max_integer_digits))
    proof = serialize(document)
    if len(proof) > prepared.policy.max_proof_bytes:
        raise refuse(RefusalCategory.RESULT_OUT_OF_RANGE, "the proof exceeds the policy's byte limit")
    return replace(result, proof=proof)


def query_certified_hydrostatics(prepared: PreparedCertifiedHydrostatics, query: HalfSpaceQuery, *,
                                 budget: WorkBudget | None = None,
                                 cancellation_check: Callable[[], bool] | None = None
                                 ) -> CertifiedHydrostaticsResult:
    """Answer ``query`` for the prepared snapshot: exact V and J, both one-sided sections of a degenerate plane,
    waterplane quantities and topology, the requested forms with bounds, and a canonical proof.

    Raises ``CertifiedHydrostaticsRefusal`` (phase ``query``, with work counts) on a budget, deadline, bit-length or
    cancellation stop, an unachievable accuracy, an out-of-range value or a symbolic degeneracy."""
    if not isinstance(prepared, PreparedCertifiedHydrostatics):
        raise TypeError("prepared must be a PreparedCertifiedHydrostatics")
    if not isinstance(query, HalfSpaceQuery):
        raise TypeError("query must be a HalfSpaceQuery")
    meter = Meter(budget, cancellation_check)
    before = meter.budget.counts()
    try:
        return _evaluate(prepared, query, meter, before)
    except CertifiedHydrostaticsRefusal as refusal:
        raise refusal.complete(phase="query", request=query.as_dict(), identity=prepared.model_identity(),
                               work=meter.budget.counts())
