"""``query_certified_hydrostatics``: the certified half-space query (design sections 3-6, producer side)."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Callable

from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory, refuse
from ._forms import build_quantity
from ._prepare import PreparedCertifiedHydrostatics
from ._query import HalfSpaceQuery, WATERPLANE_NAMES
from ._arith import Ar
from ._result import (CertifiedHydrostaticsResult, Contacts, SectionResult, Topology, TopologyLoop, proof_document,
                      write_canonical)
from ._sections import (build_section, cap_sums, clipped_sums, contacts, exact_occupancy, vertex_heights,
                        waterplane_sums)
from ._work import Meter, WorkBudget

__all__ = ["query_certified_hydrostatics"]


def _delta(after: dict, before: dict) -> dict:
    units = {kind: count - before["units"].get(kind, 0) for kind, count in after["units"].items()
             if count != before["units"].get(kind, 0)}
    return {"operations": after["operations"] - before["operations"], "units": units}


def _side(name: str, section, query: HalfSpaceQuery, nn, meter: Meter) -> SectionResult:
    if not section.points:
        return SectionResult(name, section.occupancy, "ABSENT", None, None)
    sums = waterplane_sums(section, query, meter)
    quantities = tuple(build_quantity(key, sums[key], nn, query.forms, query.accuracy(key), meter)
                       for key in WATERPLANE_NAMES)
    for _ in range(len(section.loops) + len(section.components)):
        meter.unit("topology_element")
    topology = Topology(tuple(TopologyLoop(loop.identity, loop.role) for loop in section.loops), section.components)
    return SectionResult(name, section.occupancy, "PRESENT", quantities, topology)


def _evaluate(prepared: PreparedCertifiedHydrostatics, query: HalfSpaceQuery, meter: Meter,
              before: dict) -> CertifiedHydrostaticsResult:
    snapshot = prepared.snapshot
    if snapshot.curved is not None:         # CH3c: the curved query engine (the planar engine reads polygons only)
        from ._ch3_query import evaluate_curved
        return evaluate_curved(prepared, query, meter, before)
    ar = Ar(meter)
    for number in (*query.normal, query.offset, *query.waterplane_origin, *query.waterplane_axes[0],
                   *query.waterplane_axes[1], *(w for _name, w in query.absolute_accuracy)):
        meter.fit(number)
    nn = ar.dot(query.normal, query.normal)
    heights = vertex_heights(snapshot, query, meter)
    vertices, edges, faces = contacts(snapshot, heights, meter)
    generic = not vertices
    occupancy = exact_occupancy(heights, meter)
    below = build_section(snapshot, query, heights, +1, meter)
    above = None if generic else build_section(snapshot, query, heights, -1, meter)

    total6, moment24 = clipped_sums(snapshot, heights, meter)
    cap6, cap24 = cap_sums(below, meter)
    volume = ar.div(ar.add(total6, cap6), 6)
    moments = [ar.div(ar.add(moment24[k], cap24[k]), 24) for k in range(3)]
    volume_quantity = build_quantity("V", volume, None, query.forms, query.accuracy("V"), meter, always_exact=True)
    moment_quantities = tuple(build_quantity(name, value, None, query.forms, query.accuracy(name), meter,
                                             always_exact=True)
                              for name, value in zip(("Jx", "Jy", "Jz"), moments))
    if generic:
        sections = (_side("plane", below, query, nn, meter),)
    else:
        sections = (_side("below", below, query, nn, meter), _side("above", above, query, nn, meter))
    placeholder = {"operations": 0, "units": {}}
    result = CertifiedHydrostaticsResult(query, occupancy, generic, Contacts(vertices, edges, faces), volume_quantity,
                                         moment_quantities, sections, placeholder)
    limits = dict(max_digits=prepared.policy.max_integer_digits, max_bytes=prepared.policy.max_proof_bytes)
    # one metered pass: the claim is copied out of the result and the document is written element by element,
    # both reserving units and polling; the work field is appended last (uncharged) from the budget delta
    document = proof_document(prepared, query, result.claim(meter, with_work=False), meter)
    taken: dict = {}

    def work_field():
        taken["work"] = _delta(meter.budget.counts(), before)
        return taken["work"]
    proof = write_canonical(document, meter, late=(("result",), "work", work_field), **limits)
    return replace(result, work=taken["work"], proof=proof)


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
