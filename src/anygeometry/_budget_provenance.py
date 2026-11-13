"""Per-entry provenance of validation problems that a qualification-budget denial may have caused.

A producer of one validation entry takes ``mark = denial_mark()`` before the (certified) decision and calls
``note_entry(mark)`` when it appends an UNDECIDED entry ("not certified ..."): the entry is then counted on the
plan's ``Work`` only if that very decision observed a denial. Definite verdicts (self-intersects, zero area)
are never counted, so a definite defect keeps the plan's own refusal class at every cap. Structural; no
message text. Outside a plan (no ``Work`` with counters) both functions are inert.
"""
from .plane_nurbs import CURRENT_WORK


def denial_mark():
    return getattr(CURRENT_WORK.get(), 'denied', None)


def note_entry(mark) -> None:
    if mark is not None:
        work = CURRENT_WORK.get()
        if work.denied > mark:
            work.failed_entries += 1
