"""Structured refusals of the certified hydrostatics provider (design sections 2 and 7)."""
from __future__ import annotations

import enum
from typing import Any, Mapping

from ..errors import GeometryError

__all__ = ["CertifiedHydrostaticsRefusal", "RefusalCategory"]


class RefusalCategory(enum.Enum):
    """Why a request was refused. Consumers branch on the category, never on the message."""

    UNSUPPORTED_SURFACE = "UNSUPPORTED_SURFACE"
    NOT_EXACTLY_PLANAR = "NOT_EXACTLY_PLANAR"
    INVALID_LOOP = "INVALID_LOOP"
    INVALID_HOLES = "INVALID_HOLES"
    NOT_CLOSED = "NOT_CLOSED"
    NON_MANIFOLD = "NON_MANIFOLD"
    SELF_INTERSECTING = "SELF_INTERSECTING"
    INVERTED_ORIENTATION = "INVERTED_ORIENTATION"
    NONFINITE_INPUT = "NONFINITE_INPUT"
    INVALID_QUERY = "INVALID_QUERY"
    ACCURACY_UNACHIEVABLE = "ACCURACY_UNACHIEVABLE"
    RESULT_OUT_OF_RANGE = "RESULT_OUT_OF_RANGE"
    BIT_LENGTH_EXCEEDED = "BIT_LENGTH_EXCEEDED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CONCURRENT_MUTATION = "CONCURRENT_MUTATION"
    SYMBOLIC_DEGENERACY = "SYMBOLIC_DEGENERACY"
    CANCELLED = "CANCELLED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    STALE_PREPARATION = "STALE_PREPARATION"
    MALFORMED_PROOF = "MALFORMED_PROOF"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    # CH3 kernel (design CERTIFIED_HYDROSTATICS_CH3_DESIGN.md sections 2.3 and 3.2)
    UNDECIDED_PREDICATE = "UNDECIDED_PREDICATE"
    UNSUPPORTED_TRIM = "UNSUPPORTED_TRIM"


PHASES = ("prepare", "query", "verify", "check")


class CertifiedHydrostaticsRefusal(GeometryError):
    """A request the provider refuses; final for that request (no retry with a changed problem).

    ``category`` is the public enum, ``phase`` one of ``prepare``/``query``/``verify``/``check``,
    ``request`` and ``identity`` the request and model/preparation identity known at the refusal,
    ``work`` the counts consumed so far (``WorkBudget.counts()``), ``sub_reason`` the discrepancy class of a
    ``VERIFICATION_FAILED`` refusal (``None`` for every other refusal), ``__cause__`` the original error."""

    def __init__(self, category: RefusalCategory, message: str, *, phase: str = "",
                 request: Mapping[str, Any] | None = None, identity: Mapping[str, Any] | None = None,
                 work: Mapping[str, Any] | None = None, sub_reason: str | None = None) -> None:
        super().__init__(f"{category.value}: {message}")
        self.category = category
        self.sub_reason = sub_reason          # set for VERIFICATION_FAILED (design 6a), None otherwise
        self.detail = message
        self.phase = phase
        self.request = request
        self.identity = identity
        self.work = work

    def complete(self, *, phase: str, request=None, identity=None, work=None) -> "CertifiedHydrostaticsRefusal":
        """Fill the context a deeper layer did not know (keeps the first value of each field)."""
        self.phase = self.phase or phase
        self.request = self.request if self.request is not None else request
        self.identity = self.identity if self.identity is not None else identity
        self.work = self.work if self.work is not None else work
        return self


def refuse(category: RefusalCategory, message: str, cause: BaseException | None = None):
    """Build the refusal to ``raise`` (``raise refuse(...)``), chaining ``cause`` when given."""
    error = CertifiedHydrostaticsRefusal(category, message)
    if cause is not None:
        error.__cause__ = cause
    return error
