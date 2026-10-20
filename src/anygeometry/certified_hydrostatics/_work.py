"""Shared work accounting: primitive units, deadline, cancellation and the bit-length cap (design section 7).

A primitive unit is one bounded predicate or accumulation. Each unit reserves its cost *before* it
runs and is a checkpoint for the budget, the deadline and the caller's ``cancellation_check``, so the
overshoot after a refusal and the non-preemptible span are one unit. The budget is shareable across
phases and threads (a lock guards the counters)."""
from __future__ import annotations

import threading
import time
from fractions import Fraction
from typing import Callable

from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory

__all__ = ["WorkBudget"]


class WorkBudget:
    """Limits on primitive operations, on the bit length of exact values and (optionally) on time.

    ``max_operations`` / ``max_bits``: non-negative ints, or ``None`` for no limit. ``deadline`` is an
    absolute ``clock()`` value (``time.monotonic`` by default) or ``None``."""

    def __init__(self, max_operations: int | None, max_bits: int | None, deadline: float | None = None,
                 *, clock: Callable[[], float] = time.monotonic) -> None:
        for name, value in (("max_operations", max_operations), ("max_bits", max_bits)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))):
            raise ValueError("deadline must be a number or None")
        self.max_operations = max_operations
        self.max_bits = max_bits
        self.deadline = deadline
        self._clock = clock
        self._lock = threading.Lock()
        self._operations = 0
        self._bits = 0
        self._kinds: dict[str, int] = {}

    @classmethod
    def unlimited(cls) -> "WorkBudget":
        return cls(None, None)

    @property
    def operations(self) -> int:
        return self._operations

    def counts(self) -> dict:
        """The work consumed so far: total operations, the largest bit length seen and units by kind."""
        with self._lock:
            return {"operations": self._operations, "max_bits_seen": self._bits, "units": dict(self._kinds)}

    def reserve(self, kind: str, count: int = 1, cancellation_check: Callable[[], bool] | None = None) -> None:
        """Reserve ``count`` units of ``kind`` before they run; a refusal consumes nothing."""
        if cancellation_check is not None and cancellation_check():
            raise CertifiedHydrostaticsRefusal(RefusalCategory.CANCELLED, "cancelled by the caller")
        if self.deadline is not None and self._clock() >= self.deadline:
            raise CertifiedHydrostaticsRefusal(RefusalCategory.DEADLINE_EXCEEDED, "the deadline has passed")
        with self._lock:
            if self.max_operations is not None and self._operations + count > self.max_operations:
                raise CertifiedHydrostaticsRefusal(
                    RefusalCategory.BUDGET_EXHAUSTED,
                    f"operation budget of {self.max_operations} exhausted at '{kind}'")
            self._operations += count
            self._kinds[kind] = self._kinds.get(kind, 0) + count

    def check_value(self, value):
        """Return ``value`` if its numerator and denominator fit ``max_bits`` (ints and Fractions)."""
        if isinstance(value, Fraction):
            bits = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        else:
            bits = abs(int(value)).bit_length()
        if bits > self._bits:
            self._bits = bits
        if self.max_bits is not None and bits > self.max_bits:
            raise CertifiedHydrostaticsRefusal(
                RefusalCategory.BIT_LENGTH_EXCEEDED,
                f"an exact value needs {bits} bits, the cap is {self.max_bits}")
        return value


class Meter:
    """A budget bound to one call's cancellation check: the object the algorithms use."""

    __slots__ = ("budget", "cancellation_check")

    def __init__(self, budget: WorkBudget | None, cancellation_check: Callable[[], bool] | None = None) -> None:
        self.budget = WorkBudget.unlimited() if budget is None else budget
        if not isinstance(self.budget, WorkBudget):
            raise TypeError("budget must be a WorkBudget or None")
        if cancellation_check is not None and not callable(cancellation_check):
            raise TypeError("cancellation_check must be callable or None")
        self.cancellation_check = cancellation_check

    def unit(self, kind: str, count: int = 1) -> None:
        self.budget.reserve(kind, count, self.cancellation_check)

    def fit(self, value):
        return self.budget.check_value(value)
