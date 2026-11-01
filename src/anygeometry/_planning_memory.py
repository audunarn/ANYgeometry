"""Explicit H + P/W admission for the experimental planner.

H remains outside the owned Job. Its declared peak is checked against the
current-process lifetime private-commit peak; it is not an allocation limit.
The separate P/W cap is enforced and read back by OwnedTree. No defaults here
grant a machine-wide budget, a free-RAM fraction, or whole-model authority.
"""
import ctypes
import sys
from dataclasses import dataclass

import psutil

from ._planning_executor import PlannerMemoryExceeded
from ._planning_qualification import QualificationDecline


_SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


def _integer(value, name, *, minimum=1):
    if type(value) is not int or not minimum <= value <= _SIZE_MAX:
        raise QualificationDecline('invalid memory ' + name)
    return value


def observe_memory():
    """Observe only H and aggregate available RAM, without a process census."""
    if sys.platform != 'win32':
        raise QualificationDecline('Windows private-commit accounting unavailable')
    info = psutil.Process().memory_info()
    return dict(host_peak_bytes=info.peak_pagefile,
                host_private_bytes=info.private,
                available_bytes=psutil.virtual_memory().available)


def _sample():
    value = observe_memory()
    if type(value) is not dict or set(value) != {
            'host_peak_bytes', 'host_private_bytes', 'available_bytes'}:
        raise QualificationDecline('unknown native memory accounting')
    peak = _integer(value['host_peak_bytes'], 'host peak')
    private = _integer(value['host_private_bytes'], 'host private', minimum=0)
    available = _integer(value['available_bytes'], 'available RAM', minimum=0)
    if private > peak:
        raise QualificationDecline('host private commit exceeds reported lifetime peak')
    return peak, private, available


@dataclass(frozen=True)
class MemoryBudget:
    host_peak_estimate_bytes: int
    tree_limit_bytes: int
    total_limit_bytes: int
    free_fraction: tuple

    def __post_init__(self):
        _integer(self.host_peak_estimate_bytes, 'host estimate')
        _integer(self.tree_limit_bytes, 'tree cap')
        _integer(self.total_limit_bytes, 'aggregate cap')
        fraction = self.free_fraction
        if (type(fraction) is not tuple or len(fraction) != 2 or
                any(type(v) is not int or not 0 < v <= _SIZE_MAX for v in fraction) or
                fraction[0] > fraction[1]):
            raise QualificationDecline('invalid memory free-RAM fraction')

    def begin(self):
        # Revalidate and snapshot primitive values even if a frozen object was
        # deliberately modified with object.__setattr__ before this call.
        self.__post_init__()
        host, tree, total = (self.host_peak_estimate_bytes, self.tree_limit_bytes,
                             self.total_limit_bytes)
        numerator, denominator = self.free_fraction
        try:
            peak, private, available = _sample()
        except QualificationDecline:
            raise
        except Exception as exc:
            raise QualificationDecline('native memory accounting unavailable: ' + type(exc).__name__) from exc
        if peak > host:
            raise QualificationDecline('observed host peak exceeds declared estimate')
        # Exact ceil(5*(H+PW)/4), no floating-point allowance or overflow.
        required = (5 * (host + tree) + 3) // 4
        effective = min(total, (available * numerator) // denominator)
        if required > effective:
            raise QualificationDecline('host plus tree and 25 percent headroom exceed memory admission budget')
        return MemoryLease(host, tree, total, (numerator, denominator),
                           peak, private, available, required, effective)


class MemoryLease:
    """One admitted plan; policy values never read again from caller controls."""
    def __init__(self, host, tree, total, fraction, peak, private, available, required, effective):
        self._policy = (host, tree, total, fraction, available, required, effective)
        self._peak = peak
        self._private = private
        self._samples = 1
        self._terminal_reason = None

    @property
    def tree_limit_bytes(self):
        return self._policy[1]

    def check(self):
        if self._terminal_reason is not None:
            raise PlannerMemoryExceeded(self._terminal_reason)
        try:
            peak, private, _available = _sample()
        except Exception as exc:
            self._terminal_reason = 'admitted host memory accounting lost: ' + type(exc).__name__
            raise PlannerMemoryExceeded(self._terminal_reason) from exc
        self._samples += 1
        previous = self._peak
        self._peak = max(previous, peak)
        self._private = private
        if peak < previous:
            self._terminal_reason = 'host lifetime private-commit peak decreased'
        elif peak > self._policy[0]:
            self._terminal_reason = 'host private-commit peak exceeded admitted estimate'
        if self._terminal_reason is not None:
            raise PlannerMemoryExceeded(self._terminal_reason)

    def report(self):
        host, tree, total, fraction, available, required, effective = self._policy
        return dict(scope='planning-and-CI', host_accounting='Windows lifetime peak private commit',
                    host_peak_estimate_bytes=host, tree_limit_bytes=tree,
                    total_limit_bytes=total, free_fraction=fraction,
                    initial_available_bytes=available,
                    required_with_headroom_bytes=required, effective_limit_bytes=effective,
                    observed_host_peak_bytes=self._peak, observed_host_private_bytes=self._private,
                    samples=self._samples, terminal_reason=self._terminal_reason,
                    host_allocation_enforced=False)
