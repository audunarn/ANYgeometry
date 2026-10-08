"""Opt-in runtime diagnostics counters for geometry hot paths.

``collect_runtime_diagnostics`` is a context manager yielding a fresh
``dict[str, int]`` that counts named internal operations performed within its
dynamic scope. Counters are ContextVar-scoped: nested collectors each
independently receive the increments for the work within their own scope,
concurrent contexts are unaffected, and the scope is reset safely when the
block exits through an exception.

Counting is strictly observational. Increments are a no-op when no collector
is active, and counters never influence geometry decisions, validation or
serialized output. The module holds no model state and persists nothing.
"""
from contextlib import contextmanager
from contextvars import ContextVar

COUNTER_KEYS = ("attachment_clip_calls", "attachment_bounds_tests",
                "attachment_bounds_pruned")

_COLLECTORS: ContextVar[tuple] = ContextVar(
    "anygeometry_runtime_diagnostics_collectors", default=())


@contextmanager
def collect_runtime_diagnostics():
    """Yield a fresh counter dict filled with every counter at zero."""
    counters = dict.fromkeys(COUNTER_KEYS, 0)
    token = _COLLECTORS.set((*_COLLECTORS.get(), counters))
    try:
        yield counters
    finally:
        _COLLECTORS.reset(token)


def _increment(key, amount=1):
    """Increment every collector active in the current scope; no-op without one."""
    collectors = _COLLECTORS.get()
    if not collectors:
        return
    for counters in collectors:
        counters[key] = counters.get(key, 0) + amount
