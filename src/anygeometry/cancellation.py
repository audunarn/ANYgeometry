"""Owner-issued passive cancellation state.

A ``CancellationSignal`` is opaque live state: it holds no callback, hook, caller event or mutable
attribute, and reading it never executes caller code. The state lives in a private registry keyed by the
exact issued object, so a forged, subclassed, copied or unpickled object holds no identity.
"""
from __future__ import annotations

import threading
from weakref import WeakKeyDictionary

from .errors import GeometryError

__all__ = ["CancellationSignal", "is_cancellation_signal"]

_EVENTS: "WeakKeyDictionary" = WeakKeyDictionary()
_LOCK = threading.Lock()


def _event(token):
    with _LOCK:
        event = _EVENTS.get(token) if type(token) is CancellationSignal else None
    if event is None:
        raise GeometryError('cancellation signal was not issued by this owner')
    return event


class CancellationSignal:
    """A cooperative, thread-safe cancellation flag. ``CancellationSignal()`` takes no arguments."""

    __slots__ = ('__weakref__',)

    def __init_subclass__(cls, **kwargs):
        raise TypeError('CancellationSignal cannot be subclassed')

    def __init__(self):
        with _LOCK:
            if self in _EVENTS:
                raise GeometryError('a cancellation signal cannot be reinitialized')   # monotonic: never reset
            _EVENTS[self] = threading.Event()

    def cancel(self) -> None:
        _event(self).set()

    @property
    def cancelled(self) -> bool:
        return _event(self).is_set()

    def __call__(self, phase=None) -> bool:
        """The proof frontier test: ``True`` once cancelled. ``phase`` is ignored and never executed."""
        return _event(self).is_set()

    def __copy__(self):
        raise GeometryError('a cancellation signal cannot be copied')

    def __deepcopy__(self, memo):
        raise GeometryError('a cancellation signal cannot be copied')

    def __reduce_ex__(self, protocol):
        raise TypeError('a cancellation signal cannot be serialized or cloned')

    def __repr__(self):
        return 'CancellationSignal()'


def is_cancellation_signal(value) -> bool:
    """``True`` only for an exact, live, owner-issued ``CancellationSignal`` (never for callables)."""
    if type(value) is not CancellationSignal:
        return False
    with _LOCK:
        return value in _EVENTS
