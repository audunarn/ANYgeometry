"""Conservative finite-box sweep; no geometric intersection or topology decisions."""
import heapq
import numpy as np
from .errors import GeometryError


def swept_face_pairs(bounds, padding, *, poll=None):
    """Return box candidates in original lexicographic face order.

    ``padding`` bounds every final pair tolerance. Expiry uses the *same*
    subtraction/comparison as the final box predicate, avoiding asymmetric
    rounding from adding padding to upper bounds. The caller still applies
    its exact per-pair predicate and accounts for omitted logical predicates.
    """
    ids = sorted(bounds)
    if len(ids) < 2:
        return ()
    lows = np.asarray([bounds[key][0] for key in ids])
    highs = np.asarray([bounds[key][1] for key in ids])
    if (not np.all(np.isfinite(lows)) or not np.all(np.isfinite(highs))
            or np.any(lows > highs) or not np.isfinite(padding) or padding < 0):
        raise GeometryError('intersection candidate bounds and padding must be finite and ordered')
    # Prefer the direction with the greatest separation relative to box width.
    widths = np.maximum(np.mean(highs - lows, axis=0), np.finfo(float).tiny)
    axis = int(np.argmax(np.ptp(lows, axis=0) / widths))
    order = sorted(range(len(ids)), key=lambda i: (lows[i, axis], ids[i]))
    active = {}
    expiry = []
    pairs = []
    for i in order:
        if poll is not None:
            poll()
        cutoff = lows[i, axis] - padding
        while expiry and expiry[0][0] < cutoff:
            _, expired = heapq.heappop(expiry)
            active.pop(expired, None)
        indices = np.fromiter(active, dtype=np.intp, count=len(active))
        # Test in bounded chunks so cancellation remains available in dense
        # active sets. Only bbox comparisons are vectorized.
        for start in range(0, len(indices), 4096):
            if poll is not None:
                poll()
            selected = indices[start:start + 4096]
            keep = (np.all(highs[selected] >= lows[i] - padding, axis=1)
                    & np.all(highs[i] >= lows[selected] - padding, axis=1))
            pairs.extend((min(ids[i], ids[j]), max(ids[i], ids[j]))
                         for j in selected[keep])
        active[i] = None
        heapq.heappush(expiry, (highs[i, axis], i))
    if poll is not None:
        poll()
    result = tuple(sorted(pairs))
    if poll is not None:
        poll()
    return result
