"""Deterministic member and bay layout shared by all representations."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPS = 1.0e-9


def positive_spacing(value: object, tolerance: float = EPS) -> float:
    if isinstance(value, (bool, np.bool_)):
        return 0.0
    try:
        spacing = float(value)
    except (TypeError, ValueError):
        return 0.0
    return spacing if math.isfinite(spacing) and spacing > tolerance else 0.0


def centered_member_positions(
    total_length: float,
    spacing: float,
    *,
    fallback_midpoint: bool = True,
    max_count: int | None = 1000,
    include_ends: bool = False,
) -> tuple[float, ...]:
    total_length = max(float(total_length), EPS)
    spacing = positive_spacing(spacing)
    tolerance = max(total_length * EPS, EPS)
    if spacing <= 0.0:
        return (0.5 * total_length,) if fallback_midpoint else ()
    full_count = int(math.floor(total_length / spacing + EPS))
    if full_count <= 0:
        return (0.5 * total_length,) if fallback_midpoint else ()
    offset = 0.5 * (total_length - full_count * spacing)
    if offset <= tolerance:
        positions = (
            [spacing * index for index in range(full_count + 1)]
            if include_ends
            else [spacing * index for index in range(1, full_count)]
        )
        if include_ends:
            positions[-1] = total_length
    else:
        positions = [offset + spacing * index for index in range(full_count + 1)]
    if not include_ends:
        positions = [value for value in positions if tolerance < value < total_length - tolerance]
    if not positions and fallback_midpoint:
        positions = [0.5 * total_length]
    if max_count is not None and max_count > 0 and len(positions) > max_count:
        positions = list(symmetric_samples(positions, max_count))
    return tuple(float(value) for value in positions)


def centered_bay_breaks(total_length: float, spacing: float, *, max_count: int | None = 1000) -> tuple[float, ...]:
    stations = centered_member_positions(total_length, spacing, fallback_midpoint=False, max_count=max_count)
    return cleanup_axis((0.0, *stations, total_length), total_length)


def cleanup_axis(values: Iterable[float], total_length: float, tolerance: float | None = None) -> tuple[float, ...]:
    total_length = max(float(total_length), EPS)
    tolerance = max(total_length * EPS, EPS) if tolerance is None else max(float(tolerance), 0.0)
    clean: list[float] = []
    for value in sorted(float(item) for item in values):
        value = min(max(value, 0.0), total_length)
        if not clean or abs(value - clean[-1]) > tolerance:
            clean.append(value)
    if not clean:
        clean = [0.0, total_length]
    clean[0] = 0.0 if abs(clean[0]) <= tolerance else clean[0]
    clean[-1] = total_length if abs(clean[-1] - total_length) <= tolerance else clean[-1]
    return tuple(clean)


def bay_ranges(total_length: float, supports: Iterable[float], support_gap: float = 0.0) -> tuple[tuple[float, float], ...]:
    total_length = max(float(total_length), 0.0)
    support_gap = max(float(support_gap), 0.0)
    if total_length <= EPS:
        return ()
    tolerance = max(total_length * EPS, EPS)
    internal = sorted(float(value) for value in supports if tolerance < float(value) < total_length - tolerance)
    ranges = []
    for start, end in zip((0.0, *internal), (*internal, total_length)):
        left = support_gap / 2.0 if any(abs(start - value) <= tolerance for value in internal) else 0.0
        right = support_gap / 2.0 if any(abs(end - value) <= tolerance for value in internal) else 0.0
        if end - right > start + left:
            ranges.append((float(start + left), float(end - right)))
    return tuple(ranges)


def closed_loop_member_count(total_length: float, spacing: float) -> int:
    if isinstance(total_length, (bool, np.bool_)) or isinstance(
        spacing, (bool, np.bool_)
    ):
        return 0
    try:
        total_length, spacing = float(total_length), float(spacing)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(total_length) or not math.isfinite(spacing) or total_length <= 0.0 or spacing <= EPS:
        return 0
    return max(int(round(total_length / spacing)), 1)


def symmetric_samples(positions: Sequence[float], max_count: int) -> tuple[float, ...]:
    values = list(positions)
    if max_count <= 0 or len(values) <= max_count:
        return tuple(values)
    if max_count == 1:
        return (values[len(values) // 2],)
    last = len(values) - 1
    indexes = sorted({round(index * last / (max_count - 1)) for index in range(max_count)})
    return tuple(values[int(index)] for index in indexes)
