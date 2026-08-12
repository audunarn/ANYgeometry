"""Explicit policies for topology-changing geometric operations."""

from __future__ import annotations

from enum import Enum

__all__ = ["MutationPolicy"]


class MutationPolicy(str, Enum):
    """Required caller intent when a query is promoted to a mutation."""

    REJECT = "reject"
    REUSE_EXISTING = "reuse_existing"
    WELD = "weld"
    IMPRINT = "imprint"
    KEEP_SEPARATE_PART = "keep_separate_part"
