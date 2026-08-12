"""Topology entities with persistent identity.

Identity is the load-bearing idea here.  Loads, boundary conditions, sections
and materials all reference entities by ``EntityRef``, never by coordinate or
by index into a mesh.  IDs are allocated monotonically per kind and are never
reused, so an attribute keeps pointing at the thing the user picked even after
the model is edited and re-meshed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Tuple

import numpy as np

from .curves import CurveShape
from .errors import GeometryError
from .structural import FrozenMetadata, freeze_metadata

if TYPE_CHECKING:
    from .surfaces import Surface

__all__ = [
    "Edge",
    "EntityKind",
    "EntityRef",
    "Face",
    "OrientedEdge",
    "Vertex",
]

EntityKind = Literal["vertex", "edge", "face"]


@dataclass(frozen=True)
class EntityRef:
    """A stable reference to one geometry entity."""

    kind: EntityKind
    id: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.kind}{self.id}"


@dataclass(frozen=True, slots=True)
class Vertex:
    """A modelled point."""

    id: int
    position: np.ndarray

    def __post_init__(self) -> None:
        position = np.array(self.position, dtype=float, copy=True)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise GeometryError("vertex position must be a finite 3-vector")
        position.flags.writeable = False
        object.__setattr__(self, "position", position)

    @property
    def ref(self) -> EntityRef:
        return EntityRef("vertex", self.id)


@dataclass(frozen=True, slots=True)
class Edge:
    """A line between two vertices, with a curve shape."""

    id: int
    start: int
    end: int
    curve: CurveShape

    @property
    def ref(self) -> EntityRef:
        return EntityRef("edge", self.id)

    def other_vertex(self, vertex_id: int) -> int:
        """Return the far vertex of this edge."""

        if vertex_id == self.start:
            return self.end
        if vertex_id == self.end:
            return self.start
        raise ValueError(f"vertex {vertex_id} is not on edge {self.id}")


@dataclass(frozen=True)
class OrientedEdge:
    """One edge traversed in a stated direction within a face loop."""

    edge: int
    forward: bool


@dataclass(frozen=True, slots=True)
class Face:
    """A structural surface bounded by an outer loop and optional inner loops.

    ``corners`` is deprecated mapped-meshing compatibility metadata.  It is
    either empty for a neutral arbitrary-loop face or contains four loop
    indices defining mapped sides; meshing policy remains in ANYmesher.
    """

    id: int
    loop: Tuple[OrientedEdge, ...]
    corners: Tuple[int, ...] = ()
    metadata: FrozenMetadata | Mapping[str, object] = FrozenMetadata()
    holes: Tuple[Tuple[OrientedEdge, ...], ...] = ()
    surface: "Surface | None" = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "loop", tuple(self.loop))
        object.__setattr__(self, "corners", tuple(self.corners))
        object.__setattr__(
            self,
            "holes",
            tuple(tuple(loop) for loop in self.holes),
        )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def ref(self) -> EntityRef:
        return EntityRef("face", self.id)

    def side(self, index: int) -> Tuple[OrientedEdge, ...]:
        """Return the oriented edge chain forming side ``index`` (0..3)."""

        if len(self.corners) != 4:
            raise ValueError(
                f"face {self.id} has no four-side mapped parameterization"
            )
        if not 0 <= index < 4:
            raise IndexError("a mapped face has exactly four sides (0..3)")
        start = self.corners[index]
        stop = self.corners[(index + 1) % 4]
        if stop > start:
            return self.loop[start:stop]
        # The last side wraps past the end of the loop.
        return self.loop[start:] + self.loop[:stop]

    def sides(self) -> Tuple[Tuple[OrientedEdge, ...], ...]:
        return tuple(self.side(k) for k in range(4))
