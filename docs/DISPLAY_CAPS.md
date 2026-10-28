# Explicit box-volume display caps

`anygeometry.display_caps.box_volume_cut_caps(volume, planes)` queries transient
convex cap polygons for the canonical `volume_cut.BoxVolume`. Each plane is
`(normal, offset)` and retains `normal . point >= offset`. All planes are admitted
before clipping. Results contain the originating plane index and read-only
outward-wound world-coordinate vertices.

The operand is an explicitly authored closed volume. Structural faces, closed
shells and unqualified triangle clouds are not promoted to material solids.
The source volume, topology and engineering model are never edited. Empty or
tangent lower-dimensional intersections return no cap. The query handles rigid
box transforms and intersects all supplied planes; output polygons are display
geometry, not a qualified engineering section calculation.

ANY3dView's optional BoxVolume cap adapter binds an explicit source handle and
semantic owner to this query and triangulates each convex polygon for display.
That association must match current source coordinates; transformed/deformed
sources need an owner-updated provider. Neutral caps carry no numerical values.
Other volume types remain unavailable until their owning geometry service
supplies the corresponding admission and cut query.
