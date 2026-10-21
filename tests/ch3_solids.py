"""CH3a2 test fixtures: the closed solids F-A, F-B, F-C of the CH3 design (section 8.2) and the outward B1 full hull.

``build_face`` (CH3a1 tests) makes the curved face; the planar faces close it as the design describes. Loops are
oriented consistently by the library's ``_consistent_loops`` and then all reversed when the exact signed volume of
the result is negative (``outward``), so every solid has outward loops."""
from __future__ import annotations

from dataclasses import replace

from anygeometry import Plane
from anygeometry.b1_example import _consistent_loops
from anygeometry.entities import OrientedEdge


def _reverse(loop):
    return tuple(OrientedEdge(e.edge, not e.forward) for e in reversed(loop))


def flip_all(model, face_ids) -> None:
    """Reverse every face loop (the whole shell's orientation)."""
    for fid in face_ids:
        face = model.faces[fid]
        model._put_entity("face", replace(face, loop=_reverse(face.loop)))


def _close(model, curved_face, curved_edges, planar_edge_lists, planar_surfaces, flip=False):
    loops = _consistent_loops(model, [list(curved_edges)] + [list(e) for e in planar_edge_lists])
    face = model.faces[curved_face]
    model._put_entity("face", replace(face, loop=tuple(loops[0])))
    ids = [curved_face]
    for loop, surface in zip(loops[1:], planar_surfaces):
        ids.append(model.add_face_from_loop(loop, surface=surface))
    if flip:
        flip_all(model, ids)
    return ids


def _orient_outward(model, ids):
    """Reverse the shell when it is inward (the exact signed volume is negative)."""
    from anygeometry.certified_hydrostatics import CertifiedHydrostaticsRefusal, RefusalCategory, \
        prepare_certified_hydrostatics
    try:
        prepare_certified_hydrostatics(model, ids)
    except CertifiedHydrostaticsRefusal as refusal:
        if refusal.category is not RefusalCategory.INVERTED_ORIENTATION:
            return
        flip_all(model, ids)


def solid_fa(build_face, surface_fa, planes_fa):
    """F-A: the parabolic prism: the curved face, four planar faces with one arc each, and the plane ``y = 0``."""
    model, face, vertices, edges = build_face(surface_fa(), planes_fa())
    c = [model.vertices[v].position for v in vertices]
    p = [model.add_point(float(x), 0., float(z)) for x, _y, z in c]
    n = 4
    lines = [model.add_line(vertices[i], p[i]) for i in range(n)]
    along = [model.add_line(p[i], p[(i + 1) % n]) for i in range(n)]
    planar = [[edges[i], lines[(i + 1) % n], along[i], lines[i]] for i in range(n)] + [list(along)]
    surfaces = planes_fa() + [Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.))]
    ids = _close(model, face, edges, planar, surfaces)
    _orient_outward(model, ids)
    return model, ids


def solid_fc(build_face, surface_fc, planes_fc):
    """F-C: the diamond prism: the top curved face, four vertical planar faces (the arcs are lines), the bottom."""
    model, face, vertices, edges = build_face(surface_fc(), planes_fc())
    c = [model.vertices[v].position for v in vertices]
    b = [model.add_point(float(x), float(y), 0.) for x, y, _z in c]
    n = 4
    down = [model.add_line(vertices[i], b[i]) for i in range(n)]
    bottom = [model.add_line(b[i], b[(i + 1) % n]) for i in range(n)]
    planar = [[edges[i], down[(i + 1) % n], bottom[i], down[i]] for i in range(n)] + [list(bottom)]
    surfaces = planes_fc() + [Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))]
    ids = _close(model, face, edges, planar, surfaces)
    _orient_outward(model, ids)
    return model, ids


def planes_fb4():
    """F-B with the slanted planes ``4x + z = 2`` and ``4y + z = 2``: the design's ``3x + z = 2`` puts the edge
    ``x = y = 1/6`` where no double lies (a stored vertex is not exactly on its planes), a dyadic variant does."""
    return [Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.)), Plane((0., 0., 2.), (1., 0., -4.), (0., 1., 0.)),
            Plane((0., 0., 1.5), (1., 0., 0.), (0., 1., 0.)), Plane((0., 0., 2.), (0., 1., -4.), (1., 0., 0.))]


def solid_fb(build_face, surface_fb, planes=planes_fb4):
    """F-B (dyadic variant): the conic slab ``{x, y >= (2 - z)/4, inside the conic, 1/2 <= z <= 3/2}``."""
    model, face, v, e = build_face(surface_fb(), planes())
    o1 = model.add_point(.375, .375, .5)
    o2 = model.add_point(.125, .125, 1.5)
    # vertex i starts edge i: v0 (z=1/2, y=3/8), v1 (z=1/2, x=3/8), v2 (z=3/2, x=1/8), v3 (z=3/2, y=1/8)
    l0, l1 = model.add_line(v[0], o1), model.add_line(v[1], o1)
    l2, l3 = model.add_line(v[2], o2), model.add_line(v[3], o2)
    l12 = model.add_line(o1, o2)
    planar = [[e[0], l1, l0], [e[1], l2, l12, l1], [e[2], l3, l2], [e[3], l0, l12, l3]]
    ids = _close(model, face, e, planar, planes())
    _orient_outward(model, ids)
    return model, ids


def outward_full_hull():
    """``build_b1_hull`` with every loop reversed when needed: the library builds it with inward loops."""
    from anygeometry.b1_example import build_b1_hull
    model, faces = build_b1_hull()
    ids = list(faces.values())
    flip_all(model, ids)
    return model, faces


def polygon_face(build_face, surface_fa, points):
    """A curved face of F-A whose chart loop is the polygon ``points`` ((u, v) pairs, all edges non-vertical):
    every edge is the section of the plane through the chart line. The face is written into the store (the
    library's own loop check refuses a non-simple chart loop, which these fixtures need)."""
    from anygeometry import GeometryModel
    from anygeometry.entities import Face

    def raw(self, loop, surface):
        face = self._next_id["face"]
        self._next_id["face"] = face + 1
        self._put_entity("face", Face(face, tuple(loop), surface=surface))
        return face

    n = len(points)
    planes = [Plane((points[i][0], 0., points[i][1]),
                    (points[(i + 1) % n][0] - points[i][0], 0., points[(i + 1) % n][1] - points[i][1]),
                    (0., 1., 0.)) for i in range(n)]
    original = GeometryModel.add_face_from_loop
    GeometryModel.add_face_from_loop = raw
    try:
        return build_face(surface_fa(), planes)
    finally:
        GeometryModel.add_face_from_loop = original


def hull_with_port(port_surface):
    """``build_b1_hull`` with a chosen port bow surface (to make the two bows overlap); loops left as built."""
    import anygeometry.b1_example as b1
    original = b1.b1_mirrored_bow_surface
    b1.b1_mirrored_bow_surface = lambda: port_surface
    try:
        return b1.build_b1_hull()
    finally:
        b1.b1_mirrored_bow_surface = original
