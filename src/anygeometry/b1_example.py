"""The B1 driving case: a closed half-hull of a curved bow between two frames (generality slice B1-P5b).

The bow is a rational NURBS patch of degree (2, 1) (a quarter-circle-like plan from the side to the
stem, flared towards the deck; z = 4 v), used over ``u_range = (0, 0.9)`` so the centreline plane
y = 0 misses its support (the stem itself lies in that plane, which P1 cannot section). The region
between the frame planes x = a and x = b, the bottom plane z = z0 and the deck plane z = z1, inside
the bow and on the y >= 0 side, is bounded by six faces sharing twelve edges:

- the bow piece (NURBS face) bounded by four plane x NURBS section curves, cornered at their exact
  intersections;
- the deck and the bottom (plane faces: a section curve and three centreline/frame lines);
- the two frames (plane faces: a section curve and three lines);
- the centreline face (plane y = 0, a rectangle).

``build_b1_half_hull`` returns the model and the face ids by name. Loops are oriented so that every
face's loop runs anticlockwise about its outward normal (checked by ``closed_shell`` in P5c).
"""
from __future__ import annotations

import math

from .model import GeometryModel
from .nurbs import NurbsSurface
from .support_curves import PlaneNurbsCurve
from .surfaces import Plane


def b1_bow_surface() -> NurbsSurface:
    w = math.cos(math.pi / 4)
    bottom = [(0., 6., 0.), (5.4, 6., 0.), (5.4, 0., 0.)]
    deck = [(0., 7.5, 4.), (6., 7.5, 4.), (6., 0., 4.)]
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple((b, d) for b, d in zip(bottom, deck)), ((1., 1.), (w, w), (1., 1.)), u_range=(0., .9))


def _parameter(curve, normal, offset) -> float:
    from .branch_predicates import section_roots
    roots = section_roots(curve, normal, offset, max_depth=90, refine=80)
    if len(roots) != 1:
        raise ValueError("the B1 planes must cross each section curve exactly once")
    return .5 * (roots[0].t_lo + roots[0].t_hi)


def build_b1_half_hull(frames=(1.5, 3.5), bottom: float = .5, deck: float = 3.):
    """The closed B1 half-hull between the frames ``x = frames[0], frames[1]`` and the planes
    ``z = bottom, deck``: ``(model, faces)`` with ``faces`` keyed bow, deck, bottom, aft, fore, centre."""
    a, b = frames
    surface = b1_bow_surface()
    planes = {
        "deck": (Plane((0., 0., deck), (1., 0., 0.), (0., 1., 0.)), (0, 0, 1), deck),
        "bottom": (Plane((0., 0., bottom), (1., 0., 0.), (0., 1., 0.)), (0, 0, 1), bottom),
        "aft": (Plane((a, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), a),
        "fore": (Plane((b, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), b),
    }
    curves = {key: PlaneNurbsCurve.from_plane(plane, surface, 0) for key, (plane, _n, _d) in planes.items()}
    pairs = (("deck", "aft"), ("deck", "fore"), ("bottom", "aft"), ("bottom", "fore"))
    t = {}
    for first, second in pairs:
        t[(first, second)] = _parameter(curves[first], planes[second][1], planes[second][2])
        t[(second, first)] = _parameter(curves[second], planes[first][1], planes[first][2])
    model = GeometryModel()
    corner = {frozenset(pair): model.add_point(*curves[pair[0]].evaluate(t[pair])) for pair in pairs}
    centre = {(x_key, z_key): model.add_point(x, 0., z)
              for x_key, x in (("aft", a), ("fore", b)) for z_key, z in (("bottom", bottom), ("deck", deck))}

    def section(curve, start, end):
        return model.add_curve(corner[frozenset((curve, start))], corner[frozenset((curve, end))],
                               curves[curve].subcurve(t[(curve, start)], t[(curve, end)]))

    # the four section edges of the bow piece
    s_bottom = section("bottom", "aft", "fore")
    s_fore = section("fore", "bottom", "deck")
    s_deck = section("deck", "fore", "aft")
    s_aft = section("aft", "deck", "bottom")
    # lines: from the corners to the centreline, and along the centreline
    to_centre = {(x_key, z_key): model.add_line(corner[frozenset((z_key, x_key))], centre[(x_key, z_key)])
                 for x_key in ("aft", "fore") for z_key in ("bottom", "deck")}
    along = {z_key: model.add_line(centre[("aft", z_key)], centre[("fore", z_key)]) for z_key in ("bottom", "deck")}
    up = {x_key: model.add_line(centre[(x_key, "bottom")], centre[(x_key, "deck")]) for x_key in ("aft", "fore")}

    from .entities import OrientedEdge
    # the bow's loop reversed so that every shared edge runs opposite ways in its two faces (closed_shell)
    bow_loop = model.order_loop([s_bottom, s_fore, s_deck, s_aft])
    bow_loop = tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(bow_loop))
    faces = {
        "bow": model.add_face_from_loop(bow_loop, surface=surface),
        "deck": model.add_face([s_deck, to_centre[("aft", "deck")], along["deck"], to_centre[("fore", "deck")]],
                               surface=planes["deck"][0]),
        "bottom": model.add_face([s_bottom, to_centre[("fore", "bottom")], along["bottom"],
                                  to_centre[("aft", "bottom")]], surface=planes["bottom"][0]),
        "aft": model.add_face([s_aft, to_centre[("aft", "bottom")], up["aft"], to_centre[("aft", "deck")]],
                              surface=planes["aft"][0]),
        "fore": model.add_face([s_fore, to_centre[("fore", "deck")], up["fore"], to_centre[("fore", "bottom")]],
                               surface=planes["fore"][0]),
        "centre": model.add_face([along["bottom"], up["fore"], along["deck"], up["aft"]],
                                 surface=Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.))),
    }
    return model, faces


def b1_mirrored_bow_surface() -> NurbsSurface:
    """The port bow: the starboard bow mirrored in y = 0 (controls with y negated)."""
    surface = b1_bow_surface()
    controls = tuple(tuple((x, -y, z) for x, y, z in row) for row in surface.controls)
    return NurbsSurface(surface.degree_u, surface.degree_v, surface.knots_u, surface.knots_v, controls,
                        surface.weights, surface.u_range, surface.v_range)


def _consistent_loops(model, edge_lists):
    """Loops for faces given as edge lists, oriented so that every edge shared by two faces runs opposite
    ways in them (propagated from the first face; a closed shell then has one orientation throughout)."""
    from .entities import OrientedEdge
    loops = [list(model.order_loop(edges)) for edges in edge_lists]
    done, pending = {0}, [0]
    while pending:
        index = pending.pop()
        directions = {item.edge: item.forward for item in loops[index]}
        for other, loop in enumerate(loops):
            if other in done:
                continue
            common = [item for item in loop if item.edge in directions]
            if not common:
                continue
            if common[0].forward == directions[common[0].edge]:
                loops[other] = [OrientedEdge(item.edge, not item.forward) for item in reversed(loop)]
            done.add(other)
            pending.append(other)
    return [tuple(loop) for loop in loops]


def _loop_area_z(model, loop) -> float:
    """Twice the signed xy area of the polygon through the loop's start vertices (positive: anticlockwise
    seen from +z)."""
    points = []
    for item in loop:
        edge = model.edges[item.edge]
        points.append(model.vertices[edge.start if item.forward else edge.end].position)
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def build_b1_hull(frames=(1.5, 3.5), bottom: float = .5, deck: float = 3.):
    """The closed full B1 hull between the frames and the planes ``z = bottom, deck`` (H1-F3): the
    starboard bow and its mirror image, deck/bottom/frame faces spanning port to starboard (each bounded
    by two section curves and two straight lines). ``(model, faces)`` keyed bow, bow_port, deck,
    bottom, aft, fore."""
    a, b = frames
    surfaces = {"s": b1_bow_surface(), "p": b1_mirrored_bow_surface()}
    planes = {
        "deck": (Plane((0., 0., deck), (1., 0., 0.), (0., 1., 0.)), (0, 0, 1), deck),
        "bottom": (Plane((0., 0., bottom), (1., 0., 0.), (0., 1., 0.)), (0, 0, 1), bottom),
        "aft": (Plane((a, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), a),
        "fore": (Plane((b, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), b),
    }
    pairs = (("deck", "aft"), ("deck", "fore"), ("bottom", "aft"), ("bottom", "fore"))
    model = GeometryModel()
    corner, sections = {}, {}
    for side, surface in surfaces.items():
        curves = {key: PlaneNurbsCurve.from_plane(plane, surface, 0) for key, (plane, _n, _d) in planes.items()}
        t = {}
        for first, second in pairs:
            t[(first, second)] = _parameter(curves[first], planes[second][1], planes[second][2])
            t[(second, first)] = _parameter(curves[second], planes[first][1], planes[first][2])
        for pair in pairs:
            corner[(side, frozenset(pair))] = model.add_point(*curves[pair[0]].evaluate(t[pair]))
        for key, (start, end) in (("bottom", ("aft", "fore")), ("fore", ("bottom", "deck")),
                                  ("deck", ("fore", "aft")), ("aft", ("deck", "bottom"))):
            sections[(side, key)] = model.add_curve(corner[(side, frozenset((key, start)))],
                                                    corner[(side, frozenset((key, end)))],
                                                    curves[key].subcurve(t[(key, start)], t[(key, end)]))
    across = {pair: model.add_line(corner[("s", frozenset(pair))], corner[("p", frozenset(pair))]) for pair in pairs}
    names = ["bow", "bow_port", "deck", "bottom", "aft", "fore"]
    edge_lists = [
        [sections[("s", k)] for k in ("bottom", "fore", "deck", "aft")],
        [sections[("p", k)] for k in ("bottom", "fore", "deck", "aft")],
        [sections[("s", "deck")], across[("deck", "aft")], sections[("p", "deck")], across[("deck", "fore")]],
        [sections[("s", "bottom")], across[("bottom", "aft")], sections[("p", "bottom")], across[("bottom", "fore")]],
        [sections[("s", "aft")], across[("deck", "aft")], sections[("p", "aft")], across[("bottom", "aft")]],
        [sections[("s", "fore")], across[("deck", "fore")], sections[("p", "fore")], across[("bottom", "fore")]],
    ]
    from .entities import OrientedEdge
    loops = _consistent_loops(model, edge_lists)
    if _loop_area_z(model, loops[2]) < 0.:
        # the deck's outward normal is +z: its loop must run anticlockwise seen from above; if not, the whole
        # (consistently oriented) shell is inward and is reversed so the hull is an OUTWARD closed hull
        loops = [tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(loop)) for loop in loops]
    supports = [surfaces["s"], surfaces["p"]] + [planes[key][0] for key in ("deck", "bottom", "aft", "fore")]
    faces = {name: model.add_face_from_loop(loop, surface=support)
             for name, loop, support in zip(names, loops, supports)}
    return model, faces
