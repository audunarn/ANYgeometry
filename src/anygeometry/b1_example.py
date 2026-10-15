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
