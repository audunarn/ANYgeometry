"""The real-bow probe geometry (programme step 3, owner decision Q2: two patches meeting at the stem).

A hard-chine barge bow for the 11 m x 2.5 m midship of ANYfem's ``floating_barge_large``: the
bow runs from the collision bulkhead at X = 15 to the stem near X = 20. Each side is one
bidegree-(3, 3) polynomial ``NurbsSurface``: u from the stem (u = 0, in the centreline plane
y = 0) to the aft boundary (u = 1, in the plane X = 15), v from the keel (v = 0, in the
centreline plane) over the chine (v = 1/2, a C0 crease from a triple interior knot) to above
the deck (v = 1, z = 3; the deck at z = 2.5 is a plane x NURBS section). At X = 15 the aft
boundary is exactly the midship section: the flat bottom z = 0 from the keel to the chine at
y = 5.5 and the flat side y = 5.5 (variant G0: a knuckle at X = 15; the patch is not tangent
to the midship). The port patch is the exact mirror of the starboard patch in y = 0, so the
stem and keel iso-curves of the two patches are the same exact curves.
"""
from __future__ import annotations

from .nurbs import NurbsSurface

BOW_KNOTS_U = (0., 0., 0., 0., 1., 1., 1., 1.)
BOW_KNOTS_V = (0., 0., 0., 0., .5, .5, .5, 1., 1., 1., 1.)       # the chine: a triple knot at v = 1/2

# controls[i][j]: i along u (stem -> X = 15), j along v (keel -> chine (j = 3) -> above the deck)
_STARBOARD = (
    ((18.75, 0., .75), (18.875, 0., .875), (19., 0., 1.), (19.25, 0., 1.25),          # stem, y = 0
     (19.5, 0., 1.75), (19.75, 0., 2.375), (20., 0., 3.)),
    ((18.25, 0., .5), (18.25, 1., .5), (18.25, 2., .5), (18.5, 2.75, .625),
     (18.75, 3., 1.25), (19., 3.25, 2.), (19.25, 3.25, 3.)),
    ((16.75, 0., .25), (16.75, 1.75, .25), (16.75, 3.5, .25), (16.75, 4.75, .25),
     (17., 5., 1.), (17.25, 5.25, 2.), (17.5, 5.25, 3.)),
    ((15., 0., 0.), (15., 2., 0.), (15., 4., 0.), (15., 5.5, 0.),                     # X = 15: midship
     (15., 5.5, 1.), (15., 5.5, 2.), (15., 5.5, 3.)),
)


def bow_starboard_surface() -> NurbsSurface:
    """The starboard bow patch (y >= 0)."""
    return NurbsSurface(3, 3, BOW_KNOTS_U, BOW_KNOTS_V, _STARBOARD)


def bow_port_surface() -> NurbsSurface:
    """The port bow patch: the starboard patch mirrored in y = 0, exactly (controls with y negated;
    a zero stays +0.0, so the shared stem and keel controls are bit-identical)."""
    controls = tuple(tuple((x, -y + 0.0, z) for x, y, z in row) for row in _STARBOARD)
    return NurbsSurface(3, 3, BOW_KNOTS_U, BOW_KNOTS_V, controls)


def bow_two_patch() -> tuple[NurbsSurface, NurbsSurface]:
    """``(starboard, port)`` bow patches meeting at the stem (and the keel) in the plane y = 0."""
    return bow_starboard_surface(), bow_port_surface()
