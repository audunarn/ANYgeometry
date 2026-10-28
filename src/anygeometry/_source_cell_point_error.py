"""Certified LOWER error witnesses only; never geometric admission."""
from fractions import Fraction as Q


def fan_point_error_lower_squared(support, boundary, centre_uv, xyz_side, centre_xyz, *,
                                  check, quad_uv=None, quad_xyz=None, quad_side=None):
    """Maximum lower bound at mid-edge/centre, or None for unknown family.

    A value below threshold says nothing about the whole cell. The live owner
    caller must validate all precision/transport/binding gates before using it
    to emit a resolution signal. This helper has no receipt authority.
    """
    from .curvilinear_reference_maps import _source_box, _source_quad_fan, _quad_fan
    from .regular_source_reference import SourceSide, RationalSourceSide
    from .plane_source_reference import PlaneSide
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    from .plane_nurbs import charge_box
    from ._point_fan_values import canonical_point_values
    if type(support) not in (NurbsSurface, Plane) or type(boundary) not in (SourceSide, PlaneSide, RationalSourceSide):
        return None
    maximum = Q(0)
    for s, t in ((Q(0), Q(1, 2)), (Q(1), Q(1, 2))):
        check()
        charge_box()
        if quad_uv is not None and quad_xyz is not None:
            values = canonical_point_values(quad_uv, boundary, quad_side, s, t, quad_xyz)
            if values is None:
                uv = _source_quad_fan(quad_uv, boundary, quad_side, (s, s), (t, t))[0]
                target = _quad_fan(quad_xyz, quad_side, (s, s), (t, t))[0]
            else:
                uv, target = values
        elif quad_uv is None and quad_xyz is None:
            edge = boundary.box(t, t)
            uv = tuple(((1-s)*Q(a)+s*Q(c), (1-s)*Q(b)+s*Q(c)) for (a, b), c in zip(edge, centre_uv))
            target = tuple(((1-s)*((1-t)*Q(a)+t*Q(b))+s*Q(c),)*2
                           for a, b, c in zip(*xyz_side, centre_xyz))
        else:
            return None
        if type(support) is NurbsSurface and not all(0 <= a <= b <= 1 for a, b in uv):
            check()
            continue  # no source-image witness outside the registered chart
        world = _source_box(support, uv)
        lower = sum(max(Q(a)-Q(d), Q(c)-Q(b), Q(0))**2
                    for (a, b), (c, d) in zip(world, target))
        check()
        maximum = max(maximum, lower)
    return maximum
