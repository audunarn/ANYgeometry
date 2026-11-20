"""Codex a1 H1 (polygon, 100 m) and H2 (curved thin overlap) with slope s: s=0 exactly coplanar, s=5e-10 in band."""
import sys
sys.path.insert(0, sys.argv[1])
s = float(sys.argv[2])
from anygeometry import GeometryModel, GeometryError
from anygeometry.entities import OrientedEdge as OE
from anygeometry.surfaces import Plane
from anygeometry.batch_intersections import plan_intersections, apply_intersections


def run(name, m, ids):
    try:
        plan = plan_intersections(m, [m.handle('face', k) for k in ids], policy='connect')
        apply_intersections(m, plan, policy='connect')
        print(name, 'PLANNED+APPLIED', m.validate_topology())
    except GeometryError as ex:
        print(name, 'refused', str(ex)[:110])


def loop(m, hinge, a, b, points, reverse, arc_pair=None, via=None):
    chain = [OE(hinge, x == a) if {x, y} == {a, b} else
             OE(m.add_arc(x, via, y) if arc_pair and {x, y} == arc_pair else m.add_line(x, y), True)
             for x, y in zip(points, points[1:]+points[:1])]
    return tuple(OE(x.edge, not x.forward) for x in chain[::-1]) if reverse else tuple(chain)


def polygon(span, dy, dx, reverse):
    m = GeometryModel()
    a, b, c, d, e, f, g, h, i = m.add_points(((0, 0, 0), (0, dy, 0), (0, span, 0), (1, span, 0), (1, 0, 0),
                                              (dx, dy, dx*s), (.5, -span, .5*s), (-.5, -span, -.5*s), (-.5, 0, -.5*s)))
    hinge = m.add_line(a, b)
    ids = (m.add_face_from_loop(loop(m, hinge, a, b, [a, b, c, d, e], reverse), surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))),
           m.add_face_from_loop(loop(m, hinge, a, b, [a, b, f, g, h, i], False), surface=Plane((0, 0, 0), (1, 0, s), (0, 1, 0))))
    return m, ids


def curved(dx, reverse_a, reverse_b):
    m = GeometryModel()
    a, b, c, d, e, f, g, h, i, via = m.add_points(((0, 0, 0), (0, .3, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (dx, .3, dx*s),
                                                   (dx, -1, dx*s), (-1, -1, -s), (-1, 0, -s), (1.12, .5, 0)))
    hinge = m.add_line(a, b)
    ids = (m.add_face_from_loop(loop(m, hinge, a, b, [a, b, c, d, e], reverse_a, {d, e}, via), surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))),
           m.add_face_from_loop(loop(m, hinge, a, b, [a, b, f, g, h, i], reverse_b), surface=Plane((0, 0, 0), (1, 0, s), (0, 1, 0))))
    return m, ids


for span, dy, dx in ((100., 1e-6, 1e-5), (999., 1e-5, 1e-4)):
    for reverse in (False, True):
        run(f'polygon span={span:g} dy={dy:g} dx={dx:g} rev={reverse}', *polygon(span, dy, dx, reverse))
for dx in (25e-9, 40e-9):
    for ra, rb in ((0, 0), (1, 0)):
        run(f'curved dx={dx:g} rev=({ra},{rb})', *curved(dx, ra, rb))
