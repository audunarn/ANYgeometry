"""Open main defect reproducer (docs/coplanar-edge-joined.md): exactly coplanar folds at 1e7 m plan silently.

Usage: python docs/coplanar-edge-joined-probes/scale_fold_base.py <path to ANYgeometry src>
"""
import sys; sys.path.insert(0, sys.argv[1])
from dataclasses import replace
from anygeometry import GeometryModel, GeometryError
from anygeometry.batch_intersections import plan_intersections, apply_intersections
from anygeometry.surfaces import Plane
def fold(length, depth, width=1.):
    m = GeometryModel()
    a = m.add_points(((0., 0., 0.), (length, 0., 0.), (length, width, 0.), (0., width, 0.)))
    hinge = m.add_line(a[1], a[2])
    fa = m.add_face((m.add_line(a[0], a[1]), hinge, m.add_line(a[2], a[3]), m.add_line(a[3], a[0])))
    b = m.add_points(((length-depth, 0., 0.), (length-depth, width, 0.)))
    fb = m.add_face((m.add_line(a[1], b[0]), m.add_line(b[0], b[1]), m.add_line(b[1], a[2]), hinge))
    for f in (fa, fb):
        m._put_entity('face', replace(m.faces[f], surface=Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))))
    return m, (m.handle('face', fa), m.handle('face', fb))
for length, depth in ((1., .5), (1e7, .5), (1e7, 1e-4), (1e3, 1e-4), (1e3, 1e-6)):
    m, f = fold(length, depth)
    try:
        p = plan_intersections(m, f, policy='connect'); apply_intersections(m, p, policy='connect')
        print(f'exactly coplanar fold L={length:g} depth={depth:g}: PLANNED+APPLIED (silent)')
    except GeometryError as e:
        print(f'exactly coplanar fold L={length:g} depth={depth:g}: REFUSED', str(e)[:90])
