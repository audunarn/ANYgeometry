from anygeometry import GeometryModel
from anygeometry.entities import OrientedEdge
from anygeometry.surfaces import RuledSurface
from anygeometry.tensor_interfaces import tensor_boundary_contacts


def joined_faces(far_x):
    model = GeometryModel()
    verts, edges, faces = {}, {}, []
    # A rotated network exercises the non-axis-aligned separating plane.
    def point(x, y, z):
        return (x-y, x+y, z)
    for x0, x1, h0, h1 in ((0, 1, 0, .125), (1, far_x, .125, -.125)):
        a, b, c, d = (point(x0,0,0), point(x1,0,0), point(x1,1,h1), point(x0,1,h0))
        loop = []
        for start, end in zip((a,b,c,d), (b,c,d,a)):
            for xyz in (start,end):
                if xyz not in verts:
                    verts[xyz] = model.add_point(*xyz)
            if (end,start) in edges:
                loop.append(OrientedEdge(edges[end,start],False))
            else:
                edge = edges[start,end] = model.add_line(verts[start],verts[end])
                loop.append(OrientedEdge(edge,True))
        faces.append(model.add_face_from_loop(tuple(loop), corners=(0,1,2,3),
                                              surface=RuledSurface((a,b),(d,c))))
    return model, faces, edges[point(1,0,0),point(1,1,.125)]


def test_exact_shared_boundary_and_separate_open_material():
    model, faces, edge = joined_faces(2)
    assert tensor_boundary_contacts(model) == {tuple(faces): edge}


def test_folded_back_chart_cannot_claim_a_separating_rim():
    model, _, _ = joined_faces(.25)
    # Both surfaces occupy the same side of their shared edge. The returned
    # proof must not omit the ordinary intersection query.
    assert tensor_boundary_contacts(model) is None


def test_bent_seam_with_oblique_support_plane_and_folded_control():
    from fractions import Fraction
    from anygeometry.tensor_interfaces import _opposite_open_halfspaces

    # Every rim control lies on x + y + z = 0. No coordinate plane contains it.
    rim = [(0, 0, 0), (1, -1, 0), (1, 0, -1)]
    above = [(3, -2, 0), (0, 2, 0)]
    below = [(-2, 1, 0), (0, 0, -2)]
    tolerance = Fraction(1, 1000)
    assert _opposite_open_halfspaces((rim, above), (rim[::-1], below), tolerance)
    assert not _opposite_open_halfspaces((rim, above), (rim[::-1], below + [(2, 0, 0)]), tolerance)
