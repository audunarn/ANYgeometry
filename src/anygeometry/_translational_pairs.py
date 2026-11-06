"""Fail-closed rational separation for topology Coons NURBS sweeps.

No new surface support is declared. Unrecognised definitions return None and
continue through the existing qualified intersection engine.
"""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np

from .curves import Straight
from .nurbs import NurbsCurve
from .surfaces import CoonsSurface


def _q(point):
    return tuple(Q(float(value)) for value in point)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _oriented(model, use):
    curve = model.edges[use.edge].curve
    return curve if use.forward else replace(curve, domain=curve.domain[::-1])


def _sweep(model, face_id):
    face = model.faces[face_id]
    if (not isinstance(face.surface, CoonsSurface) or face.surface.has_boundaries
            or face.parameterization is not None or face.holes or len(face.corners) != 4):
        return None
    sides = face.sides()
    if any(len(side) != 1 for side in sides):
        return None
    loop = tuple(side[0] for side in sides)
    for index in range(4):
        bottom, right, top, left = (loop[(index+k) % 4] for k in range(4))
        if (isinstance(model.edges[bottom.edge].curve, NurbsCurve)
                and isinstance(model.edges[top.edge].curve, NurbsCurve)
                and isinstance(model.edges[right.edge].curve, Straight)
                and isinstance(model.edges[left.edge].curve, Straight)):
            break
    else:
        return None
    profile = _oriented(model, bottom)
    opposite = _oriented(model, top)
    opposite = replace(opposite, domain=opposite.domain[::-1])
    if (profile.dimension != 3 or profile.degree != opposite.degree
            or profile.knots != opposite.knots or profile.domain != opposite.domain
            or profile.weights != opposite.weights or len(profile.controls) != len(opposite.controls)):
        return None
    start = model.vertex_position(model.oriented_start_vertex(bottom))
    end = model.vertex_position(model.oriented_end_vertex(bottom))
    top_start = model.vertex_position(model.oriented_end_vertex(top))
    top_end = model.vertex_position(model.oriented_start_vertex(top))
    # Coons contributes an affine endpoint correction if connector vertices
    # differ from its evaluated profile corners. Certify only zero correction.
    if not (np.array_equal(profile.evaluate(0.), start)
            and np.array_equal(profile.evaluate(1.), end)
            and np.array_equal(opposite.evaluate(0.), top_start)
            and np.array_equal(opposite.evaluate(1.), top_end)):
        return None
    vector = _sub(_q(top_start), _q(start))
    if not any(vector) or _sub(_q(top_end), _q(end)) != vector:
        return None
    if any(_sub(_q(b), _q(a)) != vector for a, b in zip(profile.controls, opposite.controls)):
        return None
    if next(value for value in vector if value) < 0:
        profile = opposite
        vector = tuple(-value for value in vector)
    return profile, vector, (left.edge, right.edge)


def separated_sweep_connectors(model, first_id, second_id):
    """Return shared connector IDs when exact profiles only meet there.

    Empty tuple certifies disjoint sweeps; None means no new classification.
    A common rational profile plane transverse to the common sweep makes the
    extrusion injective in the plane coordinate. Positive rational hulls then
    prove profile separation, rather than tessellated surface separation.
    """
    a, b = _sweep(model, first_id), _sweep(model, second_id)
    if a is None or b is None or a[1] != b[1]:
        return None
    first, vector, first_connectors = a
    second, _, second_connectors = b
    controls = [_q(point) for curve in (first, second) for point in curve.controls]
    anchor = controls[0]
    normal = None
    for point in controls[1:]:
        direction = _sub(point, anchor)
        if any(direction):
            for other in controls[1:]:
                candidate = _cross(direction, _sub(other, anchor))
                if any(candidate):
                    normal = candidate
                    break
        if normal is not None:
            break
    if normal is None or not _dot(normal, vector):
        return None
    if any(_dot(normal, _sub(point, anchor)) for point in controls):
        return None
    contacts = [(float(i), float(j)) for i in (0, 1) for j in (0, 1)
                if first.exact_point(i) == second.exact_point(j)]
    matching = {first_connectors[int(i)] for i, j in contacts
                if first_connectors[int(i)] == second_connectors[int(j)]}
    if (len(matching) != len(contacts)
            or matching != set(first_connectors) & set(second_connectors)):
        return None
    from .nurbs_audit import _certify_pieces, _BUDGET
    if _certify_pieces(first, (0., 1.), second, (0., 1.), contacts, _BUDGET,
                       _hull=lambda curve, lo, hi: curve.exact_hull(lo, hi), _qualification=True) is not True:
        return None
    return tuple(sorted(matching))


def topology_boundary_uv(model, face_id, edge_id, parameter):
    """Exact boundary chart parameter for unsegmented topology Coons sides."""
    face = model.faces[face_id]
    if (not isinstance(face.surface, CoonsSurface) or face.surface.has_boundaries
            or face.parameterization is not None or len(face.corners) != 4):
        return None
    for side, chain in enumerate(face.sides()):
        if len(chain) == 1 and chain[0].edge == edge_id:
            value = float(parameter) if chain[0].forward else 1.-float(parameter)
            return ((value, 0.), (1., value), (1.-value, 1.), (0., 1.-value))[side]
    return None


def query_translational_profile_interfaces(model, face_ids=None, *, expected_revision=None,
                                         cancellation_check=None):
    """Fresh complete pair descriptions for canonical raw rational translations.

    None means unsupported, never permission to omit an interior contact. The
    immutable result describes this exact source read; it is not persisted
    preparation authority. Connector offsets are nonzero rational upper bounds
    against the original absolute length tolerance, not zero residual claims.
    """
    from .errors import GeometryError
    from .structural import FrozenMetadata
    from .trimmed_charts import _query_start, _query_end
    from .nurbs_sweep_chart import source_sweep_chart_model
    from .curves import sample_straight

    revision, checksum = _query_start(model, expected_revision, cancellation_check)
    def check():
        _query_end(model, revision, checksum, cancellation_check)
    check()
    ids = tuple(sorted(model.faces if face_ids is None else face_ids))
    if (len(ids)<2 or len(set(ids))!=len(ids)
            or any(type(face) is not int or face not in model.faces for face in ids)):
        raise GeometryError('translational interfaces require at least two unique source face IDs')
    faces=[]
    for face_id in ids:
        check()
        face=model.faces[face_id]
        if (type(face.surface) is not CoonsSurface or face.surface.has_boundaries
                or face.parameterization is not None or face.holes
                or any(len(side)!=1 for side in face.sides())
                or any(type(model.edges[use.edge].curve) not in (Straight,NurbsCurve) for use in face.loop)):
            check();return None
        found=_sweep(model,face_id)
        if found is None:check();return None
        profile,vector,connectors=found
        # Bind the ACTUAL source frozen corners (sides 0/2), including rotations.
        for side in (face.sides()[0],face.sides()[2]):
            use=side[0];edge=model.edges[use.edge]
            for at in (0.,1.):
                t=at if use.forward else 1.-at
                start=model.vertices[edge.start].position;end=model.vertices[edge.end].position
                point=(sample_straight(start,end,np.asarray([t]))[0] if type(edge.curve) is Straight
                       else edge.curve.evaluate(t))
                vertex=start if t==0. else end
                if not np.array_equal(point,vertex):check();return None
        bounds=[]
        for u,edge_id in enumerate(connectors):
            edge=model.edges[edge_id]
            a,b=(_q(model.vertices[vertex].position) for vertex in (edge.start,edge.end))
            if _sub(b,a)==vector:base,tip=a,b
            elif _sub(a,b)==vector:base,tip=b,a
            else:check();return None
            exact=profile.exact_point(u)
            offset=_sub(base,exact)
            if _sub(tip,tuple(x+d for x,d in zip(exact,vector)))!=offset:check();return None
            bound=sum(map(abs,offset))
            if bound>Q(float(model.tolerance.length)):
                raise GeometryError('translational connector offset exceeds original absolute length tolerance')
            bounds.append(dict(edge=edge_id,offset=tuple((x.numerator,x.denominator) for x in offset),
                               upper_L1=(bound.numerator,bound.denominator)))
        # Existing full-boundary correspondence guards, without installing a chart.
        view=source_sweep_chart_model(model,face_id)
        if view is None:check();return None
        for use in face.loop:
            check();view.chart_edge(model.edges[use.edge])
        faces.append(dict(face_id=face_id,connectors=tuple(bounds)))
    pairs=[]
    for index,first in enumerate(ids):
        for second in ids[index+1:]:
            check()
            shared=separated_sweep_connectors(model,first,second)
            if shared is None:check();return None
            actual={use.edge for use in model.faces[first].loop}&{use.edge for use in model.faces[second].loop}
            if set(shared)!=actual:check();return None
            pairs.append(dict(faces=(first,second),shared_edges=shared,relation='shared-boundary-only' if shared else 'disjoint'))
    check()
    return FrozenMetadata(dict(schema='translational-profile-interfaces-v1',
        source_model_id=str(model.model_id),source_revision=revision,source_checksum=checksum,
        face_ids=ids,faces=tuple(faces),pairs=tuple(pairs)))
