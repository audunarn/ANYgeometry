"""Exact boundary-only contacts retained for bilinear Coons quadrilaterals.

This is a compatibility predicate, not a general Coons intersection algorithm.
A bilinear patch is a convex combination of its four corners. A plane whose
signed corner values have one sign can therefore meet it only on zero-valued
boundary edges (or isolated corners). Interior crossings remain unsupported.
"""
import numpy as np

from .arrangement_geometry import LinePath
from .curves import Straight
from .errors import GeometryError
from .material_arrangement import ArrangementPath, _clip
from .surfaces import CoonsSurface, Plane


def bilinear_boundary(model, face_id):
    face = model.faces[face_id]
    if (not isinstance(face.surface, CoonsSurface) or face.holes
            or len(face.loop) != 4
            or any(not isinstance(model.edges[use.edge].curve, Straight) for use in face.loop)):
        raise GeometryError(f"face {face_id} has no qualified analytic or bilinear support")
    if face.surface.has_boundaries:
        for values in (face.surface.bottom, face.surface.right, face.surface.top, face.surface.left):
            expected = np.linspace(values[0], values[-1], len(values))
            # Exact sampled descriptors must define straight boundaries. Do
            # not infer a bilinear surface from approximately straight samples.
            if not np.array_equal(values, expected):
                raise GeometryError(f"face {face_id} has non-bilinear Coons boundaries")
    points = tuple(tuple(model.vertex_position(model.oriented_start_vertex(use))) for use in face.loop)
    return points


def plane_boundary_contacts(model, face_id, corners, domain, tolerance, check):
    if not isinstance(domain.support, Plane):
        raise GeometryError("bilinear boundary contact requires a plane support")
    values = (np.asarray(corners)-domain.support.origin) @ domain.support.normal
    # Values inside the existing geometric tolerance are boundary witnesses.
    # All other corners must lie strictly on the same side; convexity then
    # excludes an interior crossing, independently of display sampling.
    zero = np.abs(values) <= tolerance
    if np.any(values > tolerance) and np.any(values < -tolerance):
        raise GeometryError(f"face {face_id} has an unsupported interior Coons intersection")
    if np.all(zero):
        raise GeometryError(f"face {face_id} is not qualified for a coplanar Coons contact")
    paths = []
    face = model.faces[face_id]
    for index, use in enumerate(face.loop):
        if zero[index] and zero[(index+1)%4]:
            curve = LinePath(corners[index], corners[(index+1)%4])
            paths.extend(ArrangementPath(part, use.edge, (domain.face_id, face_id))
                         for part in _clip(domain, curve, tolerance, check))
    return tuple(paths)
