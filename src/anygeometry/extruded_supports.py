"""Exact supports of extruded faces: recognition from topology.

``GeometryModel.extrude`` leaves a spline or oblique-arc face as a topology-backed Coons patch: four
edges, a profile curve, its translate and two parallel straight connectors. That structure *is* the exact
translational surface ``c(t) + s d``; :func:`extruded_support` recovers it as an
:class:`~anygeometry.surfaces.ExtrudedSurface` (a planar Bezier or elliptic directrix) without changing
what the model stores. A face that already carries an ``ExtrudedSurface`` returns it unchanged.
"""
from __future__ import annotations

import numpy as np

from .arrangement_geometry import BezierPath, LinePath, freeze_edge
from .errors import GeometryError
from .exact_curves import EllipticArc
from .extrusions import BezierDirectrix, EllipseDirectrix
from .surfaces import CoonsSurface, ExtrudedSurface


def _oriented(model, use):
    curve = freeze_edge(model, use.edge)
    return curve if use.forward else curve.subcurve(1., 0.)


def _directrix(curve):
    if isinstance(curve, BezierPath):
        return BezierDirectrix(curve.controls)
    if isinstance(curve, EllipticArc):
        return EllipseDirectrix(tuple(curve.center), tuple(curve.u_vector), tuple(curve.v_vector),
                                float(curve.start_angle), float(curve.sweep_angle))
    return None


def extruded_support(model, face_id):
    """The :class:`ExtrudedSurface` of a face made by sweeping a planar profile, or ``None``."""
    face = model.faces[face_id]
    if isinstance(face.surface, ExtrudedSurface):
        return face.surface
    if face.holes or len(face.loop) != 4:
        return None
    if face.surface is not None and not isinstance(face.surface, CoonsSurface):
        return None
    try:
        paths = [_oriented(model, use) for use in face.loop]
    except GeometryError:
        return None
    for rotation in range(4):
        bottom, right, top, left = (paths[(rotation + k) % 4] for k in range(4))
        if (isinstance(right, LinePath) and isinstance(left, LinePath)
                and not isinstance(bottom, LinePath) and not isinstance(top, LinePath)):
            break
    else:
        return None
    if type(bottom) is not type(top) or (isinstance(bottom, BezierPath) and len(bottom.controls) != len(top.controls)):
        return None
    vector = np.asarray(right.end) - np.asarray(right.start)            # bottom end -> top end
    profile = np.asarray(bottom.evaluate(np.linspace(0., 1., 9)))
    tolerance = model.tolerance.effective_length(max(float(np.linalg.norm(vector)),
                                                     float(np.linalg.norm(np.ptp(profile, axis=0)))))
    if float(np.linalg.norm(np.asarray(left.end) - np.asarray(left.start) + vector)) > tolerance:
        return None                                                      # the connectors are not parallel and equal
    along = np.linspace(0., 1., 17)
    translated = np.asarray(bottom.evaluate(along)) + vector
    if float(np.max(np.linalg.norm(translated - np.asarray(top.subcurve(1., 0.).evaluate(along)), axis=1))) > tolerance:
        return None                                                      # the top is not the bottom moved by the vector
    try:
        directrix = _directrix(bottom)
        return None if directrix is None else ExtrudedSurface(directrix, vector)
    except GeometryError:
        return None                                                      # non-planar profile or a vector in its plane
