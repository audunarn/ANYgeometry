"""Whole original-UV triangles in one authenticated literal current child.

Separate from original-domain coverage: a cell crossing a descendant boundary
must not inherit that child's load or reference scope. Unsupported mappings
refuse. No complete partition, discretization or publication claim is supplied.
"""
from fractions import Fraction as F
from numbers import Integral
from copy import deepcopy

from .arrangement_geometry import BezierPath, LinePath, freeze_edge
from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
from .authored_domain_coverage import _original_domain, _coons_extrusion
from .errors import GeometryError
from .entities import EntityRef
from .identity import EntityHandle
from .material_arrangement import ArrangementPath, MaterialDomain
from .material_cell_coverage import _triangle_rows, _validate_domain_triangles, _frame, _dot, _cross
from .surfaces import CoonsSurface, ExtrudedSurface, Plane


def _exact(row):
    return tuple(F(float(value)) for value in row)


def _support_correspondence(original, current, triangles, check=lambda: None):
    """Prove same plane/carrier and finite child ranges, with no fitted map."""
    _frame(current)
    if type(original) is Plane and type(current) is Plane:
        normal = _cross(_exact(original.u_vector), _exact(original.v_vector))
        delta = tuple(b-a for a, b in zip(_exact(original.origin), _exact(current.origin)))
        if (any(_dot(normal, row) for row in (delta, _exact(current.u_vector), _exact(current.v_vector)))):
            raise GeometryError('authored child coverage: literal plane is not exactly the original support')
        return
    if type(original) is not ExtrudedSurface or type(current) is not ExtrudedSurface:
        raise GeometryError('authored child coverage: unsupported literal support correspondence')
    # _frame also restricts both directrices to injective qualified Bezier form.
    if (tuple(map(_exact, original.directrix.controls)) != tuple(map(_exact, current.directrix.controls)) or
            _exact(original.vector) != _exact(current.vector)):
        raise GeometryError('authored child coverage: literal extrusion carrier differs from original')
    for axis, source_range, target_range in (
            (0, original.u_range, current.u_range), (1, original.v_range, current.v_range)):
        start, end = map(F, map(float, source_range))
        lower, upper = sorted(map(F, map(float, target_range)))
        for triangle in triangles:
            check()
            for row in triangle:
                value = start + F(float(row[axis]))*(end-start)
                if not lower <= value <= upper:
                    raise GeometryError('authored child coverage: triangle leaves literal support range')


def _literal_child_domain(model, identifier):
    """Detached literal inputs; caller authenticates before/after capture."""
    face = deepcopy(model.faces[identifier])
    boundaries = []
    for loop in (face.loop, *face.holes):
        paths = []
        for use in loop:
            curve = deepcopy(freeze_edge(model, use.edge))
            if not use.forward:
                # Reverse stored coefficients, not evaluated float endpoints.
                if type(curve) is LinePath:
                    curve = LinePath(curve.end, curve.start)
                elif type(curve) is BezierPath:
                    curve = BezierPath(curve.controls[::-1])
                else:
                    curve = curve.subcurve(1., 0.)
            paths.append(ArrangementPath(curve, use.edge, decomposition=
                'intersection_decomposition_seam' in model.tags_for(EntityRef('edge', use.edge))))
        boundaries.append(tuple(paths))
    return face, MaterialDomain(identifier, face.surface, tuple(boundaries))


def validate_prepared_authored_face_child_triangles(model, correspondence, current_face,
                                                    triangles_uv, *, cancellation_check=None):
    """Prove each CLOSED original-UV triangle lies wholly in a named child.

    The named face must be an active authenticated descendant of this original
    face. Exact coplanar affine supports and qualified same-carrier Bezier
    extrusions are supported, including restricted parameter ranges. Every
    actual child outer/hole trim is checked in the original chart. Explicit
    child parameterizations, unproved support changes and unsupported trims
    refuse. Original material containment is checked separately too.

    Success proves coverage by this child, not uniqueness against other children,
    equality of the full fragment partition, triangulation coverage, required
    station/node retention or mesh quality/publication. Consumers must retain
    those separate gates and revalidate with the same cells before publication.
    """
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    if isinstance(current_face, EntityHandle):
        if current_face.model_id != model.model_id or current_face.kind != 'face':
            raise GeometryError('authored child coverage requires a current model face handle')
        identifier = current_face.id
    elif isinstance(current_face, Integral) and not isinstance(current_face, bool):
        identifier = int(current_face)
    else:
        raise GeometryError('authored child coverage requires a current face ID or handle')
    if identifier not in correspondence.descendants or identifier not in model.faces:
        raise GeometryError('authored child coverage face is not an authenticated current descendant')
    triangles = _triangle_rows(triangles_uv)
    # Array coercion may invoke caller code. Authenticate again, then detach
    # ALL literal child inputs before the first cancellation callback. A caller
    # temporarily changing/restoring live topology must not change this proof.
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    face, current = _literal_child_domain(model, identifier)
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)

    def check():
        if cancellation_check is not None and cancellation_check('authored child coverage'):
            raise GeometryError('authored child coverage cancelled')

    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    original = _original_domain(correspondence.authored_definition, check)
    _frame(original.support)
    if face.parameterization is not None:
        raise GeometryError('authored child coverage: explicit child parameterization is unsupported')
    # Do not substitute a tolerance-recovered support for literal Coons truth.
    # The exact coefficient cancellation proof is the sole supported recovery.
    if type(face.surface) is CoonsSurface:
        if face.surface.has_boundaries:
            raise GeometryError('authored child coverage: explicit sampled Coons boundaries are unsupported')
        support = _coons_extrusion({'surface': {'type': 'coons'}, 'corners': list(face.corners)},
                                   current.boundaries, check)
    elif type(face.surface) in (Plane, ExtrudedSurface):
        support = face.surface
    else:
        raise GeometryError('authored child coverage: unsupported literal child surface')
    _support_correspondence(original.support, support, triangles, check)
    _validate_domain_triangles(original, triangles, check)
    # World trims are mapped coefficientwise into ORIGINAL UV by the existing
    # exact coverage kernel; the child's independent native frame is not used.
    _validate_domain_triangles(MaterialDomain(identifier, original.support, current.boundaries), triangles, check)
    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
