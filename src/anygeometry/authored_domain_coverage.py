"""Exact triangle coverage of an authenticated ORIGINAL face domain only.

Success does not establish equality with the literal current fragment union,
current station incidence, a material partition, or any mesh/edit permission.
"""
import json
from fractions import Fraction as F

from .arrangement_geometry import LinePath, BezierPath
from .errors import GeometryError
from .extrusions import BezierDirectrix
from .material_arrangement import ArrangementPath, MaterialDomain
from .material_cell_coverage import _triangle_rows, _validate_domain_triangles, _frame
from .serialization import _decode_surface, _QUADRIC_VERSION
from .surfaces import ExtrudedSurface


def _coons_extrusion(face, loops, check):
    """Prove the original tensor Coons formula is exactly B(u)+v*D."""
    if (face['surface'] != {'type': 'coons'} or face['corners'] != [0, 1, 2, 3]
            or len(loops) != 1 or len(loops[0]) != 4):
        raise GeometryError('authored domain requires an exact four-edge implicit Coons extrusion')
    bottom, right, top, left = (path.curve for path in loops[0])
    if (type(bottom) is not BezierPath or type(top) is not BezierPath
            or type(right) is not LinePath or type(left) is not LinePath):
        raise GeometryError('authored Coons extrusion requires Bezier boundaries and straight connectors')
    exact = lambda point: tuple(F(float(value)) for value in point)
    lower = tuple(map(exact, bottom.controls))
    upper = tuple(map(exact, top.controls[::-1]))
    if len(lower) != len(upper):
        raise GeometryError('authored Coons boundary degrees do not match exactly')
    direction = tuple(b-a for a, b in zip(lower[0], upper[0]))
    for first, second in zip(lower, upper):
        check()
        if tuple(b-a for a, b in zip(first, second)) != direction:
            raise GeometryError('authored Coons boundaries are not exact coefficient translations')
    if (exact(right.start) != lower[-1] or exact(right.end) != upper[-1]
            or exact(left.start) != upper[0] or exact(left.end) != lower[0]):
        raise GeometryError('authored Coons connector endpoints do not close exactly')
    try:
        vector = tuple(float(value) for value in direction)
        represented = tuple(F(value) for value in vector)
    except (ValueError, OverflowError) as error:
        raise GeometryError('authored Coons translation is not exactly representable') from error
    if represented != direction:
        raise GeometryError('authored Coons translation is not exactly representable')
    # The two linear connector interpolants cancel the bilinear corner term.
    # The remaining blend is (1-v)*B(u)+v*(B(u)+D), coefficientwise.
    support = ExtrudedSurface(BezierDirectrix(bottom.controls), vector)
    _frame(support)  # exact profile plane, transversality and injectivity proof
    return support


def _original_domain(definition, check):
    """Rebuild only authenticated original polynomial loops, including holes."""
    data = json.loads(definition.definition_json)
    face = data['face']
    if face['id'] != definition.face_id:
        raise GeometryError('authored domain face definition identity changed')
    if face['parameterization'] is not None:
        raise GeometryError('authored domain coverage does not support an original parameterization')
    # These are prospectively captured current-codec definitions. The codec's
    # default VERSION predates extruded supports; no document is reconstructed.
    vertices = {item['id']: tuple(item['position']) for item in data['vertices']}
    edges = {item['id']: item for item in data['edges']}
    loops = []
    for source_loop in (face['loop'], *face['holes']):
        paths = []
        for edge_id, forward in source_loop:
            check()
            edge = edges[edge_id]
            kind = edge['curve']['type']
            if kind == 'straight':
                points = vertices[edge['start']], vertices[edge['end']]
                if not forward:
                    points = points[::-1]
                curve = LinePath(*points)
            elif kind == 'spline':
                ids = (edge['start'], *edge['curve']['control_vertices'], edge['end'])
                points = tuple(vertices[identifier] for identifier in ids)
                curve = BezierPath(points if forward else points[::-1])
            else:
                raise GeometryError('authored domain coverage requires original polynomial trims')
            paths.append(ArrangementPath(curve, edge_id))
        loops.append(tuple(paths))
    loops = tuple(loops)
    if isinstance(face['surface'], dict) and face['surface'].get('type') == 'coons':
        support = _coons_extrusion(face, loops, check)
    else:
        support = _decode_surface(face['surface'], strict=True, schema_version=_QUADRIC_VERSION)
    return MaterialDomain(face['id'], support, loops)


def validate_prepared_authored_face_triangles(model, correspondence, triangles_uv, *, cancellation_check=None):
    """Certify closed UV triangles in the ORIGINAL authenticated face domain.

    UV coordinates use that face's original support/ranges. All original holes
    participate. Current-fragment coverage/partition equivalence is NOT claimed.
    Original explicit parameterizations refuse. Implicit four-edge Coons charts
    qualify only through an exact coefficient proof of Bezier extrusion form.
    The same exact polynomial correspondence, winding, event and refusal rules
    as current-region coverage apply. No source or supplied input is modified.
    """
    triangles = _triangle_rows(triangles_uv)
    def check():
        if cancellation_check is not None and cancellation_check('authored domain coverage'):
            raise GeometryError('authored domain coverage cancelled')
    from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence, cancellation_check=cancellation_check)
    domain = _original_domain(correspondence.authored_definition, check)
    _validate_domain_triangles(domain, triangles, check)
    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence, cancellation_check=cancellation_check)
