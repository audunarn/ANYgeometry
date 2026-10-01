"""Compatibility projection of the shared material engine to pair queries."""
import numpy as np

from .errors import GeometryError
from .material_arrangement import MaterialDomain
from .batch_intersections import _pair_paths, _domain_bounds
from .predicates import (IntersectionComponent, IntersectionCertificate,
                         IntersectionDimension, IntersectionKind, IntersectionQuality)
from .surfaces import Plane, Cylinder


def pair_plan_operands(model, first, second):
    """Preserve the historical transverse-query complete-band operation."""
    operands={first,second}
    if first.kind!='face' or second.kind!='face':
        return tuple(sorted(operands))
    for curved,flat in ((first,second),(second,first)):
        a,b=model.faces[curved.id].surface,model.faces[flat.id].surface
        if isinstance(a,Cylinder) and isinstance(b,Plane) and (
                abs(abs(float(a.axis @ b.normal))-1.) <= model.tolerance.angular):
            from .intersections import _same_cylinder_band
            operands.update(model.handle('face',face.id) for face in model.faces.values()
                if isinstance(face.surface,Cylinder)
                and _same_cylinder_band(face.surface,a,tolerance=model.tolerance.length))
    return tuple(sorted(operands))


def domains_for_pair(model, first, second):
    return (MaterialDomain.from_model(model, first.id),
            MaterialDomain.from_model(model, second.id))


def query_exact_pair(model, first, second):
    from .intersections import _qualified_result
    a, b = domains_for_pair(model, first, second)
    length = max(float(np.linalg.norm(hi-lo)) for domain in (a,b)
                 for lo,hi in (_domain_bounds(domain),))
    tolerance = model.tolerance.effective_length(length)
    try:
        points=[]
        curves = _pair_paths(a,b,model,tolerance,lambda: None,point_contacts=points)
    except GeometryError as error:
        return _qualified_result(model,first,second,IntersectionKind.UNCLASSIFIED,
            diagnostics=(str(error),),tolerance_used=tolerance)
    components = tuple(IntersectionComponent(
        tuple(tuple(float(value) for value in curve.evaluate(t)) for t in (0.,.5,1.)),
        IntersectionQuality.EXACT,
        first_parameter_path=tuple(tuple(a.uv(curve,t)) for t in (0.,.5,1.)),
        second_parameter_path=tuple(tuple(b.uv(curve,t)) for t in (0.,.5,1.)),
        analytic_curve=curve,
        certificate=IntersectionCertificate('analytic_material_arrangement',tolerance,
                                            complete=True),
        first_subparent=first,second_subparent=second) for curve in curves)
    point_components=tuple(IntersectionComponent((point,),IntersectionQuality.EXACT,
        first_parameter=tuple(a.support.local_uv(point)),second_parameter=tuple(b.support.local_uv(point)),
        first_subparent=first,second_subparent=second,
        certificate=IntersectionCertificate('analytic_material_point_contact',tolerance,complete=True))
        for point in points)
    return _qualified_result(model,first,second,
        IntersectionKind.CROSS if components else IntersectionKind.TOUCH_POINT if point_components else IntersectionKind.DISJOINT,
        (*components,*point_components),dimension=IntersectionDimension.CURVE if components else
        IntersectionDimension.POINT if point_components else IntersectionDimension.NONE,
        tolerance_used=tolerance)
