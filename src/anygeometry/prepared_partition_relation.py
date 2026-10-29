"""Current live owner-partition disjointness, with global world injectivity.

This adds no planar complementary-domain meaning, persisted authority or
exemption based on support identity alone.
"""
from .errors import GeometryError
from .identity import EntityHandle
from .prepared_material_curvilinear import _literal
from .prepared_material_source_transfer import (
    _prepared_partition_context, _has_current_prepared_partition_lineage, _chart_map, _effective)
from .surfaces import Plane
from .nurbs import NurbsSurface


def has_current_prepared_partition_lineage(model):
    """Whether optional partition proof reuse has complete current live lineage.

    Absent lineage or a normal committed revision advance returns False and
    requires fresh classification. Same-revision content/native changes and
    invalid authority refuse. This does not certify any face pair; explicit
    partition queries retain their strict stale-proof refusal.
    """
    from .model import GeometryModel
    if not isinstance(model, GeometryModel):
        raise GeometryError('prepared partition readiness requires a geometry model')
    try:
        return _has_current_prepared_partition_lineage(model)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:
        if isinstance(error, GeometryError):
            raise
        raise GeometryError('prepared partition lineage is malformed or changed') from error


def query_prepared_partition_disjoint_interiors(model, first_face, second_face, *, cancellation_check=None):
    """True only for distinct current complete-partition siblings on an injective parent.

    Unknown/cross-parent provenance or uncertified global injectivity returns
    False. Stale/malformed live authority, mutation, cancellation and exhausted
    proof resources refuse. Each call reconstructs current owner authority.
    """
    from .model import GeometryModel
    if not isinstance(model,GeometryModel):
        raise GeometryError('prepared partition relation requires a geometry model')
    if (type(first_face) is not EntityHandle or type(second_face) is not EntityHandle
            or first_face.kind != 'face' or second_face.kind != 'face'
            or type(first_face.id) is not int or type(second_face.id) is not int
            or first_face.model_id != model.model_id or second_face.model_id != model.model_id
            or first_face.id == second_face.id or first_face.id not in model.faces or second_face.id not in model.faces):
        raise GeometryError('prepared partition relation requires distinct current model-bound face handles')
    if cancellation_check is not None and not callable(cancellation_check):
        raise GeometryError('prepared partition cancellation_check must be callable')
    inputs = _literal((first_face,second_face))
    current,links,routes = _prepared_partition_context(model)
    def check(phase):
        cancelled = cancellation_check(phase) if cancellation_check is not None else False
        fresh,live,actual = _prepared_partition_context(model)
        if (fresh != current or live is not links or _literal((first_face,second_face)) != inputs
                or len(actual) != len(routes) or any(a[0] is not b[0] or a[1:] != b[1:] for a,b in zip(actual,routes))):
            raise GeometryError('prepared partition operands changed during callback')
        if cancelled:
            raise GeometryError('prepared partition relation cancelled')
    check('prepared partition relation binding')
    candidates = []
    for source,_,mapping in routes:
        for parent,children in mapping:
            if first_face.id in children and second_face.id in children:
                support = _effective(source,parent)
                # Complete material partition issuance already certified chart
                # cells; these fresh maps re-establish physical/chart identity.
                _chart_map(source,parent,_effective(model,first_face.id))
                _chart_map(source,parent,_effective(model,second_face.id))
                candidates.append(support)
    for support in candidates:
        if type(support) is Plane:
            check('prepared partition relation complete')
            return True  # nondegenerate affine Plane charts are globally injective
        if type(support) is not NurbsSurface:
            continue
        from .nurbs_metric import certify_projected_chart
        from .cancellation import is_cancellation_signal
        from .plane_nurbs import PlaneNurbsNotCertified
        passive = is_cancellation_signal(cancellation_check)
        callback_failure = [None]
        def arbitrary_frontier():
            try:
                check('prepared partition injectivity')
            except BaseException as error:
                callback_failure[0] = error
                raise
            return False
        guarded = (None if cancellation_check is None else cancellation_check if passive else arbitrary_frontier)
        try:
            # A SAME strongly monotone projection certifies the WHOLE convex
            # normalized parent chart, including every trimmed material cell.
            certify_projected_chart(support,cancellation_check=guarded)
        except PlaneNurbsNotCertified:
            raise  # budget/depth/kernel failure is never a disjointness False
        except GeometryError as error:
            if callback_failure[0] is error:
                raise  # even a callback matching the kernel's refusal text is caller failure
            if type(error) is not GeometryError or str(error) != 'NURBS support immersion and injective projection are not certified':
                raise
            check('prepared partition injectivity unresolved')
            continue  # only the registered kernel's explicit inability is False
        check('prepared partition relation complete')
        return True
    check('prepared partition relation complete')
    return False
