"""Source-checked curvature size proposals, never approximation admission."""
from fractions import Fraction as Q
from .errors import GeometryError
from .nurbs import NurbsSurface,_sqrt_down
from .surfaces import Plane
from .definition_binding import definition_checksum
from .prepared_material_source_transfer import _state,_effective
from .native_support_snapshots import _material_support_read_pin
from ._local_curvature import bound_surface_curvature,LocalApproximationNotCertified


def curvature_size_hints(model,face,error,*,cancellation_check=None,local_approximation=None):
    from .nurbs_sweep_chart import has_sweep_chart
    raw=model.faces[face]
    if not (type(raw.surface) in (NurbsSurface,Plane) or has_sweep_chart(raw)):
        return ()
    before=_state(model);support=_effective(model,face)
    if type(support) not in (NurbsSurface,Plane):return ()
    signature=definition_checksum(support);callback_failure=[]
    def check(phase='curvature size proposal'):
        if _state(model)!=before or definition_checksum(support)!=signature:
            raise GeometryError('size proposal source changed')
        if type(support) is NurbsSurface and _material_support_read_pin(support) is None:
            raise GeometryError('size proposal native support is unbound')
        if cancellation_check is not None:
            try:cancelled=cancellation_check(phase)
            except BaseException as failure:callback_failure.append(failure);raise
            if cancelled:raise GeometryError('size proposal cancelled')
        if _state(model)!=before or definition_checksum(support)!=signature:
            raise GeometryError('size proposal source changed during callback')
        if type(support) is NurbsSurface and _material_support_read_pin(support) is None:
            raise GeometryError('size proposal native support changed')
        return False
    check()
    try:curvature=bound_surface_curvature(support,cancellation_check=check)
    except LocalApproximationNotCertified as failure:
        if any(failure is item for item in callback_failure):raise
        check()
        # A failed advisory bound grants nothing and does not tighten the
        # legacy scalar acceptance route. Integrity/callback failures above
        # retain their hard, original exception behavior.
        return ()
    from .curvilinear_reference_maps import _source_box
    from dataclasses import replace
    from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as policy
    query=None
    if local_approximation is not None:
        from .local_approximation_field import prepare_local_cell_limit_query
        query=prepare_local_cell_limit_query(model,local_approximation)
    tiles=[];used=curvature.boxes
    for original in curvature.tiles:
        if original.curvature_upper==0:tiles.append(original);continue
        (a,b),(c,d)=original.chart_box
        for i in range(4):
            for j in range(4):
                check();remaining=policy.max_boxes_per_pair-used
                if remaining<2:return ()
                box=((a+(b-a)*Q(i,4),a+(b-a)*Q(i+1,4)),
                     (c+(d-c)*Q(j,4),c+(d-c)*Q(j+1,4)))
                try:child=bound_surface_curvature(support,box,
                    policy=replace(policy,max_boxes_per_pair=remaining),cancellation_check=check)
                except LocalApproximationNotCertified as failure:
                    if any(failure is item for item in callback_failure):raise
                    check();return ()
                used+=child.boxes;tiles.extend(child.tiles)
    rows=[]
    for tile in tiles:
        check()
        if tile.curvature_upper>0:
            allowance=Q(error) if query is None else min(Q(error),query(face,(tile.chart_box,))[0])
            size=_sqrt_down(8*allowance/tile.curvature_upper)
            rows.append((_source_box(support,tile.chart_box),size,tile.curvature_upper))
    check()
    if local_approximation is not None:
        from .local_approximation_field import validate_local_approximation_allocation
        validate_local_approximation_allocation(model,local_approximation)
    return tuple(rows)
