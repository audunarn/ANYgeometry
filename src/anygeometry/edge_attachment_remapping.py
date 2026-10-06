"""Explicit, transactional attachment remapping for a split topology edge."""
from dataclasses import replace

import numpy as np

from .arrangement_geometry import freeze_edge, point_parameters
from .errors import GeometryError
from .structural import Orientation, ParameterRange


def _one_parameter(curve, point, tolerance):
    values = point_parameters(curve, point, tolerance=tolerance)
    if len(values) != 1:
        raise GeometryError("edge attachment remap needs one certified point parameter")
    parameter = float(values[0])
    if np.linalg.norm(curve.evaluate(parameter) - point) > tolerance:
        raise GeometryError("edge attachment remap exceeds its world tolerance")
    return parameter


def _member_parameter(model, member_id, point, interval, tolerance):
    candidates = []
    for use_id in model.members[member_id].edge_use_ids:
        use = model.member_edge_uses[use_id]
        curve = freeze_edge(model, use.edge_id)
        for local in point_parameters(curve, point, tolerance=tolerance):
            if np.linalg.norm(curve.evaluate(local) - point) > tolerance:
                continue
            if use.orientation is Orientation.REVERSED:
                local = 1 - local
            parent = use.parent_range.start + local * use.parent_range.length
            if interval.contains(parent, tolerance=model.tolerance.parameter):
                parent = min(interval.end, max(interval.start, parent))
                if not any(abs(parent - old) <= model.tolerance.parameter for old in candidates):
                    candidates.append(parent)
    if len(candidates) != 1:
        raise GeometryError("edge attachment remap needs one source member parameter")
    return candidates[0]


def _capture_member_ranges(model, members, check):
    """Capture every owner-held member station before child charts change."""
    def capture(member, interval):
        endpoints = []
        for parameter in (interval.start, interval.end):
            for use_id in model.members[member].edge_use_ids:
                use = model.member_edge_uses[use_id]
                if use.parent_range.contains(parameter, tolerance=model.tolerance.parameter):
                    local = (parameter - use.parent_range.start) / use.parent_range.length
                    if use.orientation is Orientation.REVERSED:
                        local = 1 - local
                    point = model.sample_edge(use.edge_id, np.asarray([min(1., max(0., local))]))[0]
                    endpoints.append((point, use.parent_range))
                    break
            else:
                raise GeometryError("attachment source lies outside its member")
        return member, tuple(endpoints)

    attachments, junctions = {}, {}
    for member in sorted(members):
        check()
        identifiers = (model._source_attachments.get(('member', member), set())
                       | model._target_attachments.get(('member', member), set()))
        for identifier in sorted(identifiers):
            record = model.attachments[identifier]
            fields = attachments.setdefault(identifier, {})
            if record.source_kind == 'member' and record.member_id == member:
                fields['member_range'] = capture(member, record.member_range)
            if record.target_kind.value == 'member' and record.target_id == member:
                fields['target_parameters'] = capture(member, record.target_parameters[0])
        for identifier, junction in model.junctions.items():
            for index, use in enumerate(junction.member_uses):
                if use.member_id == member:
                    junctions.setdefault(identifier, {})[index] = capture(member, use.member_range)
    return attachments, junctions


def _remap_member_ranges(model, snapshots, split_targets, tolerance, check):
    def remap(captured):
        member, endpoints = captured
        values = [_member_parameter(model, member, point, interval, tolerance) for point, interval in endpoints]
        return ParameterRange(min(values), max(values))

    attachments, junctions = snapshots
    for identifier, fields in attachments.items():
        check()
        # Target-edge relations were subdivided and mapped separately.
        if identifier in split_targets:
            continue
        changes = {name: ((remap(value),) if name == 'target_parameters' else remap(value))
                   for name, value in fields.items()}
        model._put_structural('attachment', replace(model.attachments[identifier], **changes))
    for identifier, fields in junctions.items():
        check()
        record = model.junctions[identifier]
        uses = tuple(replace(use, member_range=remap(fields[index])) if index in fields else use
                     for index, use in enumerate(record.member_uses))
        model._put_structural('junction', replace(record, member_uses=uses))


def split_edge_attachments(model, edge_id, parameter, check=lambda: None):
    """Split with explicit point/interval maps inside an owner transaction.

    Child charts can reparameterize nonlinearly. Recover coordinates from their
    exact curves, never from a linear fraction rule. Unqualified direct splits
    retain their historical refusal by default.
    """
    if not 0. < parameter < 1.:
        raise GeometryError("split parameter must be strictly between 0 and 1")
    if model._source_attachments.get(('edge', edge_id)):
        raise GeometryError("source-edge attachments require a qualified source interval remap")
    attachments = [model.attachments[i] for i in sorted(model._target_attachments.get(('edge', edge_id), ()))]
    affected_members = set(model.members_using_edge(edge_id))
    if not attachments and not affected_members:
        return model.split_edge(edge_id, parameter)
    original = freeze_edge(model, edge_id)
    lo, hi = original.bounds()
    tolerance = model.tolerance.effective_length(float(np.linalg.norm(hi - lo)))
    member_snapshots = _capture_member_ranges(model, affected_members, check)
    pieces = []
    junctions = {i: model.junctions[i] for attachment in attachments
                 for i in model._attachment_junctions.get(attachment.id, ())}
    for attachment in attachments:
        check()
        if attachment.source_kind not in ('vertex', 'sheet', 'member'):
            raise GeometryError("edge attachment source has no qualified interval remap")
        interval = attachment.target_parameters[0]
        for child_index, (a, b) in enumerate(((0., parameter), (parameter, 1.))):
            lower, upper = max(a, interval.start), min(b, interval.end)
            if upper < lower or (upper == lower and not interval.is_point):
                continue
            # A point on the split belongs to the first child deterministically.
            if interval.is_point and interval.start == parameter and child_index == 1:
                continue
            endpoints = tuple(original.evaluate(t) for t in (lower, upper))
            source_interval = attachment.member_range
            if attachment.source_kind == 'member':
                values = [_member_parameter(model, attachment.member_id, p,
                                            source_interval, tolerance) for p in endpoints]
                source_interval = ParameterRange(min(values), max(values))
            elif attachment.source_kind == 'vertex':
                if not interval.is_point or np.linalg.norm(model.vertex_position(attachment.source_id) - endpoints[0]) > tolerance:
                    raise GeometryError("vertex attachment does not match the split-edge station")
            pieces.append((attachment, child_index, endpoints, source_interval))
    for attachment in attachments:
        model._delete_structural('attachment', attachment.id)
    vertex, children = model.split_edge(edge_id, parameter)
    child_curves = [freeze_edge(model, child) for child in children]
    replacements = {attachment.id: [] for attachment in attachments}
    for attachment, child_index, endpoints, member_range in pieces:
        check()
        values = [_one_parameter(child_curves[child_index], p, tolerance) for p in endpoints]
        if attachment.source_kind == 'member' and attachment.member_id in affected_members:
            values_member = [_member_parameter(model, attachment.member_id, p,
                                               ParameterRange(0., 1.), tolerance) for p in endpoints]
            member_range = ParameterRange(min(values_member), max(values_member))
        identifiers = replacements[attachment.id]
        identifier = attachment.id if not identifiers else model._allocate_structural('attachment')
        lineage = tuple(dict.fromkeys((*attachment.lineage, ('attachment',attachment.id), ('edge', edge_id))))
        model._put_structural('attachment', replace(attachment, id=identifier,
            target_id=children[child_index], target_parameters=(ParameterRange(min(values), max(values)),),
            member_range=member_range, lineage=lineage))
        identifiers.append(identifier)
    for identifier, junction in junctions.items():
        expanded = tuple(dict.fromkeys(new for old in junction.attachment_ids
                        for new in replacements.get(old, (old,))))
        model._put_structural('junction', replace(junction, attachment_ids=expanded))
    _remap_member_ranges(model, member_snapshots, replacements, tolerance, check)
    return vertex, children
