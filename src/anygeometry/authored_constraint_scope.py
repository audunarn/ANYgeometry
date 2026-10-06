"""Complete typed owner inventory for selected authored-root traces.

Literal record equality is not semantic preservation, a parameter remap,
material coverage, external-reference completeness or mesh admission.
"""
from dataclasses import dataclass
import json
from numbers import Integral

import numpy as np

from .authored_boundary_correspondence import (
    AuthoredBoundaryCorrespondence,
    query_prepared_authored_boundary_correspondence,
    validate_prepared_authored_boundary_correspondence_binding,
)
from .definition_binding import definition_checksum
from .errors import GeometryError
from .prepared_model_scope import (
    PreparedModelScope, query_prepared_model_scope, validate_prepared_model_scope_binding,
)


_RECORD_KINDS = ('members', 'member_edge_uses', 'attachments', 'junctions')


@dataclass(frozen=True, slots=True)
class PreparedAuthoredConstraintScope:
    scope: PreparedModelScope
    selected_root_ids: tuple
    current_face_ids: tuple
    outside_root_ids: tuple
    boundary_correspondences: tuple
    inventory_json: str
    typed_inventory_complete: bool = True
    semantic_mapping_qualified: bool = False
    parameter_remapping_qualified: bool = False
    material_qualified: bool = False
    publication_qualified: bool = False
    external_reference_scope_qualified: bool = False

    @property
    def inventory(self):
        """Fresh detached data; original/current IDs retain distinct meanings."""
        return json.loads(self.inventory_json)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _signature(receipt):
    try:
        return definition_checksum(receipt)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored constraint scope receipt is malformed') from error


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GeometryError(f'authored constraint scope requires integer {label}')
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f'authored constraint scope requires usable integer {label}') from error


def _check(callback):
    if callback is not None and callback('authored constraint scope'):
        raise GeometryError('authored constraint scope cancelled')


def _isolated_vertices(document):
    referenced = set()
    for edge in document['edges']:
        referenced.update((edge['start'], edge['end']))
        referenced.update(edge['curve'].get('control_vertices', ()))
        if 'via_vertex' in edge['curve']:
            referenced.add(edge['curve']['via_vertex'])
    # Face.corners are LOOP OFFSETS. Endpoints cover their derived vertices.
    return sorted({row['id'] for row in document['vertices']} - referenced)


def _document_inventory(document):
    metadata = {}
    tables = {kind: document[kind] for kind in ('vertices', 'edges', 'faces')}
    tables.update(document['structural'])
    for kind, rows in tables.items():
        metadata[kind] = [{'id': row['id'], 'metadata': row['metadata']}
                          for row in rows if 'metadata' in row]
    return {
        'records': {kind: document['structural'][kind] for kind in _RECORD_KINDS},
        'isolated_vertex_ids': _isolated_vertices(document),
        'opaque_unqualified': {
            'groups': document['groups'], 'tags': document['tags'],
            'features': document.get('features', {}), 'extensions': document['extensions'],
            'metadata': metadata, 'construction_vertices': document['construction_vertices'],
            'coordinates': document['coordinates'],
        },
    }


def _dispositions(original, current):
    result = []
    for kind in _RECORD_KINDS:
        before = {row['id']: row for row in original['structural'][kind]}
        after = {row['id']: row for row in current['structural'][kind]}
        for key in sorted(before.keys() | after.keys()):
            if key not in before:
                status = 'current_only'
            elif key not in after:
                status = 'source_only'
            elif _encode(before[key]) == _encode(after[key]):
                status = 'literal_unchanged'
            else:
                status = 'literal_changed'
            result.append({'kind': kind, 'id': key, 'literal_disposition': status,
                           'semantic_mapping_qualified': False,
                           'parameter_remapping_qualified': False})
    return result


def _complete_native_reference_inventory(original, current, descendants, selected,
                                         edge_maps, check, source_native=None, current_native=None,
                                         orientation_relations=None, material_faces=(),request_proof=None,replacement_history=(),attachment_source_ids=None,epoch_derived_attachment_ids=None):
    """Complete native caller inventory, independent of boundary cancellation.

    This additive path receives detached documents and one batch's authenticated
    native maps. Historical literal scope flags above remain unchanged.
    """
    source_native = {} if source_native is None else source_native
    current_native = {} if current_native is None else current_native
    orientation_relations = {} if orientation_relations is None else orientation_relations
    material_faces = set(material_faces)
    def dependency_vertices(edge):
        return (edge['start'],edge['end'],*edge['curve'].get('control_vertices',()),
                *((edge['curve']['via_vertex'],) if 'via_vertex' in edge['curve'] else ()))
    roots_by_face = {}
    for root, children in descendants:
        for child in children:
            roots_by_face.setdefault(child, set()).add(root)
    selected_faces = {child for root, children in descendants if root in selected for child in children}
    incidence = {}
    for face in current['faces']:
        for index, loop in enumerate((face['loop'], *face['holes'])):
            for edge, forward in loop:
                check()
                incidence.setdefault(edge, []).append((face['id'], index, forward))
    selected_edges = {edge for edge, uses in incidence.items()
                      if any(face in selected_faces for face, _, _ in uses)}
    current_edge_index = {edge['id']:edge for edge in current['edges']}
    vertex_faces = {}
    for edge, uses in incidence.items():
        for vertex in dependency_vertices(current_edge_index[edge]):
            vertex_faces.setdefault(vertex,set()).update(face for face,_,_ in uses)
    selected_vertices = {v for edge in selected_edges for v in dependency_vertices(current_edge_index[edge])}
    outside = set()
    for vertex in selected_vertices:
        outside.update(root for face in vertex_faces.get(vertex,())
                       for root in roots_by_face.get(face,()) if root not in selected)
    # Complete dependency closure includes carrier vertices and structural owners.
    graph={}
    def link(first,second):
        graph.setdefault(first,set()).add(second);graph.setdefault(second,set()).add(first)
    for edge,uses in incidence.items():
        for face,_,_ in uses:
            link(('face',face),('edge',edge))
    for edge in current['edges']:
        for vertex in dependency_vertices(edge):
            link(('edge',edge['id']),('vertex',vertex))
    structural=current['structural']
    for use in structural['face_uses']:
        link(('sheet',use['sheet_id']),('face',use['face_id']))
    for part in structural['parts']:
        for sheet in part['sheet_ids']:
            link(('part',part['id']),('sheet',sheet))
        for member in part['member_ids']:
            link(('part',part['id']),('member',member))
    for use in structural['member_edge_uses']:
        link(('member',use['member_id']),('edge',use['edge_id']))
    for member in structural['members']:
        if member.get('orientation_reference'):
            link(('member',member['id']),tuple(member['orientation_reference']))
    for attachment in structural['attachments']:
        node=('attachment',attachment['id'])
        link(node,(attachment['target_kind'],attachment['target_id']))
        if attachment['source_kind']:
            link(node,(attachment['source_kind'],attachment['source_id']))
        for kind in ('member','sheet','part'):
            if attachment.get(kind+'_id') is not None:
                link(node,(kind,attachment[kind+'_id']))
    for junction in structural['junctions']:
        node=('junction',junction['id'])
        for kind,keys in (('sheet',junction['sheet_ids']),('attachment',junction['attachment_ids']),
                          ('member',[u['member_id'] for u in junction['member_uses']])):
            for key in keys:
                link(node,(kind,key))
    reachable={('face',face) for face in selected_faces};pending=list(reachable)
    while pending:
        check()
        for node in graph.get(pending.pop(),()):
            if node not in reachable:
                reachable.add(node);pending.append(node)
    outside.update(root for kind,key in reachable if kind=='face'
                   for root in roots_by_face.get(key,()) if root not in selected)
    traces = []
    for edge, uses in sorted(incidence.items()):
        check()
        if edge not in selected_edges:
            continue
        neighbours = {root for face, _, _ in uses for root in roots_by_face.get(face, ())}
        outside.update(neighbours-set(selected))
        traces.append({'edge_id':edge, 'all_face_loop_incidence':uses,
            'selected_face_ids':sorted({f for f,_,_ in uses if f in selected_faces}),
            'outside_root_ids':sorted(neighbours-set(selected)),
            'interior_constraint':len([f for f,_,_ in uses if f in selected_faces])>1})
    maps_by_pair = {}
    for row in edge_maps:
        check()
        maps_by_pair.setdefault((row['ancestor_edge_id'],row['edge_id']), []).append(row)
    original_edges = {row['id']:row for row in original['edges']}
    current_edges = {row['id']:row for row in current['edges']}
    source_uses = {row['id']:row for row in original['structural']['member_edge_uses']}
    current_uses = {row['id']:row for row in current['structural']['member_edge_uses']}
    source_by_member = {}
    for use in source_uses.values():
        source_by_member.setdefault(use['member_id'],[]).append(use)
    from fractions import Fraction as F
    use_maps, use_ok, mapped_source = [], {}, {}
    tau = F(original['tolerance']['parameter'])
    for use in current_uses.values():
        check()
        candidates = []
        r, s = map(F,use['parent_range'])
        for source in source_by_member.get(use['member_id'],()):
            check()
            if source['member_id'] != use['member_id']:
                continue
            p, q = map(F,source['parent_range'])
            for mapping in maps_by_pair.get((source['edge_id'],use['edge_id']), ()):
                if (mapping['ancestor_source_checksum']!=original['checksum']['value'] or
                        mapping['ancestor_revision']!=original['revision'] or
                        mapping['ancestor_model_id']!=original['model_id'] or
                        use['metadata']!=source['metadata']):
                    continue
                a,b = map(lambda x:F(*x), mapping['interval'])
                lo,hi = (a,b) if use['orientation']=='forward' else (b,a)
                if source['orientation'] != 'forward':
                    lo,hi = 1-lo,1-hi
                expected = (p+(q-p)*lo,p+(q-p)*hi)
                discrepancy = max(abs(r-expected[0]),abs(s-expected[1]))
                if discrepancy <= tau and mapping['classification'] in ('exact','bounded'):
                    candidates.append((source,mapping,discrepancy))
        if len(candidates)==1:
            source,mapping,discrepancy = candidates[0]
            status = 'exact_native_map' if discrepancy==0 and mapping['classification']=='exact' else 'bounded_native_map'
            use_ok[use['id']] = True
            mapped_source.setdefault(source['id'],[]).append(mapping['interval'])
            use_maps.append({'current_use_id':use['id'],'source_use_id':source['id'],
                'member_id':use['member_id'],'source_edge_id':source['edge_id'],
                'current_edge_id':use['edge_id'],'native_map':mapping,
                'source_traversal':source['orientation'],'current_traversal':use['orientation'],
                'source_parent_range':source['parent_range'],'current_parent_range':use['parent_range'],
                'parameter_residual_bound':[discrepancy.numerator,discrepancy.denominator],
                'disposition':status})
        else:
            use_ok[use['id']] = False
            use_maps.append({'current_use_id':use['id'],'disposition':'refused_ambiguous_or_unavailable_native_map'})
    source_coverage = {}
    for source in source_uses.values():
        spans = sorted(tuple(sorted(F(*x) for x in pair)) for pair in mapped_source.get(source['id'],()))
        source_coverage[source['id']] = bool(spans and spans[0][0]==0 and spans[-1][1]==1
            and all(a[1]==b[0] for a,b in zip(spans,spans[1:])))
    vertex_before = {r['id']:r for r in original['vertices']}
    vertex_after = {r['id']:r for r in current['vertices']}
    identical_vertices = {key for key in vertex_before.keys() & vertex_after.keys()
                          if vertex_before[key]==vertex_after[key]}
    identical_edges = {key for key in original_edges.keys() & current_edges.keys()
                       if original_edges[key]==current_edges[key] and
                       all(v in identical_vertices for v in dependency_vertices(original_edges[key]))}
    original_faces = {r['id']:r for r in original['faces']}
    current_faces = {r['id']:r for r in current['faces']}
    identical_faces = {key for key in original_faces.keys() & current_faces.keys()
        if original_faces[key]==current_faces[key] and source_native.get(key)==current_native.get(key)
        and source_native.get(key,(None,()))[0] in ('plane','cylinder') and all(edge in identical_edges
            for loop in (original_faces[key]['loop'],*original_faces[key]['holes']) for edge,_ in loop)}
    member_ok = {}
    before_members = {r['id']:r for r in original['structural']['members']}
    after_members = {r['id']:r for r in current['structural']['members']}
    maps_by_use = {r['current_use_id']:r for r in use_maps}
    for key in before_members.keys() | after_members.keys():
        check()
        before,after = before_members.get(key),after_members.get(key)
        if before is None or after is None:
            member_ok[key] = False
            continue
        fields_before = {k:v for k,v in before.items() if k!='edge_use_ids'}
        fields_after = {k:v for k,v in after.items() if k!='edge_use_ids'}
        reference = before.get('orientation_reference')
        reference_ok = (reference is None or reference[1] in {'vertex':identical_vertices,
            'edge':identical_edges,'face':identical_faces}.get(reference[0],set()))
        member_ok[key] = (fields_before==fields_after and reference_ok and
            all(use_ok.get(use,False) for use in after['edge_use_ids']) and
            all(source_coverage.get(use,False) for use in before['edge_use_ids']))
        mapped_order=[maps_by_use.get(use,{}).get('source_use_id')
                      for use in after['edge_use_ids']]
        collapsed=[value for i,value in enumerate(mapped_order) if i==0 or value!=mapped_order[i-1]]
        member_ok[key] = member_ok[key] and collapsed==before['edge_use_ids']
    dispositions = _dispositions(original,current)
    before_attachments = {r['id']:r for r in original['structural']['attachments']}
    after_attachments = {r['id']:r for r in current['structural']['attachments']}
    before_junctions = {r['id']:r for r in original['structural']['junctions']}
    after_junctions = {r['id']:r for r in current['structural']['junctions']}
    uses_by_face = {}
    for use in current['structural']['face_uses']:
        uses_by_face.setdefault(use['face_id'],[]).append(use)
    derived_attachments, derived_junctions = set(),set()
    contract = {'contract':'ANYGEOMETRY_ANALYTIC_JOINT_EDGE_V1'}
    for joint in after_junctions.values():
        check()
        if (joint['id'] in before_junctions or joint['kind']!='sheet_joint'
                or joint['member_uses'] or joint['metadata'] or joint['provenance']!=contract
                or joint['connection_intent']!='connect'):
            continue
        attachments=[after_attachments.get(key) for key in joint['attachment_ids']]
        if not attachments or any(a is None or a['id'] in before_attachments for a in attachments):
            continue
        targets={a['target_id'] for a in attachments}
        if len(targets)!=1:
            continue
        edge=next(iter(targets)); faces={f for f,_,_ in incidence.get(edge,())}
        owners={u['sheet_id'] for f in faces for u in uses_by_face.get(f,())}
        if (len(owners)<2 or not faces<=material_faces or set(joint['sheet_ids'])!=owners
                or len(joint['sheet_ids'])!=len(owners) or len(attachments)!=len(owners)):
            continue
        if any(a['member_id'] is not None or a['kind']!='sheet_on_joint'
               or a['target_kind']!='edge' or a['member_range']!=[0.,1.]
               or a['target_parameters']!=[[0.,1.]] or a['source_kind']!='sheet'
               or a['source_id'] not in owners or a['connection_intent']!='connect'
               or a['evidence']!='exact' or a['max_residual']!=0
               or a['tolerance_used']<0 or a['metadata'] or a['provenance']!=contract
               or a['lineage'] or a['part_id'] is not None or a['sheet_id'] is not None
               for a in attachments):
            continue
        if {a['source_id'] for a in attachments}!=owners:
            continue
        # No unclaimed same-target records may hide an incomplete/extra owner collection.
        if {a['id'] for a in after_attachments.values() if a['target_kind']=='edge'
            and a['target_id']==edge and a['kind']=='sheet_on_joint'}!={a['id'] for a in attachments}:
            continue
        if epoch_derived_attachment_ids is not None and not {a['id'] for a in attachments}<=set(epoch_derived_attachment_ids):
            continue
        derived_attachments.update(a['id'] for a in attachments)
        derived_junctions.add(joint['id'])
    attachment_ok = {}
    for row in dispositions:
        check()
        kind,key = row['kind'],row['id']
        ok = False
        if kind=='members':
            ok = member_ok.get(key,False)
        elif kind=='member_edge_uses':
            ok = use_ok.get(key,False)
            if key in source_uses:
                ok = ok or source_coverage.get(key,False)
        elif kind=='attachments':
            source = before_attachments.get(key)
            after = after_attachments.get(key)
            ok = (source is not None and source==after and (source['member_id'] is None or member_ok.get(source['member_id'],False))
                and source['target_id'] in {'edge':identical_edges,'face':identical_faces}.get(source['target_kind'],set())
                and (source['source_kind'] is None or source['source_id'] in
                     {'vertex':identical_vertices,'edge':identical_edges,'face':identical_faces,
                      'member':{key for key,value in member_ok.items() if value}}.get(source['source_kind'],set())))
            ok = ok or key in derived_attachments
            attachment_ok[key] = ok
        row['geometry_semantic_disposition'] = 'qualified_native_owner_semantics' if ok else 'refused_owner_semantics_unresolved'
        row['semantic_mapping_qualified'] = ok
        row['parameter_remapping_qualified'] = ok
        if kind=='attachments' and key in derived_attachments:
            row['geometry_semantic_disposition']='derived_exact_joint_incidence'
            row['parameter_remapping_qualified']=False
    for row in dispositions:
        if row['kind'] != 'junctions':
            continue
        source = before_junctions.get(row['id'])
        after = after_junctions.get(row['id'])
        ok = (source is not None and source==after and all(member_ok.get(use['member_id'],False)
            for use in source['member_uses']) and all(attachment_ok.get(a,False) for a in source['attachment_ids']))
        ok=ok or row['id'] in derived_junctions
        row.update(geometry_semantic_disposition=('derived_exact_joint_incidence' if row['id'] in derived_junctions else
                   'qualified_native_owner_semantics' if ok else 'refused_owner_semantics_unresolved'),
                   semantic_mapping_qualified=ok,parameter_remapping_qualified=(ok and row['id'] not in derived_junctions))
    isolated = []
    for key in sorted(set(_isolated_vertices(original)) | set(_isolated_vertices(current))):
        isolated.append({'vertex_id':key,'disposition':'exact_identity' if key in identical_vertices else 'refused_coordinate_or_identity_change',
                         'source':vertex_before.get(key),'current':vertex_after.get(key)})
    occurrence_maps=[]
    current_occurrences=current['structural']['face_uses']
    child_map=dict(descendants)
    occurrences_by_owner = {}
    for use in current_occurrences:
        for root in roots_by_face.get(use['face_id'],()):
            occurrences_by_owner.setdefault((root,use['sheet_id'],use['orientation']),[]).append(use)
    for source in original['structural']['face_uses']:
        check()
        children=child_map.get(source['face_id'],())
        matches=occurrences_by_owner.get((source['face_id'],source['sheet_id'],source['orientation']),[])
        ok=(len(matches)==len(children) and {row['face_id'] for row in matches}==set(children)
            and all(row.get('metadata')==source.get('metadata') and
                    orientation_relations.get((source['face_id'],row['face_id']),False) for row in matches))
        occurrence_maps.append({'source_face_use_id':source['id'],'source_face_id':source['face_id'],
            'source_sheet_id':source['sheet_id'],'source_orientation':source['orientation'],
            'current_face_use_ids':[r['id'] for r in matches],
            'native_orientation_preserved':all(orientation_relations.get((source['face_id'],r['face_id']),False) for r in matches),
            'disposition':'preserved_occurrence_orientation' if ok else 'refused_occurrence_semantics'})
    claimed_uses={use for row in occurrence_maps for use in row['current_face_use_ids']}
    original_occurrence_roots={use['face_id'] for use in original['structural']['face_uses']}
    extra_occurrences=[]
    for use in current_occurrences:
        if use['id'] in claimed_uses:
            continue
        roots=roots_by_face.get(use['face_id'],set())
        peers=uses_by_face.get(use['face_id'],())
        derived=(bool(roots) and not roots & original_occurrence_roots and len(peers)==1 and not use['metadata']
                 and all(orientation_relations.get((root,use['face_id']),False) for root in roots))
        extra_occurrences.append({'current_face_use_id':use['id'],
            'disposition':'derived_preparation_occurrence' if derived else 'refused_untracked_occurrence'})
    ownership=[]
    for kind,excluded in (('parts',{'sheet_ids','member_ids'}),('sheets',{'face_use_ids'})):
        before={r['id']:r for r in original['structural'][kind]}
        after={r['id']:r for r in current['structural'][kind]}
        for key in before.keys() | after.keys():
            check()
            source,target=before.get(key),after.get(key)
            if source is not None and target is not None:
                ok=({k:v for k,v in source.items() if k not in excluded}==
                    {k:v for k,v in target.items() if k not in excluded})
                status='preserved_owner_payload' if ok else 'refused_owner_reassignment_or_payload'
            elif source is None and target is not None:
                owned_uses=([u for u in current_occurrences if u['sheet_id']==key] if kind=='sheets' else
                    [u for u in current_occurrences if u['sheet_id'] in target['sheet_ids']])
                derived=bool(owned_uses) and all(u['id'] in {r['current_face_use_id'] for r in extra_occurrences
                    if r['disposition']=='derived_preparation_occurrence'} for u in owned_uses)
                status='derived_preparation_owner' if derived and not target['metadata'] else 'refused_current_only_owner'
            else:
                status='refused_deleted_owner'
            ownership.append({'kind':kind,'id':key,'source':source,'current':target,'disposition':status})
    coedge_dispositions=[]
    source_coedges=original['structural']['coedges']
    source_uses_by_id={r['id']:r for r in original['structural']['face_uses']}
    current_uses_by_id={r['id']:r for r in current_occurrences}
    source_coedges_by_owner={}
    maps_by_edge={}
    for source in source_coedges:
        use=source_uses_by_id[source['face_use_id']]
        source_coedges_by_owner.setdefault((use['face_id'],use['sheet_id'],source['edge_id']),[]).append(source)
    for mapping in edge_maps:
        maps_by_edge.setdefault(mapping['edge_id'],[]).append(mapping)
    coedge_source_spans={}
    for coedge in current['structural']['coedges']:
        check()
        use=current_uses_by_id[coedge['face_use_id']]
        roots=roots_by_face.get(use['face_id'],set())
        candidates=[]
        for mapping in maps_by_edge.get(coedge['edge_id'],()):
            if (mapping['classification'] not in ('exact','bounded') or
                mapping['ancestor_source_checksum']!=original['checksum']['value'] or
                mapping['ancestor_revision']!=original['revision'] or
                mapping['ancestor_model_id']!=original['model_id']):
                continue
            for root in roots:
                for source in source_coedges_by_owner.get((root,use['sheet_id'],mapping['ancestor_edge_id']),()):
                    a,b=map(lambda x:F(*x),mapping['interval'])
                    mapped_forward=(coedge['orientation']=='forward')==(b>a)
                    if (mapped_forward==(source['orientation']=='forward') and source['metadata']==coedge['metadata']
                            and orientation_relations.get((root,use['face_id']),False)):
                        candidates.append((source['id'],mapping['interval']))
        source_ids={key for key,_ in candidates}
        source_carrier=any(source_coedges_by_owner.get((root,use['sheet_id'],mapping['ancestor_edge_id']))
            for mapping in maps_by_edge.get(coedge['edge_id'],()) for root in roots)
        if len(candidates)==1:
            key,interval=candidates[0]
            coedge_source_spans.setdefault(key,[]).append(interval)
        status='preserved_coedge_owner_orientation' if len(candidates)==1 else (
            'derived_preparation_coedge' if not source_carrier and not coedge['metadata'] and
            (not roots & original_occurrence_roots or len([f for f,_,_ in incidence.get(coedge['edge_id'],())
                if f in selected_faces])>1) else 'refused_coedge_semantics')
        coedge_dispositions.append({'current_coedge_id':coedge['id'],'source_coedge_ids':sorted(source_ids),
            'current_face_use_id':use['id'],'disposition':status})
    coedge_coverage={}
    for source in source_coedges:
        spans=sorted(tuple(sorted(F(*x) for x in interval)) for interval in coedge_source_spans.get(source['id'],()))
        coedge_coverage[source['id']]=bool(spans and spans[0][0]==0 and spans[-1][1]==1
            and all(a[1]==b[0] for a,b in zip(spans,spans[1:])))
    occurrence_by_source={r['source_face_use_id']:r for r in occurrence_maps}
    coedges_by_source_use={}
    for source in source_coedges:
        coedges_by_source_use.setdefault(source['face_use_id'],[]).append(source['id'])
    sheet_ok={}
    for row in ownership:
        if row['kind']!='sheets' or row['source'] is None:
            continue
        sheet=row['source']
        expected={key for use in sheet['face_use_ids'] for key in occurrence_by_source.get(use,{}).get('current_face_use_ids',())}
        sheet_ok[row['id']]=(row['disposition']=='preserved_owner_payload' and row['current'] is not None
            and expected==set(row['current']['face_use_ids'])
            and all(occurrence_by_source.get(use,{}).get('disposition')=='preserved_occurrence_orientation'
                and all(coedge_coverage.get(key,False) for key in coedges_by_source_use.get(use,()))
                for use in sheet['face_use_ids']))
    for row in dispositions:
        if row['kind']!='attachments':
            continue
        source=before_attachments.get(row['id']);target=after_attachments.get(row['id'])
        if source is not None and source==target and source['source_kind']=='sheet':
            ok=(sheet_ok.get(source['source_id'],False) and (source['member_id'] is None or member_ok.get(source['member_id'],False))
                and source['target_id'] in {'edge':identical_edges,'face':identical_faces}.get(source['target_kind'],set()))
            attachment_ok[row['id']]=ok
            row.update(geometry_semantic_disposition='qualified_native_owner_semantics' if ok else 'refused_owner_semantics_unresolved',
                       semantic_mapping_qualified=ok,parameter_remapping_qualified=ok)
    for row in dispositions:
        if row['kind']!='junctions' or row['id'] in derived_junctions:
            continue
        source=before_junctions.get(row['id']);target=after_junctions.get(row['id'])
        ok=(source is not None and source==target and all(member_ok.get(use['member_id'],False) for use in source['member_uses'])
            and all(sheet_ok.get(key,False) for key in source['sheet_ids'])
            and all(attachment_ok.get(key,False) for key in source['attachment_ids']))
        row.update(geometry_semantic_disposition='qualified_native_owner_semantics' if ok else 'refused_owner_semantics_unresolved',
                   semantic_mapping_qualified=ok,parameter_remapping_qualified=ok)
    attachment_maps=[];junction_maps=[];untracked_attachments=[]
    if request_proof is not None:
        from .native_attachment_maps import map_native_attachment_references, map_native_junction_references
        attachment_maps,current_sources,untracked_attachments,context=map_native_attachment_references(
            original,current,descendants,edge_maps,use_maps,member_ok,sheet_ok,source_native,current_native,material_faces,request_proof,
            {r['id']:r['disposition']=='preserved_owner_payload' for r in ownership if r['kind']=='parts'},replacement_history,attachment_source_ids)
        by_source={r['source_attachment_id']:r for r in attachment_maps}
        for row in dispositions:
            if row['kind']!='attachments':continue
            source_key=(row['id'] if row['id'] in by_source else current_sources.get(row['id']))
            if source_key is not None:
                mapping=by_source[source_key];ok=mapping['classification']!='refused'
                attachment_ok[row['id']]=ok
                literal=(before_attachments.get(row['id'])==after_attachments.get(row['id']))
                label=('qualified_native_owner_semantics' if literal else 'qualified_captured_real_relation') if ok else 'refused_attachment_relation'
                row.update(geometry_semantic_disposition=label,
                           semantic_mapping_qualified=ok,parameter_remapping_qualified=ok,
                           source_attachment_id=source_key)
        junction_maps=map_native_junction_references(original,current,attachment_maps,member_ok,sheet_ok,context)
        junction_by_source={r['source_junction_id']:r for r in junction_maps}
        for row in dispositions:
            if row['kind']=='junctions' and row['id'] in junction_by_source:
                mapping=junction_by_source[row['id']];ok=mapping['classification']!='refused'
                literal=(before_junctions.get(row['id'])==after_junctions.get(row['id']))
                label=('qualified_native_owner_semantics' if literal else 'qualified_captured_real_relation') if ok else 'refused_junction_relation'
                row.update(geometry_semantic_disposition=label,
                           semantic_mapping_qualified=ok,parameter_remapping_qualified=ok)
    inventory = {'original':_document_inventory(original),'current':_document_inventory(current),
        'attachment_native_maps':attachment_maps,'junction_native_maps':junction_maps,
        'untracked_attachment_ids':untracked_attachments,
        'record_dispositions':dispositions,'member_native_maps':use_maps,'source_use_coverage':source_coverage,
        'isolated_vertex_dispositions':isolated,'traces':traces,'outside_root_ids':sorted(outside),
        'all_face_use_occurrences':{'original':original['structural']['face_uses'],'current':current['structural']['face_uses']},
        'face_use_dispositions':occurrence_maps,
        'current_extra_face_use_dispositions':extra_occurrences,
        'part_sheet_dispositions':ownership,'coedge_dispositions':coedge_dispositions,
        'source_coedge_coverage':coedge_coverage,'selected_dependency_vertex_ids':sorted(selected_vertices),
        'dependency_closure':sorted(reachable),
        'opaque_semantics_disposition':'unknown_consumer_semantics_retained',
        'external_reference_obligation':'consumer must inventory and adjudicate external references',
        'external_reference_scope_qualified':False}
    return inventory,tuple(sorted(outside)),(not untracked_attachments and all(r['semantic_mapping_qualified'] for r in dispositions)
        and all(row['disposition']=='exact_identity' for row in isolated)
        and all(row['disposition']=='preserved_occurrence_orientation' for row in occurrence_maps)
        and all(coedge_coverage.values())
        and all(not row['disposition'].startswith('refused') for row in ownership+extra_occurrences+coedge_dispositions))


def _trace_inventory(current, correspondences, roots_by_face, selected, check):
    faces = {row['id']: row for row in current['faces']}
    edges = {row['id']: row for row in current['edges']}
    vertices = {row['id']: row for row in current['vertices']}
    structural = current['structural']
    uses = {row['id']: row for row in structural['face_uses']}
    sheets = {row['id']: row for row in structural['sheets']}
    coedges = {row['id']: row for row in structural['coedges']}
    face_incidence, coedge_incidence, uses_by_face = {}, {}, {}
    for face in faces.values():
        check()
        for loop in (face['loop'], *face['holes']):
            for edge, forward in loop:
                face_incidence.setdefault(edge, []).append((face['id'], forward))
    for row in coedges.values():
        check()
        coedge_incidence.setdefault(row['edge_id'], []).append(row['id'])
    for key, use in uses.items():
        check()
        uses_by_face.setdefault(use['face_id'], []).append(key)
    result, outside = [], set()
    for correspondence in correspondences:
        root = correspondence.authored_definition.face_id
        exterior = {edge for loop in correspondence.exterior_loops
                    for _, _, current_edges in loop for edge in current_edges}
        interior = dict(correspondence.interior_incidence)
        if exterior & interior.keys():
            raise GeometryError('authored constraint scope has ambiguous root trace incidence')
        for edge_id in sorted(exterior | interior.keys()):
            check()
            adjacent = tuple(sorted(face_incidence.get(edge_id, ())))
            if not adjacent:
                raise GeometryError('authored constraint scope trace has no literal adjacent faces')
            root_incidence = tuple(row for row in adjacent if row[0] in correspondence.descendants)
            if edge_id in interior and root_incidence != tuple(sorted(interior[edge_id])):
                raise GeometryError('authored constraint scope interior incidence changed')
            if edge_id in exterior and len(root_incidence) != 1:
                raise GeometryError('authored constraint scope exterior incidence changed')
            adjacent_faces = sorted({face for face, _ in adjacent})
            neighbour_roots = set()
            for face in adjacent_faces:
                if face not in roots_by_face:
                    raise GeometryError('authored constraint scope has an unmapped adjacent face')
                neighbour_roots.update(roots_by_face[face])
            outside_roots = neighbour_roots - selected
            outside.update(outside_roots)
            adjacent_uses = sorted(key for face in adjacent_faces
                                   for key in uses_by_face.get(face, ()))
            adjacent_coedges = sorted(coedge_incidence.get(edge_id, ()))
            adjacent_sheets = sorted({uses[key]['sheet_id'] for key in adjacent_uses})
            edge = edges[edge_id]
            result.append({
                'authored_root_id': root, 'current_edge_id': edge_id,
                'trace_kind': 'exterior' if edge_id in exterior else 'paired_interior',
                'root_child_incidence': root_incidence, 'adjacent_face_incidence': adjacent,
                'endpoint_ids': (edge['start'], edge['end']),
                'endpoint_records': [vertices[edge['start']], vertices[edge['end']]],
                'edge_definition': edge,
                'adjacent_face_ids': adjacent_faces,
                'adjacent_faces': [faces[key] for key in adjacent_faces],
                'adjacent_face_use_ids': adjacent_uses,
                'adjacent_face_uses': [uses[key] for key in adjacent_uses],
                'adjacent_sheet_ids': adjacent_sheets,
                'adjacent_sheets': [sheets[key] for key in adjacent_sheets],
                'adjacent_coedge_ids': adjacent_coedges,
                'adjacent_coedges': [coedges[key] for key in adjacent_coedges],
                'adjacent_authored_root_ids': sorted(neighbour_roots),
                'outside_authored_root_ids': sorted(outside_roots),
                'semantic_mapping_qualified': False, 'parameter_remapping_qualified': False,
            })
    return result, tuple(sorted(outside))


def query_prepared_authored_constraint_scope(model, authored_face_ids, *, expected_revision=None,
                                             cancellation_check=None):
    """Inventory all typed records and literal selected-root trace neighbours.

    Whole original/current Member, MemberEdgeUse, Attachment and Junction tables
    prevent incoming-reference omission. Dispositions compare literal payloads,
    not semantic meaning or parameter carriers. Outside roots are reported,
    never silently selected or granted shared-publication permission.
    """
    scope = query_prepared_model_scope(model)
    try:
        identifiers = tuple(_integer(value, 'authored face IDs') for value in tuple(authored_face_ids))
    except TypeError as error:
        raise GeometryError('authored constraint scope requires iterable authored face IDs') from error
    revision = scope.face_preimages.revision
    if expected_revision is not None and _integer(expected_revision, 'revision') != revision:
        raise GeometryError('authored constraint scope revision is stale')
    validate_prepared_model_scope_binding(model, scope)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise GeometryError('authored constraint scope requires nonempty unique authored face IDs')
    selected = tuple(sorted(identifiers))
    if not set(selected) <= set(scope.face_preimages.authored_face_ids):
        raise GeometryError('authored constraint scope has missing authored face IDs')
    original, current = scope.authored_document, scope.current_document

    def check():
        _check(cancellation_check)

    check()
    correspondences = []
    for root in selected:
        check()
        validate_prepared_model_scope_binding(model, scope)
        correspondence = query_prepared_authored_boundary_correspondence(model, root,
            expected_revision=revision, cancellation_check=cancellation_check)
        if correspondence.face_preimages != scope.face_preimages:
            raise GeometryError('authored constraint scope boundaries bind a different preparation')
        correspondences.append(correspondence)
    roots_by_face = {}
    for root, descendants in scope.face_preimages.face_descendants:
        for face in descendants:
            roots_by_face.setdefault(face, set()).add(root)
    traces, outside = _trace_inventory(current, correspondences, roots_by_face, set(selected), check)
    current_faces = tuple(sorted({face for item in correspondences for face in item.descendants}))
    inventory = {'original': _document_inventory(original), 'current': _document_inventory(current),
                 'record_dispositions': _dispositions(original, current), 'traces': traces,
                 'external_reference_scope_qualified': False,
                 'semantic_mapping_qualified': False, 'parameter_remapping_qualified': False}
    result = PreparedAuthoredConstraintScope(scope, selected, current_faces, outside,
                                            tuple(correspondences), _encode(inventory))
    check()
    # Final owner guards are callback-free; no caller action follows them.
    for correspondence in correspondences:
        validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    validate_prepared_model_scope_binding(model, scope)
    return result


def validate_prepared_authored_constraint_scope_binding(model, receipt, *, cancellation_check=None):
    """Recompute the owner inventory; forged omissions cannot inherit freshness."""
    if (type(receipt) is not PreparedAuthoredConstraintScope or
            type(receipt.scope) is not PreparedModelScope or type(receipt.inventory_json) is not str or
            any(type(getattr(receipt, field)) is not tuple or
                any(type(key) is not int for key in getattr(receipt, field))
                for field in ('selected_root_ids', 'current_face_ids', 'outside_root_ids')) or
            type(receipt.boundary_correspondences) is not tuple or
            any(type(item) is not AuthoredBoundaryCorrespondence
                for item in receipt.boundary_correspondences)):
        raise GeometryError('authored constraint scope requires a plain immutable owner receipt')
    validate_prepared_model_scope_binding(model, receipt.scope)
    signature = _signature(receipt)
    expected = query_prepared_authored_constraint_scope(model, receipt.selected_root_ids,
        expected_revision=receipt.scope.face_preimages.revision, cancellation_check=cancellation_check)
    if _signature(expected) != signature or _signature(receipt) != signature:
        raise GeometryError('authored constraint scope definition binding changed')
    validate_prepared_model_scope_binding(model, receipt.scope)
