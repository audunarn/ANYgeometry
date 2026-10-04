"""Complete current dependencies of a restricted prepared Sheet joint component.

This is reference/occurrence visibility, not parameter remapping, semantic load
equivalence, geometric partition completeness or mesh publication permission.
"""
from dataclasses import dataclass
import json
from numbers import Integral

import numpy as np

from .arrangement_geometry import LinePath
from .authored_child_coverage import _support_correspondence
from .definition_binding import definition_checksum
from .errors import GeometryError
from .joint_edges import query_joint_edge
from .material_cell_coverage import _chart_polynomial, _frame
from .prepared_model_scope import PreparedModelScope, query_prepared_model_scope, validate_prepared_model_scope_binding
from .serialization import _decode_surface, _QUADRIC_VERSION
from .surfaces import Plane


@dataclass(frozen=True, slots=True)
class PreparedSheetJointComponent:
    scope: object
    joint_edge_id: int
    sheet_ids: tuple
    part_ids: tuple
    authored_face_ids: tuple
    current_face_ids: tuple
    joint_edge_ids: tuple
    junction_ids: tuple
    attachment_ids: tuple
    # (Sheet ID, authored root ID, original FaceUse ID, current FaceUse IDs).
    occurrence_correspondence: tuple
    source_records_json: str
    current_records_json: str
    unqualified_semantics: tuple
    occurrence_mapping_qualified: bool = True
    semantic_mapping_qualified: bool = False
    publication_qualified: bool = False
    # IDs in the fresh working-model source namespace, not Project IDs.
    # Only literal unchanged joint relations are qualified; no split remap.
    preserved_joint_attachment_ids: tuple = ()
    preserved_joint_junction_ids: tuple = ()

    @property
    def source_records(self):
        return json.loads(self.source_records_json)

    @property
    def current_records(self):
        return json.loads(self.current_records_json)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GeometryError(f'{name} must be an integer')
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f'{name} must be a usable integer') from error


def _signature(value):
    try:
        return definition_checksum(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('prepared Sheet joint component definition is malformed') from error


def _index(document):
    result = {kind: {row['id']: row for row in document[kind]} for kind in ('vertices','edges','faces')}
    result.update({kind: {row['id']: row for row in values} for kind,values in document['structural'].items()})
    return result


def _literal_users(data, edge):
    coedges = tuple(sorted(row['id'] for row in data['coedges'].values() if row['edge_id']==edge))
    uses = tuple(sorted({data['coedges'][key]['face_use_id'] for key in coedges}))
    sheets = tuple(sorted({data['face_uses'][key]['sheet_id'] for key in uses}))
    return coedges, uses, sheets


def _live_incidence(model, data, edge):
    return (_literal_users(data,edge),
            (tuple(model.coedges_using_edge(edge)),tuple(model.face_uses_using_edge(edge)),
             tuple(model.sheets_using_edge(edge))))


def _live_joint(model, data, edge):
    joint = query_joint_edge(model,edge)
    return (*_live_incidence(model,data,edge),
            tuple(row.id for row in joint.sheets),tuple(row.id for row in joint.junctions))


def _crosscheck(snapshot):
    literal, derived, owners, junctions = snapshot
    if literal != derived or owners != literal[2]:
        raise GeometryError('prepared Sheet joint component literal occurrence/index mismatch')
    if len(owners)<2 or not junctions:
        raise GeometryError('prepared Sheet joint component requires a declared multi-Sheet joint')
    return set(owners)


def _dependencies(data, sheets):
    parts = {data['sheets'][key]['part_id'] for key in sheets}
    uses = {key for sheet in sheets for key in data['sheets'][sheet]['face_use_ids']}
    for sheet in sheets:
        actual = {row['id'] for row in data['face_uses'].values() if row['sheet_id']==sheet}
        if actual != set(data['sheets'][sheet]['face_use_ids']):
            raise GeometryError('prepared Sheet joint component has inconsistent Sheet membership')
    faces = {data['face_uses'][key]['face_id'] for key in uses}
    coedges = {key for use in uses for loop in data['face_uses'][use]['loops'] for key in loop}
    actual = {row['id'] for row in data['coedges'].values() if row['face_use_id'] in uses}
    if actual != coedges:
        raise GeometryError('prepared Sheet joint component has inconsistent coedge membership')
    edges = {edge for face in faces for loop in (data['faces'][face]['loop'],*data['faces'][face]['holes'])
             for edge,_ in loop}
    if edges != {data['coedges'][key]['edge_id'] for key in coedges}:
        raise GeometryError('prepared Sheet joint component occurrence geometry mismatch')
    vertices = {value for edge in edges for value in (data['edges'][edge]['start'],data['edges'][edge]['end'])}
    return dict(parts=parts,sheets=set(sheets),face_uses=uses,coedges=coedges,faces=faces,edges=edges,vertices=vertices)


def _touch_attachment(row, deps, near_edges, attachments=(), junctions=()):
    names = {'sheet':deps['sheets'],'part':deps['parts'],'face':deps['faces'],
             'edge':near_edges,'vertex':deps['vertices'], 'face_use':deps['face_uses'],
             'coedge':deps['coedges'],'attachment':attachments,'junction':junctions}
    return (row['target_id'] in names.get(row['target_kind'],()) or
            row['source_id'] in names.get(row['source_kind'],()) or
            row['sheet_id'] in deps['sheets'] or row['part_id'] in deps['parts'] or
            any(key in names.get(kind,()) for kind,key in row['lineage']))


def _qualify_joint(data, row, observed):
    if row['kind']!='sheet_joint' or row['connection_intent']!='connect' or row['member_uses']:
        raise GeometryError('prepared Sheet joint component has unsupported Junction/member semantics')
    attachments = [data['attachments'][key] for key in row['attachment_ids']]
    if not attachments:
        raise GeometryError('prepared Sheet joint component has an empty joint declaration')
    edge = attachments[0]['target_id']
    if edge not in observed:
        raise GeometryError('prepared Sheet joint component has an unsupported joint target')
    owners = _crosscheck(observed[edge])
    if set(row['sheet_ids'])!=owners or row['id'] not in observed[edge][3]:
        raise GeometryError('prepared Sheet joint component lacks complete declared owner coverage')
    for attachment in attachments:
        if attachment['lineage']:
            raise GeometryError('prepared Sheet joint component has unqualified Attachment lineage')
        if (attachment['kind']!='sheet_on_joint' or attachment['target_kind']!='edge' or
                attachment['target_id']!=edge or attachment['source_kind']!='sheet' or
                attachment['source_id'] not in owners or attachment['member_id'] is not None or
                attachment['connection_intent']!='connect' or attachment['evidence']!='exact' or
                attachment['target_parameters']!=[[0.0,1.0]] or attachment['member_range']!=[0.0,1.0] or
                attachment['max_residual']>attachment['tolerance_used'] or
                attachment['sheet_id'] not in (None,attachment['source_id']) or
                attachment['part_id'] not in (None,data['sheets'][attachment['source_id']]['part_id'])):
            raise GeometryError('prepared Sheet joint component has unsupported Attachment semantics')
    if sorted(a['source_id'] for a in attachments)!=sorted(owners):
        raise GeometryError('prepared Sheet joint component requires one complete attachment per owner')
    return edge,owners,{a['id'] for a in attachments}


def _refuse_members(data,deps,near_edges):
    for row in data['member_edge_uses'].values():
        if row['edge_id'] in near_edges:
            raise GeometryError('prepared Sheet joint component touches unsupported Member semantics')
    for row in data['members'].values():
        reference=row['orientation_reference']
        names={'vertex':'vertices','edge':'edges','face':'faces','sheet':'sheets','part':'parts'}
        if row['part_id'] in deps['parts'] or (reference and reference[1] in deps.get(names.get(reference[0],''),())):
            raise GeometryError('prepared Sheet joint component touches unsupported Member semantics')
    if any(data['parts'][key]['member_ids'] for key in deps['parts']):
        raise GeometryError('prepared Sheet joint component Part carries unsupported Members')


def _plane_geometry(data,faces,check):
    supports={}
    for key in sorted(faces):
        check()
        face=data['faces'][key]
        if face['parameterization'] is not None:
            raise GeometryError('prepared Sheet joint component does not qualify parameterizations')
        support=_decode_surface(face['surface'],strict=True,schema_version=_QUADRIC_VERSION)
        if type(support) is not Plane:
            raise GeometryError('prepared Sheet joint component requires original/current Plane faces')
        frame=_frame(support)
        for loop in (face['loop'],*face['holes']):
            for edge_id,_ in loop:
                check()
                edge=data['edges'][edge_id]
                if edge['curve']!={'type':'straight'}:
                    raise GeometryError('prepared Sheet joint component requires literal Straight boundaries')
                _chart_polynomial(frame,LinePath(*(tuple(data['vertices'][v]['position'])
                                      for v in (edge['start'],edge['end']))),check)
        supports[key]=support
    return supports


def _records(document,data,deps,attachments,junctions):
    result={kind:[data[kind][key] for key in sorted(values)] for kind,values in deps.items()}
    result['attachments']=[data['attachments'][key] for key in sorted(attachments)]
    result['junctions']=[data['junctions'][key] for key in sorted(junctions)]
    # Vertex has an irregular plural; preserve exact geometric reference keys.
    refs={('vertex' if kind=='vertices' else kind[:-1],key)
          for kind in ('vertices','edges','faces') for key in deps[kind]}
    result['groups']={name:[ref for ref in rows if tuple(ref) in refs]
                      for name,rows in document['groups'].items() if any(tuple(ref) in refs for ref in rows)}
    result['tags']=[row for row in document['tags'] if tuple(row['entity']) in refs]
    return json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)


def _qualify_preserved_source_joints(source, current, source_attachments,
                                    source_junctions, attachments, junctions, check):
    """Prove unchanged source relations against already qualified CURRENT joints.

    Equality of IDs alone cannot prove an unchanged parameter carrier. Retain
    the entire attachment/junction payload, edge definition and endpoint
    records. Fragmented/rebound carriers and unsupported relation types refuse.
    """
    for key in sorted(source_attachments):
        check()
        original = source['attachments'][key]
        if key not in attachments or original != current['attachments'].get(key):
            raise GeometryError('prepared Sheet joint component has unqualified original Attachment remapping')
        edge = original['target_id']
        old_edge = source['edges'].get(edge)
        if old_edge is None or old_edge != current['edges'].get(edge):
            raise GeometryError('prepared Sheet joint component original joint carrier changed')
        for vertex in (old_edge['start'], old_edge['end']):
            if (source['vertices'].get(vertex) is None or
                    source['vertices'][vertex] != current['vertices'].get(vertex)):
                raise GeometryError('prepared Sheet joint component original joint endpoint changed')
    for key in sorted(source_junctions):
        check()
        if key not in junctions or source['junctions'][key] != current['junctions'].get(key):
            raise GeometryError('prepared Sheet joint component has unqualified original Junction remapping')


def _source_relations(source, deps, near_edges, check):
    """Include original incoming attachment/junction references to a fixed point."""
    attachments, junctions = set(), set()
    while True:
        before = (len(attachments), len(junctions))
        for key, row in source['attachments'].items():
            check()
            if _touch_attachment(row, deps, near_edges, attachments, junctions):
                attachments.add(key)
        for key, row in source['junctions'].items():
            check()
            if (set(row['sheet_ids']) & deps['sheets'] or
                    set(row['attachment_ids']) & attachments):
                junctions.add(key)
        if before == (len(attachments), len(junctions)):
            return attachments, junctions


def query_prepared_sheet_joint_component(model,current_joint_edge_id,*,expected_revision=None,
                                         cancellation_check=None):
    """Capture the entire CURRENT declared Plane/Straight Sheet-joint component.

    Stable source Sheet/Part IDs and unique root/Sheet occurrences are required.
    Raw definitions remain visible; arbitrary property/reference semantics are
    not remapped. Batch-created source-less structural owners refuse.
    """
    return _query_component(model, current_joint_edge_id,
        expected_revision=expected_revision, cancellation_check=cancellation_check)


def _query_component(model,current_joint_edge_id,*,expected_revision=None,
                     cancellation_check=None, relation_factory=None, result_factory=None):
    """Shared owner engine; additional relations need a separate qualified policy."""
    scope=query_prepared_model_scope(model)
    identifier=_integer(current_joint_edge_id,'joint edge ID')
    if expected_revision is not None and _integer(expected_revision,'expected revision')!=scope.face_preimages.revision:
        raise GeometryError('prepared Sheet joint component revision is stale')
    validate_prepared_model_scope_binding(model,scope)
    original,current=scope.authored_document,scope.current_document
    source,data=_index(original),_index(current)
    relations = (None if relation_factory is None else
                 relation_factory(model, scope, source, data, cancellation_check))
    occurrence_edges={row['edge_id'] for row in data['coedges'].values()}
    incidences={edge:_live_incidence(model,data,edge) for edge in sorted(occurrence_edges)}
    candidates={identifier}|{edge for edge,(literal,_) in incidences.items() if len(literal[2])>=2}|\
        {row['target_id'] for row in data['attachments'].values()
                             if row['kind']=='sheet_on_joint' and row['target_kind']=='edge'}
    observed={key:_live_joint(model,data,key) for key in sorted(candidates)}
    validate_prepared_model_scope_binding(model,scope)
    sheets=_crosscheck(observed[identifier])

    def check():
        if cancellation_check is not None and cancellation_check('prepared Sheet joint component'):
            raise GeometryError('prepared Sheet joint component cancelled')

    junctions,attachments,joint_edges,inspected=set(),set(),set(),set()
    while True:
        check()
        deps=_dependencies(data,sheets)
        incidence_junctions=set()
        for edge in sorted(deps['edges']):
            check()
            inspected.add(edge)
            literal,derived=incidences[edge]
            if literal!=derived:
                raise GeometryError('prepared Sheet joint component literal occurrence/index mismatch')
            if len(literal[2])>=2:
                # A shared edge is an explicit incoming occurrence. It must
                # have complete qualified declaration, even if no Sheet's
                # junction index would otherwise lead us to its other owners.
                _crosscheck(observed[edge])
                incidence_junctions.update(observed[edge][3])
        near_edges={key for key,row in data['edges'].items()
                    if key in deps['edges'] or row['start'] in deps['vertices'] or row['end'] in deps['vertices']}
        if relations is None:
            _refuse_members(data,deps,near_edges)
        else:
            relations.validate_dependencies(data, deps, original=False)
        touching={key for key,row in data['attachments'].items()
                  if _touch_attachment(row,deps,near_edges,attachments,junctions)}
        selected={key for key,row in data['junctions'].items()
                  if set(row['sheet_ids'])&sheets or set(row['attachment_ids'])&touching}|incidence_junctions
        expanded=set(sheets)
        for key in sorted(selected):
            check()
            edge,owners,links=_qualify_joint(data,data['junctions'][key],observed)
            joint_edges.add(edge);attachments.update(links);junctions.add(key);expanded.update(owners)
        touching.update(key for key,row in data['attachments'].items()
                        if _touch_attachment(row,deps,near_edges,attachments,junctions))
        extra = set() if relations is None else set(relations.attachment_ids)
        if touching-attachments-extra:
            raise GeometryError('prepared Sheet joint component touches unsupported/uncontained Attachments')
        if expanded==sheets:break
        sheets=expanded
    if identifier not in joint_edges:
        raise GeometryError('prepared Sheet joint component lacks its requested declaration')
    parts=deps['parts']
    for key in sorted(sheets):
        check()
        if key not in source['sheets']:
            raise GeometryError('prepared Sheet joint component has a batch-created source-less Sheet')
        if {k:v for k,v in data['sheets'][key].items() if k!='face_use_ids'}!=\
                {k:v for k,v in source['sheets'][key].items() if k!='face_use_ids'}:
            raise GeometryError('prepared Sheet joint component Sheet semantic fields changed')
    for key in sorted(parts):
        if key not in source['parts']:
            raise GeometryError('prepared Sheet joint component has a source-less Part')
        if {k:v for k,v in data['parts'][key].items() if k not in ('sheet_ids','member_ids')}!=\
                {k:v for k,v in source['parts'][key].items() if k not in ('sheet_ids','member_ids')}:
            raise GeometryError('prepared Sheet joint component Part semantic fields changed')
    descendants=dict(scope.face_preimages.face_descendants)
    roots={child:root for root,children in descendants.items() for child in children}
    occurrence=[];authored=set()
    for sheet in sorted(sheets):
        old_uses=[source['face_uses'][key] for key in source['sheets'][sheet]['face_use_ids']]
        new_uses=[data['face_uses'][key] for key in data['sheets'][sheet]['face_use_ids']]
        old_roots={row['face_id'] for row in old_uses}
        if old_roots!={roots[row['face_id']] for row in new_uses}:
            raise GeometryError('prepared Sheet joint component Sheet root membership changed')
        for root in sorted(old_roots):
            check()
            originals=[row for row in old_uses if row['face_id']==root]
            uses=[row for row in new_uses if roots[row['face_id']]==root]
            if len(originals)!=1 or sorted(row['face_id'] for row in uses)!=list(descendants[root]):
                raise GeometryError('prepared Sheet joint component lacks unique complete root/Sheet occurrences')
            if any(row['orientation']!=originals[0]['orientation'] or row['metadata']!=originals[0]['metadata']
                   for row in uses):
                raise GeometryError('prepared Sheet joint component occurrence properties changed')
            occurrence.append((sheet,root,originals[0]['id'],tuple(sorted(row['id'] for row in uses))))
            authored.add(root)
    source_deps=_dependencies(source,sheets)
    source_near={key for key,row in source['edges'].items() if key in source_deps['edges'] or
                 row['start'] in source_deps['vertices'] or row['end'] in source_deps['vertices']}
    if relations is None:
        _refuse_members(source,source_deps,source_near)
    else:
        relations.validate_dependencies(source, source_deps, original=True)
    source_attachments,source_junctions=_source_relations(source,source_deps,source_near,check)
    if relations is not None:
        # These exact point relations were independently qualified by the new
        # policy. They cannot inherit the legacy unchanged-joint proof.
        source_attachments -= set(relations.attachment_ids)
    # Current classification alone does not qualify original relations. Only
    # literal preservation on the same unchanged carrier is established here.
    _qualify_preserved_source_joints(source,data,source_attachments,source_junctions,
                                    attachments,junctions,check)
    original_supports=_plane_geometry(source,authored,check)
    current_supports=_plane_geometry(data,deps['faces'],check)
    for face,support in current_supports.items():
        check()
        _support_correspondence(original_supports[roots[face]],support,np.empty((0,3,2)))
    factory = PreparedSheetJointComponent if result_factory is None else result_factory
    all_attachments = attachments if relations is None else attachments | set(relations.attachment_ids)
    all_source_attachments = (source_attachments if relations is None else
                              source_attachments | set(relations.attachment_ids))
    original_records = _records(original,source,source_deps,all_source_attachments,source_junctions)
    current_records = _records(current,data,deps,all_attachments,junctions)
    if relations is not None:
        original_records = relations.complete_records(original_records, source)
        current_records = relations.complete_records(current_records, data)
        parts = parts | relations.part_ids(data)
    result=factory(scope,identifier,tuple(sorted(sheets)),tuple(sorted(parts)),
        tuple(sorted(authored)),tuple(sorted(deps['faces'])),tuple(sorted(joint_edges)),
        tuple(sorted(junctions)),tuple(sorted(all_attachments)),tuple(occurrence),
        original_records,current_records,
        ('raw metadata/group/tag/feature/extension semantics and parameter remapping',),
        preserved_joint_attachment_ids=tuple(sorted(source_attachments)),
        preserved_joint_junction_ids=tuple(sorted(source_junctions)))
    result_signature = None if relations is None else relations.output_signature(result)
    check()
    validate_prepared_model_scope_binding(model,scope)
    for edge in result.joint_edge_ids:
        if _live_joint(model,data,edge)!=observed[edge]:
            raise GeometryError('prepared Sheet joint component derived declaration changed')
    for edge in sorted(inspected):
        if _live_incidence(model,data,edge)!=incidences[edge]:
            raise GeometryError('prepared Sheet joint component derived occurrence changed')
    if relations is not None:
        relations.validate_final(model, scope, cancellation_check=None)
        if relations.output_signature(result) != result_signature:
            raise GeometryError('prepared member Sheet component output definition changed')
    return result


def validate_prepared_sheet_joint_component_binding(model,receipt,*,cancellation_check=None):
    """Rederive immutable current coverage; no omitted/forged records may pass."""
    if type(receipt) is not PreparedSheetJointComponent:
        raise GeometryError('prepared Sheet joint component requires an owner receipt')
    if type(receipt.scope) is not PreparedModelScope:
        raise GeometryError('prepared Sheet joint component requires an owner scope')
    validate_prepared_model_scope_binding(model,receipt.scope)
    signature=_signature(receipt)
    expected=query_prepared_sheet_joint_component(model,receipt.joint_edge_id,
        expected_revision=receipt.scope.face_preimages.revision,cancellation_check=cancellation_check)
    if _signature(expected)!=signature or _signature(receipt)!=signature:
        raise GeometryError('prepared Sheet joint component definition binding changed')


def validate_prepared_sheet_joint_component_selection(model,receipt,authored_face_ids,*,cancellation_check=None):
    """Require every component root in a selection; extra roots are unqualified here."""
    if type(receipt) is not PreparedSheetJointComponent:
        raise GeometryError('prepared Sheet joint component requires an owner receipt')
    signature=_signature(receipt)
    try:
        identifiers=tuple(_integer(value,'authored face ID') for value in authored_face_ids)
    except TypeError as error:
        raise GeometryError('prepared Sheet joint component requires iterable authored face IDs') from error
    validate_prepared_sheet_joint_component_binding(model,receipt,cancellation_check=cancellation_check)
    if _signature(receipt)!=signature or len(set(identifiers))!=len(identifiers):
        raise GeometryError('prepared Sheet joint component selection/definition changed')
    if not set(identifiers)<=set(receipt.scope.face_preimages.authored_face_ids):
        raise GeometryError('prepared Sheet joint component selection has unknown authored roots')
    if not set(receipt.authored_face_ids)<=set(identifiers):
        raise GeometryError('prepared Sheet joint component selection omits connected authored roots')
