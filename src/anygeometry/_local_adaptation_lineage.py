"""Pre-proof, owner-authenticated effective-chart lineage for local fields.

No cells, material coverage or approximation are certified here. Maps use the existing owner-qualified source authored_query_uv convention;
prepared inputs are effective-native UV, not arbitrary face coordinates.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from weakref import WeakKeyDictionary, ref
from .errors import GeometryError
from .definition_binding import definition_checksum
from .nurbs import NurbsSurface
from .surfaces import Plane
from .prepared_material_source_transfer import _LINKS, _state, _effective, _chart_map

_ISSUED=WeakKeyDictionary()

@dataclass(frozen=True, eq=False)
class LocalAdaptationLineage:
    source_model_id: object
    prepared_model_id: object
    source_revision: int
    prepared_revision: int
    source_checksum: str
    prepared_checksum: str
    # (prepared face, original face, source authored_query_uv from effective-native UV)
    face_maps: tuple
    edge_parameter_maps: tuple


def _rows(source,prepared,faces):
    a,b=_state(source),_state(prepared)
    if source is prepared:
        mapping=tuple((f,(f,)) for f in sorted(source.faces))
        edges=tuple((e,e,Q(0),Q(1)) for e in sorted(source.edges))
    else:
        routes=[r for r in _LINKS.get(prepared,()) if r[0]() is source]
        if len(routes)!=1 or routes[0][1:3]!=(a,b):
            raise GeometryError('local field needs current owner-issued source lineage')
        mapping,edges=routes[0][3:5]
    parents={child:parent for parent,children in mapping for child in children}
    if len(parents)!=sum(len(children) for _,children in mapping):
        raise GeometryError('local field lineage has overlapping parents')
    rows=[]
    for face in faces:
        if face not in parents or face not in prepared.faces:
            raise GeometryError('local field face has no authenticated parent')
        parent=parents[face]
        left,right=_effective(source,parent),_effective(prepared,face)
        if definition_checksum(left)!=definition_checksum(right):
            raise GeometryError('local field descendant changed effective source support')
        matrix=_chart_map(source,parent,right)
        rows.append((face,parent,matrix))
    return a,b,tuple(rows),tuple(edges)


def query_local_adaptation_lineage(source,prepared,faces,*,cancellation_check=None):
    selected=tuple(sorted(set(faces)))
    if not selected: raise GeometryError('local field lineage selection is empty')
    before=_rows(source,prepared,selected)
    if cancellation_check is not None and cancellation_check('local field source lineage'):
        raise GeometryError('local field source lineage cancelled')
    after=_rows(source,prepared,selected)
    if before!=after: raise GeometryError('local field lineage changed during query')
    a,b,rows,edges=after
    result=LocalAdaptationLineage(source.model_id,prepared.model_id,a[0],b[0],a[1],b[1],rows,edges)
    _ISSUED[result]=(ref(source),ref(prepared),after,definition_checksum(result))
    return result


def validate_local_adaptation_lineage(source,prepared,result,*,cancellation_check=None):
    stored=_ISSUED.get(result) if type(result) is LocalAdaptationLineage else None
    if stored is None or stored[0]() is not source or stored[1]() is not prepared:
        raise GeometryError('local field lineage receipt was not issued for these sources')
    if definition_checksum(result)!=stored[3]: raise GeometryError('local field lineage receipt changed')
    if cancellation_check is not None and cancellation_check('local field lineage binding'):
        raise GeometryError('local field lineage binding cancelled')
    if _rows(source,prepared,tuple(row[0] for row in result.face_maps))!=stored[2]:
        raise GeometryError('local field lineage binding is stale')
    if definition_checksum(result)!=stored[3]:
        raise GeometryError('local field lineage receipt changed during callback')
