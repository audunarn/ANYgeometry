"""Production closed-prism contract tests, with independent geometric oracles."""
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, EntityHandle, from_dict, to_dict
from anygeometry.entities import Vertex

PROFILE = ((0,0,0),(2,0,0),(2,3,0),(0,3,0))


def _oracle(snapshot):
    xyz = snapshot.coordinates
    center = xyz.mean(axis=0)
    directed = {}
    volume = 0
    for loop in snapshot.face_loops:
        p = xyz[list(loop)]
        normal = np.cross(p[1]-p[0], p[2]-p[0])
        assert np.dot(normal, p.mean(axis=0)-center) > 0
        assert np.max(np.abs((p-p[0]) @ normal)) < 1e-9
        for j in range(1,len(p)-1):
            volume += np.dot(p[0]-center, np.cross(p[j]-center,p[j+1]-center))/6
        for a,b in zip(loop,loop[1:]+loop[:1]):
            directed[(a,b)] = directed.get((a,b),0)+1
    assert all(count == 1 and directed.get((b,a)) == 1 for (a,b),count in directed.items())
    assert volume == pytest.approx(snapshot.volume, rel=2e-13)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('height',[4,-4,0.0001,-0.0001])
def test_signed_prisms_closed_outward_and_analytic(reverse,height):
    m = GeometryModel()
    profile = PROFILE[::-1] if reverse else PROFILE
    rid = m.add_solid_prism(profile,(0,0,height),offset=-0.0002)
    s = m.solid_region_snapshot(rid)
    assert s.volume == pytest.approx(6*abs(height))
    assert s.revision == 1
    assert s.schema_version == 1 and s.geometry_class == 'convex_normal_prism'
    assert s.units == 'm'
    _oracle(s)
    assert all(h.kind == 'face' and h.model_id == m.model_id for h in s.face_refs)
    with pytest.raises(ValueError):
        s.coordinates.flags.writeable = True
    with pytest.raises(TypeError):
        s.face_groups['wrong'] = ()


def test_box_and_arbitrary_plane_prism():
    m = GeometryModel()
    rid = m.add_solid_box((1,2,3),(3,5,7),name='box')
    assert m.solid_region_snapshot(rid).volume == pytest.approx(24)
    # Rotate profile with orthonormal basis, no axis-aligned implementation assumption.
    u = np.array((1.,1.,0.))/np.sqrt(2)
    v = np.array((-1.,1.,2.))/np.sqrt(6)
    n = np.cross(u,v)
    p = [np.array((3,4,5))+a*u+b*v for a,b in ((0,0),(2,0),(2,3),(0,3))]
    s = m.solid_region_snapshot(m.add_solid_prism(p,2*n))
    assert s.volume == pytest.approx(12)
    _oracle(s)


@pytest.mark.parametrize('profile,vector',[
    (PROFILE,(1,0,1)), (PROFILE,(0,0,0)),
    (((0,0,0),(1,0,0),(0,1,0))*2,(0,0,1)),
    (((0,0,0),(1,0,0),(0.2,0.2,0),(0,1,0)),(0,0,1)),
    (((0,0,0),(1,1,0),(0,1,0),(1,0,0)),(0,0,1)),
    (((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,0)),(0,0,1)),
    (((0,0,0),(0.5,0,0),(1,0,0),(1,1,0),(0,1,0)),(0,0,1)),
    (((0,0,0),(1,0,0),(1,1,0.2),(0,1,0)),(0,0,1)),
    (PROFILE,(0,0,float('nan'))),
])
def test_invalid_prisms_atomic(profile,vector):
    m=GeometryModel()
    before=to_dict(m)
    with pytest.raises(GeometryError):
        m.add_solid_prism(profile,vector)
    assert to_dict(m)==before


def test_nested_failure_rolls_back_region_and_outer_geometry():
    m=GeometryModel()
    with pytest.raises(GeometryError,match='nested'):
        with m.transaction():
            m.add_solid_box((0,0,0),(1,1,1))
            try:
                m.add_solid_prism(PROFILE,(1,0,1))
            except GeometryError:
                pass
    assert not m.solid_regions and not m.vertices and m.revision == 0


def test_groups_snapshot_stale_and_regeneration_preserve_roles():
    m=GeometryModel()
    rid=m.add_solid_prism(PROFILE,(0,0,1))
    first=m.solid_region_snapshot(rid)
    m.set_solid_face_group(rid,'loaded',(first.face_refs[1],first.face_refs[2]))
    with pytest.raises(GeometryError,match='stale'):
        m.validate_solid_region_snapshot(first)
    selected=m.solid_region_snapshot(rid)
    m.regenerate_solid_prism(rid,np.array(PROFILE)*2,(0,0,3))
    updated=m.solid_region_snapshot(rid)
    assert updated.face_refs == first.face_refs
    assert updated.face_groups['loaded']==selected.face_groups['loaded']
    assert updated.volume == pytest.approx(72)
    assert first.volume==pytest.approx(6) and first.coordinates[4,2]==1
    assert updated.source_signature != first.source_signature
    before=to_dict(m)
    with pytest.raises(GeometryError,match='conflict'):
        m.regenerate_solid_prism(rid,((0,0,0),(1,0,0),(0,1,0)),(0,0,1))
    assert to_dict(m)==before
    foreign=EntityHandle(GeometryModel().model_id,'face',first.face_refs[0].id)
    with pytest.raises(GeometryError):
        m.set_solid_face_group(rid,'foreign',(foreign,))
    with pytest.raises(GeometryError):
        m.set_solid_face_group(rid,'start',(first.face_refs[1],))


def test_invalid_source_mutation_rolls_back_and_strict_audit_detects_corruption():
    m=GeometryModel()
    rid=m.add_solid_box((0,0,0),(1,1,1))
    region=m.solid_regions[rid]
    before=to_dict(m)
    with pytest.raises(GeometryError):
        m.move_point(region.vertex_ids[4],0.1,0,1)
    assert to_dict(m)==before
    with pytest.raises(GeometryError):
        m.remove_face(region.face_refs[0].id)
    assert to_dict(m)==before
    assert m.strict_audit().certifiable, m.strict_audit().to_dict()
    m._vertices[region.vertex_ids[4]]=Vertex(region.vertex_ids[4],np.array((0.1,0,1)))
    report=m.strict_audit()
    assert not report.certifiable
    assert any(i.classification=='invalid_closed_solid_region' for i in report.issues)


def test_clone_roundtrip_checksums_and_owner_identities():
    m=GeometryModel()
    rid=m.add_solid_box((0,0,0),(1,2,3))
    s=m.solid_region_snapshot(rid)
    m.set_solid_face_group(rid,'selected',(s.face_refs[0],s.face_refs[1]))
    doc=to_dict(m)
    restored=from_dict(json.loads(json.dumps(doc)))
    assert to_dict(restored)==doc
    assert restored.solid_region_snapshot(rid).source_signature==m.solid_region_snapshot(rid).source_signature
    other=m.clone()
    assert other.model_id != m.model_id
    assert all(h.model_id==other.model_id for h in other.solid_regions[rid].face_refs)
    assert all(h.model_id==other.model_id for h in other.solid_regions[rid].face_groups['selected'])
    assert m.clone(preserve_identity=True).solid_region_snapshot(rid).source_signature==m.solid_region_snapshot(rid).source_signature
    doc['solid_regions'][0]['name']='tampered'
    with pytest.raises(GeometryError,match='checksum'):
        from_dict(doc)


def test_builtin_feature_regeneration_roundtrip_conflict_and_suppression():
    m=GeometryModel()
    m.features.capture_baseline(m)
    feature=m.features.append('geometry.solid_prism',name='native',parameters={'profile':PROFILE,'vector':(0,0,1)})
    report=m.regenerate_features()
    assert report.success, report.diagnostic
    rid=next(iter(m.solid_regions))
    s=m.solid_region_snapshot(rid)
    m.set_solid_face_group(rid,'loaded',(s.face_refs[1],))
    m.features.update(feature.feature_id,parameters={'profile':PROFILE,'vector':(0,0,3)})
    report=m.regenerate_features()
    assert report.success, report.diagnostic
    current=m.solid_region_snapshot(rid)
    assert current.face_refs==s.face_refs and current.face_groups['loaded']==(s.face_refs[1],)
    assert current.volume==pytest.approx(18)
    restored=from_dict(to_dict(m))
    assert restored.solid_region_snapshot(rid).source_signature==current.source_signature
    restored.features.update(feature.feature_id,parameters={'profile':((0,0,0),(1,0,0),(0,1,0)),'vector':(0,0,1)})
    report=restored.regenerate_features()
    assert not report.success and 'conflict' in report.diagnostic
    assert restored.solid_region_snapshot(rid).source_signature==current.source_signature
    m.features.set_suppressed(feature.feature_id,True)
    report=m.regenerate_features()
    assert report.success,report.diagnostic
    assert not m.solid_regions and not m.faces


def test_ordinary_import_has_no_consumer_or_mesher_imports():
    code='import sys; import anygeometry; assert not any(k in sys.modules for k in ("anymesh", "anymesher", "anysolid", "pytetwild", "scipy"))'
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True,timeout=30)


@pytest.mark.parametrize('n',[3,5,7])
def test_polygon_prisms_use_analytic_regular_polygon_oracle(n):
    theta=np.arange(n)*2*np.pi/n
    profile=np.column_stack((2*np.cos(theta),2*np.sin(theta),np.zeros(n)))
    m=GeometryModel()
    s=m.solid_region_snapshot(m.add_solid_prism(profile,(0,0,-0.02),offset=0.01))
    assert s.volume==pytest.approx(n*2*np.sin(2*np.pi/n)*0.02)
    assert len(s.face_loops)==n+2
    _oracle(s)
    assert m.strict_audit().certifiable


def test_full_history_replay_preserves_region_face_and_group_roles():
    m=GeometryModel()
    m.features.capture_baseline(m)
    first=m.features.append('geometry.solid_prism',parameters={'profile':PROFILE,'vector':(0,0,1)})
    second=m.features.append('geometry.solid_prism',parameters={'profile':(np.array(PROFILE)+(5,0,0)).tolist(),'vector':(0,0,2)})
    assert m.regenerate_features().success
    before={rid:m.solid_region_snapshot(rid) for rid in m.solid_regions}
    for rid,s in before.items():
        m.set_solid_face_group(rid,'load',(s.face_refs[1],))
    m.features.move(second.feature_id,0)
    report=m.regenerate_features()
    assert report.success,report.diagnostic
    assert set(m.solid_regions)==set(before)
    for rid,s in before.items():
        current=m.solid_region_snapshot(rid)
        assert current.face_refs==s.face_refs
        assert current.face_groups['load']==(s.face_refs[1],)
        assert current.volume==s.volume
    assert to_dict(from_dict(to_dict(m)))==to_dict(m)


def test_strict_region_codec_rejects_valid_checksum_but_invalid_owner_orientation_or_schema():
    from anygeometry.serialization import _checksum
    m=GeometryModel()
    m.add_solid_box((0,0,0),(1,1,1))
    baseline=to_dict(m)
    for mutate in (
        lambda r:r.update(model_id=str(GeometryModel().model_id)),
        lambda r:r.update(schema_version=True),
        lambda r:r['face_orientations'].__setitem__(0,False),
        lambda r:r['face_groups'].__setitem__('start',[r['face_refs'][1]]),
        lambda r:r['vertex_ids'].__setitem__(0,r['vertex_ids'][1]),
    ):
        doc=json.loads(json.dumps(baseline))
        mutate(doc['solid_regions'][0])
        doc['checksum']=_checksum(doc)
        with pytest.raises(GeometryError):
            from_dict(doc)


def test_snapshot_content_and_document_units_fail_closed():
    m=GeometryModel()
    rid=m.add_solid_box((0,0,0),(1,1,1))
    s=m.solid_region_snapshot(rid)
    with pytest.raises(GeometryError,match='content'):
        m.validate_solid_region_snapshot(replace(s,volume=2))
    with pytest.raises(GeometryError,match='metres'):
        m.set_document_settings(units='mm')
    assert m.units=='m'
    with pytest.raises(GeometryError,match='committed'):
        with m.transaction():
            m.solid_region_snapshot(rid)
    assert m.revision==1


def test_snapshot_restore_and_group_edit_nested_failure_are_atomic():
    m=GeometryModel()
    rid=m.add_solid_box((0,0,0),(1,1,1))
    saved=m.topology_snapshot()
    s=m.solid_region_snapshot(rid)
    before=to_dict(m)
    with pytest.raises(GeometryError):
        with m.transaction():
            m.set_solid_face_group(rid,'new',(s.face_refs[0],))
            m.add_solid_prism(PROFILE,(1,0,0))
    assert to_dict(m)==before
    m.regenerate_solid_prism(rid,((0,0,0),(2,0,0),(2,1,0),(0,1,0)),(0,0,1))
    m.restore_topology(saved)
    assert m.solid_region_snapshot(rid).source_signature==s.source_signature
    assert m.solid_region_snapshot(rid).face_refs==s.face_refs


def test_coincident_duplicate_edge_cannot_break_closed_source_topology():
    from anygeometry.entities import OrientedEdge
    m=GeometryModel()
    rid=m.add_solid_box((0,0,0),(1,1,1))
    region=m.solid_regions[rid]
    face=m.faces[region.face_refs[0].id]
    use=face.loop[0]
    edge=m.edges[use.edge]
    before=to_dict(m)
    with pytest.raises(GeometryError,match='source-edge'):
        with m.transaction():
            duplicate=m.add_line(edge.start,edge.end)
            loop=(OrientedEdge(duplicate,use.forward),)+face.loop[1:]
            m._put_entity('face',replace(face,loop=loop))
    assert m.solid_region_snapshot(rid).source_signature==from_dict(before).solid_region_snapshot(rid).source_signature
    assert len(m.edges)==12


@pytest.mark.parametrize('version',[5,6,7])
def test_region_records_require_dedicated_top_level_schema(version):
    from anygeometry.serialization import _checksum
    m=GeometryModel()
    assert to_dict(m)['version']==5
    m.add_solid_box((0,0,0),(1,1,1))
    doc=to_dict(m)
    assert doc['version']==8
    from anygeometry.serialization import required_version
    assert required_version(doc)==8
    assert from_dict(doc).solid_regions
    doc['version']=version
    doc['checksum']=_checksum(doc)
    with pytest.raises(GeometryError,match='unexpected'):
        from_dict(doc)


def test_full_replay_blocked_prism_does_not_retain_materialized_topology():
    m = GeometryModel()
    m.features.capture_baseline(m)
    dependency = m.features.append('geometry.point', parameters={'position':[10,0,0]})
    prism = m.features.append('geometry.solid_prism',
        parameters={'profile':PROFILE,'vector':[0,0,1]}, dependencies=(dependency.feature_id,))
    unrelated = m.features.append('geometry.point', parameters={'position':[20,0,0]})
    assert m.regenerate_features().success
    old_region = next(iter(m.solid_regions.values()))
    old_faces = {h.id for h in old_region.face_refs}
    old_vertices = set(old_region.vertex_ids)
    m.features.set_suppressed(dependency.feature_id)
    # Removing a feature explicitly selects the real full-replay route;
    # simple history reordering may still use incremental additive replay.
    m.features.remove(unrelated.feature_id)
    report = m.regenerate_features()
    assert report.success, report.diagnostic
    assert m.features.get(prism.feature_id).state == 'blocked'
    assert not m.solid_regions
    assert not old_faces.intersection(m.faces)
    assert not old_vertices.intersection(m.vertices)
    assert not m.edges
    assert to_dict(from_dict(to_dict(m))) == to_dict(m)
