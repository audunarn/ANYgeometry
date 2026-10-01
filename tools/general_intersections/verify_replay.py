"""Retained exact application replay and independent mesh incidence checks."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import json
import math
import sys
from time import perf_counter
import numpy as np
from replay import project
from anygeometry import to_dict
from anyfem.io.project_file import save_project, load_project
from anymesher import mesh_to_dict, mesh_from_dict


def verify_lifecycle(made,commands):
    from anyfem.commands import EditFeature
    from anygeometry import EntityRef
    before=to_dict(made.geometry)
    identity=made.geometry.model_id
    assert commands.undo()
    assert len(made.geometry.faces)==3
    assert made.geometry.model_id==identity
    assert commands.redo()
    after=to_dict(made.geometry)
    for record in (before,after):
        record.pop('revision',None)
        record.pop('checksum',None)
    assert before==after,[key for key in before if before[key]!=after[key]]
    cylinder=next(feature for feature in made.geometry.features.records
                  if feature.kind=='generator.cylinder')
    references=tuple(EntityRef('face',face) for face in made.geometry.faces)
    commands.run(EditFeature(cylinder.feature_id,name='Cylinder regenerated',
                             parameters=dict(cylinder.parameters)))
    assert made.geometry.model_id==identity
    assert made.geometry.validate_topology()==()
    assert len(made.geometry.faces)==55
    assert all(made.geometry.resolve_ref(reference) for reference in references)
    regenerated=made.regenerate_geometry_features()
    assert regenerated.success,regenerated.diagnostic
    assert made.geometry.validate_topology()==()
    return {'undo_redo_source_identity':True,'undo_redo_source_content':True,
            'feature_regeneration':True,'retained_downstream_face_references':len(references)}


def verify(mesh, source):
    faces = sorted(source.faces)
    groups = [set([face]) for face in faces[:3]] + [set(faces[3:])]
    edges, nodes = [], []
    areas = []
    for category, group in enumerate(groups):
        incidence, member_nodes, area = Counter(), set(), 0.
        for face in group:
            for element in mesh.elements_of_face[face]:
                ids = mesh.corners_of(element)
                member_nodes.update(ids)
                incidence.update(tuple(sorted(pair)) for pair in zip(ids, ids[1:]+ids[:1]))
                xyz = np.asarray([mesh.nodes[node] for node in ids])
                if category < 3:
                    area += sum(np.linalg.norm(np.cross(xyz[i]-xyz[0], xyz[i+1]-xyz[0]))/2
                                for i in range(1, len(ids)-1))
                else:
                    angles = np.unwrap(np.arctan2(xyz[:,1]-2.5, xyz[:,0]-2.5))
                    heights = xyz[:,2]
                    area += abs(np.dot(angles, np.roll(heights,-1))
                                -np.dot(heights,np.roll(angles,-1)))/2
        edges.append(set(incidence))
        nodes.append(member_nodes)
        areas.append(float(area))
    assert np.allclose(areas, [25.,25.,25.,8*math.pi], rtol=0, atol=1e-8), areas
    joints = []
    for a,b in combinations(range(4),2):
        shared = edges[a] & edges[b]
        assert shared, (a,b)
        shared_nodes = {node for edge in shared for node in edge}
        assert shared_nodes == nodes[a]&nodes[b], (a,b,"unconnected joint station")
        length = 0.
        for first,second in shared:
            p,q = mesh.nodes[first],mesh.nodes[second]
            if (a,b)==(0,3):
                assert abs(p[2]) < 1e-8 and abs(q[2]) < 1e-8
                a_xy,b_xy=p[:2]-2.5,q[:2]-2.5
                angle = math.atan2(a_xy[0]*b_xy[1]-a_xy[1]*b_xy[0],np.dot(a_xy,b_xy))
                length += abs(angle)
            else:
                length += float(np.linalg.norm(q-p))
        expected = 2*math.pi if (a,b)==(0,3) else (8. if b==3 else 5.)
        assert abs(length-expected)<1e-8, (a,b,length,expected)
        joints.append({"owners":[a,b],"shared_nodes":len(shared_nodes),
                       "shared_segments":len(shared),"analytic_length":length})
    assert len(nodes[0]&nodes[1]&nodes[2]) == 1
    assert len(nodes[0]&nodes[1]&nodes[3]) == 2
    assert len(nodes[0]&nodes[2]&nodes[3]) == 2
    return {"material_areas":areas,"joints":joints}


if __name__ == '__main__':
    from anymesher import MeshAutomationOptions
    size = float(sys.argv[1])
    label = f"{size:.2f}".replace('.','')
    directory = Path(__file__).parent
    made,commands = project()
    before = to_dict(made.geometry)
    start = perf_counter()
    mesh = made.generate_mesh(size,strategy='quad_first',order='linear',automation=MeshAutomationOptions())
    assert to_dict(made.geometry)==before
    codec = mesh_to_dict(mesh)
    # Preserve the actual accepted candidate before additional verification.
    (directory/f'accepted-mesh-{label}.json').write_text(json.dumps(codec,allow_nan=False))
    assert mesh_to_dict(mesh_from_dict(codec)) == codec
    checks = verify(mesh,made.geometry)
    saved = save_project(made,directory/f'accepted-source-{label}.anyfem')
    restored = load_project(saved)
    assert to_dict(restored.geometry)==before
    automation = mesh.hybrid_diagnostics['automation']
    assert automation['status']=='ready' and automation['solver_admission']=='ADMITTED'
    lifecycle=verify_lifecycle(made,commands)
    report = {"target_size":size,"nodes":len(mesh.nodes),"quads":len(mesh.quads),
              "triangles":len(mesh.tris),"source_unchanged":True,"source_save_load":True,
              "mesh_codec_roundtrip":True,"automation":automation,"checks":checks,
              "lifecycle":lifecycle,"seconds":perf_counter()-start}
    (directory/f'accepted-replay-{label}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)
