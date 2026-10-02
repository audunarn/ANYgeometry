"""Public consumer probe; a ready result is required, never inspection fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import anygeometry
import anymesher
from anygeometry import EntityRef, to_dict
from anymesher.preparation import prepare_structural_closure
from anymesher.recovery import generate_automatic_mesh_result, MeshAutomationOptions

from large_connected_fixtures import connected_strip, connected_hub, connected_mixed


def run(args):
    if not __debug__:
        raise RuntimeError('consumer checks require Python without optimization')
    report = {'status': 'running', 'family': args.family, 'count': args.count,
              'order': args.order, 'target_size': args.target_size,
              'geometry_origin': anygeometry.__file__, 'mesher_origin': anymesher.__file__,
              'python': sys.version, 'platform': platform.platform(), 'stages': {}}
    report['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['fixture_sha256'] = hashlib.sha256(
        Path(__file__).with_name('large_connected_fixtures.py').read_bytes()).hexdigest()
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def save():
        destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    def stage(name, operation):
        report['active_stage'] = name
        save()
        started = time.perf_counter()
        value = operation()
        report['stages'][name] = time.perf_counter()-started
        save()
        return value
    try:
        fixture = stage('build', lambda: {'strip': connected_strip, 'hub': connected_hub,
            'mixed': connected_mixed}[args.family](args.count))
        authored = to_dict(fixture.model)
        beam_edges = sorted({fixture.model.member_edge_uses[uid].edge_id
            for member in fixture.model.members.values() for uid in member.edge_use_ids})
        prepared, _preparation_report = stage('prepare', lambda: prepare_structural_closure(
            fixture.model, beam_edges=beam_edges))
        prepared_document = to_dict(prepared)
        result = stage('mesh', lambda: generate_automatic_mesh_result(prepared,
            target_size=args.target_size, strategy='auto', native_backend='python',
            order=args.order, qualified_s3=True, structural_preparation=False,
            automation=MeshAutomationOptions(max_seconds=args.max_seconds, allow_inspection=False)))
        report['mesh_result'] = result.to_dict()
        assert result.status == 'ready', result.status
        assert to_dict(fixture.model) == authored, 'authored geometry changed'
        assert to_dict(prepared) == prepared_document, 'prepared geometry changed'
        mesh = result.mesh
        nodes_by_face = {face: {node for element in elements for node in
            (mesh.quads[element] if element in mesh.quads else mesh.tris[element])}
            for face, elements in mesh.elements_of_face.items()}
        def owner_nodes(owner):
            kind, identifier = owner
            if kind == 'member':
                return set(mesh.nodes_of_member[identifier])
            return set().union(*(nodes_by_face.get(ref.id, set())
                for ref in prepared.resolve_ref(EntityRef(kind, identifier))))
        for joint in fixture.joints:
            start, end = np.asarray(joint['start']), np.asarray(joint['end'])
            direction = end-start
            length2 = float(direction@direction)
            sequences = []
            for owner in joint['owners']:
                located = []
                for node in owner_nodes(owner):
                    point = np.asarray(mesh.nodes[node])
                    parameter = float((point-start)@direction/length2)
                    if (-1e-9 <= parameter <= 1.+1e-9 and
                            np.max(np.abs(point-start-parameter*direction)) <= 1e-9):
                        located.append((parameter, node))
                located.sort()
                assert len(located) >= 2, ('missing intended joint', joint, owner)
                assert abs(located[0][0]) <= 1e-9 and abs(located[-1][0]-1.) <= 1e-9
                assert all(b[0]-a[0] > 1e-9 for a,b in zip(located,located[1:])), (
                    'duplicate coincident joint nodes', joint, owner)
                sequences.append([node for _,node in located])
            assert sequences[0] == sequences[1], ('nonconforming intended joint', joint, sequences)
        checked = 0
        for edge in prepared.edges:
            faces, members = prepared.faces_using_edge(edge), prepared.members_using_edge(edge)
            if len(faces)+len(members) < 2:
                continue
            sequence = set(mesh.nodes_of_edge[edge])
            assert len(sequence) >= 2, ('empty joint sequence', edge)
            for face in faces:
                assert sequence <= nodes_by_face[face], ('missing face joint nodes', edge, face)
            for member in members:
                assert sequence <= set(mesh.nodes_of_member[member]), ('missing member joint nodes', edge, member)
            checked += 1
        report.update(status='passed', shared_edges_checked=checked,
            intended_joints_checked=len(fixture.joints), nodes=len(mesh.nodes),
            quads=len(mesh.quads), tris=len(mesh.tris), beams=len(mesh.beams),
            authored_checksum=authored['checksum'], prepared_checksum=prepared_document['checksum'])
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('strip', 'hub', 'mixed'), required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--order', choices=('linear', 'quadratic'), default='linear')
    parser.add_argument('--target-size', type=float, default=.5)
    parser.add_argument('--max-seconds', type=float, default=120.)
    parser.add_argument('--report', required=True)
    run(parser.parse_args())
