"""Bounded development/consumer fixture runner; not a release qualification claim."""
import argparse
import cProfile
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import anygeometry
from anygeometry import (EntityRef, IntersectionBatchPolicy, apply_intersections,
                         from_dict, plan_intersections, query_trimmed_surface_charts, to_dict)
from anygeometry.curves import Straight

from large_connected_fixtures import connected_hub, connected_mixed, connected_strip


def owner_edges(model, owner):
    kind, identifier = owner
    if kind == 'member':
        return {model.member_edge_uses[u].edge_id for u in model.members[identifier].edge_use_ids}
    return {use.edge for ref in model.resolve_ref(EntityRef(kind, identifier))
            for loop in (model.faces[ref.id].loop, *model.faces[ref.id].holes) for use in loop}


def verify(fixture, model, charts):
    if not __debug__:
        raise RuntimeError('acceptance checks require Python without optimization')
    assert model.validate_topology() == ()
    assert {(h.kind, h.id) for h in fixture.operands} == (
        {('face', f) for f in fixture.model.faces} |
        {('member', m) for m in fixture.model.members}), 'unselected authored operand'
    # Batch application may establish sheets for authored, previously unowned faces.
    assert set(fixture.model.sheets) <= set(model.sheets)
    assert set(model.members) == set(fixture.model.members)
    for identifier, original in fixture.model.sheets.items():
        current = model.sheets[identifier]
        assert (current.part_id, current.name, current.metadata) == (
            original.part_id, original.name, original.metadata)
        expected_uses = sorted((ref.id, use.orientation) for uid in original.face_use_ids
            for use in (fixture.model.face_uses[uid],)
            for ref in model.resolve_ref(EntityRef('face', use.face_id)))
        actual_uses = sorted((model.face_uses[uid].face_id, model.face_uses[uid].orientation)
                             for uid in current.face_use_ids)
        assert actual_uses == expected_uses, ('sheet ownership/orientation', identifier)
    owners = {(h.kind, h.id): owner_edges(model, (h.kind, h.id)) for h in fixture.operands}
    edge_owners = {}
    for owner, edges in owners.items():
        for edge in edges:
            edge_owners.setdefault(edge, set()).add(owner)
    connected = {next(iter(owners))}
    while True:
        expanded = set(connected)
        for group in edge_owners.values():
            if group & connected:
                expanded.update(group)
        if expanded == connected:
            break
        connected = expanded
    assert connected == set(owners), ('disconnected source owners', set(owners)-connected)
    areas = {chart.face.id: chart.material_area for chart in charts.charts}
    for parent, expected in fixture.areas.items():
        actual = sum(areas[ref.id] for ref in model.resolve_ref(EntityRef('face', parent)))
        assert abs(actual - expected) <= 1e-8, (parent, expected, actual)
    for joint in fixture.joints:
        common = owners[joint['owners'][0]] & owners[joint['owners'][1]]
        start, end = np.asarray(joint['start']), np.asarray(joint['end'])
        direction = end - start
        length2 = float(direction @ direction)
        intervals = []
        for identifier in common:
            edge = model.edges[identifier]
            assert isinstance(edge.curve, Straight)
            points = np.array([model.vertices[v].position for v in (edge.start, edge.end)])
            t = (points-start) @ direction / length2
            if np.max(np.abs(points - (start + t[:, None]*direction))) > 1e-9:
                continue
            a, b = sorted(t)
            if a >= -1e-9 and b <= 1.+1e-9:
                intervals.append((a, b))
        cursor = 0.
        for a, b in sorted(intervals):
            assert abs(a-cursor) <= 1e-9, ('gap/overlap', joint, intervals)
            cursor = b
        assert abs(cursor-1.) <= 1e-9, ('missing joint', joint, intervals)
    owner_vertices = {owner: {v for e in edges for v in (model.edges[e].start, model.edges[e].end)}
                      for owner, edges in owners.items()}
    for joint in fixture.point_joints:
        common = set.intersection(*(owner_vertices[owner] for owner in joint['owners']))
        matches = [v for v in common if np.max(np.abs(model.vertices[v].position-joint['position'])) <= 1e-9]
        assert len(matches) == 1, ('noncanonical point joint', joint, matches)
    for joint in fixture.curve_joints:
        common = owners[joint['wall']] & set.union(*(owners[p] for p in joint['panels']))
        assert common, ('missing wall/quadric joint', joint)
        assert any(type(model.edges[e].curve).__name__ == joint['required_curve'] for e in common), (
            'missing exact intersection family', joint, common)


def peak_memory():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set',
                'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        info = Counters()
        info.cb = ctypes.sizeof(info)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        fn = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not fn(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'peak_working_set_bytes': info.peak_working_set}
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_rss_bytes': value if sys.platform == 'darwin' else value*1024}


def run(args):
    if not __debug__:
        raise RuntimeError('acceptance checks require Python without optimization')
    report = {'family': args.family, 'count': args.count, 'python': sys.version,
              'platform': platform.platform(), 'status': 'running', 'stages': {}}
    report['geometry_origin'] = anygeometry.__file__
    report['profiled'] = bool(args.profile)
    sources = [Path(__file__), Path(__file__).with_name('large_connected_fixtures.py')]
    package = Path(anygeometry.__file__).parent
    sources += [package/name for name in ('batch_intersections.py', 'member_arrangements.py',
                                         'material_arrangement.py', 'arrangement_geometry.py')]
    report['source_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def save():
        destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    def stage(name, operation):
        report['active_stage'] = name
        save()
        started = time.perf_counter()
        result = operation()
        report['stages'][name] = time.perf_counter()-started
        save()
        return result
    profiler = cProfile.Profile() if args.profile else None
    try:
        fixture = stage('build', lambda: {'strip': connected_strip, 'hub': connected_hub,
                                         'mixed': connected_mixed}[args.family](args.count))
        report['authored_faces'] = len(fixture.model.faces)
        report['authored_members'] = len(fixture.model.members)
        original = to_dict(fixture.model)
        model = from_dict(original)
        deadline = time.monotonic()+args.deadline_seconds
        policy = IntersectionBatchPolicy(cancellation_check=lambda: time.monotonic() >= deadline)
        if profiler:
            profiler.enable()
        plan = stage('plan', lambda: plan_intersections(model, fixture.operands, policy=policy))
        assert to_dict(model) == original
        if profiler:
            profiler.disable()
            report['profile_counts'] = {name: sum(item.callcount for item in profiler.getstats()
                if getattr(item.code, 'co_name', None) == name)
                for name in ('check', 'check_pair', '_domain_bounds', '_pair_paths')}
        stage('apply', lambda: apply_intersections(model, plan, policy=policy))
        charts = stage('charts', lambda: query_trimmed_surface_charts(
            model, cancellation_check=lambda _phase: policy.cancellation_check()))
        stage('verify', lambda: verify(fixture, model, charts))
        document = to_dict(model)
        assert apply_intersections(model, plan, policy=policy).reused
        assert to_dict(model) == document
        assert to_dict(from_dict(document)) == document
        assert to_dict(fixture.model) == original
        if args.export:
            directory = Path(args.export)
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in (('authored', original), ('prepared', document)):
                (directory/f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
            manifest = {'name': fixture.name, 'operands': [(h.kind, h.id) for h in fixture.operands],
                        'areas': fixture.areas, 'joints': fixture.joints, 'point_joints': fixture.point_joints,
                        'curve_joints': fixture.curve_joints,
                        'authored_checksum': original['checksum'], 'prepared_checksum': document['checksum']}
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        report.update(status='passed', operands=len(fixture.operands), expected_joints=len(fixture.joints),
                      faces=len(model.faces), edges=len(model.edges), vertices=len(model.vertices),
                      members=len(model.members), **peak_memory())
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), **peak_memory())
        raise
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(args.profile)
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('strip', 'hub', 'mixed'), default='strip')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--report', required=True)
    parser.add_argument('--profile')
    parser.add_argument('--export')
    parser.add_argument('--deadline-seconds', type=float, default=120.)
    run(parser.parse_args())
