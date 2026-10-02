"""Small owner-built schema-6 fixtures for installed ANYmesher/ANYfem consumers.

Only public ANYgeometry APIs are used. Run with an installed candidate and an
output directory to write authored/prepared documents and attachment oracles.
Recorded topology IDs are integration inputs, not independent numerical proof.
"""
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np

from anygeometry import (GeometryModel, ParameterRange, apply_intersections, from_dict,
                         plan_intersections, to_dict)
from anygeometry.generators import cylinder


@dataclass
class Fixture:
    name: str
    model: GeometryModel
    operands: list[int]
    expected: dict


def _wall(controls=((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.)),
          vector=(.25, 0., 1.5)):
    model = GeometryModel()
    points = model.add_points(controls)
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    return model, list(model.extrude([edge], vector))


def _pipe(model, faces, *, radius=.7, origin=(-2., .4, .6), axis=(1., 0., 0.), length=8.):
    previous = set(model.faces)
    model.insert_model(cylinder(radius, length, origin=origin, axis=axis,
                                radial_direction=(0., 1., 0.), circumferential_segments=8))
    return faces + sorted(set(model.faces) - previous)


def _apply(model, operands):
    plan = plan_intersections(model, [model.handle('face', f) for f in operands], policy='connect')
    return apply_intersections(model, plan, policy='connect')


def cubic_oblique_pipe():
    model, faces = _wall()
    faces = _pipe(model, faces, axis=(1., .2, .1))
    return Fixture('cubic-oblique-pipe', model, faces, {
        'supported': True, 'required_curve_type': 'BezierQuadricCurve',
        'requires_regularized_fold_end': True, 'source_must_remain_unchanged': True})


def parabolic_pipe():
    model, faces = _wall(((0., 0., 0.), (2., 2., 0.), (4., 0., 0.)), (0., 0., 2.))
    faces = _pipe(model, faces, radius=.6, origin=(-1., .5, .8), length=6.)
    return Fixture('parabolic-pipe', model, faces, {
        'supported': True, 'required_curve_type': 'QuadricIntersectionCurve',
        'source_must_remain_unchanged': True})


def tangent_pipe(offset=0.):
    model = GeometryModel()
    plate = model.add_plate(model.add_points(((0., -2., -1.), (0., 2., -1.),
                                              (0., 2., 3.), (0., -2., 3.))))
    model.add_sheet((plate,))
    model.insert_model(cylinder(1., 2., origin=(1. + offset, 0., 0.),
        radial_direction=(math.cos(.7), math.sin(.7), 0.), circumferential_segments=8))
    name = 'tangent' if offset == 0 else 'secant' if offset < 0 else 'separated'
    ys = [0.] if offset == 0 else [-math.sqrt(1 - (1+offset)**2), math.sqrt(1 - (1+offset)**2)] if offset < 0 else []
    return Fixture('plate-pipe-' + name, model, list(model.faces), {
        'supported': True, 'generator_lines': [[[0., y, 0.], [0., y, 2.]] for y in ys],
        'material_area': 16. + 4. * math.pi, 'source_must_remain_unchanged': True})


def second_cut_with_attachments():
    model, faces = _wall()
    faces = _pipe(model, faces, radius=.5, origin=(-2., .4, .5))
    first = _apply(model, faces)
    children = [f for f, face in model.faces.items() if type(face.surface).__name__ == 'ExtrudedSurface']
    operands = _pipe(model, children, radius=.25, origin=(-2., .6, 1.1))
    plan = plan_intersections(model, [model.handle('face', f) for f in operands], policy='connect')
    old_joints = {edge.id for edge in first.joint_edges}
    selected = None
    for arrangement in plan.arrangements:
        for path in arrangement.paths:
            if path.source_edge not in old_joints:
                continue
            for endpoint in path.curve.evaluate(np.array([0., 1.])):
                _point, t, distance = model.closest_edge_point(path.source_edge, endpoint)
                if distance < 1e-10 and 1e-6 < t < 1 - 1e-6:
                    selected = path.source_edge
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        raise AssertionError('fixture no longer splits an existing declared joint')
    stations = []
    for kind, t in (('vertex', .31), ('member', .73)):
        point = model.sample_edge(selected, np.array([t]))[0]
        vertex = model.add_point(*point)
        if kind == 'vertex':
            identifier = model.add_attachment(None, 'vertex_on_edge', 'edge', selected,
                ParameterRange.point(0.), (ParameterRange.point(t),), source_kind='vertex', source_id=vertex,
                evidence='exact', tolerance_used=1e-9)
        else:
            other = model.add_point(*(point + (0., 0., .2)))
            member = model.add_member((model.add_line(vertex, other),))
            identifier = model.add_attachment(member, 'member_on_boundary', 'edge', selected,
                ParameterRange.point(0.), (ParameterRange.point(t),), evidence='exact', tolerance_used=1e-9)
        stations.append({'attachment_id': identifier, 'source_kind': kind, 'position': point.tolist()})
    return Fixture('second-cut-attachments', model, operands, {
        'supported': True, 'old_joint_edge': selected, 'attachment_stations': stations,
        'source_must_remain_unchanged': True})


def fixtures():
    return [cubic_oblique_pipe(), parabolic_pipe(), tangent_pipe(), tangent_pipe(-1e-6),
            tangent_pipe(1e-6), second_cut_with_attachments()]


def prepare(fixture):
    candidate = from_dict(to_dict(fixture.model))
    application = _apply(candidate, fixture.operands)
    return candidate, application


def write_documents(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for fixture in fixtures():
        authored = to_dict(fixture.model)
        prepared, application = prepare(fixture)
        assert to_dict(fixture.model) == authored
        for suffix, document in (('authored', authored), ('prepared', to_dict(prepared))):
            (directory / f'{fixture.name}-{suffix}.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
        manifest.append({'name': fixture.name, 'operands': fixture.operands, 'expected': fixture.expected,
            'source_checksum': authored['checksum']['value'],
            'prepared_checksum': to_dict(prepared)['checksum']['value'],
            'joint_edge_ids': [edge.id for edge in application.joint_edges]})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory')
    args = parser.parse_args()
    print(json.dumps(write_documents(args.output_directory), indent=2))
