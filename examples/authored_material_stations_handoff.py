"""Exact current traces and rational partitions; no native mesher."""
from fractions import Fraction as F

from anygeometry import (Plane, query_prepared_authored_boundary_correspondence,
    query_prepared_authored_material_stations, validate_prepared_authored_face_partition)
from examples.prepared_sheet_joint_component_handoff import build as build_sheets


def unpack(rows):
    return tuple(tuple(F(*value) for value in row) for row in rows)


def build():
    def frame(model,roots,sheets):
        model.set_face_surface(roots[1],Plane((3,-1,-1),(0,6,0),(0,0,2)))
    model,roots,sheets,edge=build_sheets(before_prepare=frame)
    bindings=tuple(query_prepared_authored_boundary_correspondence(model,root) for root in roots)
    return model,bindings,sheets,edge


def partition(model,binding):
    result={}
    for face in binding.descendants:
        points=[]
        for use in model.faces[face].loop:
            receipt=query_prepared_authored_material_stations(model,binding,use.edge,(0,1))
            points.append(unpack(receipt.authored_uv)[0 if use.forward else 1])
        assert len(points)==4
        result[face]=((points[0],points[1],points[2]),(points[0],points[2],points[3]))
    return result


def verify():
    model,bindings,sheets,edge=build()
    for binding in bindings:
        validate_prepared_authored_face_partition(model,binding,partition(model,binding))
    return {'authored_roots':tuple(binding.authored_definition.face_id for binding in bindings),
            'current_faces':tuple(face for binding in bindings for face in binding.descendants),
            'joint_edge':edge,'sheets':sheets,'exact_partition_coverage':True,'accepted_mesh':False}


if __name__=='__main__':
    print(verify())
