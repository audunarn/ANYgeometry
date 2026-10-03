"""Complete coverage discriminator; no mesh or publication acceptance."""
from anygeometry import GeometryError,validate_prepared_authored_face_partition
from examples.authored_child_coverage_handoff import build


def verify():
    model,correspondence,left,right=build()
    rows={left:[[[0,0],[3,0],[3,4]],[[0,0],[3,4],[0,4]]],
          right:[[[3,0],[4,0],[4,4]],[[3,0],[4,4],[3,4]]]}
    validate_prepared_authored_face_partition(model,correspondence,rows)
    try:
        validate_prepared_authored_face_partition(model,correspondence,{left:rows[left]})
    except GeometryError:
        pass
    else:
        raise AssertionError('left-only containment must not prove whole-root coverage')
    return dict(original_area=16,left_area=12,right_area=4,complete_partition=True,
                accepted_mesh=False)


if __name__=='__main__':
    print(verify())
