"""Exact user replay, retained as a development fixture (source imports)."""
from anyfem import Project
from anyfem.commands import CommandStack, AddPoint, AddPlate, AddFeature
from anygeometry import to_dict
from time import perf_counter
import traceback
import json
import pickle


def project():
    project = Project("three plates and cylinder")
    commands = CommandStack(project)
    for x, y, z in ((0,0,0),(5,0,0),(5,5,0),(0,5,0)):
        commands.run(AddPoint(x=x,y=y,z=z))
    commands.run(AddPlate(vertex_ids=[1,2,3,4]))
    for x,y,z in ((0,5,0),(5,0,2.5),(5,2.5,2.5),(0,2.5,2.5),(0,2.5,-2.5),
                  (5,2.5,-2.5),(2.5,5,-2.5),(2.5,5,2.5),(2.5,0,2.5),(2.5,0,-2.5)):
        commands.run(AddPoint(x=x,y=y,z=z))
    commands.run(AddPlate(vertex_ids=[11,12,13,14]))
    commands.run(AddPlate(vertex_ids=[7,8,9,10]))
    parameters={'radius':.5,'height':4.,'circumferential_segments':12,'origin':(1.,1.,-2.),
                'axis':(0.,0.,1.),'radial_direction':(1.,0.,0.),'longitudinal_spacing':.5,'ring_spacing':1.}
    commands.run(AddFeature(kind='generator.cylinder',name='Cylinder',parameters=parameters,label='add cylinder'))
    commands.undo()
    parameters={**parameters,'radius':1.,'origin':(2.5,2.5,-2.)}
    commands.run(AddFeature(kind='generator.cylinder',name='Cylinder',parameters=parameters,label='add cylinder'))
    return project, commands


if __name__ == '__main__':
    from anygeometry.batch_intersections import plan_intersections, apply_intersections
    from anygeometry import ConnectionIntent
    made, commands = project()
    before = to_dict(made.geometry)
    print('original faces',len(made.geometry.faces),flush=True)
    start=perf_counter()
    failed=False
    try:
        operands=(*made.geometry.faces,*(made.geometry.handle('member',member)
                                        for member in made.geometry.members))
        plan=plan_intersections(made.geometry, operands, policy=ConnectionIntent.CONNECT)
        print('planned',len(plan.arrangements),'seconds',perf_counter()-start,flush=True)
        with open('reports/general_intersections/replay-source.json','w') as file:
            json.dump(before,file)
        with open('reports/general_intersections/replay-plan.pickle','wb') as file:
            pickle.dump(plan,file)
        applied=apply_intersections(made.geometry,plan,policy=ConnectionIntent.CONNECT)
        print('applied faces',len(made.geometry.faces),'joints',len(applied.joint_edges),'seconds',perf_counter()-start,flush=True)
        print('topology',made.geometry.validate_topology(),flush=True)
    except Exception:
        failed=True
        traceback.print_exc()
        import sys
        error=sys.exc_info()[1]
        while error.__cause__ is not None: error=error.__cause__
        tb=error.__traceback__
        while tb.tb_next: tb=tb.tb_next
        values=tb.tb_frame.f_locals
        if 'edges' in values and 'vertices' in values:
            data={'face_id':values['domain'].face_id,'chain':values.get('chain'),
                  'vertices':[list(point) for point in values['vertices']],
                  'endpoints':values.get('endpoints_by_edge'),
                  'outgoing':values.get('outgoing'),
                  'paths':[{'type':type(path.curve).__name__, 'start':list(path.curve.evaluate(0)),
                            'end':list(path.curve.evaluate(1)),'owners':path.owners,'seam':path.decomposition}
                           for path in values['edges']]}
            with open('reports/general_intersections/replay-arrangement-debug.json','w') as file:
                json.dump(data,file,indent=2)
            print('failed arrangement face',data['face_id'],'chain',data['chain'],flush=True)
    print('source unchanged on failure' if failed else 'explicit batch changed source topology',
          to_dict(made.geometry)==before if failed else to_dict(made.geometry)!=before,flush=True)
