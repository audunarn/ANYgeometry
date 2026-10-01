"""Independent analytic material and complete joint coverage checks.

The oracles use the cylinder equations and their closed-form intersections.
They do not sample ANYgeometry's intersection curves as reference truth.
"""
from collections import Counter
from pathlib import Path
import json
import math
import sys
from decimal import Decimal,localcontext
import numpy as np
from anymesher import mesh_from_dict,mesh_to_dict


def coverage(intervals,lower,upper):
    ordered=sorted(intervals)
    right=lower
    for start,end in ordered:
        assert abs(start-right)<1e-8,(start,right)
        assert end>start
        right=end
    assert abs(right-upper)<1e-8,(right,upper)


def automation_record(mesh,require_admission):
    automation=mesh.hybrid_diagnostics.get('automation',
        {'status':'inspection_only','solver_admission':'BLOCKED'})
    if require_admission:
        assert automation['status']=='ready' and automation['solver_admission']=='ADMITTED'
    return automation


def verify(mesh,name,*,require_admission=True):
    groups=(range(1,9),range(9,17))
    if name=='parallel':
        frames=((np.array([0.,0.,-1.5]),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.),
                (np.array([1.,0.,-1.]),np.array([0.,0.,1.]),np.array([1.,0.,0.]),.9))
    elif name=='perpendicular':
        frames=((np.array([0.,0.,-1.5]),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.),
                (np.array([0.,-1.5,0.]),np.array([0.,1.,0.]),np.array([1.,0.,0.]),1.))
    elif name=='skew':
        frames=((np.array([0.,0.,-1.5]),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.),
                (np.array([-.5,-1.,-.5]),np.array([1.,2.,1.])/math.sqrt(6),
                 np.array([5.,-2.,-1.])/math.sqrt(30),.8))
    elif name=='tangent':
        frames=((np.array([0.,0.,-1.5]),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.),
                (np.array([2.,0.,-1.5]),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.))
    else:
        raise ValueError('this independent closed-form oracle covers parallel and perpendicular axes')
    edges=[];nodes=[];areas=[]
    for selected,(origin,axis,radial,radius) in zip(groups,frames):
        transverse=np.cross(axis,radial)
        incidence=Counter();stations=set();area=0.
        for face in selected:
            for element in mesh.elements_of_face[face]:
                ids=mesh.corners_of(element)
                stations.update(ids)
                incidence.update(tuple(sorted(pair)) for pair in zip(ids,ids[1:]+ids[:1]))
                delta=np.array([mesh.nodes[node] for node in ids])-origin
                assert np.max(np.abs((delta@radial)**2+(delta@transverse)**2-radius**2))<1e-8
                angles=np.unwrap(np.arctan2(delta@transverse,delta@radial))
                height=delta@axis
                assert min(height)>-1e-8 and max(height)<3+1e-8
                area+=radius*abs(angles@np.roll(height,-1)-height@np.roll(angles,-1))/2
        assert abs(area-2*math.pi*radius*3)<1e-8,(area,radius)
        edges.append(set(incidence));nodes.append(stations);areas.append(float(area))
    shared=edges[0]&edges[1]
    shared_nodes={node for edge in shared for node in edge}
    assert shared and shared_nodes==nodes[0]&nodes[1]
    if name=='tangent':
        intervals=[]
        for edge in shared:
            points=np.array([mesh.nodes[node] for node in edge])
            assert np.max(np.abs(points[:,:2]-(1.,0.)))<1e-8
            intervals.append(tuple(sorted(points[:,2])))
        coverage(intervals,-1.5,1.5)
        degree=Counter(node for edge in shared for node in edge)
        assert sum(value==1 for value in degree.values())==2
        assert all(value in (1,2) for value in degree.values())
        automation=automation_record(mesh,require_admission)
        assert mesh_to_dict(mesh_from_dict(mesh_to_dict(mesh)))==mesh_to_dict(mesh)
        return {'case':name,'material_areas':areas,'shared_segments':len(shared),
                'shared_nodes':len(shared_nodes),'complete_branches':1,'joint_components':1,
                'closed_joint_components':0,'open_joint_components':1,'high_valence_nodes':0,
                'nodes':len(mesh.nodes),'elements':len(mesh.shells),'automation':automation,
                'oracle':'independent tangent circle equations, complete axial coverage and graph incidence'}
    if name=='skew':
        # The second axis passes through zero. Eliminating x^2+y^2=1
        # gives 5*z^2-2*A*z+2.16-A^2=0, A=x+2*y.
        # Real branches require |A|>=sqrt(1.8). Finite-height clipping
        # on cylinder two requires A+z>=-3; its upper bound is inactive.
        phase=math.atan2(2.,1.)
        delta=math.acos(.6)
        clipped=math.acos((3.-math.sqrt(1.14))/math.sqrt(5.))
        ranges={
            (1,1):((phase-delta,phase+delta),),
            (1,-1):((phase-delta,phase+delta),),
            (-1,1):((phase+math.pi-delta,phase+math.pi+delta),),
            (-1,-1):((phase+math.pi-delta,phase+math.pi-clipped),
                      (phase+math.pi+clipped,phase+math.pi+delta)),
        }
        intervals={key:[] for key in ranges}
        with localcontext() as context:
            context.prec=70
            for node in shared_nodes:
                x,y,z=(Decimal(str(float(value))) for value in mesh.nodes[node])
                assert abs(x*x+y*y-1)<Decimal('1e-8')
                assert abs(x*x+y*y+z*z-(x+2*y+z)**2/6-Decimal('.64'))<Decimal('1e-8')
        for edge in shared:
            points=np.array([mesh.nodes[node] for node in edge])
            a=points[:,0]+2*points[:,1]
            sign=1 if np.mean(a)>0 else -1
            selector=float(np.sum(5*points[:,2]-a))
            assert abs(selector)>1e-8,'branch endpoints alone do not identify this chart'
            branch=1 if selector>0 else -1
            angles=np.mod(np.arctan2(points[:,1],points[:,0]),2*math.pi)
            lo,hi=sorted(angles)
            assert hi-lo<math.pi
            intervals[sign,branch].append((lo,hi))
        for key,expected in ranges.items():
            values=intervals[key]
            for lower,upper in expected:
                selected=[value for value in values if lower-1e-8<=sum(value)/2<=upper+1e-8]
                coverage(selected,lower,upper)
            assert all(any(lower-1e-8<=value[0] and value[1]<=upper+1e-8
                           for lower,upper in expected) for value in values)
        degree=Counter(node for edge in shared for node in edge)
        assert sum(value==1 for value in degree.values())==2
        assert all(value in (1,2) for value in degree.values())
        graph={node:set() for node in degree}
        for a,b in shared:
            graph[a].add(b);graph[b].add(a)
        remaining=set(graph);components=[]
        while remaining:
            visited=set();pending=[min(remaining)]
            while pending:
                node=pending.pop()
                if node in visited:continue
                visited.add(node);pending.extend(graph[node]-visited)
            remaining-=visited;components.append(visited)
        assert len(components)==2
        assert sorted(sum(degree[node]==1 for node in component) for component in components)==[0,2]
        automation=automation_record(mesh,require_admission)
        assert mesh_to_dict(mesh_from_dict(mesh_to_dict(mesh)))==mesh_to_dict(mesh)
        return {'case':name,'material_areas':areas,'shared_segments':len(shared),
                'shared_nodes':len(shared_nodes),'complete_branches':4,'joint_components':2,
                'closed_joint_components':1,'open_joint_components':1,'high_valence_nodes':0,
                'nodes':len(mesh.nodes),'elements':len(mesh.shells),'automation':automation,
                'oracle':'independent eliminated cylinder equations, 70-digit residuals and finite-height interval coverage'}
    branch_edges=[[],[]]
    intervals=[[],[]]
    if name=='parallel':
        x=(1-.9**2+1)/2
        y=math.sqrt(1-x*x)
        for edge in shared:
            points=np.array([mesh.nodes[node] for node in edge])
            assert np.max(np.abs(points[:,0]-x))<1e-8
            branch=0 if np.mean(points[:,1])>0 else 1
            assert np.max(np.abs(points[:,1]-(y if branch==0 else -y)))<1e-8
            branch_edges[branch].append(edge)
            intervals[branch].append(tuple(sorted(points[:,2])))
        for values in intervals:
            coverage(values,-1.,1.5)
    else:
        for edge in shared:
            points=np.array([mesh.nodes[node] for node in edge])
            branch=0 if np.max(np.abs(points[:,1]-points[:,2]))<1e-8 else 1
            assert np.max(np.abs(points[:,1]-(points[:,2] if branch==0 else -points[:,2])))<1e-8
            branch_edges[branch].append(edge)
            angles=np.mod(np.arctan2(points[:,1],points[:,0]),2*math.pi)
            # A seam node with a tiny signed rounding error belongs to the
            # exact zero station; the existing world tolerance qualifies it.
            angles[np.minimum(angles,2*math.pi-angles)<1e-8]=0.
            lo,hi=sorted(angles)
            if hi-lo>math.pi:
                intervals[branch].extend(((hi,2*math.pi),(0.,lo)) if lo>1e-8 else ((hi,2*math.pi),))
            else:
                intervals[branch].append((lo,hi))
        for values in intervals:
            coverage(values,0.,2*math.pi)
    degrees=[]
    for branch in branch_edges:
        degree=Counter(node for edge in branch for node in edge)
        assert degree
        if name=='parallel':
            assert sorted(degree.values()).count(1)==2
            assert all(value in (1,2) for value in degree.values())
        else:
            assert set(degree.values())=={2}
        degrees.append(degree)
    common=set(degrees[0])&set(degrees[1])
    assert len(common)==(2 if name=='perpendicular' else 0)
    if common:
        assert sorted(round(float(mesh.nodes[node][0])) for node in common)==[-1,1]
    automation=automation_record(mesh,require_admission)
    assert mesh_to_dict(mesh_from_dict(mesh_to_dict(mesh)))==mesh_to_dict(mesh)
    return {'case':name,'material_areas':areas,'shared_segments':len(shared),
            'shared_nodes':len(shared_nodes),'complete_branches':2,'high_valence_nodes':len(common),
            'nodes':len(mesh.nodes),'elements':len(mesh.shells),'automation':automation,
            'oracle':'independent cylinder equations, angular/axial interval coverage and graph incidence'}


if __name__=='__main__':
    name=sys.argv[1]
    directory=Path(__file__).parent
    mesh=mesh_from_dict(json.loads((directory/f'accepted-cylinders-{name}.json').read_text()))
    result=verify(mesh,name)
    (directory/f'accepted-cylinders-{name}-checks.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))
