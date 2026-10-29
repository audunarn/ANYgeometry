"""Fail-closed planar sketch trimming using owner curve predicates."""
from dataclasses import replace
import numpy as np
from .curves import Straight, Arc, Spline
from .nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .sketch import face_sketch_plane


def validate_sketch_cut(geometry,face_id,wire_edges):
    from .sketch_entities import _order_sketch_contour
    from .arrangement_geometry import freeze_edge, curve_junctions, point_parameters
    from .material_arrangement import MaterialDomain, ArrangementPath
    from .nurbs_audit import certify_simple, certify_only_shared_ends
    face=geometry.faces[face_id]; plane=face_sketch_plane(geometry,face_id)
    tolerance=geometry.tolerance.length
    if not wire_edges: raise GeometryError('select at least one closed contour')
    added=tuple(_order_sketch_contour(geometry,edges) for edges in wire_edges)
    loops=(face.loop,)+face.holes+added
    # Topology order is proven by order_loop. Coplanarity is checked on every
    # defining control, not by display samples.
    curves=[]
    for loop in loops:
        group=[]
        for item in loop:
            edge=geometry.edges[item.edge]; curve=edge.curve
            if isinstance(curve,NurbsCurve): controls=np.asarray(curve.controls)
            elif isinstance(curve,Arc): controls=np.asarray([geometry.vertex_position(k) for k in (edge.start,curve.via_vertex,edge.end)])
            elif isinstance(curve,Straight): controls=np.asarray([geometry.vertex_position(k) for k in (edge.start,edge.end)])
            else: raise GeometryError('cut contour curve type is unsupported')
            if np.max(np.abs((controls-plane.origin)@plane.normal))>tolerance:
                raise GeometryError('cut contours must be coplanar with the plate')
            group.append((item,edge))
        curves.append(group)
    has_nurbs=any(isinstance(e.curve,NurbsCurve) for loop in curves for _,e in loop)
    has_arcs=any(isinstance(e.curve,Arc) for loop in curves for _,e in loop)
    if has_nurbs and has_arcs:
        # Different contour families can be certified independently when their
        # original owner bounds are strictly separated. No analytic arc is
        # reclassified as a rounded rational approximation for acceptance.
        families=[('nurbs' if any(isinstance(e.curve,NurbsCurve) for _,e in loop) else
                   'arc' if any(isinstance(e.curve,Arc) for _,e in loop) else 'line') for loop in curves]
        if families[0]!='line' or any(any(isinstance(e.curve,NurbsCurve) for _,e in loop) and any(isinstance(e.curve,Arc) for _,e in loop) for loop in curves):
            raise GeometryError('mixed arc/NURBS within one contour cannot yet be certified')
        bounds=[]
        for loop in loops:
            boxes=np.asarray(geometry.entity_bounds_many(tuple(('edge',item.edge) for item in loop)))
            bounds.append((boxes[:,:3].min(axis=0),boxes[:,3:].max(axis=0)))
        for i in range(1,len(loops)):
            for j in range(i+1,len(loops)):
                if {families[i],families[j]}=={'arc','nurbs'}:
                    low_a,high_a=bounds[i]; low_b,high_b=bounds[j]
                    if not (np.any(high_a+tolerance<low_b) or np.any(high_b+tolerance<low_a)):
                        raise GeometryError('arc/NURBS contour separation is unresolved; conservative curve bounds overlap')
        start=1+len(face.holes)
        for family in ('arc','nurbs'):
            proposed=[tuple(item.edge for item in loops[i]) for i in range(start,len(loops)) if families[i] in (family,'line')]
            if not proposed: continue
            working=geometry.clone(include_features=False)
            retained=tuple(loops[i] for i in range(1,start) if families[i] in (family,'line'))
            working._put_entity('face',replace(working.faces[face_id],holes=retained))
            validate_sketch_cut(working,face_id,proposed)
        return added
    if has_nurbs:
        # Use the established exact rational chart predicates. The temporary
        # chart encloses every defining control; it does not alter the model.
        all_points=np.vstack([np.asarray(e.curve.controls) if isinstance(e.curve,NurbsCurve) else np.asarray([geometry.vertex_position(e.start),geometry.vertex_position(e.end)]) for loop in curves for _,e in loop])
        local=np.asarray([plane.local(p) for p in all_points]); lo=local.min(axis=0)-1; span=local.max(axis=0)-lo+1
        pcurves=[]
        for loop in curves:
            group=[]
            for item,e in loop:
                if isinstance(e.curve,NurbsCurve):
                    controls=tuple(tuple((plane.local(p)-lo)/span) for p in e.curve.controls)
                    curve=replace(e.curve,controls=controls)
                else:
                    controls=tuple(tuple((plane.local(geometry.vertex_position(k))-lo)/span) for k in (e.start,e.end))
                    curve=NurbsCurve(1,(0.,0.,1.,1.),controls)
                if not item.forward: curve=curve.subcurve(1.,0.)
                if certify_simple(curve) is not True: raise GeometryError('cut contour is not certified simple')
                group.append(curve)
            pcurves.append(group)
        for i,group in enumerate(pcurves):
            for a,first in enumerate(group):
                for j in range(i,len(pcurves)):
                    for b,second in enumerate(pcurves[j]):
                        if i==j and b<=a: continue
                        if not certify_only_shared_ends(first,second): raise GeometryError('cut contours self-intersect or cannot be certified disjoint')
                        common={first.exact_point(0),first.exact_point(1)}&{second.exact_point(0),second.exact_point(1)}
                        adjacent=i==j and ((a-b)%len(group) in (1,len(group)-1))
                        if common and not adjacent: raise GeometryError('cut contours touch or intersect')
        support=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),(((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))
        paths=tuple(tuple(ArrangementPath(CurveOnSurface(support,c)) for c in group) for group in pcurves)
        domain=MaterialDomain(face_id,support,paths)
        def contains(i,j): return domain.contains_loop(paths[i],pcurves[j][0].exact_point(0),tolerance/max(span),boundary=False)
    else:
        paths=tuple(tuple(ArrangementPath(freeze_edge(geometry,item.edge) if item.forward else freeze_edge(geometry,item.edge).subcurve(1.,0.)) for item,_ in loop) for loop in curves)
        # Analytical line/conic root predicates classify all curve pairs.
        for i,group in enumerate(paths):
            for a,path in enumerate(group):
                for j in range(i,len(paths)):
                    for b,other in enumerate(paths[j]):
                        if i==j and b<=a: continue
                        contacts=curve_junctions(path.curve,other.curve,tolerance=tolerance)
                        if point_parameters(other.curve,path.curve.evaluate(.5),tolerance=tolerance):
                            raise GeometryError('cut contour edges overlap or intersect')
                        adjacent=i==j and ((a-b)%len(group) in (1,len(group)-1))
                        for u,v in contacts:
                            if not adjacent or not ((u<=1e-10 or u>=1-1e-10) and (v<=1e-10 or v>=1-1e-10)):
                                raise GeometryError('cut contours intersect or touch')
        # face_sketch_plane has orthonormal axes, while the support may have
        # arbitrary units; containment uses that original support's chart.
        from .surfaces import Plane
        support=Plane(plane.origin,plane.x_axis,plane.y_axis)
        domain=MaterialDomain(face_id,support,paths)
        def contains(i,j): return domain.contains_loop(paths[i],plane.local(paths[j][0].curve.evaluate(0.)),tolerance,boundary=False)
    start=1+len(face.holes)
    for j in range(start,len(loops)):
        if not contains(0,j): raise GeometryError('cut contour must lie strictly inside the plate')
        for i in range(1,len(loops)):
            if i!=j and (contains(i,j) or contains(j,i)):
                raise GeometryError('cut contours overlap existing holes or contain nested islands')
    return added


def apply_sketch_cut(geometry,face_id,wire_edges):
    with geometry.transaction():
        added=validate_sketch_cut(geometry,face_id,wire_edges)
        # Face mutation currently requires three edges per trim loop. Split a
        # private boundary copy, leaving the driving sketch and its output
        # identities untouched. Prefer splitting the straight chord.
        compatible=[]
        for loop in added:
            if len(loop)!=2:
                compatible.append(loop)
                continue
            chosen=next((item for item in loop if isinstance(geometry.edges[item.edge].curve,Straight)),loop[0])
            edge=geometry.edges[chosen.edge]
            middle=geometry.add_point(*geometry.sample_edge(edge.id,np.asarray((.5,)))[0])
            if isinstance(edge.curve,Straight):
                children=(geometry.add_line(edge.start,middle),geometry.add_line(middle,edge.end))
            elif isinstance(edge.curve,Arc):
                samples=geometry.sample_edge(edge.id,np.asarray((.25,.75)))
                via=tuple(geometry.add_point(*point) for point in samples)
                geometry.mark_construction_vertices(via)
                children=(geometry.add_arc(edge.start,via[0],middle),geometry.add_arc(middle,via[1],edge.end))
            else:
                children=(geometry.add_curve(edge.start,middle,edge.curve.subcurve(0.,.5)),
                          geometry.add_curve(middle,edge.end,edge.curve.subcurve(.5,1.)))
            replacements=tuple(OrientedEdge(child,chosen.forward)
                               for child in (children if chosen.forward else reversed(children)))
            compatible.append(tuple(part for item in loop
                                    for part in (replacements if item==chosen else (item,))))
        added=tuple(compatible)
        face=geometry.faces[face_id]
        geometry._put_entity('face',replace(face,holes=face.holes+added))
    return {'face':EntityRef('face',face_id)}
