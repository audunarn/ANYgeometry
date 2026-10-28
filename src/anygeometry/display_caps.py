"""Transient cut polygons for an explicitly authored closed box volume.

This display query never promotes structural surface faces into material solids.
It accepts the canonical BoxVolume operand, not an unqualified triangle cloud.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from .volume_cut import BoxVolume, VolumeCutError


@dataclass(frozen=True)
class DisplayCap:
    plane_index: int
    vertices: np.ndarray


def box_volume_cut_caps(volume,planes):
    """Cap the intersection with positive halfspaces n.x >= offset.

    Return only newly cut polygons, with outward winding. Original volume and
    plane data are unchanged. Rendering triangulation does not create topology.
    """
    if not isinstance(volume,BoxVolume):raise VolumeCutError('CAP_SOURCE_UNQUALIFIED','caps require an explicit BoxVolume')
    origin=volume.world((np.asarray(volume.lower)+volume.upper)/2)[0]
    def admit(plane):
        try:normal,offset=plane;normal=np.asarray(normal,dtype=float);offset=float(offset)
        except (ValueError,TypeError) as error:raise VolumeCutError('CAP_PLANE_INVALID','planes require (normal, offset)') from error
        if normal.shape!=(3,) or not np.isfinite(normal).all() or not np.isfinite(offset) or not np.any(normal):
            raise VolumeCutError('CAP_PLANE_INVALID','cap plane must be finite with nonzero normal')
        magnitude=np.max(np.abs(normal));normal=normal/magnitude;length=np.linalg.norm(normal)
        normal=normal/length;offset=offset/magnitude/length-normal@origin
        if not np.isfinite(offset):raise VolumeCutError('CAP_PLANE_INVALID','normalized cap plane exceeds display range')
        return normal,offset
    prepared=tuple(admit(plane) for plane in planes)
    corners={bits:volume.world([np.where(bits,volume.upper,volume.lower)])[0]-origin for bits in product((0,1),repeat=3)}
    faces=[]
    for axis in range(3):
        others=[i for i in range(3) if i!=axis]
        for side in (0,1):
            points=[]
            for a,b in ((0,0),(1,0),(1,1),(0,1)):
                bits=[0,0,0];bits[axis]=side;bits[others[0]]=a;bits[others[1]]=b
                points.append(corners[tuple(bits)])
            faces.append((None,np.array(points)))
    tolerance=32*np.finfo(float).eps*max(volume.diagonal,1.)
    for index,(normal,offset) in enumerate(prepared):
        if faces:
            distances=np.concatenate([points@normal-offset for _,points in faces])
            if np.max(distances)<=0:return ()
            if np.min(distances)>=0:continue
        kept=[];crossings=[]
        for owner,polygon in faces:
            output=[]
            for first,last in zip(polygon,np.roll(polygon,-1,axis=0)):
                a=float(normal@first-offset);b=float(normal@last-offset)
                if a>=0:output.append(first)
                if (a<0<=b) or (b<0<=a):
                    point=first+(last-first)*(a/(a-b));output.append(point);crossings.append(point)
            clean=[]
            for point in output:
                if not clean or np.linalg.norm(point-clean[-1])>tolerance:clean.append(point)
            if len(clean)>1 and np.linalg.norm(clean[0]-clean[-1])<=tolerance:clean.pop()
            if len(clean)>=3:kept.append((owner,np.array(clean)))
        unique=[]
        for point in crossings:
            if not any(np.linalg.norm(point-other)<=tolerance for other in unique):unique.append(point)
        if len(unique)>=3:
            points=np.array(unique);center=points.mean(axis=0)
            axis=np.eye(3)[np.argmin(np.abs(normal))]
            u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
            relative=points-center
            order=np.argsort(np.arctan2(relative@v,relative@u))[::-1]
            kept.append((index,points[order]))
        faces=kept
        if not faces:break
    result=[]
    for owner,polygon in faces:
        if owner is not None:
            points=polygon+origin;points.setflags(write=False);result.append(DisplayCap(owner,points))
    return tuple(result)
