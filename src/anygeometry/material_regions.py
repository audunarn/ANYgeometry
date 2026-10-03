"""Read-only material regions across qualified artificial extrusion seams.

Regions retain the source charts and exact boundary curves. They do not merge
document faces or discard physical joints, references or ownership transitions.
Only straight, oppositely traversed construction seams on identical extrusion
definitions and compatible structural occurrences are eligible for cancellation.
Other chart families and protected seams remain separate.
"""
from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

import numpy as np

from .arrangement_geometry import LinePath
from .definition_binding import definition_checksum
from .entities import EntityRef
from .errors import GeometryError
from .identity import EntityHandle
from .material_arrangement import ArrangementPath, MaterialDomain, arrange_material
from .serialization import _serialized_model_state
from .surfaces import ExtrudedSurface, _surface_derivatives_many
from .trimmed_charts import (TrimmedSurfaceChart, TrimmedSurfaceCharts, _check, _rows,
    query_trimmed_surface_charts, validate_trimmed_surface_charts_binding)


@dataclass(frozen=True, slots=True)
class MaterialSurfaceRegion:
    sources: tuple[TrimmedSurfaceChart, ...]
    domain: MaterialDomain
    cancelled_seams: tuple[EntityHandle, ...]
    interior_constraints: tuple[ArrangementPath, ...]
    retained_vertices: tuple[EntityHandle, ...]
    source_attachments: tuple[EntityHandle, ...]
    world_tolerance: float
    material_area: float

    @property
    def faces(self):
        return tuple(chart.face for chart in self.sources)

    @property
    def face_uses(self):
        return tuple(use for chart in self.sources for use in chart.face_uses)

    @property
    def support(self):
        return self.domain.support

    @property
    def boundaries(self):
        return self.domain.boundaries


@dataclass(frozen=True, slots=True)
class MaterialSurfaceRegions:
    source: TrimmedSurfaceCharts
    regions: tuple[MaterialSurfaceRegion, ...]


_validated = WeakKeyDictionary()


def _regions(model, source, cancellation_check):
    charts={chart.face.id: chart for chart in source.charts}
    groups_by_ref={}
    for name,refs in model.groups.items():
        for ref in refs:
            groups_by_ref.setdefault(ref,[]).append(name)
    protected_sources={attachment.source_id for attachment in model.attachments.values()
                       if attachment.source_kind=='edge'}
    keys={}
    for face_id,chart in charts.items():
        _check(cancellation_check, 'material region eligibility')
        support=chart.support
        if not isinstance(support, ExtrudedSurface) or not chart.face_uses:
            continue
        if any(hi<=lo for lo,hi in (support.u_range,support.v_range)):
            continue
        face=model.faces[face_id]
        if face.parameterization is not None:
            continue
        occurrences=tuple(sorted((model.face_uses[use.id].sheet_id,
                                 model.face_uses[use.id].orientation.value,
                                 definition_checksum(model.face_uses[use.id].metadata),
                                 model.tags_for(EntityRef('face_use',use.id)),
                                 tuple(sorted(groups_by_ref.get(EntityRef('face_use',use.id),()))))
                                for use in chart.face_uses))
        groups=tuple(sorted(groups_by_ref.get(face.ref,())))
        keys[face_id]=(definition_checksum(replace(support,u_range=(0.,1.),v_range=(0.,1.))),
                      occurrences,definition_checksum(face.metadata),model.tags_for(face.ref),groups)
    incidence={}
    for face_id in keys:
        for loop in (model.faces[face_id].loop, *model.faces[face_id].holes):
            for use in loop:
                incidence.setdefault(use.edge,[]).append((face_id,use))
    eligible={}
    adjacency={face:set() for face in charts}
    for edge_id, uses in sorted(incidence.items()):
        _check(cancellation_check, 'material region seam qualification')
        ref=EntityRef('edge',edge_id)
        if len(uses)!=2 or len(model.faces_using_edge(edge_id))!=2:
            continue
        (first,a),(second,b)=uses
        if first==second or keys[first]!=keys[second] or a.forward==b.forward:
            continue
        if model.tags_for(ref)!=('intersection_decomposition_seam',):
            continue
        if model.members_using_edge(edge_id) or model._target_attachments.get(('edge',edge_id)):
            continue
        if model._orientation_members.get(('edge',edge_id)):
            continue
        if any(model.coedges[use].metadata or model.tags_for(EntityRef('coedge',use)) or
               EntityRef('coedge',use) in groups_by_ref for use in model._edge_coedges.get(edge_id,())):
            continue
        if ref in groups_by_ref or edge_id in protected_sources:
            continue
        from .joint_edges import query_joint_edge
        if query_joint_edge(model,edge_id).declared:
            continue
        paths=[path for loop in charts[first].boundaries for path in loop if path.source_edge==edge_id]
        if len(paths)!=1 or not paths[0].decomposition or not isinstance(paths[0].curve,LinePath):
            continue
        eligible[edge_id]=(first,second)
        adjacency[first].add(second); adjacency[second].add(first)
    components=[]
    unseen=set(charts)
    for seed in sorted(charts):
        if seed not in unseen:
            continue
        pending=[seed]; component=set()
        while pending:
            _check(cancellation_check, 'material region connectivity')
            face=pending.pop()
            if face in component:
                continue
            component.add(face); pending.extend(sorted(adjacency[face]-component,reverse=True))
        unseen-=component; components.append(tuple(sorted(component)))
    component_by_face={face:index for index,component in enumerate(components) for face in component}
    seams_by_component={}
    for edge,faces in eligible.items():
        seams_by_component.setdefault(component_by_face[faces[0]],[]).append(edge)
    regions=[]
    for component_index,component in enumerate(components):
        sources=tuple(charts[face] for face in component)
        attachment_ids=tuple(sorted({attachment for face in component
                                    for attachment in model.attachments_for_face(face)}))
        attachments=tuple(model.handle('attachment',identifier) for identifier in attachment_ids)
        attachment_vertices={model.attachments[identifier].source_id for identifier in attachment_ids
                             if model.attachments[identifier].source_kind=='vertex'}
        seams=tuple(seams_by_component.get(component_index,()))
        seam_set=set(seams)
        if not seams:
            chart=sources[0]
            boundary_vertices={vertex for loop in (model.faces[component[0]].loop,*model.faces[component[0]].holes)
                               for use in loop for vertex in
                               (model.edges[use.edge].start,model.edges[use.edge].end)}
            retained=tuple(model.handle('vertex',vertex) for vertex in sorted(attachment_vertices-boundary_vertices))
            regions.append(MaterialSurfaceRegion(sources,chart.domain,(),(),retained,attachments,
                                                 chart.world_tolerance,chart.material_area))
            continue
        support=replace(sources[0].support,
            u_range=(min(c.support.u_range[0] for c in sources),max(c.support.u_range[1] for c in sources)),
            v_range=(min(c.support.v_range[0] for c in sources),max(c.support.v_range[1] for c in sources)))
        directed={}
        component_uses={}
        for chart in sources:
            for uses,paths in zip((model.faces[chart.face.id].loop,*model.faces[chart.face.id].holes),
                                  chart.boundaries):
                for use,path in zip(uses,paths):
                    component_uses.setdefault(use.edge,[]).append((use,path))
        interior=[]
        for edge,uses in sorted(component_uses.items()):
            if len(uses)>1:
                if len(uses)!=2 or uses[0][0].forward==uses[1][0].forward:
                    raise GeometryError('material region has ambiguous internal edge incidence')
                if edge not in seam_set:
                    interior.append(next(path for use,path in uses if use.forward))
        for chart in sources:
            face=model.faces[chart.face.id]
            for uses,paths in zip((face.loop, *face.holes),chart.boundaries):
                for use,path in zip(uses,paths):
                    if len(component_uses[use.edge])==2:
                        continue
                    start=model.oriented_start_vertex(use); end=model.oriented_end_vertex(use)
                    if start in directed:
                        raise GeometryError('material region has an ambiguous boundary junction')
                    directed[start]=(end,path)
        loops=[]; remaining=set(directed)
        while remaining:
            seed=min(remaining); station=seed; loop=[]
            while True:
                _check(cancellation_check, 'material region boundary traversal')
                if station not in remaining:
                    raise GeometryError('material region boundary is not a simple closed cycle')
                remaining.remove(station)
                station,path=directed[station]; loop.append(path)
                if station==seed:
                    break
            loops.append(tuple(loop))
        tolerance=max(chart.world_tolerance for chart in sources)
        domain=MaterialDomain(component[0],support,tuple(loops))
        signs=[domain.area_loop(loop,model.tolerance.area*.05) for loop in loops]
        source_sign=np.sign(domain.area_loop(sources[0].boundaries[0],model.tolerance.area*.05))
        outer=[index for index,area in enumerate(signs) if area*source_sign>0]
        if len(outer)!=1 or any(abs(area)<=model.tolerance.area for area in signs):
            raise GeometryError('material region needs one nondegenerate outer cycle')
        index=outer[0]
        domain=replace(domain,boundaries=(loops[index],*(loop for i,loop in enumerate(loops) if i!=index)))
        boxes=[path.curve.bounds() for loop in domain.boundaries for path in loop]
        length=float(np.linalg.norm(np.max([box[1] for box in boxes],axis=0)-
                                    np.min([box[0] for box in boxes],axis=0)))
        tolerance=model.tolerance.effective_length(length)
        area_tolerance=model.tolerance.effective_area(length)
        arrangement=arrange_material(domain,(),tolerance=tolerance,area_tolerance=area_tolerance,
            cancellation_check=lambda: (_check(cancellation_check,'material region exact arrangement') or False))
        area=domain.material_world_area(arrangement)
        if abs(area-sum(chart.material_area for chart in sources))>area_tolerance:
            raise GeometryError('material region does not conserve source material')
        boundary_vertices=set(directed)
        retained={vertex for seam in seams for vertex in (model.edges[seam].start,model.edges[seam].end)}
        retained.update(attachment_vertices)
        regions.append(MaterialSurfaceRegion(sources,domain,
            tuple(model.handle('edge',edge) for edge in seams),
            tuple(interior),
            tuple(model.handle('vertex',vertex) for vertex in sorted(retained-boundary_vertices)),
            attachments,
            tolerance,area))
    return tuple(regions)


def query_material_surface_regions(model, operands=None, *, expected_revision=None, cancellation_check=None):
    """Return exact source-bound regions without changing authored topology.

    Artificial straight extrusion seams may disappear from region boundaries;
    physical boundaries, protected references and ownership transitions remain.
    ``sources`` retains every original face and its parameter mapping. This is
    an additive consumer contract; document schema and existing chart APIs stay
    unchanged. Ambiguous boundary incidence fails with GeometryError.
    Consumers must honour ``interior_constraints``, ``retained_vertices`` and
    ``source_attachments`` and preserve authored face provenance. A region is
    not permission to discard a reference or assign a cell to one source face
    when its material spans multiple independently authored faces.
    """
    source=query_trimmed_surface_charts(model,operands,expected_revision=expected_revision,
                                      cancellation_check=cancellation_check)
    result=MaterialSurfaceRegions(source,_regions(model,source,cancellation_check))
    validate_material_surface_regions_binding(model,result,cancellation_check=cancellation_check)
    return result


def validate_material_surface_regions_binding(model,result,*,cancellation_check=None):
    if not isinstance(result,MaterialSurfaceRegions):
        raise GeometryError('material region binding needs a MaterialSurfaceRegions result')
    validate_trimmed_surface_charts_binding(model,result.source,cancellation_check=cancellation_check)
    signature=(result.source.source_checksum,definition_checksum(result))
    if signature in _validated.get(model,()):
        return
    expected=_regions(model,result.source,cancellation_check)
    if definition_checksum(expected)!=definition_checksum(result.regions):
        raise GeometryError('material region definition binding changed')
    _check(cancellation_check,'material region binding complete')
    if _serialized_model_state(model)['checksum']['value']!=result.source.source_checksum:
        raise GeometryError('material region source changed during validation')
    previous=tuple(item for item in _validated.get(model,()) if item[0]==signature[0])
    _validated[model]=(*previous[-7:],signature)


def evaluate_material_surface_region(model,result,face,parameters,*,derivatives=False,
                                     require_material=False,cancellation_check=None):
    """Evaluate the common support keyed by any of the region's source faces.

    Parameters belong to ``region.support``, not to an individual source chart.
    Setting derivatives returns the two exact differential arrays instead.
    """
    validate_material_surface_regions_binding(model,result,cancellation_check=cancellation_check)
    if not isinstance(face,int):
        if face.model_id!=model.model_id or face.kind!='face':
            raise GeometryError('material region evaluation requires a bound face')
        face=face.id
    region=next((region for region in result.regions if face in {handle.id for handle in region.faces}),None)
    if region is None:
        raise GeometryError('face is not selected by the material region binding')
    values,rows=_rows(parameters)
    positions=np.asarray([region.support.evaluate(*uv) for uv in rows],dtype=float).reshape(-1,3)
    if require_material:
        for point in positions:
            _check(cancellation_check,'material region membership')
            if not region.domain.contains(LinePath(tuple(point),tuple(point)),0.,region.world_tolerance):
                raise GeometryError('material region sample is outside material')
    arrays=_surface_derivatives_many(region.support,rows) if derivatives else (positions,)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise GeometryError('material region evaluation produced nonfinite results')
    validate_material_surface_regions_binding(model,result,cancellation_check=cancellation_check)
    arrays=tuple(array.reshape((*values.shape[:-1],3)).copy() for array in arrays)
    return arrays if derivatives else arrays[0]
