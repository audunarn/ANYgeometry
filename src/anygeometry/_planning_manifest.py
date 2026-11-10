"""Bounded loaded-callable closure; cache contents never become code authority.

This is engineering admission under the existing native runtime trust boundary,
not a proof of Python purity. Module objects bind namespaces; they do not make
every loaded application, UI or test symbol a numerical root.
"""
import ast
import builtins
import contextvars
import dataclasses
import dis
import enum
import functools
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import types
import weakref

from ._planning_qualification import QualificationDecline, NUMERICAL_SETTINGS, is_package_module, ModuleObserver
from ._planning_code import code_digest,passive_data

_THREADS=tuple(name for name in NUMERICAL_SETTINGS if 'THREAD' in name)
_INFRASTRUCTURE=frozenset(('anygeometry._planning_manifest','anygeometry._planning_controller',
    'anygeometry._planning_native','anygeometry._planning_process','anygeometry._planning_windows_launch'))


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _instructions(code):
    yield from dis.get_instructions(code)
    for value in code.co_consts:
        if type(value) is types.CodeType:yield from _instructions(value)


def _registry_factory_instructions(function,cache=None):
    """Constructed feature executors are definitions, not persistence calls."""
    from . import features
    if function is not features.builtin_feature_registry or function.__globals__ is not vars(features):
        raise QualificationDecline('feature registry factory binding changed')
    cache=_ImmutableEvidence() if cache is None else cache
    digest,_,_,codes=cache.source(features,function.__code__.co_filename,syntax=False)
    if digest!=_ROLE_MODEL_CALLER_SOURCE_PINS['host'][features.__name__]:
        raise QualificationDecline('feature registry source changed')
    expected=codes['builtin_feature_registry']
    if (cache.digest(function.__code__)!=cache.digest(expected) or function.__closure__ is not None or
            function.__defaults__ is not None or function.__kwdefaults__ is not None):
        raise QualificationDecline('feature registry factory code/default/closure changed')
    # code_digest above includes ALL nested executors and generator.execute.
    # Only generator construction is called; it captures its literal name in
    # the returned executor. No executor body is called by register or has.
    yield from cache.instructions(function.__code__,recursive=False)
    generator=next(code for code in function.__code__.co_consts if type(code) is types.CodeType and code.co_name=='generator')
    yield from cache.instructions(generator,recursive=False)


def _is_type(value):
    # Inspect the actual metaclass, never the instance's user __class__ property.
    return any(base is type for base in type.__getattribute__(type(value),'__mro__'))


def context_state(variable,path,role='host'):
    """Named effective-value policies; presence is not inferred from get(None)."""
    from . import analytic_roots,arrangement_geometry,runtime_diagnostics
    try:value=variable.get()
    except LookupError:value=None
    present=variable in contextvars.copy_context()
    if variable is arrangement_geometry._JUNCTION_BOUNDS:
        if value is not False:raise QualificationDecline('active junction bounds')
    elif variable is analytic_roots._ROOT_MEMO:
        if value is not None and not (role=='host' and type(value) is dict):
            raise QualificationDecline('active root memo outside H operation')
    elif variable in (runtime_diagnostics._COLLECTORS,runtime_diagnostics._RECORD_COLLECTORS):
        if type(value) is not tuple or any(type(item) is not dict for item in value):
            raise QualificationDecline('diagnostic collector schema')
        if role!='host' and value:raise QualificationDecline('worker collector leaked')
    elif present:
        raise QualificationDecline('active callable context: '+path)
    # Effective state is attested separately per role. The common numerical
    # fingerprint binds the variable and its policy, never mutable payloads.
    return variable.name


def _runtime_types():
    """Finite external type boundary, separate from owned record dispatch.

    These standard-library/numpy types use the existing runtime-image trust
    boundary. Their actual class/method bindings and Python implementations are
    still fingerprinted; no arbitrary external type is admitted by its prefix.
    """
    import collections,contextlib,decimal,fractions,numbers,pathlib,uuid
    import numpy as np
    return (fractions.Fraction,numbers.Number,numbers.Complex,numbers.Real,numbers.Rational,numbers.Integral,
        decimal.Decimal,uuid.UUID,pathlib.Path,pathlib.PurePath,pathlib.WindowsPath,pathlib.PureWindowsPath,
        pathlib.PosixPath,pathlib.PurePosixPath,collections.OrderedDict,collections.deque,
        contextlib.AbstractContextManager,contextlib.ContextDecorator,contextlib._GeneratorContextManager,
        functools.partial,enum.Enum,enum.EnumType,enum.IntEnum,enum.IntFlag,enum.Flag,enum.StrEnum,
        dataclasses.Field,dataclasses._DataclassParams,type(dataclasses.MISSING),
        np.ndarray,np.generic,np.number,np.inexact,np.floating,np.integer,np.signedinteger,
        np.float64,np.int64,np.bool_,np.dtype)


def _dataclass_slot_values(value):
    """Actual native slots; labels are edges, not invented tuple/name nodes."""
    typ=type(value)
    if typ is dataclasses.Field:
        names=('name','type','default','default_factory','repr','hash','init','compare',
               'metadata','kw_only',*(('doc',) if sys.version_info >= (3,14) else ()), '_field_type')
    elif typ is dataclasses._DataclassParams:
        names=('init','repr','eq','order','unsafe_hash','frozen',
               *(('match_args','kw_only','slots','weakref_slot') if sys.version_info >= (3,12) else ()))
    else:raise QualificationDecline('unregistered dataclass metadata type')
    slots=type.__getattribute__(typ,'__slots__')
    if type(slots) is not tuple or any(type(name) is not str for name in slots) or slots!=names:
        raise QualificationDecline('dataclass metadata slot schema changed')
    namespace=type.__getattribute__(typ,'__dict__')
    for name in names:
        descriptor=namespace.get(name)
        if (type(descriptor) is not types.MemberDescriptorType or descriptor.__objclass__ is not typ or
                descriptor.__name__!=name):
            raise QualificationDecline('dataclass metadata slot descriptor changed: '+name)
        yield name,descriptor,types.MemberDescriptorType.__get__(descriptor,value,typ)



# Source-reviewed operation interface. These are executable edges, not a waiver
# for unselected definitions: every owned class namespace remains fingerprinted.
_MODEL_OPERATION_METHODS = {'common': ('_angle_on_arc_sweep', '_arc_frame', '_capture_edge_caches', '_contains_entity', '_entity_bounds', '_entity_store', '_geometry_stamp', '_require_edge', '_require_face', '_require_vertex', '_spline_points', '_structural_store', 'arc_frame', 'conservative_face_bounds', 'face_corner_vertices', 'handle', 'oriented_end_vertex', 'oriented_start_vertex', 'tags_for', 'vertex_position'), 'host': ('_add_edge', '_allocate', '_allocate_structural', '_angle_on_arc_sweep', '_arc_frame', '_attach_entity', '_attach_structural_incidence', '_batch_topology_supported', '_builtin_local_uv_many', '_capture_edge_caches', '_capture_entity', '_capture_mapping', '_chain_data', '_chain_eval', '_check_plane_nurbs_ends', '_contains_entity', '_delete_structural', '_detach_entity', '_detach_structural_incidence', '_edge_parameter_derivative_object', '_entity_bounds', '_entity_store', '_face_local_uv_rows', '_geometry_stamp', '_install_feature_history', '_point_in_polygon', '_polygon_self_intersects', '_polygons_intersect', '_put_entity', '_put_structural', '_rebuild_incidence', '_rebuild_member_incidence', '_rebuild_structural_incidence', '_regularize_apex_uv', '_require_edge', '_require_face', '_require_nurbs_face_loops', '_require_vertex', '_sample_edge_object', '_segment_pairs_possible', '_segments_intersect_2d', '_spline_points', '_straight_loop_self_intersects_3d', '_structural_store', '_synchronize_face_uses', '_topology_coons_outer_loop_uv', '_topology_face_point', '_topology_face_point_and_derivatives', '_topology_face_points_many', '_validate_face_geometry', '_validate_nurbs_trims', '_validate_planar_straight_face_geometry', '_validate_planar_straight_quad_geometry', '_validate_replacement_history', '_validate_structural', '_validate_structural_replacement_history', '_validate_trim_polygons', '_validation_face_local_uv_many', '_validation_loop_points', 'add_arc', 'add_curve', 'add_line', 'add_point', 'add_spline', 'add_to_group', 'arc_frame', 'conservative_face_bounds', 'edge_length', 'entity_keys', 'entity_ref', 'face_corner_vertices', 'face_local_uv', 'face_local_uv_many', 'face_point', 'faces_using_edge', 'group', 'handle', 'id_state', 'mark_construction_vertices', 'oriented_end_vertex', 'oriented_start_vertex', 'replacement_history', 'resolve_handle', 'resolve_ref', 'resolve_structural_key', 'restore_id_state', 'sample_edge', 'structural_replacement_history', 'tag', 'transaction', 'validate_topology', 'vertex_position'), 'constructor': ('_add_edge', '_allocate', '_allocate_structural', '_angle_on_arc_sweep', '_arc_frame', '_attach_entity', '_attach_structural_incidence', '_batch_topology_supported', '_builtin_local_uv_many', '_capture_edge_caches', '_capture_entity', '_capture_mapping', '_chain_data', '_chain_eval', '_check_plane_nurbs_ends', '_contains_entity', '_delete_structural', '_detach_entity', '_detach_structural_incidence', '_edge_parameter_derivative_object', '_entity_bounds', '_entity_store', '_face_local_uv_rows', '_geometry_stamp', '_install_feature_history', '_point_in_polygon', '_polygon_self_intersects', '_polygons_intersect', '_put_entity', '_put_structural', '_rebuild_incidence', '_rebuild_member_incidence', '_rebuild_structural_incidence', '_regularize_apex_uv', '_require_edge', '_require_face', '_require_nurbs_face_loops', '_require_vertex', '_sample_edge_object', '_segment_pairs_possible', '_segments_intersect_2d', '_spline_points', '_straight_loop_self_intersects_3d', '_structural_store', '_synchronize_face_uses', '_topology_coons_outer_loop_uv', '_topology_face_point', '_topology_face_point_and_derivatives', '_topology_face_points_many', '_validate_face_geometry', '_validate_nurbs_trims', '_validate_planar_straight_face_geometry', '_validate_planar_straight_quad_geometry', '_validate_replacement_history', '_validate_structural', '_validate_structural_replacement_history', '_validate_trim_polygons', '_validation_face_local_uv_many', '_validation_loop_points', 'add_arc', 'add_curve', 'add_line', 'add_point', 'add_spline', 'add_to_group', 'conservative_face_bounds', 'edge_length', 'entity_keys', 'entity_ref', 'face_corner_vertices', 'face_local_uv', 'face_local_uv_many', 'face_point', 'faces_using_edge', 'group', 'handle', 'id_state', 'mark_construction_vertices', 'oriented_end_vertex', 'oriented_start_vertex', 'replacement_history', 'resolve_ref', 'restore_id_state', 'sample_edge', 'structural_replacement_history', 'tag', 'transaction', 'validate_topology', 'vertex_position')}
_VIEW_OPERATION_METHODS=frozenset(('__init__','__getattr__','chart_face','chart_edge','_connector_edge'))


_MODEL_CALLER_SOURCE_PINS = {'anygeometry.arrangement_geometry': '504574a1e4b7f8a37bbb96133fee6189522a0d9627c8b4214d17d8c3144ac18e', 'anygeometry.batch_intersections': '44e01224b0337127a7d8ffa60b3d593c6a484dd37b4c3c8d0b55cd2489ba9ee9', 'anygeometry.intersections': '2588de65b3741ca6332259112bb0bd383183dd6b3aebe3cee6d5d2a7ebca58df', 'anygeometry.material_arrangement': '0f9a9e786e96a21d1fa556866704bd905681387fef60360781fa43c5fe4fb066', 'anygeometry.model': '606bafa49bd5ff280f7d9694de1b312e19c61ce61a907c3ce4e06a94d53bd250', 'anygeometry.nurbs_sweep_chart': 'b89ff25198bab5cab66a3b0b9b8cc91fa7544f548d52a47cf9da06ef770c6f8a', 'anygeometry._translational_pairs': 'c92f8149fe6b742ad010a84b36bddedb252e0139618051171d4ed438366fa896', 'anygeometry': '1e7ea0a5dbf684d1f412115be6945b0a38d1624f67ee1c528ee89bc5f7bc9cf5'}
_MODEL_CALLER_EDGES = (('anygeometry._translational_pairs._sweep', 'model', 'vertex_position'), ('anygeometry._translational_pairs._sweep', 'model', 'oriented_start_vertex'), ('anygeometry._translational_pairs._sweep', 'model', 'vertex_position'), ('anygeometry._translational_pairs._sweep', 'model', 'oriented_end_vertex'), ('anygeometry._translational_pairs._sweep', 'model', 'vertex_position'), ('anygeometry._translational_pairs._sweep', 'model', 'oriented_end_vertex'), ('anygeometry._translational_pairs._sweep', 'model', 'vertex_position'), ('anygeometry._translational_pairs._sweep', 'model', 'oriented_start_vertex'), ('anygeometry.arrangement_geometry.freeze_edge', 'model', 'vertex_position'), ('anygeometry.arrangement_geometry.freeze_edge', 'model', 'vertex_position'), ('anygeometry.arrangement_geometry.freeze_edge', 'model', 'arc_frame'), ('anygeometry.arrangement_geometry.freeze_edge', 'model', '_spline_points'), ('anygeometry.batch_intersections._coplanar_traces', 'model', 'handle'), ('anygeometry.batch_intersections._coplanar_traces', 'model', 'handle'), ('anygeometry.intersections._exact_face_region', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._exact_face_support', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._exact_face_support', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._face_length_scale', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._face_length_scale', 'geometry', '_entity_bounds'), ('anygeometry.intersections._face_plane', 'geometry', 'vertex_position'), ('anygeometry.intersections._face_plane', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._face_polygon_in_plane', 'geometry', 'vertex_position'), ('anygeometry.intersections._face_polygon_in_plane', 'geometry', 'oriented_start_vertex'), ('anygeometry.material_arrangement.MaterialDomain.from_model', 'model', 'vertex_position'), ('anygeometry.material_arrangement.MaterialDomain.from_model', 'model', 'oriented_start_vertex'), ('anygeometry.material_arrangement.MaterialDomain.from_model', 'model', 'tags_for'), ('anygeometry.model.GeometryModel._arc_frame', 'self', '_geometry_stamp'), ('anygeometry.model.GeometryModel._arc_frame', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel._contains_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._contains_entity', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_angle_on_arc_sweep'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', 'conservative_face_bounds'), ('anygeometry.model.GeometryModel.arc_frame', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.arc_frame', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', '_entity_bounds'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', '_entity_bounds'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', 'face_corner_vertices'), ('anygeometry.model.GeometryModel.face_corner_vertices', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_corner_vertices', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel.handle', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.oriented_end_vertex', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.oriented_start_vertex', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.vertex_position', 'self', '_require_vertex'), ('anygeometry.nurbs_sweep_chart.SweepChartModel._connector_edge', 'self.raw', 'vertex_position'), ('anygeometry.nurbs_sweep_chart.SweepChartModel._connector_edge', 'self.raw', 'vertex_position'), ('anygeometry.nurbs_sweep_chart.SweepChartModel._connector_edge', 'self.raw', 'vertex_position'), ('anygeometry.nurbs_sweep_chart.SweepChartModel.chart_edge', 'self.raw', 'vertex_position'), ('anygeometry.nurbs_sweep_chart._qualified_sweep_chart_face', 'model', 'vertex_position'))


_ROLE_MODEL_CALLER_EDGES = {'host': (('anygeometry.arrangement_geometry.freeze_edge', 'model', '_spline_points'), ('anygeometry.arrangement_geometry.freeze_edge', 'model', 'arc_frame'), ('anygeometry.arrangement_geometry.freeze_edge', 'model', 'vertex_position'), ('anygeometry.batch_intersections._coplanar_traces', 'model', 'handle'), ('anygeometry.boundary_contacts.bilinear_boundary', 'model', 'oriented_start_vertex'), ('anygeometry.boundary_contacts.bilinear_boundary', 'model', 'vertex_position'), ('anygeometry.closure.extract_model_closure', 'work', '_allocate'), ('anygeometry.closure.extract_model_closure', 'work', '_allocate_structural'), ('anygeometry.closure.extract_model_closure', 'work', '_put_entity'), ('anygeometry.closure.extract_model_closure', 'work', '_put_structural'), ('anygeometry.closure.extract_model_closure', 'work', 'add_arc'), ('anygeometry.closure.extract_model_closure', 'work', 'add_curve'), ('anygeometry.closure.extract_model_closure', 'work', 'add_line'), ('anygeometry.closure.extract_model_closure', 'work', 'add_point'), ('anygeometry.closure.extract_model_closure', 'work', 'add_spline'), ('anygeometry.closure.extract_model_closure', 'work', 'add_to_group'), ('anygeometry.closure.extract_model_closure', 'work', 'handle'), ('anygeometry.closure.extract_model_closure', 'work', 'mark_construction_vertices'), ('anygeometry.closure.extract_model_closure', 'work', 'tag'), ('anygeometry.closure.extract_model_closure', 'work', 'transaction'), ('anygeometry.features.FeatureHistory.materialization_checksum', 'geometry', 'group'), ('anygeometry.features.FeatureHistory.materialization_checksum', 'geometry', 'resolve_ref'), ('anygeometry.features.FeatureHistory.validate_materialization', 'geometry', 'resolve_ref'), ('anygeometry.features.FeatureHistory.validate_persistence', 'geometry', 'resolve_ref'), ('anygeometry.intersections._exact_face_region', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._exact_face_support', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._face_length_scale', 'geometry', '_entity_bounds'), ('anygeometry.intersections._face_length_scale', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._face_polygon_in_plane', 'geometry', 'oriented_start_vertex'), ('anygeometry.intersections._face_polygon_in_plane', 'geometry', 'vertex_position'), ('anygeometry.intersections._normalize_operand', 'geometry', 'handle'), ('anygeometry.intersections._normalize_operand', 'geometry', 'resolve_handle'), ('anygeometry.model.GeometryModel._add_edge', 'self', '_allocate'), ('anygeometry.model.GeometryModel._add_edge', 'self', '_put_entity'), ('anygeometry.model.GeometryModel._arc_frame', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel._arc_frame', 'self', '_geometry_stamp'), ('anygeometry.model.GeometryModel._capture_entity', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel._capture_entity', 'self', '_entity_bounds'), ('anygeometry.model.GeometryModel._capture_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._chain_data', 'self', '_require_edge'), ('anygeometry.model.GeometryModel._chain_data', 'self', 'edge_length'), ('anygeometry.model.GeometryModel._chain_eval', 'self', '_edge_parameter_derivative_object'), ('anygeometry.model.GeometryModel._chain_eval', 'self', '_sample_edge_object'), ('anygeometry.model.GeometryModel._contains_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._contains_entity', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._delete_structural', 'self', '_detach_structural_incidence'), ('anygeometry.model.GeometryModel._delete_structural', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._edge_parameter_derivative_object', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._edge_parameter_derivative_object', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_angle_on_arc_sweep'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', 'conservative_face_bounds'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', '_batch_topology_supported'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', '_require_face'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', '_topology_face_points_many'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', 'face_local_uv'), ('anygeometry.model.GeometryModel._polygon_self_intersects', 'cls', '_segment_pairs_possible'), ('anygeometry.model.GeometryModel._polygon_self_intersects', 'cls', '_segments_intersect_2d'), ('anygeometry.model.GeometryModel._polygons_intersect', 'cls', '_segment_pairs_possible'), ('anygeometry.model.GeometryModel._polygons_intersect', 'cls', '_segments_intersect_2d'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_attach_entity'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_capture_entity'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_detach_entity'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_synchronize_face_uses'), ('anygeometry.model.GeometryModel._put_structural', 'self', '_attach_structural_incidence'), ('anygeometry.model.GeometryModel._put_structural', 'self', '_detach_structural_incidence'), ('anygeometry.model.GeometryModel._put_structural', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._rebuild_incidence', 'self', '_attach_entity'), ('anygeometry.model.GeometryModel._rebuild_member_incidence', 'self', '_rebuild_structural_incidence'), ('anygeometry.model.GeometryModel._rebuild_structural_incidence', 'self', '_attach_structural_incidence'), ('anygeometry.model.GeometryModel._sample_edge_object', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._sample_edge_object', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._synchronize_face_uses', 'self', '_allocate_structural'), ('anygeometry.model.GeometryModel._synchronize_face_uses', 'self', '_delete_structural'), ('anygeometry.model.GeometryModel._synchronize_face_uses', 'self', '_put_structural'), ('anygeometry.model.GeometryModel._topology_coons_outer_loop_uv', 'self', '_require_face'), ('anygeometry.model.GeometryModel._topology_coons_outer_loop_uv', 'self', 'edge_length'), ('anygeometry.model.GeometryModel._topology_face_point', 'self', '_topology_face_point_and_derivatives'), ('anygeometry.model.GeometryModel._topology_face_point_and_derivatives', 'self', '_chain_data'), ('anygeometry.model.GeometryModel._topology_face_point_and_derivatives', 'self', '_chain_eval'), ('anygeometry.model.GeometryModel._topology_face_point_and_derivatives', 'self', '_require_face'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_chain_data'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_chain_eval'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_require_face'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_sample_edge_object'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_topology_face_point'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_builtin_local_uv_many'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_regularize_apex_uv'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_require_nurbs_face_loops'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_straight_loop_self_intersects_3d'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_topology_coons_outer_loop_uv'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_nurbs_trims'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_planar_straight_face_geometry'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_planar_straight_quad_geometry'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_trim_polygons'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validation_face_local_uv_many'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validation_loop_points'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', '_point_in_polygon'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', '_polygon_self_intersects'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', '_polygons_intersect'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel._validate_planar_straight_quad_geometry', 'self', '_polygon_self_intersects'), ('anygeometry.model.GeometryModel._validate_planar_straight_quad_geometry', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel._validate_replacement_history', 'self', 'entity_keys'), ('anygeometry.model.GeometryModel._validate_structural', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel._validate_structural', 'self', '_validate_structural_replacement_history'), ('anygeometry.model.GeometryModel._validate_structural_replacement_history', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel._validate_trim_polygons', 'self', '_point_in_polygon'), ('anygeometry.model.GeometryModel._validate_trim_polygons', 'self', '_polygon_self_intersects'), ('anygeometry.model.GeometryModel._validate_trim_polygons', 'self', '_polygons_intersect'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', '_batch_topology_supported'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', '_require_face'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', '_topology_face_points_many'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', 'face_local_uv_many'), ('anygeometry.model.GeometryModel._validation_loop_points', 'self', 'sample_edge'), ('anygeometry.model.GeometryModel.add_arc', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_arc', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.add_curve', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_curve', 'self', '_check_plane_nurbs_ends'), ('anygeometry.model.GeometryModel.add_curve', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.add_line', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_line', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.add_point', 'self', '_allocate'), ('anygeometry.model.GeometryModel.add_point', 'self', '_put_entity'), ('anygeometry.model.GeometryModel.add_spline', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_spline', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.add_to_group', 'self', '_capture_mapping'), ('anygeometry.model.GeometryModel.add_to_group', 'self', 'entity_ref'), ('anygeometry.model.GeometryModel.arc_frame', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel.arc_frame', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', '_entity_bounds'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', 'face_corner_vertices'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_spline_points'), ('anygeometry.model.GeometryModel.entity_ref', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.entity_ref', 'self', '_require_face'), ('anygeometry.model.GeometryModel.entity_ref', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.face_corner_vertices', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_corner_vertices', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel.face_local_uv', 'model', 'face_point'), ('anygeometry.model.GeometryModel.face_local_uv', 'self', '_batch_topology_supported'), ('anygeometry.model.GeometryModel.face_local_uv', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_local_uv', 'self', '_topology_face_points_many'), ('anygeometry.model.GeometryModel.face_local_uv_many', 'self', '_builtin_local_uv_many'), ('anygeometry.model.GeometryModel.face_local_uv_many', 'self', '_face_local_uv_rows'), ('anygeometry.model.GeometryModel.face_local_uv_many', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_point', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_point', 'self', '_topology_face_point'), ('anygeometry.model.GeometryModel.faces_using_edge', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.group', 'self', 'resolve_ref'), ('anygeometry.model.GeometryModel.handle', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.mark_construction_vertices', 'self', '_capture_mapping'), ('anygeometry.model.GeometryModel.mark_construction_vertices', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.oriented_end_vertex', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.oriented_start_vertex', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.resolve_handle', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.resolve_handle', 'self', 'resolve_ref'), ('anygeometry.model.GeometryModel.resolve_handle', 'self', 'resolve_structural_key'), ('anygeometry.model.GeometryModel.resolve_ref', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.resolve_structural_key', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.sample_edge', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.sample_edge', 'self', '_sample_edge_object'), ('anygeometry.model.GeometryModel.tag', 'self', '_capture_mapping'), ('anygeometry.model.GeometryModel.tag', 'self', 'entity_ref'), ('anygeometry.model.GeometryModel.validate_topology', 'self', '_validate_face_geometry'), ('anygeometry.model.GeometryModel.validate_topology', 'self', '_validate_replacement_history'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'entity_keys'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'oriented_end_vertex'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.vertex_position', 'self', '_require_vertex'), ('anygeometry.nurbs_section_faces.validate_nurbs_section_face', 'model', 'oriented_start_vertex'), ('anygeometry.nurbs_sweep_chart.SweepChartModel._connector_edge', 'self.raw', 'vertex_position'), ('anygeometry.nurbs_sweep_chart.SweepChartModel.chart_edge', 'self.raw', 'vertex_position'), ('anygeometry.plane_section_plate._curve_box', 'model', 'vertex_position'), ('anygeometry.plane_section_plate._line_endpoints', 'model', 'oriented_end_vertex'), ('anygeometry.plane_section_plate._line_endpoints', 'model', 'oriented_start_vertex'), ('anygeometry.plane_section_plate._line_endpoints', 'model', 'vertex_position'), ('anygeometry.plane_section_plate._translation_cycle_embedding', 'model', 'oriented_end_vertex'), ('anygeometry.plane_section_plate._translation_cycle_embedding', 'model', 'oriented_start_vertex'), ('anygeometry.plane_section_plate.validate_complete_path_face', 'model', 'oriented_end_vertex'), ('anygeometry.plane_section_plate.validate_complete_path_face', 'model', 'oriented_start_vertex'), ('anygeometry.plane_section_plate.validate_complete_path_face', 'model', 'vertex_position'), ('anygeometry.section_faces._edge_points', 'model', 'faces_using_edge'), ('anygeometry.section_faces._retained_line_pair_verdict', 'model', 'oriented_end_vertex'), ('anygeometry.section_faces._retained_line_pair_verdict', 'model', 'oriented_start_vertex'), ('anygeometry.section_faces.loop_area', 'model', 'oriented_start_vertex'), ('anygeometry.section_faces.validate_plane_section_face', 'model', '_segments_intersect_2d'), ('anygeometry.section_faces.validate_plane_section_face', 'model', 'oriented_start_vertex'), ('anygeometry.serialization._decode_geometry_records', 'geometry', '_check_plane_nurbs_ends'), ('anygeometry.serialization._decode_geometry_records', 'geometry', '_put_entity'), ('anygeometry.serialization._decode_structural', 'geometry', '_put_structural'), ('anygeometry.serialization._migrate_legacy_structural', 'geometry', '_put_structural'), ('anygeometry.serialization._migrate_legacy_structural', 'geometry', '_structural_store'), ('anygeometry.serialization._qualify_document', 'geometry', '_validate_structural'), ('anygeometry.serialization._qualify_document', 'geometry', 'validate_topology'), ('anygeometry.serialization._restore_counters', 'geometry', '_entity_store'), ('anygeometry.serialization._restore_counters', 'geometry', '_structural_store'), ('anygeometry.serialization._restore_counters', 'geometry', 'restore_id_state'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'group'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'id_state'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'replacement_history'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'structural_replacement_history'), ('anygeometry.serialization.from_dict', 'geometry', '_install_feature_history'), ('anygeometry.serialization.from_dict', 'geometry', '_rebuild_incidence'), ('anygeometry.serialization.from_dict', 'geometry', '_rebuild_member_incidence'), ('anygeometry.serialization.from_dict', 'geometry', '_validate_structural'), ('anygeometry.serialization.from_dict', 'geometry', 'entity_keys'), ('anygeometry.serialization.from_dict', 'geometry', 'id_state'), ('anygeometry.serialization.from_dict', 'geometry', 'replacement_history'), ('anygeometry.serialization.from_dict', 'geometry', 'validate_topology')), 'constructor': (('anygeometry.closure.extract_model_closure', 'work', '_allocate'), ('anygeometry.closure.extract_model_closure', 'work', '_allocate_structural'), ('anygeometry.closure.extract_model_closure', 'work', '_put_entity'), ('anygeometry.closure.extract_model_closure', 'work', '_put_structural'), ('anygeometry.closure.extract_model_closure', 'work', 'add_arc'), ('anygeometry.closure.extract_model_closure', 'work', 'add_curve'), ('anygeometry.closure.extract_model_closure', 'work', 'add_line'), ('anygeometry.closure.extract_model_closure', 'work', 'add_point'), ('anygeometry.closure.extract_model_closure', 'work', 'add_spline'), ('anygeometry.closure.extract_model_closure', 'work', 'add_to_group'), ('anygeometry.closure.extract_model_closure', 'work', 'handle'), ('anygeometry.closure.extract_model_closure', 'work', 'mark_construction_vertices'), ('anygeometry.closure.extract_model_closure', 'work', 'tag'), ('anygeometry.closure.extract_model_closure', 'work', 'transaction'), ('anygeometry.features.FeatureHistory.materialization_checksum', 'geometry', 'group'), ('anygeometry.features.FeatureHistory.materialization_checksum', 'geometry', 'resolve_ref'), ('anygeometry.features.FeatureHistory.validate_materialization', 'geometry', 'resolve_ref'), ('anygeometry.features.FeatureHistory.validate_persistence', 'geometry', 'resolve_ref'), ('anygeometry.model.GeometryModel._add_edge', 'self', '_allocate'), ('anygeometry.model.GeometryModel._add_edge', 'self', '_put_entity'), ('anygeometry.model.GeometryModel._arc_frame', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel._arc_frame', 'self', '_geometry_stamp'), ('anygeometry.model.GeometryModel._capture_entity', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel._capture_entity', 'self', '_entity_bounds'), ('anygeometry.model.GeometryModel._capture_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._chain_data', 'self', '_require_edge'), ('anygeometry.model.GeometryModel._chain_data', 'self', 'edge_length'), ('anygeometry.model.GeometryModel._chain_eval', 'self', '_edge_parameter_derivative_object'), ('anygeometry.model.GeometryModel._chain_eval', 'self', '_sample_edge_object'), ('anygeometry.model.GeometryModel._contains_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._contains_entity', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._delete_structural', 'self', '_detach_structural_incidence'), ('anygeometry.model.GeometryModel._delete_structural', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._edge_parameter_derivative_object', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._edge_parameter_derivative_object', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_angle_on_arc_sweep'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._entity_bounds', 'self', 'conservative_face_bounds'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', '_batch_topology_supported'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', '_require_face'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', '_topology_face_points_many'), ('anygeometry.model.GeometryModel._face_local_uv_rows', 'self', 'face_local_uv'), ('anygeometry.model.GeometryModel._polygon_self_intersects', 'cls', '_segment_pairs_possible'), ('anygeometry.model.GeometryModel._polygon_self_intersects', 'cls', '_segments_intersect_2d'), ('anygeometry.model.GeometryModel._polygons_intersect', 'cls', '_segment_pairs_possible'), ('anygeometry.model.GeometryModel._polygons_intersect', 'cls', '_segments_intersect_2d'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_attach_entity'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_capture_entity'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_detach_entity'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_entity_store'), ('anygeometry.model.GeometryModel._put_entity', 'self', '_synchronize_face_uses'), ('anygeometry.model.GeometryModel._put_structural', 'self', '_attach_structural_incidence'), ('anygeometry.model.GeometryModel._put_structural', 'self', '_detach_structural_incidence'), ('anygeometry.model.GeometryModel._put_structural', 'self', '_structural_store'), ('anygeometry.model.GeometryModel._rebuild_incidence', 'self', '_attach_entity'), ('anygeometry.model.GeometryModel._rebuild_member_incidence', 'self', '_rebuild_structural_incidence'), ('anygeometry.model.GeometryModel._rebuild_structural_incidence', 'self', '_attach_structural_incidence'), ('anygeometry.model.GeometryModel._sample_edge_object', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel._sample_edge_object', 'self', '_spline_points'), ('anygeometry.model.GeometryModel._synchronize_face_uses', 'self', '_allocate_structural'), ('anygeometry.model.GeometryModel._synchronize_face_uses', 'self', '_delete_structural'), ('anygeometry.model.GeometryModel._synchronize_face_uses', 'self', '_put_structural'), ('anygeometry.model.GeometryModel._topology_coons_outer_loop_uv', 'self', '_require_face'), ('anygeometry.model.GeometryModel._topology_coons_outer_loop_uv', 'self', 'edge_length'), ('anygeometry.model.GeometryModel._topology_face_point', 'self', '_topology_face_point_and_derivatives'), ('anygeometry.model.GeometryModel._topology_face_point_and_derivatives', 'self', '_chain_data'), ('anygeometry.model.GeometryModel._topology_face_point_and_derivatives', 'self', '_chain_eval'), ('anygeometry.model.GeometryModel._topology_face_point_and_derivatives', 'self', '_require_face'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_chain_data'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_chain_eval'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_require_face'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_sample_edge_object'), ('anygeometry.model.GeometryModel._topology_face_points_many', 'self', '_topology_face_point'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_builtin_local_uv_many'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_regularize_apex_uv'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_require_nurbs_face_loops'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_straight_loop_self_intersects_3d'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_topology_coons_outer_loop_uv'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_nurbs_trims'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_planar_straight_face_geometry'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_planar_straight_quad_geometry'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validate_trim_polygons'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validation_face_local_uv_many'), ('anygeometry.model.GeometryModel._validate_face_geometry', 'self', '_validation_loop_points'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', '_point_in_polygon'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', '_polygon_self_intersects'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', '_polygons_intersect'), ('anygeometry.model.GeometryModel._validate_planar_straight_face_geometry', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel._validate_planar_straight_quad_geometry', 'self', '_polygon_self_intersects'), ('anygeometry.model.GeometryModel._validate_planar_straight_quad_geometry', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel._validate_replacement_history', 'self', 'entity_keys'), ('anygeometry.model.GeometryModel._validate_structural', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel._validate_structural', 'self', '_validate_structural_replacement_history'), ('anygeometry.model.GeometryModel._validate_structural_replacement_history', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel._validate_trim_polygons', 'self', '_point_in_polygon'), ('anygeometry.model.GeometryModel._validate_trim_polygons', 'self', '_polygon_self_intersects'), ('anygeometry.model.GeometryModel._validate_trim_polygons', 'self', '_polygons_intersect'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', '_batch_topology_supported'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', '_require_face'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', '_topology_face_points_many'), ('anygeometry.model.GeometryModel._validation_face_local_uv_many', 'self', 'face_local_uv_many'), ('anygeometry.model.GeometryModel._validation_loop_points', 'self', 'sample_edge'), ('anygeometry.model.GeometryModel.add_arc', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_arc', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.add_curve', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_curve', 'self', '_check_plane_nurbs_ends'), ('anygeometry.model.GeometryModel.add_curve', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.add_line', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_line', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.add_point', 'self', '_allocate'), ('anygeometry.model.GeometryModel.add_point', 'self', '_put_entity'), ('anygeometry.model.GeometryModel.add_spline', 'self', '_add_edge'), ('anygeometry.model.GeometryModel.add_spline', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.add_to_group', 'self', '_capture_mapping'), ('anygeometry.model.GeometryModel.add_to_group', 'self', 'entity_ref'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', '_entity_bounds'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', 'face_corner_vertices'), ('anygeometry.model.GeometryModel.conservative_face_bounds', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_arc_frame'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_capture_edge_caches'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.edge_length', 'self', '_spline_points'), ('anygeometry.model.GeometryModel.entity_ref', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.entity_ref', 'self', '_require_face'), ('anygeometry.model.GeometryModel.entity_ref', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.face_corner_vertices', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_corner_vertices', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel.face_local_uv', 'model', 'face_point'), ('anygeometry.model.GeometryModel.face_local_uv', 'self', '_batch_topology_supported'), ('anygeometry.model.GeometryModel.face_local_uv', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_local_uv', 'self', '_topology_face_points_many'), ('anygeometry.model.GeometryModel.face_local_uv_many', 'self', '_builtin_local_uv_many'), ('anygeometry.model.GeometryModel.face_local_uv_many', 'self', '_face_local_uv_rows'), ('anygeometry.model.GeometryModel.face_local_uv_many', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_point', 'self', '_require_face'), ('anygeometry.model.GeometryModel.face_point', 'self', '_topology_face_point'), ('anygeometry.model.GeometryModel.faces_using_edge', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.group', 'self', 'resolve_ref'), ('anygeometry.model.GeometryModel.handle', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.mark_construction_vertices', 'self', '_capture_mapping'), ('anygeometry.model.GeometryModel.mark_construction_vertices', 'self', '_require_vertex'), ('anygeometry.model.GeometryModel.oriented_end_vertex', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.oriented_start_vertex', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.resolve_ref', 'self', '_contains_entity'), ('anygeometry.model.GeometryModel.sample_edge', 'self', '_require_edge'), ('anygeometry.model.GeometryModel.sample_edge', 'self', '_sample_edge_object'), ('anygeometry.model.GeometryModel.tag', 'self', '_capture_mapping'), ('anygeometry.model.GeometryModel.tag', 'self', 'entity_ref'), ('anygeometry.model.GeometryModel.validate_topology', 'self', '_validate_face_geometry'), ('anygeometry.model.GeometryModel.validate_topology', 'self', '_validate_replacement_history'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'entity_keys'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'oriented_end_vertex'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'oriented_start_vertex'), ('anygeometry.model.GeometryModel.validate_topology', 'self', 'vertex_position'), ('anygeometry.model.GeometryModel.vertex_position', 'self', '_require_vertex'), ('anygeometry.nurbs_section_faces.validate_nurbs_section_face', 'model', 'oriented_start_vertex'), ('anygeometry.nurbs_sweep_chart.SweepChartModel._connector_edge', 'self.raw', 'vertex_position'), ('anygeometry.nurbs_sweep_chart.SweepChartModel.chart_edge', 'self.raw', 'vertex_position'), ('anygeometry.plane_section_plate._curve_box', 'model', 'vertex_position'), ('anygeometry.plane_section_plate._line_endpoints', 'model', 'oriented_end_vertex'), ('anygeometry.plane_section_plate._line_endpoints', 'model', 'oriented_start_vertex'), ('anygeometry.plane_section_plate._line_endpoints', 'model', 'vertex_position'), ('anygeometry.plane_section_plate._translation_cycle_embedding', 'model', 'oriented_end_vertex'), ('anygeometry.plane_section_plate._translation_cycle_embedding', 'model', 'oriented_start_vertex'), ('anygeometry.plane_section_plate.validate_complete_path_face', 'model', 'oriented_end_vertex'), ('anygeometry.plane_section_plate.validate_complete_path_face', 'model', 'oriented_start_vertex'), ('anygeometry.plane_section_plate.validate_complete_path_face', 'model', 'vertex_position'), ('anygeometry.section_faces._edge_points', 'model', 'faces_using_edge'), ('anygeometry.section_faces._retained_line_pair_verdict', 'model', 'oriented_end_vertex'), ('anygeometry.section_faces._retained_line_pair_verdict', 'model', 'oriented_start_vertex'), ('anygeometry.section_faces.loop_area', 'model', 'oriented_start_vertex'), ('anygeometry.section_faces.validate_plane_section_face', 'model', '_segments_intersect_2d'), ('anygeometry.section_faces.validate_plane_section_face', 'model', 'oriented_start_vertex'), ('anygeometry.serialization._decode_geometry_records', 'geometry', '_check_plane_nurbs_ends'), ('anygeometry.serialization._decode_geometry_records', 'geometry', '_put_entity'), ('anygeometry.serialization._decode_structural', 'geometry', '_put_structural'), ('anygeometry.serialization._migrate_legacy_structural', 'geometry', '_put_structural'), ('anygeometry.serialization._migrate_legacy_structural', 'geometry', '_structural_store'), ('anygeometry.serialization._qualify_document', 'geometry', '_validate_structural'), ('anygeometry.serialization._qualify_document', 'geometry', 'validate_topology'), ('anygeometry.serialization._restore_counters', 'geometry', '_entity_store'), ('anygeometry.serialization._restore_counters', 'geometry', '_structural_store'), ('anygeometry.serialization._restore_counters', 'geometry', 'restore_id_state'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'group'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'id_state'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'replacement_history'), ('anygeometry.serialization._serialized_model_state', 'geometry', 'structural_replacement_history'), ('anygeometry.serialization.from_dict', 'geometry', '_install_feature_history'), ('anygeometry.serialization.from_dict', 'geometry', '_rebuild_incidence'), ('anygeometry.serialization.from_dict', 'geometry', '_rebuild_member_incidence'), ('anygeometry.serialization.from_dict', 'geometry', '_validate_structural'), ('anygeometry.serialization.from_dict', 'geometry', 'entity_keys'), ('anygeometry.serialization.from_dict', 'geometry', 'id_state'), ('anygeometry.serialization.from_dict', 'geometry', 'replacement_history'), ('anygeometry.serialization.from_dict', 'geometry', 'validate_topology'))}
_ROLE_MODEL_CALLER_SOURCE_PINS = {'host': {'anygeometry.arrangement_geometry': '504574a1e4b7f8a37bbb96133fee6189522a0d9627c8b4214d17d8c3144ac18e', 'anygeometry.batch_intersections': '44e01224b0337127a7d8ffa60b3d593c6a484dd37b4c3c8d0b55cd2489ba9ee9', 'anygeometry.boundary_contacts': '53b81147663fbb27bae07f60687dc914c092406240ec596ee7217ae1f83cc2a7', 'anygeometry.closure': 'd58d31bbd39fd397ab6b7e059ff058ebddf92e0b123780bc642caf566d3d713a', 'anygeometry.features': '19976c0cc4b357d5b1dbddaa8a8f73f885a6b98f3ec9c6ab4a14badc350e85ee', 'anygeometry.intersections': '2588de65b3741ca6332259112bb0bd383183dd6b3aebe3cee6d5d2a7ebca58df', 'anygeometry.model': '606bafa49bd5ff280f7d9694de1b312e19c61ce61a907c3ce4e06a94d53bd250', 'anygeometry.nurbs_section_faces': '6e72989a338f0c17bcd555366db7cfd64ff81bfa342f16495ff489d30c8187fd', 'anygeometry.nurbs_sweep_chart': 'b89ff25198bab5cab66a3b0b9b8cc91fa7544f548d52a47cf9da06ef770c6f8a', 'anygeometry.plane_section_plate': '2c2293529a30d0e57af19127e1076db702fd8df2e4588404cfafbb1295cc5648', 'anygeometry.section_faces': 'fe36ac1f88701acc0fd2ab3c45e4d6de6b0c9af7d2b4ccf8375a824d863d20fe', 'anygeometry.serialization': '641c0ebbc2c27d4e22ad99eec51617e3d9c2e85040b54d6f883a8e38ceeb3154'}, 'constructor': {'anygeometry.closure': 'd58d31bbd39fd397ab6b7e059ff058ebddf92e0b123780bc642caf566d3d713a', 'anygeometry.features': '19976c0cc4b357d5b1dbddaa8a8f73f885a6b98f3ec9c6ab4a14badc350e85ee', 'anygeometry.model': '606bafa49bd5ff280f7d9694de1b312e19c61ce61a907c3ce4e06a94d53bd250', 'anygeometry.nurbs_section_faces': '6e72989a338f0c17bcd555366db7cfd64ff81bfa342f16495ff489d30c8187fd', 'anygeometry.nurbs_sweep_chart': 'b89ff25198bab5cab66a3b0b9b8cc91fa7544f548d52a47cf9da06ef770c6f8a', 'anygeometry.plane_section_plate': '2c2293529a30d0e57af19127e1076db702fd8df2e4588404cfafbb1295cc5648', 'anygeometry.section_faces': 'fe36ac1f88701acc0fd2ab3c45e4d6de6b0c9af7d2b4ccf8375a824d863d20fe', 'anygeometry.serialization': '641c0ebbc2c27d4e22ad99eec51617e3d9c2e85040b54d6f883a8e38ceeb3154'}}


class _ImmutableEvidence:
    """Reuse immutable compiled source/code, while hashing live files each boundary."""
    def __init__(self,*,cache_traversal=True):
        if type(cache_traversal) is not bool:raise TypeError('cache_traversal must be bool')
        self.cache_traversal=cache_traversal
        self.compiled={};self.fingerprints={};self.current={};self.source_pins={};self._class_declarations={}
        self.path_cache={};self.instruction_cache={};self.path_observations={};self.instruction_slots=0
    def begin(self):
        self.current={};self.path_cache.clear();self.instruction_cache.clear();self.path_observations.clear();self.instruction_slots=0
    def resolve(self,value):
        # Memoization never replaces original accepted path bindings.
        if type(value) is not str or not Path(value).is_absolute():
            return Path(value).resolve()
        if self.cache_traversal and value in self.path_cache:
            resolved=self.path_cache[value]
        else:
            resolved=str(Path(value).resolve())
            if self.cache_traversal:self.path_cache[value]=resolved
        previous=self.path_observations.setdefault(value,resolved)
        if previous!=resolved:raise QualificationDecline('source path resolution changed: '+value)
        return Path(resolved)
    def verify_paths(self):
        for spelling,resolved in self.path_observations.items():
            if str(Path(spelling).resolve())!=resolved:
                raise QualificationDecline('source path resolution changed: '+spelling)
    def instructions(self,code,*,recursive=True):
        if type(code) is not types.CodeType:raise TypeError('instruction evidence requires exact code')
        if not self.cache_traversal:
            return tuple(_instructions(code) if recursive else dis.get_instructions(code))
        key=(id(code),recursive)
        if key in self.instruction_cache:
            original,ops=self.instruction_cache[key]
            if original is not code:raise QualificationDecline('immutable instruction identity recycled')
            return ops
        ops=tuple(dis.get_instructions(code))
        if recursive:
            ops+=tuple(op for child in code.co_consts if type(child) is types.CodeType
                for op in self.instructions(child))
        # Performance storage only: no instruction or dependency is omitted
        # when the shared assembly budget is exhausted or a body is oversized.
        if self.instruction_slots+len(ops)<=4096:
            self.instruction_cache[key]=(code,ops)
            self.instruction_slots+=len(ops)
        return ops
    def source(self,owner,filename,*,syntax=True):
        if type(syntax) is not bool:raise TypeError('source syntax demand must be bool')
        path=self.resolve(owner.__file__)
        if path not in self.current:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if path in self.source_pins and self.source_pins[path]!=digest:
                raise QualificationDecline('immutable source evidence changed: '+str(path))
            self.source_pins[path]=digest;self.current[path]=(raw,digest)
        raw,digest=self.current[path];key=(str(path),filename,digest)
        if key not in self.compiled:
            root=compile(raw,filename,'exec',dont_inherit=True)
            codes={};pending=[root]
            while pending:
                code=pending.pop();codes[code.co_qualname]=code
                codes[(code.co_qualname,code.co_firstlineno)]=code
                pending.extend(item for item in code.co_consts if type(item) is types.CodeType)
            self.compiled[key]=(digest,None,{},codes)
        if syntax and self.compiled[key][1] is None:
            tree=ast.parse(raw);codes=self.compiled[key][3]
            nodes={}
            for declaration in tree.body:
                if type(declaration) is ast.ClassDef:
                    for body in declaration.body:
                        if type(body) is ast.FunctionDef:nodes[declaration.name+'.'+body.name]=body
            self.compiled[key]=(digest,tree,nodes,codes)
        return self.compiled[key]
    def class_declarations(self,owner,filename):
        """Exact qualified ClassDef names from pinned syntax, without retaining ASTs."""
        digest,tree,_,_=self.source(owner,filename,syntax=False)
        path=self.resolve(owner.__file__);key=(str(path),filename,digest)
        if key not in self._class_declarations:
            if tree is None:tree=ast.parse(self.current[path][0])
            names=set()
            def visit(node,scope):
                if type(node) is ast.ClassDef:
                    scope=scope+node.name;names.add(scope);scope+='.'
                elif type(node) in (ast.FunctionDef,ast.AsyncFunctionDef):
                    scope+=node.name+'.<locals>.'
                for child in ast.iter_child_nodes(node):visit(child,scope)
            visit(tree,'')
            self._class_declarations[key]=frozenset(names)
        return self._class_declarations[key]
    def release_parsing(self):
        """Discard only owned parser scaffolding after an assembled proof closes."""
        trees=visits=0
        for key,(digest,tree,nodes,codes) in tuple(self.compiled.items()):
            if tree is None:continue
            trees+=1;visits+=sum(1 for _ in ast.walk(tree))
            self.compiled[key]=(digest,None,{},codes)
        self.path_cache.clear();self.instruction_cache.clear();self.path_observations.clear();self.instruction_slots=0
        return dict(source_trees=trees,ast_node_visits=visits)
    def digest(self,code):
        key=id(code)
        if key not in self.fingerprints:self.fingerprints[key]=(code,code_digest(code))
        original,digest=self.fingerprints[key]
        if original is not code:raise QualificationDecline('immutable code identity recycled')
        return digest


class _RetainedChecks:
    """Checks of original graph edges; never follow a replacement object."""
    def __init__(self):self.checks=[];self.early=[];self.references=[];self._namespace_bindings={}
    @staticmethod
    def atom(value):
        typ=type(value)
        if value is None:return ('none',)
        if typ in (str,bool,int,bytes):return (typ.__name__,value)
        if typ is float:return ('float',struct.pack('>d',value))
        if typ is complex:return ('complex',struct.pack('>dd',value.real,value.imag))
        return ('binding',id(typ),id(value))
    def add(self,path,getter,refs=(),*,early=False):
        self.references.extend(refs);(self.early if early else self.checks).append((path,getter,getter()))
    def binding(self,path,namespace,name):
        identity_key=name is types.ModuleType
        if type(name) is not str and not identity_key:
            raise QualificationDecline('retained namespace key must be exact string or registered module type')
        if type(namespace) is types.MappingProxyType:
            backing=gc.get_referents(namespace)
            if len(backing)!=1 or type(backing[0]) is not dict:
                raise QualificationDecline('unsupported retained namespace backing: '+path)
        elif type(namespace) is not dict:raise QualificationDecline('unsupported retained namespace type: '+path)
        if identity_key:
            native=namespace if type(namespace) is dict else backing[0]
            def snapshot():
                for key,value in dict.items(native):
                    if key is name:return True,self.atom(value)
                return False,self.atom(None)
            value=next((value for key,value in dict.items(native) if key is name),None)
            self.references.extend((namespace,value))
            key=(id(namespace),id(name))
            if key in self._namespace_bindings:
                original,expected=self._namespace_bindings[key]
                if namespace is not original or snapshot()!=expected:
                    raise QualificationDecline('retained namespace changed during admission: '+path)
                return
            self.add(path,snapshot,early=True)
            self._namespace_bindings[key]=(namespace,self.early[-1][2])
            return
        # Retain the same strong-reference multiplicity. Only the duplicate
        # check/getter is shared; every graph edge remains separately counted.
        self.references.extend((namespace,namespace.get(name)))
        key=(id(namespace),name)
        if key in self._namespace_bindings:
            original,expected=self._namespace_bindings[key]
            if namespace is not original or (name in namespace,self.atom(namespace.get(name)))!=expected:
                raise QualificationDecline('retained namespace changed during admission: '+path)
            return
        self.add(path,lambda:(name in namespace,self.atom(namespace.get(name))),early=True)
        self._namespace_bindings[key]=(namespace,self.early[-1][2])
    def sequence(self,path,getter):
        self.references.extend(getter())
        self.add(path,lambda:tuple(self.atom(item) for item in getter()))
    def mapping(self,path,value):
        if type(value) is types.MappingProxyType:
            backing=gc.get_referents(value)
            if len(backing)!=1 or type(backing[0]) is not dict:
                raise QualificationDecline('unsupported retained mapping backing: '+path)
        elif type(value) is not dict:
            raise QualificationDecline('unsupported retained mapping type: '+path)
        self.references.extend(item for pair in value.items() for item in pair)
        self.add(path,lambda:tuple(sorted((self.atom(key),self.atom(item)) for key,item in value.items())))
    def verify(self):
        for path,getter,expected in (*self.early,*self.checks):
            if getter()!=expected:raise QualificationDecline('retained graph binding changed: '+path)


def _authority_signature(value):
    if type(value) in (tuple,frozenset):
        rows=tuple(_authority_signature(item) for item in value)
        return (type(value).__name__,tuple(sorted(rows)) if type(value) is frozenset else rows)
    return _RetainedChecks.atom(value)


def _source_bound_body(value,owner,qualname,codes,*,fingerprint=code_digest):
    """Bind a direct body or the exact registered transactional producer chain."""
    expected=codes.get(qualname)
    if expected is None or type(value) is not types.FunctionType or value.__globals__ is not vars(owner):
        raise QualificationDecline('operation caller binding changed: '+qualname)
    if fingerprint(value.__code__)==fingerprint(expected):return value
    if owner.__name__!='anygeometry.model' or not qualname.startswith('GeometryModel.'):
        raise QualificationDecline('operation caller live code differs from source: '+qualname)
    factory=vars(owner).get('_transactional');factory_code=codes.get('_transactional')
    wrapper_code=codes.get('_transactional.<locals>.wrapped')
    body=vars(value).get('__wrapped__')
    if (type(factory) is not types.FunctionType or factory.__globals__ is not vars(owner) or
            factory_code is None or fingerprint(factory.__code__)!=fingerprint(factory_code) or
            factory.__closure__ is not None or factory.__defaults__ is not None or factory.__kwdefaults__ is not None or
            wrapper_code is None or fingerprint(value.__code__)!=fingerprint(wrapper_code) or
            value.__code__.co_freevars!=('method',) or len(value.__closure__ or ())!=1 or
            type(body) is not types.FunctionType or value.__closure__[0].cell_contents is not body or
            body.__globals__ is not vars(owner) or body.__qualname__!=qualname or
            fingerprint(body.__code__)!=fingerprint(expected)):
        raise QualificationDecline('transactional producer/wrapper/body binding changed: '+qualname)
    return body


def verify_operation_registry(role="common",cache=None):
    """A finite source-bound map, never an inference from observed canary calls."""
    from .model import GeometryModel
    cache=_ImmutableEvidence() if cache is None else cache
    pins=_MODEL_CALLER_SOURCE_PINS if role=="common" else _ROLE_MODEL_CALLER_SOURCE_PINS[role]
    edges=_MODEL_CALLER_EDGES if role=="common" else _ROLE_MODEL_CALLER_EDGES[role]
    for name,expected in pins.items():
        module=importlib.import_module(name)
        if cache.source(module,str(cache.resolve(module.__file__)),syntax=False)[0]!=expected:
            raise QualificationDecline('operation caller source outside fixed registry: '+name)
    # A fixed source file is insufficient if a live caller was replaced. Every
    # registered call site must still execute its source definition. New direct
    # or dynamic model calls therefore require a new reviewed registry.
    compiled={}
    for caller in sorted({row[0] for row in edges}):
        module_name=max((name for name in pins
            if caller.startswith(name+'.')),key=len)
        module=sys.modules[module_name];qualname=caller[len(module_name)+1:]
        value=module
        for part in qualname.split('.'):
            value=vars(value).get(part)
            if type(value) in (staticmethod,classmethod):value=value.__func__
        if type(value) is not types.FunctionType or value.__globals__ is not vars(module):
            raise QualificationDecline('operation caller binding changed: '+caller)
        key=(module_name,value.__code__.co_filename)
        if key not in compiled:compiled[key]=cache.source(module,key[1],syntax=False)[3]
        _source_bound_body(value,module,qualname,compiled[key],fingerprint=cache.digest)
    allowed=operation_methods(GeometryModel,role,cache=cache)
    if any(method not in allowed for caller,receiver,method in edges):
        raise QualificationDecline('operation registry omits a declared model method edge')
    from . import nurbs_sweep_chart as views
    if views._SourceSweepChartView.__mro__[1] is not views.SweepChartModel:
        raise QualificationDecline('model view MRO forwarding changed')
    # stage_task's literal route=None plus the public controller's exact-only
    # admission is the enforced boundary excluding route.pair in this role.
    from ._planning_stages import stage_task
    source=Path(sys.modules[stage_task.__module__].__file__).read_text(encoding='utf-8')
    tree=ast.parse(source)
    body=next(node for node in tree.body if type(node) is ast.FunctionDef and node.name=='stage_task')
    pairs=[node for node in ast.walk(body) if type(node) is ast.Call and type(node.func) is ast.Name and node.func.id=='_pair_paths']
    if len(pairs)!=1 or not any(kw.arg=='route' and type(kw.value) is ast.Constant and kw.value.value is None for kw in pairs[0].keywords):
        raise QualificationDecline('exact stage route invariant unavailable')


def _transaction_model_edges(cache=None):
    cache=_ImmutableEvidence() if cache is None else cache
    """Exact constructor alias from TopologyTransaction to its GeometryModel."""
    from . import transactions
    cls=transactions.TopologyTransaction
    if not _is_type(cls) or cls.__module__!=transactions.__name__ or cls.__qualname__!='TopologyTransaction':
        raise QualificationDecline('transaction context owner changed')
    _,tree,_,_=cache.source(transactions,str(cache.resolve(transactions.__file__)),syntax=True)
    declaration=next(node for node in tree.body if type(node) is ast.ClassDef and node.name=='TopologyTransaction')
    bodies={node.name:node for node in declaration.body if type(node) is ast.FunctionDef}
    methods=('__init__','__enter__','__exit__');edges=set()
    for name in methods:
        function=vars(cls).get(name)
        if type(function) is not types.FunctionType:raise QualificationDecline('transaction context method changed')
        codes=cache.source(transactions,function.__code__.co_filename,syntax=False)[3]
        _source_bound_body(function,transactions,'TopologyTransaction.'+name,codes,fingerprint=cache.digest)
        for node in ast.walk(bodies[name]):
            if type(node) is not ast.Call or type(node.func) is not ast.Attribute:continue
            receiver=node.func.value
            if (type(receiver) is ast.Attribute and receiver.attr=='_model' and
                    type(receiver.value) is ast.Name and receiver.value.id=='self'):
                edges.add(node.func.attr)
    assignments=[node for node in bodies['__init__'].body if type(node) is ast.Assign and
        any(type(target) is ast.Attribute and target.attr=='_model' and type(target.value) is ast.Name and
            target.value.id=='self' for target in node.targets)]
    if len(assignments)!=1 or type(assignments[0].value) is not ast.Name or assignments[0].value.id!='model':
        raise QualificationDecline('transaction context constructor alias changed')
    if edges!={'_enter_transaction','_exit_transaction'}:
        raise QualificationDecline('transaction context model edges changed')
    return edges


def _method_closure(cls,seeds,cache=None):
    """Source-bound transitive self/cls edges for the finite filtered classes."""
    cache=_ImmutableEvidence() if cache is None else cache
    pending=list(seeds);selected=set();sources={}
    namespace={}
    for base in reversed(type.__getattribute__(cls,'__mro__')):
        namespace.update(type.__getattribute__(base,'__dict__'))
    while pending:
        name=pending.pop()
        if name in selected:continue
        selected.add(name);member=namespace.get(name)
        functions=(member.fget,member.fset,member.fdel) if type(member) is property else (
            member.__func__ if type(member) in (staticmethod,classmethod) else member,)
        for function in functions:
            if function is None:continue
            if type(function) is not types.FunctionType:
                raise QualificationDecline('unknown filtered method binding: '+name)
            owner=sys.modules.get(function.__module__)
            if owner is None or function.__globals__ is not vars(owner):
                raise QualificationDecline('filtered method owner changed: '+name)
            key=(owner.__name__,function.__code__.co_filename)
            if key not in sources:
                _,_,nodes,codes=cache.source(owner,key[1],syntax=True);sources[key]=(nodes,codes)
            nodes,codes=sources[key]
            try:source_function=_source_bound_body(function,owner,function.__qualname__,codes,fingerprint=cache.digest)
            except QualificationDecline as error:
                raise QualificationDecline('filtered method binding changed: '+name) from error
            if source_function is not function:
                # The wrapper executes a transaction context before its body;
                # constructor-pinned TopologyTransaction forwards these edges.
                pending.extend(('transaction',*_transaction_model_edges(cache)))
            function=source_function
            body=nodes.get(function.__qualname__);code=codes.get(function.__qualname__)
            if body is None or code is None or cache.digest(code)!=cache.digest(function.__code__):
                raise QualificationDecline('filtered method source/code changed: '+name)
            for node in ast.walk(body):
                if type(node) is ast.Attribute and type(node.value) is ast.Name and node.value.id in ('self','cls'):
                    target=namespace.get(node.attr)
                    if type(target) in (types.FunctionType,staticmethod,classmethod,property):pending.append(node.attr)
                if type(node) is not ast.Call:continue
                target=node.func
                if (type(target) is ast.Attribute and type(target.value) is ast.Name and
                        target.value.id in ('self','cls') and target.attr not in namespace):
                    raise QualificationDecline('unknown filtered receiver call: '+target.attr)
                if type(target) is ast.Name and target.id=='getattr' and node.args:
                    receiver=node.args[0]
                    if type(receiver) is ast.Name and receiver.id in ('self','cls'):
                        if len(node.args)<2 or type(node.args[1]) is not ast.Constant or type(node.args[1].value) is not str:
                            raise QualificationDecline('dynamic filtered receiver lookup')
                        attribute=node.args[1].value
                        if type(namespace.get(attribute)) in (types.FunctionType,staticmethod,classmethod,property):pending.append(attribute)
    return frozenset(selected)


def operation_methods(cls,role,cache=None):
    name=type.__getattribute__(cls,'__module__')+'.'+type.__getattribute__(cls,'__qualname__')
    if name=='anygeometry.model.GeometryModel':
        methods=set(_MODEL_OPERATION_METHODS[role])
        methods.add('__init__')
        methods.update(key for key,value in vars(cls).items() if type(value) is property)
        return _method_closure(cls,methods,cache=cache)
    if name in ('anygeometry.nurbs_sweep_chart.SweepChartModel','anygeometry.nurbs_sweep_chart._SourceSweepChartView') and role=='common':
        return _method_closure(cls,_VIEW_OPERATION_METHODS,cache=cache)
    if name=='anygeometry.features.FeatureHistory':
        seeds=('__init__','_bind_owner') if role=='common' else (
            '__init__','_bind_owner','baseline','records','next_id','__len__','validate',
            'validate_persistence','materialization_checksum','validate_materialization',
            '_capture_baseline_unchecked','_state_view','_restore_unchecked')
        return _method_closure(cls,seeds,cache=cache)
    if name=='anygeometry.features.FeatureRegistry':
        return _method_closure(cls,('__init__','register','has','topology_role','kinds'),cache=cache)
    if name=='anygeometry._planning_stages.WorkerModel':
        return _method_closure(cls,('view',) if role=='common' else ('__init__','verify','view'),cache=cache)
    return frozenset(vars(cls))


def prepare_interfaces():
    """Load declared interfaces and prepare one annotation-free class metadata."""
    import numpy.polynomial.legendre
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from . import nurbs
    cls=nurbs._PreparedCurveState
    slots=('_prepared','_prepared_binding','_endpoint_values','_point_values')
    if (type(cls) is not type or cls.__module__!=nurbs.__name__ or cls.__qualname__!='_PreparedCurveState'):
        raise QualificationDecline('prepared-curve metadata class owner changed')
    tree=ast.parse(Path(nurbs.__file__).read_bytes())
    declarations=[node for node in tree.body if type(node) is ast.ClassDef and node.name=='_PreparedCurveState']
    if len(declarations)!=1:raise QualificationDecline('prepared-curve metadata source class changed')
    declaration=declarations[0]
    if (declaration.bases or declaration.decorator_list or len(declaration.body)!=1 or
            type(declaration.body[0]) is not ast.Assign or len(declaration.body[0].targets)!=1 or
            type(declaration.body[0].targets[0]) is not ast.Name or declaration.body[0].targets[0].id!='__slots__' or
            ast.literal_eval(declaration.body[0].value)!=slots):
        raise QualificationDecline('prepared-curve annotation-free source changed')
    namespace=vars(cls)
    annotation_name='__annotations_cache__' if sys.version_info >= (3,14) else '__annotations__'
    allowed={'__module__','__slots__','__static_attributes__','__doc__',annotation_name,'__annotate_func__','__slotnames__',*slots}
    if sys.version_info >= (3,13):allowed.add('__firstlineno__')
    if (set(namespace)-allowed or namespace.get('__slots__')!=slots or (sys.version_info >= (3,13) and namespace.get('__firstlineno__')!=declaration.lineno)
            or namespace.get('__annotate_func__') is not None):
        raise QualificationDecline('prepared-curve metadata namespace changed')
    for name in slots:
        descriptor=namespace.get(name)
        if type(descriptor) is not types.MemberDescriptorType or descriptor.__objclass__ is not cls or descriptor.__name__!=name:
            raise QualificationDecline('prepared-curve metadata slot changed')
    prior=namespace.get(annotation_name)
    if annotation_name in namespace and (type(prior) is not dict or prior):
        raise QualificationDecline('prepared-curve annotation cache is not empty metadata')
    # Native type metadata read, with no annotation producer or caller hook.
    annotations=type.__getattribute__(cls,'__annotations__')
    if type(annotations) is not dict or annotations or (prior is not None and annotations is not prior):
        raise QualificationDecline('prepared-curve annotation metadata preparation changed')
    _prepare_slot_metadata(cls,_ImmutableEvidence(),initial=True)


def _prepare_slot_metadata(cls,evidence,*,initial):
    """Materialize copyreg's derived class metadata once, without calling hooks."""
    owner=sys.modules.get(type.__getattribute__(cls,'__module__'))
    if not is_package_module(owner) or not owner.__name__.startswith('anygeometry.'):
        return
    current=owner
    for name in type.__getattribute__(cls,'__qualname__').split('.'):
        if type(current) is not types.ModuleType and not _is_type(current):return
        current=vars(current).get(name)
    if current is not cls:return # retained pre-slots class is not a registered replacement
    if not evidence.resolve(owner.__file__).is_relative_to(evidence.resolve(__file__).parent):
        raise QualificationDecline('slot metadata source outside package')
    declarations=evidence.class_declarations(owner,str(evidence.resolve(owner.__file__)))
    if type.__getattribute__(cls,'__qualname__') not in declarations:
        raise QualificationDecline('slot metadata class lacks registered source declaration')
    expected=_slot_names(cls)
    namespace=type.__getattribute__(cls,'__dict__')
    if '__slotnames__' not in namespace:
        if not initial:raise QualificationDecline('prepared slot metadata deleted')
        type.__setattr__(cls,'__slotnames__',expected)
    else:
        value=namespace['__slotnames__']
        if type(value) is not list or any(type(name) is not str for name in value) or value!=expected:
            raise QualificationDecline('prepared slot metadata content changed')


def _slot_names(cls):
    """Native slot schema traversal; no metaclass or descriptor hook invocation."""
    expected=[]
    for base in type.__getattribute__(cls,'__mro__'):
        namespace=type.__getattribute__(base,'__dict__')
        slots=namespace.get('__slots__',())
        if type(slots) is str:slots=(slots,)
        if type(slots) not in (tuple,list) or any(type(name) is not str for name in slots):
            raise QualificationDecline('slot metadata declaration schema')
        for name in slots:
            if name in ('__dict__','__weakref__'):continue
            if name.startswith('__') and not name.endswith('__'):
                stripped=type.__getattribute__(base,'__name__').lstrip('_')
                if stripped:name='_'+stripped+name
            descriptor=namespace.get(name)
            if (type(descriptor) is not types.MemberDescriptorType or descriptor.__objclass__ is not base or descriptor.__name__!=name):
                raise QualificationDecline('slot metadata native descriptor changed')
            expected.append(name)
    return expected


def operation_roots():
    prepare_interfaces()
    from ._planning_stages import stage_task,WorkerModel
    from ._planning_values import encode_value,decode_value
    from ._planning_executor import freeze_model,thaw_model,pack,unpack
    from .batch_intersections import _plan_intersections
    from .serialization import from_dict,to_dict
    return dict(common=(stage_task,encode_value,decode_value,pack,unpack),
        host=(_plan_intersections,),
        constructor=(from_dict,to_dict,freeze_model,thaw_model,WorkerModel.__init__,WorkerModel.verify))


def cancellation_registry(*,worker=False):
    """Owner authority inspected under its declared lock; never reset as cache."""
    from . import cancellation
    import _thread,threading
    if type(cancellation._LOCK) is not _thread.LockType:
        raise QualificationDecline('cancellation lock owner/type changed')
    if type(cancellation._EVENTS) is not weakref.WeakKeyDictionary:
        raise QualificationDecline('cancellation registry type changed')
    with cancellation._LOCK:
        rows=tuple(cancellation._EVENTS.items())
        if any(type(token) is not cancellation.CancellationSignal or type(event) is not threading.Event
               for token,event in rows):
            raise QualificationDecline('cancellation registry entry type changed')
        if worker and rows:raise QualificationDecline('worker cancellation authority is not empty')
    return cancellation,cancellation._LOCK,cancellation._EVENTS,rows


class CancellationAuthority:
    def __init__(self,signal=None):
        self.signal=signal
        owner,lock,registry,rows=cancellation_registry()
        self.owner,self.lock,self.registry=owner,lock,registry
        self.event=None
        if type(signal) is owner.CancellationSignal:
            self.event=next((event for token,event in rows if token is signal),None)
            if self.event is None:raise QualificationDecline('cancellation signal was not issued')
        # Ordinary callbacks are invoked only by the controller, never here.

    def verify(self):
        owner,lock,registry,rows=cancellation_registry()
        if owner is not self.owner or lock is not self.lock or registry is not self.registry:
            raise QualificationDecline('cancellation registry owner binding changed')
        if self.event is not None:
            if next((event for token,event in rows if token is self.signal),None) is not self.event:
                raise QualificationDecline('selected cancellation event changed')
            if self.event.is_set():
                from ._planning_executor import PlannerCancelled
                raise PlannerCancelled('intersection planning cancelled')


def reference_namespace():
    """Exact source-defined Python backend namespace, not a dict exemption."""
    from . import backend
    namespace=backend._RUNTIME_PYTHON_GLOBALS
    if namespace is None:
        if backend._CORE is None:raise QualificationDecline('reference namespace missing')
        return None
    owner=sys.modules.get('anygeometry.reference.bspline')
    if (not is_package_module(owner) or namespace is not vars(owner) or
            type(backend.bspline_evaluate) is not types.FunctionType or
            backend.bspline_evaluate.__globals__ is not namespace or
            vars(owner).get('bspline_evaluate') is not backend.bspline_evaluate):
        raise QualificationDecline('reference namespace owner binding changed')
    if backend._RUNTIME_REFERENCE_NAMES!=('find_spans','basis_derivatives'):
        raise QualificationDecline('reference namespace finite names changed')
    if namespace.get('np') is not backend.np:
        raise QualificationDecline('reference namespace numerical module changed')
    rows=backend._RUNTIME_TRUSTED_REFERENCE
    if type(rows) is not tuple or len(rows)!=2:
        raise QualificationDecline('reference namespace callable table changed')
    for name,row in zip(backend._RUNTIME_REFERENCE_NAMES,rows):
        if (type(row) is not tuple or len(row)!=3 or row[0]!=name or
                namespace.get(name) is not row[1] or type(row[1]) is not types.FunctionType or
                row[1].__globals__ is not namespace or row[1].__code__ is not row[2]):
            raise QualificationDecline('reference namespace callable binding changed')
    return owner


class CallableManifest:
    def __init__(self,roots,*,branch_profile=None,role=None,budget=None,document_model=None,shared=None):
        self.document_model=document_model
        self._epoch_initial={}
        self.registered_stores={}
        self.role=role
        self.budget=budget
        self.roots=tuple(roots)
        self.branch_profile=branch_profile
        self.branch_methods=()
        if branch_profile is not None:
            from . import _planning_profiles as profiles
            if type(branch_profile) is not profiles._GuardedSelector:
                raise QualificationDecline('unknown guarded branch profile')
            self.branch_methods=tuple((cls,name,vars(cls)[name],vars(cls)[name].__code__)
                for cls,names in ((profiles._GuardedSelector,('validate','dependencies','wire')),
                                  (profiles.BranchProfiles,('verify','face','json')))
                for name in names)
            self.branch_module=profiles
            self.branch_source=hashlib.sha256(Path(profiles.__file__).read_bytes()).hexdigest()
        self.root_bindings=tuple((root.__globals__,root.__name__,root) for root in self.roots
            if type(root) is types.FunctionType and root.__globals__.get(root.__name__) is root)
        self.root_bindings+=tuple((vars(sys.modules[type.__getattribute__(root,'__module__')]),type.__getattribute__(root,'__name__'),root) for root in self.roots
            if _is_type(root) and type.__getattribute__(root,'__module__') in sys.modules and
            vars(sys.modules[type.__getattribute__(root,'__module__')]).get(type.__getattribute__(root,'__name__')) is root)
        self._immutable_evidence=(shared.setdefault("immutable_evidence",_ImmutableEvidence()) if shared is not None else _ImmutableEvidence())
        self.strong_references=[]
        self._preparing=True
        self._digest,self._identities=self._capture(self.strong_references,shared=shared)
        self._preparing=False
        self._reference_tokens=self._last_tokens
        self._function_bindings=tuple((value,value.__code__) for value in self.strong_references if type(value) is types.FunctionType)
        self.budget=None
        self.parsing_release=self._immutable_evidence.release_parsing() if shared is None else None

    def _capture(self,references,shared=None):
        from . import _planning_ledger as ledger
        from ._planning_qualification import pin_module,runtime_files
        import numpy as np
        shared={} if shared is None else shared
        evidence=self._immutable_evidence
        checks=_RetainedChecks()
        module_checks={}
        if "immutable_epoch" not in shared:
            evidence.begin();shared["immutable_epoch"]=evidence
        if self.role is not None:verify_operation_registry(self.role,cache=evidence)
        try:
            from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
        except ModuleNotFoundError as error:
            if error.name not in ('numpy._core', 'numpy._core._multiarray_umath'):
                raise
            raise QualificationDecline('NumPy array dispatcher interface unavailable') from error
        if self.branch_profile is not None:
            if (sys.modules.get(self.branch_module.__name__) is not self.branch_module or
                    hashlib.sha256(Path(self.branch_module.__file__).read_bytes()).hexdigest()!=self.branch_source or
                    any(vars(cls).get(name) is not function or function.__code__ is not code
                        for cls,name,function,code in self.branch_methods)):
                raise QualificationDecline('guarded branch definition changed')
            self.branch_profile.validate()
        runtime_types=_runtime_types()
        module_pins=shared.setdefault("module_pins",{});runtime=shared.get("runtime_files")
        module_observers=shared.setdefault('module_observers',{})
        shared={} if shared is None else shared
        queue=[];scheduled_edges=shared.setdefault('edges',set())
        initial_edges=len(scheduled_edges)
        refusals=[]
        seen=shared.setdefault('definitions',{});executed=shared.setdefault('executed',set())
        execution_requests=shared.setdefault('execution_requests',set())
        tokens=[];identities=[];emit_definition=True
        definition_count=execution_count=binding_count=0
        node_kinds={};metadata_kinds={};synthetic_nodes=0;scalar_nodes=0
        def emit(record):
            if emit_definition:tokens.append(record)
        cache_ids=shared.setdefault('cache_ids',set());definition_ids=shared.setdefault('definition_ids',set())
        nodes=0;store_rows=shared.setdefault('stores',{})
        duplicate_edges=0
        classes=shared.setdefault('classes',{});scheduled=set();method_selections={}
        generated_classes=shared.setdefault('generated_classes',{})
        owned_members=shared.setdefault('owned_members',{})
        provider_codes=shared.setdefault('provider_codes',{})
        owned_sentinels=shared.setdefault('owned_sentinels',{})
        abc_owners=shared.setdefault('abc_owners',{})
        if self.branch_profile is not None:emit(('guarded-branch',self.branch_profile.wire()))
        attributes=set('__init__ __new__ __post_init__ __getattribute__ __getattr__ __iter__ __getitem__ __len__ __eq__ __hash__ __lt__ __le__ __add__ __radd__ __sub__ __rsub__ __mul__ __rmul__ __truediv__ __rtruediv__ __neg__ __float__ __int__ __enter__ __exit__'.split())
        protected={'__dataclass_fields__','__annotations__','__annotate_func__','__annotations_cache__',
                   '__conditional_annotations__','_field_defaults','_member_names_','_member_map_','_value2member_map_'}
        def put(path,value,definition=False,synthetic=False,execute=True):
            # Identical binding reads are one edge; different owner/key bindings
            # remain separate even when they alias the same value.
            capable=(value is not None and value is not Ellipsis and value is not NotImplemented and
                type(value) not in (bool,int,str,bytes,float,complex,np.ndarray,types.CodeType,contextvars.ContextVar) and not is_package_module(value))
            execute=execute and capable
            key=('binding',path,id(value) if not synthetic else None,definition)
            execution_key=(key,self.role if _is_type(value) else None)
            nonlocal duplicate_edges
            if key in scheduled_edges:
                if not execute or execution_key in execution_requests:
                    duplicate_edges+=1;return
                key=('execution-upgrade',key,execution_key[1])
            if len(scheduled_edges)>=80000:raise QualificationDecline('callable edge bound 80000 exceeded')
            scheduled_edges.add(key)
            if execute:execution_requests.add(execution_key)
            queue.append((path,value,definition,synthetic,execute))
        # Class method selection is role dependent. Revisit the finite classes
        # already reached, while sharing their unchanged definition expansions.
        for name,value in sorted(classes.items()):put('role-class:'+name,value,True)
        for i,value in enumerate(self.roots):put('root:'+str(self.role)+':'+str(i),value)

        def module_pin(module):
            nonlocal runtime
            if not is_package_module(module):raise QualificationDecline('callable module binding type')
            if sys.modules.get(module.__name__) is not module:
                raise QualificationDecline('callable module is not registered: '+module.__name__)
            if module.__name__ not in module_pins:
                if runtime is None:
                    runtime=runtime_files();shared["runtime_files"]=runtime
                module_observers[module.__name__]=None if type(module) is types.ModuleType else ModuleObserver(module)
                module_pins[module.__name__]=pin_module(module,runtime=runtime,package_only=True,observer=module_observers[module.__name__])
            module_checks[module.__name__]=(module,module_pins[module.__name__],
                vars(module).get('__file__'),vars(module).get('__spec__'),type(module),module_observers[module.__name__])
            return module_pins[module.__name__]
        def symbol(value):
            name=getattr(value,'__module__',None)
            qualname=getattr(value,'__qualname__',getattr(value,'__name__',None))
            if name is None and type(value) is types.BuiltinFunctionType:
                owner=value.__self__
                if (type(owner) is type and owner.__module__=='builtins' and
                        vars(builtins).get(owner.__name__) is owner and
                        vars(owner).get(value.__name__) is value):
                    references.append(owner)
                    identities.append(('builtin-owner',id(owner),id(value)))
                    return ('builtins',owner.__qualname__+'.'+value.__name__,module_pin(builtins))
            if type(name) is not str or type(qualname) is not str or name not in sys.modules:
                raise QualificationDecline('unbound runtime callable '+str(type(value)))
            return (name,qualname,module_pin(sys.modules[name]))
        def authority_data(value):
            nonlocal nodes
            if nodes+(0 if self.budget is None else self.budget[0])>=80000:
                raise QualificationDecline('authority metadata bound 80000 exceeded')
            nodes+=1
            typ=type(value)
            if value is None or typ in (bool,int,str):return (typ.__name__,value)
            if typ is float:return ('float',struct.pack('>d',value).hex())
            if typ is bytes:return ('bytes',value.hex())
            if typ in (tuple,frozenset):
                parts=[authority_data(item) for item in value]
                if typ is frozenset:parts.sort(key=lambda item:json.dumps(item,sort_keys=True))
                return (typ.__name__,parts)
            if _is_type(value):
                owner=sys.modules.get(value.__module__)
                if owner is None or vars(owner).get(value.__name__) is not value:raise QualificationDecline('authority type owner')
                references.append(value);identities.append(('authority-type',id(value)))
                return ('type',value.__module__,value.__qualname__)
            raise QualificationDecline('unsupported admitted document authority value')
        def ordered_key(value):
            if type(value) is frozenset:
                return b'frozenset:'+json.dumps(sorted(ordered_key(item).hex() for item in value)).encode()
            if _is_type(value):
                name=type.__getattribute__(value,'__module__')
                qualname=type.__getattribute__(value,'__qualname__')
                owner=sys.modules.get(name)
                if not is_package_module(owner) or vars(owner).get(value.__name__) is not value:
                    raise QualificationDecline('unregistered type dictionary key')
                return ('type:'+name+'.'+qualname).encode()
            return passive_data(value,key=True)
        def registered_wrapper(value,owner):
            # Only a live registered package binding can confer generated-edge
            # authority. External __module__ spelling alone is never sufficient.
            bound=(is_package_module(owner) and
                (any(item is value for item in vars(owner).values()) or id(value) in owned_members))
            actual=sys.modules.get(value.__globals__.get('__name__'))
            if not bound or not is_package_module(actual) or value.__globals__ is not vars(actual):return False
            origin=vars(actual).get('__file__')
            if origin is None or evidence.resolve(value.__code__.co_filename)!=evidence.resolve(origin):return False
            if actual.__name__ not in ('reprlib','contextlib','anygeometry.nurbs_intersection_curve'):return False
            wrapped=vars(value).get('__wrapped__')
            cells=tuple(cell.cell_contents for cell in value.__closure__ or ())
            if type(wrapped) is not types.FunctionType or not any(item is wrapped for item in cells):return False
            # The provider wrapper body is pinned against its source; its owned
            # wrapped function and recursion/default state still enter this walk.
            pin=module_pin(actual)
            if actual.__name__ not in provider_codes:
                compiled=compile(Path(pin['path']).read_bytes(),pin['path'],'exec',dont_inherit=True)
                pending=[compiled];codes=set()
                while pending:
                    code=pending.pop();codes.add(evidence.digest(code))
                    pending.extend(item for item in code.co_consts if type(item) is types.CodeType)
                provider_codes[actual.__name__]=codes
            if evidence.digest(value.__code__) not in provider_codes[actual.__name__]:return False
            emit(('generated-provider',actual.__name__,pin))
            if actual.__name__=='anygeometry.nurbs_intersection_curve':
                cells_by_name=dict(zip(value.__code__.co_freevars,cells))
                namespace=cells_by_name.get('namespace')
                if type(namespace) is not object:return False
                owned_sentinels[id(namespace)]=(value.__module__,value.__qualname__,'operation-namespace')
            return True
        def owned_function(value):
            owner=sys.modules.get(value.__module__)
            if not value.__module__.startswith('anygeometry.'):
                if not is_package_module(owner):return True
                origin=vars(owner).get('__file__')
                trusted=(value.__module__.split('.')[0] in ('numpy','scipy','shapely') or
                    origin is not None and evidence.resolve(origin).is_relative_to(evidence.resolve(sys.base_prefix)/'Lib'))
                return not trusted
            if owner is None and value in self.roots and value.__code__.co_filename.startswith('<'):return True
            if not is_package_module(owner) or value.__globals__ is not vars(owner):
                if registered_wrapper(value,owner):return True
                raise QualificationDecline('owned function namespace/origin mismatch')
            filename=value.__code__.co_filename
            if not filename.startswith('<') and not evidence.resolve(filename).is_relative_to(evidence.resolve(__file__).parent):
                raise QualificationDecline('owned function source outside package')
            return True
        while queue:
            if nodes+(0 if self.budget is None else self.budget[0])>=80000:
                refusals.append('callable closure truncated at unchanged 80000-node bound');break
            path,value,definition,synthetic,execute=queue.pop();nodes+=1;binding_count+=1
            emit_definition=True
            typ=type(value)
            type_module=type.__dict__['__module__'].__get__(typ)
            type_name=type.__dict__['__qualname__'].__get__(typ)
            if type(type_module) is not str or type(type_name) is not str:
                refusals.append(path+': type metadata must be exact strings')
                break
            kind=type_module+'.'+type_name
            node_kinds[kind]=node_kinds.get(kind,0)+1
            synthetic_nodes+=int(synthetic)
            if value is None or value is Ellipsis or value is NotImplemented:
                scalar_nodes+=1
                emit((path,str(value)));continue
            if typ in (bool,int,str,bytes,float,complex):
                scalar_nodes+=1
                encoded=(value.hex() if typ is bytes else struct.pack('>d',value).hex() if typ is float else
                    (struct.pack('>dd',value.real,value.imag).hex() if typ is complex else value))
                emit((path,typ.__name__,encoded));continue
            identity=id(value)
            if definition:definition_ids.add(identity)
            if not synthetic:
                references.append(value);identities.append((path,identity,id(typ)))
            seen_key=('synthetic',path) if synthetic else identity
            execution_key=(seen_key,self.role if _is_type(value) else None)
            first_definition=seen_key not in seen
            if first_definition:
                seen[seen_key]=path;definition_count+=1
            else:
                tokens.append((path,'alias',seen[seen_key]))
                if not execute or execution_key in executed:continue
            if execute:
                executed.add(execution_key);execution_count+=1
                tokens.append((path,'execution',seen[seen_key],self.role if _is_type(value) else None))
            # All outgoing definition records use the first canonical owner
            # path. Upgrades add dependencies without duplicating definitions.
            path=seen[seen_key];emit_definition=first_definition
            if identity in cache_ids:
                emit((path,'registered-cache',typ.__module__,typ.__name__));continue
            try:
                if typ is types.FunctionType:
                    if type(value.__module__) is not str or type(value.__qualname__) is not str:
                        raise QualificationDecline('function metadata must be exact strings')
                    checks.add(path+'.function-state',lambda value=value:(id(value.__code__),id(value.__globals__),
                        id(value.__defaults__),id(value.__kwdefaults__),id(value.__closure__),
                        tuple(checks.atom(cell.cell_contents) for cell in value.__closure__ or ()),checks.atom(value.__module__),checks.atom(value.__qualname__),
                        id(object.__getattribute__(value,'__annotate__')) if hasattr(types.FunctionType,'__annotate__') else None),[value.__code__],early=True)
                    checks.mapping(path+'.function-attributes',vars(value))
                    if (value.__module__=='anygeometry.model' and
                            value.__code__.co_qualname=='_transactional.<locals>.wrapped'):
                        owner=sys.modules.get(value.__module__)
                        if owner is None:raise QualificationDecline('transactional owner missing')
                        put('anygeometry.model._transactional',vars(owner).get('_transactional'),True,execute=execute)
                    emit((path,'function',value.__module__,value.__qualname__,evidence.digest(value.__code__)))
                    references.extend((value.__code__,value.__globals__))
                    identities.extend(((path+'.code',id(value.__code__),id(types.CodeType)),
                                       (path+'.globals',id(value.__globals__),id(dict))))
                    if not owned_function(value):
                        # Trusted external interface: retain the live callable/code
                        # and source image, without walking provider internals.
                        emit((path,'external-interface',module_pin(sys.modules[value.__module__])))
                        for key,item in (('defaults',value.__defaults__),('kwdefaults',value.__kwdefaults__)):
                            references.append(item);identities.append((path+'.'+key,id(item)))
                            try: emit((path+'.'+key,passive_data(item).hex()))
                            except QualificationDecline:
                                emit((path+'.'+key,type(item).__module__,type(item).__name__))
                            try:passive_data(item)
                            except QualificationDecline:pass
                            else:checks.add(path+'.external-'+key,lambda item=item:passive_data(item),[item])
                        continue
                    put(path+'.defaults',value.__defaults__,True,execute=execute);put(path+'.kwdefaults',value.__kwdefaults__,True,execute=execute)
                    annotation=object.__getattribute__(value,'__annotate__') if hasattr(types.FunctionType,'__annotate__') else None
                    if hasattr(types.FunctionType,'__type_params__'):
                        parameters=object.__getattribute__(value,'__type_params__')
                        put(path+'.type-parameters',parameters,True,execute=False)
                        checks.add(path+'.type-parameters-binding',lambda value=value:id(object.__getattribute__(value,'__type_params__')),[parameters],early=True)
                    if annotation is not None:put(path+'.annotate',annotation,True,execute=False)
                    else:
                        put(path+'.annotations',value.__annotations__,True,execute=False)
                        checks.add(path+'.annotations-binding',lambda value=value:id(value.__annotations__),[value.__annotations__])
                    if value.__dict__:put(path+'.attributes',value.__dict__,True,execute=execute)
                    for i,cell in enumerate(value.__closure__ or ()):
                        try:put(path+'.cell'+str(i),cell.cell_contents,True,execute=execute)
                        except ValueError as exc:raise QualificationDecline('empty callable closure cell') from exc
                    if value.__module__=='anygeometry.features' and value.__qualname__=='builtin_feature_registry' and self.role is not None:
                        ops=list(_registry_factory_instructions(value,cache=evidence))
                    else:ops=evidence.instructions(value.__code__)
                    if not execute:continue
                    if self.branch_profile is not None:
                        selected=self.branch_profile.dependencies(value)
                        if selected is not None:
                            emit((path,'completed-guarded-selector'))
                            for name,item in selected:put(path+'.'+name,item,True)
                            continue
                    for i,op in enumerate(ops):
                        if op.opname in ('LOAD_GLOBAL','LOAD_NAME'):
                            name=op.argval
                            if name=='globals' and value.__module__=='anygeometry.backend' and value.__name__=='_canonical_curve_runtime_context':
                                owner=sys.modules.get(value.__module__)
                                if owner is None or value.__globals__ is not vars(owner) or vars(owner).get(value.__name__) is not value:
                                    raise QualificationDecline('kernel context dispatch owner binding')
                                pin=module_pin(owner)
                                compiled=compile(Path(pin['path']).read_bytes(),value.__code__.co_filename,'exec',dont_inherit=True)
                                expected=next((item for item in compiled.co_consts if type(item) is types.CodeType and item.co_name==value.__name__),None)
                                if expected is None or evidence.digest(value.__code__)!=evidence.digest(expected):
                                    raise QualificationDecline('kernel context source/code mismatch')
                                names=vars(owner).get('_RUNTIME_OWNER_NAMES');rows=vars(owner).get('_RUNTIME_TRUSTED_FUNCTIONS')
                                if type(names) is not tuple or type(rows) is not tuple or tuple(row[0] for row in rows if type(row) is tuple and len(row)==3)!=names:
                                    raise QualificationDecline('kernel context finite dispatch table')
                                for key,function,code in rows:
                                    if type(key) is not str or vars(owner).get(key) is not function or getattr(function,'__code__',None) is not code:
                                        raise QualificationDecline('kernel context finite dispatch binding')
                                    put(owner.__name__+'.'+key,function,True)
                                    checks.binding(owner.__name__+'.'+key,vars(owner),key)
                                continue
                            if name=='globals' and value.__module__=='anygeometry.batch_intersections' and value.__name__=='_validate_route_carrier':
                                owner=sys.modules.get(value.__module__)
                                if owner is None or value.__globals__ is not vars(owner) or vars(owner).get(value.__name__) is not value:
                                    raise QualificationDecline('route dispatch owner binding')
                                pin=module_pin(owner)
                                compiled=compile(Path(pin['path']).read_bytes(),pin['path'],'exec',dont_inherit=True)
                                expected=next((item for item in compiled.co_consts if type(item) is types.CodeType and item.co_name==value.__name__),None)
                                if expected is None or evidence.digest(value.__code__)!=evidence.digest(expected):
                                    raise QualificationDecline('route dispatch source/code mismatch')
                                rows=vars(owner).get('_CARRIER_ROWS')
                                if type(rows) is not dict or any(not _is_type(key) or type(items) is not tuple or
                                        any(type(row) is not int for row in items) for key,items in rows.items()):
                                    raise QualificationDecline('route dispatch table schema')
                                put(value.__module__+'._CARRIER_ROWS',rows,True)
                                checks.binding(value.__module__+'._CARRIER_ROWS',vars(owner),'_CARRIER_ROWS')
                                for row in sorted({row for items in rows.values() for row in items}):
                                    key='_route_row_'+str(row);item=vars(owner).get(key)
                                    if type(item) is not types.FunctionType or item.__globals__ is not vars(owner):
                                        raise QualificationDecline('route dispatch callable binding')
                                    put(value.__module__+'.'+key,item,True)
                                    checks.binding(value.__module__+'.'+key,vars(owner),key)
                                continue
                            if name in ('eval','exec','globals','locals','__import__'):
                                raise QualificationDecline('unsupported dynamic callable dependency: '+name)
                            if name in value.__globals__:bound=value.__globals__[name]
                            elif name in value.__builtins__:bound=value.__builtins__[name]
                            else:raise QualificationDecline('unresolved callable global: '+name)
                            global_owner=value.__globals__.get('__name__',value.__module__)
                            key=global_owner+'.'+name
                            owner=sys.modules.get(global_owner)
                            if (global_owner,name)!=('anygeometry._canonical_curve_runtime','_CONTEXT'):
                                checks.binding(key,value.__globals__ if name in value.__globals__ else value.__builtins__,name)
                            if global_owner=='anygeometry.exact_predicates' and name=='_NATIVE_FRACTION_NAMESPACE':
                                from fractions import Fraction
                                if type(bound) is not tuple or len(bound)!=len(vars(Fraction)) or any(
                                        type(row) is not tuple or len(row)!=2 or type(row[0]) is not str or
                                        vars(Fraction).get(row[0]) is not row[1] for row in bound):
                                    raise QualificationDecline('external Fraction namespace snapshot changed')
                                references.extend((bound,Fraction))
                                checks.mapping(key+'.live-namespace',vars(Fraction))
                                identities.append((key,id(bound),id(Fraction),tuple((field,id(item)) for field,item in bound)))
                                emit((key,'external-class-namespace-interface',tuple(field for field,item in bound),module_pin(sys.modules['fractions'])))
                                continue
                            if global_owner=='anygeometry.cancellation' and name=='_EVENTS':
                                registry_owner,lock,registry,rows=cancellation_registry()
                                if bound is not registry:raise QualificationDecline('cancellation registry binding')
                                references.extend((registry_owner,lock,registry))
                                identities.append((key,id(registry_owner),id(lock),id(registry)))
                                emit((key,'owner-issued-cancellation-authority'))
                                checks.add(key+'.authority',lambda:(id(cancellation_registry()[1]),id(cancellation_registry()[2])))
                                continue
                            cache=(global_owner.startswith('anygeometry.') and is_package_module(owner) and
                                   value.__globals__ is vars(owner) and vars(owner).get(name) is bound and
                                   name in ledger._NUMERICAL_ROOTS and
                                   type(bound) in (dict,list,set,contextvars.ContextVar))
                            if global_owner=='anygeometry._canonical_curve_runtime' and name=='_CONTEXT':
                                from . import backend
                                original=self._epoch_initial.setdefault(key,bound)
                                expected=backend._canonical_curve_runtime_context()
                                def epoch_value(item):
                                    if type(item) not in (tuple,int,str,bytes,float,type(None)):raise QualificationDecline('kernel epoch value schema')
                                    return passive_data(item)
                                if not (bound is original and type(bound) is object) and epoch_value(bound)!=epoch_value(expected):
                                    raise QualificationDecline('canonical runtime epoch outside effective kernel context')
                                references.extend((owner,original))
                                identities.append((key,id(owner),id(original)))
                                emit((key,'effective-kernel-cache-epoch'))
                                def epoch_check(owner=owner,original=original,backend=backend):
                                    live=vars(owner).get('_CONTEXT')
                                    if live is original and type(live) is object:return True
                                    return passive_data(live)==passive_data(backend._canonical_curve_runtime_context())
                                checks.add(key+'.effective-epoch',epoch_check)
                                continue
                            exact_cache=(global_owner,name) in (('anygeometry._canonical_curve_runtime','_STANDALONE'),('anygeometry._canonical_curve_runtime','_MODELS'),('anygeometry._canonical_curve_runtime','_SHARED'))
                            if exact_cache:
                                from collections import OrderedDict
                                allowed={'_STANDALONE':OrderedDict,'_MODELS':weakref.WeakKeyDictionary,'_SHARED':weakref.WeakValueDictionary}
                                if type(bound) is not allowed[name]:raise QualificationDecline('registered runtime cache type changed')
                                cache=True
                            if cache:
                                cache_ids.add(id(bound));references.append(bound)
                                store_rows[key]=(owner,name,bound,type(bound))
                            if global_owner=='anygeometry.serialization' and name=='_qualified_documents':
                                if type(bound) is not weakref.WeakKeyDictionary:raise QualificationDecline('document authority registry type')
                                if self.role=='common':raise QualificationDecline('document authority escaped constructor role')
                                # Only the admitted model's qualification row is
                                # consumed by this request. Other H documents are
                                # neither cleared nor transported.
                                row=None if self.document_model is None else bound.get(self.document_model)
                                if self.document_model is not None and row is None:raise QualificationDecline('admitted document qualification absent')
                                references.append(bound);identities.append((key,id(bound)))
                                emit((key,'request-document-authority',authority_data(row)))
                                checks.add(key+'.selected-row',lambda bound=bound,model=self.document_model:
                                    _authority_signature(None if model is None else bound.get(model)),[bound,row,self.document_model])
                                continue
                            put(key,bound,not cache)
                            current=bound
                            for following in ops[i+1:]:
                                if following.opname not in ('LOAD_ATTR','LOAD_METHOD'):break
                                if not is_package_module(current):break
                                if following.argval not in vars(current):
                                    raise QualificationDecline('dynamic module attribute: '+str(following.argval))
                                checks.binding(key+'.'+following.argval,vars(current),following.argval)
                                key+='.'+following.argval;current=vars(current)[following.argval]
                                put(key,current,True)
                        elif op.opname=='IMPORT_NAME':
                            level=ops[i-2].argval if i>=2 and ops[i-2].opname in ('LOAD_CONST','LOAD_SMALL_INT') else 0
                            name=('.'*level+op.argval) if type(level) is int else op.argval
                            module=importlib.import_module(name,value.__globals__.get('__package__'))
                            put(path+'.import:'+module.__name__,module)
                            for following in ops[i+1:]:
                                if following.opname=='IMPORT_FROM':
                                    if following.argval not in vars(module):
                                        importlib.import_module(module.__name__+'.'+following.argval)
                                    if following.argval not in vars(module):raise QualificationDecline('unresolved finite lazy import')
                                    checks.binding(module.__name__+'.'+following.argval,vars(module),following.argval)
                                    put(module.__name__+'.'+following.argval,vars(module)[following.argval],True)
                                elif following.opname not in ('STORE_FAST','STORE_NAME','STORE_DEREF','POP_TOP'):break
                elif is_package_module(value):
                    emit((path,'module',value.__name__,module_pin(value)))
                elif typ is types.CodeType:
                    emit((path,'code',evidence.digest(value)))
                elif typ in (tuple,list):
                    checks.sequence(path,lambda value=value:tuple(value))
                    emit((path,typ.__name__,len(value)))
                    for i,item in enumerate(value):put(path+'['+str(i)+']',item,definition,synthetic and type(item) is tuple,execute)
                elif typ in (dict,types.MappingProxyType):
                    if path=='anygeometry.backend._RUNTIME_PYTHON_GLOBALS':
                        checks.mapping(path,value)
                        checks.add(path+'.reference-owner',lambda value=value:reference_namespace() is not None and vars(reference_namespace()) is value)
                        owner=reference_namespace()
                        if owner is None or vars(owner) is not value:
                            raise QualificationDecline('reference namespace dictionary owner changed')
                        import __future__
                        from importlib.machinery import SourceFileLoader,ModuleSpec
                        loader=value.get('__loader__');spec=value.get('__spec__')
                        if (type(loader) is not SourceFileLoader or type(spec) is not ModuleSpec or
                                loader.name!=owner.__name__ or evidence.resolve(loader.path)!=evidence.resolve(owner.__file__) or
                                spec.name!=owner.__name__ or spec.loader is not loader or
                                evidence.resolve(spec.origin)!=evidence.resolve(owner.__file__) or
                                value.get('annotations') is not __future__.annotations):
                            raise QualificationDecline('reference import interface binding changed')
                        interface={'__loader__':loader,'__spec__':spec,'annotations':__future__.annotations}
                        checks.add(path+'.import-interface',lambda loader=loader,spec=spec:(type(loader),type(spec),loader.name,loader.path,spec.name,id(spec.loader),spec.origin),[loader,spec])
                        emit((path,'registered-reference-namespace',module_pin(owner),sorted(value)))
                        for key,item in sorted(value.items()):
                            if key in interface:
                                references.append(item);identities.append((path+'.'+key,id(item),id(type(item))))
                                emit((path+'.'+key,'import-provider-interface',type(item).__module__,type(item).__qualname__,module_pin(sys.modules[type(item).__module__])))
                            else:put(owner.__name__+'.'+key,item,True,execute=execute)
                        continue
                    external=sys.modules.get(value.get('__name__')) if typ is dict and type(value.get('__name__')) is str else None
                    if external is not None and vars(external) is value and not external.__name__.startswith('anygeometry'):
                        emit((path,'external-namespace-interface',module_pin(external)));continue
                    checks.mapping(path,value)
                    if typ is types.MappingProxyType:
                        backing=gc.get_referents(value)
                        if len(backing)!=1 or type(backing[0]) is not dict:
                            raise QualificationDecline('unsupported callable mapping backing: '+path)
                    emit((path,typ.__name__,len(value)))
                    for i,(key,item) in enumerate(sorted(value.items(),key=lambda row:ordered_key(row[0]))):
                        put(path+'.key'+str(i),key,definition,execute=execute);put(path+'.value'+str(i),item,definition,execute=execute)
                elif typ in (set,frozenset):
                    checks.add(path,lambda value=value:tuple(sorted(checks.atom(item) for item in value)),tuple(value))
                    emit((path,typ.__name__,len(value)))
                    for i,item in enumerate(sorted(value,key=ordered_key)):put(path+'.'+str(i),item,definition)
                elif typ is functools._lru_cache_wrapper:
                    checks.add(path,lambda value=value:(id(value.__wrapped__),tuple(sorted(value.cache_parameters().items()))),[value.__wrapped__])
                    emit((path,'registered-lru',value.cache_parameters()))
                    if value.__wrapped__.__module__.startswith('anygeometry.'):
                        store_rows[path]=(None,path,value,typ)
                    put(path+'.wrapped',value.__wrapped__,True)
                elif typ is _ArrayFunctionDispatcher:
                    checks.add(path,lambda value=value:(value.__name__,id(value.__wrapped__)),[value.__wrapped__])
                    emit((path,'owner-anchored-numpy-dispatcher',value.__name__,
                        module_pin(sys.modules['numpy._core._multiarray_umath'])))
                    implementation=value.__wrapped__
                    put(path+'.interface',implementation,True)
                elif typ is functools.cached_property:
                    checks.mapping(path,vars(value))
                    if set(vars(value)) not in ({'func','attrname','__doc__'},{'func','attrname','__doc__','__module__'}):
                        raise QualificationDecline('cached property metadata schema')
                    for key,item in sorted(vars(value).items()):put(path+'.'+key,item,True,execute=execute)
                elif typ in (staticmethod,classmethod):
                    checks.add(path,lambda value=value:id(value.__func__),[value.__func__])
                    put(path+'.function',value.__func__,True,execute=execute)
                elif typ is property:
                    checks.add(path,lambda value=value:(id(value.fget),id(value.fset),id(value.fdel),value.__doc__))
                    for key in ('fget','fset','fdel'):put(path+'.'+key,getattr(value,key),True,execute=execute)
                elif typ is contextvars.ContextVar:
                    checks.add(path,lambda value=value,path=path:context_state(value,path,role='worker' if self.role in ('worker','planner') else 'host'))
                    emit((path,'context',context_state(value,path,role='worker' if self.role in ('worker','planner') else 'host')))
                    if value.name=='anygeometry_root_memo':
                        active=value.get();references.append(active)
                        identities.append((path+'.effective',id(active),value in contextvars.copy_context()))
                        checks.add(path+'.effective',lambda value=value:(id(value.get()),value in contextvars.copy_context()),[active])
                elif typ in (weakref.WeakKeyDictionary,weakref.WeakValueDictionary,weakref.WeakSet):
                    raise QualificationDecline('unregistered weak callable state: '+path)
                elif enum.Enum in type.__getattribute__(typ,'__mro__'):
                    checks.add(path,lambda value=value:(checks.atom(object.__getattribute__(value,'_name_')),checks.atom(object.__getattribute__(value,'_value_'))))
                    put(path+'.enum-type',typ,True)
                    put(path+'.enum-name',object.__getattribute__(value,'_name_'),True)
                    put(path+'.enum-value',object.__getattribute__(value,'_value_'),True)
                elif np.generic in type.__getattribute__(typ,'__mro__') and typ.__module__=='numpy':
                    checks.add(path,lambda value=value:(value.dtype.str,value.tobytes()))
                    emit((path,'numpy-scalar',value.dtype.str,value.tobytes().hex()))
                elif typ in (range,slice):
                    put(path+'.range',(value.start,value.stop,value.step),True)
                elif typ is np.ndarray:
                    checks.add(path,lambda value=value:(value.dtype.str,tuple(value.shape),bool(value.flags.writeable),bool(value.flags.c_contiguous),value.tobytes()))
                    seeds=sys.modules.get('anygeometry.surface_capabilities')
                    registered=seeds is not None and vars(seeds).get('_SEEDS') is value
                    for module_name in ('anygeometry.section_faces','anygeometry.nurbs_section_faces'):
                        owner=sys.modules.get(module_name);quadrature=None if owner is None else vars(owner).get('_GAUSS')
                        if type(quadrature) is tuple and len(quadrature)==2 and any(item is value for item in quadrature):
                            if any(type(item) is not np.ndarray or item.dtype!=np.dtype(float) or item.shape!=(24,) for item in quadrature):
                                raise QualificationDecline('registered quadrature definition schema')
                            registered=True
                    if value.flags.writeable and not registered:raise QualificationDecline('mutable callable array: '+path)
                    emit((path,'array-flags',bool(value.flags.writeable),bool(value.flags.c_contiguous)))
                    emit((path,'array',value.dtype.str,list(value.shape),hashlib.sha256(value.tobytes()).hexdigest()))
                elif _is_type(value):
                    if type.__getattribute__(value,'__module__').startswith('anygeometry.'):
                        for base in type.__getattribute__(value,'__mro__'):
                            _prepare_slot_metadata(base,evidence,initial=self._preparing)
                        classes[value.__module__+'.'+value.__qualname__]=value
                    name=type.__getattribute__(value,'__module__')
                    emit((path,'type',name,type.__getattribute__(value,'__qualname__')))
                    if name=='builtins':continue
                    emit((path,'type-module',module_pin(sys.modules[name])))
                    if not name.startswith('anygeometry.'):
                        continue
                    checks.add(path+'.class-namespace',lambda value=value:(tuple(map(id,type.__getattribute__(value,'__mro__'))),tuple(sorted((key,checks.atom(item)) for key,item in type.__getattribute__(value,'__dict__').items()))),list(vars(value).values()),early=True)
                    owner=sys.modules.get(name)
                    if not is_package_module(owner) or (vars(owner).get(value.__name__) is not value and id(value) not in generated_classes):
                        raise QualificationDecline('unregistered owned class binding')
                    for base in type.__getattribute__(value,'__mro__'):
                        references.append(base);identities.append((path+'.mro',id(base),id(type(base))))
                        if not type.__getattribute__(base,'__module__').startswith('anygeometry.'):
                            put(path+'.base:'+base.__qualname__,base,True);continue
                        if base is not value:
                            put('mro:'+name+'.'+value.__qualname__+'.'+base.__qualname__,base,True,execute=execute)
                            continue
                        # Each owned namespace once, rather than class x all
                        # attribute names observed in unrelated functions.
                        for key,member in sorted(type.__getattribute__(base,'__dict__').items()):
                            if key in ('__dict__','__weakref__','__doc__','__module__'):continue
                            if key=='_abc_impl':abc_owners[id(member)]=base
                            if type(member) is types.FunctionType:
                                owned_members[id(member)]=(base,key)
                                if key in ('__setattr__','__delattr__') and member.__code__.co_filename=='<string>':
                                    for cell in member.__closure__ or ():
                                        original=cell.cell_contents
                                        if not _is_type(original) or original is base:continue
                                        namespace=type.__getattribute__(original,'__dict__')
                                        registered=vars(owner).get(value.__name__) is value
                                        if (registered and '__slots__' in vars(base) and
                                                type.__getattribute__(original,'__module__')==name and
                                                type.__getattribute__(original,'__qualname__')==base.__qualname__ and
                                                namespace.get(key) is member and
                                                namespace.get('__dataclass_fields__') is vars(base).get('__dataclass_fields__') and
                                                namespace.get('__dataclass_params__') is vars(base).get('__dataclass_params__')):
                                            generated_classes[id(original)]=(base,key,member)
                            if self.role is not None and base not in method_selections:
                                method_selections[base]=operation_methods(base,self.role,cache=evidence)
                            selected=execute and key not in protected and (self.role is None or key in method_selections[base])
                            owner_path=name+'.'+base.__qualname__+('@pre-slots' if id(base) in generated_classes else '')
                            put(owner_path+'.'+key,member,True,execute=selected)
                elif typ in (types.BuiltinFunctionType,types.MethodDescriptorType,types.WrapperDescriptorType,
                             types.MemberDescriptorType,types.GetSetDescriptorType,types.ClassMethodDescriptorType) or typ is np.ufunc:
                    if typ is np.ufunc:emit((path,'numpy-ufunc',value.__name__,module_pin(np)))
                    elif typ in (types.MethodDescriptorType,types.WrapperDescriptorType,types.MemberDescriptorType,
                                 types.GetSetDescriptorType,types.ClassMethodDescriptorType):
                        emit((path,'native-descriptor',value.__objclass__.__module__,value.__objclass__.__qualname__,value.__name__))
                    else:emit((path,'native-callable',symbol(value)))
                elif typ is types.MethodType:
                    put(path+'.function',value.__func__,True,execute=execute);put(path+'.self',value.__self__,True)
                else:
                    import _collections,_abc
                    if sys.version_info >= (3,14):import annotationlib
                    if typ in (dataclasses.Field,dataclasses._DataclassParams):
                        checks.add(path,lambda value=value:tuple((name,id(descriptor),checks.atom(part)) for name,descriptor,part in _dataclass_slot_values(value)))
                        metadata_kinds[kind]=metadata_kinds.get(kind,0)+1
                        emit((path,'native-dataclass-slots',typ.__name__))
                        put('dataclass-metadata-type:'+typ.__name__,typ,True,execute=False)
                        for name,descriptor,part in _dataclass_slot_values(value):
                            put('dataclasses.'+typ.__name__+'.'+name,descriptor,True,execute=False)
                            put(path+'.'+name,part,True,execute=execute)
                        continue
                    if typ is _collections._tuplegetter:
                        restored,args=value.__reduce__()
                        if restored is not _collections._tuplegetter or type(args) is not tuple or len(args)!=2 or type(args[0]) is not int or type(args[1]) not in (str,type(None)):
                            raise QualificationDecline('tuple field descriptor schema')
                        emit((path,'tuple-field',args,module_pin(sys.modules['_collections'])))
                        continue
                    if sys.version_info >= (3,14) and typ is annotationlib.ForwardRef:
                        def forward_state(value=value,typ=typ):
                            rows=[]
                            for field in type.__getattribute__(typ,'__slots__'):
                                if field=='__weakref__':continue
                                try:rows.append((field,checks.atom(object.__getattribute__(value,field))))
                                except AttributeError:rows.append((field,'absent'))
                            return tuple(rows)
                        checks.add(path,forward_state)
                        fields=type.__getattribute__(typ,'__slots__')
                        for field in fields:
                            if field=='__weakref__':continue
                            try:part=object.__getattribute__(value,field)
                            except AttributeError:emit((path+'.'+field,'absent'));continue
                            put(path+'.'+field,part,True,execute=False)
                        continue
                    if value is sys.float_info:
                        put(path+'.float-info',tuple(value),True,True,False);continue
                    if identity in abc_owners:
                        checks.add(path,lambda cls=abc_owners[identity]:tuple(sorted(id(reference()) for reference in _abc._get_dump(cls)[0])))
                        cls=abc_owners[identity]
                        registry,positive,negative,version=_abc._get_dump(cls)
                        # Registry is authority; positive/negative lookup caches
                        # are runtime-derived, remain owned and are never cleared.
                        live=tuple(reference() for reference in registry)
                        if any(item is None or not _is_type(item) for item in live):
                            raise QualificationDecline('owned ABC registration state')
                        emit((path,'owned-abc-registry',sorted(ordered_key(item).hex() for item in live),module_pin(sys.modules['_abc'])))
                        for item in live:put(path+'.registration:'+item.__module__+'.'+item.__qualname__,item,True)
                        continue
                    sentinel_names=(('anygeometry.nurbs_surface_intersection.curve','_INTERVAL_PAYLOAD'),('anygeometry.nurbs_surface_intersection.curve','_RATIONAL_PAYLOAD'),('anygeometry._canonical_curve_runtime','_CONTEXT'),('anygeometry._source_box_hull','_UNKNOWN'))
                    sentinel=next(((owner,key) for owner,key in sentinel_names if owner in sys.modules and vars(sys.modules[owner]).get(key) is value),None)
                    if typ is object and (sentinel is not None or identity in owned_sentinels):
                        emit((path,'owner-sentinel',sentinel or owned_sentinels[identity]));continue
                    locks=(('anygeometry.surface_capabilities','_LOCK'),('anygeometry.exact_predicates','_PI_CACHE_LOCK'),('anygeometry._canonical_curve_runtime','_LOCK'),('anygeometry.cancellation','_LOCK'))
                    lock=next(((owner,key) for owner,key in locks if owner in sys.modules and vars(sys.modules[owner]).get(key) is value),None)
                    if lock is not None:
                        import _thread
                        if typ is _thread.RLock and 'locked' not in vars(_thread.RLock):
                            raise QualificationDecline('runtime RLock state interface unavailable')
                        if typ not in (_thread.LockType,_thread.RLock) or value.locked():
                            raise QualificationDecline('registered lock state/type')
                        checks.add(path,lambda value=value:value.locked())
                        emit((path,'registered-unlocked',lock,module_pin(sys.modules['_thread'])));continue
                    records=(('anygeometry.nurbs_surface_intersection._filtered','ZERO',('lo','hi')),
                             ('anygeometry.nurbs_surface_intersection._filtered','ONE',('lo','hi')),
                             ('anygeometry.curved_intersections','_BOUNDED_CHART',('unbounded','box')),
                             ('anygeometry.nurbs_surface_intersection._topology','_PLANE',('origin','u_vector','v_vector')))
                    record=next(((owner,key,fields) for owner,key,fields in records if owner in sys.modules and vars(sys.modules[owner]).get(key) is value),None)
                    if record is not None:
                        owner,key,fields=record
                        if typ is types.SimpleNamespace:
                            if set(vars(value))!=set(fields):raise QualificationDecline('registered record extra state')
                        else:
                            if type.__getattribute__(typ,'__module__')!=owner or tuple(vars(typ).get('__slots__',()))!=fields:
                                raise QualificationDecline('registered slot record type/fields')
                        checks.add(path,lambda value=value,fields=fields:tuple((field,checks.atom(object.__getattribute__(value,field))) for field in fields))
                        if typ is types.SimpleNamespace:checks.mapping(path+'.namespace',vars(value))
                        put(path+'.type',typ,True,execute=execute)
                        for field in fields:put(path+'.'+field,object.__getattribute__(value,field),True,execute=execute)
                        continue
                    metadata=ledger._metadata_parts(value)
                    if metadata is not None:
                        checks.add(path,lambda value=value:_authority_signature(ledger._metadata_parts(value)))
                        metadata_kinds[kind]=metadata_kinds.get(kind,0)+1
                        put(path+'.metadata',metadata,True,True,execute)
                    elif '__dataclass_fields__' in type.__getattribute__(typ,'__dict__') and typ.__module__.startswith('anygeometry.'):
                        checks.add(path,lambda value=value,typ=typ:tuple((field.name,checks.atom(object.__getattribute__(value,field.name))) for field in dataclasses.fields(typ)))
                        put(path+'.type',typ,True)
                        for field in dataclasses.fields(value):put(path+'.'+field.name,object.__getattribute__(value,field.name),True)
                    else:raise QualificationDecline('unsupported callable state: '+path)
            except (QualificationDecline,ImportError) as error:
                refusals.append(path+': '+str(error))
                if 'bound 80000 exceeded' in str(error):break
        if cache_ids & definition_ids:raise QualificationDecline('registered cache aliases protected callable definition')
        self.nodes=nodes;self.edges=len(scheduled_edges)-initial_edges;self.duplicate_edges=duplicate_edges
        self.registered_stores=store_rows
        graph=dict(nodes=nodes,edges=self.edges,assembled_edges=len(scheduled_edges),duplicate_edges=duplicate_edges,
            definition_expansions=definition_count,execution_expansions=execution_count,binding_records=binding_count,
            node_kinds=node_kinds,metadata_kinds=metadata_kinds,synthetic_nodes=synthetic_nodes,scalar_nodes=scalar_nodes,
            role=self.role,functions=sum(type(item) is types.FunctionType for item in references),
            unique_objects=len({id(item) for item in references}),pending_edges=len(queue),
            assembled_nodes_before=0 if self.budget is None else self.budget[0])
        self.graph=graph
        if self.budget is not None:
            self.budget[0]+=nodes
            if self.budget[0]>80000:refusals.append('assembled role manifest bound 80000 exceeded')
        if refusals:
            error=QualificationDecline('callable closure refused: '+str(refusals[:5]));error.refusals=refusals
            error.nodes=nodes;error.graph=graph
            raise error
        evidence.verify_paths()
        if self._preparing:
            # Values were observed during this proof, not resnapshotted here.
            for spelling,resolved in evidence.path_observations.items():
                checks.early.append(('source-resolution:'+spelling,
                    lambda spelling=spelling:str(Path(spelling).resolve()),resolved))
            self._retained_checks=checks;self._module_checks=module_checks
        self._last_tokens=tuple(tokens)
        return _hash(tokens),tuple(identities)

    @property
    def digest(self):return self._digest

    def verify(self,shared=None,*,full=False):
        if full:return self.verify_full(shared=shared)
        from ._planning_executor import PlannerSourceChanged
        from ._planning_qualification import pin_module,runtime_files
        shared={} if shared is None else shared
        try:
            verified=shared.setdefault('verified_module_pins',{})
            for name,(module,expected,filename,spec,kind,observer) in self._module_checks.items():
                if type(module) is not kind:raise QualificationDecline('retained module class changed: '+name)
                if observer is not None:observer.verify()
                if (sys.modules.get(name) is not module or vars(module).get('__file__')!=filename or
                        vars(module).get('__spec__') is not spec):
                    raise QualificationDecline('retained module origin/binding changed: '+name)
                if name in verified:
                    if verified[name]!=(id(module),expected):raise QualificationDecline('role module pin disagreement')
                    continue
                if 'runtime_files' not in shared:shared['runtime_files']=runtime_files()
                if pin_module(module,runtime=shared['runtime_files'],package_only=True,observer=observer)!=expected:
                    raise QualificationDecline('retained source/native image changed: '+name)
                verified[name]=(id(module),expected)
            if any(function.__code__ is not code for function,code in self._function_bindings):
                raise QualificationDecline('retained live code object replaced')
            if any(namespace.get(name) is not original for namespace,name,original in self.root_bindings):
                raise QualificationDecline('root symbol rebound')
            self._retained_checks.verify()
        except Exception as error:
            raise PlannerSourceChanged('retained callable closure changed: '+str(error)) from error

    def verify_full(self,shared=None):
        from ._planning_executor import PlannerSourceChanged
        try:
            for module,_,_,_,kind,observer in self._module_checks.values():
                if type(module) is not kind:raise QualificationDecline('module class changed')
                if observer is not None:observer.verify()
            if any(function.__code__ is not code for function,code in self._function_bindings):
                raise QualificationDecline('retained live code object replaced')
            if any(namespace.get(name) is not original for namespace,name,original in self.root_bindings):
                raise QualificationDecline('root symbol rebound')
            refs=[];digest,identities=self._capture(refs,shared=shared)
            if digest!=self._digest or identities!=self._identities:
                def grouped(records):
                    result={}
                    for record in records:result.setdefault(record[0],[]).append(record[1:])
                    return result
                def changes(before,after):
                    first,second=grouped(before),grouped(after)
                    return [dict(path=key,before=first.get(key),after=second.get(key))
                        for key in sorted(first.keys()|second.keys()) if first.get(key)!=second.get(key)]
                error=QualificationDecline('loaded callable binding changed')
                error.manifest_changes=dict(role=self.role,record_changes=changes(self._reference_tokens,self._last_tokens),
                    identity_changes=changes(self._identities,identities))
                raise error
        except Exception as exc:
            error=PlannerSourceChanged('loaded callable closure changed: '+str(exc))
            if hasattr(exc,'manifest_changes'):error.manifest_changes=exc.manifest_changes
            raise error from exc
        finally:
            if shared is None:self._immutable_evidence.release_parsing()


def capture_callable_manifest(roots,*,role=None,budget=None,document_model=None,shared=None):return CallableManifest(roots,role=role,budget=budget,document_model=document_model,shared=shared)


def numerical_roots():
    return operation_roots()['common']


def effective_settings():
    from . import batch_intersections as batch
    from threadpoolctl import threadpool_info
    from ._planning_numpy_context import capture_numpy_context
    numpy_context=capture_numpy_context(package_only=True)
    return dict(environment={name:os.environ.get(name) for name in NUMERICAL_SETTINGS},
        numpy=numpy_context.wire,numpy_binding=numpy_context.binding,
        retry={name:value for name,value in vars(batch).items() if 'RETRY' in name and type(value) in (bool,int,str,float)},
        pools=[dict(filepath=pool.get('filepath'),num_threads=pool.get('num_threads')) for pool in threadpool_info()])


class LoadedManifest:
    def __init__(self,model=None,cancellation=None,*,include_host=True,_cache_traversal=True):
        from ._planning_numpy_context import capture_numpy_context
        self.cancellation=CancellationAuthority(cancellation)
        import time
        budget=[0];shared={"immutable_evidence":_ImmutableEvidence(cache_traversal=_cache_traversal)}
        self.role_seconds={}
        roots=operation_roots()
        stamp=time.perf_counter()
        self.callables=capture_callable_manifest(roots['common'],role='common',budget=budget,shared=shared)
        self.role_seconds['common']=time.perf_counter()-stamp
        self.roles={}
        for role in (('constructor','host') if include_host else ('constructor',)):
            stamp=time.perf_counter()
            try:self.roles[role]=capture_callable_manifest(roots[role],role=role,budget=budget,document_model=model,shared=shared)
            except Exception as error:
                error.completed_roles={'common':self.callables.graph,**{name:item.graph for name,item in self.roles.items()}}
                raise
            self.role_seconds[role]=time.perf_counter()-stamp
        self.nodes=budget[0]
        self.numpy_context=capture_numpy_context(package_only=True)
        self.settings=effective_settings()
        worker_env=dict(self.settings['environment']);worker_env.update({name:'1' for name in _THREADS})
        self.wire=dict(schema='PPQ-loaded-v1',workload=dict(kind='package-numerical-v1'),callables_sha256=self.callables.digest,
            roles={role:manifest.digest for role,manifest in self.roles.items()},host=self.settings,worker=dict(environment=worker_env,retry=self.settings['retry'],
                numpy=self.settings['numpy'],numpy_binding=self.settings['numpy_binding']))
        self.wire['sha256']=_hash(self.wire)
        self.callables._immutable_evidence.verify_paths()
        self.parsing_release=self.callables._immutable_evidence.release_parsing()
    def verify(self,*,full=False):
        from ._planning_executor import PlannerSourceChanged
        self.cancellation.verify()
        self.numpy_context.verify()
        shared={}
        try:
            self.callables.verify(shared=shared,full=full)
            for manifest in self.roles.values():manifest.verify(shared=shared,full=full)
        finally:
            if full:self.callables._immutable_evidence.release_parsing()
        if sum(item.nodes for item in (self.callables,*self.roles.values()))>80000:
            raise PlannerSourceChanged('assembled role manifest bound 80000 exceeded')
        try:
            if effective_settings()!=self.settings:
                raise PlannerSourceChanged('loaded settings binding changed')
        except PlannerSourceChanged:raise
        except Exception as error:raise PlannerSourceChanged('loaded settings binding changed: '+str(error)) from error


def capture_loaded_manifest(*,model=None,face_ids=None,cancellation=None):
    # Legacy selector arguments remain accepted, but no Plane branch profile
    # supplies authority. Constructor/selection validation is independent.
    if model is not None:
        from ._planning_executor import freeze_model
        freeze_model(model)
        if face_ids is not None and any(face not in model.faces for face in face_ids):
            raise QualificationDecline('manifest face selection absent from model')
    elif face_ids is not None:raise QualificationDecline('face selection needs model')
    return LoadedManifest(model=model,cancellation=cancellation)
