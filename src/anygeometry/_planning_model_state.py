"""Explicit derived-state registry for declarative planner transport.

These caches remain owned by H. Exclusion neither clears them nor certifies their
contents. Constructor extras, extension aliases and authority-bearing state retain
their separate checks. In particular, ``definition=False`` alone is not a cache
registration (policy switches carry effective behavior).
"""
import gc
import types
from ._planning_qualification import QualificationDecline


_MODEL_DICT_CACHES = (
    '_arc_cache', '_edge_length_cache', '_trim_loops_cache',
    '_topology_corner_cache', '_chain_arrays',
)

_MODEL_PUBLIC_VIEWS = (
    'vertices', 'edges', 'faces', 'parts', 'sheets', 'face_uses', 'coedges',
    'members', 'member_edge_uses', 'attachments', 'junctions',
    'solid_regions',
)


def declarative_state(model):
    from .model import GeometryModel
    from .spatial import AABBTree
    if type(model) is not GeometryModel:
        raise QualificationDecline('unsupported model type')
    state = object.__getattribute__(model, '__dict__')
    if type(state) is not dict:
        raise QualificationDecline('unsupported model instance namespace')
    pristine = GeometryModel(model_id='00000000-0000-0000-0000-000000000001')
    declared = object.__getattribute__(pristine, '__dict__')
    keys = set(declared)
    if set(state) not in (keys, keys | {'_serialization_extensions'}):
        raise QualificationDecline('non-field model instance state')
    if {name for name,value in declared.items() if type(value) is types.MappingProxyType} != set(_MODEL_PUBLIC_VIEWS):
        raise QualificationDecline('unregistered model public view schema')
    for name in _MODEL_PUBLIC_VIEWS:
        private=state['_'+name];view=state[name]
        if type(private) is not dict or type(view) is not types.MappingProxyType:
            raise QualificationDecline('model public view type: '+name)
        backing=gc.get_referents(view)
        if len(backing)!=1 or backing[0] is not private:
            raise QualificationDecline('model public view backing: '+name)
    if state['_solid_regions']:
        # Solid-region records have no qualified parallel-planning contract; decline (serial route) rather than accept.
        raise QualificationDecline('model solid region state is not qualified for planning')
    # This view is a computed property, so validate its private owner without
    # invoking the property or any injected mapping operation.
    if type(state['_construction_vertices']) is not dict:
        raise QualificationDecline('model construction view backing type')
    for name in _MODEL_DICT_CACHES:
        if type(state[name]) is not dict:
            raise QualificationDecline('derived model cache type: ' + name)
    if state['_spatial_index'] is not None and type(state['_spatial_index']) is not AABBTree:
        raise QualificationDecline('derived model cache type: _spatial_index')
    if (type(state['_change_hooks']) is not list or state['_change_hooks']
            or state['_transaction_journal'] is not None
            or state['_notifying_hooks'] is not False):
        raise QualificationDecline('active model instance state')
    omitted = (*_MODEL_DICT_CACHES, '_spatial_index')
    return {name: part for name, part in state.items() if name not in omitted}


def registered_record_cache(record, field, value):
    """Validate exact registered containers; never traverse or evaluate payloads."""
    from .material_arrangement import MaterialDomain
    from .branch_curves import BezierQuadricCurve
    expected = None
    if type(record) is MaterialDomain and field.name in (
            '_exact_chart_cache', '_exact_boundary_cache', '_float_scale_cache'):
        expected = tuple
    elif type(record) is BezierQuadricCurve and field.name == '_inversions':
        expected = dict
    if expected is None:
        return False
    if field.metadata.get('definition') is not False:
        raise QualificationDecline('derived record cache declaration changed')
    if type(value) is not expected and not (expected is tuple and value is None):
        raise QualificationDecline('derived record cache type: ' + field.name)
    return True


def production_snapshot(model, *, payload_limit=4_000_000, walk_limit=100000):
    """Consolidate production rechecks without retaining model-array aliases.

    Initial admission still uses freeze_model. Original constructor/refcount/GC
    and extension checks run on both sides of one serialization. The same typed
    document walk and packet bounds apply; legacy transport is unchanged.
    """
    from .model import GeometryModel
    from .serialization import to_dict
    from ._planning_constructor import audit
    from ._planning_executor import pack
    from ._planning_qualification import _input_types,_type_name,input_type_names,check_types,model_extension_record
    allowed=input_type_names()
    if (type(model) is not GeometryModel or not any(type(model) is typ for typ in _input_types()) or
            _type_name(type(model)) not in allowed):
        raise QualificationDecline('unsupported production model type')
    state=declarative_state(model)
    before_extension=model_extension_record(model,limit=walk_limit)
    before_rows=audit(state,limit=walk_limit)
    document=to_dict(model)
    check_types(document,{'types':allowed},limit=walk_limit)
    after_extension=model_extension_record(model,limit=walk_limit)
    after_rows=audit(declarative_state(model),limit=walk_limit)
    if before_extension!=after_extension or before_rows!=after_rows:
        raise QualificationDecline('model constructor/extension state changed during snapshot')
    body=dict(document=document,extension_record=after_extension)
    if after_rows:body['constructor_state']=after_rows
    return pack(body,'model',limit=payload_limit)


def clone_for_planning(source, *, preserve_identity=False):
    """Construct an opt-in caller clone before its first H admission.

    Validate the original authority before historical cloning can omit it. Only
    empty feature history is supported, keeping constructor paths unchanged.
    Existing transport restores registered derived state on new records only.
    """
    from ._planning_executor import freeze_model,thaw_model,unpack,pack,PlannerSourceChanged
    from ._planning_qualification import model_extension_record
    from .serialization import to_dict
    from .surfaces import Plane,Cylinder
    if type(preserve_identity) is not bool:raise QualificationDecline('clone identity option type')
    original=freeze_model(source)
    try:
        body=unpack(original,'model')
        if not source.faces or any(type(face.surface) not in (Plane,Cylinder) for face in source.faces.values()):
            raise QualificationDecline('clone adapter requires ordinary Plane/Cylinder faces')
        empty_features=dict(baseline=None,next_id=1,records=[],version=1)
        if pack(body['document']['features'],'features')!=pack(empty_features,'features'):
            raise QualificationDecline('clone adapter requires default empty feature history')
        cloned=source.clone(include_features=False,preserve_identity=preserve_identity)
        declarative_state(cloned)
        document=to_dict(cloned)
        expected=dict(body['document'],model_id=document['model_id'],checksum=document['checksum'])
        if pack(document,'clone-document')!=pack(expected,'clone-document'):
            raise QualificationDecline('historical clone changed declarative document')
        if (cloned.revision!=source.revision or (cloned.model_id==source.model_id) is not preserve_identity):
            raise QualificationDecline('historical clone identity/revision contract')
        transported=dict(body,document=document,extension_record=model_extension_record(cloned))
        packet=pack(transported,'model')
        made=thaw_model(packet)
        if freeze_model(made)!=packet:
            raise QualificationDecline('clone constructor/document roundtrip mismatch')
        return made
    finally:
        # Once original authority was captured, drift outranks an otherwise
        # eligible construction refusal. Unchanged-source errors keep their type.
        try:current=freeze_model(source)
        except Exception as error:
            raise PlannerSourceChanged('source changed during caller clone construction') from error
        if current!=original:
            raise PlannerSourceChanged('source changed during caller clone construction')
