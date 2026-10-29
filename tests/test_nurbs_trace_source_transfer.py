"""Source-transfer classifier on real public fragmentation (not a mesh gate)."""
from contextlib import contextmanager

import pytest

from anygeometry import GeometryError,clone_prepared_geometry,plan_intersections,apply_intersections
from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
from anygeometry.serialization import to_dict,from_dict
import anygeometry.prepared_material_source_transfer as transfer
from test_nurbs_pair_public import model_pair


@pytest.fixture(scope='module')
def fragmented():
    source,parents = model_pair()
    for parent in parents:
        source.add_sheet((parent,))
    before = to_dict(source)
    prepared = clone_prepared_geometry(source)
    with intersection_curve_operation():
        plan = plan_intersections(prepared,parents,policy='connect')
        apply_intersections(prepared,plan,policy='connect')
    traces = tuple(edge.id for edge in prepared.edges.values()
                   if type(edge.curve) is NurbsSurfaceIntersectionCurve)
    assert len(traces) >= 3 and len(prepared.faces) == 4
    assert to_dict(source) == before
    return source,prepared,parents,traces,before


def classify(fixture,*,represented=None,selected=None,declared=None,check=None,work=None):
    source,prepared,_,traces,_ = fixture
    prepared = prepared if work is None else work
    edge = traces[0]
    incident = tuple(prepared.faces_using_edge(edge))
    return transfer._qualify_two_support_virtual_trace(source,prepared,edge,
        incident if represented is None else represented,
        set(prepared.faces) if selected is None else selected,
        incident if declared is None else declared,check or (lambda phase:None),
        (lambda:check('two-support virtual trace continuation')) if check is not None else None)


@contextmanager
def raw_attribute(item,name,value):
    original = getattr(item,name)
    object.__setattr__(item,name,value)
    try:
        yield
    finally:
        object.__setattr__(item,name,original)


def test_two_support_trace_classifies_both_original_parents_without_source_mutation(fragmented):
    source,prepared,parents,traces,before = fragmented
    with intersection_curve_operation():
        owners,incident = classify(fragmented)
    assert owners == tuple(sorted(parents))
    assert incident == tuple(sorted(prepared.faces))
    assert to_dict(source) == before
    assert all(len(prepared.faces_using_edge(edge)) == 4 for edge in traces)


@pytest.mark.parametrize('missing',('representation','selection','declaration','duplicate_declaration'))
def test_two_support_trace_requires_every_current_material_use(fragmented,missing):
    _,prepared,_,_,_ = fragmented
    faces = tuple(sorted(prepared.faces))
    kwargs = {'representation':{'represented':faces[:-1]},
              'selection':{'selected':faces[:-1]},
              'declaration':{'declared':faces[:-1]},
              'duplicate_declaration':{'declared':(*faces[:-1],faces[0])}}[missing]
    with pytest.raises(GeometryError,match='four|descendants'):
        classify(fragmented,**kwargs)


@pytest.mark.parametrize('copy',('clone','persisted'))
def test_ordinary_copy_or_persisted_curve_is_not_source_lineage(fragmented,copy):
    _,prepared,_,_,_ = fragmented
    made = prepared.clone(preserve_identity=True) if copy == 'clone' else from_dict(to_dict(prepared))
    with pytest.raises(GeometryError,match='authenticated source lineage'):
        classify(fragmented,work=made)


def test_trace_same_revision_source_control_mutation_refuses(fragmented):
    source,_,parents,_,_ = fragmented
    surface = source.faces[parents[0]].surface
    controls = surface.controls
    point = controls[0][0]
    changed = (((point[0],point[1],point[2]+.125),*controls[0][1:]),*controls[1:])
    revision = source.revision
    with raw_attribute(surface,'controls',changed),pytest.raises(GeometryError):
        classify(fragmented)
    assert source.revision == revision


def test_trace_same_revision_vertex_mutation_refuses(fragmented):
    _,prepared,_,traces,_ = fragmented
    vertex = prepared.vertices[prepared.edges[traces[0]].start]
    # Coordinates are public vertex definition fields, not a revision stamp.
    changed = vertex.position.copy()
    changed[0] += .125
    with raw_attribute(vertex,'position',changed),pytest.raises(GeometryError):
        classify(fragmented)


def test_trace_changed_owner_or_missing_parent_never_grants_receipt(fragmented,monkeypatch):
    source,prepared,_,_,_ = fragmented
    routes = transfer._LINKS[prepared]
    row = routes[0]
    # Simulates a corrupt owner ledger: no current material parent can be
    # recovered by geometry proximity, tags, or matching only one support.
    monkeypatch.setitem(transfer._LINKS,prepared,((row[0],row[1],row[2],row[3][:1],row[4]),))
    with pytest.raises(GeometryError,match='parent lineage|original parents'):
        classify(fragmented)


def test_trace_qualified_cache_cannot_replace_issued_owner_shape(fragmented):
    import anygeometry.nurbs_intersection_curve as curves
    _,prepared,_,traces,_ = fragmented
    curve = prepared.edges[traces[0]].curve
    with intersection_curve_operation():
        classify(fragmented)
        curves._OPERATIONS.get()[('shape',curve)] = object()
        with pytest.raises(GeometryError):
            classify(fragmented)


def test_trace_callback_cancellation_and_same_revision_mutation_propagate(fragmented):
    source,_,_,_,_ = fragmented
    class Cancelled(BaseException):
        pass
    def stop(phase):
        if 'continuation' in phase:
            raise Cancelled()
    with pytest.raises(Cancelled):
        classify(fragmented,check=stop)
    # A source mutation at the final callback is caught before classification.
    vertex = next(iter(source.vertices.values()))
    original = vertex.position
    changed = original.copy()
    changed[0] += .125
    def mutate(phase):
        if phase.endswith('complete'):
            object.__setattr__(vertex,'position',changed)
    try:
        with pytest.raises(GeometryError):
            classify(fragmented,check=mutate)
    finally:
        object.__setattr__(vertex,'position',original)


def test_virtual_classifier_does_not_issue_a_source_transfer_receipt(fragmented):
    source,prepared,_,_,_ = fragmented
    before = len(transfer._TRANSFERS)
    classify(fragmented)
    assert len(transfer._TRANSFERS) == before
    with pytest.raises(GeometryError,match='nonliteral|live owner cell receipt'):
        transfer.query_prepared_material_source_transfer(source,prepared,(object(),))


def test_changed_joint_attachment_does_not_retain_virtual_trace_authority(fragmented):
    from anygeometry.structural import ConnectionIntent
    _,prepared,_,traces,_ = fragmented
    attachment = next(item for item in prepared.attachments.values()
                      if item.target_kind == 'edge' and item.target_id == traces[0])
    revision = prepared.revision
    with raw_attribute(attachment,'connection_intent',ConnectionIntent.IMPRINT),pytest.raises(GeometryError):
        classify(fragmented)
    assert prepared.revision == revision


def test_raw_reverse_incidence_cannot_hide_an_incident_child(fragmented):
    _,prepared,_,traces,_ = fragmented
    edge = traces[0]
    original = prepared._edge_faces[edge]
    prepared._edge_faces[edge] = set(sorted(original)[:-1])
    try:
        with pytest.raises(GeometryError):
            classify(fragmented)
    finally:
        prepared._edge_faces[edge] = original


def test_public_imprint_retains_undeclared_trace_and_source_transfer_refuses():
    source,parents = model_pair()
    for parent in parents:
        source.add_sheet((parent,))
    prepared = clone_prepared_geometry(source)
    with intersection_curve_operation():
        plan = plan_intersections(prepared,parents,policy='imprint')
        apply_intersections(prepared,plan,policy='imprint')
        traces = tuple(edge.id for edge in prepared.edges.values()
                       if type(edge.curve) is NurbsSurfaceIntersectionCurve)
        from anygeometry.joint_edges import query_joint_edge
        assert traces and all(not query_joint_edge(prepared,edge).declared for edge in traces)
        fixture = source,prepared,parents,traces,to_dict(source)
        with pytest.raises(GeometryError,match='explicit CONNECT'):
            classify(fixture)


@pytest.mark.parametrize('failure',('exception','cancel'))
def test_generated_connect_declaration_failure_is_candidate_only(failure,monkeypatch):
    import anygeometry.joint_edges as joints
    source,parents = model_pair()
    for parent in parents:
        source.add_sheet((parent,))
    before = to_dict(source)
    allocator = dict(source._next_id)
    structural_allocator = dict(source._next_structural_id)
    original = joints.declare_joint_edge
    class Cancelled(BaseException):
        pass
    def fail_after_declaration(*args):
        original(*args)
        if failure == 'cancel':
            raise Cancelled()
        raise GeometryError('injected generated declaration failure')
    monkeypatch.setattr(joints,'declare_joint_edge',fail_after_declaration)
    with intersection_curve_operation(),pytest.raises(Cancelled if failure == 'cancel' else GeometryError):
        plan_intersections(source,parents,policy='connect')
    assert to_dict(source) == before
    assert source._next_id == allocator and source._next_structural_id == structural_allocator


def test_connect_declaration_uses_certified_extent_without_newtrace_arclength(monkeypatch):
    from anygeometry import GeometryModel
    from anygeometry.joint_edges import query_joint_edge
    original = GeometryModel.edge_length
    def length(model,identifier):
        if type(model.edges[identifier].curve) is NurbsSurfaceIntersectionCurve:
            raise AssertionError('CONNECT declaration must not integrate newtrace arclength')
        return original(model,identifier)
    monkeypatch.setattr(GeometryModel,'edge_length',length)
    source,parents = model_pair()
    for parent in parents:
        source.add_sheet((parent,))
    before = to_dict(source)
    with intersection_curve_operation():
        plan = plan_intersections(source,parents,policy='connect')
        assert to_dict(source) == before
        apply_intersections(source,plan,policy='connect')
        traces = [edge.id for edge in source.edges.values()
                  if type(edge.curve) is NurbsSurfaceIntersectionCurve]
        assert traces and all(query_joint_edge(source,edge).declared for edge in traces)


def test_declaration_feature_extent_is_outward_and_origin_independent(monkeypatch):
    from fractions import Fraction as Q
    from types import SimpleNamespace
    from anygeometry.nurbs_pair_imprint import _declaration_feature_extent
    import anygeometry.plane_nurbs as work
    calls = []
    monkeypatch.setattr(work,'charge_box',lambda:calls.append(1))
    class Enclosure:
        def __init__(self,offset):
            self.offset = offset
        def enclose(self,a,b):
            assert (a,b) == (Q(0),Q(1))
            return (SimpleNamespace(world_lo=(self.offset,)*3,
                world_hi=tuple(self.offset+x for x in (Q(1,3),Q(2,5),Q(3,7)))),)
    exact = Q(1,3)+Q(2,5)+Q(3,7)
    values = [_declaration_feature_extent(Enclosure(offset),lambda:None)
              for offset in (Q(0),Q(10**15))]
    assert values[0] == values[1] and Q(values[0]) >= exact
    assert len(calls) == 2


def test_connect_registration_failure_restores_published_joints_lineage_and_allocators(monkeypatch):
    source,parents = model_pair()
    for parent in parents:
        source.add_sheet((parent,))
    source_before = to_dict(source)
    prepared = clone_prepared_geometry(source)
    with intersection_curve_operation():
        plan = plan_intersections(prepared,parents,policy='connect')
        before = to_dict(prepared)
        revision = prepared.revision
        allocator,structural = dict(prepared._next_id),dict(prepared._next_structural_id)
        lineage = transfer._LINKS[prepared]
        receipts = {name:value for name,value in vars(prepared).items() if 'receipt' in name}
        names = ('_edge_faces','_edge_coedges','_face_structural_uses','_sheet_junctions','_target_attachments')
        reverse = {name:{key:frozenset(values) for key,values in getattr(prepared,name).items()} for name in names}
        original = transfer._publish_source_lineage
        reached = []
        def fail_after_publication(model,staged):
            original(model,staged)
            # Injection occurs after committed topology, attachments/junctions
            # and the actual new lineage have reached the live destination.
            assert model is prepared and len(model.faces) == 4
            assert model.junctions and model.attachments
            assert transfer._LINKS[model] is not lineage
            reached.append(True)
            raise RuntimeError('injected CONNECT registration failure')
        monkeypatch.setattr(transfer,'_publish_source_lineage',fail_after_publication)
        with pytest.raises(RuntimeError,match='CONNECT registration failure'):
            apply_intersections(prepared,plan,policy='connect')
        assert reached == [True]
        assert to_dict(prepared) == before and prepared.revision == revision
        assert prepared._next_id == allocator and prepared._next_structural_id == structural
        assert transfer._LINKS[prepared] is lineage
        now = {name:value for name,value in vars(prepared).items() if 'receipt' in name}
        assert set(now) == set(receipts) and all(now[name] is value for name,value in receipts.items())
        assert all({key:frozenset(values) for key,values in getattr(prepared,name).items()} == reverse[name]
                   for name in names)
        assert not prepared.validate_topology() and not prepared._validate_structural()
    assert to_dict(source) == source_before
