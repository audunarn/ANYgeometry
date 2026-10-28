"""Actual public two-support query/plan/apply/persistence vertical slice."""
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryModel,GeometryError
from anygeometry.nurbs import NurbsSurface,NurbsCurve,CurveOnSurface
from anygeometry.intersections import query_intersection,plan_imprint,apply_imprint
from anygeometry.predicates import IntersectionKind
from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
from anygeometry.serialization import to_dict,from_dict


def model_pair():
    model = GeometryModel()
    knots = (0.,0.,0.,1.,1.,1.)
    square = (1.,-1.,1.)
    faces = []
    for scale,offset in ((1.,0.),(2.,-.25)):
        controls = tuple(tuple((float(i-1),float(j-1),scale*(square[i]+square[j])+offset)
            for j in range(3)) for i in range(3))
        support = NurbsSurface(2,2,knots,knots,controls)
        corners = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
        vertices = [model.add_point(*map(float,support.exact_point(*uv))) for uv in corners]
        edges = []
        for i in range(4):
            pcurve = NurbsCurve(1,(0.,0.,1.,1.),(corners[i],corners[(i+1)%4]))
            edges.append(model.add_curve(vertices[i],vertices[(i+1)%4],CurveOnSurface(support,pcurve)))
        faces.append(model.add_face(edges,surface=support))
    return model,tuple(faces)


def test_public_circle_query_plan_apply_persist_shared_world_edges():
    model,faces = model_pair()
    before = to_dict(model)
    result = query_intersection(model,*faces)
    assert result.kind is IntersectionKind.CROSS,result.diagnostics
    assert len(result.components) == 1
    assert isinstance(result.components[0].analytic_curve,NurbsSurfaceIntersectionCurve)
    assert to_dict(model) == before
    with intersection_curve_operation():
        plan = plan_imprint(model,result,policy='imprint')
        assert to_dict(model) == before
        application = apply_imprint(model,plan,policy='imprint')
        assert len(model.faces) == 4
        assert not model.validate_topology()
        traces = [edge for edge in model.edges.values() if isinstance(edge.curve,NurbsSurfaceIntersectionCurve)]
        assert len(traces) >= 3
        assert all(len(model.faces_using_edge(edge.id)) == 4 for edge in traces)
        assert all(any(old.id == parent for old,new in application.change_set.replacements) for parent in faces)
        document = to_dict(model)
        restored = from_dict(document)
        assert to_dict(restored) == document


def test_public_two_support_stale_plan_refuses_without_mutation():
    model,faces = model_pair()
    with intersection_curve_operation():
        plan = plan_imprint(model,*faces,policy='imprint')
        model.add_point(8.,9.,10.)
        before = to_dict(model)
        with pytest.raises(GeometryError):
            apply_imprint(model,plan,policy='imprint')
        assert to_dict(model) == before


def test_public_circle_profile(monkeypatch):
    """Bounded diagnostic: retain phase timings/counts plus a stalled stack."""
    import faulthandler
    import importlib
    import json
    import time
    rows = {}
    def instrument(module,name):
        original = getattr(module,name)
        label = module.__name__+'.'+name
        def wrapped(*args,**kwargs):
            record = rows.setdefault(label,[0,0.])
            record[0] += 1
            first = record[0] == 1
            if first:
                print('ENTER '+label,flush=True)
            started = time.perf_counter()
            try:
                return original(*args,**kwargs)
            finally:
                record[1] += time.perf_counter()-started
                if first:
                    print('EXIT '+label+' '+str(record),flush=True)
        monkeypatch.setattr(module,name,wrapped)
    for module,names in (
        ('anygeometry.nurbs_pair_imprint',('_curves','_fragment','candidate_pair','query_pair')),
        ('anygeometry.nurbs_intersection_faces',('loop_area','validate_intersection_face')),
        ('anygeometry.translational_imprint',('_checksum','_publish_qualified')),
        ('anygeometry.prepared_face_preimages',('_compose_application_preimages',)),
        ('anygeometry.edge_subcurve_preimages',('_finalize_edge_subcurve_preimages',))):
        target = importlib.import_module(module)
        for name in names:
            instrument(target,name)
    faulthandler.dump_traceback_later(20,repeat=False)
    try:
        test_public_circle_query_plan_apply_persist_shared_world_edges()
    finally:
        faulthandler.cancel_dump_traceback_later()
        print('PUBLIC_PHASE_TOTALS '+json.dumps(rows,sort_keys=True),flush=True)


def test_intersection_containment_reuses_declared_joint_witnesses(monkeypatch):
    from anygeometry.intersection_chart import contains_owner_loop
    import anygeometry.intersection_chart as chart_module
    import numpy as np
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    pieces = tuple(curve.subcurve(Q(i,3),Q(i+1,3)) for i in range(3))
    original = NurbsSurfaceIntersectionCurve.chart
    counts = {}
    def perturbed(self,parameter):
        identity = self.point_identity(parameter)
        counts[identity] = counts.get(identity,0)+1
        # Distinct valid approximation witnesses must never create a connector
        # to a newly evaluated value at the next occurrence of the same joint.
        return original(self,parameter)+np.array([1e-7*counts[identity],0.])
    monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'chart',perturbed)
    # This regression owns the independent winding fallback's connector
    # contract. Force only sufficient-sign inapplicability; keep real winding.
    monkeypatch.setattr(chart_module,'_closed_sign_contains',lambda *args:None)
    with intersection_curve_operation():
        assert contains_owner_loop(pieces,model.faces[faces[0]].surface,(Q(1,2),Q(1,2)))
    assert counts and max(counts.values()) == 1


def test_intersection_containment_sign_route_needs_no_joint_witnesses(monkeypatch):
    import anygeometry.intersection_chart as chart_module
    model,faces=model_pair()
    curve=query_intersection(model,*faces).components[0].analytic_curve
    pieces=tuple(curve.subcurve(Q(i,3),Q(i+1,3)) for i in range(3))
    def unnecessary(*args,**kwargs):
        raise AssertionError('certified strict sign should not need winding or point witnesses')
    monkeypatch.setattr(chart_module,'_contains_owner_loop_winding',unnecessary)
    monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'chart',unnecessary)
    with intersection_curve_operation():
        assert chart_module.contains_owner_loop(pieces,model.faces[faces[0]].surface,(Q(1,2),Q(1,2)))
        assert not chart_module.contains_owner_loop(pieces,model.faces[faces[0]].surface,(Q(7,8),Q(7,8)))


def test_containment_only_reclassifies_unresolved_contact(monkeypatch):
    from types import SimpleNamespace
    from anygeometry import nurbs_domain
    from anygeometry.intersection_chart import IntersectionChartContactUndecided
    from anygeometry.nurbs_surface_intersection import NurbsSurfaceIntersectionCancelled
    import anygeometry.intersection_chart as chart_module
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    domain = SimpleNamespace(support=model.faces[faces[0]].surface)
    loop = (SimpleNamespace(curve=curve),)
    for error in (NurbsSurfaceIntersectionCancelled('cancelled'),GeometryError('piece budget exhausted')):
        def refuse(*args,**kwargs):
            raise error
        monkeypatch.setattr(chart_module,'contains_owner_loop',refuse)
        with pytest.raises(type(error),match=str(error)):
            nurbs_domain.contains_loop(domain,loop,(.5,.5),1e-9)
    def unresolved(*args,**kwargs):
        raise IntersectionChartContactUndecided('contact unresolved')
    monkeypatch.setattr(chart_module,'contains_owner_loop',unresolved)
    with pytest.raises(nurbs_domain.ChartContainmentUndecided):
        nurbs_domain.contains_loop(domain,loop,(.5,.5),1e-9)


def test_operation_issuer_registry_does_not_retain_proof_context_after_operation():
    import gc
    import weakref
    from anygeometry.nurbs_surface_intersection.curve import _ISSUED_CONTEXTS
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    with intersection_curve_operation():
        shape = curve._shape()
        identity = id(shape._context)
        reference = weakref.ref(shape._context)
    del shape
    gc.collect()
    assert reference() is None
    assert identity not in _ISSUED_CONTEXTS


def test_operation_checker_reentry_unwinds_and_later_cancellation_remains_active():
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    state = {'entered':False,'calls':0,'cancel':False,'raise':False}
    other = {'calls':0}
    def composed():
        other['calls'] += 1
        return False
    def check():
        state['calls'] += 1
        if not state['entered']:
            state['entered'] = True
            # Fresh preparation captures a callback Context while this checker
            # is active. Its later checks must see the dynamically cleared set.
            curve.bounds(Q(1,4),Q(1,3))
        if state['raise']:
            state['raise'] = False
            raise GeometryError('injected checker failure')
        return state['cancel']
    with intersection_curve_operation(check),intersection_curve_operation(composed):
        curve.bounds(Q(1,5),Q(1,4))
        assert other['calls'] > 0
        before = state['calls']
        state['raise'] = True
        with pytest.raises(GeometryError,match='injected checker failure'):
            curve.bounds(Q(1,5),Q(1,4))
        assert state['calls'] > before
        before = state['calls']
        state['cancel'] = True
        with pytest.raises(GeometryError,match='cancelled'):
            curve.bounds(Q(1,5),Q(1,4))
        assert state['calls'] > before


@pytest.mark.parametrize('kind',('controls','vertex','reverse_incidence'))
def test_pair_callback_same_revision_mutation_refuses_before_candidate_publication(kind,monkeypatch):
    import importlib
    import numpy as np
    from anygeometry.batch_intersections import IntersectionBatchPolicy
    from anygeometry.nurbs_pair_imprint import candidate_pair
    source = importlib.import_module('anygeometry.serialization')
    model,faces = model_pair()
    revision = model.revision
    identifiers = (tuple(model.vertices),tuple(model.edges),tuple(model.faces),model.id_state())
    before_checksum = to_dict(model)['checksum']['value']
    calls = []
    qualified = source._qualified_model_state
    def observe(*args,**kwargs):
        calls.append(kwargs.get('expected_checksum'))
        return qualified(*args,**kwargs)
    monkeypatch.setattr(source,'_qualified_model_state',observe)
    changed = [False]
    def callback():
        if not changed[0]:
            changed[0] = True
            if kind == 'controls':
                surface = model.faces[faces[0]].surface
                controls = [[tuple(p) for p in row] for row in surface.controls]
                x,y,z = controls[1][1]
                controls[1][1] = (x,y,z+.125)
                object.__setattr__(surface,'controls',tuple(tuple(row) for row in controls))
            elif kind == 'vertex':
                vertex = model.vertices[next(iter(model.vertices))]
                object.__setattr__(vertex,'position',np.asarray(vertex.position)+np.array([.125,0.,0.]))
            else:
                model._edge_coedges.setdefault(next(iter(model.edges)),set()).add(987654321)
        return False
    policy = IntersectionBatchPolicy('imprint',cancellation_check=callback)
    with pytest.raises(GeometryError):
        candidate_pair(model,tuple(model.handle('face',face) for face in faces),policy,None,require_cells=True)
    assert changed[0] and calls and all(value == before_checksum for value in calls)
    assert model.revision == revision
    assert (tuple(model.vertices),tuple(model.edges),tuple(model.faces),model.id_state()) == identifiers
    assert not hasattr(model,'_intersection_application_receipt')


def test_advisory_area_reuse_is_literal_scoped_and_regenerates_for_settings_or_orientation(monkeypatch):
    import dataclasses
    import anygeometry.nurbs_domain as domain_module
    from anygeometry.material_arrangement import MaterialDomain,ArrangementPath
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    loop = (ArrangementPath(curve,source_edge=19),)
    domain = MaterialDomain(faces[0],model.faces[faces[0]].surface,(loop,))
    original = domain_module._settle
    calls = []
    def count(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(domain_module,'_settle',count)
    with intersection_curve_operation():
        from anygeometry.nurbs_intersection_curve import operation_fact,operation_cached
        public_key = lambda:domain_module._world_area_estimate_key(domain,loop,1e-7)
        forged = (-123.,0.)
        assert operation_fact(public_key,lambda:forged) == forged
        own = operation_cached(lambda *_:public_key())(lambda *_:forged)
        assert own(domain,loop,1e-7) == forged
        initial = domain_module.world_area_estimate(domain,loop,1e-7)
        assert initial != forged
        assert domain_module.world_area_estimate(domain,loop,1e-7) == initial
        assert len(calls) == 1
        domain_module.world_area_estimate(domain,loop,5e-8)
        assert len(calls) == 2
        reverse = (dataclasses.replace(loop[0],curve=curve.subcurve(1,0)),)
        reversed_area = domain_module.world_area_estimate(domain,reverse,1e-7)
        assert reversed_area[0] == pytest.approx(-initial[0],abs=1e-7)
        assert len(calls) == 3
        # Complete path metadata also belongs to the literal key.
        changed_path = (dataclasses.replace(loop[0],source_edge=20),)
        assert domain_module.world_area_estimate(domain,changed_path,1e-7) == initial
        assert len(calls) == 4
        controls = domain.support.controls
        rows = [list(row) for row in controls]
        x,y,z = rows[1][1]
        rows[1][1] = (x,y,z+.125)
        object.__setattr__(domain.support,'controls',tuple(tuple(row) for row in rows))
        try:
            with pytest.raises(GeometryError):
                domain_module.world_area_estimate(domain,loop,1e-7)
        finally:
            object.__setattr__(domain.support,'controls',controls)
        # The changed source refuses before any new Gauss estimate starts.
        assert len(calls) == 4
    # Ending the owner operation must discard the advisory value.
    with intersection_curve_operation():
        assert domain_module.world_area_estimate(domain,loop,1e-7) == initial
        assert len(calls) == 5


def test_cached_area_does_not_authenticate_forged_trimmed_chart_material_area():
    import dataclasses
    from anygeometry.trimmed_charts import query_trimmed_surface_charts,validate_trimmed_surface_charts_binding
    model,faces = model_pair()
    with intersection_curve_operation():
        plan = plan_imprint(model,*faces,policy='imprint')
        apply_imprint(model,plan,policy='imprint')
        inner = next(face.id for face in model.faces.values() if not face.holes and
                     all(isinstance(model.edges[e.edge].curve,NurbsSurfaceIntersectionCurve) for e in face.loop))
        collection = query_trimmed_surface_charts(model,(model.handle('face',inner),))
        validate_trimmed_surface_charts_binding(model,collection)
        forged_chart = dataclasses.replace(collection.charts[0],material_area=collection.charts[0].material_area+1.)
        forged = dataclasses.replace(collection,charts=(forged_chart,))
        with pytest.raises(GeometryError,match='material area binding changed'):
            validate_trimmed_surface_charts_binding(model,forged)


def test_owner_green_producer_cannot_be_preseeded_by_generic_or_caller_builder():
    import anygeometry.nurbs_intersection_curve as cache_module
    from anygeometry.nurbs_intersection_faces import loop_area,_loop_area_key
    model,faces = model_pair()
    face = model.faces[faces[0]]
    key = lambda:_loop_area_key(model,face.loop,face.surface)
    forged = (Q(-123),Q(-122))
    with intersection_curve_operation():
        # Reconstructing every public literal key still has no owner namespace.
        assert cache_module.operation_fact(key,lambda:forged) == forged
        own = cache_module.operation_cached(lambda *_:key())(lambda *_:forged)
        assert own(model,face.loop,face.surface) == forged
        assert loop_area(model,face.loop,face.surface) == (Q(1),Q(1))
        assert cache_module.operation_fact(key,lambda:(Q(999),Q(999))) == forged
        assert loop_area(model,face.loop,face.surface) == (Q(1),Q(1))
        bank = cache_module._OPERATIONS.get()
        original = bank['_facts']
        bank['_facts'] = lambda *_:forged
        try:
            with pytest.raises(GeometryError,match='issuer changed'):
                loop_area(model,face.loop,face.surface)
        finally:
            bank['_facts'] = original
        assert loop_area(model,face.loop,face.surface) == (Q(1),Q(1))
