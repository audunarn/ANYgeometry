"""Explicit preparation epochs use the actual public owner path."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import json,os,sys
from importlib.metadata import version
import pytest,anygeometry
from anygeometry import clone_prepared_geometry,GeometryError,plan_intersections,apply_intersections
from anygeometry.batch_intersections import IntersectionBatchPolicy
from anygeometry.serialization import to_dict
from anygeometry.prepared_face_preimages import query_prepared_face_preimages,_binding_checksum
from anygeometry.preparation_epochs import _has_epoch_permit
from test_native_attachment_reference_scope import _axis_fixture,_cut,_prepare,query,validate


def test_runtime():
    origin=Path(anygeometry.__file__).resolve()
    assert origin==Path(__file__).resolve().parents[1]/'src/anygeometry/__init__.py'
    assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    Path(os.environ['EPOCH_RUNTIME_EVIDENCE']).write_text(json.dumps(dict(executable=sys.executable,python_version=sys.version,
        source_origin=str(origin),numpy_version=version('numpy'),pytest_version=version('pytest'),threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2))


def fresh(duplicates=False):
    old,*_=_axis_fixture(cuts=(1.,),duplicates=duplicates)
    epoch=clone_prepared_geometry(old,new_preparation_epoch=True)
    _cut(epoch,3.)
    return old,epoch


def test_production_subsequent_cutter_epoch_and_duplicate_identity():
    old,epoch=fresh(True);original=to_dict(old);baseline=set(epoch.attachments);roots=tuple(epoch.faces)
    assert epoch.model_id!=old.model_id and _has_epoch_permit(epoch)
    _prepare(epoch)
    binding=query_prepared_face_preimages(epoch)
    assert set(source for _,source in binding.attachment_source_ids)==baseline
    assert set(key for key,_ in binding.attachment_source_ids)<=set(epoch.attachments)
    assert not _has_epoch_permit(epoch)
    receipt=query(epoch,roots);rows=receipt.inventory['attachment_native_maps']
    assert {row['source_attachment_id'] for row in rows}==baseline
    assert all(row['classification']!='refused' for row in rows),rows
    assert all(row['source_identity_semantics']=='authenticated_epoch_map' for row in rows)
    validate(epoch,receipt);assert to_dict(old)==original
    copied=clone_prepared_geometry(epoch);query_prepared_face_preimages(copied)
    _cut(epoch,2.)
    with pytest.raises(GeometryError):query_prepared_face_preimages(epoch)


def test_permit_not_inherited_and_plan_identity_and_preapply_staleness():
    old,epoch=fresh();copied=epoch.clone(preserve_identity=True)
    assert not _has_epoch_permit(copied)
    assert epoch.revision==copied.revision
    plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    with pytest.raises(GeometryError):apply_intersections(old,plan,policy='connect')
    _cut(epoch,2.)
    with pytest.raises(GeometryError,match='stale'):apply_intersections(epoch,plan,policy='connect')
    assert _has_epoch_permit(epoch)
    _prepare(epoch);query_prepared_face_preimages(epoch)


def test_cancellation_after_actual_remap_retry(monkeypatch):
    import anygeometry.preparation_epochs as lifecycle
    old,epoch=fresh();before=to_dict(epoch);plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    hit=[False];actual=lifecycle._record_attachment_descendants
    def remap(*args):
        actual(*args);hit[0]=True
    monkeypatch.setattr(lifecycle,'_record_attachment_descendants',remap)
    with pytest.raises(GeometryError,match='cancelled'):
        apply_intersections(epoch,plan,policy=IntersectionBatchPolicy(cancellation_check=lambda:hit[0]))
    assert hit[0] and to_dict(epoch)==before and _has_epoch_permit(epoch)
    apply_intersections(epoch,plan,policy='connect');query_prepared_face_preimages(epoch)


def test_publication_failure_unchanged_and_forged_map_refused(monkeypatch):
    import anygeometry.prepared_face_preimages as owner
    old,epoch=fresh();before=to_dict(epoch);plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    actual=owner._publish_application_preimages
    def fail(*args):raise RuntimeError('publication sentinel')
    monkeypatch.setattr(owner,'_publish_application_preimages',fail)
    with pytest.raises(RuntimeError,match='sentinel'):apply_intersections(epoch,plan,policy='connect')
    assert to_dict(epoch)==before and _has_epoch_permit(epoch)
    monkeypatch.setattr(owner,'_publish_application_preimages',actual)
    apply_intersections(epoch,plan,policy='connect')
    binding=query_prepared_face_preimages(epoch)
    forged=replace(binding,attachment_source_ids=tuple(reversed(binding.attachment_source_ids)))
    epoch._prepared_face_preimages_receipt=(forged,_binding_checksum(forged))
    with pytest.raises(GeometryError):query_prepared_face_preimages(epoch)


def test_clone_retains_raw_support_and_source_guard(monkeypatch):
    from anygeometry import GeometryModel
    from anygeometry.surfaces import Cylinder
    from anygeometry.native_support_snapshots import capture_native_supports
    from anygeometry.generators.structural import cylinder
    model=cylinder(1.,2.,circumferential_segments=3)
    for face in model.faces.values():
        if isinstance(face.surface,Cylinder):object.__setattr__(face.surface,'_circumferential',face.surface._circumferential*(1+2**-50))
    native=capture_native_supports(model);before=to_dict(model)
    epoch=clone_prepared_geometry(model,new_preparation_epoch=True)
    assert capture_native_supports(epoch)==native and epoch.revision==model.revision
    assert to_dict(model)==before and not hasattr(epoch,'_prepared_face_preimages_receipt')


def test_post_adoption_sealing_failure_rolls_back_and_retry(monkeypatch):
    import anygeometry.preparation_epochs as lifecycle
    old,epoch=fresh();before=to_dict(epoch);plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    actual=lifecycle._seal_sources
    def fail(model,binding):
        if model is epoch:raise RuntimeError('actual owner sealing sentinel')
        return actual(model,binding)
    monkeypatch.setattr(lifecycle,'_seal_sources',fail)
    with pytest.raises(RuntimeError,match='sentinel'):apply_intersections(epoch,plan,policy='connect')
    assert to_dict(epoch)==before and _has_epoch_permit(epoch)
    assert not hasattr(epoch,'_prepared_face_preimages_receipt')
    monkeypatch.setattr(lifecycle,'_seal_sources',actual)
    apply_intersections(epoch,plan,policy='connect');query_prepared_face_preimages(epoch)


def test_actual_owner_publisher_failure_rollback(monkeypatch):
    import anygeometry.prepared_face_preimages as owner
    old,epoch=fresh();before=to_dict(epoch);plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    actual=owner._publish_application_preimages
    def fail(model,*args):
        if model is epoch:raise RuntimeError('original publisher sentinel')
        return actual(model,*args)
    monkeypatch.setattr(owner,'_publish_application_preimages',fail)
    with pytest.raises(RuntimeError,match='sentinel'):apply_intersections(epoch,plan,policy='connect')
    assert to_dict(epoch)==before and _has_epoch_permit(epoch)
    monkeypatch.setattr(owner,'_publish_application_preimages',actual)
    apply_intersections(epoch,plan,policy='connect');query_prepared_face_preimages(epoch)


def test_omitted_descendant_cannot_be_relabelled_derived():
    old,epoch=fresh();_prepare(epoch);binding=query_prepared_face_preimages(epoch)
    original_ids={source for _,source in binding.attachment_source_ids}
    key=next(key for key,_ in binding.attachment_source_ids if key not in original_ids)
    forged=replace(binding,attachment_source_ids=tuple(row for row in binding.attachment_source_ids if row[0]!=key),
        epoch_derived_attachment_ids=tuple(sorted(binding.epoch_derived_attachment_ids+(key,))))
    epoch._prepared_face_preimages_receipt=(forged,_binding_checksum(forged))
    with pytest.raises(GeometryError,match='owner issued'):query_prepared_face_preimages(epoch)


def test_cached_epoch_apply_rechecks_callback_authoring_without_rollback():
    old,epoch=fresh();plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    apply_intersections(epoch,plan,policy='connect');before=epoch.revision;added=[]
    def author():
        added.append(epoch.add_point(10.,10.,10.))
        return False
    with pytest.raises(GeometryError,match='changed'):
        apply_intersections(epoch,plan,policy=IntersectionBatchPolicy(cancellation_check=author))
    assert epoch.revision>before and added[0] in epoch.vertices


@pytest.mark.parametrize('edit', ('lookup', 'receipt'))
def test_cached_epoch_apply_rechecks_same_revision_callback_state(edit):
    _,epoch=fresh();plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    apply_intersections(epoch,plan,policy='connect')
    revision=epoch.revision;replacement=[]
    def change():
        if edit=='lookup':
            first,second=sorted(epoch.vertices)[:2]
            epoch._vertices[first],epoch._vertices[second]=epoch._vertices[second],epoch._vertices[first]
        else:
            receipt=tuple(list(epoch._intersection_application_receipt))
            epoch._intersection_application_receipt=receipt
            replacement.append(receipt)
        return False
    with pytest.raises(GeometryError):
        apply_intersections(epoch,plan,policy=IntersectionBatchPolicy(cancellation_check=change))
    assert epoch.revision==revision
    if replacement:
        assert epoch._intersection_application_receipt is replacement[0]


def test_cached_epoch_apply_rejects_unserialized_raw_support_mutation():
    from anygeometry.generators.structural import cylinder
    from anygeometry.surfaces import Cylinder
    authored=cylinder(1.,1.,circumferential_segments=3)
    epoch=clone_prepared_geometry(authored,new_preparation_epoch=True)
    epoch.add_plate(epoch.add_points(((2.,.5,-.5),(2.,.5,1.5),(-2.,.5,1.5),(-2.,.5,-.5))))
    plan=plan_intersections(epoch,tuple(epoch.faces),policy='connect')
    support=next(face.surface for face in epoch.faces.values() if isinstance(face.surface,Cylinder))
    raw=support._circumferential.copy();before=to_dict(epoch)
    object.__setattr__(support,'_circumferential',raw*(1+2**-50))
    assert to_dict(epoch)==before
    with pytest.raises(GeometryError,match='native support'):
        apply_intersections(epoch,plan,policy='connect')
    object.__setattr__(support,'_circumferential',raw)
    apply_intersections(epoch,plan,policy='connect');before=to_dict(epoch);revision=epoch.revision
    support=next(face.surface for face in epoch.faces.values() if isinstance(face.surface,Cylinder))
    object.__setattr__(support,'_circumferential',support._circumferential*(1+2**-50))
    assert to_dict(epoch)==before and epoch.revision==revision
    with pytest.raises(GeometryError,match='native support'):
        apply_intersections(epoch,plan,policy='connect')
