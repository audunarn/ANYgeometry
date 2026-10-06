"""Owner-issued first-preparation permits and transient attachment identity drafts."""
from weakref import WeakKeyDictionary,ref

from .errors import GeometryError


_permits=WeakKeyDictionary()
_drafts=WeakKeyDictionary()
_seals=WeakKeyDictionary()
_derived=WeakKeyDictionary()
_plan_bindings={}


def _seal_sources(model,binding):
    from .prepared_face_preimages import _binding_checksum
    if binding.attachment_source_ids is not None:
        _seals[model]=_binding_checksum(binding)


def _validate_sources(model,binding):
    from .prepared_face_preimages import _binding_checksum
    if binding.attachment_source_ids is not None and _seals.get(model)!=_binding_checksum(binding):
        raise GeometryError('preparation epoch attachment map is not owner issued')


def _issue_epoch(model):
    _permits[model]=model.model_id


def _has_epoch_permit(model):
    return _permits.get(model)==model.model_id


def _consume_epoch(model):
    _permits.pop(model,None)


def _begin_attachment_draft(model,binding):
    if binding is None or binding.attachment_source_ids is None:
        return
    _drafts[model]=dict(binding.attachment_source_ids)
    _derived[model]=set(binding.epoch_derived_attachment_ids or ())


def _record_attachment_descendants(model,source,children):
    draft=_drafts.get(model)
    if draft is None:return
    if source not in draft:
        if source in _derived.get(model,set()):
            _derived[model].discard(source);_derived[model].update(children)
        return
    ancestor=draft.pop(source)
    for child in children:
        existing=draft.get(child)
        if existing is not None and existing!=ancestor:
            raise GeometryError('preparation epoch attachment identity collision')
        draft[child]=ancestor


def _finish_attachment_draft(model):
    draft=_drafts.pop(model,None)
    if draft is None:
        return None
    if not set(draft)<=set(model.attachments):
        raise GeometryError('preparation epoch attachment identity was lost')
    derived=_derived.pop(model,set())
    if set(draft)&derived or set(draft)|derived!=set(model.attachments):
        raise GeometryError('epoch attachments lack complete owner creation provenance')
    return tuple(sorted(draft.items())),tuple(sorted(derived))


def _discard_attachment_draft(model):
    _drafts.pop(model,None)
    _derived.pop(model,None)


def _record_created_attachment(model,identifier):
    if model in _drafts:_derived[model].add(identifier)


def _epoch_plan_required(model):
    receipt=getattr(model,'_prepared_face_preimages_receipt',None)
    return (_has_epoch_permit(model) or model in _seals or (receipt is not None and
            getattr(receipt[0],'attachment_source_ids',None) is not None))


def _issue_native_plan(plan,supports):
    key=id(plan)
    def cleanup(reference):
        current=_plan_bindings.get(key)
        if current is not None and current[0] is reference:_plan_bindings.pop(key,None)
    _plan_bindings[key]=(ref(plan,cleanup),supports)


def _native_plan_supports(plan):
    entry=_plan_bindings.get(id(plan))
    if entry is None or entry[0]() is not plan:
        raise GeometryError('epoch intersection plan native support binding is unavailable')
    return entry[1]
