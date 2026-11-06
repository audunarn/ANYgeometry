"""Passive test evidence only: initial declines never qualify parallel/capture."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import types
import uuid

_ROOT=Path(__file__).resolve().parents[1]
FILE_LIMIT=4_000_000
_LABELS=frozenset(('warm8-cold','warm8-warm','callback-model','callback-code','callback-exception','callback-cancel'))


def capture_witness():
    """Observe only the real controller poll calling this test's callback."""
    from anygeometry import _planning_controller as controller
    run_code=controller.run.__code__
    poll_code=next(value for value in run_code.co_consts
                   if type(value) is types.CodeType and value.co_name=='poll')
    try:frame=sys._getframe(2)
    except ValueError:return dict(exact_poll=False,captured=False)
    if frame.f_code is not poll_code:return dict(exact_poll=False,captured=False)
    local=frame.f_locals
    source=local.get('source') is not None
    manifest=local.get('manifest') is not None
    bridge=local.get('bridge') is not None
    receipt={}
    parent=frame.f_back
    for _ in range(16):
        if parent is None:break
        if parent.f_code is run_code:
            candidate=parent.f_locals.get('receipt')
            if type(candidate) is dict:receipt=candidate
            break
        parent=parent.f_back
    return dict(exact_poll=True,captured=source and manifest and bridge and
                type(receipt.get('manifest_sha256')) is str,
                source_captured=source,manifest_captured=manifest,bridge_created=bridge,
                manifest_sha256=receipt.get('manifest_sha256'),
                model_sha256=receipt.get('model_sha256'),
                callback_calls=local.get('callback_calls'))


def _plain(value,depth=0):
    if depth>64:raise ValueError('observation nesting exceeds bound')
    kind=type(value)
    if kind in (str,int,float,bool) or value is None:return
    if kind is dict:
        if any(type(key) is not str for key in value):raise TypeError('observation keys must be strings')
        for item in value.values():_plain(item,depth+1)
    elif kind in (tuple,list):
        for item in value:_plain(item,depth+1)
    else:raise TypeError('observation requires plain JSON values')


def _retain(request,records,*,label,witness):
    if type(label) is not str or label not in _LABELS:raise ValueError('unknown observation label')
    raw=Path(request.getfixturevalue('tmp_path'))
    if raw.is_symlink():raise ValueError('observation base is a symlink')
    base=raw.resolve()
    if not base.is_dir():raise ValueError('observation base is not owned directory')
    payload=dict(schema='planning-runtime-observation-v1',status='UNVALIDATED',
        parallel_qualified=False,postcapture_qualified=False,records=records,witness=witness)
    _plain(payload)
    body=json.dumps(payload,sort_keys=True,allow_nan=False).encode('utf-8')
    if len(body)>FILE_LIMIT:raise ValueError('observation exceeds fixed file bound')
    target=base/('runtime-observation-'+label+'.json')
    if target.is_symlink() or target.exists():raise ValueError('observation target already exists')
    with target.open('xb') as stream:stream.write(body)
    if os.environ.get('CI')=='true':
        root=_ROOT.resolve()
        parent=root
        for name in ('ci-isolation','runtime-observations'):
            parent=parent/name
            if parent.is_symlink():raise ValueError('CI observation directory is a symlink')
            parent.mkdir(exist_ok=True)
            if not parent.resolve().is_relative_to(root):raise ValueError('CI observation escapes checkout')
        destination=parent/(label+'-'+uuid.uuid4().hex)
        destination.mkdir(exist_ok=False)
        with (destination/'observation.json').open('xb') as stream:stream.write(body)


@contextmanager
def retain_observation(request,records,*,label,witness=()):
    """Ordinary retention errors preserve a primary failure.

    KeyboardInterrupt/SystemExit propagate as terminal failures, with any
    original error retained by Python as exception context.
    """
    primary=None
    try:yield
    except BaseException as error:
        primary=error
        raise
    finally:
        try:_retain(request,records,label=label,witness=witness)
        except Exception as retention_error:
            if primary is None:raise
            BaseException.add_note(primary,'runtime observation retention failed: '+type(retention_error).__name__)
