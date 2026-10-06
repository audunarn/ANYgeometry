"""Content binding for immutable analytic owner results."""
from dataclasses import fields, is_dataclass
from enum import Enum
from collections.abc import Mapping
from uuid import UUID
import hashlib
import json
import numpy as np


def definition_value(value):
    # Preserve independent mutable expansions for each public output occurrence.
    return _definition_value(value, None)


def _definition_value(value, memo):
    # Exact primitive types are already their canonical expansion. Avoid
    # NumPy/dataclass/mapping dispatch for the many scalar control coordinates.
    # Subclasses (notably str-backed Enum) retain the original dispatch below.
    if value is None or type(value) in (str, int, float, bool):
        return value
    cached = memo.get(id(value)) if memo is not None else None
    if cached is not None:
        return cached[1]
    if isinstance(value, np.ndarray):
        result = {"array":value.tolist(),"shape":list(value.shape)}
    elif isinstance(value, np.generic):
        return value.item()
    elif is_dataclass(value):
        # Only explicitly marked implementation caches are outside the definition.
        # Equality/representation settings do not weaken analytic content binding.
        result = {"type":type(value).__qualname__,"fields":{
            field.name:_definition_value(getattr(value,field.name), memo) for field in fields(value)
            if field.metadata.get("definition") is not False}}
    elif isinstance(value, Mapping):
        result = {str(key):_definition_value(item, memo) for key,item in value.items()}
    elif isinstance(value, (tuple,list)):
        result = [_definition_value(item, memo) for item in value]
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, UUID):
        return str(value)
    else:
        return value
    # Publish only complete expansions, preserving rejection of recursive input.
    if memo is not None:
        memo[id(value)] = (value, result)
    return result


def definition_checksum(value):
    # Checksum serialization exposes no mutable output. Reuse only within this
    # call, retaining inputs to prevent reuse of transient object identities.
    payload=json.dumps(_definition_value(value, {}),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
