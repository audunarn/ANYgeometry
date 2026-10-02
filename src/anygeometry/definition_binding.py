"""Content binding for immutable analytic owner results."""
from dataclasses import fields, is_dataclass
from enum import Enum
from collections.abc import Mapping
from uuid import UUID
import hashlib
import json
import numpy as np


def definition_value(value):
    if isinstance(value, np.ndarray):
        return {"array":value.tolist(),"shape":list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        # Only explicitly marked implementation caches are outside the definition.
        # Equality/representation settings do not weaken analytic content binding.
        return {"type":type(value).__qualname__,"fields":{
            field.name:definition_value(getattr(value,field.name)) for field in fields(value)
            if field.metadata.get("definition") is not False}}
    if isinstance(value, Mapping):
        return {str(key):definition_value(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [definition_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    return value


def definition_checksum(value):
    payload=json.dumps(definition_value(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
