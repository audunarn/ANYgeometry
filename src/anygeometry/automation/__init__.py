"""ANYgeometry automation protocol version 1.

Natural-language interpretation and transport belong in adapters.  This
package accepts only bounded declarative data and delegates every mutation to
the existing kernel owner APIs.
"""

from .planning import apply_plan, execute_query, plan_commands
from .schema import (
    automation_dumps,
    automation_json_schema,
    automation_loads,
    describe_capabilities,
    tool_catalog,
)
from .selection import describe_entities, describe_model, select_entities
from .types import (
    ApplyResult,
    AutomationError,
    AutomationResponse,
    Command,
    CommandBatch,
    EditPlan,
    EntitySummary,
    PROTOCOL_VERSION,
    Quantity,
    SelectionResult,
    SelectionSpec,
)

__all__ = [
    "ApplyResult",
    "AutomationError",
    "AutomationResponse",
    "Command",
    "CommandBatch",
    "EditPlan",
    "EntitySummary",
    "PROTOCOL_VERSION",
    "Quantity",
    "SelectionResult",
    "SelectionSpec",
    "apply_plan",
    "automation_dumps",
    "automation_json_schema",
    "automation_loads",
    "describe_capabilities",
    "describe_entities",
    "describe_model",
    "execute_query",
    "plan_commands",
    "select_entities",
    "tool_catalog",
]
