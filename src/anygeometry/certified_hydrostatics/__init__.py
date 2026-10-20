"""Certified hydrostatics for closed planar shells (design: docs/agent_plans/CERTIFIED_HYDROSTATICS_DESIGN.md).

Slice CH1a: exact snapshot, the six admission checks, refusal categories, work budget, implementation
identity and ``check_preparation``. Slice CH1b: ``HalfSpaceQuery`` and ``query_certified_hydrostatics``. The verifier comes in CH1c."""
from ._engine import query_certified_hydrostatics
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._identity import ImplementationIdentity
from ._prepare import (CertifiedHydrostaticsPolicy, PreparedCertifiedHydrostatics, check_preparation,
                       prepare_certified_hydrostatics)
from ._forms import QuantityResult
from ._query import HalfSpaceQuery
from ._result import CertifiedHydrostaticsResult, Contacts, SectionResult, Topology, TopologyLoop
from ._snapshot import ExactSnapshot
from ._work import WorkBudget

__all__ = ["CertifiedHydrostaticsPolicy", "CertifiedHydrostaticsRefusal", "CertifiedHydrostaticsResult", "Contacts",
           "ExactSnapshot", "HalfSpaceQuery", "ImplementationIdentity", "PreparedCertifiedHydrostatics",
           "QuantityResult", "RefusalCategory", "SectionResult", "Topology", "TopologyLoop", "WorkBudget",
           "check_preparation", "prepare_certified_hydrostatics", "query_certified_hydrostatics"]
