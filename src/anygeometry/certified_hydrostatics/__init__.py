"""Certified hydrostatics for closed planar shells (design: docs/agent_plans/CERTIFIED_HYDROSTATICS_DESIGN.md).

Slice CH1a: exact snapshot, the six admission checks, refusal categories, work budget, implementation
identity and ``check_preparation``. Queries, sections, quantities and the verifier come in later slices."""
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._identity import ImplementationIdentity
from ._prepare import (CertifiedHydrostaticsPolicy, PreparedCertifiedHydrostatics, check_preparation,
                       prepare_certified_hydrostatics)
from ._snapshot import ExactSnapshot
from ._work import WorkBudget

__all__ = ["CertifiedHydrostaticsPolicy", "CertifiedHydrostaticsRefusal", "ExactSnapshot",
           "ImplementationIdentity", "PreparedCertifiedHydrostatics", "RefusalCategory", "WorkBudget",
           "check_preparation", "prepare_certified_hydrostatics"]
