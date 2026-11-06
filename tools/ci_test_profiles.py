"""Explicit test tiers, independent of slow marks and runtime scientific gates.

Unit is a deliberately small curated development cohort. Unknown modules are
integration. Qualification is a selected scientific tier; only full is the
release union. This module imports no tests or numerical packages.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

TIERS = ('unit', 'integration', 'qualification')
PROFILES = ('unit', 'fast', 'integration', 'qualification', 'full')
UNIT_MODULES = frozenset({
    # Source-backed primitive/tolerance cohort; hosted development CI measures
    # actual runtime. This declaration is not a measured speed guarantee.
    'tests/test_predicates.py',
    'tests/test_ci_test_profiles.py',
    'tests/test_ci_structure_followup.py',
    'tests/test_ci_module_scheduling.py',
    'tests/test_ci_tools.py',
    'tests/test_ci_collection_durations.py',
    'tests/test_ci_planning_profile.py',
})
QUALIFICATION_MODULES = frozenset({
    # Root-adopted engineering proof tier; full release selection still includes every case.
    'tests/test_bounded_nurbs_clipping.py',
    'tests/test_bounded_nurbs.py',
    'tests/test_local_sweep_competition.py',
    'tests/test_local_contact_field.py',
    'tests/test_local_contact_transport.py',
    'tests/test_local_contact_stars.py',
    'tests/nurbs_surface_intersection/test_generic.py',
    'tests/test_certified_hydrostatics_admission.py',
    'tests/test_certified_hydrostatics_ch3_admission.py',
    'tests/test_certified_hydrostatics_ch3_kernel.py',
    'tests/test_certified_hydrostatics_ch3_loops.py',
    'tests/test_certified_hydrostatics_ch3_query.py',
    'tests/test_certified_hydrostatics_ch3_verify.py',
    'tests/test_certified_hydrostatics_ch3e_verify.py',
    'tests/test_certified_hydrostatics_query.py',
    'tests/test_certified_hydrostatics_verify.py',
    'tests/test_t2b_planning_route.py',
    'tests/test_planning_r8_integration.py',
    'tests/test_planning_numpy_context.py',
    'tests/test_planning_host_connected.py',
    'tests/test_planning_loaded_manifest.py',
    'tests/test_planning_resource_lifetime.py',
    'tests/test_planning_r8_controller.py',
})
POLICY = 'explicit-unit-qualification-default-integration-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def canonical_profile(profile):
    if profile not in PROFILES:
        raise ValueError('unknown test profile: ' + str(profile))
    return 'unit' if profile == 'fast' else profile


def tier(module):
    if UNIT_MODULES & QUALIFICATION_MODULES:
        raise ValueError('test tier module declarations overlap')
    return 'unit' if module in UNIT_MODULES else 'qualification' if module in QUALIFICATION_MODULES else 'integration'


def partition(nodes):
    from ci_isolated_tests import validate_nodes
    validate_nodes(nodes)
    groups = {name: [] for name in TIERS}
    for node in nodes:
        groups[tier(node.split('::', 1)[0])].append(node)
    flat = [node for group in groups.values() for node in group]
    if len(flat) != len(set(flat)) or set(flat) != set(nodes):
        raise ValueError('test tiers are not disjoint/exhaustive')
    return groups


def select(nodes, profile):
    profile = canonical_profile(profile)
    groups = partition(nodes)
    return list(nodes) if profile == 'full' else groups[profile]


def validate_selection(profile, selectors, keyword, marker, env):
    """No inherited narrowing is allowed to masquerade as a full release run."""
    profile = canonical_profile(profile)
    if env.get('ANYGEOMETRY_TEST_PROFILE') or env.get('ANYGEOMETRY_TEST_TIER'):
        raise ValueError('inherited test profile/tier selector is forbidden; use --profile explicitly')
    if profile == 'full' and (selectors or keyword or marker or env.get('PYTEST_ADDOPTS', '').strip()):
        raise ValueError('full profile forbids selectors, -k/-m and inherited PYTEST_ADDOPTS')


def module_manifest(root, profile):
    root = Path(root).resolve()
    profile = canonical_profile(profile)
    files = sorted((root / 'tests').rglob('test_*.py'))
    if not files or len(files) > 10000:
        raise ValueError('invalid source module inventory size')
    modules = []
    source_hashes = {}
    groups = {name: [] for name in TIERS}
    for path in files:
        if not path.resolve().is_relative_to(root / 'tests'):
            raise ValueError('test source escapes repository')
        name = path.relative_to(root).as_posix()
        modules.append(name)
        groups[tier(name)].append(name)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    chosen = modules if profile == 'full' else groups[profile]
    if not chosen:
        raise ValueError('test profile has no modules: ' + profile)
    return dict(schema='ci-test-profile-modules-v1', policy=POLICY, profile=profile,
                scope='full-source-module-inventory' if profile == 'full' else 'partial-development-only',
                release_acceptance=False, inventory_modules=modules, tiers=groups,
                tier_counts={name: len(group) for name, group in groups.items()},
                selected_modules=chosen, inventory_sha256=digest(modules),
                source_sha256=source_hashes,
                declarations_sha256=digest([sorted(UNIT_MODULES), sorted(QUALIFICATION_MODULES)]))


def node_receipt(nodes, profile):
    profile = canonical_profile(profile)
    return dict(schema='ci-test-profile-nodes-v1', policy=POLICY, profile=profile,
                scope='full-collected-node-inventory' if profile == 'full' else 'partial-development-only',
                release_acceptance=False, inventory_count=len(nodes), inventory_sha256=digest(nodes),
                tiers=partition(nodes), selected=select(nodes, profile))
