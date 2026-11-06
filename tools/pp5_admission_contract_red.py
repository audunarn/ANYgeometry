"""Unmet PP5 admission contracts, deliberately red on the named PP2 base.

Run explicitly; not part of normal test discovery. No planning or meshing.
The support-mutation counterexample is a passing guard, not an admission waiver.
"""
import dataclasses
from pathlib import Path
import sys
import numpy as np
import pytest
from anygeometry import _planning_qualification as q
from anygeometry._planning_executor import freeze_model
from anygeometry.definition_binding import definition_checksum
from anygeometry.surfaces import Cylinder

@pytest.fixture(scope='module')
def project_geometry():
    repo = Path(__file__).resolve().parents[1]
    root = repo.parents[1]
    sys.path.insert(0, str(root/'resume'))
    import env_setup
    original = env_setup.repo_root
    env_setup.repo_root = lambda name: repo if name == 'ANYgeometry' else (root/'int/ANYmesh' if name == 'ANYmesh' else original(name))
    env_setup.setup()
    from anyfem.io import load_project
    project = load_project(Path('C:/ANYworkspaceWork/AIP Gen2/ANYfem-local-assessment/hydrodynamics/AIP-Gen2-full-tripod-no-tower.anyfem'))
    assert not env_setup.foreign_modules()
    return project.geometry

@pytest.mark.parametrize('route,operand_count', [('elbow_sector60',97),('elbow_r7',240)])
def test_required_reference_input_can_enter_parallel_transport(project_geometry, route, operand_count):
    # Both authoritative reference scripts pass this full source model with subset operands.
    # The name/count identifies the blocked acceptance case, not two distinct model snapshots.
    assert len(project_geometry.faces) == 1083
    freeze_model(project_geometry)

def test_pristine_cylinder_record_has_a_qualified_adapter():
    support = Cylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),1.,1.)
    q._check_model_record_state({'surface':support})

def test_omitted_array_is_semantic_and_must_not_be_ignored():
    support = Cylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),1.,1.)
    assert set(vars(support)) - {f.name for f in dataclasses.fields(support)} == {'_circumferential'}
    before = support.evaluate(.25, 0.)
    digest = definition_checksum(support)
    support._circumferential[1] = 2.
    after = support.evaluate(.25, 0.)
    assert digest == definition_checksum(support)
    assert before[1] == 1. and after[1] == 2.
    assert not np.array_equal(before, after)
    with pytest.raises(q.QualificationDecline, match='non-field model record instance state'):
        q._check_model_record_state({'surface':support})