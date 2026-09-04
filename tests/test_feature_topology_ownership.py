"""Exact feature/topology ownership for intent-first consumers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from anygeometry import (
    EntityRef,
    FeatureOutputRef,
    FeatureRegistry,
    FeatureTopologyRole,
    GeometryModel,
    feature_entity_owners,
)


def test_composite_generator_owns_internal_topology_but_authored_geometry_does_not():
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    point = geometry.features.append(
        "geometry.point",
        parameters={"position": (-2.0, 0.0, 0.0)},
    )
    cylinder = geometry.features.append(
        "generator.cylinder",
        parameters={
            "radius": 1.0,
            "height": 2.0,
            "circumferential_segments": 8,
        },
    )

    assert geometry.regenerate_features().success
    owners = feature_entity_owners(geometry)
    point_ref = geometry.features.get(point.feature_id).outputs["point"]
    assert point_ref not in owners

    current_outputs = {
        current
        for output in geometry.features.get(cylinder.feature_id).outputs.values()
        for current in geometry.resolve_ref(output)
    }
    assert current_outputs
    assert {owners[output] for output in current_outputs} == {
        cylinder.feature_id
    }
    assert len([ref for ref in owners if ref.kind == "vertex"]) == 32
    assert len([ref for ref in owners if ref.kind == "edge"]) == 24
    assert len([ref for ref in owners if ref.kind == "face"]) == 8


def test_modifier_does_not_claim_authored_topology():
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    point = geometry.features.append(
        "geometry.point",
        parameters={"position": (1.0, 0.0, 0.0)},
    )
    geometry.features.append(
        "geometry.transform",
        parameters={"matrix": np.eye(4)},
        inputs={
            "entities": (
                FeatureOutputRef(point.feature_id, "point", "vertex"),
            )
        },
    )

    assert geometry.regenerate_features().success
    assert feature_entity_owners(geometry) == {}


def test_modifier_preserves_composite_owner_through_exact_lineage():
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    plate = geometry.features.append(
        "generator.plate",
        parameters={"length": 2.0, "width": 1.0},
    )
    geometry.features.append(
        "geometry.split_face",
        parameters={"axis": 0, "fraction": 0.5},
        inputs={
            "face": (
                FeatureOutputRef(plate.feature_id, "face/1", "face"),
            )
        },
    )

    assert geometry.regenerate_features().success
    owners = feature_entity_owners(geometry)
    assert len(geometry.faces) == 2
    assert {owners[EntityRef("face", face_id)] for face_id in geometry.faces} == {
        plate.feature_id
    }


def test_registry_topology_roles_are_strict_and_removed_with_executor():
    registry = FeatureRegistry()

    def executor(_geometry, _feature, _inputs):
        return {}

    with pytest.raises(ValueError, match="topology_role"):
        registry.register("vendor.bad", executor, topology_role="container")

    registry.register(
        "vendor.feature",
        executor,
        topology_role=FeatureTopologyRole.AUTHORED,
    )
    assert registry.topology_role("vendor.feature") is FeatureTopologyRole.AUTHORED
    with pytest.raises(ValueError, match="conflicts with.*topology_role"):
        registry.register(
            "vendor.feature",
            executor,
            replace=True,
            topology_role=FeatureTopologyRole.COMPOSITE,
        )
    registry.unregister("vendor.feature")
    assert registry.topology_role("vendor.feature") is None


def test_owner_query_does_not_deep_copy_feature_history():
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    geometry.features.append(
        "generator.plate",
        parameters={"length": 2.0, "width": 1.0},
    )
    assert geometry.regenerate_features().success

    with patch("anygeometry.features.deepcopy", side_effect=AssertionError):
        owners = feature_entity_owners(geometry)

    assert owners
    assert len(geometry.features) == 1
