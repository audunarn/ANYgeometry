"""Typed primitive predicates, tolerance boundaries, and metamorphic cases."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anygeometry.errors import GeometryError
from anygeometry.predicates import (
    IntersectionKind,
    IntersectionQuality,
    ParameterRange,
    qualified_line_line,
    qualified_line_cylinder,
    qualified_line_plane,
    qualified_plane_plane,
    qualified_segment_segment,
)
from anygeometry.surfaces import Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy, feature_extent


def test_tolerance_policy_validates_and_scales_dimensional_values() -> None:
    policy = TolerancePolicy(
        length=2.0e-8,
        merge_length=3.0e-7,
        angular=4.0e-10,
        parameter=5.0e-10,
        area=6.0e-16,
        surface_residual=7.0e-8,
        relative_length=8.0e-12,
        relative_area=9.0e-14,
    )

    scaled = policy.scaled(1000.0)

    assert scaled.length == pytest.approx(2.0e-5)
    assert scaled.merge_length == pytest.approx(3.0e-4)
    assert scaled.area == pytest.approx(6.0e-10)
    assert scaled.surface_residual == pytest.approx(7.0e-5)
    assert scaled.angular == policy.angular
    assert scaled.parameter == policy.parameter
    assert scaled.relative_length == policy.relative_length
    assert scaled.relative_area == policy.relative_area
    with pytest.raises(GeometryError, match="positive finite"):
        TolerancePolicy(length=0.0)
    with pytest.raises(GeometryError, match="positive finite"):
        TolerancePolicy(angular=float("nan"))
    with pytest.raises(FrozenInstanceError):
        policy.length = 1.0  # type: ignore[misc]


def test_feature_extent_and_effective_tolerance_ignore_global_origin() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)))
    translated = points + np.asarray((8.0e11, -4.0e11, 2.0e11))
    policy = TolerancePolicy(relative_length=1.0e-6)

    assert feature_extent(points) == pytest.approx(13.0)
    assert feature_extent(translated) == pytest.approx(13.0)
    assert policy.effective_length(feature_extent(points)) == pytest.approx(
        policy.effective_length(feature_extent(translated))
    )


def test_qualified_line_line_cross_skew_coincident_and_physical_parameters() -> None:
    crossing = qualified_line_line(
        (-2.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (0.5, -3.0, 0.0),
        (0.0, 0.25, 0.0),
    )
    skew = qualified_line_line(
        (-2.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, -3.0, 1.0),
        (0.0, 1.0, 0.0),
    )
    coincident = qualified_line_line(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (-3.0, 0.0, 0.0),
    )

    assert crossing.kind is IntersectionKind.CROSS
    assert np.asarray(crossing.witnesses) == pytest.approx(
        np.asarray(((0.5, 0.0, 0.0),))
    )
    assert crossing.components[0].first_parameter == pytest.approx((2.5,))
    assert crossing.components[0].second_parameter == pytest.approx((3.0,))
    assert crossing.quality is IntersectionQuality.VERIFIED_APPROXIMATE
    assert skew.kind is IntersectionKind.DISJOINT
    assert skew.components == ()
    assert coincident.kind is IntersectionKind.COINCIDENT
    assert coincident.components[0].direction == pytest.approx((1.0, 0.0, 0.0))


def test_qualified_line_line_fails_closed_for_invalid_and_ill_conditioned_lines() -> None:
    degenerate = qualified_line_line(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    ill_conditioned = qualified_line_line(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0e-16, 0.0),
        policy=TolerancePolicy(angular=1.0e-20),
    )

    assert degenerate.kind is IntersectionKind.UNCLASSIFIED
    assert not degenerate.classified
    assert degenerate.quality is IntersectionQuality.UNVERIFIED
    assert ill_conditioned.kind is IntersectionKind.UNCLASSIFIED
    assert ill_conditioned.diagnostics == ("ill_conditioned_line_line",)


def test_qualified_line_cylinder_distinguishes_cross_tangent_and_generatrix() -> None:
    cylinder = Cylinder(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        4.0,
    )
    crossing = qualified_line_cylinder(
        (-3.0, 0.0, 2.0), (5.0, 0.0, 0.0), cylinder
    )
    tangent = qualified_line_cylinder(
        (-3.0, 2.0, 2.0), (1.0, 0.0, 0.0), cylinder
    )
    generatrix = qualified_line_cylinder(
        (2.0, 0.0, -2.0), (0.0, 0.0, 7.0), cylinder
    )

    assert crossing.kind is IntersectionKind.CROSS
    assert np.asarray(crossing.witnesses) == pytest.approx(
        np.asarray(((-2.0, 0.0, 2.0), (2.0, 0.0, 2.0)))
    )
    assert crossing.components[0].first_parameter == pytest.approx((1.0,))
    assert crossing.components[1].first_parameter == pytest.approx((5.0,))
    assert tangent.kind is IntersectionKind.TANGENT
    assert generatrix.kind is IntersectionKind.OVERLAP_CURVE
    assert generatrix.components[0].first_parameter_range == ParameterRange(2.0, 6.0)


def test_qualified_line_plane_distinguishes_cross_containment_and_parallel() -> None:
    plane = Plane(
        np.asarray((10.0, 20.0, 3.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 4.0, 0.0)),
    )
    crossing = qualified_line_plane(
        (11.0, 22.0, -2.0), (0.0, 0.0, 20.0), plane
    )
    contained = qualified_line_plane(
        (11.0, 22.0, 3.0), (5.0, 0.0, 0.0), plane
    )
    parallel = qualified_line_plane(
        (11.0, 22.0, 4.0), (5.0, 0.0, 0.0), plane
    )

    assert crossing.kind is IntersectionKind.CROSS
    assert np.asarray(crossing.witnesses) == pytest.approx(
        np.asarray(((11.0, 22.0, 3.0),))
    )
    assert crossing.components[0].first_parameter == pytest.approx((5.0,))
    assert crossing.components[0].second_parameter == pytest.approx((0.5, 0.5))
    assert contained.kind is IntersectionKind.OVERLAP_CURVE
    assert contained.components[0].direction == pytest.approx((1.0, 0.0, 0.0))
    assert parallel.kind is IntersectionKind.DISJOINT


def test_qualified_plane_plane_distinguishes_cross_coincident_and_parallel() -> None:
    horizontal = Plane(
        np.asarray((0.0, 0.0, 2.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 3.0, 0.0)),
    )
    vertical = Plane(
        np.asarray((0.5, 0.0, 0.0)),
        np.asarray((0.0, 3.0, 0.0)),
        np.asarray((0.0, 0.0, 4.0)),
    )
    crossing = qualified_plane_plane(horizontal, vertical)
    coincident = qualified_plane_plane(
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 1.0),
        (3.0, -8.0, 2.0),
        (0.0, 0.0, -2.0),
    )
    parallel = qualified_plane_plane(
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 3.0),
        (0.0, 0.0, 1.0),
    )

    assert crossing.kind is IntersectionKind.CROSS
    assert crossing.components[0].direction == pytest.approx((0.0, 1.0, 0.0))
    point = np.asarray(crossing.witnesses[0])
    assert point[0] == pytest.approx(0.5)
    assert point[2] == pytest.approx(2.0)
    assert coincident.kind is IntersectionKind.COINCIDENT
    assert parallel.kind is IntersectionKind.DISJOINT


def test_segment_segment_classifies_interior_3d_cross_and_skew() -> None:
    crossing = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),
        (0.0, 2.0, 1.0),
        (2.0, 0.0, 1.0),
    )
    skew = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),
        (0.0, 2.0, 1.25),
        (2.0, 0.0, 1.25),
    )

    assert crossing.kind is IntersectionKind.CROSS
    assert np.asarray(crossing.witnesses) == pytest.approx(
        np.asarray(((1.0, 1.0, 1.0),))
    )
    assert crossing.components[0].first_parameter == pytest.approx((0.5,))
    assert crossing.components[0].second_parameter == pytest.approx((0.5,))
    assert skew.kind is IntersectionKind.DISJOINT


def test_segment_segment_classifies_endpoint_touch_and_outside_crossing() -> None:
    endpoint = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 3.0, 0.0),
    )
    outside = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, -1.0, 0.0),
        (2.0, 1.0, 0.0),
    )

    assert endpoint.kind is IntersectionKind.TOUCH_POINT
    assert endpoint.components[0].first_parameter == (1.0,)
    assert endpoint.components[0].second_parameter == (0.0,)
    assert outside.kind is IntersectionKind.DISJOINT


def test_collinear_segment_partial_and_full_reversed_overlap_have_oriented_ranges() -> None:
    partial = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),
    )
    full_reversed = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    assert partial.kind is IntersectionKind.OVERLAP_CURVE
    assert np.asarray(partial.witnesses) == pytest.approx(
        np.asarray(((2.0, 0.0, 0.0), (4.0, 0.0, 0.0)))
    )
    assert partial.components[0].first_parameter_range == ParameterRange(0.5, 1.0)
    assert partial.components[0].second_parameter_range == ParameterRange(0.0, 0.5)
    assert full_reversed.kind is IntersectionKind.COINCIDENT
    assert full_reversed.components[0].first_parameter_range == ParameterRange(0.0, 1.0)
    assert full_reversed.components[0].second_parameter_range == ParameterRange(1.0, 0.0)
    assert full_reversed.components[0].second_parameter_range.lower == 0.0
    assert full_reversed.components[0].second_parameter_range.upper == 1.0


def test_segment_near_tolerance_cases_use_local_length_not_global_coordinates() -> None:
    policy = TolerancePolicy(
        length=1.0e-6,
        merge_length=2.0e-6,
        angular=1.0e-12,
        parameter=1.0e-12,
        area=1.0e-12,
        surface_residual=1.0e-6,
        relative_length=1.0e-15,
        relative_area=1.0e-15,
    )
    near_touch = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0 + 0.5e-6, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        policy=policy,
    )
    outside_touch = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0 + 1.5e-6, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        policy=policy,
    )
    large_origin = 1.0e8
    translated_parallel = qualified_segment_segment(
        (large_origin, large_origin, large_origin),
        (large_origin + 1.0, large_origin, large_origin),
        (large_origin, large_origin + 1.0e-5, large_origin),
        (large_origin + 1.0, large_origin + 1.0e-5, large_origin),
        policy=policy,
    )

    assert near_touch.kind is IntersectionKind.TOUCH_POINT
    assert outside_touch.kind is IntersectionKind.DISJOINT
    assert translated_parallel.kind is IntersectionKind.DISJOINT


def test_parallel_and_skew_segment_residuals_straddle_length_tolerance() -> None:
    policy = TolerancePolicy(
        length=1.0e-6,
        merge_length=2.0e-6,
        angular=1.0e-12,
        parameter=1.0e-12,
        area=1.0e-12,
        surface_residual=1.0e-6,
        relative_length=1.0e-15,
        relative_area=1.0e-15,
    )

    parallel_near = qualified_segment_segment(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.5e-6, 0.0),
        (1.0, 0.5e-6, 0.0),
        policy=policy,
    )
    parallel_far = qualified_segment_segment(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 1.5e-6, 0.0),
        (1.0, 1.5e-6, 0.0),
        policy=policy,
    )
    skew_near = qualified_segment_segment(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.5e-6),
        (0.0, 1.0, 0.5e-6),
        policy=policy,
    )
    skew_far = qualified_segment_segment(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 1.5e-6),
        (0.0, 1.0, 1.5e-6),
        policy=policy,
    )

    assert parallel_near.kind is IntersectionKind.COINCIDENT
    assert parallel_far.kind is IntersectionKind.DISJOINT
    assert skew_near.kind is IntersectionKind.CROSS
    assert skew_far.kind is IntersectionKind.DISJOINT


@pytest.mark.parametrize("scale", (1.0e-6, 1.0, 1.0e6))
def test_segment_classification_is_uniform_scale_metamorphic(scale: float) -> None:
    policy = TolerancePolicy(
        length=1.0e-7,
        merge_length=2.0e-7,
        surface_residual=3.0e-7,
    ).scaled(scale)
    result = qualified_segment_segment(
        np.asarray((-2.0, 0.0, 0.0)) * scale,
        np.asarray((2.0, 0.0, 0.0)) * scale,
        np.asarray((0.5, -3.0, 0.0)) * scale,
        np.asarray((0.5, 3.0, 0.0)) * scale,
        policy=policy,
    )

    assert result.kind is IntersectionKind.CROSS
    assert result.components[0].first_parameter == pytest.approx((0.625,))
    assert result.components[0].second_parameter == pytest.approx((0.5,))
    assert np.asarray(result.witnesses[0]) / scale == pytest.approx((0.5, 0.0, 0.0))


def test_predicates_are_translation_metamorphic_and_do_not_mutate_inputs() -> None:
    inputs = [
        np.asarray((-2.0, 0.0, 0.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.5, -3.0, 0.0)),
        np.asarray((0.5, 3.0, 0.0)),
    ]
    originals = [item.copy() for item in inputs]
    translation = np.asarray((7.5e8, -2.5e8, 4.0e8))

    base = qualified_segment_segment(*inputs)
    translated = qualified_segment_segment(*(item + translation for item in inputs))

    assert base.kind is translated.kind is IntersectionKind.CROSS
    assert translated.components[0].first_parameter == pytest.approx(
        base.components[0].first_parameter
    )
    assert translated.components[0].second_parameter == pytest.approx(
        base.components[0].second_parameter
    )
    assert np.asarray(translated.witnesses[0]) - translation == pytest.approx(
        base.witnesses[0]
    )
    assert all(np.array_equal(item, original) for item, original in zip(inputs, originals))


def test_infinite_predicates_are_translation_metamorphic() -> None:
    translation = np.asarray((6.0e8, -9.0e8, 3.0e8))
    first_line_point = np.asarray((-2.0, 1.0, 4.0))
    second_line_point = np.asarray((0.5, -3.0, 4.0))
    first_direction = np.asarray((1.0, 0.0, 0.0))
    second_direction = np.asarray((0.0, 1.0, 0.0))
    base_lines = qualified_line_line(
        first_line_point,
        first_direction,
        second_line_point,
        second_direction,
    )
    moved_lines = qualified_line_line(
        first_line_point + translation,
        first_direction,
        second_line_point + translation,
        second_direction,
    )

    plane_origin = np.asarray((0.0, 0.0, 2.0))
    plane_normal = np.asarray((0.0, 0.0, 1.0))
    base_line_plane = qualified_line_plane(
        (1.0, 2.0, -3.0), (0.0, 0.0, 1.0), plane_origin, plane_normal
    )
    moved_line_plane = qualified_line_plane(
        np.asarray((1.0, 2.0, -3.0)) + translation,
        (0.0, 0.0, 1.0),
        plane_origin + translation,
        plane_normal,
    )

    vertical_origin = np.asarray((0.5, 0.0, 0.0))
    vertical_normal = np.asarray((1.0, 0.0, 0.0))
    base_planes = qualified_plane_plane(
        plane_origin, plane_normal, vertical_origin, vertical_normal
    )
    moved_planes = qualified_plane_plane(
        plane_origin + translation,
        plane_normal,
        vertical_origin + translation,
        vertical_normal,
    )

    for base, moved in (
        (base_lines, moved_lines),
        (base_line_plane, moved_line_plane),
        (base_planes, moved_planes),
    ):
        assert base.kind is moved.kind is IntersectionKind.CROSS
        assert np.asarray(moved.witnesses[0]) - translation == pytest.approx(
            base.witnesses[0]
        )
        assert moved.components[0].first_parameter == pytest.approx(
            base.components[0].first_parameter
        )
        assert moved.components[0].second_parameter == pytest.approx(
            base.components[0].second_parameter
        )


def test_uncertain_or_unsupported_segment_inputs_fail_closed() -> None:
    zero_length = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    non_finite = qualified_segment_segment(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, float("nan")),
        (0.0, 1.0, 0.0),
    )

    assert zero_length.kind is IntersectionKind.UNCLASSIFIED
    assert non_finite.kind is IntersectionKind.UNCLASSIFIED
    assert zero_length.diagnostics
    assert non_finite.diagnostics
