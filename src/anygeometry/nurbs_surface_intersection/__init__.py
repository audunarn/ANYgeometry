"""Experimental certified NURBS x NURBS intersection kernel (not exported from the package root)."""
from ._certificate import LocalArcCertificate, certify_local_arc
from ._source import (IntersectionResult, IntersectionStats, IntersectionStatus,
                      NurbsSurfaceIntersectionCancelled, NurbsSurfaceIntersectionIntegrityError,
                      SourceDefinition)
from ._evaluate import ArcEvaluation, evaluate_arc, evaluate_arcs
from ._records import ArcRecord, ComponentRecord, StructuralCertificate, RationalProjectionCertificate
from ._source import NurbsSurfaceIntersectionRefused
from .query import query_nurbs_surface_intersection
from .curve import NurbsIntersectionCurve, PreparedCurveShape, encode_curve, decode_curve
from ._curve_bounds import CurveSegmentBounds

__all__ = [
    "NurbsIntersectionCurve", "PreparedCurveShape", "CurveSegmentBounds", "encode_curve", "decode_curve",
    "ArcEvaluation", "ArcRecord", "ComponentRecord", "StructuralCertificate", "RationalProjectionCertificate", "evaluate_arc", "evaluate_arcs",
    "NurbsSurfaceIntersectionRefused",
    "IntersectionResult", "IntersectionStats", "IntersectionStatus", "LocalArcCertificate",
    "NurbsSurfaceIntersectionCancelled", "NurbsSurfaceIntersectionIntegrityError", "SourceDefinition",
    "certify_local_arc", "query_nurbs_surface_intersection",
]
