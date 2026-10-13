"""Pure-Python references for the optional compiled kernels.

Every function exported by the compiled module ``anygeometry._core`` has a
reference here with the same signature and the same caller-allocated output
convention. ``anygeometry.backend`` selects one of the two at import; the
differential tests call both and compare them (owner decision D1: identical
classification, coordinates within a stated bound).
"""
from .analytic_surfaces import plane_evaluate
from .bspline import bspline_evaluate

__all__ = ["bspline_evaluate", "plane_evaluate"]
