"""``_exact_point`` is memoised per plan (value key, cache owned by the plan's Work): repeated chart points in one
plane x NURBS plan are computed once (kernel strategy review proposal 2a)."""
from anygeometry.nurbs import NurbsSurface
from test_tight_x_memo import _plan


def test_repeated_chart_points_are_not_recomputed(monkeypatch):
    calls = [0]
    original = NurbsSurface.exact_point

    def counting(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NurbsSurface, "exact_point", counting)
    plan = _plan(.5, 200000)
    assert plan is not None
    # 2840 surface evaluations before the memo (2827 chart points, 257 distinct)
    assert calls[0] < 800, calls[0]
