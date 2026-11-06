"""Shared, read-only B1 hulls, independent of pytest conftest loading order."""

_B1_MODELS: dict = {}


def b1_half_hull_shared(**options):
    """The B1 half-hull for these options, built once per test session (CI tiers C1: each build costs
    5-10 s). Shared: callers only read it; a test that changes its model takes ``model.clone()``."""
    from anygeometry.b1_example import build_b1_half_hull
    key = ("half", tuple(sorted(options.items())))
    if key not in _B1_MODELS:
        _B1_MODELS[key] = build_b1_half_hull(**options)
    return _B1_MODELS[key]


def b1_full_hull_shared():
    """The full B1 hull, built once per test session (read-only, as ``b1_half_hull_shared``)."""
    from anygeometry.b1_example import build_b1_hull
    if "full" not in _B1_MODELS:
        _B1_MODELS["full"] = build_b1_hull()
    return _B1_MODELS["full"]
