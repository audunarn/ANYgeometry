"""Session-shared B1 builders, independent of pytest's conftest module names.

Callers read these models; mutations must use a private model.clone().
"""

_B1_MODELS: dict = {}


def b1_half_hull_shared(**options):
    from anygeometry.b1_example import build_b1_half_hull
    key = ("half", tuple(sorted(options.items())))
    if key not in _B1_MODELS:
        _B1_MODELS[key] = build_b1_half_hull(**options)
    return _B1_MODELS[key]


def b1_full_hull_shared():
    from anygeometry.b1_example import build_b1_hull
    if "full" not in _B1_MODELS:
        _B1_MODELS["full"] = build_b1_hull()
    return _B1_MODELS["full"]
