"""Reviewed source identities are portable across line endings, never content."""
from pathlib import Path

import pytest

from anygeometry.nurbs_surface_intersection import _dependencies
from anygeometry.nurbs_surface_intersection._source import NurbsSurfaceIntersectionIntegrityError


@pytest.fixture
def dependency_tree(tmp_path, monkeypatch):
    real = Path(_dependencies.__file__).parent.parent
    for name, _pin in _dependencies.PINS:
        (tmp_path / name).write_bytes((real / name).read_bytes().replace(b'\r\n', b'\n'))
    monkeypatch.setattr(_dependencies, '__file__', str(tmp_path / 'adapter' / '_dependencies.py'))
    return tmp_path


@pytest.mark.parametrize('ending', [b'\n', b'\r\n'])
def test_actual_dependency_files_accept_only_canonical_line_ending_difference(dependency_tree, ending):
    for name, _pin in _dependencies.PINS:
        path = dependency_tree / name
        path.write_bytes(path.read_bytes().replace(b'\n', ending))
    _dependencies.verify_dependencies()


@pytest.mark.parametrize('name', [name for name, _pin in _dependencies.PINS])
@pytest.mark.parametrize('change', [b'# changed comment\n', b'\xef\xbb\xbf', b'\r', b' '])
def test_non_line_ending_source_bytes_still_refuse(dependency_tree, name, change):
    path = dependency_tree / name
    path.write_bytes(change + path.read_bytes())
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError, match='dependency changed'):
        _dependencies.verify_dependencies()


def test_missing_dependency_is_not_replaced_by_another_source(dependency_tree):
    (dependency_tree / _dependencies.PINS[0][0]).unlink()
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError, match='dependency unavailable'):
        _dependencies.verify_dependencies()


def test_actual_current_source_supports_a_public_query_at_existing_tolerances():
    from test_rational_projection import _cheap_rational
    from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection, IntersectionStatus
    from anygeometry import TolerancePolicy
    first, second = _cheap_rational()
    result = query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED
