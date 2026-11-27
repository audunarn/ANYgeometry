"""The lift's arithmetic is the same on every platform and memory layout (suite run 2026-11-27).

Main's 0.4.7 gate (``test_quadrature_fixed_order_sums_20261124``) found Apple Accelerate's BLAS dot
not repeatable across memory offsets on Intel macOS.  The lift writes every three-term dot out as
``x0*w0 + x1*w1 + x2*w2`` (left to right, IEEE products and sums), so none of its reductions goes
through BLAS (``@``, ``np.dot``, ``.dot``, ``np.matmul``, ``np.inner``, ``np.vdot``, ``np.einsum``).
Its tests do not read git history either: a shallow CI checkout has none, and a rebase drops it.
"""
import ast
import pathlib
import tempfile

from anygeometry import _display_lift as L

TESTS = pathlib.Path(__file__).resolve().parent


_BLAS = ("dot", "matmul", "inner", "vdot", "einsum")


def _blas_sites(path):
    hits = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            hits.append(node.lineno)
        elif isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Attribute) and node.func.attr in _BLAS)
                or (isinstance(node.func, ast.Name) and node.func.id in _BLAS)):
            hits.append(node.lineno)
    return sorted(set(hits))


def test_the_lift_has_no_blas_reduction():
    assert _blas_sites(pathlib.Path(L.__file__)) == []


def test_the_guard_sees_every_form():
    """``@``, ``np.dot``, ``x.dot`` and a bare imported ``dot`` are each caught (Codex N3 suite L)."""
    text = "a @ b" + chr(10) + "np.dot(a, b)" + chr(10) + "a.dot(b)" + chr(10) + "dot(a, b)" + chr(10) + "np.einsum('i,i', a, b)" + chr(10)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "probe.py"
        path.write_text(text, encoding="utf-8")
        assert _blas_sites(path) == [1, 2, 3, 4, 5]


def test_no_lift_test_reads_git_history():
    own = pathlib.Path(__file__).name
    readers = []
    for p in sorted(TESTS.glob("test_display_tessellation_lift*.py")):
        text = p.read_text(encoding="utf-8")
        if p.name != own and ('"git"' in text or "'git'" in text):
            readers.append(p.name)
    assert readers == []
