"""Implementation identity of the running artifact (design section 6, proof schema ``artifact``).

Distribution name and version, the SHA-256 of every file of the installed ``anygeometry`` package
(``__pycache__`` and ``.pyc`` files are build products and excluded), the versions of the runtime
dependencies and the Python version. Hashing reads 64 KiB chunks, one work unit per chunk."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ._work import Meter

__all__ = ["ImplementationIdentity", "implementation_identity", "canonical_json"]

CHUNK = 64 * 1024
DISTRIBUTION = "anygeometry"


def canonical_json(value) -> bytes:
    """Canonical JSON bytes: sorted keys, no spaces, UTF-8, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")


def sha256_chunks(data: bytes, meter: Meter) -> str:
    """SHA-256 of ``data`` hashed in 64 KiB chunks, one unit per chunk (at least one)."""
    digest = hashlib.sha256()
    for start in range(0, max(len(data), 1), CHUNK):
        meter.unit("hash_chunk")
        digest.update(data[start:start + CHUNK])
    return digest.hexdigest()


def _file_digest(path: Path, meter: Meter) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            meter.unit("hash_chunk")
            block = stream.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImplementationIdentity:
    distribution: str
    distribution_version: str | None       # None: not installed as a distribution (source tree on the path)
    package_version: str                   # ``anygeometry.__version__``
    files: tuple[tuple[str, str], ...]     # (posix path relative to the package, SHA-256), sorted
    dependencies: tuple[tuple[str, str | None], ...]   # (name, installed version or None), sorted
    python: str

    def as_dict(self) -> dict:
        return {"distribution": self.distribution, "distribution_version": self.distribution_version,
                "package_version": self.package_version, "files": [list(item) for item in self.files],
                "dependencies": [list(item) for item in self.dependencies], "python": self.python}

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()


def _dependencies() -> tuple[tuple[str, str | None], ...]:
    names: set[str] = set()
    try:
        requirements = metadata.requires(DISTRIBUTION) or []
    except metadata.PackageNotFoundError:
        requirements = None
    if requirements is None:                  # source tree: read the declaration next to ``src``
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        try:
            import tomllib
            requirements = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get("dependencies", [])
        except (OSError, KeyError, ValueError):
            requirements = []
    for requirement in requirements:
        if "extra ==" in requirement.replace('"', "'").replace("'", ""):
            continue                          # optional extras are not runtime dependencies
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    versions = []
    for name in sorted(names):
        try:
            versions.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            versions.append((name, None))
    return tuple(versions)


def implementation_identity(meter: Meter) -> ImplementationIdentity:
    from .. import __version__ as package_version      # relative: the package must not import itself by name
    root = Path(__file__).resolve().parent.parent       # the anygeometry package directory
    paths = sorted((path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"), key=lambda path: path.relative_to(root).as_posix())
    files = tuple((path.relative_to(root).as_posix(), _file_digest(path, meter)) for path in paths)
    try:
        installed = metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        installed = None
    python = f"{platform.python_implementation()} {sys.version.split()[0]}"
    return ImplementationIdentity(DISTRIBUTION, installed, str(package_version), files, _dependencies(), python)
