"""Command-line inspection and example generation for ANYgeometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .generators import stiffened_panel
from .serialization import read_geometry, write_geometry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anygeometry",
        description="Inspect or create a neutral ANYgeometry document.",
    )
    parser.add_argument("geometry_file", nargs="?", type=Path)
    parser.add_argument(
        "--write-example",
        metavar="PATH",
        type=Path,
        help="write a small stiffened-panel geometry document",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the inspection summary as JSON",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _summary(path: Path) -> dict[str, object]:
    geometry = read_geometry(path)
    return {
        "path": str(path),
        "vertices": len(geometry.vertices),
        "edges": len(geometry.edges),
        "faces": len(geometry.faces),
        "groups": {
            name: len(geometry.group(name)) for name in sorted(geometry.groups)
        },
        "replacement_history": len(geometry.replacement_history()),
        "topology": "valid",
    }


def main(args: Sequence[str] | None = None) -> int:
    """Run the lightweight CLI and return a process exit status."""

    parser = _parser()
    options = parser.parse_args(args)
    if options.write_example is not None and options.geometry_file is not None:
        parser.error("geometry_file and --write-example are mutually exclusive")
    if options.write_example is not None:
        geometry = stiffened_panel(
            4.0,
            3.0,
            longitudinal_spacing=1.0,
            transverse_spacing=2.0,
        )
        write_geometry(options.write_example, geometry)
        print(f"Wrote {options.write_example}")
        return 0
    if options.geometry_file is None:
        parser.print_help()
        return 0

    summary = _summary(options.geometry_file)
    if options.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"{summary['path']}: {summary['vertices']} vertices, "
            f"{summary['edges']} edges, {summary['faces']} faces; topology valid"
        )
        groups = summary["groups"]
        if groups:
            print(
                "Groups: "
                + ", ".join(
                    f"{name}={count}" for name, count in groups.items()
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
