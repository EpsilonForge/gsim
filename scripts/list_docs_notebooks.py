"""List the source notebooks referenced by the documentation navigation."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "docs" / "zensical.toml"


class DocsNotebookConfigError(ValueError):
    """Raised when the docs navigation cannot produce a safe notebook list."""


def _walk_nav_leaves(value: Any) -> Iterator[str]:
    """Yield string leaves from a nested Zensical navigation structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_nav_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_nav_leaves(item)
    else:
        raise DocsNotebookConfigError(
            f"Unsupported project.nav value of type {type(value).__name__}"
        )


def _local_nav_path(value: str) -> PurePosixPath | None:
    """Return a validated local path, ignoring external navigation links."""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return None
    if parsed.query or parsed.fragment:
        value = parsed.path
    if "\\" in value:
        raise DocsNotebookConfigError(f"Navigation path uses a backslash: {value!r}")

    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise DocsNotebookConfigError(
            f"Navigation path escapes the docs root: {value!r}"
        )
    return path


def list_docs_notebooks(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    repository_root: Path | None = None,
) -> list[str]:
    """Return repository-relative notebooks referenced under ``nbs/`` in nav."""
    config_path = Path(config_path)
    if repository_root is None:
        repository_root = config_path.parent.parent

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    try:
        nav = config["project"]["nav"]
    except (KeyError, TypeError) as error:
        raise DocsNotebookConfigError(
            "Missing project.nav in Zensical config"
        ) from error

    notebooks: list[str] = []
    seen: set[str] = set()
    for leaf in _walk_nav_leaves(nav):
        nav_path = _local_nav_path(leaf)
        if (
            nav_path is None
            or not nav_path.parts
            or nav_path.parts[0] != "nbs"
            or nav_path.suffix != ".md"
        ):
            continue

        notebook_path = nav_path.with_suffix(".ipynb").as_posix()
        if notebook_path in seen:
            raise DocsNotebookConfigError(
                f"Duplicate docs notebook reference: {notebook_path}"
            )
        seen.add(notebook_path)

        if not (repository_root / notebook_path).is_file():
            raise DocsNotebookConfigError(
                f"Docs notebook does not exist: {notebook_path}"
            )
        notebooks.append(notebook_path)

    return notebooks


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON array for use as a GitHub Actions matrix",
    )
    return parser


def main() -> int:
    """Run the command-line interface."""
    args = _build_parser().parse_args()
    try:
        notebooks = list_docs_notebooks(
            args.config, repository_root=args.repository_root
        )
    except (DocsNotebookConfigError, OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"error: {error}") from error

    if args.json:
        print(json.dumps(notebooks, separators=(",", ":")))
    else:
        print(*notebooks, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
