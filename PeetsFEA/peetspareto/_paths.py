"""
Shared helpers for locating bundled legacy assets (LightGBM models, etc.).
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, TypeVar

Exc = TypeVar("Exc", bound=Exception)


def _candidate_roots(relative_path: Path, env_var: str | None) -> Iterable[Path]:
    """
    Yield possible installation roots for a legacy asset directory.

    The search order is:
    - explicit environment variable override
    - package root (site-packages/PeetsFEA)
    - source checkout root (one level above the package)
    - site-packages-style locations (purelib/platlib/data)
    - sys.prefix/PeetsFEA (where setuptools data-files land)
    """

    if env_var:
        override = os.environ.get(env_var)
        if override:
            yield Path(override).expanduser()

    here = Path(__file__).resolve()
    package_root = here.parents[1]
    project_root = here.parents[2]
    for base in (package_root, project_root):
        yield (base / relative_path).expanduser()

    for key in ("purelib", "platlib", "data"):
        path = sysconfig.get_path(key)
        if path:
            yield Path(path) / "PeetsFEA" / relative_path

    yield Path(sys.prefix) / "PeetsFEA" / relative_path


def resolve_legacy_root(
    relative_path: Path,
    *,
    env_var: str | None = None,
    description: str,
    error_cls: type[Exc] = FileNotFoundError,
) -> Path:
    """
    Resolve the first existing legacy asset directory.

    Args:
        relative_path: Path to the asset directory relative to the project/package root.
        env_var: Optional environment variable that can override the lookup.
        description: Human-readable description used in error messages.
        error_cls: Exception type to raise on failure.
    """

    checked: list[str] = []
    for candidate in _candidate_roots(relative_path, env_var):
        resolved = candidate.expanduser().resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    msg = f"Unable to locate {description}. Checked: {checked}"
    if env_var:
        msg += f". Set {env_var} to override."
    raise error_cls(msg)


__all__ = ["resolve_legacy_root"]
