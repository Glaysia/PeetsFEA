"""
Configuration helpers for PCB-on-PCB optimization, inference, and schema defaults.

This module houses:
    - NSGA-II loop/export configs used by Agent C
    - LightGBM artifact selection helpers used by Agent B
    - Design-default loaders/mergers maintained by Agent A
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
import tomllib
from typing import Iterable, Mapping, MutableMapping, Sequence

from . import schemas
from .protocols import ObjectiveDefinition

__all__ = [
    "OptimizationLoopConfig",
    "ExportConfig",
    "ParetoRunConfig",
    "DEFAULT_TARGETS",
    "ModelArtifactSelectionError",
    "ModelArtifactMetadata",
    "PCBPCBModelConfig",
    "default_legacy_model_root",
    "default_legacy_config",
    "ConfigError",
    "DEFAULT_CHARACTERISTICS_PATH",
    "load_design_defaults",
    "merge_overrides",
    "resolve_design_vector",
]

# --------------------------------------------------------------------------- #
# Optimization configs (Agent C)
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class OptimizationLoopConfig:
    """
    NSGA-II loop tuning knobs (mirrors ``pymoo.algorithms.moo.nsga2.NSGA2``).
    """

    population_size: int = 64
    n_generations: int = 25
    n_offsprings: int | None = None
    seed: int = 7
    eliminate_duplicates: bool = True

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.n_generations <= 0:
            raise ValueError("n_generations must be positive")
        if self.n_offsprings is not None and self.n_offsprings <= 0:
            raise ValueError("n_offsprings must be positive when provided")


@dataclass(slots=True, frozen=True)
class ExportConfig:
    """Controls where Pareto CSVs and provenance bundles land."""

    directory: Path = Path("data/pareto")
    csv_filename: str = "pcbpcb_pareto.csv"
    provenance_filename: str = "pcbpcb_provenance.json"
    include_measures: bool = True
    include_provenance: bool = True

    def resolved_directory(self) -> Path:
        return self.directory.expanduser().resolve()

    def csv_path(self) -> Path:
        return self.resolved_directory() / self.csv_filename

    def provenance_path(self) -> Path:
        return self.resolved_directory() / self.provenance_filename


@dataclass(slots=True, frozen=True)
class ParetoRunConfig:
    """Full payload describing an optimization run."""

    objectives: Sequence[ObjectiveDefinition]
    loop: OptimizationLoopConfig = field(default_factory=OptimizationLoopConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    tag: str | None = None

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("At least one objective must be declared")


# --------------------------------------------------------------------------- #
# LightGBM artifact configs (Agent B)
# --------------------------------------------------------------------------- #

DEFAULT_TARGETS: tuple[str, ...] = (
    "Lmt",
    "Llt",
    "copperloss_Tx",
    "copperloss_Rx1",
    "copperloss_Rx2",
    "coreloss",
    "magnetizing_copperloss_Tx",
    "magnetizing_copperloss_Rx1",
    "magnetizing_copperloss_Rx2",
    "B_left",
    "B_center",
    "B_top_left",
)


class ModelArtifactSelectionError(RuntimeError):
    """Raised when the configured LightGBM artifacts cannot be located."""


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Describes a single artifact used during inference."""

    target: str
    path: Path
    checksum: str
    n_features: int


@dataclass(frozen=True)
class PCBPCBModelConfig:
    """
    Configuration for the LightGBM inference service.

    Attributes:
        artifact_paths: Mapping from target name to absolute artifact path.
        tx_current_amps: Default Tx current appended to the feature vector when the
            caller does not supply one.
        rx_current_amps: Default Rx current appended to the feature vector when the
            caller does not supply one.
    """

    artifact_paths: Mapping[str, Path] = field(default_factory=dict)
    tx_current_amps: float = 4.0
    rx_current_amps: float = 11.0

    def __post_init__(self) -> None:
        paths: MutableMapping[str, Path] = {}
        for target, path in self.artifact_paths.items():
            resolved = Path(path).expanduser().resolve()
            if not resolved.exists():
                raise ModelArtifactSelectionError(
                    f"Artifact for target '{target}' not found: {resolved}"
                )
            paths[target] = resolved
        object.__setattr__(self, "artifact_paths", dict(paths))

        missing = [t for t in DEFAULT_TARGETS if t not in self.artifact_paths]
        if missing:
            raise ModelArtifactSelectionError(
                f"Missing LightGBM artifacts for targets: {missing}"
            )

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(sorted(self.artifact_paths.keys()))

    def fingerprint(self) -> str:
        """Create a deterministic hash covering paths + runtime knobs."""
        payload = "|".join(
            f"{target}:{self.artifact_paths[target]}"
            for target in sorted(self.artifact_paths)
        )
        payload += f"|tx_current={self.tx_current_amps:.6f}"
        payload += f"|rx_current={self.rx_current_amps:.6f}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_directory(
        cls,
        model_root: Path,
        *,
        suffix: str | None = None,
        required_targets: Sequence[str] = DEFAULT_TARGETS,
    ) -> "PCBPCBModelConfig":
        """
        Build a configuration by scanning a directory containing ``*.pkl`` artifacts.

        Args:
            model_root: Directory containing LightGBM pickle artifacts.
            suffix: Optional explicit suffix (e.g., ``\"251021\"``) to use for every
                target. When omitted, the newest timestamped artifact per target
                (based on lexicographical order) will be selected automatically.
            required_targets: Override the list of target names to load.
        """

        root = Path(model_root).expanduser().resolve()
        if not root.exists():
            raise ModelArtifactSelectionError(
                f"Model directory does not exist: {root}"
            )

        artifacts: dict[str, Path] = {}
        for target in required_targets:
            artifacts[target] = (
                _select_by_suffix(root, target, suffix)
                if suffix
                else _select_latest(root, target)
            )

        return cls(artifacts)


def _select_latest(root: Path, target: str) -> Path:
    candidates = sorted(_iter_candidates(root, target))
    if not candidates:
        raise ModelArtifactSelectionError(
            f"No artifacts found for target '{target}' under {root}"
        )
    return candidates[-1][1]


def _select_by_suffix(root: Path, target: str, suffix: str) -> Path:
    path = root / f"{target}_{suffix}.pkl"
    if not path.exists():
        raise ModelArtifactSelectionError(
            f"Artifact '{path}' missing for target '{target}'"
        )
    return path


def _iter_candidates(root: Path, target: str) -> Iterable[tuple[str, Path]]:
    prefix = f"{target}_"
    for child in root.glob(f"{target}_*.pkl"):
        suffix = child.name.removeprefix(prefix).removesuffix(".pkl")
        yield suffix, child


def default_legacy_model_root() -> Path:
    """
    Locate ``legacy_codes/EVDD_PCB_PCB/model`` relative to the repo root.
    """

    candidate = (
        Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("legacy_codes", "EVDD_PCB_PCB", "model")
    )
    if not candidate.exists():
        raise ModelArtifactSelectionError(
            "Unable to locate the legacy LightGBM artifacts. "
            "Specify a model directory explicitly."
        )
    return candidate


def default_legacy_config(*, suffix: str | None = None) -> PCBPCBModelConfig:
    """Convenience helper that loads artifacts from the checked-in legacy folder."""

    return PCBPCBModelConfig.from_directory(
        default_legacy_model_root(), suffix=suffix
    )


# --------------------------------------------------------------------------- #
# Design defaults (Agent A)
# --------------------------------------------------------------------------- #


class ConfigError(RuntimeError):
    """Raised for malformed or missing PCB-PCB configuration files."""


DEFAULT_CHARACTERISTICS_PATH = (
    Path(__file__).resolve().parents[3] / "tmp" / "characteristics_defaults.toml"
)


def load_design_defaults(path: str | Path | None = None) -> schemas.DesignVector:
    """
    Load the default design vector from TOML.

    Parameters
    ----------
    path:
        Optional override path. When ``None``, the repo-level ``tmp`` directory
        is used.
    """
    target_path = Path(path) if path else DEFAULT_CHARACTERISTICS_PATH
    if not target_path.exists():
        raise ConfigError(f"Missing defaults file: {target_path}")

    try:
        payload = tomllib.loads(target_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - tomllib raises ValueError
        raise ConfigError(f"Invalid TOML in {target_path}") from exc

    try:
        defaults = payload["pcbpcb"]["defaults"]
    except KeyError as exc:
        raise ConfigError(
            f"{target_path} is missing the [pcbpcb.defaults] section."
        ) from exc

    return schemas.DesignVector.from_mapping(defaults)


def merge_overrides(
    base: schemas.DesignVector, overrides: Mapping[str, float | int] | None
) -> schemas.DesignVector:
    """
    Merge validated overrides onto a base design vector.

    ``overrides`` may contain a subset of the design fields; unknown keys raise
    :class:`ConfigError`.
    """
    if not overrides:
        return base

    unknown = set(overrides) - set(schemas.DESIGN_FIELD_ORDER)
    if unknown:
        raise ConfigError(f"Unknown override keys: {sorted(unknown)}")

    clean_kwargs: MutableMapping[str, float | int] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        schemas.VARIABLE_SPECS[key].validate_physical(float(value))
        clean_kwargs[key] = value

    return replace(base, **clean_kwargs)


def resolve_design_vector(
    overrides: Mapping[str, float | int] | None = None,
    defaults_path: str | Path | None = None,
) -> schemas.DesignVector:
    """
    Convenience helper that loads defaults and applies optional overrides.
    """
    base = load_design_defaults(defaults_path)
    return merge_overrides(base, overrides)
