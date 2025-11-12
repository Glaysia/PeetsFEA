"""
Export helpers for PCB-on-PCB Pareto runs.

Responsibilities:
- convert evaluation records into tidy tabular data
- persist Pareto CSVs and provenance bundles
- capture environment metadata for reproducibility
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ExportConfig
from .optimizer import ParetoResult
from .protocols import EvaluationRecord


def records_to_dataframe(
    records: Sequence[EvaluationRecord],
    *,
    include_measures: bool = True,
    include_provenance: bool = True,
) -> pd.DataFrame:
    """Transforms evaluation records into a flat dataframe ready for CSV export."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "spec_id": record.spec.identifier,
        }
        row.update(_flatten_mapping(record.spec.to_parameters(), prefix="param"))
        row.update(_flatten_mapping(record.objectives, prefix="obj"))
        if include_measures:
            row.update(_flatten_mapping(record.measures, prefix="measure"))
        if include_provenance:
            row.update(_flatten_mapping(record.provenance, prefix="prov"))
        rows.append(row)

    return pd.DataFrame(rows)


def write_export_bundle(
    result: ParetoResult,
    *,
    extra_provenance: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """
    Writes the Pareto CSV and provenance JSON based on the run's ExportConfig.

    Returns:
        Tuple of (csv_path, provenance_path).
    """

    export_config: ExportConfig = result.config.export
    export_dir = export_config.resolved_directory()
    export_dir.mkdir(parents=True, exist_ok=True)

    df = records_to_dataframe(
        result.front_records,
        include_measures=export_config.include_measures,
        include_provenance=export_config.include_provenance,
    )
    df.to_csv(export_config.csv_path(), index=False)

    provenance_payload = build_provenance_payload(result, extra=extra_provenance)
    export_config.provenance_path().write_text(json.dumps(provenance_payload, indent=2, sort_keys=True))
    return export_config.csv_path(), export_config.provenance_path()


def build_provenance_payload(
    result: ParetoResult,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Captures reproducibility metadata for a Pareto run."""

    payload: dict[str, Any] = {
        "timestamp": result.timestamp.isoformat(),
        "runtime_seconds": result.runtime_seconds,
        "pareto_size": len(result.front_records),
        "evaluations": len(result.all_records),
        "config": _serialize_dataclass(result.config),
        "model": result.model_metadata,
        "algorithm": result.algorithm_metadata,
        "environment": _capture_environment_snapshot(),
        "objective_names": [obj.name for obj in result.config.objectives],
    }
    if extra:
        payload["extra"] = _serialize_mapping(extra)
    return payload


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _flatten_mapping(mapping: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        flattened[f"{prefix}__{key}"] = _coerce_scalar(value)
    return flattened


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.generic,)):
        return value.item()
    return json.dumps(value, sort_keys=True, default=str)


def _serialize_dataclass(obj: Any) -> Any:
    if not is_dataclass(obj):
        return obj
    return _serialize_mapping(asdict(obj))


def _serialize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in mapping.items():
        serialized[key] = _serialize_value(value)
    return serialized


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize_dataclass(value)
    if isinstance(value, Mapping):
        return _serialize_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _capture_environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("pymoo", "numpy", "pandas", "lightgbm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
