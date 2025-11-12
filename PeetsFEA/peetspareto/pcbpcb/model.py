from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import joblib
import lightgbm  # type: ignore
import numpy as np

from .config import (
    ModelArtifactMetadata,
    ModelArtifactSelectionError,
    PCBPCBModelConfig,
    default_legacy_config,
)
from .schemas import DesignVector

COPPER_MODEL_FEATURE_NAMES: tuple[str, ...] = (
    "freq",
    "w1",
    "l1_leg",
    "l1_top",
    "l2",
    "h1",
    "l1_center",
    "Tx_turns",
    "Tx_width",
    "Tx_height",
    "Tx_space_x",
    "Tx_space_y",
    "Tx_preg",
    "Rx_width",
    "Rx_height",
    "Rx_space_x",
    "Rx_space_y",
    "Rx_preg",
    "g2",
    "Tx_layer_space_x",
    "Tx_layer_space_y",
    "Tx_current",
    "Rx_current_optimetric",
)

CORE_MODEL_FEATURE_NAMES: tuple[str, ...] = (
    *COPPER_MODEL_FEATURE_NAMES[:-1],
    "magnetizing_current_optimetric",
)

COPPER_TARGETS: tuple[str, ...] = (
    "Lmt",
    "Llt",
    "copperloss_Tx",
    "copperloss_Rx1",
    "copperloss_Rx2",
)

CORE_TARGETS: tuple[str, ...] = (
    "coreloss",
    "magnetizing_copperloss_Tx",
    "magnetizing_copperloss_Rx1",
    "magnetizing_copperloss_Rx2",
    "B_left",
    "B_center",
    "B_top_left",
)


@dataclass(frozen=True)
class PredictionProvenance:
    generated_at: datetime
    batch_size: int
    config_hash: str
    artifact_metadata: tuple[ModelArtifactMetadata, ...]
    library_versions: Mapping[str, str]


@dataclass(frozen=True)
class PCBPCBPrediction(Mapping[str, float]):
    design: DesignVector
    lmt: float
    llt: float
    copperloss_tx: float
    copperloss_rx1: float
    copperloss_rx2: float
    coreloss: float
    magnetizing_copperloss_tx: float
    magnetizing_copperloss_rx1: float
    magnetizing_copperloss_rx2: float
    b_left: float
    b_center: float
    b_top_left: float
    magnetizing_current: float
    total_loss: float
    provenance: PredictionProvenance
    _mapping: Mapping[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        payload = {
            "Lmt": self.lmt,
            "Llt": self.llt,
            "copperloss_Tx": self.copperloss_tx,
            "copperloss_Rx1": self.copperloss_rx1,
            "copperloss_Rx2": self.copperloss_rx2,
            "coreloss": self.coreloss,
            "magnetizing_copperloss_Tx": self.magnetizing_copperloss_tx,
            "magnetizing_copperloss_Rx1": self.magnetizing_copperloss_rx1,
            "magnetizing_copperloss_Rx2": self.magnetizing_copperloss_rx2,
            "B_left": self.b_left,
            "B_center": self.b_center,
            "B_top_left": self.b_top_left,
            "magnetizing_current": self.magnetizing_current,
            "total_loss": self.total_loss,
        }
        object.__setattr__(self, "_mapping", payload)

    def __getitem__(self, key: str) -> float:
        return float(self._mapping[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def as_dict(self) -> dict[str, float]:
        return dict(self._mapping)


class PCBPCBModel:
    """
    Production-ready LightGBM inference harness for the pcbpcb domain.

    ``PCBPCBModel()`` may be constructed with no arguments, in which case it attempts
    to locate the legacy ``legacy_codes/EVDD_PCB_PCB/model`` directory. This mirrors
    the behavior expected by the zero-config ``run_pcbpcb_nsga2`` helper.
    """

    def __init__(self, config: PCBPCBModelConfig | None = None):
        self.config = config or default_legacy_config()
        self._models: dict[str, object] = {}
        self._artifact_metadata: tuple[ModelArtifactMetadata, ...] = ()
        self._library_versions = _detect_library_versions()

        self._load_models()
        self.model_version = self.config.fingerprint()
        self.artifact_checksum = _combined_checksum(self._artifact_metadata)

    def predict_batch(
        self, specs: Sequence[DesignVector]
    ) -> list[PCBPCBPrediction]:
        if not specs:
            return []

        copper_rows, copper_matrix = self._build_copper_matrix(specs)
        predictions: dict[str, np.ndarray] = {}

        for target in COPPER_TARGETS:
            predictions[target] = self._predict(target, copper_matrix)

        magnetizing_current = self._compute_magnetizing_current(
            specs, predictions["Lmt"]
        )

        core_matrix = self._build_core_matrix(copper_rows, magnetizing_current)
        for target in CORE_TARGETS:
            predictions[target] = self._predict(target, core_matrix)

        total_loss = (
            predictions["copperloss_Tx"]
            + predictions["copperloss_Rx1"]
            + predictions["copperloss_Rx2"]
            + predictions["coreloss"]
            + predictions["magnetizing_copperloss_Tx"]
            + predictions["magnetizing_copperloss_Rx1"]
            + predictions["magnetizing_copperloss_Rx2"]
        )

        provenance = PredictionProvenance(
            generated_at=datetime.now(timezone.utc),
            batch_size=len(specs),
            config_hash=self.config.fingerprint(),
            artifact_metadata=self._artifact_metadata,
            library_versions=self._library_versions,
        )

        results: list[PCBPCBPrediction] = []
        for idx, spec in enumerate(specs):
            results.append(
                PCBPCBPrediction(
                    design=spec,
                    lmt=float(predictions["Lmt"][idx]),
                    llt=float(predictions["Llt"][idx]),
                    copperloss_tx=float(predictions["copperloss_Tx"][idx]),
                    copperloss_rx1=float(predictions["copperloss_Rx1"][idx]),
                    copperloss_rx2=float(predictions["copperloss_Rx2"][idx]),
                    coreloss=float(predictions["coreloss"][idx]),
                    magnetizing_copperloss_tx=float(
                        predictions["magnetizing_copperloss_Tx"][idx]
                    ),
                    magnetizing_copperloss_rx1=float(
                        predictions["magnetizing_copperloss_Rx1"][idx]
                    ),
                    magnetizing_copperloss_rx2=float(
                        predictions["magnetizing_copperloss_Rx2"][idx]
                    ),
                    b_left=float(predictions["B_left"][idx]),
                    b_center=float(predictions["B_center"][idx]),
                    b_top_left=float(predictions["B_top_left"][idx]),
                    magnetizing_current=float(magnetizing_current[idx]),
                    total_loss=float(total_loss[idx]),
                    provenance=provenance,
                )
            )

        return results

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load_models(self) -> None:
        models: dict[str, object] = {}
        artifact_metadata: list[ModelArtifactMetadata] = []

        for target, path in sorted(self.config.artifact_paths.items()):
            try:
                model = joblib.load(path)
            except Exception as exc:  # pragma: no cover - exercised via monkeypatch
                raise ModelArtifactSelectionError(
                    "Failed to load LightGBM artifact for "
                    f"target '{target}' at '{path}'. "
                    "If the pickle was moved or corrupted, regenerate it before "
                    "calling `run_pcbpcb_nsga2()`."
                ) from exc
            _validate_feature_schema(target, model)
            models[target] = model
            artifact_metadata.append(
                ModelArtifactMetadata(
                    target=target,
                    path=Path(path),
                    checksum=_sha256(path),
                    n_features=len(getattr(model, "feature_name_", []) or []),
                )
            )

        missing = set(COPPER_TARGETS + CORE_TARGETS) - set(models)
        if missing:
            raise ModelArtifactSelectionError(
                f"Config missing LightGBM models for targets: {sorted(missing)}"
            )

        self._models = models
        self._artifact_metadata = tuple(
            sorted(artifact_metadata, key=lambda meta: meta.target)
        )

    def _predict(self, target: str, matrix: np.ndarray) -> np.ndarray:
        model = self._models[target]
        try:
            raw = model.predict(matrix)  # type: ignore[attr-defined]
        except TypeError:
            booster = getattr(model, "booster_", None)
            if booster is None:
                raise
            raw = booster.predict(matrix)
        predictions = np.asarray(raw, dtype=np.float64)
        return predictions

    def _build_copper_matrix(
        self, specs: Sequence[DesignVector]
    ) -> tuple[list[list[float]], np.ndarray]:
        rows: list[list[float]] = []
        for spec in specs:
            row = spec.as_feature_row(
                tx_current=spec.effective_tx_current(self.config.tx_current_amps),
                rx_current=spec.effective_rx_current(self.config.rx_current_amps),
            )
            rows.append(row)
        return rows, np.asarray(rows, dtype=np.float64)

    def _build_core_matrix(
        self,
        copper_rows: Sequence[Sequence[float]],
        magnetizing_current: np.ndarray,
    ) -> np.ndarray:
        rows = []
        for base_row, mag in zip(copper_rows, magnetizing_current, strict=True):
            row = list(base_row)
            row[-1] = float(mag)
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def _compute_magnetizing_current(
        self, specs: Sequence[PCBPCBDesignSpec], lmt: np.ndarray
    ) -> np.ndarray:
        freqs = np.asarray([spec.freq for spec in specs], dtype=np.float64)
        voltages = np.asarray(
            [spec.input_voltage for spec in specs], dtype=np.float64
        )
        lmt_values = np.asarray(lmt, dtype=np.float64)

        numerator = voltages * math.sqrt(2.0)
        denominator = (
            2.0 * math.pi * freqs * 1e3 * lmt_values * 1e-6
        )  # matches legacy formula

        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0.0,
        )


def _validate_feature_schema(target: str, model: object) -> None:
    names = tuple(getattr(model, "feature_name_", ()) or ())
    if not names:
        return

    expected = (
        COPPER_MODEL_FEATURE_NAMES
        if target in COPPER_TARGETS
        else CORE_MODEL_FEATURE_NAMES
    )
    if names != expected:
        raise ValueError(
            f"Unexpected feature order for '{target}'. "
            f"Expected {expected}, received {names}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_checksum(
    metadata: Sequence[ModelArtifactMetadata],
) -> str:
    digest = hashlib.sha256()
    for item in sorted(metadata, key=lambda meta: meta.target):
        digest.update(item.checksum.encode("utf-8"))
    return digest.hexdigest()


def _detect_library_versions() -> Mapping[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }
