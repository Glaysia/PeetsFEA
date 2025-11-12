from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from PeetsFEA.peetspareto.pcbpcb.config import default_legacy_config
from PeetsFEA.peetspareto.pcbpcb.model import (
    CORE_TARGETS,
    COPPER_TARGETS,
    PCBPCBModel,
)
from PeetsFEA.peetspareto.pcbpcb.schemas import DesignVector

LEGACY_CSV = Path("legacy_codes/EVDD_PCB_PCB/output_data.csv")


def _load_sample_spec(strict: bool = False) -> DesignVector:
    row = pd.read_csv(LEGACY_CSV, nrows=1).iloc[0]
    return DesignVector.from_mapping(row, strict=strict)


def test_design_vector_from_dataset_allows_non_quantized_values() -> None:
    spec = _load_sample_spec(strict=False)
    assert spec.identifier.startswith("pcbpcb-")
    assert spec.effective_tx_current() > 0
    assert spec.effective_rx_current() > 0
    params = spec.to_parameters()
    assert params["spec_id"] == spec.identifier
    assert "freq" in params and "Tx_current" in params


def test_pcbpcb_model_matches_manual_pipeline() -> None:
    config = default_legacy_config()
    service = PCBPCBModel(config)
    spec = _load_sample_spec(strict=False)

    prediction = service.predict_batch([spec])[0]

    tx_current = spec.effective_tx_current(service.config.tx_current_amps)
    rx_current = spec.effective_rx_current(service.config.rx_current_amps)
    copper_row = np.asarray(
        [spec.as_feature_row(tx_current=tx_current, rx_current=rx_current)],
        dtype=np.float64,
    )

    expected: dict[str, float] = {}

    def predict_raw(target: str, matrix: np.ndarray) -> float:
        model = service._models[target]  # type: ignore[attr-defined]
        try:
            values = model.predict(matrix)
        except TypeError:
            booster = getattr(model, "booster_", None)
            if booster is None:
                raise
            values = booster.predict(matrix)
        return float(np.asarray(values, dtype=np.float64)[0])

    for target in COPPER_TARGETS:
        expected[target] = predict_raw(target, copper_row)

    freq = spec.freq
    voltage = spec.input_voltage
    denominator = 2.0 * math.pi * freq * 1e3 * expected["Lmt"] * 1e-6
    magnetizing_current = (
        voltage * math.sqrt(2.0) / denominator if denominator else 0.0
    )

    core_row = copper_row.copy()
    core_row[:, -1] = magnetizing_current
    for target in CORE_TARGETS:
        expected[target] = predict_raw(target, core_row)

    expected_total = (
        expected["copperloss_Tx"]
        + expected["copperloss_Rx1"]
        + expected["copperloss_Rx2"]
        + expected["coreloss"]
        + expected["magnetizing_copperloss_Tx"]
        + expected["magnetizing_copperloss_Rx1"]
        + expected["magnetizing_copperloss_Rx2"]
    )

    np.testing.assert_allclose(prediction["total_loss"], expected_total, rtol=1e-6)
    for target, value in expected.items():
        np.testing.assert_allclose(prediction[target], value, rtol=1e-6)

    np.testing.assert_allclose(
        prediction.magnetizing_current, magnetizing_current, rtol=1e-6
    )
    assert prediction.provenance.batch_size == 1
    assert prediction.provenance.artifact_metadata
