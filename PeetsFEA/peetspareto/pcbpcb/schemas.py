"""
Typed schema definitions and validators for the PCB-to-PCB optimizer.

The legacy NSGA-II notebook used integer-valued design vectors that get
scaled into physical units before reaching the LightGBM predictors. This
module makes those translations explicit so downstream components can rely
on typed objects instead of loose dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Mapping, Sequence

__all__ = [
    "SchemaValidationError",
    "VariableSpec",
    "VARIABLE_SPECS",
    "DESIGN_FIELD_ORDER",
    "DEFAULT_DESIGN_PARAMS",
    "DEFAULT_DESIGN_VECTOR",
    "DEFAULT_DECISION_VECTOR",
    "OPERATING_LIMITS",
    "CLEARANCE_LIMITS",
    "HEIGHT_LIMITS",
    "DesignVector",
    "GeometryMetrics",
    "compute_geometry_metrics",
    "DEFAULT_TX_CURRENT_A",
    "DEFAULT_RX_CURRENT_A",
    "decision_vector_to_design",
]


class SchemaValidationError(ValueError):
    """Raised when design parameters fall outside the supported domain."""


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Metadata describing how an integer search variable maps to physics."""

    name: str
    raw_min: int
    raw_max: int
    raw_step: int
    scale: float
    offset: float = 0.0
    unit: str | None = None
    description: str = ""

    def decode(self, raw_value: float) -> float:
        """Integer value → physical units."""
        return raw_value * self.scale + self.offset

    def encode(self, physical_value: float) -> float:
        """Physical value → integer space (not rounded)."""
        return (physical_value - self.offset) / self.scale

    @property
    def physical_min(self) -> float:
        return self.decode(self.raw_min)

    @property
    def physical_max(self) -> float:
        return self.decode(self.raw_max)

    def validate_physical(self, value: float) -> float:
        """Return the quantized raw value if ``value`` sits on the allowed grid."""
        raw_value = self.encode(value)
        if not math.isfinite(raw_value):
            raise SchemaValidationError(f"{self.name} must be finite (got {value}).")

        min_allowed = self.raw_min - 1e-9
        max_allowed = self.raw_max + 1e-9
        if raw_value < min_allowed or raw_value > max_allowed:
            raise SchemaValidationError(
                f"{self.name}={value} out of bounds "
                f"[{self.physical_min}, {self.physical_max}]."
            )

        if self.raw_step > 0:
            steps_from_min = (raw_value - self.raw_min) / self.raw_step
            quantized = self.raw_min + round(steps_from_min) * self.raw_step
            if not math.isclose(raw_value, quantized, abs_tol=1e-9):
                raise SchemaValidationError(
                    f"{self.name}={value} violates step size of {self.scale * self.raw_step}."
                )
            raw_value = quantized
        return raw_value


DESIGN_FIELD_ORDER: tuple[str, ...] = (
    "freq",
    "input_voltage",
    "w1",
    "l1_leg",
    "l1_top",
    "l2",
    "h1",
    "l1_center",
    "Tx_turns",
    "Tx_space_x",
    "Tx_space_y",
    "Tx_preg",
    "Tx_width",
    "Tx_height",
    "Rx_width",
    "Rx_height",
    "Rx_space_x",
    "Rx_space_y",
    "Rx_preg",
    "g2",
    "Tx_layer_space_x",
    "Tx_layer_space_y",
)

_VARIABLE_SPEC_DATA: Mapping[str, Mapping[str, object]] = {
    "freq": dict(
        raw_min=1,
        raw_max=80,
        raw_step=1,
        scale=5.0,
        offset=0.0,
        unit="kHz",
        description="Switching frequency (freq_raw * 5 kHz).",
    ),
    "input_voltage": dict(
        raw_min=1,
        raw_max=8,
        raw_step=1,
        scale=50.0,
        offset=100.0,
        unit="Vrms",
        description="Input voltage (raw * 50 + 100).",
    ),
    "w1": dict(
        raw_min=20,
        raw_max=200,
        raw_step=1,
        scale=1.0,
        unit="mm",
        description="Window width along Y.",
    ),
    "l1_leg": dict(
        raw_min=20,
        raw_max=150,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Outer-leg length per side of the core frame.",
    ),
    "l1_top": dict(
        raw_min=5,
        raw_max=20,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Upper yoke thickness.",
    ),
    "l2": dict(
        raw_min=50,
        raw_max=300,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Effective horizontal window length.",
    ),
    "h1": dict(
        raw_min=10,
        raw_max=300,
        raw_step=1,
        scale=0.01,
        unit="mm",
        description="Window height available to copper + gaps.",
    ),
    "l1_center": dict(
        raw_min=1,
        raw_max=25,
        raw_step=1,
        scale=1.0,
        unit="mm",
        description="Center-leg width.",
    ),
    "Tx_turns": dict(
        raw_min=2,
        raw_max=20,
        raw_step=1,
        scale=1.0,
        unit="turns",
        description="Primary winding turn count.",
    ),
    "Tx_space_x": dict(
        raw_min=1,
        raw_max=50,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Primary horizontal clearance to Rx stack.",
    ),
    "Tx_space_y": dict(
        raw_min=1,
        raw_max=50,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Primary vertical spacing between stack layers.",
    ),
    "Tx_preg": dict(
        raw_min=5,
        raw_max=30,
        raw_step=1,
        scale=0.01,
        unit="mm",
        description="Primary prepreg thickness (dielectric).",
    ),
    "Tx_width": dict(
        raw_min=5,
        raw_max=30,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Primary trace width.",
    ),
    "Tx_height": dict(
        raw_min=1,
        raw_max=5,
        raw_step=1,
        scale=0.035,
        unit="mm",
        description="Primary copper thickness.",
    ),
    "Rx_width": dict(
        raw_min=4,
        raw_max=20,
        raw_step=1,
        scale=1.0,
        unit="mm",
        description="Secondary trace width.",
    ),
    "Rx_height": dict(
        raw_min=1,
        raw_max=5,
        raw_step=1,
        scale=0.035,
        unit="mm",
        description="Secondary copper thickness.",
    ),
    "Rx_space_x": dict(
        raw_min=1,
        raw_max=50,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Secondary horizontal gap to Tx stack.",
    ),
    "Rx_space_y": dict(
        raw_min=1,
        raw_max=50,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Secondary inter-layer spacing.",
    ),
    "Rx_preg": dict(
        raw_min=5,
        raw_max=30,
        raw_step=1,
        scale=0.01,
        unit="mm",
        description="Secondary prepreg thickness.",
    ),
    "g2": dict(
        raw_min=10,
        raw_max=300,
        raw_step=1,
        scale=0.01,
        unit="mm",
        description="Vertical air gap below the window.",
    ),
    "Tx_layer_space_x": dict(
        raw_min=2,
        raw_max=50,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Horizontal spacing between Tx layers.",
    ),
    "Tx_layer_space_y": dict(
        raw_min=2,
        raw_max=50,
        raw_step=1,
        scale=0.1,
        unit="mm",
        description="Vertical spacing between Tx layers.",
    ),
}

VARIABLE_SPECS: dict[str, VariableSpec] = {
    name: VariableSpec(name=name, **_VARIABLE_SPEC_DATA[name]) for name in DESIGN_FIELD_ORDER
}

DEFAULT_DESIGN_PARAMS: Mapping[str, float | int] = {
    "freq": 130.0,
    "input_voltage": 400.0,
    "w1": 80.0,
    "l1_leg": 10.0,
    "l1_top": 1.1,
    "l2": 26.0,
    "h1": 2.4,
    "l1_center": 15.0,
    "Tx_turns": 12,
    "Tx_space_x": 2.5,
    "Tx_space_y": 2.0,
    "Tx_preg": 0.2,
    "Tx_width": 1.0,
    "Tx_height": 0.07,
    "Rx_width": 12.0,
    "Rx_height": 0.07,
    "Rx_space_x": 1.0,
    "Rx_space_y": 0.5,
    "Rx_preg": 0.15,
    "g2": 1.5,
    "Tx_layer_space_x": 2.5,
    "Tx_layer_space_y": 0.8,
}

OPERATING_LIMITS: Mapping[str, tuple[float, float]] = {
    "freq": (100.0, 150.0),
    "input_voltage": (380.0, 420.0),
}

CLEARANCE_LIMITS: Mapping[str, float] = {
    "max_length": 1e-5,
    "coil_cavity": 1e-5,
    "gap": 1e-5,
    "tx_rx_spacing": 4e-4,
}

HEIGHT_LIMITS: tuple[float, float] = (4.5, 5.0)

DEFAULT_TX_CURRENT_A = 4.0
DEFAULT_RX_CURRENT_A = 11.0


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Convenience container mirroring ``compute_area`` from the legacy notebook."""

    area: float
    height: float
    max_length: float
    tx_rx_spacing: float
    coil_cavity: float
    gap: float
    volume: float
    tx_span: float
    rx_span: float


def compute_geometry_metrics(design: DesignVector) -> GeometryMetrics:
    """Replicate the notebook geometry calculations using physical units."""
    tx_span = design.Tx_width * 5 + design.Tx_layer_space_y * 4 + design.Tx_space_y
    rx_span = design.Rx_width + design.Rx_space_y
    area = (design.l1_leg + design.l2 + design.l1_center / 2) * 2 * (
        design.w1
        + design.Tx_layer_space_x * 4
        + design.Tx_width * 5
        + design.Tx_space_x
        + design.Rx_space_x
        + design.Rx_width
    )
    height = design.l1_top * 2 + design.h1
    volume = area * height
    max_length = design.l2 - (tx_span + rx_span)
    coil_height = (
        design.Tx_height * 2
        + design.Tx_preg * 2
        + design.Rx_height * 2
        + design.Rx_preg * 2
    )
    coil_cavity = design.h1 - coil_height
    gap = design.h1 - design.g2
    tx_rx_spacing = design.Tx_preg / 2 + design.Rx_preg
    return GeometryMetrics(
        area=area,
        height=height,
        max_length=max_length,
        tx_rx_spacing=tx_rx_spacing,
        coil_cavity=coil_cavity,
        gap=gap,
        volume=volume,
        tx_span=tx_span,
        rx_span=rx_span,
    )


@dataclass(frozen=True, slots=True)
class DesignVector:
    """
    Physical representation of the 22-variable PCB-to-PCB design vector.

    The field names intentionally match the original dataset/LightGBM feature
    set to minimize translation overhead.
    """

    freq: float
    input_voltage: float
    w1: float
    l1_leg: float
    l1_top: float
    l2: float
    h1: float
    l1_center: float
    Tx_turns: int
    Tx_space_x: float
    Tx_space_y: float
    Tx_preg: float
    Tx_width: float
    Tx_height: float
    Rx_width: float
    Rx_height: float
    Rx_space_x: float
    Rx_space_y: float
    Rx_preg: float
    g2: float
    Tx_layer_space_x: float
    Tx_layer_space_y: float
    tx_current: float | None = None
    rx_current_optimetric: float | None = None
    label: str | None = None
    _identifier: str = field(init=False, repr=False)
    enforce_limits: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.enforce_limits:
            self.validate()
        object.__setattr__(self, "_identifier", self.label or self._build_identifier())

    @classmethod
    def from_raw(cls, raw_values: Sequence[float | int]) -> DesignVector:
        if len(raw_values) != len(DESIGN_FIELD_ORDER):
            raise SchemaValidationError(
                f"Expected {len(DESIGN_FIELD_ORDER)} raw values, received {len(raw_values)}."
            )
        decoded = {
            name: VARIABLE_SPECS[name].decode(value)
            for name, value in zip(DESIGN_FIELD_ORDER, raw_values)
        }
        return cls(**decoded)  # type: ignore[arg-type]

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, float | int], *, strict: bool = True
    ) -> DesignVector:
        missing = [name for name in DESIGN_FIELD_ORDER if name not in data]
        if missing:
            raise SchemaValidationError(f"Missing fields for design vector: {missing}")
        ordered = {name: data[name] for name in DESIGN_FIELD_ORDER}
        extras: dict[str, float | str] = {}
        tx_key = "Tx_current"
        rx_key = "Rx_current_optimetric"
        if tx_key in data:
            value = data[tx_key]
            if value is not None:
                fv = float(value)
                if not math.isnan(fv):
                    extras["tx_current"] = fv
        if rx_key in data:
            value = data[rx_key]
            if value is not None:
                fv = float(value)
                if not math.isnan(fv):
                    extras["rx_current_optimetric"] = fv
        label = data.get("spec_id") or data.get("identifier")
        if label is not None:
            extras["label"] = str(label)
        extras["enforce_limits"] = strict
        return cls(**ordered, **extras)  # type: ignore[arg-type]

    def to_raw(self) -> tuple[int, ...]:
        """Return the canonical integer vector consumed by NSGA-II."""
        ints: list[int] = []
        for name in DESIGN_FIELD_ORDER:
            spec = VARIABLE_SPECS[name]
            value = getattr(self, name)
            raw = spec.validate_physical(value)
            ints.append(int(round(raw)))
        return tuple(ints)

    def validate(self) -> None:
        """Run all per-variable and derived-geometry checks."""
        for name in DESIGN_FIELD_ORDER:
            spec = VARIABLE_SPECS[name]
            spec.validate_physical(getattr(self, name))

        for name, (lower, upper) in OPERATING_LIMITS.items():
            value = getattr(self, name)
            if not (lower <= value <= upper):
                raise SchemaValidationError(
                    f"{name}={value} outside approved operating window [{lower}, {upper}]."
                )

        geom = self.geometry
        for metric, minimum in CLEARANCE_LIMITS.items():
            value = getattr(geom, metric)
            if value < minimum:
                raise SchemaValidationError(
                    f"{metric}={value} below minimum clearance of {minimum}."
                )

        if not (HEIGHT_LIMITS[0] <= geom.height <= HEIGHT_LIMITS[1]):
            raise SchemaValidationError(
                f"height={geom.height} outside [{HEIGHT_LIMITS[0]}, {HEIGHT_LIMITS[1]}]."
            )

    @property
    def geometry(self) -> GeometryMetrics:
        return compute_geometry_metrics(self)

    def with_updates(self, **updates: float | int | str | None) -> DesignVector:
        """Produce a new instance with selected fields replaced."""
        allowed = set(DESIGN_FIELD_ORDER) | {
            "tx_current",
            "rx_current_optimetric",
            "label",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise SchemaValidationError(f"Unknown fields for update: {sorted(unknown)}")
        return replace(self, **updates)

    def as_feature_row(
        self,
        tx_current: float | None = None,
        rx_current: float | None = None,
    ) -> list[float]:
        """Return features in the order expected by the LightGBM models."""
        tx_value = (
            float(tx_current)
            if tx_current is not None
            else self.effective_tx_current()
        )
        rx_value = (
            float(rx_current)
            if rx_current is not None
            else self.effective_rx_current()
        )
        base = [
            self.freq,
            self.w1,
            self.l1_leg,
            self.l1_top,
            self.l2,
            self.h1,
            self.l1_center,
            float(self.Tx_turns),
            self.Tx_width,
            self.Tx_height,
            self.Tx_space_x,
            self.Tx_space_y,
            self.Tx_preg,
            self.Rx_width,
            self.Rx_height,
            self.Rx_space_x,
            self.Rx_space_y,
            self.Rx_preg,
            self.g2,
            self.Tx_layer_space_x,
            self.Tx_layer_space_y,
        ]
        base.append(tx_value)
        base.append(rx_value)
        return base

    def effective_tx_current(self, fallback: float = DEFAULT_TX_CURRENT_A) -> float:
        return float(self.tx_current if self.tx_current is not None else fallback)

    def effective_rx_current(self, fallback: float = DEFAULT_RX_CURRENT_A) -> float:
        return float(
            self.rx_current_optimetric
            if self.rx_current_optimetric is not None
            else fallback
        )

    @property
    def identifier(self) -> str:
        return self._identifier

    def to_feature_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in DESIGN_FIELD_ORDER)

    def to_parameters(self) -> dict[str, float | int | str | None]:
        params = {name: getattr(self, name) for name in DESIGN_FIELD_ORDER}
        params["Tx_current"] = self.tx_current
        params["Rx_current_optimetric"] = self.rx_current_optimetric
        params["spec_id"] = self.identifier
        return params

    def _build_identifier(self) -> str:
        payload_parts = [
            f"{name}:{getattr(self, name)}" for name in DESIGN_FIELD_ORDER
        ]
        if self.tx_current is not None:
            payload_parts.append(f"Tx_current:{self.tx_current}")
        if self.rx_current_optimetric is not None:
            payload_parts.append(
                f"Rx_current_optimetric:{self.rx_current_optimetric}"
            )
        payload = "|".join(payload_parts)
        digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()
        return f"pcbpcb-{digest}"


DEFAULT_DESIGN_VECTOR = DesignVector.from_mapping(DEFAULT_DESIGN_PARAMS)
DEFAULT_DECISION_VECTOR = DEFAULT_DESIGN_VECTOR.to_raw()


def decision_vector_to_design(
    decision_vector: Sequence[int | float],
) -> DesignVector:
    """Decode a discrete decision vector into a validated :class:`DesignVector`."""
    return DesignVector.from_raw(decision_vector)
