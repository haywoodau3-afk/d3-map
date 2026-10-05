from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError

EnergyKind = Literal["free_energy", "electronic_energy"]
EnergyUnit = Literal["kcal/mol", "kJ/mol", "hartree"]

_KCAL_PER_UNIT: dict[str, float] = {
    "kcal/mol": 1.0,
    "kJ/mol": 1.0 / 4.184,
    "hartree": 627.5094740631,
}
_R_KCAL_PER_MOL_K = 0.00198720425864083


@dataclass(frozen=True)
class PopulationWeights:
    weights: NDArray[np.float64]
    temperature: float
    energy_kind: EnergyKind
    unit: EnergyUnit


def boltzmann_weights(
    energies: NDArray[np.float64],
    *,
    temperature: float,
    energy_kind: EnergyKind,
    unit: EnergyUnit = "kcal/mol",
    degeneracies: NDArray[np.float64] | None = None,
) -> PopulationWeights:
    """Return stable energy-derived weights with explicit thermodynamic semantics."""
    values = np.asarray(energies, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValidationError("energies must be a non-empty finite one-dimensional array")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValidationError("temperature must be a positive finite value")
    if energy_kind not in ("free_energy", "electronic_energy"):
        raise ValidationError("energy_kind must be free_energy or electronic_energy")
    if unit not in _KCAL_PER_UNIT:
        raise ValidationError(f"unsupported energy unit: {unit}")

    if degeneracies is None:
        factors = np.ones_like(values)
    else:
        factors = np.asarray(degeneracies, dtype=float)
        if (
            factors.shape != values.shape
            or np.any(factors <= 0.0)
            or not np.all(np.isfinite(factors))
        ):
            raise ValidationError("degeneracies must be one positive finite value per energy")

    relative_kcal = (values - values.min()) * _KCAL_PER_UNIT[unit]
    log_populations = np.log(factors) - relative_kcal / (_R_KCAL_PER_MOL_K * temperature)
    log_populations -= log_populations.max()
    populations = np.exp(log_populations)
    populations /= populations.sum()
    return PopulationWeights(populations, temperature, energy_kind, unit)
