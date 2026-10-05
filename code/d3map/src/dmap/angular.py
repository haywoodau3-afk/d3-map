from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .models import CatalyticFrame, Ensemble
from .static import angular_mask, equal_area_directions


@dataclass(frozen=True)
class ShieldingResult:
    directions: NDArray[np.float64]
    shielding_probability: NDArray[np.float64]
    per_geometry_shielding: NDArray[np.bool_]
    per_geometry_g_percent: NDArray[np.float64]
    mean_g_percent: float


def analyze_shielding(
    ensemble: Ensemble,
    *,
    frame: CatalyticFrame,
    weights: NDArray[np.float64],
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    direction_count: int = 40_962,
) -> ShieldingResult:
    """Aggregate per-geometry angular masks into a shielding-probability field."""
    population = np.asarray(weights, dtype=float)
    if population.shape != (ensemble.size,) or np.any(population < 0) or population.sum() <= 0:
        raise ValidationError("weights must be one non-negative value per geometry")
    population = population / population.sum()
    indices = np.asarray(tuple(steric_atom_indices), dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(ensemble.elements)):
        raise ValidationError("steric atom indices must identify at least one valid atom")
    try:
        atom_radii = np.asarray([radii[ensemble.elements[index]] for index in indices])
    except KeyError as error:
        raise ValidationError(f"no radius configured for element: {error.args[0]}") from error

    directions = equal_area_directions(direction_count)
    probability = np.zeros(direction_count, dtype=float)
    per_geometry_shielding = np.zeros((ensemble.size, direction_count), dtype=bool)
    per_geometry = np.zeros(ensemble.size, dtype=float)
    for frame_index, (geometry, weight) in enumerate(
        zip(ensemble.geometries, population, strict=True)
    ):
        coordinates = frame.transform(geometry.coordinates)[indices]
        mask = angular_mask(coordinates, atom_radii, directions)
        per_geometry_shielding[frame_index] = mask
        per_geometry[frame_index] = 100.0 * np.mean(mask)
        probability += weight * mask
    return ShieldingResult(
        directions,
        probability,
        per_geometry_shielding,
        per_geometry,
        float(np.dot(population, per_geometry)),
    )
