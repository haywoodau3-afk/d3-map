from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .chemistry import CoordinationEnvironment, perceive_coordination
from .errors import ValidationError
from .features import PocketFieldResult, analyze_pocket_field
from .models import CatalyticFrame, Ensemble


@dataclass(frozen=True)
class StateFieldDifference:
    difference: NDArray[np.float64]
    positive_fraction: float
    negative_fraction: float


def coordination_distribution(
    ensemble: Ensemble, center_index: int, weights: NDArray[np.float64]
) -> tuple[dict[str, float], tuple[CoordinationEnvironment, ...]]:
    population = np.asarray(weights, dtype=float)
    population = population / population.sum()
    environments = tuple(
        perceive_coordination(geometry, center_index) for geometry in ensemble.geometries
    )
    distribution: dict[str, float] = {}
    for environment, weight in zip(environments, population, strict=True):
        key = f"CN{environment.coordination_number}:{environment.geometry}"
        distribution[key] = distribution.get(key, 0.0) + float(weight)
    return distribution, environments


def coordination_sector_shielding(
    directions: NDArray[np.float64], probability: NDArray[np.float64]
) -> dict[str, float]:
    """Partition the sphere into six signed catalytic-frame sectors."""
    labels = np.argmax(np.abs(directions), axis=1)
    signs = np.sign(np.take_along_axis(directions, labels[:, None], axis=1)[:, 0])
    names = ("x", "y", "z")
    return {
        f"{'+' if sign > 0 else '-'}{names[axis]}": float(
            np.mean(probability[(labels == axis) & (signs == sign)])
        )
        for axis in range(3)
        for sign in (-1, 1)
    }


def ligand_resolved_fields(
    ensemble: Ensemble,
    *,
    frame: CatalyticFrame,
    weights: NDArray[np.float64],
    ligand_atom_indices: dict[str, tuple[int, ...]],
    radii: dict[str, float],
    sphere_radius: float,
    spacing: float,
) -> dict[str, PocketFieldResult]:
    return {
        label: analyze_pocket_field(
            ensemble,
            frame=frame,
            weights=weights,
            steric_atom_indices=indices,
            radii=radii,
            sphere_radius=sphere_radius,
            spacing=spacing,
        )
        for label, indices in ligand_atom_indices.items()
    }


def coupled_ligand_motion(
    frame_coordinates: tuple[NDArray[np.float64], ...],
    ligand_atom_indices: dict[str, tuple[int, ...]],
    weights: NDArray[np.float64],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Weighted correlations between ligand-centroid displacement amplitudes."""
    labels = tuple(ligand_atom_indices)
    if len(labels) < 2:
        raise ValidationError("coupled motion requires at least two ligand groups")
    population = np.asarray(weights, dtype=float)
    population /= population.sum()
    signals = []
    for label in labels:
        indices = np.asarray(ligand_atom_indices[label], dtype=int)
        centroids = np.asarray(
            [coordinates[indices].mean(axis=0) for coordinates in frame_coordinates]
        )
        reference = np.average(centroids, axis=0, weights=population)
        signals.append(np.linalg.norm(centroids - reference, axis=1))
    values = np.asarray(signals)
    centered = values - np.average(values, axis=1, weights=population)[:, None]
    covariance = (centered * population) @ centered.T
    scale = np.sqrt(np.diag(covariance))
    denominator = np.outer(scale, scale)
    correlation = np.divide(covariance, denominator, out=np.eye(len(labels)), where=denominator > 0)
    return labels, correlation


def compare_state_fields(
    first: PocketFieldResult, second: PocketFieldResult
) -> StateFieldDifference:
    if first.occupation_probability.shape != second.occupation_probability.shape or not np.allclose(
        first.axis, second.axis
    ):
        raise ValidationError("state fields must share the same grid and profile")
    difference = second.occupation_probability - first.occupation_probability
    domain = first.domain_mask & second.domain_mask
    return StateFieldDifference(
        difference,
        float(np.mean(difference[domain] > 0)),
        float(np.mean(difference[domain] < 0)),
    )
