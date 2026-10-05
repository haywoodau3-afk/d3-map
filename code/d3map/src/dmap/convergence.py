from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import analyze_pocket_field
from .models import CatalyticFrame, Ensemble
from .probability_v2 import analyze_probability_field_v2


@dataclass(frozen=True)
class GridConvergencePoint:
    spacing: float
    weighted_vbur_percent: float
    change_from_previous_percent: float | None


@dataclass(frozen=True)
class ProbabilityFieldConvergencePoint:
    spacing: float
    weighted_vbur_percent: float
    entropy_volume_angstrom3_nat: float | None
    adaptive_volume_angstrom3: float
    persistent_open_volume_angstrom3: float
    persistent_excluded_volume_angstrom3: float
    open_curve_l1_change_angstrom3: float | None
    entropy_change_angstrom3_nat: float | None
    adaptive_volume_change_angstrom3: float | None


def occupation_grid_convergence(
    ensemble: Ensemble,
    *,
    frame: CatalyticFrame,
    weights: np.ndarray,
    steric_atom_indices: tuple[int, ...],
    radii: dict[str, float],
    sphere_radius: float,
    spacings: tuple[float, ...] = (0.3, 0.2, 0.15),
) -> tuple[GridConvergencePoint, ...]:
    points = []
    previous: float | None = None
    for spacing in spacings:
        result = analyze_pocket_field(
            ensemble,
            frame=frame,
            weights=weights,
            steric_atom_indices=steric_atom_indices,
            radii=radii,
            sphere_radius=sphere_radius,
            spacing=spacing,
        )
        value = result.weighted_buried_percent
        points.append(
            GridConvergencePoint(spacing, value, None if previous is None else value - previous)
        )
        previous = value
    return tuple(points)


def probability_field_grid_convergence(
    ensemble: Ensemble,
    *,
    frame: CatalyticFrame,
    weights: np.ndarray,
    steric_atom_indices: tuple[int, ...],
    radii: dict[str, float],
    sphere_radius: float,
    spacings: tuple[float, ...] = (0.3, 0.2, 0.15),
) -> tuple[ProbabilityFieldConvergencePoint, ...]:
    """Converge v2 entropy, persistence volumes, and the complete open-volume curve."""
    points: list[ProbabilityFieldConvergencePoint] = []
    previous_curve: np.ndarray | None = None
    previous_entropy: float | None = None
    previous_adaptive: float | None = None
    for spacing in spacings:
        pocket = analyze_pocket_field(
            ensemble,
            frame=frame,
            weights=weights,
            steric_atom_indices=steric_atom_indices,
            radii=radii,
            sphere_radius=sphere_radius,
            spacing=spacing,
        )
        result = analyze_probability_field_v2(pocket, weights=weights)
        entropy = result.features["steric_occupancy_entropy_volume_angstrom3_nat"]
        morphology = result.features["classified_morphology"]
        adaptive = float(morphology["adaptive"]["volume_angstrom3"])
        curve = np.asarray(
            result.features["accessibility_persistence_curve"]["volume_angstrom3"], dtype=float
        )
        points.append(
            ProbabilityFieldConvergencePoint(
                spacing=spacing,
                weighted_vbur_percent=pocket.weighted_buried_percent,
                entropy_volume_angstrom3_nat=entropy,
                adaptive_volume_angstrom3=adaptive,
                persistent_open_volume_angstrom3=float(
                    morphology["persistent_open"]["volume_angstrom3"]
                ),
                persistent_excluded_volume_angstrom3=float(
                    morphology["persistent_excluded"]["volume_angstrom3"]
                ),
                open_curve_l1_change_angstrom3=(
                    None if previous_curve is None else float(np.mean(np.abs(curve - previous_curve)))
                ),
                entropy_change_angstrom3_nat=(
                    None
                    if previous_entropy is None or entropy is None
                    else float(entropy - previous_entropy)
                ),
                adaptive_volume_change_angstrom3=(
                    None if previous_adaptive is None else adaptive - previous_adaptive
                ),
            )
        )
        previous_curve = curve
        previous_entropy = entropy
        previous_adaptive = adaptive
    return tuple(points)
