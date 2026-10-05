from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .models import CatalyticFrame, Ensemble


@dataclass(frozen=True)
class PocketFieldResult:
    axis: NDArray[np.float64]
    occupation_probability: NDArray[np.float64]
    entropy: NDArray[np.float64]
    domain_mask: NDArray[np.bool_]
    per_geometry_occupation: NDArray[np.bool_]
    per_geometry_buried_percent: NDArray[np.float64]
    weighted_buried_percent: float
    integrated_buried_percent: float
    spacing: float
    voxel_volume: float


def analyze_pocket_field(
    ensemble: Ensemble,
    *,
    frame: CatalyticFrame,
    weights: NDArray[np.float64],
    steric_atom_indices: tuple[int, ...],
    radii: dict[str, float],
    sphere_radius: float = 3.5,
    spacing: float = 0.15,
) -> PocketFieldResult:
    """Build the canonical unsmoothed spherical 3-D occupation probability field."""
    if sphere_radius <= 0 or spacing <= 0:
        raise ValidationError("field sphere radius and spacing must be positive")
    population = np.asarray(weights, dtype=float)
    if population.shape != (ensemble.size,) or np.any(population < 0) or population.sum() <= 0:
        raise ValidationError("weights must be one non-negative value per geometry")
    population = population / population.sum()
    count = round(2 * sphere_radius / spacing) + 1
    axis = np.linspace(-sphere_radius, sphere_radius, count)
    actual_spacing = float(axis[1] - axis[0])
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    domain = xx * xx + yy * yy + zz * zz <= sphere_radius * sphere_radius + 1e-12
    occupation = np.zeros(domain.shape, dtype=float)
    per_geometry = []
    per_geometry_occupation = np.zeros((ensemble.size, *domain.shape), dtype=bool)
    indices = np.asarray(steric_atom_indices, dtype=int)
    atom_radii = np.asarray([radii[ensemble.elements[index]] for index in indices])
    for geometry_index, (geometry, weight) in enumerate(
        zip(ensemble.geometries, population, strict=True)
    ):
        occupied = np.zeros(domain.shape, dtype=bool)
        coordinates = frame.transform(geometry.coordinates)[indices]
        for atom, radius in zip(coordinates, atom_radii, strict=True):
            occupied |= domain & (
                (xx - atom[0]) ** 2 + (yy - atom[1]) ** 2 + (zz - atom[2]) ** 2 < radius * radius
            )
        per_geometry.append(100.0 * float(occupied.sum()) / float(domain.sum()))
        per_geometry_occupation[geometry_index] = occupied
        occupation += float(weight) * occupied
    entropy = np.zeros_like(occupation)
    mixed = domain & (occupation > 0) & (occupation < 1)
    entropy[mixed] = -(
        occupation[mixed] * np.log(occupation[mixed])
        + (1 - occupation[mixed]) * np.log(1 - occupation[mixed])
    )
    occupation[~domain] = np.nan
    entropy[~domain] = np.nan
    weighted = float(np.dot(population, per_geometry))
    integrated = 100.0 * float(np.nansum(occupation)) / float(domain.sum())
    return PocketFieldResult(
        axis,
        occupation,
        entropy,
        domain,
        per_geometry_occupation,
        np.asarray(per_geometry),
        weighted,
        integrated,
        actual_spacing,
        actual_spacing**3,
    )


def ensemble_feature_set(
    pocket: PocketFieldResult,
    *,
    weights: NDArray[np.float64],
    frame_coordinates: tuple[NDArray[np.float64], ...],
    steric_atom_indices: tuple[int, ...],
    radii: dict[str, float],
    elements: tuple[str, ...],
    probe_radius: float = 1.2,
) -> dict[str, Any]:
    """Return a frozen v0.1 set of interpretable and ML-ready pocket features."""
    p = pocket.occupation_probability
    domain = pocket.domain_mask
    x = pocket.axis
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    quadrants = (
        (xx >= 0) & (yy >= 0),
        (xx < 0) & (yy >= 0),
        (xx < 0) & (yy < 0),
        (xx >= 0) & (yy < 0),
    )
    quadrant_occ = [float(np.nanmean(p[domain & mask])) for mask in quadrants]
    radii_grid = np.sqrt(xx * xx + yy * yy + zz * zz)
    edges = np.linspace(0, float(abs(x[-1])), 8)
    radial = []
    for low, high in pairwise(edges):
        mask = domain & (radii_grid >= low) & (radii_grid < high)
        radial.append(float(np.nanmean(p[mask])) if np.any(mask) else 0.0)
    persistent_open = float(np.mean(p[domain] <= 0.1))
    persistent_blocked = float(np.mean(p[domain] >= 0.9))
    breathing = 1.0 - persistent_open - persistent_blocked
    entropy_voxel_sum = float(np.nansum(pocket.entropy))
    entropy_integral = entropy_voxel_sum * pocket.voxel_volume
    normalized_flexibility = (
        None
        if len(frame_coordinates) == 1
        else float(entropy_integral / (float(domain.sum()) * pocket.voxel_volume * np.log(2.0)))
    )

    # Probe-sized straight-line clearance along +z at each transverse point.
    xy_shape = p.shape[:2]
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights = normalized_weights / normalized_weights.sum()
    accessible = np.ones(xy_shape, dtype=float)
    legacy_accessible = np.ones(xy_shape, dtype=float)
    for coordinates, weight in zip(frame_coordinates, normalized_weights, strict=True):
        blocked = np.zeros(xy_shape, dtype=bool)
        gx, gy = np.meshgrid(x, x, indexing="ij")
        for index in steric_atom_indices:
            atom = coordinates[index]
            radius = radii[elements[index]] + probe_radius
            if atom[2] >= 0:
                blocked |= (gx - atom[0]) ** 2 + (gy - atom[1]) ** 2 < radius * radius
        accessible -= weight * blocked
        legacy_accessible -= blocked / len(frame_coordinates)
    return {
        "feature_schema": "d3map.features.v2",
        "pocket_weighted_vbur_percent": pocket.weighted_buried_percent,
        "pocket_integrated_vbur_percent": pocket.integrated_buried_percent,
        "persistent_open_fraction": persistent_open,
        "breathing_fraction": breathing,
        "persistent_blocked_fraction": persistent_blocked,
        "variability_status": "not_estimable" if len(frame_coordinates) == 1 else "estimated",
        "pocket_steric_entropy_volume_angstrom3_nat": (
            None if len(frame_coordinates) == 1 else entropy_integral
        ),
        "normalized_pocket_flexibility": normalized_flexibility,
        "legacy_pocket_steric_entropy_voxel_sum": entropy_voxel_sum,
        "pocket_steric_entropy_sum": entropy_voxel_sum,
        "quadrant_mean_occupation": quadrant_occ,
        "quadrant_asymmetry_variance": float(np.var(quadrant_occ)),
        "radial_mean_occupation_7bins": radial,
        "probe_radius_angstrom": probe_radius,
        "approach_accessibility_mean": float(np.mean(accessible)),
        "approach_accessibility_min": float(np.min(accessible)),
        "legacy_uniform_approach_accessibility_mean": float(np.mean(legacy_accessible)),
        "legacy_uniform_approach_accessibility_min": float(np.min(legacy_accessible)),
    }


def angular_fingerprint(
    directions: NDArray[np.float64], probability: NDArray[np.float64], *, bins: int = 12
) -> list[float]:
    """Rotation-fixed latitude/azimuth fingerprint in the confirmed catalytic frame."""
    azimuth = (np.arctan2(directions[:, 1], directions[:, 0]) + 2 * np.pi) % (2 * np.pi)
    labels = np.minimum((azimuth / (2 * np.pi) * bins).astype(int), bins - 1)
    return [float(np.mean(probability[labels == index])) for index in range(bins)]
