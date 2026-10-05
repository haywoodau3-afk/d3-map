from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .models import AxialGrid, AxialResult, AxialSliceSummary, CatalyticFrame, Ensemble


def _axis(start: float, stop: float, spacing: float) -> NDArray[np.float64]:
    count = round((stop - start) / spacing) + 1
    return start + np.arange(count, dtype=float) * spacing


def _normalized_weights(weights: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    values = np.asarray(weights, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValidationError("weights must be one finite value per geometry")
    if np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValidationError("weights must be non-negative with a positive sum")
    return np.asarray(values / values.sum(), dtype=np.float64)


def _largest_region(
    mask: NDArray[np.bool_],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    domain_count: int,
) -> tuple[float, float | None, float | None, float | None]:
    seen = np.zeros_like(mask)
    largest: list[tuple[int, int]] = []
    for row, column in zip(*np.nonzero(mask), strict=True):
        if seen[row, column]:
            continue
        component: list[tuple[int, int]] = []
        stack = [(int(row), int(column))]
        seen[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < mask.shape[0]
                    and 0 <= next_column < mask.shape[1]
                    and mask[next_row, next_column]
                    and not seen[next_row, next_column]
                ):
                    seen[next_row, next_column] = True
                    stack.append((next_row, next_column))
        if len(component) > len(largest):
            largest = component
    if not largest:
        return 0.0, None, None, None
    points = np.asarray([(x[column], y[row]) for row, column in largest], dtype=float)
    covariance = np.cov(points.T) if len(points) > 1 else np.zeros((2, 2))
    eigenvalues = np.linalg.eigvalsh(covariance)
    anisotropy = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-12 else None
    return (
        float(len(largest) / domain_count),
        float(points[:, 0].mean()),
        float(points[:, 1].mean()),
        anisotropy,
    )


def analyze_axial(
    ensemble: Ensemble,
    *,
    frame: CatalyticFrame,
    grid: AxialGrid,
    weights: NDArray[np.float64],
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    open_threshold: float = 0.1,
    blocked_threshold: float = 0.9,
) -> AxialResult:
    """Calculate fixed-corridor axial occupation and entropy through a public seam."""
    normalized_weights = _normalized_weights(weights, ensemble.size)
    atom_indices = tuple(steric_atom_indices)
    if not atom_indices:
        raise ValidationError("steric atom set cannot be empty")
    if not 0.0 <= open_threshold < blocked_threshold <= 1.0:
        raise ValidationError("persistence thresholds must satisfy 0 <= open < blocked <= 1")

    atom_count = len(ensemble.elements)
    if any(index < 0 or index >= atom_count for index in atom_indices):
        raise ValidationError("steric atom index is out of range")
    atom_radii = np.array([radii[ensemble.elements[index]] for index in atom_indices])
    if np.any(atom_radii <= 0.0):
        raise ValidationError("atomic radii must be positive")

    x = _axis(-grid.transverse_radius, grid.transverse_radius, grid.spacing)
    y = _axis(-grid.transverse_radius, grid.transverse_radius, grid.spacing)
    z = _axis(grid.z_min, grid.z_max, grid.spacing)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    corridor_mask = xx * xx + yy * yy <= grid.transverse_radius**2 + 1e-12
    occupation = np.zeros((len(z), len(y), len(x)), dtype=float)

    for geometry, weight in zip(ensemble.geometries, normalized_weights, strict=True):
        coordinates = frame.transform(geometry.coordinates)[np.array(atom_indices)]
        for z_index, z_value in enumerate(z):
            occupied = np.zeros_like(corridor_mask)
            for atom, radius in zip(coordinates, atom_radii, strict=True):
                distance_squared = (
                    (xx - atom[0]) ** 2 + (yy - atom[1]) ** 2 + (z_value - atom[2]) ** 2
                )
                occupied |= distance_squared < radius * radius
            occupation[z_index] += weight * occupied

    occupation[:, ~corridor_mask] = np.nan
    entropy = np.zeros_like(occupation)
    interior = np.broadcast_to(corridor_mask, occupation.shape)
    probabilities = occupation[interior]
    mixed = (probabilities > 0.0) & (probabilities < 1.0)
    entropy_values = np.zeros_like(probabilities)
    entropy_values[mixed] = -(
        probabilities[mixed] * np.log(probabilities[mixed])
        + (1.0 - probabilities[mixed]) * np.log(1.0 - probabilities[mixed])
    )
    entropy[interior] = entropy_values
    entropy[:, ~corridor_mask] = np.nan

    cell_area = grid.spacing**2
    summaries = []
    for z_index, z_value in enumerate(z):
        values = occupation[z_index][corridor_mask]
        entropy_slice = entropy[z_index][corridor_mask]
        open_mask = corridor_mask & (occupation[z_index] <= open_threshold)
        region = _largest_region(open_mask, x, y, int(np.count_nonzero(corridor_mask)))
        quadrant_masks = (
            corridor_mask & (xx >= 0) & (yy >= 0),
            corridor_mask & (xx < 0) & (yy >= 0),
            corridor_mask & (xx < 0) & (yy < 0),
            corridor_mask & (xx >= 0) & (yy < 0),
        )
        quadrant_values = tuple(
            float(np.nanmean(occupation[z_index][mask])) if np.any(mask) else 0.0
            for mask in quadrant_masks
        )
        summaries.append(
            AxialSliceSummary(
                z=float(z_value),
                mean_occupation=float(values.mean()),
                open_fraction=float(np.mean(values <= open_threshold)),
                breathing_fraction=float(
                    np.mean((values > open_threshold) & (values < blocked_threshold))
                ),
                blocked_fraction=float(np.mean(values >= blocked_threshold)),
                mean_entropy=float(entropy_slice.mean()),
                integrated_entropy=float(entropy_slice.sum() * cell_area),
                quadrant_mean_occupation=(
                    quadrant_values[0],
                    quadrant_values[1],
                    quadrant_values[2],
                    quadrant_values[3],
                ),
                largest_open_region_fraction=region[0],
                largest_open_region_centroid_x=region[1],
                largest_open_region_centroid_y=region[2],
                largest_open_region_anisotropy=region[3],
            )
        )

    return AxialResult(x, y, z, occupation, entropy, tuple(summaries))
