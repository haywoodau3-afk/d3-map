from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .features import PocketFieldResult

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ProbabilityFieldV2Result:
    features: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


@dataclass(frozen=True)
class ProbeAccessibilityV2Result:
    features: dict[str, Any]
    arrays: dict[str, NDArray[Any]]


def _normalized_weights(weights: FloatArray, size: int) -> FloatArray:
    values = np.asarray(weights, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValidationError("weights must be one finite value per geometry")
    if np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValidationError("weights must be non-negative with a positive sum")
    return np.asarray(values / values.sum(), dtype=float)


def persistence_curve(
    probability: FloatArray,
    domain: BoolArray,
    voxel_volume: float,
    *,
    thresholds: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return accessibility-persistence thresholds, volumes, and domain fractions."""
    if voxel_volume <= 0.0 or not np.isfinite(voxel_volume):
        raise ValidationError("voxel volume must be positive and finite")
    q = np.linspace(0.0, 1.0, 21) if thresholds is None else np.asarray(thresholds, dtype=float)
    if q.ndim != 1 or q.size == 0 or np.any((q < 0.0) | (q > 1.0)):
        raise ValidationError("persistence thresholds must be a non-empty vector in [0, 1]")
    p_open = 1.0 - np.asarray(probability, dtype=float)[domain]
    counts = np.asarray([np.count_nonzero(p_open >= value) for value in q], dtype=float)
    return q, counts * voxel_volume, counts / float(np.count_nonzero(domain))


def _morphology(mask: BoolArray, axis: FloatArray, voxel_volume: float) -> dict[str, Any]:
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "volume_angstrom3": 0.0,
            "signed_halfspace_volume_angstrom3": {
                name: 0.0 for name in ("+x", "-x", "+y", "-y", "+z", "-z")
            },
            "extent_angstrom": None,
            "centroid_angstrom": None,
            "principal_variances_angstrom2": None,
            "principal_axes": None,
            "anisotropy": None,
        }
    coordinates = np.column_stack((xx[mask], yy[mask], zz[mask]))
    centroid = coordinates.mean(axis=0)
    centered = coordinates - centroid
    covariance = centered.T @ centered / float(count)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    anisotropy = None if eigenvalues[-1] <= 1e-15 else float(eigenvalues[0] / eigenvalues[-1])
    return {
        "volume_angstrom3": float(count * voxel_volume),
        "signed_halfspace_volume_angstrom3": {
            "+x": float(np.count_nonzero(mask & (xx >= 0.0)) * voxel_volume),
            "-x": float(np.count_nonzero(mask & (xx < 0.0)) * voxel_volume),
            "+y": float(np.count_nonzero(mask & (yy >= 0.0)) * voxel_volume),
            "-y": float(np.count_nonzero(mask & (yy < 0.0)) * voxel_volume),
            "+z": float(np.count_nonzero(mask & (zz >= 0.0)) * voxel_volume),
            "-z": float(np.count_nonzero(mask & (zz < 0.0)) * voxel_volume),
        },
        "extent_angstrom": {
            "x_min": float(coordinates[:, 0].min()),
            "x_max": float(coordinates[:, 0].max()),
            "y_min": float(coordinates[:, 1].min()),
            "y_max": float(coordinates[:, 1].max()),
            "z_min": float(coordinates[:, 2].min()),
            "z_max": float(coordinates[:, 2].max()),
        },
        "centroid_angstrom": centroid.tolist(),
        "principal_variances_angstrom2": eigenvalues.tolist(),
        "principal_axes": eigenvectors.T.tolist(),
        "anisotropy": anisotropy,
    }


def _energy_fields(
    masks: BoolArray,
    weights: FloatArray,
    energies_kcal: FloatArray,
    domain: BoolArray,
    cutoffs_kcal: Sequence[float],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    energies = np.asarray(energies_kcal, dtype=float)
    if energies.shape != (masks.shape[0],) or not np.all(np.isfinite(energies)):
        raise ValidationError("energies must be one finite kcal/mol value per geometry")
    relative = energies - energies.min()
    mean_energy = float(np.dot(weights, relative))
    covariance = np.tensordot(weights * (relative - mean_energy), masks.astype(float), axes=(0, 0))
    covariance[~domain] = np.nan
    occupied_probability = np.tensordot(weights, masks.astype(float), axes=(0, 0))
    occupied_energy_sum = np.tensordot(weights * relative, masks.astype(float), axes=(0, 0))
    open_probability = 1.0 - occupied_probability
    total_energy = float(np.dot(weights, relative))
    open_energy_sum = total_energy - occupied_energy_sum
    contrast = np.full(occupied_probability.shape, np.nan, dtype=float)
    supported = domain & (occupied_probability > 1e-12) & (open_probability > 1e-12)
    contrast[supported] = (
        open_energy_sum[supported] / open_probability[supported]
        - occupied_energy_sum[supported] / occupied_probability[supported]
    )
    cutoffs = np.asarray(tuple(float(value) for value in cutoffs_kcal), dtype=float)
    conditioned = np.full((len(cutoffs), *domain.shape), np.nan, dtype=float)
    support = np.zeros(len(cutoffs), dtype=float)
    for index, cutoff in enumerate(cutoffs):
        selected = relative <= cutoff + 1e-12
        support[index] = float(weights[selected].sum())
        if support[index] > 0.0:
            conditioned[index] = np.tensordot(
                weights[selected] / support[index], masks[selected].astype(float), axes=(0, 0)
            )
            conditioned[index][~domain] = np.nan
    return (
        {
            "availability": "available",
            "energy_unit": "kcal/mol",
            "cutoffs_kcal_per_mol": cutoffs.tolist(),
            "population_support_by_cutoff": support.tolist(),
            "mean_relative_energy_kcal_per_mol": mean_energy,
        },
        {
            "energy_covariance_kcal_per_mol": covariance,
            "conditional_open_minus_occupied_energy_kcal_per_mol": contrast,
            "energy_conditioned_occupation": conditioned,
            "energy_cutoffs_kcal_per_mol": cutoffs,
            "energy_cutoff_population_support": support,
        },
    )


def analyze_probability_field_v2(
    pocket: PocketFieldResult,
    *,
    weights: FloatArray,
    energies_kcal: FloatArray | None = None,
    block_labels: Sequence[str] | None = None,
    energy_cutoffs_kcal: Sequence[float] = (0.0, 1.0, 3.0, 6.0),
    open_threshold: float = 0.1,
    blocked_threshold: float = 0.9,
) -> ProbabilityFieldV2Result:
    """Derive versioned persistence, entropy, morphology, and energy outputs."""
    if not 0.0 <= open_threshold < blocked_threshold <= 1.0:
        raise ValidationError("persistence thresholds must satisfy 0 <= open < blocked <= 1")
    masks = np.asarray(pocket.per_geometry_occupation, dtype=bool)
    population = _normalized_weights(weights, masks.shape[0])
    p = pocket.occupation_probability
    domain = pocket.domain_mask
    variability_status = "not_estimable" if masks.shape[0] == 1 else "estimated"
    thresholds, open_volumes, open_fractions = persistence_curve(
        p, domain, pocket.voxel_volume
    )
    persistent_open = domain & (p <= open_threshold)
    adaptive = domain & (p > open_threshold) & (p < blocked_threshold)
    persistent_excluded = domain & (p >= blocked_threshold)
    entropy_integral = float(np.nansum(pocket.entropy) * pocket.voxel_volume)
    domain_volume = float(np.count_nonzero(domain) * pocket.voxel_volume)
    entropy_value = None if variability_status == "not_estimable" else entropy_integral
    normalized_entropy = (
        None
        if variability_status == "not_estimable"
        else float(entropy_integral / (domain_volume * np.log(2.0)))
    )
    morphology = {
        "persistent_open": _morphology(persistent_open, pocket.axis, pocket.voxel_volume),
        "adaptive": _morphology(adaptive, pocket.axis, pocket.voxel_volume),
        "persistent_excluded": _morphology(
            persistent_excluded, pocket.axis, pocket.voxel_volume
        ),
    }
    features: dict[str, Any] = {
        "schema": "d3map.features.v2",
        "variability_status": variability_status,
        "evidence_status": "exploratory" if masks.shape[0] > 1 else "not_estimable",
        "domain_volume_angstrom3": domain_volume,
        "voxel_spacing_angstrom": pocket.spacing,
        "voxel_volume_angstrom3": pocket.voxel_volume,
        "steric_occupancy_entropy_volume_angstrom3_nat": entropy_value,
        "normalized_pocket_flexibility": normalized_entropy,
        "persistence_thresholds": {
            "persistent_open_maximum_occupation": open_threshold,
            "persistent_excluded_minimum_occupation": blocked_threshold,
        },
        "accessibility_persistence_curve": {
            "minimum_open_probability": thresholds.tolist(),
            "volume_angstrom3": open_volumes.tolist(),
            "domain_fraction": open_fractions.tolist(),
        },
        "classified_morphology": morphology,
        "energy_analysis": {"availability": "unavailable", "reason": "compatible energies absent"},
        "uncertainty": {
            "availability": "unavailable",
            "reason": "replicate or block provenance absent from this field input",
        },
    }
    arrays: dict[str, NDArray[Any]] = {
        "persistence_thresholds": thresholds,
        "open_volume_angstrom3": open_volumes,
        "open_domain_fraction": open_fractions,
        "persistent_open_mask": persistent_open,
        "adaptive_mask": adaptive,
        "persistent_excluded_mask": persistent_excluded,
        "per_geometry_occupation": masks,
    }
    if block_labels is not None:
        labels = np.asarray(tuple(str(label) for label in block_labels), dtype=str)
        if labels.shape != (masks.shape[0],):
            raise ValidationError("block_labels must provide one label per geometry")
        unique_labels = tuple(dict.fromkeys(labels.tolist()))
        if len(unique_labels) >= 3:
            estimates: list[dict[str, float]] = []
            occupation_estimates: list[NDArray[np.float64]] = []
            for label in unique_labels:
                retained = labels != label
                retained_weights = population[retained]
                if not np.any(retained) or float(retained_weights.sum()) <= 0.0:
                    continue
                retained_weights = retained_weights / retained_weights.sum()
                occupation = np.tensordot(
                    retained_weights,
                    masks[retained].astype(float),
                    axes=(0, 0),
                )
                occupation_estimates.append(occupation)
                mixed = domain & (occupation > 0.0) & (occupation < 1.0)
                entropy = np.zeros_like(occupation)
                entropy[mixed] = -(
                    occupation[mixed] * np.log(occupation[mixed])
                    + (1.0 - occupation[mixed]) * np.log(1.0 - occupation[mixed])
                )
                estimates.append(
                    {
                        "steric_occupancy_entropy_volume_angstrom3_nat": float(
                            np.nansum(entropy) * pocket.voxel_volume
                        ),
                        "normalized_pocket_flexibility": float(
                            np.nansum(entropy)
                            * pocket.voxel_volume
                            / (domain_volume * np.log(2.0))
                        ),
                        "persistent_open_volume_angstrom3": float(
                            np.count_nonzero(domain & (occupation <= open_threshold))
                            * pocket.voxel_volume
                        ),
                        "adaptive_volume_angstrom3": float(
                            np.count_nonzero(
                                domain
                                & (occupation > open_threshold)
                                & (occupation < blocked_threshold)
                            )
                            * pocket.voxel_volume
                        ),
                        "persistent_excluded_volume_angstrom3": float(
                            np.count_nonzero(domain & (occupation >= blocked_threshold))
                            * pocket.voxel_volume
                        ),
                    }
                )
            block_count = len(estimates)
            if block_count >= 3:
                primary = {
                    "steric_occupancy_entropy_volume_angstrom3_nat": entropy_integral,
                    "normalized_pocket_flexibility": float(
                        entropy_integral / (domain_volume * np.log(2.0))
                    ),
                    "persistent_open_volume_angstrom3": morphology["persistent_open"][
                        "volume_angstrom3"
                    ],
                    "adaptive_volume_angstrom3": morphology["adaptive"][
                        "volume_angstrom3"
                    ],
                    "persistent_excluded_volume_angstrom3": morphology[
                        "persistent_excluded"
                    ]["volume_angstrom3"],
                }
                scalar_uncertainty: dict[str, dict[str, float]] = {}
                for name, estimate in primary.items():
                    leave_one_out = np.asarray([row[name] for row in estimates], dtype=float)
                    mean_leave_one_out = float(leave_one_out.mean())
                    standard_error = float(
                        np.sqrt(
                            (block_count - 1)
                            / block_count
                            * np.sum((leave_one_out - mean_leave_one_out) ** 2)
                        )
                    )
                    scalar_uncertainty[name] = {
                        "estimate": float(estimate),
                        "jackknife_standard_error": standard_error,
                        "normal_approximation_95_percent_lower": float(
                            estimate - 1.96 * standard_error
                        ),
                        "normal_approximation_95_percent_upper": float(
                            estimate + 1.96 * standard_error
                        ),
                        "leave_one_block_out_minimum": float(leave_one_out.min()),
                        "leave_one_block_out_maximum": float(leave_one_out.max()),
                    }
                occupation_stack = np.stack(occupation_estimates)
                occupation_mean = occupation_stack.mean(axis=0)
                occupation_se = np.sqrt(
                    (block_count - 1)
                    / block_count
                    * np.sum((occupation_stack - occupation_mean) ** 2, axis=0)
                )
                occupation_se[~domain] = np.nan
                arrays["occupation_block_jackknife_standard_error"] = occupation_se
                features["uncertainty"] = {
                    "availability": "available",
                    "method": "leave-one-CREST-origin-block-out jackknife",
                    "block_count": block_count,
                    "block_labels": list(unique_labels),
                    "interpretation": (
                        "sampling-origin sensitivity within one CREST search; "
                        "not an independent-experiment confidence interval"
                    ),
                    "scalar_features": scalar_uncertainty,
                }
        elif block_labels is not None:
            features["uncertainty"] = {
                "availability": "unavailable",
                "reason": "fewer than three distinct retained CREST origin blocks",
                "block_count": len(unique_labels),
                "block_labels": list(unique_labels),
            }
    if energies_kcal is not None:
        energy_features, energy_arrays = _energy_fields(
            masks,
            population,
            energies_kcal,
            domain,
            energy_cutoffs_kcal,
        )
        features["energy_analysis"] = energy_features
        arrays.update(energy_arrays)
    return ProbabilityFieldV2Result(features, arrays)


def connected_free_volume(
    occupied: BoolArray,
    domain: BoolArray,
) -> BoolArray:
    """Return free voxels connected to the declared domain boundary by 6-neighbour paths."""
    free = np.asarray(domain, dtype=bool) & ~np.asarray(occupied, dtype=bool)
    if free.shape != domain.shape or free.ndim != 3:
        raise ValidationError("connected accessibility requires matching three-dimensional masks")
    reachable = np.zeros_like(free)
    boundary = np.zeros_like(free)
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = 0
        high[axis] = -1
        boundary[tuple(low)] = True
        boundary[tuple(high)] = True
    # A curved domain such as a sphere has an interior array boundary. Seed any free
    # voxel adjacent to a non-domain voxel as part of the external surface.
    for axis in range(3):
        for shift in (-1, 1):
            shifted_domain = np.roll(domain, shift, axis=axis)
            boundary |= domain & ~shifted_domain
    seeds = np.argwhere(free & boundary)
    queue: deque[tuple[int, int, int]] = deque(tuple(int(value) for value in row) for row in seeds)
    for seed in queue:
        reachable[seed] = True
    shape = free.shape
    while queue:
        current = queue.popleft()
        for axis in range(3):
            for shift in (-1, 1):
                neighbour = list(current)
                neighbour[axis] += shift
                if not 0 <= neighbour[axis] < shape[axis]:
                    continue
                key = tuple(neighbour)
                if free[key] and not reachable[key]:
                    reachable[key] = True
                    queue.append(key)
    return reachable


def analyze_probe_accessibility_v2(
    *,
    axis: FloatArray,
    domain: BoolArray,
    frame_coordinates: Sequence[FloatArray],
    weights: FloatArray,
    steric_atom_indices: Sequence[int],
    radii: dict[str, float],
    elements: Sequence[str],
    probe_radii: Sequence[float] = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    target_distances: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
) -> ProbeAccessibilityV2Result:
    """Calculate geometry-level point, connected, and six-direction radial accessibility."""
    population = _normalized_weights(np.asarray(weights, dtype=float), len(frame_coordinates))
    probes = np.asarray(tuple(float(value) for value in probe_radii), dtype=float)
    targets = np.asarray(tuple(float(value) for value in target_distances), dtype=float)
    if probes.ndim != 1 or probes.size == 0 or np.any(probes < 0.0):
        raise ValidationError("probe radii must be a non-empty non-negative vector")
    if targets.ndim != 1 or targets.size == 0 or np.any(targets < 0.0):
        raise ValidationError("target distances must be a non-empty non-negative vector")
    indices = np.asarray(tuple(steric_atom_indices), dtype=int)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    radius_grid = np.sqrt(xx * xx + yy * yy + zz * zz)
    voxel_volume = float(abs(axis[1] - axis[0]) ** 3)
    point_volume = np.zeros((len(frame_coordinates), len(probes)), dtype=float)
    connected_volume = np.zeros_like(point_volume)
    target_reached = np.zeros(
        (len(frame_coordinates), len(probes), len(targets)), dtype=bool
    )
    direction_vectors = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=float,
    )
    radial_clear = np.ones(
        (len(frame_coordinates), len(probes), len(targets), len(direction_vectors)), dtype=bool
    )
    for frame_index, coordinates in enumerate(frame_coordinates):
        atom_coordinates = np.asarray(coordinates, dtype=float)[indices]
        base_radii = np.asarray([radii[elements[index]] for index in indices], dtype=float)
        for probe_index, probe in enumerate(probes):
            occupied = np.zeros(domain.shape, dtype=bool)
            inflated = base_radii + probe
            for atom, radius in zip(atom_coordinates, inflated, strict=True):
                occupied |= domain & (
                    (xx - atom[0]) ** 2
                    + (yy - atom[1]) ** 2
                    + (zz - atom[2]) ** 2
                    < radius * radius
                )
            free = domain & ~occupied
            connected = connected_free_volume(occupied, domain)
            point_volume[frame_index, probe_index] = np.count_nonzero(free) * voxel_volume
            connected_volume[frame_index, probe_index] = (
                np.count_nonzero(connected) * voxel_volume
            )
            for target_index, target in enumerate(targets):
                target_reached[frame_index, probe_index, target_index] = bool(
                    np.any(connected & (radius_grid <= target + 1e-12))
                )
                for direction_index, direction in enumerate(direction_vectors):
                    projection = atom_coordinates @ direction
                    perpendicular_squared = np.sum(atom_coordinates * atom_coordinates, axis=1) - (
                        projection * projection
                    )
                    radial_clear[frame_index, probe_index, target_index, direction_index] = not bool(
                        np.any(
                            (projection >= target - 1e-12)
                            & (projection <= float(np.max(np.abs(axis))) + 1e-12)
                            & (perpendicular_squared < inflated * inflated)
                        )
                    )
    point_mean = np.tensordot(population, point_volume, axes=(0, 0))
    connected_mean = np.tensordot(population, connected_volume, axes=(0, 0))
    target_probability = np.tensordot(population, target_reached.astype(float), axes=(0, 0))
    radial_probability = np.tensordot(population, radial_clear.astype(float), axes=(0, 0))
    features = {
        "schema": "d3map.probe-accessibility.v2",
        "probe_radii_angstrom": probes.tolist(),
        "target_distances_angstrom": targets.tolist(),
        "direction_names": ["+x", "-x", "+y", "-y", "+z", "-z"],
        "point_accessible_volume_angstrom3": point_mean.tolist(),
        "connected_accessible_volume_angstrom3": connected_mean.tolist(),
        "connected_target_reach_probability": target_probability.tolist(),
        "radial_clearance_probability": radial_probability.tolist(),
        "evidence_status": "exploratory" if len(frame_coordinates) > 1 else "not_estimable",
    }
    return ProbeAccessibilityV2Result(
        features,
        {
            "probe_radii_angstrom": probes,
            "target_distances_angstrom": targets,
            "point_accessible_volume_per_geometry_angstrom3": point_volume,
            "connected_accessible_volume_per_geometry_angstrom3": connected_volume,
            "connected_target_reached_per_geometry": target_reached,
            "radial_clear_per_geometry": radial_clear,
            "connected_target_reach_probability": target_probability,
            "radial_clearance_probability": radial_probability,
        },
    )


def write_probability_v2_figures(
    output: Path,
    pocket: PocketFieldResult,
    result: ProbabilityFieldV2Result,
) -> tuple[str, ...]:
    """Write fixed-scale v2 figures from numerical fields without altering them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = output / "figures" / "probability-v2"
    target.mkdir(parents=True, exist_ok=True)
    entropy_by_slice = np.nansum(pocket.entropy, axis=(0, 1))
    slice_index = int(np.argmax(entropy_by_slice))
    probability = pocket.occupation_probability[:, :, slice_index]
    entropy = pocket.entropy[:, :, slice_index]
    classification = np.full(probability.shape, np.nan, dtype=float)
    classification[probability <= 0.1] = 0.0
    classification[(probability > 0.1) & (probability < 0.9)] = 0.5
    classification[probability >= 0.9] = 1.0
    energy = result.arrays.get("energy_covariance_kcal_per_mol")
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.1), constrained_layout=True)
    extent = (pocket.axis[0], pocket.axis[-1], pocket.axis[0], pocket.axis[-1])
    panels = (
        (probability, "viridis", 0.0, 1.0, "Mean occupation"),
        (entropy, "magma", 0.0, float(np.log(2.0)), "Occupancy entropy"),
        (classification, "RdYlBu_r", 0.0, 1.0, "Persistent / adaptive"),
    )
    for axis_object, (values, cmap, low, high, title) in zip(axes[:3], panels, strict=True):
        image = axis_object.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=low,
            vmax=high,
            interpolation="nearest",
        )
        axis_object.set(title=title, xlabel="x / Å", ylabel="y / Å", aspect="equal")
        figure.colorbar(image, ax=axis_object, shrink=0.75)
    if energy is None:
        axes[3].axis("off")
        axes[3].text(
            0.5,
            0.5,
            "Energy sensitivity\nunavailable",
            ha="center",
            va="center",
            fontsize=13,
        )
    else:
        energy_slice = np.asarray(energy)[:, :, slice_index]
        finite = np.abs(energy_slice[np.isfinite(energy_slice)])
        limit = float(finite.max()) if finite.size else 1.0
        limit = max(limit, 1e-12)
        image = axes[3].imshow(
            energy_slice.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axes[3].set(title="Energy sensitivity", xlabel="x / Å", ylabel="y / Å", aspect="equal")
        figure.colorbar(image, ax=axes[3], shrink=0.75, label="cov(E,O) / kcal mol⁻¹")
    figure.suptitle(f"Probability-field v2 at z = {pocket.axis[slice_index]:.2f} Å")
    four_layer = target / "four-layer-pocket.png"
    figure.savefig(four_layer, dpi=180)
    plt.close(figure)

    curve = result.features["accessibility_persistence_curve"]
    figure, axis_object = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis_object.plot(
        curve["minimum_open_probability"],
        curve["volume_angstrom3"],
        marker="o",
        markersize=3,
    )
    for landmark in (0.1, 0.5, 0.9):
        axis_object.axvline(landmark, color="#777777", linewidth=0.8, linestyle="--")
    axis_object.set(
        title="Accessibility-persistence curve",
        xlabel="Minimum open probability",
        ylabel="Accessible volume / Å³",
        xlim=(0.0, 1.0),
    )
    persistence = target / "accessibility-persistence-curve.png"
    figure.savefig(persistence, dpi=180)
    plt.close(figure)

    morphology = result.features["classified_morphology"]["adaptive"]
    signed = morphology["signed_halfspace_volume_angstrom3"]
    figure, axis_object = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    names = ("+x", "-x", "+y", "-y", "+z", "-z")
    axis_object.bar(names, [signed[name] for name in names], color="#6a51a3")
    axis_object.set(
        title="Signed adaptive-volume morphology",
        xlabel="Catalytic-frame direction",
        ylabel="Adaptive volume / Å³",
    )
    morphology_path = target / "signed-adaptive-volume.png"
    figure.savefig(morphology_path, dpi=180)
    plt.close(figure)

    summary_path = output / "probability-v2-summary.json"
    summary_path.write_text(
        json.dumps(result.features, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tuple(
        str(path.relative_to(output))
        for path in (four_layer, persistence, morphology_path, summary_path)
    )
