from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .angular import ShieldingResult
from .features import PocketFieldResult
from .ml import write_ml_spatial_tables
from .models import AxialResult, CatalyticFrame, Ensemble
from .static import PROJECTION_DIRECTIONS, _projection_basis

PLANE_AXES = {
    "+z": ("+x", "+y"),
    "-z": ("+x", "-y"),
    "+x": ("+y", "+z"),
    "-x": ("+y", "-z"),
    "+y": ("+z", "+x"),
    "-y": ("+z", "-x"),
}

LAYER_SPECS = {
    "contact_probability": ("Contact probability", "viridis", "probability"),
    "first_contact_median": ("Conditional median first contact", "RdYlBu_r", "depth"),
    "first_contact_interval_10_90": ("Conditional 10–90% interval", "cividis", "span"),
    "occupied_depth": ("Population-weighted occupied depth", "magma", "span"),
}


def _topographic_geometry_fields(
    coordinates: np.ndarray,
    atom_radii: np.ndarray,
    offsets: np.ndarray,
    sphere_radius: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuous first-contact and exact occupied-union depth fields."""
    if spacing <= 0:
        raise ValueError("topographic spacing must be positive")
    count = round(2 * sphere_radius / spacing) + 1
    axis = np.linspace(-sphere_radius, sphere_radius, count)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    sphere_term = sphere_radius**2 - xx**2 - yy**2
    inside = sphere_term >= 0
    sphere_cap = np.sqrt(np.maximum(sphere_term, 0))
    lower_intervals = []
    upper_intervals = []
    for atom, radius in zip(coordinates, atom_radii, strict=True):
        atom_term = radius**2 - (xx - atom[0]) ** 2 - (yy - atom[1]) ** 2
        atom_inside = atom_term >= 0
        atom_cap = np.sqrt(np.maximum(atom_term, 0))
        lower_intervals.append(np.where(atom_inside, atom[2] - atom_cap, np.inf))
        upper_intervals.append(np.where(atom_inside, atom[2] + atom_cap, -np.inf))

    lowers = np.stack(lower_intervals)
    uppers = np.stack(upper_intervals)
    order = np.argsort(lowers, axis=0)
    lowers = np.take_along_axis(lowers, order, axis=0)
    uppers = np.take_along_axis(uppers, order, axis=0)

    offset_grid = np.asarray(offsets, dtype=float)[:, None, None]
    window_lower = offset_grid - sphere_cap[None, :, :]
    window_upper = offset_grid + sphere_cap[None, :, :]
    union_end = window_lower.copy()
    occupied_depth = np.zeros_like(window_lower)
    top_absolute = np.full_like(window_lower, -np.inf)
    for lower, upper in zip(lowers, uppers, strict=True):
        clipped_lower = np.maximum(lower[None, :, :], window_lower)
        clipped_upper = np.minimum(upper[None, :, :], window_upper)
        valid = inside[None, :, :] & (clipped_lower <= clipped_upper)
        new_length = np.maximum(0.0, clipped_upper - np.maximum(clipped_lower, union_end))
        occupied_depth += np.where(valid, new_length, 0.0)
        union_end = np.where(valid, np.maximum(union_end, clipped_upper), union_end)
        top_absolute = np.where(valid, np.maximum(top_absolute, clipped_upper), top_absolute)

    first_contact = top_absolute - offset_grid
    first_contact[~np.isfinite(top_absolute)] = np.nan
    occupied_depth[:, ~inside] = np.nan
    return axis, first_contact, occupied_depth


def _weighted_conditional_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> np.ndarray:
    """Calculate a discrete weighted quantile along the geometry axis."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    present = np.isfinite(values)
    sortable = np.where(present, values, np.inf)
    order = np.argsort(sortable, axis=1)
    sorted_values = np.take_along_axis(sortable, order, axis=1)
    geometry_weights = np.broadcast_to(
        np.asarray(weights, dtype=float)[None, :, None, None, None], values.shape
    )
    sorted_weights = np.take_along_axis(np.where(present, geometry_weights, 0.0), order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    total = cumulative[:, -1]
    selected = np.argmax(cumulative >= quantile * total[:, None, :, :, :], axis=1)
    result = np.take_along_axis(sorted_values, selected[:, None, :, :, :], axis=1)[:, 0]
    return np.where(total > 0.0, result, np.nan)


def _ensemble_topographic_fields(
    *,
    ensemble: Ensemble,
    frame: CatalyticFrame,
    weights: np.ndarray,
    steric_atom_indices: tuple[int, ...],
    atom_radii: np.ndarray,
    offsets: np.ndarray,
    sphere_radius: float,
    spacing: float,
) -> dict[str, np.ndarray]:
    first_contact_by_direction = []
    occupied_depth_by_direction = []
    axis: np.ndarray | None = None
    atom_indices = np.asarray(steric_atom_indices, dtype=int)
    for direction in PROJECTION_DIRECTIONS:
        local_frame = CatalyticFrame(frame.origin, _projection_basis(frame, direction))
        direction_first_contact = []
        direction_occupied_depth = []
        for geometry in ensemble.geometries:
            axis, first_contact, occupied_depth = _topographic_geometry_fields(
                local_frame.transform(geometry.coordinates)[atom_indices],
                atom_radii,
                offsets,
                sphere_radius,
                spacing,
            )
            direction_first_contact.append(first_contact)
            direction_occupied_depth.append(occupied_depth)
        first_contact_by_direction.append(np.stack(direction_first_contact))
        occupied_depth_by_direction.append(np.stack(direction_occupied_depth))
    if axis is None:  # pragma: no cover - Ensemble rejects empty geometry collections
        raise RuntimeError("topographic reporting requires at least one geometry")

    first_contact = np.stack(first_contact_by_direction)
    occupied_depth_per_frame = np.stack(occupied_depth_by_direction)
    present = np.isfinite(first_contact)
    geometry_weights = weights[None, :, None, None, None]
    contact_probability = np.clip(np.sum(geometry_weights * present, axis=1), 0.0, 1.0)
    weighted_surface = np.sum(geometry_weights * np.where(present, first_contact, 0.0), axis=1)
    conditional_mean = np.full_like(contact_probability, np.nan)
    np.divide(
        weighted_surface,
        contact_probability,
        out=conditional_mean,
        where=contact_probability > 0.0,
    )
    q10 = _weighted_conditional_quantile(first_contact, weights, 0.10)
    q50 = _weighted_conditional_quantile(first_contact, weights, 0.50)
    q90 = _weighted_conditional_quantile(first_contact, weights, 0.90)
    occupied_depth = np.sum(
        geometry_weights * np.nan_to_num(occupied_depth_per_frame, nan=0.0), axis=1
    )
    disk = np.isfinite(occupied_depth_per_frame).any(axis=1)
    contact_probability = np.where(disk, contact_probability, np.nan)
    occupied_depth = np.where(disk, occupied_depth, np.nan)
    return {
        "axis": axis,
        "first_contact_per_frame": first_contact,
        "occupied_depth_per_frame": occupied_depth_per_frame,
        "contact_probability": contact_probability,
        "conditional_first_contact": conditional_mean,
        "first_contact_q10": q10,
        "first_contact_q50": q50,
        "first_contact_q90": q90,
        "first_contact_interval_10_90": q90 - q10,
        "occupied_depth": occupied_depth,
    }


def _selected_indices(values: np.ndarray, minimum: float, maximum: float, spacing: float) -> list[int]:
    requested = np.arange(minimum, maximum + spacing * 0.5, spacing)
    return sorted({int(np.argmin(np.abs(values - value))) for value in requested})


def _direction_label(direction: str) -> str:
    return {
        "+x": "x-positive",
        "-x": "x-negative",
        "+y": "y-positive",
        "-y": "y-negative",
        "+z": "z-positive",
        "-z": "z-negative",
    }[direction]


def _save_figure(figure: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _occupation_plot(result: AxialResult, index: int, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.7), constrained_layout=True)
    image = axis.imshow(
        result.occupation[index], origin="lower",
        extent=(result.x[0], result.x[-1], result.y[0], result.y[-1]),
        cmap="magma", vmin=0, vmax=1,
    )
    axis.set(
        title=f"d2-map axial occupation probability at z = {result.z[index]:.1f} Å",
        xlabel="x / Å", ylabel="y / Å", aspect="equal",
    )
    figure.colorbar(image, ax=axis, label="occupation probability")
    _save_figure(figure, path)


def _topographic_scale(kind: str, sphere_radius: float) -> tuple[float, float]:
    return {
        "probability": (0.0, 1.0),
        "depth": (-sphere_radius, sphere_radius),
        "span": (0.0, 2.0 * sphere_radius),
    }[kind]


def _topographic_plot(
    axis: plt.Axes,
    grid_axis: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    direction: str,
    sphere_radius: float,
    layer: str,
    probability: np.ndarray | None = None,
) -> Any:
    _label, cmap_name, scale_kind = LAYER_SPECS[layer]
    minimum, maximum = _topographic_scale(scale_kind, sphere_radius)
    cmap = matplotlib.colormaps[cmap_name].with_extremes(bad="#eeeeee")
    masked = np.ma.masked_invalid(values.T)
    image = axis.imshow(
        masked,
        origin="lower",
        extent=(grid_axis[0], grid_axis[-1], grid_axis[0], grid_axis[-1]),
        cmap=cmap,
        vmin=minimum,
        vmax=maximum,
        interpolation="nearest",
    )
    if probability is not None and layer != "contact_probability":
        finite = probability[np.isfinite(probability)]
        available = [
            level for level in (0.1, 0.5, 0.9)
            if finite.size and float(finite.min()) < level < float(finite.max())
        ]
        if available:
            axis.contour(
                grid_axis,
                grid_axis,
                probability.T,
                levels=available,
                colors="white",
                linewidths=0.7,
            )
    horizontal, vertical = PLANE_AXES[direction]
    axis.set(
        title=title,
        xlabel=f"catalytic {horizontal} / Å",
        ylabel=f"catalytic {vertical} / Å",
        aspect="equal",
    )
    axis.text(
        0.02,
        0.02,
        f"view from {direction} → origin",
        transform=axis.transAxes,
        fontsize=8,
        color="#222222",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    return image


def _layer_values(fields: dict[str, np.ndarray], layer: str) -> np.ndarray:
    return {
        "contact_probability": fields["contact_probability"],
        "first_contact_median": fields["first_contact_q50"],
        "first_contact_interval_10_90": fields["first_contact_interval_10_90"],
        "occupied_depth": fields["occupied_depth"],
    }[layer]


def _render_topographic_outputs(
    output: Path,
    *,
    fields: dict[str, np.ndarray],
    directions: tuple[str, ...],
    offsets: np.ndarray,
    sphere_radius: float,
) -> None:
    figures = output / "figures"
    topographic_directory = figures / "topographic-projections"
    map_axis = fields["axis"]
    probability = fields["contact_probability"]
    layers = tuple(LAYER_SPECS)

    for direction_index, direction in enumerate(directions):
        direction_directory = topographic_directory / _direction_label(direction)
        for offset_index, offset in enumerate(offsets):
            figure, axes = plt.subplots(2, 2, figsize=(12.4, 10.7), constrained_layout=True)
            for axis, layer in zip(axes.flat, layers, strict=True):
                title = LAYER_SPECS[layer][0]
                image = _topographic_plot(
                    axis,
                    map_axis,
                    _layer_values(fields, layer)[direction_index, offset_index],
                    title,
                    direction=direction,
                    sphere_radius=sphere_radius,
                    layer=layer,
                    probability=probability[direction_index, offset_index],
                )
                figure.colorbar(image, ax=axis, label=_layer_colorbar_label(layer))
            figure.suptitle(
                f"Ensemble topographic diagnostic: {direction}, outward offset {offset:.1f} Å"
            )
            name = f"offset-{offset:.1f}".replace(".", "p") + "A.png"
            _save_figure(figure, direction_directory / name)
            if direction == "+z":
                source = direction_directory / name
                alias = topographic_directory / (f"z-{offset:.1f}".replace(".", "p") + "A.png")
                alias.write_bytes(source.read_bytes())

        selected = [
            int(np.argmin(np.abs(offsets - value)))
            for value in np.linspace(float(offsets[0]), float(offsets[-1]), 5)
        ]
        figure, axes = plt.subplots(4, 5, figsize=(20, 15), constrained_layout=True)
        for row, layer in enumerate(layers):
            image = None
            for column, offset_index in enumerate(selected):
                image = _topographic_plot(
                    axes[row, column],
                    map_axis,
                    _layer_values(fields, layer)[direction_index, offset_index],
                    f"{offsets[offset_index]:.1f} Å",
                    direction=direction,
                    sphere_radius=sphere_radius,
                    layer=layer,
                    probability=probability[direction_index, offset_index],
                )
            if image is not None:
                figure.colorbar(
                    image,
                    ax=axes[row, :],
                    label=_layer_colorbar_label(layer),
                    shrink=0.85,
                )
            axes[row, 0].set_ylabel(
                f"{LAYER_SPECS[layer][0]}\n{axes[row, 0].get_ylabel()}"
            )
        figure.suptitle(f"Ensemble topographic diagnostic along {direction}")
        overview_path = direction_directory / "overview-five-distances.png"
        _save_figure(figure, overview_path, dpi=160)
        if direction == "+z":
            (figures / "topographic-projections-five.png").write_bytes(overview_path.read_bytes())

    zero_index = int(np.argmin(np.abs(offsets)))
    figure, axes = plt.subplots(6, 4, figsize=(18, 25), constrained_layout=True)
    for column, layer in enumerate(layers):
        image = None
        for row, direction in enumerate(directions):
            image = _topographic_plot(
                axes[row, column],
                map_axis,
                _layer_values(fields, layer)[row, zero_index],
                LAYER_SPECS[layer][0],
                direction=direction,
                sphere_radius=sphere_radius,
                layer=layer,
                probability=probability[row, zero_index],
            )
            if column == 0:
                axes[row, 0].set_ylabel(f"{direction}\n{axes[row, 0].get_ylabel()}")
        if image is not None:
            figure.colorbar(
                image,
                ax=axes[:, column],
                label=_layer_colorbar_label(layer),
                shrink=0.75,
            )
    figure.suptitle(
        f"Six signed catalytic-frame directions at outward offset {offsets[zero_index]:.1f} Å"
    )
    _save_figure(figure, figures / "topographic-projections-six.png", dpi=160)


def _layer_colorbar_label(layer: str) -> str:
    return {
        "contact_probability": "contact probability",
        "first_contact_median": "conditional first-contact depth / Å",
        "first_contact_interval_10_90": "conditional q90 − q10 / Å",
        "occupied_depth": "population-weighted occupied depth / Å",
    }[layer]


def _topographic_analysis(
    fields: dict[str, np.ndarray],
    directions: tuple[str, ...],
    offsets: np.ndarray,
    sphere_radius: float,
) -> dict[str, Any]:
    rows = []
    for direction_index, direction in enumerate(directions):
        for offset_index, offset in enumerate(offsets):
            probability = fields["contact_probability"][direction_index, offset_index]
            median = fields["first_contact_q50"][direction_index, offset_index]
            interval = fields["first_contact_interval_10_90"][direction_index, offset_index]
            occupied = fields["occupied_depth"][direction_index, offset_index]
            disk = np.isfinite(occupied)
            contacted = np.isfinite(median)
            rows.append(
                {
                    "direction": direction,
                    "offset_angstrom": float(offset),
                    "mean_contact_probability": float(np.mean(probability[disk])),
                    "maximum_contact_probability": float(np.max(probability[disk])),
                    "deepest_median_first_contact_angstrom": (
                        float(np.max(median[contacted])) if np.any(contacted) else None
                    ),
                    "mean_conditional_interval_10_90_angstrom": (
                        float(np.mean(interval[contacted])) if np.any(contacted) else None
                    ),
                    "mean_occupied_depth_angstrom": float(np.mean(occupied[disk])),
                    "maximum_occupied_depth_angstrom": float(np.max(occupied[disk])),
                    "fraction_at_positive_depth_boundary": float(
                        np.mean(median[contacted] >= sphere_radius - 1e-9)
                    ) if np.any(contacted) else 0.0,
                }
            )

    def extreme(key: str, maximum: bool = True) -> dict[str, Any]:
        available = [row for row in rows if row[key] is not None]
        return dict(max(available, key=lambda row: row[key]) if maximum else min(available, key=lambda row: row[key]))

    asymmetry = []
    for positive, negative in (("+x", "-x"), ("+y", "-y"), ("+z", "-z")):
        for offset in offsets:
            first = next(
                row for row in rows
                if row["direction"] == positive and np.isclose(row["offset_angstrom"], offset)
            )
            second = next(
                row for row in rows
                if row["direction"] == negative and np.isclose(row["offset_angstrom"], offset)
            )
            asymmetry.append(
                {
                    "axis": positive[-1],
                    "offset_angstrom": float(offset),
                    "absolute_mean_contact_probability_difference": abs(
                        first["mean_contact_probability"] - second["mean_contact_probability"]
                    ),
                }
            )
    boundary_rows = [row for row in rows if row["fraction_at_positive_depth_boundary"] > 0.0]
    return {
        "schema": "d3map.topographic-analysis.v1",
        "scope": "descriptive steric analysis; no mechanistic or energetic inference",
        "direction_offset_summaries": rows,
        "highlights": {
            "most_blocked": extreme("mean_contact_probability"),
            "least_blocked": extreme("mean_contact_probability", maximum=False),
            "deepest_median_contact": extreme("deepest_median_first_contact_angstrom"),
            "greatest_conditional_variability": extreme(
                "mean_conditional_interval_10_90_angstrom"
            ),
            "greatest_mean_occupied_depth": extreme("mean_occupied_depth_angstrom"),
            "greatest_directional_contact_asymmetry": max(
                asymmetry,
                key=lambda item: item["absolute_mean_contact_probability_difference"],
            ),
        },
        "directional_asymmetry": asymmetry,
        "scale_clipping": {
            "detected": bool(boundary_rows),
            "reason": (
                "first-contact medians reach the analysis-sphere boundary; this is physical "
                "domain truncation, not per-plot colour autoscaling"
                if boundary_rows
                else "no summary field reaches a physical scale boundary"
            ),
            "direction_offset_pairs": [
                {
                    "direction": row["direction"],
                    "offset_angstrom": row["offset_angstrom"],
                    "affected_fraction": row["fraction_at_positive_depth_boundary"],
                }
                for row in boundary_rows
            ],
        },
    }


def _write_topographic_analysis(path: Path, analysis: dict[str, Any]) -> None:
    path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    highlights = analysis["highlights"]
    lines = [
        "# Ensemble topographic analysis",
        "",
        "Descriptive steric analysis only; these findings do not imply a mechanism or energy barrier.",
        "",
    ]
    labels = {
        "most_blocked": "Most blocked",
        "least_blocked": "Least blocked",
        "deepest_median_contact": "Deepest median contact",
        "greatest_conditional_variability": "Greatest conditional variability",
        "greatest_mean_occupied_depth": "Greatest mean occupied depth",
    }
    for key, label in labels.items():
        item = highlights[key]
        lines.append(
            f"- **{label}:** {item['direction']} at {item['offset_angstrom']:.2f} Å outward offset"
        )
    asymmetric = highlights["greatest_directional_contact_asymmetry"]
    lines.append(
        f"- **Greatest signed-direction contact asymmetry:** {asymmetric['axis']} axis at "
        f"{asymmetric['offset_angstrom']:.2f} Å outward offset"
    )
    clipping = analysis["scale_clipping"]
    if clipping["detected"]:
        lines.extend(("", f"> **Boundary warning:** {clipping['reason']}."))
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plot_bundle(
    output: Path,
    *,
    result: AxialResult,
    shielding: ShieldingResult,
    pocket: PocketFieldResult,
    metadata: dict[str, Any],
    ensemble: Ensemble,
    frame: CatalyticFrame,
    weights: np.ndarray,
    steric_atom_indices: tuple[int, ...],
    topographic_radii: dict[str, float],
    displaced_offsets: np.ndarray,
    displaced_vbur: np.ndarray,
    displaced_g: np.ndarray,
    sphere_radius: float,
    topographic_spacing: float,
    reporting: dict[str, Any] | None = None,
) -> None:
    """Generate the complete standard plot and plot-data suite for every project."""
    options = reporting or {}
    figures = output / "figures"
    data = output / "plot-data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    z_min = max(float(result.z.min()), float(options.get("z_min", 0.0)))
    z_max = min(float(result.z.max()), float(options.get("z_max", 2.0)))
    z_spacing = float(options.get("z_spacing", 0.1))
    if z_spacing <= 0:
        raise ValueError("reporting z_spacing must be positive")
    if z_max < z_min:
        z_min = float(result.z.min())
        z_max = min(float(result.z.max()), z_min + 2.0)
    axial_indices = _selected_indices(result.z, z_min, z_max, z_spacing)

    axial_directory = figures / "axial-occupation"
    for index in axial_indices:
        name = f"z-{result.z[index]:.1f}".replace(".", "p") + "A.png"
        _occupation_plot(result, index, axial_directory / name)

    direction_names = PROJECTION_DIRECTIONS
    direction_vectors = np.stack(
        [_projection_basis(frame, direction)[:, 2] for direction in direction_names]
    )
    atom_indices = np.asarray(steric_atom_indices, dtype=int)
    atom_radii = np.asarray(
        [topographic_radii[ensemble.elements[index]] for index in atom_indices]
    )
    topographic_fields = _ensemble_topographic_fields(
        ensemble=ensemble,
        frame=frame,
        weights=weights,
        steric_atom_indices=steric_atom_indices,
        atom_radii=atom_radii,
        offsets=displaced_offsets,
        sphere_radius=sphere_radius,
        spacing=topographic_spacing,
    )
    map_axis = topographic_fields["axis"]
    surfaces = topographic_fields["first_contact_per_frame"]
    present = np.isfinite(surfaces)
    contact_probability = topographic_fields["contact_probability"]
    conditional_surface = topographic_fields["conditional_first_contact"]
    _render_topographic_outputs(
        output,
        fields=topographic_fields,
        directions=direction_names,
        offsets=displaced_offsets,
        sphere_radius=sphere_radius,
    )

    positive_z_index = direction_names.index("+z")
    zero_index = int(np.argmin(np.abs(displaced_offsets)))
    static_surface = surfaces[positive_z_index, 0, zero_index]
    static_probability = present[positive_z_index, 0, zero_index].astype(float)
    static_domain = np.isfinite(
        topographic_fields["occupied_depth"][positive_z_index, zero_index]
    )
    static_probability[~static_domain] = np.nan
    static_limit_error = float(
        np.nanmax(
            np.abs(contact_probability[positive_z_index, zero_index] - static_probability)
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    first = _topographic_plot(
        axes[0], map_axis, static_surface, "Reference geometry (static)",
        direction="+z", sphere_radius=sphere_radius, layer="first_contact_median",
        probability=static_probability,
    )
    second = _topographic_plot(
        axes[1],
        map_axis,
        topographic_fields["first_contact_q50"][positive_z_index, zero_index],
        "Weighted ensemble d2-map",
        direction="+z",
        sphere_radius=sphere_radius,
        layer="first_contact_median",
        probability=contact_probability[positive_z_index, zero_index],
    )
    difference = contact_probability[positive_z_index, zero_index] - static_probability
    delta = axes[2].imshow(
        difference.T,
        origin="lower",
        extent=(map_axis[0], map_axis[-1], map_axis[0], map_axis[-1]),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[2].set(
        title="Occupation-probability change",
        xlabel="catalytic +x / Å",
        ylabel="catalytic +y / Å",
        aspect="equal",
    )
    figure.colorbar(first, ax=axes[0], label="first-contact z / Å")
    figure.colorbar(second, ax=axes[1], label="conditional first-contact z / Å")
    figure.colorbar(delta, ax=axes[2], label="dynamic − reference probability")
    figure.suptitle("Static reference versus dynamic d2-map at z = 0 Å")
    _save_figure(figure, figures / "static-vs-ensemble.png", dpi=200)

    directions = shielding.directions
    longitude = np.arctan2(directions[:, 1], directions[:, 0])
    latitude = np.arcsin(np.clip(directions[:, 2], -1, 1))
    figure = plt.figure(figsize=(10, 5.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="mollweide")
    scatter = axis.scatter(longitude, latitude, c=shielding.shielding_probability, s=5, cmap="viridis", vmin=0, vmax=1)
    axis.grid(alpha=0.3)
    axis.set_title("Ensemble angular shielding probability")
    figure.colorbar(scatter, ax=axis, label="shielding probability", shrink=0.8)
    _save_figure(figure, figures / "angular-shielding-map.png")

    summaries = result.summaries
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(result.z, [item.open_fraction for item in summaries], label="persistent open")
    axes[0, 0].plot(result.z, [item.breathing_fraction for item in summaries], label="breathing")
    axes[0, 0].plot(result.z, [item.blocked_fraction for item in summaries], label="persistent blocked")
    axes[0, 0].set(title="Axial persistence", xlabel="z / Å", ylabel="area fraction")
    axes[0, 0].legend()
    axes[0, 1].plot(result.z, [item.mean_entropy for item in summaries], color="#756bb1")
    axes[0, 1].set(title="Pocket steric flexibility", xlabel="z / Å", ylabel="mean binary entropy")
    axes[1, 0].plot(displaced_offsets, weights @ displaced_vbur, label="dVbur")
    axes[1, 0].plot(displaced_offsets, weights @ displaced_g, label="dG")
    axes[1, 0].set(title="Displaced-centre descriptors", xlabel="translated origin / Å", ylabel="percent")
    axes[1, 0].legend()
    radial = metadata["features"]["radial_mean_occupation_7bins"]
    axes[1, 1].plot(np.arange(1, len(radial) + 1), radial, marker="o")
    axes[1, 1].set(title="Radial occupation profile", xlabel="radial bin", ylabel="mean occupation")
    _save_figure(figure, figures / "dynamic-feature-profiles.png")

    vbur = np.asarray(metadata["ensemble_scalar_descriptors"]["vbur_percent"]["values"])
    g_values = np.asarray(metadata["ensemble_scalar_descriptors"]["g_percent"]["values"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].hist(vbur, bins=min(12, max(1, ensemble.size)), weights=weights, color="#7a0177")
    axes[0].set(title="Vbur distribution", xlabel="Vbur / %", ylabel="population")
    axes[1].hist(g_values, bins=min(12, max(1, ensemble.size)), weights=weights, color="#2b8cbe")
    axes[1].set(title="G distribution", xlabel="G / %", ylabel="population")
    _save_figure(figure, figures / "descriptor-distributions.png")

    selected = [axial_indices[round(value)] for value in np.linspace(0, len(axial_indices) - 1, 5)]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    image = None
    for axis, index in zip(axes.flat[:5], selected, strict=True):
        image = axis.imshow(
            result.occupation[index], origin="lower",
            extent=(result.x[0], result.x[-1], result.y[0], result.y[-1]),
            cmap="magma", vmin=0, vmax=1,
        )
        axis.set(title=f"z = {result.z[index]:.1f} Å", xlabel="x / Å", ylabel="y / Å", aspect="equal")
    axes.flat[5].axis("off")
    if image is not None:
        figure.colorbar(image, ax=axes, label="occupation probability", shrink=0.8)
    figure.suptitle("Axial occupation probability along the reactive direction")
    _save_figure(figure, figures / "axial-occupation-five.png", dpi=200)

    zero_axial = int(np.argmin(np.abs(result.z)))
    probability = result.occupation[zero_axial]
    entropy = result.entropy[zero_axial]
    persistence = np.where(probability >= 0.9, 1.0, np.where(probability <= 0.1, 0.0, 0.5))
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    extent = (result.x[0], result.x[-1], result.y[0], result.y[-1])
    panels = (
        (probability, "magma", 0.0, 1.0, "Occupation probability at z = 0 Å"),
        (entropy, "viridis", 0.0, np.log(2), "Local steric entropy at z = 0 Å"),
        (persistence, "RdYlGn_r", 0.0, 1.0, "Open / breathing / blocked regions"),
    )
    for axis, (values, cmap, minimum, maximum, title) in zip(axes[0], panels, strict=True):
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=minimum,
            vmax=maximum,
        )
        axis.set(title=title, xlabel="x / Å", ylabel="y / Å", aspect="equal")
        figure.colorbar(image, ax=axis, shrink=0.8)
    axes[1, 0].plot(displaced_offsets, weights @ displaced_vbur, label="dVbur")
    axes[1, 0].plot(displaced_offsets, weights @ displaced_g, label="dG")
    axes[1, 0].set(title="Axially displaced descriptors", xlabel="z / Å", ylabel="percent")
    axes[1, 0].legend()
    axes[1, 1].hist(vbur, bins=min(12, max(1, ensemble.size)), weights=weights, alpha=0.7)
    axes[1, 1].hist(g_values, bins=min(12, max(1, ensemble.size)), weights=weights, alpha=0.7)
    axes[1, 1].set(title="Weighted descriptor distributions", xlabel="percent", ylabel="population")
    axes[1, 1].legend(("Vbur", "G"))
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.98,
        f"mode: {metadata['analysis_mode']}\nframes: {ensemble.size}\n"
        f"effective ensemble size: {metadata['effective_ensemble_size']:.2f}\n"
        f"dVbur: {metadata['ensemble_scalar_descriptors']['vbur_percent']['mean']:.2f}%\n"
        f"dG: {metadata['ensemble_scalar_descriptors']['g_percent']['mean']:.2f}%",
        va="top",
        fontsize=13,
    )
    figure.suptitle("d3map feature dashboard")
    _save_figure(figure, figures / "all-features-dashboard.png", dpi=200)

    with (data / "per-geometry-descriptors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "weight", "vbur_percent", "g_percent"))
        for index, (weight, value_vbur, value_g) in enumerate(zip(weights, vbur, g_values, strict=True), start=1):
            writer.writerow((index, weight, value_vbur, value_g))
    with (data / "displaced-scan.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("z_angstrom", "vbur_mean", "vbur_sd", "g_mean", "g_sd"))
        for index, offset in enumerate(displaced_offsets):
            mean_v, mean_g = float(weights @ displaced_vbur[:, index]), float(weights @ displaced_g[:, index])
            sd_v = float(np.sqrt(weights @ (displaced_vbur[:, index] - mean_v) ** 2))
            sd_g = float(np.sqrt(weights @ (displaced_g[:, index] - mean_g) ** 2))
            writer.writerow((offset, mean_v, sd_v, mean_g, sd_g))
    with (data / "angular-shielding.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ux", "uy", "uz", "shielding_probability"))
        for direction, probability in zip(directions, shielding.shielding_probability, strict=True):
            writer.writerow((*direction, probability))
    topographic_payload = {
        "schema": np.asarray("d3map.topographic-fields.v1"),
        "axis": map_axis,
        "direction_names": np.asarray(direction_names),
        "direction_vectors": direction_vectors,
        "plane_horizontal_axes": np.asarray([PLANE_AXES[name][0] for name in direction_names]),
        "plane_vertical_axes": np.asarray([PLANE_AXES[name][1] for name in direction_names]),
        "view_convention": np.asarray("outside-toward-origin; right-handed local frame"),
        "offsets": displaced_offsets,
        "sphere_radius": np.asarray(sphere_radius),
        "scale_probability": np.asarray((0.0, 1.0)),
        "scale_first_contact_angstrom": np.asarray((-sphere_radius, sphere_radius)),
        "scale_span_angstrom": np.asarray((0.0, 2.0 * sphere_radius)),
        "conditional_first_contact": conditional_surface,
        "contact_probability": contact_probability,
        "first_contact_q10": topographic_fields["first_contact_q10"],
        "first_contact_q50": topographic_fields["first_contact_q50"],
        "first_contact_q90": topographic_fields["first_contact_q90"],
        "first_contact_interval_10_90": topographic_fields[
            "first_contact_interval_10_90"
        ],
        "occupied_depth": topographic_fields["occupied_depth"],
        "reference_first_contact": static_surface,
        "reference_presence": static_probability,
        "reference_first_contact_all_directions": surfaces[:, 0, zero_index],
        "reference_presence_all_directions": present[:, 0, zero_index],
    }
    if bool(options.get("retain_per_frame_topographic_fields", False)):
        topographic_payload["first_contact_per_frame"] = surfaces
        topographic_payload["occupied_depth_per_frame"] = topographic_fields[
            "occupied_depth_per_frame"
        ]
    configured_resolutions = options.get("ml_fraction_resolutions")
    fraction_resolutions = tuple(
        int(value)
        for value in (
            configured_resolutions
            if configured_resolutions is not None
            else tuple(value for value in (1, 2, 4, 8) if value <= len(map_axis))
        )
    )
    topographic_payload["ml_fraction_resolutions"] = np.asarray(fraction_resolutions)
    np.savez_compressed(data / "topographic-fields.npz", **topographic_payload)
    write_ml_spatial_tables(
        data,
        fields=topographic_fields,
        directions=direction_names,
        offsets=displaced_offsets,
        sphere_radius=sphere_radius,
        resolutions=fraction_resolutions,
        analysis_id=str(metadata["input_sha256"])[:16],
    )
    analysis = _topographic_analysis(
        topographic_fields, direction_names, displaced_offsets, sphere_radius
    )
    _write_topographic_analysis(data / "topographic-analysis.json", analysis)
    with (data / "axial-summaries.json").open("w", encoding="utf-8") as stream:
        json.dump([asdict(item) for item in summaries], stream, indent=2)
        stream.write("\n")
    plot_manifest = {
        "schema": "d3map.plot-bundle.v1",
        "reporting_range_angstrom": [z_min, z_max, z_spacing],
        "topographic_directions": list(direction_names),
        "static_limit_max_abs_probability_error": static_limit_error,
        "figure_files": sorted(
            str(path.relative_to(output)) for path in figures.rglob("*.png")
        ),
        "data_files": sorted(
            str(path.relative_to(output))
            for path in data.iterdir()
            if path.is_file() and path.name != "plot-manifest.json"
        ),
    }
    (data / "plot-manifest.json").write_text(
        json.dumps(plot_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def regenerate_topographic_outputs(output: Path) -> None:
    """Regenerate topographic figures and descriptive analysis from stored summaries."""
    data = output / "plot-data"
    source = data / "topographic-fields.npz"
    if not source.is_file():
        raise ValueError(f"topographic summary artifact not found: {source}")
    with np.load(source, allow_pickle=False) as stored:
        schema = str(stored["schema"]) if "schema" in stored.files else ""
        if schema not in {"dmap.topographic-fields.v0.3", "d3map.topographic-fields.v1"}:
            raise ValueError(
                "report regeneration requires d3map.topographic-fields.v1; "
                "run d3map analyze once to upgrade this result"
            )
        directions = tuple(str(item) for item in stored["direction_names"])
        offsets = np.asarray(stored["offsets"], dtype=float)
        sphere_radius = float(stored["sphere_radius"])
        resolution_values = (
            stored["ml_fraction_resolutions"]
            if "ml_fraction_resolutions" in stored.files
            else np.asarray((1, 2, 4, 8))
        )
        resolutions = tuple(int(value) for value in resolution_values)
        fields = {
            key: np.asarray(stored[key])
            for key in (
                "axis",
                "contact_probability",
                "conditional_first_contact",
                "first_contact_q10",
                "first_contact_q50",
                "first_contact_q90",
                "first_contact_interval_10_90",
                "occupied_depth",
            )
        }
    _render_topographic_outputs(
        output,
        fields=fields,
        directions=directions,
        offsets=offsets,
        sphere_radius=sphere_radius,
    )
    _write_topographic_analysis(
        data / "topographic-analysis.json",
        _topographic_analysis(fields, directions, offsets, sphere_radius),
    )
    descriptors_path = output / "descriptors.json"
    descriptors = (
        json.loads(descriptors_path.read_text(encoding="utf-8"))
        if descriptors_path.is_file()
        else {}
    )
    write_ml_spatial_tables(
        data,
        fields=fields,
        directions=directions,
        offsets=offsets,
        sphere_radius=sphere_radius,
        resolutions=resolutions,
        analysis_id=str(descriptors.get("input_sha256", ""))[:16],
    )
    manifest_path = data / "plot-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    manifest.update(
        {
            "schema": "d3map.plot-bundle.v1",
            "topographic_directions": list(directions),
            "figure_files": sorted(
                str(path.relative_to(output))
                for path in (output / "figures").rglob("*.png")
            ),
            "data_files": sorted(
                str(path.relative_to(output))
                for path in data.iterdir()
                if path.is_file() and path.name != "plot-manifest.json"
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
