from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ML_LAYERS = (
    "contact_probability",
    "first_contact_q50",
    "first_contact_interval_10_90",
    "occupied_depth",
)


def _layer_scale(layer: str, sphere_radius: float) -> tuple[float, float]:
    if layer == "contact_probability":
        return 0.0, 1.0
    if layer == "first_contact_q50":
        return -sphere_radius, sphere_radius
    return 0.0, 2.0 * sphere_radius


def _bin_bounds(axis: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    spacing = float(np.mean(np.diff(axis))) if len(axis) > 1 else 0.0
    return float(axis[indices[0]] - spacing / 2), float(axis[indices[-1]] + spacing / 2)


def multiscale_spatial_fractions(
    *,
    fields: dict[str, np.ndarray],
    directions: tuple[str, ...],
    offsets: np.ndarray,
    sphere_radius: float,
    resolutions: tuple[int, ...] = (1, 2, 4, 8),
    analysis_id: str = "",
) -> tuple[list[str], list[list[Any]], list[str], list[list[Any]]]:
    """Create tidy and direction-offset-wide multiresolution steric feature tables."""
    if not resolutions or any(value <= 0 for value in resolutions):
        raise ValueError("ML fraction resolutions must be positive integers")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("ML fraction resolutions cannot contain duplicates")
    axis = np.asarray(fields["axis"], dtype=float)
    if any(value > len(axis) for value in resolutions):
        raise ValueError("ML fraction resolution cannot exceed the topographic grid size")
    long_header = [
        "analysis_id",
        "direction",
        "offset_angstrom",
        "layer",
        "resolution",
        "bin_x",
        "bin_y",
        "x_min_angstrom",
        "x_max_angstrom",
        "y_min_angstrom",
        "y_max_angstrom",
        "pixel_count",
        "valid_pixel_count",
        "valid_fraction",
        "value_mean",
        "value_spatial_standard_deviation",
        "value_minimum",
        "value_maximum",
        "normalized_mean",
    ]
    long_rows: list[list[Any]] = []
    wide_header = ["analysis_id", "direction", "offset_angstrom"]
    feature_names: list[str] = []
    for layer in ML_LAYERS:
        for resolution in resolutions:
            for bin_x in range(resolution):
                for bin_y in range(resolution):
                    stem = f"topo__{layer}__r{resolution}__x{bin_x}__y{bin_y}"
                    feature_names.extend((f"{stem}__mean", f"{stem}__valid_fraction"))
    wide_header.extend(feature_names)
    wide_rows: list[list[Any]] = []

    for direction_index, direction in enumerate(directions):
        for offset_index, offset in enumerate(offsets):
            feature_values: list[Any] = []
            for layer in ML_LAYERS:
                values = np.asarray(fields[layer][direction_index, offset_index], dtype=float)
                scale_minimum, scale_maximum = _layer_scale(layer, sphere_radius)
                for resolution in resolutions:
                    x_groups = np.array_split(np.arange(values.shape[0]), resolution)
                    y_groups = np.array_split(np.arange(values.shape[1]), resolution)
                    for bin_x, x_indices in enumerate(x_groups):
                        for bin_y, y_indices in enumerate(y_groups):
                            block = values[np.ix_(x_indices, y_indices)]
                            valid = np.isfinite(block)
                            valid_values = block[valid]
                            pixel_count = int(block.size)
                            valid_count = int(valid_values.size)
                            valid_fraction = valid_count / pixel_count
                            if valid_count:
                                mean = float(np.mean(valid_values))
                                standard_deviation = float(np.std(valid_values))
                                minimum = float(np.min(valid_values))
                                maximum = float(np.max(valid_values))
                                normalized = (mean - scale_minimum) / (
                                    scale_maximum - scale_minimum
                                )
                            else:
                                mean = standard_deviation = minimum = maximum = normalized = None
                            x_minimum, x_maximum = _bin_bounds(axis, x_indices)
                            y_minimum, y_maximum = _bin_bounds(axis, y_indices)
                            long_rows.append(
                                [
                                    analysis_id,
                                    direction,
                                    float(offset),
                                    layer,
                                    resolution,
                                    bin_x,
                                    bin_y,
                                    x_minimum,
                                    x_maximum,
                                    y_minimum,
                                    y_maximum,
                                    pixel_count,
                                    valid_count,
                                    valid_fraction,
                                    mean,
                                    standard_deviation,
                                    minimum,
                                    maximum,
                                    normalized,
                                ]
                            )
                            feature_values.extend((mean, valid_fraction))
            wide_rows.append([analysis_id, direction, float(offset), *feature_values])
    return long_header, long_rows, wide_header, wide_rows


def write_ml_spatial_tables(
    output: Path,
    *,
    fields: dict[str, np.ndarray],
    directions: tuple[str, ...],
    offsets: np.ndarray,
    sphere_radius: float,
    resolutions: tuple[int, ...],
    analysis_id: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    long_header, long_rows, wide_header, wide_rows = multiscale_spatial_fractions(
        fields=fields,
        directions=directions,
        offsets=offsets,
        sphere_radius=sphere_radius,
        resolutions=resolutions,
        analysis_id=analysis_id,
    )
    for path, header, rows in (
        (output / "ml-spatial-fractions.csv", long_header, long_rows),
        (output / "ml-direction-offset-features.csv", wide_header, wide_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    schema = {
        "schema": "d3map.ml-spatial-fractions.v1",
        "analysis_id": analysis_id,
        "sample_unit_long": "one spatial fraction of one layer, direction, and offset",
        "sample_unit_wide": "one signed direction and outward offset",
        "directions": list(directions),
        "offsets_angstrom": np.asarray(offsets, dtype=float).tolist(),
        "resolutions": list(resolutions),
        "layers": {
            "contact_probability": {"unit": "probability", "scale": [0.0, 1.0]},
            "first_contact_q50": {
                "unit": "angstrom",
                "semantic": "conditional weighted median first-contact depth",
                "scale": [-sphere_radius, sphere_radius],
            },
            "first_contact_interval_10_90": {
                "unit": "angstrom",
                "semantic": "conditional weighted q90 minus q10",
                "scale": [0.0, 2.0 * sphere_radius],
            },
            "occupied_depth": {
                "unit": "angstrom",
                "semantic": "population-weighted union length of occupied ray segments",
                "scale": [0.0, 2.0 * sphere_radius],
            },
        },
        "missing_value_policy": (
            "blank means outside the analysis domain or no conditional contact; "
            "valid_fraction must be retained as a model feature"
        ),
        "normalization": "(value_mean - scale_minimum) / (scale_maximum - scale_minimum)",
        "spatial_standard_deviation": (
            "population-form standard deviation across valid numerical grid pixels within one bin"
        ),
    }
    (output / "ml-feature-schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
