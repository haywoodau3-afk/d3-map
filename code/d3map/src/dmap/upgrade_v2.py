from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .angular import analyze_shielding
from .errors import ValidationError
from .features import analyze_pocket_field
from .io import read_xyz_ensemble
from .models import CatalyticFrame
from .probability_v2 import (
    ProbabilityFieldV2Result,
    analyze_probability_field_v2,
    analyze_probe_accessibility_v2,
    write_probability_v2_figures,
)
from .radii import radii_for_elements


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_shielding_v2(
    result: ProbabilityFieldV2Result,
    *,
    shielding: Any,
    baseline_index: int,
    geometry_count: int,
) -> None:
    probability = shielding.shielding_probability
    entropy = np.zeros_like(probability)
    mixed = (probability > 0.0) & (probability < 1.0)
    entropy[mixed] = -(
        probability[mixed] * np.log(probability[mixed])
        + (1.0 - probability[mixed]) * np.log(1.0 - probability[mixed])
    )
    baseline = shielding.per_geometry_shielding[baseline_index].astype(float)
    result.arrays.update(
        {
            "shielding_directions": shielding.directions,
            "shielding_probability": probability,
            "shielding_entropy": entropy,
            "shielding_per_geometry": shielding.per_geometry_shielding,
            "baseline_shielding": baseline,
            "ensemble_residual_shielding": probability - baseline,
        }
    )
    fractions = {
        "persistent_open_fraction": float(np.mean(probability <= 0.1)),
        "adaptive_fraction": float(np.mean((probability > 0.1) & (probability < 0.9))),
        "persistent_shielded_fraction": float(np.mean(probability >= 0.9)),
    }
    result.features["shielding"] = {
        **fractions,
        "persistent_open_solid_angle_steradian": 4.0
        * float(np.pi)
        * fractions["persistent_open_fraction"],
        "adaptive_solid_angle_steradian": 4.0
        * float(np.pi)
        * fractions["adaptive_fraction"],
        "persistent_shielded_solid_angle_steradian": 4.0
        * float(np.pi)
        * fractions["persistent_shielded_fraction"],
        "mean_binary_entropy_nat": None if geometry_count == 1 else float(np.mean(entropy)),
    }


def upgrade_existing_output_v2(output: Path) -> dict[str, Any]:
    """Deterministically add v2 analysis to one completed d2 branch without sampling."""
    output = Path(output)
    descriptors_path = output / "descriptors.json"
    ensemble_path = output / "aligned-ensemble.xyz"
    legacy_fields_path = output / "axial-fields.npz"
    if not all(path.is_file() for path in (descriptors_path, ensemble_path, legacy_fields_path)):
        raise ValidationError(f"completed d2 artifacts are unavailable: {output}")
    descriptors = _read_json(descriptors_path)
    ensemble = read_xyz_ensemble(ensemble_path)
    weights = np.asarray(descriptors.get("weights", []), dtype=float)
    if weights.shape != (ensemble.size,):
        raise ValidationError(f"stored weights do not match aligned ensemble: {output}")
    frame_values = descriptors["catalytic_frame"]
    frame = CatalyticFrame(
        np.asarray(frame_values["origin"], dtype=float),
        np.asarray(frame_values["basis"], dtype=float),
    )
    steric_indices = tuple(int(value) - 1 for value in descriptors["steric_atoms"])
    radii_profile = descriptors["field_profile"]["radii_profile"]
    radii = radii_for_elements(
        [ensemble.elements[index] for index in steric_indices], profile=radii_profile
    )
    with np.load(legacy_fields_path, allow_pickle=False) as legacy:
        legacy_axis = np.asarray(legacy["pocket_axis"], dtype=float)
        legacy_occupation = np.asarray(legacy["pocket_occupation"], dtype=float)
    spacing = float(legacy_axis[1] - legacy_axis[0])
    sphere_radius = float(max(abs(legacy_axis[0]), abs(legacy_axis[-1])))
    pocket = analyze_pocket_field(
        ensemble,
        frame=frame,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
        sphere_radius=sphere_radius,
        spacing=spacing,
    )
    if pocket.occupation_probability.shape != legacy_occupation.shape:
        raise ValidationError(f"rebuilt occupation shape differs from legacy field: {output}")
    field_rebuild_max_error = float(
        np.nanmax(np.abs(pocket.occupation_probability - legacy_occupation))
    )
    if field_rebuild_max_error > 1e-10:
        raise ValidationError(
            f"rebuilt occupation differs from frozen legacy field by {field_rebuild_max_error:g}: {output}"
        )
    result = analyze_probability_field_v2(pocket, weights=weights)
    baseline_index = 0
    baseline = pocket.per_geometry_occupation[baseline_index].astype(float)
    baseline[~pocket.domain_mask] = np.nan
    result.arrays["baseline_occupation"] = baseline
    result.arrays["ensemble_residual_occupation"] = pocket.occupation_probability - baseline
    result.features["ensemble_residual"] = {
        "baseline": "reference-frame",
        "baseline_frame_number": 1,
        "status": "diagnostic",
        "reason": "stored compatible conformer energies unavailable",
    }
    transformed = tuple(frame.transform(geometry.coordinates) for geometry in ensemble.geometries)
    probe = analyze_probe_accessibility_v2(
        axis=pocket.axis,
        domain=pocket.domain_mask,
        frame_coordinates=transformed,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
        elements=ensemble.elements,
    )
    result.features["probe_accessibility"] = probe.features
    result.arrays.update(probe.arrays)
    shielding = analyze_shielding(
        ensemble,
        frame=frame,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
        direction_count=int(descriptors["static_profile"]["direction_count"]),
    )
    _add_shielding_v2(
        result,
        shielding=shielding,
        baseline_index=baseline_index,
        geometry_count=ensemble.size,
    )
    result.features.update(
        {
            "schema": "d3map.features.v2",
            "software_version": __version__,
            "analysis_source": "frozen-artifact-reanalysis",
            "analysis_mode": descriptors.get("analysis_mode"),
            "source_output": str(output),
            "source_aligned_ensemble_sha256": _sha256(ensemble_path),
            "source_descriptors_sha256": _sha256(descriptors_path),
            "source_probability_semantics": descriptors.get("population_model"),
            "field_rebuild_max_absolute_error": field_rebuild_max_error,
            "field_radii_profile": radii_profile,
            "hydrogen_policy": descriptors.get("hydrogen_policy", "unknown"),
        }
    )
    np.savez_compressed(
        output / "probability-fields-v2.npz",
        schema=np.asarray("d3map.fields.v2"),
        axis=pocket.axis,
        domain=pocket.domain_mask,
        occupation=pocket.occupation_probability,
        entropy=pocket.entropy,
        **result.arrays,
    )
    summary = output / "probability-v2-summary.json"
    summary.write_text(json.dumps(result.features, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = write_probability_v2_figures(output, pocket, result)
    manifest = {
        "schema": "d3map.probability-v2-manifest.v1",
        "software_version": __version__,
        "source_data_policy": "frozen-existing-artifacts-no-new-sampling",
        "source_aligned_ensemble_sha256": _sha256(ensemble_path),
        "artifacts": {
            str(path.relative_to(output)): _sha256(path)
            for path in (
                output / "probability-fields-v2.npz",
                summary,
                *(output / value for value in written if value != summary.name),
            )
            if path.is_file()
        },
    }
    (output / "probability-v2-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result.features


def upgrade_existing_comparison_v2(comparison_output: Path) -> dict[str, Any]:
    """Build direct and residual-double-difference v2 artifacts for one completed d3 pair."""
    comparison_output = Path(comparison_output)
    results = comparison_output.parent
    reference_output = results / "reference"
    extended_output = results / "extended"
    reference_summary = _read_json(reference_output / "probability-v2-summary.json")
    extended_summary = _read_json(extended_output / "probability-v2-summary.json")
    with np.load(reference_output / "probability-fields-v2.npz", allow_pickle=False) as left, np.load(
        extended_output / "probability-fields-v2.npz", allow_pickle=False
    ) as right:
        if not np.allclose(left["axis"], right["axis"], atol=1e-10):
            raise ValidationError(f"paired v2 axes differ: {comparison_output}")
        arrays: dict[str, np.ndarray] = {"axis": left["axis"]}
        for field in ("occupation", "entropy", "shielding_probability"):
            arrays[f"{field}_reference"] = left[field]
            arrays[f"{field}_extended"] = right[field]
            arrays[f"{field}_difference"] = right[field] - left[field]
        arrays["residual_double_difference_occupation"] = (
            right["ensemble_residual_occupation"] - left["ensemble_residual_occupation"]
        )
        arrays["residual_double_difference_shielding"] = (
            right["ensemble_residual_shielding"] - left["ensemble_residual_shielding"]
        )
        arrays["adaptive_class_changed"] = left["adaptive_mask"] != right["adaptive_mask"]
    left_morphology = reference_summary["classified_morphology"]
    right_morphology = extended_summary["classified_morphology"]
    scalar_features: dict[str, Any] = {}
    for label in ("persistent_open", "adaptive", "persistent_excluded"):
        left_value = float(left_morphology[label]["volume_angstrom3"])
        right_value = float(right_morphology[label]["volume_angstrom3"])
        scalar_features[f"{label}_volume_angstrom3"] = {
            "reference": left_value,
            "extended": right_value,
            "difference_extended_minus_reference": right_value - left_value,
        }
    for key in (
        "steric_occupancy_entropy_volume_angstrom3_nat",
        "normalized_pocket_flexibility",
    ):
        left_value = reference_summary.get(key)
        right_value = extended_summary.get(key)
        scalar_features[key] = {
            "reference": left_value,
            "extended": right_value,
            "difference_extended_minus_reference": (
                None if left_value is None or right_value is None else right_value - left_value
            ),
        }
    finite_residual = np.abs(arrays["residual_double_difference_occupation"])
    finite_residual = finite_residual[np.isfinite(finite_residual)]
    payload = {
        "schema": "d3map.comparison-v2.v1",
        "software_version": __version__,
        "difference_sign": "extended-minus-reference",
        "source_data_policy": "frozen-existing-artifacts-no-new-sampling",
        "evidence_status": "exploratory",
        "uncertainty_status": "unavailable-from-frozen-source",
        "scalar_features": scalar_features,
        "field_summaries": {
            "occupation_mean_absolute_difference": float(
                np.nanmean(np.abs(arrays["occupation_difference"]))
            ),
            "entropy_mean_absolute_difference": float(
                np.nanmean(np.abs(arrays["entropy_difference"]))
            ),
            "adaptive_class_changed_fraction": float(np.mean(arrays["adaptive_class_changed"])),
            "residual_double_difference_mean_absolute": (
                float(finite_residual.mean()) if finite_residual.size else 0.0
            ),
            "residual_double_difference_max_absolute": (
                float(finite_residual.max()) if finite_residual.size else 0.0
            ),
        },
        "reference_output": "../reference",
        "extended_output": "../extended",
    }
    comparison_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        comparison_output / "difference-fields-v2.npz",
        schema=np.asarray("d3map.comparison-fields.v2"),
        **arrays,
    )
    (comparison_output / "comparison-v2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
