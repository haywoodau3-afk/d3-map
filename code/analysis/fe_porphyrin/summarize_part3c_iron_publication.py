"""Summarize publication-facing hidden steric features for iron-only Part 3c."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CHEMICAL_ROOT = Path(__file__).resolve().parents[1]
RESULTS = CHEMICAL_ROOT / "part3" / "3c"
DOC = CHEMICAL_ROOT.parent / "Doc" / "part3" / "part3c-iron-publication-results.md"
LIGANDS = ("tdcpp", "tpp", "tmp")
DISPLAY = {"tdcpp": "Fe(TDCPP)Cl", "tpp": "Fe(TPP)Cl", "tmp": "Fe(TMP)Cl"}
HARTREE_TO_KCAL_PER_MOL = 627.5094740631


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _curve_volume(summary: dict[str, Any], minimum_open_probability: float) -> float:
    curve = summary["accessibility_persistence_curve"]
    thresholds = np.asarray(curve["minimum_open_probability"], dtype=float)
    index = int(np.argmin(np.abs(thresholds - minimum_open_probability)))
    return float(curve["volume_angstrom3"][index])


def _extreme(field: np.ndarray, axis: np.ndarray, mode: str) -> dict[str, Any]:
    finite = np.isfinite(field)
    if not np.any(finite):
        return {"value": None, "coordinate_angstrom": None}
    values = np.where(finite, field, -np.inf if mode == "max" else np.inf)
    flat = int(np.argmax(values) if mode == "max" else np.argmin(values))
    index = np.unravel_index(flat, field.shape)
    return {
        "value": float(field[index]),
        "coordinate_angstrom": [float(axis[item]) for item in index],
    }


def _d2_record(ligand: str) -> dict[str, Any]:
    root = RESULTS / "d2-map" / f"fecl-{ligand}-d2map"
    output = root / "results" / "reference"
    descriptors = _read_json(output / "descriptors.json")
    probability = _read_json(output / "probability-v2-summary.json")
    baselines = _read_json(output / "static-baselines.json")["baselines"]["input"]
    sampling = _read_json(root / "stages" / "reference" / "stages" / "sampling" / "manifest.json")
    analysis_project = _read_json(
        root / "stages" / "reference" / "stages" / "sampling" / "analysis-project.json"
    )
    energies = np.asarray(analysis_project["population"]["energies"], dtype=float)
    energies = (energies - energies.min()) * HARTREE_TO_KCAL_PER_MOL
    uncertainty = probability["uncertainty"]
    scalar_uncertainty = uncertainty.get("scalar_features", {})
    morphology = probability["classified_morphology"]
    with np.load(output / "probability-fields-v2.npz", allow_pickle=False) as fields:
        axis = np.asarray(fields["axis"], dtype=float)
        covariance = np.asarray(fields["energy_covariance_kcal_per_mol"], dtype=float)
        energy_opening = _extreme(covariance, axis, "max")
        energy_closing = _extreme(covariance, axis, "min")
    return {
        "ligand": ligand,
        "system": DISPLAY[ligand],
        "conformer_count": len(descriptors["weights"]),
        "effective_ensemble_size": descriptors["effective_ensemble_size"],
        "population_model": descriptors["population_model"],
        "temperature_kelvin": descriptors["temperature"],
        "energy_range_kcal_per_mol": float(energies.max() - energies.min()),
        "crest_origin_blocks": sampling.get("conformer_origin_blocks", []),
        "crest_origin_block_count": len(set(sampling.get("conformer_origin_blocks", []))),
        "static_vbur_percent": baselines["vbur_percent"],
        "dynamic_vbur_percent": descriptors["ensemble_scalar_descriptors"]["vbur_percent"][
            "mean"
        ],
        "static_g_percent": baselines["g_percent"],
        "dynamic_g_percent": descriptors["ensemble_scalar_descriptors"]["g_percent"]["mean"],
        "v_open_90_angstrom3": _curve_volume(probability, 0.9),
        "v_open_10_angstrom3": _curve_volume(probability, 0.1),
        "persistent_open_volume_angstrom3": morphology["persistent_open"]["volume_angstrom3"],
        "adaptive_volume_angstrom3": morphology["adaptive"]["volume_angstrom3"],
        "persistent_excluded_volume_angstrom3": morphology["persistent_excluded"][
            "volume_angstrom3"
        ],
        "steric_entropy_volume_angstrom3_nat": probability[
            "steric_occupancy_entropy_volume_angstrom3_nat"
        ],
        "normalized_pocket_flexibility": probability["normalized_pocket_flexibility"],
        "adaptive_signed_halfspace_volume_angstrom3": morphology["adaptive"][
            "signed_halfspace_volume_angstrom3"
        ],
        "adaptive_anisotropy": morphology["adaptive"]["anisotropy"],
        "adaptive_principal_axis": morphology["adaptive"]["principal_axes"][0],
        "energy_analysis": probability["energy_analysis"],
        "highest_energy_associated_occupation": energy_opening,
        "lowest_energy_associated_occupation": energy_closing,
        "uncertainty": uncertainty,
        "adaptive_volume_jackknife_standard_error": scalar_uncertainty.get(
            "adaptive_volume_angstrom3", {}
        ).get("jackknife_standard_error"),
    }


def _comparison_record(extended: str, d2: dict[str, dict[str, Any]]) -> dict[str, Any]:
    root = RESULTS / "d3-map" / f"fecl-tdcpp-vs-{extended}-d3map" / "results" / "comparison"
    comparison = _read_json(root / "comparison.json")
    scalar = {
        (row["descriptor"], row["statistic"]): row for row in comparison["scalar_comparisons"]
    }
    static = {
        (row["descriptor"], row["baseline"]): row
        for row in comparison["static_baseline_comparisons"]
    }
    with np.load(root / "difference-fields.npz", allow_pickle=False) as fields:
        axis = np.asarray(fields["v2_axis"], dtype=float)
        occupation = np.asarray(fields["v2_occupation_difference"], dtype=float)
        entropy = np.asarray(fields["v2_entropy_difference"], dtype=float)
        residual = np.asarray(fields["residual_double_difference_occupation"], dtype=float)
        occupation_gain = _extreme(occupation, axis, "max")
        occupation_loss = _extreme(occupation, axis, "min")
        entropy_gain = _extreme(entropy, axis, "max")
        entropy_loss = _extreme(entropy, axis, "min")
        finite_occupation = occupation[np.isfinite(occupation)]
        finite_entropy = entropy[np.isfinite(entropy)]
        finite_residual = residual[np.isfinite(residual)]
    reference = d2["tdcpp"]
    member = d2[extended]
    return {
        "comparison": f"{DISPLAY[extended]} minus {DISPLAY['tdcpp']}",
        "extended_ligand": extended,
        "static_delta_vbur_percent": static[("vbur_percent", "input")][
            "difference_extended_minus_reference"
        ],
        "dynamic_delta_vbur_percent": scalar[("vbur_percent", "mean")][
            "difference_extended_minus_reference"
        ],
        "static_delta_g_percent": static[("g_percent", "input")][
            "difference_extended_minus_reference"
        ],
        "dynamic_delta_g_percent": scalar[("g_percent", "mean")][
            "difference_extended_minus_reference"
        ],
        "delta_persistent_open_volume_angstrom3": member[
            "persistent_open_volume_angstrom3"
        ]
        - reference["persistent_open_volume_angstrom3"],
        "delta_adaptive_volume_angstrom3": member["adaptive_volume_angstrom3"]
        - reference["adaptive_volume_angstrom3"],
        "delta_persistent_excluded_volume_angstrom3": member[
            "persistent_excluded_volume_angstrom3"
        ]
        - reference["persistent_excluded_volume_angstrom3"],
        "delta_steric_entropy_volume_angstrom3_nat": member[
            "steric_entropy_volume_angstrom3_nat"
        ]
        - reference["steric_entropy_volume_angstrom3_nat"],
        "occupation_mean_absolute_difference": float(np.mean(np.abs(finite_occupation))),
        "entropy_mean_absolute_difference": float(np.mean(np.abs(finite_entropy))),
        "ensemble_residual_mean_absolute_difference": float(np.mean(np.abs(finite_residual))),
        "largest_local_occupation_gain": occupation_gain,
        "largest_local_occupation_loss": occupation_loss,
        "largest_local_entropy_gain": entropy_gain,
        "largest_local_entropy_loss": entropy_loss,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    d2 = {ligand: _d2_record(ligand) for ligand in LIGANDS}
    comparisons = [_comparison_record(ligand, d2) for ligand in ("tpp", "tmp")]
    payload = {
        "schema": "part3c.iron-publication-summary.v1",
        "interpretation": (
            "fixed-sextet super-reduced-search thermodynamic conformer ensembles; not kinetic "
            "probabilities or a validation of spin-state ordering"
        ),
        "d2_maps": list(d2.values()),
        "d3_maps": comparisons,
    }
    target = RESULTS / "part3c-iron-publication-summary.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Part 3c iron-porphyrin publication results",
        "",
        "## Scope and interpretation",
        "",
        "This analysis contains only Fe(TDCPP)Cl, Fe(TPP)Cl, and Fe(TMP)Cl. Each structure was tightly preoptimized without distance restraints in the neutral sextet Fe(III) state, then sampled by a CREST super-reduced search with relaxed Fe–N and Fe–Cl distances protected. Probabilities are energy-weighted thermodynamic conformer probabilities at 298.15 K, not kinetic occupancies. GFN2-xTB does not establish spin-state ordering, and exhaustive-search convergence is not claimed.",
        "",
        "## d2-map results",
        "",
        "| System | Conformers | Neff | CREST blocks | static %Vbur | d2 %Vbur | static G | d2 G | Vopen90 / Å³ | Vopen10 / Å³ | adaptive / Å³ | entropy / Å³ nat |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in d2.values():
        lines.append(
            f"| {row['system']} | {row['conformer_count']} | {_fmt(row['effective_ensemble_size'], 2)} | {row['crest_origin_block_count']} | {_fmt(row['static_vbur_percent'], 2)} | {_fmt(row['dynamic_vbur_percent'], 2)} | {_fmt(row['static_g_percent'], 2)} | {_fmt(row['dynamic_g_percent'], 2)} | {_fmt(row['v_open_90_angstrom3'], 2)} | {_fmt(row['v_open_10_angstrom3'], 2)} | {_fmt(row['adaptive_volume_angstrom3'], 2)} | {_fmt(row['steric_entropy_volume_angstrom3_nat'], 2)} |"
        )
    lines.extend(
        [
            "",
            "`Vopen90` is accessible in at least 90% of the weighted ensemble (`Pocc ≤ 0.1`); `Vopen10` becomes accessible in at least 10% (`Pocc ≤ 0.9`).",
            "",
            "## d3-map differences",
            "",
            "All signs are extended minus Fe(TDCPP)Cl.",
            "",
            "| Comparison | static Δ%Vbur | d3 Δ%Vbur | static ΔG | d3 ΔG | Δopen / Å³ | Δadaptive / Å³ | Δexcluded / Å³ | Δentropy / Å³ nat | occupation MAD |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['comparison']} | {_fmt(row['static_delta_vbur_percent'], 2)} | {_fmt(row['dynamic_delta_vbur_percent'], 2)} | {_fmt(row['static_delta_g_percent'], 2)} | {_fmt(row['dynamic_delta_g_percent'], 2)} | {_fmt(row['delta_persistent_open_volume_angstrom3'], 2)} | {_fmt(row['delta_adaptive_volume_angstrom3'], 2)} | {_fmt(row['delta_persistent_excluded_volume_angstrom3'], 2)} | {_fmt(row['delta_steric_entropy_volume_angstrom3_nat'], 2)} | {_fmt(row['occupation_mean_absolute_difference'], 3)} |"
        )
    lines.extend(["", "## Hidden local features", ""])
    for row in comparisons:
        gain = row["largest_local_occupation_gain"]
        loss = row["largest_local_occupation_loss"]
        lines.append(
            f"- **{row['comparison']}**: the strongest local occupation gain is {_fmt(gain['value'])} at {gain['coordinate_angstrom']} Å; the strongest local opening is {_fmt(loss['value'])} at {loss['coordinate_angstrom']} Å. Mean absolute occupation redistribution is {_fmt(row['occupation_mean_absolute_difference'])}, even when scalar averages partially cancel."
        )
    lines.extend(
        [
            "",
            "## Uncertainty and energy sensitivity",
            "",
            "Every d2-map stores conformer energies, energy-conditioned occupation fields, and CREST-origin labels. Leave-one-origin-block-out jackknife standard errors quantify sensitivity to metadynamics-origin blocks; they are not independent-experiment confidence intervals.",
            "",
        ]
    )
    for row in d2.values():
        lines.append(
            f"- **{row['system']}**: {row['crest_origin_block_count']} retained origin blocks; adaptive-volume jackknife SE {_fmt(row['adaptive_volume_jackknife_standard_error'], 2)} Å³; conformer energy range {_fmt(row['energy_range_kcal_per_mol'], 2)} kcal mol⁻¹."
        )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)
    print(DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
