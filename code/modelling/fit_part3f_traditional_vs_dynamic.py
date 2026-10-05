#!/usr/bin/env python3
"""Compare a traditional steric-descriptor Bayesian fit with Case 3f d2/d3.

Traditional descriptors are evaluated on the same static input structures used
by the Case 3f static-map comparison. The two 3b geometries are averaged into
one catalyst-level row, matching the single published yield observation.
Bayesian selection, prior/noise evidence grids, and nested LOO are matched to
the existing Case 3f 27-feature dynamic-core fit.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dbstep.Dbstep import dbstep
from morfeus import BuriedVolume, SASA, SolidAngle, Sterimol

import run_part3f as part3f


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "03-chemical-validation" / "part3" / "3f"
STATIC_COMPARE = OUT / "case3f-case3d-static-model-inputs.csv"
DYNAMIC_RESULTS = OUT / "case3f-bayesian-results.json"
RESULTS_PATH = OUT / "case3f-traditional-vs-dynamic-bayesian-results.json"
FEATURES_PATH = OUT / "case3f-traditional-static-features.csv"
STRUCTURES_PATH = OUT / "case3f-traditional-per-structure.csv"
PREDICTIONS_PATH = OUT / "case3f-traditional-vs-dynamic-predictions.csv"
PLOT_PATH = OUT / "yield_traditional_vs_dynamic_bayesian.png"

YIELDS = {"3a": 98.0, "3b": 94.0, "3c": 92.0, "3d": 40.0, "3e": 36.0, "IMes": 63.0, "IPr": 42.0}

STRUCTURES: dict[str, list[tuple[str, Path, Path]]] = {
    "3a": [
        (
            "3a",
            ROOT / "03-chemical-validation/part3/3b/d2-map/3a-d2map/results/reference/baselines/input/aligned-ensemble.xyz",
            ROOT / "03-chemical-validation/part3/3b/d2-map/3a-d2map/project.d3map.json",
        )
    ],
    "3b": [
        (
            name,
            ROOT / f"03-chemical-validation/part3/3b/d2-map/{name}-d2map/results/reference/baselines/input/aligned-ensemble.xyz",
            ROOT / f"03-chemical-validation/part3/3b/d2-map/{name}-d2map/project.d3map.json",
        )
        for name in ("3b-syn", "3b-anti")
    ],
    "3c": [
        (
            "3c",
            ROOT / "03-chemical-validation/part3/3b/d2-map/3c-d2map/results/reference/baselines/input/aligned-ensemble.xyz",
            ROOT / "03-chemical-validation/part3/3b/d2-map/3c-d2map/project.d3map.json",
        )
    ],
    "3d": [
        (
            "3d",
            ROOT / "03-chemical-validation/part3/3b/d2-map/3d-d2map/results/reference/baselines/input/aligned-ensemble.xyz",
            ROOT / "03-chemical-validation/part3/3b/d2-map/3d-d2map/project.d3map.json",
        )
    ],
    **{
        name: [
            (
                name,
                ROOT / f"03-chemical-validation/part3/3f/maps/d2-map/{name}-d2map/results/reference/baselines/input/aligned-ensemble.xyz",
                ROOT / f"03-chemical-validation/part3/3f/maps/d2-map/{name}-d2map/project.d3map.json",
            )
        ]
        for name in ("3e", "IMes", "IPr")
    },
}

FEATURE_NAMES = [
    "static_vbur_percent",
    "static_g_percent",
    "dbstep_vbur_percent",
    "dbstep_L_A",
    "dbstep_Bmin_A",
    "dbstep_Bmax_A",
    "morfeus_vbur_percent",
    "morfeus_G_percent",
    "morfeus_cone_angle_deg",
    "morfeus_solid_angle_sr",
    "morfeus_L_A",
    "morfeus_B1_A",
    "morfeus_B5_A",
    "morfeus_SASA_A2",
    "morfeus_SASA_volume_A3",
]


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty XYZ: {path}")
    count = int(lines[0].strip())
    rows = [line.split() for line in lines[2 : 2 + count]]
    if len(rows) != count:
        raise ValueError(f"truncated XYZ: {path}")
    return [row[0] for row in rows], np.asarray([[float(v) for v in row[1:4]] for row in rows])


def reduced_geometry(
    elements: list[str], coords: np.ndarray, center: int, axis: int, steric: list[int]
) -> tuple[list[str], np.ndarray]:
    ordered = [center, axis, *[atom for atom in steric if atom not in {center, axis}]]
    indices = np.asarray(ordered, dtype=int) - 1
    return [elements[index] for index in indices], coords[indices]


def write_xyz(path: Path, elements: list[str], coords: np.ndarray) -> None:
    lines = [str(len(elements)), "Case 3f static input geometry; coordinates unchanged"]
    lines.extend(
        f"{element:<3s} {x: .10f} {y: .10f} {z: .10f}"
        for element, (x, y, z) in zip(elements, coords, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gaussian(path: Path, elements: list[str], coords: np.ndarray) -> None:
    lines = ["# hf/sto-3g", "", "Case 3f static input geometry", "", "0 1"]
    lines.extend(
        f"{element:<3s} {x: .10f} {y: .10f} {z: .10f}"
        for element, (x, y, z) in zip(elements, coords, strict=True)
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_structure(xyz: Path, project: Path, temp: Path, tag: str) -> dict[str, float]:
    elements, coords = read_xyz(xyz)
    config = json.loads(project.read_text(encoding="utf-8"))
    settings = config["structures"]["reference"]
    center = int(settings["center_atom"])
    # Case 3f Cu(I)-NHC structures consistently use Cu atom 1 -> carbene C atom 2.
    axis = 2
    steric = [int(atom) for atom in settings["steric_atoms"]]
    if axis not in steric:
        steric.append(axis)
    red_elements, red_coords = reduced_geometry(elements, coords, center, axis, steric)

    xyz_path = temp / f"{tag}.xyz"
    com_path = temp / f"{tag}.com"
    write_xyz(xyz_path, red_elements, red_coords)
    write_gaussian(com_path, red_elements, red_coords)

    dbv = dbstep(
        str(xyz_path), atom1=1, volume=True, r=3.5, grid=0.10,
        scalevdw=1.17, quiet=True,
    )
    dbs = dbstep(
        str(xyz_path), atom1=1, atom2=2, sterimol=True,
        measure="classic", scalevdw=1.0, quiet=True,
    )
    bv = BuriedVolume(
        red_elements, red_coords, 1, include_hs=True, radius=3.5,
        radii_type="bondi", radii_scale=1.17, density=0.01,
    )
    solid_angle = SolidAngle(red_elements, red_coords, 1, radii_type="crc", density=0.01)
    sterimol = Sterimol(red_elements, red_coords, 1, 2, radii_type="bondi")
    sasa = SASA(red_elements[1:], red_coords[1:], radii_type="crc", density=0.05)

    scalar = lambda value: float(np.asarray(value).reshape(-1)[0])
    return {
        "dbstep_vbur_percent": scalar(dbv.bur_vol),
        "dbstep_L_A": scalar(dbs.L),
        "dbstep_Bmin_A": scalar(dbs.Bmin),
        "dbstep_Bmax_A": scalar(dbs.Bmax),
        "morfeus_vbur_percent": float(bv.fraction_buried_volume * 100.0),
        "morfeus_G_percent": float(solid_angle.G),
        "morfeus_cone_angle_deg": float(solid_angle.cone_angle),
        "morfeus_solid_angle_sr": float(solid_angle.solid_angle),
        "morfeus_L_A": float(sterimol.L_value),
        "morfeus_B1_A": float(sterimol.B_1_value),
        "morfeus_B5_A": float(sterimol.B_5_value),
        "morfeus_SASA_A2": float(sasa.area),
        "morfeus_SASA_volume_A3": float(sasa.volume),
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    static_rows = {row["catalyst"]: row for row in read_rows(STATIC_COMPARE)}
    grouped: list[dict[str, Any]] = []
    per_structure: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="part3f-traditional-descriptors-") as directory:
        temp = Path(directory)
        for catalyst, structures in STRUCTURES.items():
            values_by_structure = []
            for name, xyz, project in structures:
                if not xyz.is_file() or not project.is_file():
                    raise FileNotFoundError(f"missing static structure/project for {name}: {xyz} / {project}")
                metrics = evaluate_structure(xyz, project, temp, name)
                per_structure.append({"catalyst": catalyst, "structure": name, **metrics})
                values_by_structure.append(metrics)
                print(f"{catalyst}/{name}: conventional descriptors complete", flush=True)
            static = static_rows[catalyst]
            merged: dict[str, Any] = {
                "catalyst": catalyst,
                "yield_percent": YIELDS[catalyst],
                "n_structures_averaged": len(structures),
                "static_vbur_percent": float(static["vbur_percent"]),
                "static_g_percent": float(static["g_percent"]),
            }
            for name in values_by_structure[0]:
                merged[name] = float(np.mean([row[name] for row in values_by_structure]))
            grouped.append(merged)
    write_csv(FEATURES_PATH, grouped, ["catalyst", "yield_percent", "n_structures_averaged", *FEATURE_NAMES])
    write_csv(
        STRUCTURES_PATH,
        per_structure,
        ["catalyst", "structure", *[name for name in per_structure[0] if name not in {"catalyst", "structure"}]],
    )
    return grouped, per_structure


def comparison_plot(
    conventional: dict[str, Any],
    traditional_all: dict[str, Any],
    dynamic: dict[str, Any],
    labels: list[str],
    observed: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), constrained_layout=True, sharex=True, sharey=True)
    max_predicted = max(
        float(np.max(np.asarray(result["nested_loo_predicted"], dtype=float)))
        for result in (conventional, traditional_all, dynamic)
    )
    y_upper = max(105.0, 5.0 * math.ceil((max_predicted + 5.0) / 5.0))
    panels = [
        (axes[0], conventional, "Traditional pool; evidence selects ≤3", "#8a4f7d"),
        (axes[1], traditional_all, "All 15 traditional features; Bayesian ridge", "#477a67"),
        (axes[2], dynamic, "Dynamic d2/d3-map core (27 candidates)", "#b23a48"),
    ]
    for ax, result, title, color in panels:
        fitted = np.asarray(result["predicted"], dtype=float)
        loo = np.asarray(result["nested_loo_predicted"], dtype=float)
        ax.plot([0, 100], [0, 100], color="#7a7a7a", lw=1.2, ls="--", label="perfect prediction")
        ax.scatter(observed, fitted, s=58, color=color, marker="o", label="full-data fit", zorder=3)
        ax.scatter(observed, loo, s=74, facecolors="white", edgecolors=color, linewidths=1.8, marker="^", label="nested LOO prediction", zorder=4)
        offsets = {
            "3a": (5, 10), "3b": (5, -18), "3c": (-19, 3),
            "3d": (5, -15), "3e": (-20, -12), "IMes": (5, 5), "IPr": (7, 10),
        }
        for x, y, label in zip(observed, loo, labels, strict=True):
            ax.annotate(label, (x, y), xytext=offsets.get(label, (4, 4)), textcoords="offset points", fontsize=8)
        ax.set_title(
            f"{title}\nin-sample R²={result['r2_in_sample']:.3f}; nested LOO R²={result['nested_loo_r2']:.3f}"
        )
        ax.set_xlim(0, 105)
        ax.set_ylim(0, y_upper)
        ax.set_xlabel("Observed Table 1 yield (%)")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=8, frameon=True)
    axes[0].set_ylabel("Bayesian predicted yield (%)")
    fig.suptitle("Case 3f: conventional steric descriptors vs dynamic d2/d3 maps", fontsize=14)
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def fit_all_features_bayesian(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Fit the complete traditional feature pool with Bayesian ridge shrinkage.

    This fixed-feature-set sensitivity model complements evidence selection:
    every traditional variable is included, while prior/noise evidence and all
    scaling are refit within each outer LOO training fold.
    """
    size = len(names)
    fitted = part3f._fit_evidence_fast(
        x, y, names, include_diagnostics=False, min_size=size, max_size=size
    )
    loo = part3f.bayes._loo_metrics(x, y, names, fitted)
    predictions: list[float] = []
    predictive_sd: list[float] = []
    for held_out in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[held_out] = False
        fold = part3f._fit_evidence_fast(
            x[train_mask], y[train_mask], names,
            include_diagnostics=False, min_size=size, max_size=size,
        )
        prediction, sd = part3f.bayes._posterior_predict(
            x[train_mask], y[train_mask], x[held_out], names, fold
        )
        predictions.append(prediction)
        predictive_sd.append(sd)
    metrics = part3f.bayes._summarize_predictions(y, predictions, predictive_sd)
    fitted.update(loo)
    fitted.update(
        {
            "nested_loo_predicted": predictions,
            "nested_loo_predictive_sd": predictive_sd,
            "nested_loo_rmse": metrics["rmse"],
            "nested_loo_r2": metrics["r2"],
            "nested_loo_coverage_95": metrics["coverage_95"],
            "candidate_feature_count": size,
            "candidate_features": names,
            "selection_policy": "fixed all-feature set; evidence-selected prior/noise and standardization refit in each nested LOO fold; no feature subset selection",
        }
    )
    return fitted


def main() -> None:
    grouped, per_structure = build_feature_tables()
    names = FEATURE_NAMES.copy()
    x = np.asarray([[float(row[name]) for name in names] for row in grouped], dtype=float)
    y = np.asarray([YIELDS[row["catalyst"]] for row in grouped], dtype=float)
    # Route nested folds through Case 3f's vectorized implementation of the
    # same Case 3d marginal-likelihood Bayesian ridge model.
    part3f.bayes._fit_evidence = part3f._fit_evidence_fast
    conventional = part3f._fit_evidence_fast(x, y, names)
    traditional_all = fit_all_features_bayesian(x, y, names)

    dynamic_results = json.loads(DYNAMIC_RESULTS.read_text(encoding="utf-8"))
    dynamic = dynamic_results["core_feature_ensemble_model"]
    labels = [row["catalyst"] for row in grouped]
    comparison_plot(conventional, traditional_all, dynamic, labels, y)

    prediction_rows = []
    for index, row in enumerate(grouped):
        prediction_rows.append(
            {
                "catalyst": row["catalyst"],
                "yield_observed": y[index],
                "traditional_selected_fit": conventional["predicted"][index],
                "traditional_selected_nested_loo": conventional["nested_loo_predicted"][index],
                "traditional_all15_fit": traditional_all["predicted"][index],
                "traditional_all15_nested_loo": traditional_all["nested_loo_predicted"][index],
                "dynamic_27_core_fit": dynamic["predicted"][index],
                "dynamic_27_core_nested_loo": dynamic["nested_loo_predicted"][index],
            }
        )
    write_csv(PREDICTIONS_PATH, prediction_rows, list(prediction_rows[0]))

    result = {
        "schema": "part3f.traditional-vs-dynamic-bayesian.v1",
        "status": "completed_exploratory_same_yield_rows_nested_loo_comparison",
        "unit_of_analysis": "seven catalyst labels; 3b syn/anti conventional descriptor values averaged to its one yield row",
        "outcome": {"name": "yield_percent", "values": YIELDS, "source": "attached article Table 1 at 0.02 mol%"},
        "traditional_descriptor_model": {
            **conventional,
            "candidate_feature_count": len(names),
            "candidate_features": names,
            "feature_origin": "static descriptors only; no d2/d3 dynamic-map-derived features; DBSTEP 1.1.0 and Morfeus-ML 0.8.0 evaluated on the unchanged static input geometries",
            "static_vbur_and_solid_g_origin": "existing Case 3f static input-geometry d2 descriptor outputs; no CREST population averaging",
            "selection_policy": "Gaussian Bayesian ridge marginal-likelihood selection of 1-3 descriptors from the traditional candidate set; all scaling, subset selection, prior/noise selection, and posterior fitting repeated inside every leave-one-catalyst-out training fold",
            "cone_angle_note": "Morfeus SolidAngle cone_angle, a generalized Tolman-style steric cone-angle descriptor for Cu-NHC structures; not a phosphine-specific canonical Tolman angle",
            "sterimol_note": "Morfeus Sterimol L/B1/B5 and DBSTEP classic L/Bmin/Bmax, with the Cu-to-carbene axis declared as the measurement axis",
        },
        "traditional_all_features_model": traditional_all,
        "dynamic_map_model": {
            "candidate_feature_count": 27,
            "candidate_features": dynamic_results["feature_pool"]["candidate_features"],
            "r2_in_sample": dynamic["r2_in_sample"],
            "nested_loo_r2": dynamic["nested_loo_r2"],
            "nested_loo_rmse": dynamic["nested_loo_rmse"],
            "nested_loo_predicted": dynamic["nested_loo_predicted"],
            "selected_features_full_data": dynamic["feature_names"],
            "selected_features_by_nested_fold": dynamic["nested_loo_selected_features"],
            "feature_origin": "dynamic CREST d2 ensemble and paired d3-map-derived features from the existing Case 3f 27-feature core",
        },
        "bayesian_protocol": "Same Case 3d/Case 3f Gaussian Bayesian ridge evidence grid, standardized outcomes and predictors, model-size cap of three, and nested leave-one-catalyst-out feature/hyperparameter selection for both representations.",
        "interpretation_limit": "Exploratory comparison only (n=7). Three catalysts use xTB-constructed exploratory geometries; negative or positive LOO R2 is unstable at this sample size and does not establish prospective predictive performance or causation.",
        "files": {
            "traditional_feature_matrix": str(FEATURES_PATH),
            "per_structure_descriptors": str(STRUCTURES_PATH),
            "catalyst_level_predictions": str(PREDICTIONS_PATH),
            "comparison_plot": str(PLOT_PATH),
        },
        "traditional_structure_count": len(per_structure),
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "traditional_candidate_count": len(names),
                "traditional_selected_features": conventional["feature_names"],
                "traditional_r2_in_sample": conventional["r2_in_sample"],
                "traditional_nested_loo_r2": conventional["nested_loo_r2"],
                "traditional_all_features_r2_in_sample": traditional_all["r2_in_sample"],
                "traditional_all_features_nested_loo_r2": traditional_all["nested_loo_r2"],
                "dynamic_r2_in_sample": dynamic["r2_in_sample"],
                "dynamic_nested_loo_r2": dynamic["nested_loo_r2"],
                "plot": str(PLOT_PATH),
                "results": str(RESULTS_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
