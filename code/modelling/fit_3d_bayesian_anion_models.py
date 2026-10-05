#!/usr/bin/env python3
"""Fit small-sample Bayesian ee/dr models for the Figure 3a anion screen.

The unit of analysis is one anion state.  The three structural seeds and two
solvent labels are first averaged within anion, so they are not treated as six
independent chemical experiments.  A conjugate Gaussian Bayesian ridge model
with evidence-selected feature subsets (up to three descriptors) supplies
shrinkage, posterior weights, credible intervals, and posterior predictive
diagnostics without requiring PyMC/ArviZ.

All seven anion states are xTB-constructed analogues; this script therefore
reports an exploratory state-conditioned fit and never labels it causal,
DFT-validated, thermodynamic, or kinetic.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PART3D = ROOT / "03-chemical-validation" / "part3" / "3d"
MATRIX = PART3D / "descriptor-matrix-static.csv"
OUT = PART3D / "bayesian-anion-regression"
INTERACTION = PART3D / "interaction-figure3a.csv"

ANIONS = ("OMs", "OTs", "OTf", "PF6", "BF4", "Cl", "BArF4")
FEATURES = (
    "g_percent",
    "vbur_percent",
    "approach_accessibility_mean",
    "persistent_open_fraction",
    "persistent_blocked_fraction",
    "pocket_integrated_vbur_percent",
    "pocket_weighted_vbur_percent",
    "quadrant_asymmetry_variance",
    "d3_changed_occupation_area_fraction",
    "d3_changed_pocket_occupation_area_fraction",
    "d3_changed_topographic_contact_probability_area_fraction",
    "d3_changed_topographic_first_contact_q50_area_fraction",
    "d3_changed_topographic_occupied_depth_area_fraction",
    "d3_delta_vbur_percent_mean",
    "d3_delta_g_percent_mean",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _state_rows(rows: list[dict[str, str]], anion: str) -> list[dict[str, str]]:
    label = f"exploratory-3b-4d-{anion}"
    selected = [row for row in rows if row.get("chemical_state") == label]
    if len(selected) != 6:
        raise ValueError(f"{anion}: expected six seed/solvent descriptor rows, found {len(selected)}")
    return selected


def _aggregate(rows: list[dict[str, str]], anion: str, outcome: dict[str, str]) -> dict[str, Any]:
    values: dict[str, float] = {}
    sd_values: dict[str, float] = {}
    for feature in FEATURES:
        numeric = [_finite(row.get(feature, "")) for row in rows]
        numeric = [item for item in numeric if item is not None]
        if numeric:
            values[feature] = float(np.mean(numeric))
            sd_values[f"{feature}_sd"] = float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
    result: dict[str, Any] = {
        "anion": anion,
        "n_structural_rows": len(rows),
        "n_seeds": len({row["seed"] for row in rows}),
        "n_solvents": len({row["solvent"] for row in rows}),
        "ee_percent": float(outcome["ee_pct"]),
        "dr_ratio": float(outcome["dr_numeric"]),
        "log_dr": float(np.log(float(outcome["dr_numeric"]))),
        "source_status": "xTB_constructed_state_conditioned_analogue",
        "condition_match_status": "analogue_not_exact",
    }
    result.update(values)
    result.update(sd_values)
    return result


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale, mean, scale


def _posterior_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    names: list[str],
    result: dict[str, Any],
) -> tuple[float, float]:
    """Predict one held-out row using a fitted subset and hyperparameters."""
    selected = list(result["feature_names"])
    selected_indices = [names.index(name) for name in selected]
    tau = float(result["prior_tau"])
    sigma = float(result["noise_sigma_standardized"])
    y_mean = float(np.mean(y_train))
    y_sd = float(np.std(y_train, ddof=1)) or 1.0
    ys = (y_train - y_mean) / y_sd
    xs, x_mean, x_sd = _standardize(x_train)
    design = np.column_stack([np.ones(len(y_train)), xs[:, selected_indices]])
    prior_var = np.asarray([100.0**2, *([tau**2] * len(selected))], dtype=float)
    prior_cov = np.diag(prior_var)
    covariance = sigma**2 * np.eye(len(y_train)) + design @ prior_cov @ design.T
    inv_cov = np.linalg.inv(covariance)
    posterior_mean = prior_cov @ design.T @ inv_cov @ ys
    posterior_cov = prior_cov - prior_cov @ design.T @ inv_cov @ design @ prior_cov
    x_test_standardized = (x_test[selected_indices] - x_mean[selected_indices]) / x_sd[selected_indices]
    test_design = np.asarray([1.0, *x_test_standardized], dtype=float)
    pred_standardized = float(test_design @ posterior_mean)
    pred_variance_standardized = float(sigma**2 + test_design @ posterior_cov @ test_design)
    return (
        float(y_mean + y_sd * pred_standardized),
        float(y_sd * math.sqrt(max(pred_variance_standardized, 0.0))),
    )


def _summarize_predictions(y: np.ndarray, predictions: list[float], predictive_sd: list[float]) -> dict[str, Any]:
    residual = y - np.asarray(predictions, dtype=float)
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / sst) if sst > 0 else 0.0
    sd = np.asarray(predictive_sd, dtype=float)
    coverage = float(np.mean(np.abs(residual) <= 1.96 * sd))
    return {"rmse": rmse, "r2": r2, "coverage_95": coverage}


def _loo_metrics(x: np.ndarray, y: np.ndarray, names: list[str], result: dict[str, Any]) -> dict[str, Any]:
    """Compute fixed-subset leave-one-out posterior predictions.

    The feature subset, prior scale, and noise scale are selected on the full
    seven-state dataset. Each held-out row is standardized using the training
    rows only, so these diagnostics are leakage-free but not nested model
    selection estimates.
    """
    predictions: list[float] = []
    predictive_sd: list[float] = []
    for held_out in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[held_out] = False
        prediction, sd = _posterior_predict(x[train_mask], y[train_mask], x[held_out], names, result)
        predictions.append(prediction)
        predictive_sd.append(sd)
    metrics = _summarize_predictions(y, predictions, predictive_sd)
    return {
        "loo_predicted": predictions,
        "loo_predictive_sd": predictive_sd,
        "loo_rmse": metrics["rmse"],
        "loo_r2": metrics["r2"],
        "loo_coverage_95": metrics["coverage_95"],
    }


def _loo_intercept_metrics(y: np.ndarray) -> dict[str, Any]:
    """Return the leave-one-out intercept-only baseline for comparison."""
    predictions = np.asarray([np.mean(np.delete(y, index)) for index in range(len(y))], dtype=float)
    residual = y - predictions
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / sst) if sst > 0 else 0.0
    return {"loo_intercept_rmse": rmse, "loo_intercept_r2": r2}


def _nested_loo_metrics(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Select descriptors and hyperparameters inside every leave-one-out fold."""
    predictions: list[float] = []
    predictive_sd: list[float] = []
    selected_features: list[list[str]] = []
    for held_out in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[held_out] = False
        fitted = _fit_evidence(x[train_mask], y[train_mask], names, include_diagnostics=False)
        prediction, sd = _posterior_predict(x[train_mask], y[train_mask], x[held_out], names, fitted)
        predictions.append(prediction)
        predictive_sd.append(sd)
        selected_features.append(list(fitted["feature_names"]))
    metrics = _summarize_predictions(y, predictions, predictive_sd)
    return {
        "nested_loo_predicted": predictions,
        "nested_loo_predictive_sd": predictive_sd,
        "nested_loo_rmse": metrics["rmse"],
        "nested_loo_r2": metrics["r2"],
        "nested_loo_coverage_95": metrics["coverage_95"],
        "nested_loo_selected_features": selected_features,
        "nested_loo_feature_selection_frequency": dict(Counter(feature for fold in selected_features for feature in fold)),
    }


def _permutation_metrics(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    result: dict[str, Any],
    n_permutations: int = 2000,
    seed: int = 20260922,
) -> dict[str, Any]:
    """Calibrate the fixed-subset LOO R² against outcome-shuffle controls."""
    rng = np.random.default_rng(seed)
    null_r2 = []
    for _ in range(n_permutations):
        permuted = rng.permutation(y)
        null_r2.append(float(_loo_metrics(x, permuted, names, result)["loo_r2"]))
    null = np.asarray(null_r2, dtype=float)
    observed = float(result["loo_r2"])
    return {
        "permutation_count": n_permutations,
        "permutation_seed": seed,
        "permutation_loo_r2_mean": float(np.mean(null)),
        "permutation_loo_r2_sd": float(np.std(null, ddof=1)),
        "permutation_loo_r2_95_interval": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "permutation_p_ge_observed": float((1 + np.sum(null >= observed)) / (n_permutations + 1)),
    }


def _fit_evidence(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Select a <=3-feature model by Gaussian marginal likelihood."""
    n, p = x.shape
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y, ddof=1)) or 1.0
    ys = (y - y_mean) / y_sd
    xs, x_mean, x_sd = _standardize(x)
    candidates: list[tuple[float, tuple[int, ...], float, float]] = []
    tau_grid = np.geomspace(0.08, 4.0, 12)
    sigma_grid = np.geomspace(0.08, 2.0, 14)
    for size in range(1, min(3, p) + 1):
        for subset in itertools.combinations(range(p), size):
            design = np.column_stack([np.ones(n), xs[:, subset]])
            for tau in tau_grid:
                prior_var = np.asarray([100.0**2, *([tau**2] * size)], dtype=float)
                prior_cov = np.diag(prior_var)
                for sigma in sigma_grid:
                    covariance = sigma**2 * np.eye(n) + design @ prior_cov @ design.T
                    sign, logdet = np.linalg.slogdet(covariance)
                    if sign <= 0:
                        continue
                    solution = np.linalg.solve(covariance, ys)
                    evidence = -0.5 * (float(ys @ solution) + float(logdet) + n * math.log(2.0 * math.pi))
                    candidates.append((evidence, subset, tau, sigma))
    if not candidates:
        raise ValueError("no Bayesian model candidates were generated")
    evidence, subset, tau, sigma = max(candidates, key=lambda item: item[0])
    design = np.column_stack([np.ones(n), xs[:, subset]])
    prior_var = np.asarray([100.0**2, *([tau**2] * len(subset))], dtype=float)
    prior_cov = np.diag(prior_var)
    covariance = sigma**2 * np.eye(n) + design @ prior_cov @ design.T
    inv_cov = np.linalg.inv(covariance)
    posterior_mean = prior_cov @ design.T @ inv_cov @ ys
    posterior_cov = prior_cov - prior_cov @ design.T @ inv_cov @ design @ prior_cov
    posterior_sd = np.sqrt(np.clip(np.diag(posterior_cov), 0.0, None))
    # Separate the descriptor contribution from the intercept.  The former is
    # the x-axis for descriptor-response plots; using the complete fitted
    # outcome there would create a response-versus-response plot and impose a
    # positive slope by construction.
    descriptor_score = xs[:, subset] @ posterior_mean[1:]
    fitted = posterior_mean[0] + descriptor_score
    residual = ys - fitted
    rmse = float(np.sqrt(np.mean((residual * y_sd) ** 2)))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - np.sum((residual * y_sd) ** 2) / sst) if sst > 0 else 0.0
    prediction_sd = np.sqrt(np.clip(sigma**2 + np.sum((design @ posterior_cov) * design, axis=1), 0.0, None)) * y_sd
    beta = posterior_mean[1:]
    abs_beta = np.abs(beta)
    contribution = abs_beta / float(np.sum(abs_beta)) if np.sum(abs_beta) else abs_beta
    weights = []
    for index, name in zip(subset, [names[i] for i in subset], strict=True):
        weights.append(
            {
                "feature": name,
                "posterior_weight_standardized": float(posterior_mean[len(weights) + 1]),
                "posterior_sd_standardized": float(posterior_sd[len(weights) + 1]),
                "credible_interval_95_standardized": [
                    float(posterior_mean[len(weights) + 1] - 1.96 * posterior_sd[len(weights) + 1]),
                    float(posterior_mean[len(weights) + 1] + 1.96 * posterior_sd[len(weights) + 1]),
                ],
                "absolute_contribution_fraction": float(contribution[len(weights)]),
            }
        )
    result = {
        "feature_names": [names[i] for i in subset],
        "evidence": float(evidence),
        "prior_tau": float(tau),
        "noise_sigma_standardized": float(sigma),
        "outcome_mean": y_mean,
        "outcome_sd": y_sd,
        "predictor_means": {name: float(x_mean[i]) for i, name in enumerate(names)},
        "predictor_sds": {name: float(x_sd[i]) for i, name in enumerate(names)},
        "weights": weights,
        "intercept_standardized": float(posterior_mean[0]),
        "intercept_sd_standardized": float(posterior_sd[0]),
        "rmse": rmse,
        "r2_in_sample": r2,
        "observed": [float(item) for item in y],
        "predicted": [float(y_mean + y_sd * value) for value in fitted],
        "predictive_sd": [float(value) for value in prediction_sd],
        "descriptor_score": [float(value) for value in descriptor_score],
        # Retain the legacy key as an alias for downstream readers, but it now
        # contains the true posterior-weighted descriptor contribution rather
        # than the fitted outcome.
        "weighted_score": [float(value) for value in descriptor_score],
        "fitted_standardized": [float(value) for value in fitted],
    }
    if include_diagnostics:
        result.update(_loo_metrics(x, y, names, result))
        result.update(_loo_intercept_metrics(y))
        result["loo_improves_intercept"] = bool(result["loo_rmse"] < result["loo_intercept_rmse"])
        result.update(_nested_loo_metrics(x, y, names))
        result["nested_loo_intercept_rmse"] = result["loo_intercept_rmse"]
        result["nested_loo_intercept_r2"] = result["loo_intercept_r2"]
        result["nested_loo_improves_intercept"] = bool(
            result["nested_loo_rmse"] < result["nested_loo_intercept_rmse"]
        )
        result.update(_permutation_metrics(x, y, names, result))
    return result


def _plot(result: dict[str, Any], data: list[dict[str, Any]], outcome_name: str, path: Path) -> None:
    x = np.asarray(result["weighted_score"], dtype=float)
    if outcome_name == "dr":
        # The model is fitted on log(dr), but the published-facing plot is
        # returned to the raw major/minor ratio for interpretability.
        y = np.exp(np.asarray(result["observed"], dtype=float))
        fitted = np.exp(np.asarray(result["predicted"], dtype=float))
        lower = np.exp(np.asarray(result["predicted"], dtype=float) - 1.96 * np.asarray(result["predictive_sd"], dtype=float))
        upper = np.exp(np.asarray(result["predicted"], dtype=float) + 1.96 * np.asarray(result["predictive_sd"], dtype=float))
        yerr = np.vstack([y - lower, upper - y])
        ylabel = "dr major/minor"
    elif outcome_name == "dr_raw":
        y = np.asarray(result["observed"], dtype=float)
        fitted = np.asarray(result["predicted"], dtype=float)
        yerr = np.asarray(result["predictive_sd"], dtype=float) * 1.96
        ylabel = "dr major/minor (raw-scale sensitivity)"
    else:
        y = np.asarray(result["observed"], dtype=float)
        fitted = np.asarray(result["predicted"], dtype=float)
        yerr = np.asarray(result["predictive_sd"], dtype=float) * 1.96
        ylabel = "ee (%)"
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color="#245b9e")
    ax.plot(x[order], fitted[order], color="#b23a48", lw=1.8)
    if np.ndim(yerr) == 1:
        interval_low = y - yerr
        interval_high = y + yerr
    else:
        interval_low = y - yerr[0]
        interval_high = y + yerr[1]
    y_low = float(np.min(interval_low))
    y_high = float(np.max(interval_high))
    y_margin = max((y_high - y_low) * 0.08, 1.0)
    ax.set_ylim(y_low - y_margin, y_high + y_margin)
    for xi, yi, row in zip(x, y, data, strict=True):
        ax.annotate(row["anion"], (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Bayesian weighted descriptor score (standardized posterior mean)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Exploratory {outcome_name} fit: anion-level state-conditioned model")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_nested_validation(
    results: list[tuple[str, dict[str, Any], str]],
    data: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot nested leave-one-anion-out predictions against observations."""
    fig, axes = plt.subplots(1, len(results), figsize=(13.2, 4.2), constrained_layout=True)
    for ax, (label, result, outcome_name) in zip(axes, results, strict=True):
        observed = np.asarray(result["observed"], dtype=float)
        predicted = np.asarray(result["nested_loo_predicted"], dtype=float)
        predictive_sd = np.asarray(result["nested_loo_predictive_sd"], dtype=float)
        if outcome_name == "dr":
            observed = np.exp(observed)
            predicted = np.exp(predicted)
            lower = np.exp(np.asarray(result["nested_loo_predicted"], dtype=float) - 1.96 * predictive_sd)
            upper = np.exp(np.asarray(result["nested_loo_predicted"], dtype=float) + 1.96 * predictive_sd)
            yerr = np.vstack([predicted - lower, upper - predicted])
        else:
            yerr = 1.96 * predictive_sd
        ax.errorbar(observed, predicted, yerr=yerr, fmt="o", capsize=3, color="#245b9e")
        for xi, yi, row in zip(observed, predicted, data, strict=True):
            ax.annotate(row["anion"], (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        low = float(min(np.min(observed), np.min(predicted) - np.max(yerr)))
        high = float(max(np.max(observed), np.max(predicted) + np.max(yerr)))
        margin = max((high - low) * 0.08, 1.0)
        ax.plot([low - margin, high + margin], [low - margin, high + margin], color="#b23a48", lw=1.4, linestyle="--")
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Nested LOO predicted" if ax is axes[0] else "")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle("Case 3d nested leave-one-anion-out diagnostics\nstate-conditioned xTB analogue analysis", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    descriptor_rows = _read_csv(MATRIX)
    outcomes = {row["condition"]: row for row in _read_csv(INTERACTION) if row["condition_class"] == "anion"}
    aggregate_rows = []
    for anion in ANIONS:
        aggregate_rows.append(_aggregate(_state_rows(descriptor_rows, anion), anion, outcomes[anion]))
    OUT.mkdir(parents=True, exist_ok=True)
    feature_names = [feature for feature in FEATURES if all(feature in row and math.isfinite(float(row[feature])) for row in aggregate_rows)]
    if len(feature_names) < 2:
        raise ValueError(f"too few complete features for model: {feature_names}")
    with (OUT / "anion-level-model-inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["anion", "n_structural_rows", "n_seeds", "n_solvents", "ee_percent", "dr_ratio", "log_dr", *feature_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    x = np.asarray([[row[feature] for feature in feature_names] for row in aggregate_rows], dtype=float)
    ee = np.asarray([row["ee_percent"] for row in aggregate_rows], dtype=float)
    dr = np.asarray([row["dr_ratio"] for row in aggregate_rows], dtype=float)
    log_dr = np.asarray([row["log_dr"] for row in aggregate_rows], dtype=float)
    ee_result = _fit_evidence(x, ee, feature_names)
    dr_result = _fit_evidence(x, log_dr, feature_names)
    dr_raw_result = _fit_evidence(x, dr, feature_names)
    _plot(ee_result, aggregate_rows, "ee", OUT / "ee_vs_weighted_descriptor_score.png")
    _plot(dr_result, aggregate_rows, "dr", OUT / "dr_vs_weighted_descriptor_score.png")
    _plot(dr_raw_result, aggregate_rows, "dr_raw", OUT / "dr_raw_sensitivity_vs_weighted_descriptor_score.png")
    _plot_nested_validation(
        [("ee (%)", ee_result, "ee"), ("log(dr)", dr_result, "dr"), ("dr raw sensitivity", dr_raw_result, "dr_raw")],
        aggregate_rows,
        OUT / "nested_loo_validation.png",
    )
    with (OUT / "nested-loo-predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "anion",
            "ee_observed",
            "ee_nested_loo_predicted",
            "ee_nested_loo_predictive_sd",
            "ee_nested_loo_selected_features",
            "log_dr_observed",
            "log_dr_nested_loo_predicted",
            "log_dr_nested_loo_predictive_sd",
            "log_dr_nested_loo_selected_features",
            "dr_raw_observed",
            "dr_raw_nested_loo_predicted",
            "dr_raw_nested_loo_predictive_sd",
            "dr_raw_nested_loo_selected_features",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(aggregate_rows):
            writer.writerow(
                {
                    "anion": row["anion"],
                    "ee_observed": row["ee_percent"],
                    "ee_nested_loo_predicted": ee_result["nested_loo_predicted"][index],
                    "ee_nested_loo_predictive_sd": ee_result["nested_loo_predictive_sd"][index],
                    "ee_nested_loo_selected_features": ";".join(ee_result["nested_loo_selected_features"][index]),
                    "log_dr_observed": row["log_dr"],
                    "log_dr_nested_loo_predicted": dr_result["nested_loo_predicted"][index],
                    "log_dr_nested_loo_predictive_sd": dr_result["nested_loo_predictive_sd"][index],
                    "log_dr_nested_loo_selected_features": ";".join(dr_result["nested_loo_selected_features"][index]),
                    "dr_raw_observed": row["dr_ratio"],
                    "dr_raw_nested_loo_predicted": dr_raw_result["nested_loo_predicted"][index],
                    "dr_raw_nested_loo_predictive_sd": dr_raw_result["nested_loo_predictive_sd"][index],
                    "dr_raw_nested_loo_selected_features": ";".join(dr_raw_result["nested_loo_selected_features"][index]),
                }
            )
    summary = {
        "status": "completed_exploratory_bayesian_anion_fits_with_nested_validation",
        "unit_of_analysis": "one_anion_state; six structural rows aggregated within anion",
        "n_anion_states": len(aggregate_rows),
        "features_considered": feature_names,
        "maximum_model_features": 3,
        "outcome_transform": {"ee": "percent", "dr": "natural_log_of_major_minor_ratio", "dr_raw_sensitivity": "raw_major_minor_ratio"},
        "validation_protocol": {
            "nested_leave_one_anion_out": True,
            "outcome_shuffle_permutations": 2000,
            "permutation_seed": 20260922,
            "permutation_scope": "fixed_full-data_subset_and_prior_hyperparameters; calibration control, not nested selection",
            "structural_replicate_handling": "three seeds and two solvents aggregated within anion before fitting",
        },
        "scientific_boundary": "All seven states are xTB-constructed analogues; coefficients describe state-conditioned association only and are not DFT validation, binding free energies, barriers, kinetic probabilities, or causal selectivity mechanisms.",
        "ee": ee_result,
        "dr": dr_result,
        "dr_raw_sensitivity": dr_raw_result,
        "files": {
            "inputs": str(OUT / "anion-level-model-inputs.csv"),
            "nested_loo_predictions": str(OUT / "nested-loo-predictions.csv"),
            "ee_plot": str(OUT / "ee_vs_weighted_descriptor_score.png"),
            "dr_plot": str(OUT / "dr_vs_weighted_descriptor_score.png"),
            "dr_raw_sensitivity_plot": str(OUT / "dr_raw_sensitivity_vs_weighted_descriptor_score.png"),
            "nested_loo_validation_plot": str(OUT / "nested_loo_validation.png"),
        },
    }
    (OUT / "bayesian-regression-results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Case 3d exploratory Bayesian ee/dr anion fit",
        "",
        "The model unit is one Figure 3a anion state. Three structural seeds and two solvent labels were averaged within each anion; they are not six independent reaction observations.",
        "",
        f"Seven anion states were fitted with evidence-selected Bayesian ridge models using at most three descriptors from {len(feature_names)} complete candidates. Descriptor selection and prior/noise-grid selection were repeated inside each leave-one-anion-out fold for nested predictive diagnostics; 2,000 fixed-subset outcome-shuffle permutations provide a calibration control.",
        "",
        "## Boundary",
        "",
        "All seven coordinates are xTB-constructed analogues. The fit is exploratory and state-conditioned; it is not DFT validation, a binding free-energy model, a barrier model, a kinetic probability, or a causal selectivity mechanism.",
        "",
    ]
    for label, result in (("ee (%)", ee_result), ("log(dr)", dr_result), ("dr raw-scale sensitivity", dr_raw_result)):
        lines.extend([f"## {label}", "", f"Selected descriptors: {', '.join(result['feature_names'])}", f"Evidence: {result['evidence']:.3f}; in-sample RMSE: {result['rmse']:.3f}; in-sample R²: {result['r2_in_sample']:.3f}; fixed-subset LOO RMSE: {result['loo_rmse']:.3f}; fixed-subset LOO R²: {result['loo_r2']:.3f}", f"Nested LOO RMSE: {result['nested_loo_rmse']:.3f}; nested LOO R²: {result['nested_loo_r2']:.3f}; nested 95% coverage: {result['nested_loo_coverage_95']:.3f}", f"Intercept-only LOO RMSE/R²: {result['nested_loo_intercept_rmse']:.3f}/{result['nested_loo_intercept_r2']:.3f}; nested model improves intercept: {result['nested_loo_improves_intercept']}", f"Fixed-subset permutation p(R² ≥ observed): {result['permutation_p_ge_observed']:.4f} (n={result['permutation_count']})", "", "In-sample scores are descriptive. Nested LOO refits descriptor-subset and prior/noise selection inside each held-out fold; it is the primary small-sample predictive diagnostic. The permutation control keeps the full-data selected subset/hyperparameters fixed and is calibration evidence, not a fully nested selection test.", ""])
        if label == "log(dr)":
            lines.append("The dr figure exponentiates the fitted log(dr) predictions and predictive intervals back to the raw major/minor ratio.")
            lines.append("")
        if label == "dr raw-scale sensitivity":
            lines.append("This is a sensitivity model fitted directly to raw dr; it is separate from the primary log(dr) model and is included to test outcome-scale dependence.")
            lines.append("")
        lines.extend(["| Descriptor | Posterior weight | 95% credible interval | Absolute contribution |", "|---|---:|---:|---:|"])
        for weight in result["weights"]:
            interval = weight["credible_interval_95_standardized"]
            lines.append(f"| {weight['feature']} | {weight['posterior_weight_standardized']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] | {weight['absolute_contribution_fraction']:.3f} |")
        lines.append("")
    lines.extend(["## Outputs", "", f"- Inputs: `{OUT / 'anion-level-model-inputs.csv'}`", f"- Nested LOO predictions: `{OUT / 'nested-loo-predictions.csv'}`", f"- Results: `{OUT / 'bayesian-regression-results.json'}`", f"- ee plot: `{OUT / 'ee_vs_weighted_descriptor_score.png'}`", f"- dr plot: `{OUT / 'dr_vs_weighted_descriptor_score.png'}`", f"- raw-dr sensitivity plot: `{OUT / 'dr_raw_sensitivity_vs_weighted_descriptor_score.png'}`", f"- nested LOO validation plot: `{OUT / 'nested_loo_validation.png'}`", ""])
    (OUT / "bayesian-regression-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "feature_count": len(feature_names), "ee_features": ee_result["feature_names"], "dr_features": dr_result["feature_names"], "dr_raw_features": dr_raw_result["feature_names"], "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
