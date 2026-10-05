#!/usr/bin/env python3
"""Selection-aware LOO for the complete Case 4 dynamic feature pool.

The evidence calculation is algebraically equivalent to Case 3f's Gaussian
evidence grid. Diagonalizing each subset Gram matrix once makes the 128-feature
search tractable for every held-out structure.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "03-chemical-validation" / "4"
RESULTS = CASE / "analysis" / "results"
sys.path.insert(0, str(ROOT / "02-dynamic-methodology" / "src"))
sys.path.insert(0, str(ROOT / "03-chemical-validation" / "scripts"))
import run_part3f as case3f  # noqa: E402


def evidence_select(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Select 1-3 features and prior/noise scales on these rows only."""
    n, p = x.shape
    xs, _, _ = case3f.bayes._standardize(x)
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y, ddof=1)) or 1.0
    ys = (y - y_mean) / y_sd
    y_bar = float(np.mean(ys))
    y_centered = ys - y_bar
    y_centered_norm2 = float(y_centered @ y_centered)
    full_gram = xs.T @ xs
    xy = xs.T @ y_centered
    tau_grid = np.geomspace(0.08, 4.0, 12)
    sigma_grid = np.geomspace(0.08, 2.0, 14)
    best: tuple[float, tuple[int, ...], float, float] | None = None

    # Preserve the original search order and first-winner rule for ties.
    for size in (1, 2, 3):
        combinations = np.asarray(list(itertools.combinations(range(p), size)), dtype=np.int32)
        for start in range(0, len(combinations), 50000):
            subsets = combinations[start : start + 50000]
            gram = full_gram[subsets[:, :, None], subsets[:, None, :]]
            eigvals, eigvecs = np.linalg.eigh(gram)
            cross_y = xy[subsets]
            cross_projected2 = np.einsum("mik,mi->mk", eigvecs, cross_y) ** 2
            for tau in tau_grid:
                for sigma in sigma_grid:
                    sigma2 = float(sigma**2)
                    alpha = float(tau**2 / sigma2)
                    denom = 1.0 + alpha * eigvals
                    logdet_beta = np.sum(np.log(denom), axis=1)
                    correction = float(tau**2 / sigma2**2) * np.sum(
                        cross_projected2 / denom, axis=1
                    )
                    energy = (
                        n * y_bar**2 / (sigma2 + 100.0**2 * n)
                        + y_centered_norm2 / sigma2
                        - correction
                    )
                    logdet = (
                        math.log(sigma2 + 100.0**2 * n)
                        + (n - 1) * math.log(sigma2)
                        + logdet_beta
                    )
                    evidence = -0.5 * (energy + logdet + n * math.log(2.0 * math.pi))
                    local_index = int(np.argmax(evidence))
                    value = float(evidence[local_index])
                    if best is None or value > best[0]:
                        best = (
                            value,
                            tuple(int(i) for i in subsets[local_index]),
                            float(tau),
                            float(sigma),
                        )
    if best is None:
        raise RuntimeError("no evidence candidates were evaluated")
    evidence, subset, tau, sigma = best
    return {
        "feature_names": [names[i] for i in subset],
        "evidence": evidence,
        "prior_tau": tau,
        "noise_sigma_standardized": sigma,
    }


def main() -> None:
    start = time.monotonic()
    with (RESULTS / "case4-feature-matrix.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((RESULTS / "case4-feature-pool-manifest.json").read_text())
    saved = json.loads((RESULTS / "case4-bayesian-results.json").read_text())
    names = manifest["variable_full_pool_features"]
    x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    if x.shape != (7, 128):
        raise ValueError(f"unexpected full-pool matrix shape: {x.shape}")

    outcome_names = ("yield_percent", "de_percent", "ee_cis_percent")
    y_by_outcome = {
        outcome: np.asarray([float(row[outcome]) for row in rows], dtype=float)
        for outcome in outcome_names
    }
    verification: dict[str, dict] = {}
    for outcome, y in y_by_outcome.items():
        fitted = evidence_select(x, y, names)
        original = saved["models"][outcome]["full_dynamic_d2_d3_pool"]
        for key in ("prior_tau", "noise_sigma_standardized"):
            if fitted[key] != original[key]:
                raise AssertionError(f"full-data evidence mismatch for {outcome}: {key}")
        if not np.isclose(fitted["evidence"], original["evidence"], atol=1e-7):
            raise AssertionError(f"full-data evidence mismatch for {outcome}: evidence")
        # Some d2 values and their common-reference deltas differ only by a
        # constant, so either name gives identical standardized predictions.
        for held_out in range(len(rows)):
            train_mask = np.arange(len(rows)) != held_out
            predicted, _ = case3f.bayes._posterior_predict(
                x[train_mask], y[train_mask], x[held_out], names, fitted
            )
            if not np.isclose(predicted, original["loo_predicted"][held_out], atol=1e-7):
                raise AssertionError(
                    f"full-data fixed-subset prediction mismatch: {outcome}, row {held_out}"
                )
        verification[outcome] = fitted
        print(f"verified full-data search: {outcome}; elapsed {time.monotonic()-start:.1f}s", flush=True)

    result: dict = {
        "schema": "case4.full-pool-selection-aware-loo.v1",
        "candidate_feature_count": len(names),
        "selection_cap": 3,
        "population_size": len(rows),
        "selection_policy": "Evidence selection of feature subset and prior/noise scales repeated within each six-row training fold",
        "full_data_equivalence_check": verification,
        "outcomes": {},
    }
    prediction_rows: list[dict] = []
    for held_out in range(len(rows)):
        train_mask = np.arange(len(rows)) != held_out
        x_train = x[train_mask]
        x_test = x[held_out]
        for outcome, y in y_by_outcome.items():
            fitted = evidence_select(x_train, y[train_mask], names)
            pred, sd = case3f.bayes._posterior_predict(
                x_train, y[train_mask], x_test, names, fitted
            )
            prediction_rows.append({
                "outcome": outcome,
                "held_out_structure": rows[held_out]["structure_id"],
                "observed": float(y[held_out]),
                "predicted": float(pred),
                "predictive_sd": float(sd),
                "selected_features": ";".join(fitted["feature_names"]),
                "prior_tau": fitted["prior_tau"],
                "noise_sigma_standardized": fitted["noise_sigma_standardized"],
            })
        print(
            f"completed held-out structure {held_out+1}/{len(rows)} "
            f"({rows[held_out]['structure_id']}); elapsed {time.monotonic()-start:.1f}s",
            flush=True,
        )
    for outcome, y in y_by_outcome.items():
        subset = [r for r in prediction_rows if r["outcome"] == outcome]
        predictions = [r["predicted"] for r in subset]
        sds = [r["predictive_sd"] for r in subset]
        metrics = case3f.bayes._summarize_predictions(y, predictions, sds)
        intercept = case3f.bayes._loo_intercept_metrics(y)
        result["outcomes"][outcome] = {
            "nested_loo": metrics,
            "loo_intercept": intercept,
            "selected_feature_frequency": dict(
                Counter(name for row in subset for name in row["selected_features"].split(";"))
            ),
            "selected_features_by_structure": {
                r["held_out_structure"]: r["selected_features"].split(";") for r in subset
            },
        }
    result["runtime_seconds"] = time.monotonic() - start
    (RESULTS / "case4-full-pool-nested-loo.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "case4-full-pool-nested-loo-predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    comparison_rows: list[dict] = []
    for outcome in outcome_names:
        models = saved["models"][outcome]
        fixed = models["full_dynamic_d2_d3_pool"]
        nested = result["outcomes"][outcome]["nested_loo"]
        core = models["d3map_27_feature_core"]
        traditional = models["traditional_case3f_all_15"]
        intercept = result["outcomes"][outcome]["loo_intercept"]
        for model, validation, count, r2, rmse in (
            ("full_dynamic_d2_d3_pool", "fixed_subset_loo", 128, fixed["loo_r2"], fixed["loo_rmse"]),
            ("full_dynamic_d2_d3_pool", "selection_aware_loo", 128, nested["r2"], nested["rmse"]),
            ("d3map_27_feature_core", "selection_aware_loo", 27, core["nested_loo_r2"], core["nested_loo_rmse"]),
            ("traditional_case3f_all_15", "fold_local_tuning_all_15", 15, traditional["nested_loo_r2"], traditional["nested_loo_rmse"]),
            ("intercept_only", "loo_mean_of_six", 0, intercept["loo_intercept_r2"], intercept["loo_intercept_rmse"]),
        ):
            comparison_rows.append({
                "outcome": outcome,
                "model": model,
                "validation": validation,
                "candidate_features": count,
                "r2": float(r2),
                "rmse": float(rmse),
            })
    with (RESULTS / "case4-validation-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps(result["outcomes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
