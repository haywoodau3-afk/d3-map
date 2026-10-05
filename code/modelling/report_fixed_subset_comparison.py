#!/usr/bin/env python3
"""Report the same fixed-subset LOO diagnostic for every Case 4 model."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "03-chemical-validation" / "4" / "analysis" / "results"
MODEL_ORDER = (
    ("full_dynamic_d2_d3_pool", "Full dynamic\n128 candidates"),
    ("d3map_27_feature_core", "Focused dynamic\n27 candidates"),
    ("traditional_case3f_selected_1_to_3", "Traditional\nselected from 15"),
    ("traditional_case3f_all_15", "Traditional\nall 15"),
)
OUTCOMES = (
    ("yield_percent", "Yield"),
    ("de_percent", "de"),
    ("ee_cis_percent", "cis-ee"),
)
COLORS = ("#7452a3", "#277f8e", "#ce7928", "#aa4454")


def main() -> None:
    bayesian = json.loads((RESULTS / "case4-bayesian-results.json").read_text())
    with (RESULTS / "case4-feature-matrix.csv").open(newline="", encoding="utf-8") as stream:
        structures = list(csv.DictReader(stream))
    comparison: list[dict] = []
    predictions: list[dict] = []
    for outcome, outcome_label in OUTCOMES:
        observed = np.asarray([float(row[outcome]) for row in structures])
        fig, axes = plt.subplots(1, 4, figsize=(17.8, 4.9), sharex=True, sharey=True)
        all_preds = [np.asarray(bayesian["models"][outcome][name]["loo_predicted"])
                     for name, _ in MODEL_ORDER]
        lower = min(float(observed.min()), *(float(v.min()) for v in all_preds))
        upper = max(float(observed.max()), *(float(v.max()) for v in all_preds))
        pad = max(2.0, 0.09 * (upper - lower))
        lower, upper = lower - pad, upper + pad
        for ax, (name, display_name), color in zip(axes, MODEL_ORDER, COLORS, strict=True):
            fit = bayesian["models"][outcome][name]
            predicted = np.asarray(fit["loo_predicted"], dtype=float)
            rmse = float(np.sqrt(np.mean((observed - predicted)**2)))
            sst = float(np.sum((observed - observed.mean())**2))
            r2 = float(1 - np.sum((observed - predicted)**2) / sst)
            if not math.isclose(rmse, fit["loo_rmse"], rel_tol=1e-8):
                raise AssertionError(f"RMSE mismatch: {outcome}, {name}")
            if not math.isclose(r2, fit["loo_r2"], rel_tol=1e-8):
                raise AssertionError(f"R² mismatch: {outcome}, {name}")
            comparison.append({
                "outcome": outcome,
                "model": name,
                "candidate_features": fit["candidate_feature_count"],
                "fixed_selected_features": ";".join(fit["feature_names"]),
                "r2_in_sample": fit["r2_in_sample"],
                "fixed_subset_loo_r2": r2,
                "fixed_subset_loo_rmse": rmse,
                "selection_and_hyperparameters": "chosen using all seven outcomes",
            })
            for row, yhat in zip(structures, predicted, strict=True):
                predictions.append({
                    "outcome": outcome,
                    "model": name,
                    "held_out_structure": row["structure_id"],
                    "observed": float(row[outcome]),
                    "fixed_subset_loo_predicted": float(yhat),
                })
            ax.plot([lower, upper], [lower, upper], color="#9b9b9b", linewidth=1, linestyle="--")
            ax.scatter(observed, predicted, facecolor="white", edgecolor=color, linewidth=1.8, s=55)
            for row, x_val, y_val in zip(structures, observed, predicted, strict=True):
                ax.annotate(str(row["structure_number"]), (x_val, y_val),
                            xytext=(4, 3), textcoords="offset points", fontsize=7)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{display_name}\n$R^2$ = {r2:.3f} · RMSE = {rmse:.2f}", fontsize=10)
            ax.set_xlabel(f"Observed {outcome_label} (%)", fontsize=9)
            ax.grid(alpha=0.17)
        axes[0].set_ylabel(f"Fixed-subset LOO prediction (%)", fontsize=9)
        fig.suptitle(f"Case 4 · {outcome_label}: consistent fixed-subset comparison", fontsize=14)
        fig.text(0.5, 0.014,
                 "Selected features and prior/noise settings use all seven outcomes; LOO scores are descriptive diagnostics.",
                 ha="center", fontsize=8.5, color="#575757")
        fig.subplots_adjust(left=0.055, right=0.99, top=0.78, bottom=0.17, wspace=0.14)
        figure_path = RESULTS / f"case4-fixed-subset-{outcome}.png"
        fig.savefig(figure_path, dpi=210)
        plt.close(fig)
    for path, data in (
        (RESULTS / "case4-fixed-subset-comparison.csv", comparison),
        (RESULTS / "case4-fixed-subset-predictions.csv", predictions),
    ):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved {len(comparison)} model comparisons and {len(predictions)} predictions")
    for row in comparison:
        print(row["outcome"], row["model"],
              f"R²={row['fixed_subset_loo_r2']:.3f}",
              f"RMSE={row['fixed_subset_loo_rmse']:.2f}")


if __name__ == "__main__":
    main()
