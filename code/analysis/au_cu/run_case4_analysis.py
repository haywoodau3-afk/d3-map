#!/usr/bin/env python3
"""Build Case 4 xTB-metadynamics d2/d3 maps and fit Bayesian comparisons.

The D3 reference is a constructed AuCu rotaxane carrying the shared
15-atom Au-carbene core, capped with H at the variable acyl position (R=H).
Its provenance and limits are recorded in the output manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "03-chemical-validation" / "4"
SCRIPTS = ROOT / "03-chemical-validation" / "scripts"
sys.path.insert(0, str(ROOT / "02-dynamic-methodology" / "src"))
sys.path.insert(0, str(SCRIPTS))

from dmap.io import read_xyz_ensemble
from dmap.release import (
    compare_release_outputs,
    create_release_project,
    default_structure_setup,
    execute_release_project,
    read_release_project,
    validate_pair_compatibility,
    write_portable_zip,
)
import run_part3f as case3f


FULL = CASE / "structures" / "full-complexes"
SAMPLE_ROOT = CASE / "sampling" / "runs"
ANALYSIS = CASE / "analysis"
MAPS = ANALYSIS / "maps"
INPUTS = ANALYSIS / "inputs"
OUT = ANALYSIS / "results"
XTB = ROOT / ".micromamba" / "part3c" / "bin" / "xtb"
CREST = ROOT / ".micromamba" / "part3c-crest212" / "bin" / "crest"
UNITS = [
    ("benzoate", "Benzoate", 9, 86.0, "94:6", "78.5:21.5", "62:38"),
    ("pivalate", "Pivalate", 13, 90.0, "97:3", "55:45", ""),
    ("phenylacetate", "Phenylacetate", 14, 40.0, "97:3", "77:23", ""),
    ("para-CF3-benzoate", "4-CF3 benzoate", 15, 45.0, "96:4", "73:27", ""),
    ("para-OMe-benzoate", "4-OMe benzoate", 16, 48.0, "94:6", "71:29", "73:27"),
    ("para-tBu-benzoate", "4-tBu benzoate", 17, 79.0, "95:5", "87:13", "65:35"),
    ("35-di-tBu-benzoate", "3,5-di-tBu benzoate", 18, 68.0, "96:4", "89:11", "70:30"),
]
STRUCTURE_NUMBERS = {
    "benzoate": 1,
    "pivalate": 2,
    "phenylacetate": 3,
    "para-CF3-benzoate": 4,
    "para-OMe-benzoate": 5,
    "para-tBu-benzoate": 6,
    "35-di-tBu-benzoate": 7,
}
OUTCOME_TABLES = {
    "benzoate": "S4",
    "pivalate": "S8",
    "phenylacetate": "S9",
    "para-CF3-benzoate": "S10",
    "para-OMe-benzoate": "S11",
    "para-tBu-benzoate": "S12",
    "35-di-tBu-benzoate": "S13",
}
AU, P, CU, CU_N, CARBENE = 2, 1, 73, (26, 77, 78), 148
ALIGNMENT = [AU, CARBENE, P, CU, *CU_N]
PROTECTED = [[AU, CARBENE], [AU, P], *[[CU, n] for n in CU_N]]
EXCLUDED_FIELDS = {
    "d2_feature_schema", "d2_variability_status", "d2_probability_field_v2",
    "d2_coordination_sectors", "d2_coordination_distribution",
    "d2_source_count", "d3_source_count",
}
TRADITIONAL_FEATURES = [
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
TRADITIONAL_DESCRIPTOR_CSV = OUT / "case4-traditional-per-structure.csv"
TRADITIONAL_DESCRIPTOR_MANIFEST = OUT / "case4-traditional-descriptor-manifest.json"


def read_xyz_rows(path: Path) -> tuple[list[str], np.ndarray]:
    data = read_xyz_ensemble(path)
    return list(data.elements), data.geometries[0].coordinates.copy()


def write_xyz(path: Path, elements: list[str], coords: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(elements)), comment]
    lines += [f"{e:<3s} {x: .10f} {y: .10f} {z: .10f}" for e, (x, y, z) in zip(elements, coords, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_common_reference() -> Path:
    """Retain the shared carbene core and replace the variable acyl R with H."""
    path = INPUTS / "common-core-R-H-aucu-reference.xyz"
    if path.is_file():
        return path
    source = FULL / "cu-au-carbene-benzoate.xyz"
    atom_map = list(csv.DictReader(source.with_suffix(".atom-map.csv").open(encoding="utf-8")))
    elements, coords = read_xyz_rows(source)
    core_indices = [i for i, row in enumerate(atom_map) if row["role"] == "common_carbene_core"]
    tail_indices = [i for i, row in enumerate(atom_map) if row["role"] == "acyl_substituent"]
    if len(core_indices) != 15 or core_indices != list(range(146, 161)):
        raise ValueError(f"unexpected common-core atom mapping: {core_indices}")
    # The carbonyl carbon is the common-core carbon bonded to the first
    # variable acyl atom.  Cap that bond with a 1.09-Angstrom formyl H.
    attachment: tuple[int, int, float] | None = None
    for i in core_indices:
        if elements[i] != "C":
            continue
        for j in tail_indices:
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if 1.15 <= distance <= 1.75 and (attachment is None or distance < attachment[2]):
                attachment = (i, j, distance)
    if attachment is None:
        raise ValueError("could not identify the common-core/acyl attachment")
    acyl_carbon, substituent_atom, _ = attachment
    cap = coords[acyl_carbon] + 1.09 * (coords[substituent_atom] - coords[acyl_carbon]) / np.linalg.norm(coords[substituent_atom] - coords[acyl_carbon])
    base_indices = list(range(146)) + core_indices
    base_elements = [elements[i] for i in base_indices] + ["H"]
    base_coords = np.vstack([coords[base_indices], cap])
    write_xyz(path, base_elements, base_coords, "Case 4 matched D3 reference; common carbene core; R=H; constructed, unoptimized seed; charge +2; singlet")
    metadata = {
        "schema": "case4.common-core-reference.v1",
        "reference": str(path.relative_to(CASE)),
        "status": "constructed_common_core_R_equals_H_reference",
        "source_template": str(source.relative_to(CASE)),
        "construction": "retained the shared 146-atom AuCu scaffold and the identical 15-atom C6H7O2 carbene core; replaced the variable acyl substituent bond with H at 1.09 A; xTB optimization and metadynamics sampling follow",
        "charge": 2,
        "multiplicity": 1,
        "atom_count": len(base_elements),
        "carbene_carbon_index_1based": CARBENE,
        "capped_core_atom_index_1based": acyl_carbon + 1,
        "source_substituent_atom_index_1based": substituent_atom + 1,
        "seed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scientific_boundary": "methodological common-core reference, not an experimentally reported catalyst or isolated intermediate",
    }
    (INPUTS / "common-core-reference-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_reference_ensemble() -> Path:
    stage = SAMPLE_ROOT / "common-core-R-H-reference" / "sampling"
    trajectory = stage / "crest_dynamics.trj"
    optimized = stage / "preoptimized.xyz"
    if trajectory.is_file() and optimized.is_file():
        _write_reference_sampling_manifest(trajectory, optimized)
        return trajectory
    raise FileNotFoundError(
        "The completed xTB metadynamics trajectory and optimized reference are required at "
        f"{stage}; this analysis driver reuses the approved xTB data and will not launch CREST."
    )


def _write_reference_sampling_manifest(trajectory: Path, optimized: Path) -> None:
    manifest_path = INPUTS / "common-core-reference-manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = trajectory.parent / "sampling-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata.update({
        "status": "xTB_optimized_with_completed_metadynamics_trajectory",
        "construction": "retained the shared 146-atom AuCu scaffold and identical 15-atom C6H7O2 carbene core; replaced the variable acyl substituent with H at 1.09 A; the capped structure was xTB optimized and sampled as a uniformly weighted metadynamics trajectory",
        "optimized_geometry": str(optimized.relative_to(CASE)),
        "sampling_trajectory": str(trajectory.relative_to(CASE)),
        "sampling_summary": str(summary_path.relative_to(CASE)),
        "sampling": summary,
    })
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def setup_for(input_path: Path) -> dict[str, Any]:
    elements, coords = read_xyz_rows(input_path)
    center = coords[AU - 1]
    reactive = coords[CARBENE - 1] - center
    secondary = coords[P - 1] - center
    secondary = secondary - reactive * (float(secondary @ reactive) / float(reactive @ reactive))
    reactive = reactive / np.linalg.norm(reactive)
    secondary = secondary / np.linalg.norm(secondary)
    steric = [i + 1 for i, el in enumerate(elements) if el != "H" and i + 1 != AU]
    setup = default_structure_setup(
        input_path,
        center_atom=AU,
        reactive_direction=reactive.tolist(),
        secondary_direction=secondary.tolist(),
        alignment_atoms=ALIGNMENT,
        charge=2,
        multiplicity=1,
        solvent="chcl3",
        confirmed=True,
        protected_contacts=PROTECTED,
        frozen_atoms=[AU],
        steric_atoms=steric,
        xtb_executable=str(XTB),
        crest_executable=str(CREST),
        threads=4,
    )
    # These are already-sampled, energy-annotated xTB trajectories. Treat them
    # as time-series geometry ensembles and weight frames uniformly.
    setup["analysis_mode"] = "trajectory"
    setup["population"] = {"model": "uniform"}
    # The field-radii profile has no transition-metal defaults. Use the
    # release's SambVca 2.1 Cu radius explicitly for the second metal rather
    # than silently dropping Cu from the full-complex steric atom set.
    setup["radii_overrides"] = {"Cu": 1.64}
    setup["frame_semantics"] = "right-handed:+z-toward-bound-carbene:+x-projected-Au-to-P"
    setup["chemical_setup_status"] = "Au-centred Au(I)-carbene; +2 singlet CuAu complex; xTB metadynamics trajectory; Au-C, Au-P and Cu-N contacts protected; Au frozen; no CREST ensemble used"
    return setup


def create_or_run_d2(name: str, trajectory: Path, template: Path) -> Path:
    project = MAPS / "d2-map" / f"{name}-d2map" / "project.d3map.json"
    if project.is_file():
        state = project.parent / "run-state.json"
        if state.is_file() and json.loads(state.read_text()).get("status") == "success":
            return project
    else:
        project = create_release_project(
            kind="d2-map", reference_xyz=trajectory,
            output_directory=project.parent,
            reference_setup=setup_for(template),
            shared=case3f.SHARED_PROFILE,
        )
    config_path = project
    config = read_release_project(config_path)
    config["structures"]["reference"]["analysis_mode"] = "trajectory"
    config["structures"]["reference"]["population"] = {"model": "uniform"}
    config["structures"]["reference"]["radii_overrides"] = {"Cu": 1.64}
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Starting D2 map analysis: {name}", flush=True)
    execute_release_project(config_path, sampling=False)
    print(f"D2 map analysis complete: {name}", flush=True)
    return config_path


def d3_comparison_features(path: Path) -> dict[str, float]:
    return case3f._comparison_features(path)


def create_static_d3(name: str, reference_d2: Path, extended_d2: Path) -> Path:
    """Build Case 3d's single-geometry d3 comparison from cached input branches."""
    project = MAPS / "case3d-static-d3" / f"common-core-R-H-vs-{name}-case3d-static-d3map" / "project.d3map.json"
    reference_input = reference_d2.parent / "stages/reference/baselines/input/input.xyz"
    extended_input = extended_d2.parent / "stages/reference/baselines/input/input.xyz"
    if not reference_input.is_file() or not extended_input.is_file():
        raise FileNotFoundError("single-input-geometry d2 baseline is missing")
    if not project.is_file():
        project = create_release_project(
            kind="d3-map", reference_xyz=reference_input,
            extended_xyz=extended_input,
            output_directory=project.parent,
            reference_setup=setup_for(reference_input),
            extended_setup=setup_for(extended_input),
            shared=case3f.SHARED_PROFILE,
        )
    state_path = project.parent / "run-state.json"
    if state_path.is_file() and json.loads(state_path.read_text()).get("status") == "success":
        return project
    config = read_release_project(project)
    root = project.parent
    for role, source_project in (("reference", reference_d2), ("extended", extended_d2)):
        source_root = source_project.parent
        source_results = source_root / "results/reference/baselines/input"
        source_stage = source_root / "stages/reference/baselines/input"
        if not (source_results / "descriptors.json").is_file():
            raise RuntimeError(f"static d2 input baseline missing for {role}: {source_project}")
        shutil.copytree(source_results, root / "results" / role, dirs_exist_ok=True)
        shutil.copytree(source_stage, root / "stages" / role, dirs_exist_ok=True)
    mismatches = validate_pair_compatibility(root, config)
    compare_release_outputs(project, config, mismatches)
    (root / "run-state.json").write_text(json.dumps({
        "schema": "case4.case3d-static-d3.run-state.v1",
        "status": "success",
        "sampling_strategy": "cached single-input-geometry d2 branches; no CREST averaging",
        "comparison_status": "standard" if not mismatches else "nonstandard",
        "compatibility_mismatches": mismatches,
        "reference": "common-core-R-H-reference",
        "extended": name,
        "difference_definition": "single-input full-complex minus common-core reference",
    }, indent=2) + "\n", encoding="utf-8")
    write_portable_zip(project)
    return project


def run_maps() -> dict[str, Path]:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    reference_xyz = build_common_reference()
    reference_trajectory = ensure_reference_ensemble()
    reference_template = SAMPLE_ROOT / "common-core-R-H-reference" / "sampling" / "preoptimized.xyz"
    reference_d2 = create_or_run_d2("common-core-R-H-reference", reference_trajectory, reference_template)
    (MAPS / "d2-map").mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    static_outputs: dict[str, Path] = {}
    for ident, *_ in UNITS:
        sample_dir = SAMPLE_ROOT / ident / "sampling"
        trajectory = sample_dir / "crest_dynamics.trj"
        template = sample_dir / "preoptimized.xyz"
        if not trajectory.is_file() or not template.is_file():
            raise FileNotFoundError(f"completed xTB metadynamics artifacts missing for {ident}")
        full_d2 = create_or_run_d2(ident, trajectory, template)
        print(f"Building dynamic D3 comparison: {ident}", flush=True)
        project = MAPS / "d3-map" / f"common-core-R-H-vs-{ident}-d3map" / "project.d3map.json"
        if not project.is_file():
            project = create_release_project(
                kind="d3-map",
                reference_xyz=reference_trajectory,
                extended_xyz=trajectory,
                output_directory=project.parent,
                reference_setup=setup_for(reference_template),
                extended_setup=setup_for(template),
                shared=case3f.SHARED_PROFILE,
            )
        state_path = project.parent / "run-state.json"
        if not (state_path.is_file() and json.loads(state_path.read_text()).get("status") == "success"):
            config = read_release_project(project)
            root = project.parent
            for role, source_project in (("reference", reference_d2), ("extended", full_d2)):
                source_root = source_project.parent
                source_results = source_root / "results" / "reference"
                source_stages = source_root / "stages" / "reference"
                if not (source_results / "descriptors.json").is_file():
                    raise RuntimeError(f"completed d2-map output is missing for {role}: {source_project}")
                shutil.copytree(source_results, root / "results" / role, dirs_exist_ok=True)
                if source_stages.is_dir():
                    shutil.copytree(source_stages, root / "stages" / role, dirs_exist_ok=True)
            mismatches = validate_pair_compatibility(root, config)
            compare_release_outputs(project, config, mismatches)
            (root / "run-state.json").write_text(json.dumps({
                "schema": "case4.d3map.run-state.v1", "status": "success",
                "sampling_strategy": "reused completed, uniformly weighted xTB metadynamics trajectories through cached d2 branches",
                "comparison_status": "standard" if not mismatches else "nonstandard",
                "compatibility_mismatches": mismatches,
                "reference": "common-core-R-H-reference",
                "extended": ident,
                "difference_definition": "selected full AuCu-carbene complex minus common-core R=H AuCu-carbene reference",
            }, indent=2) + "\n", encoding="utf-8")
            write_portable_zip(project)
        outputs[ident] = project
        print(f"Building single-geometry static D3 comparison: {ident}", flush=True)
        static_outputs[ident] = create_static_d3(ident, reference_d2, full_d2)
        print(f"D3 comparison complete: {ident}", flush=True)
    (ANALYSIS / "map-index.json").write_text(json.dumps({
        "schema": "case4.map-index.v1",
        "status": "maps_completed",
        "reference": str(reference_xyz.relative_to(CASE)),
        "reference_sampling": str(reference_trajectory.relative_to(CASE)),
        "d2_projects": {p.parent.name.removesuffix("-d2map"): str(p.relative_to(CASE)) for p in sorted((MAPS / "d2-map").glob("*/project.d3map.json"))},
        "d3_projects": {name: str(path.relative_to(CASE)) for name, path in outputs.items()},
        "static_d3_projects": {name: str(path.relative_to(CASE)) for name, path in static_outputs.items()},
        "profile": case3f.SHARED_PROFILE,
    }, indent=2) + "\n", encoding="utf-8")
    return outputs


def read_feature_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected one feature row from {path}, found {len(rows)}")
    return rows[0]


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_feature_matrix() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_project = MAPS / "d2-map" / "common-core-R-H-reference-d2map" / "project.d3map.json"
    if not (ref_project.parent / "results/reference/features.csv").is_file():
        raise FileNotFoundError("complete d2/d3 maps before feature fitting")
    if not TRADITIONAL_DESCRIPTOR_CSV.is_file() or not TRADITIONAL_DESCRIPTOR_MANIFEST.is_file():
        raise FileNotFoundError("generate the Case 3f DBSTEP/Morfeus descriptors before fitting")
    with TRADITIONAL_DESCRIPTOR_CSV.open(newline="", encoding="utf-8") as stream:
        conventional_rows = {row["structure_id"]: row for row in csv.DictReader(stream)}
    traditional_manifest = json.loads(TRADITIONAL_DESCRIPTOR_MANIFEST.read_text(encoding="utf-8"))
    if set(conventional_rows) != {ident for ident, *_ in UNITS}:
        raise ValueError("Case 3f traditional descriptor table does not contain exactly the seven Case 4 structures")
    if traditional_manifest.get("case3f_complete_traditional_pool") != TRADITIONAL_FEATURES:
        raise ValueError("Case 3f traditional descriptor feature schema does not match the expected 15-feature pool")
    ref = read_feature_row(ref_project.parent / "results/reference/features.csv")
    rows: list[dict[str, Any]] = []
    for ident, name, product, yld, dr, ercis, ertrans in UNITS:
        sample_dir = SAMPLE_ROOT / ident / "sampling"
        full_project = MAPS / "d2-map" / f"{ident}-d2map" / "project.d3map.json"
        full = read_feature_row(full_project.parent / "results/reference/features.csv")
        traditional_external = conventional_rows[ident]
        d3_project = MAPS / "d3-map" / f"common-core-R-H-vs-{ident}-d3map"
        comparison = d3_comparison_features(d3_project)
        static_d2 = case3f._case3d_static_d2_values(full_project)
        sampling_summary_path = sample_dir / "sampling-summary.json"
        if sampling_summary_path.is_file():
            sampling_summary = json.loads(sampling_summary_path.read_text(encoding="utf-8"))
        else:
            sampling_manifest = json.loads((CASE / "sampling/runs/run-manifest.json").read_text(encoding="utf-8"))
            sampling_record = next(item for item in sampling_manifest["structures"] if item["id"] == ident)
            sampling_summary = {
                "analysis_mode": sampling_record.get("analysis_mode", "trajectory"),
                "frame_count": sampling_record.get("frame_count", sampling_record.get("conformers")),
                "sampling_note": "Uniformly weighted xTB metadynamics trajectory; no CREST conformer ensemble or Boltzmann population used.",
            }
        row: dict[str, Any] = {
            "structure_number": STRUCTURE_NUMBERS[ident],
            "structure_id": ident, "catalyst": name, "product": product,
            "yield_percent": yld,
            "de_percent": _ratio_excess(dr),
            "ee_cis_percent": _ratio_excess(ercis),
            "ee_trans_percent": _ratio_excess(ertrans) if ertrans else "",
            "dr_major_minor": dr, "er_cis_major_minor": ercis,
            "er_trans_major_minor": ertrans,
            "outcome_source": f"Heard & Goldup Chem 2020 SI Table {OUTCOME_TABLES[ident]}, [Au((Rmp)-6)(Cl)] catalyst row",
            "analysis_mode": sampling_summary["analysis_mode"],
            "sampling_frame_count": sampling_summary["frame_count"],
            "sampling_description": "Uniformly weighted xTB metadynamics trajectory; no CREST conformer ensemble or Boltzmann population used.",
        }
        for key in full:
            val = finite(full.get(key))
            if val is not None and key not in EXCLUDED_FIELDS:
                row[f"d2_{key}"] = val
        for key in ref:
            left, right = finite(ref.get(key)), finite(full.get(key))
            if left is not None and right is not None and key not in EXCLUDED_FIELDS:
                row[f"d3_d2_delta_{key}"] = right - left
        row.update(comparison)
        traditional_values = {
            "static_vbur_percent": static_d2.get("vbur_percent"),
            "static_g_percent": static_d2.get("g_percent"),
            **{name: traditional_external.get(name) for name in TRADITIONAL_FEATURES[2:]},
        }
        for feature in TRADITIONAL_FEATURES:
            value = traditional_values[feature]
            if finite(value) is None:
                raise ValueError(f"{ident}: missing Case 3f traditional descriptor {feature}")
            row[f"traditional_{feature}"] = float(value)
        rows.append(row)
    all_features = sorted({k for row in rows for k in row if k.startswith(("d2_", "d3_"))})
    complete = [feature for feature in all_features if all(finite(row.get(feature)) is not None for row in rows)]
    variable = [feature for feature in complete if np.std([float(row[feature]) for row in rows], ddof=1) > 1e-12]
    case3f_core = json.loads((case3f.OUT / "feature-pool-manifest.json").read_text())["candidate_features"]
    core_complete = [feature for feature in case3f_core if feature in complete]
    core_variable = [feature for feature in core_complete if feature in variable]
    metadata = {
        "schema": "case4.feature-pool.v1",
        "unit_of_analysis": "one of seven carbene/catalyst units with the matched experimental cyclopropane product",
        "row_count": len(rows),
        "all_numeric_features": all_features,
        "complete_features": complete,
        "variable_full_pool_features": variable,
        "full_pool_feature_count": len(variable),
        "case3f_27_feature_schema": case3f_core,
        "case3f_core_complete_in_case4": core_complete,
        "case3f_core_variable_in_case4": core_variable,
        "case3f_core_missing": sorted(set(case3f_core) - set(core_complete)),
        "case3f_core_constant": sorted(set(core_complete) - set(core_variable)),
        "traditional_case3f_schema": TRADITIONAL_FEATURES,
        "traditional_case3f_feature_count": len(TRADITIONAL_FEATURES),
        "traditional_case3f_definition": "Exact Case 3f 15-feature conventional descriptor pool: static %Vbur/Solid-G plus DBSTEP and Morfeus descriptors evaluated on each complex's single xTB-optimized D2 input geometry; external descriptors are not averaged over the metadynamics trajectory",
        "traditional_descriptor_manifest": traditional_manifest,
        "traditional_descriptor_software": traditional_manifest.get("packages"),
        "traditional_descriptor_geometry_policy": traditional_manifest.get("geometry_policy"),
        "traditional_descriptor_axis_policy": traditional_manifest.get("axis_policy"),
        "traditional_descriptor_settings": traditional_manifest.get("descriptor_settings"),
        "case3d_static_map_schema_retained_as_model": False,
        "feature_origin_boundary": "The 27-feature and full feature pools come from uniformly weighted xTB metadynamics-based d2/d3 maps; the Case 3f traditional pool is calculated on one static xTB-optimized geometry per complex, with no trajectory averaging.",
        "feature_policy": "complete, variable numeric full-complex d2 fields, matched full-minus-common-core d2 deltas, and direct d3-map comparisons; the Case 3f 27-feature schema is retained with missing/constant members listed explicitly",
        "reference": json.loads((INPUTS / "common-core-reference-manifest.json").read_text()),
        "small_sample_boundary": "n=7; exploratory; no causal or general predictive claim",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    traditional_columns = [f"traditional_{name}" for name in TRADITIONAL_FEATURES]
    fields = ["structure_number", "structure_id", "catalyst", "product", "yield_percent", "de_percent", "ee_cis_percent", "ee_trans_percent", "dr_major_minor", "er_cis_major_minor", "er_trans_major_minor", "outcome_source", "analysis_mode", "sampling_frame_count", "sampling_description", *traditional_columns, *variable]
    with (OUT / "case4-feature-matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (OUT / "case4-feature-pool-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return rows, metadata


def _ratio_excess(ratio: str) -> float:
    parts = [float(x) for x in ratio.replace(" ", "").split(":")]
    if len(parts) != 2 or sum(parts) <= 0:
        raise ValueError(f"invalid experimental ratio: {ratio}")
    return 100.0 * abs(parts[0] - parts[1]) / sum(parts)


def _fit_all_traditional_features(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Case 3f-style all-15 sensitivity fit with fold-local evidence tuning."""
    count = len(names)
    fitted = case3f._fit_evidence_fast(
        x, y, names, include_diagnostics=False, min_size=count, max_size=count
    )
    fitted.update(case3f.bayes._loo_metrics(x, y, names, fitted))
    predictions: list[float] = []
    predictive_sd: list[float] = []
    for held_out in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[held_out] = False
        fold = case3f._fit_evidence_fast(
            x[train_mask], y[train_mask], names,
            include_diagnostics=False, min_size=count, max_size=count,
        )
        prediction, sd = case3f.bayes._posterior_predict(
            x[train_mask], y[train_mask], x[held_out], names, fold
        )
        predictions.append(float(prediction))
        predictive_sd.append(float(sd))
    metrics = case3f.bayes._summarize_predictions(y, predictions, predictive_sd)
    fitted.update({
        "nested_loo_predicted": predictions,
        "nested_loo_predictive_sd": predictive_sd,
        "nested_loo_rmse": metrics["rmse"],
        "nested_loo_r2": metrics["r2"],
        "nested_loo_coverage_95": metrics["coverage_95"],
        "candidate_feature_count": count,
        "candidate_features": names,
        "selection_policy": "fixed all-15 traditional descriptor set; standardization and prior/noise evidence selection refit within each LOO training fold; no feature subset selection",
    })
    return fitted


def fit_models(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Fit the dynamic-map and Case 3f traditional models for each target."""
    case3f.bayes._fit_evidence = case3f._fit_evidence_fast
    core_names = metadata["case3f_core_variable_in_case4"]
    full_names = metadata["variable_full_pool_features"]
    traditional_names = [f"traditional_{name}" for name in TRADITIONAL_FEATURES]
    model_features = {
        "d3map_27_feature_core": core_names,
        "full_dynamic_d2_d3_pool": full_names,
        "traditional_case3f_selected_1_to_3": traditional_names,
        "traditional_case3f_all_15": traditional_names,
    }
    outcomes = {
        "yield_percent": [float(row["yield_percent"]) for row in rows],
        "de_percent": [float(row["de_percent"]) for row in rows],
        "ee_cis_percent": [float(row["ee_cis_percent"]) for row in rows],
    }
    results: dict[str, Any] = {
        "schema": "case4.bayesian-comparison.v1",
        "status": "completed_exploratory_three_outcome_comparison",
        "unit_of_analysis": f"one matched experimental carbene/product row; n={len(rows)}",
        "reference": "constructed common-core AuCu carbene (R=H); see analysis/inputs/common-core-reference-manifest.json",
        "outcomes": {},
        "feature_policy": metadata,
        "models": {},
        "scientific_boundary": f"All fits are exploratory associations over {len(rows)} distinct carbene/substrate rows. Leave-one-out predictions are uncertain at this sample size and do not establish general predictive performance or causality.",
        "validation": {
            "selection_cap": 3,
            "dynamic_core_and_selected_traditional": "nested leave-one-out; feature subset, standardization, and prior/noise evidence selection repeated in each training fold",
            "all_15_traditional": "all 15 Case 3f descriptors fixed; standardization and prior/noise evidence selection repeated in each training fold",
            "full_dynamic_pool": "full-data evidence selects up to three features; fixed-subset leave-one-out only; no nested high-dimensional feature search",
            "permutation_controls": "2,000 outcome shuffles for dynamic core and selected-traditional models; all-15 and full-pool sensitivity models omit shuffles",
        },
        "files": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for outcome_name, y_values in outcomes.items():
        y = np.asarray(y_values, dtype=float)
        results["outcomes"][outcome_name] = {
            "values_by_structure": {row["structure_id"]: float(row[outcome_name]) for row in rows},
            "source": "SI Tables S4 and S8-S13, [Au((Rmp)-6)(Cl)] catalyst row; yield as reported; de from cis:trans dr; ee from cis er",
            "unit": "percent",
        }
        results["models"][outcome_name] = {}
        predictions_for_plot: dict[str, dict[str, Any]] = {}
        for model_name, names in model_features.items():
            if not names:
                raise ValueError(f"no available candidate features for {model_name}")
            x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
            if model_name == "full_dynamic_d2_d3_pool":
                fit = case3f._fit_evidence_fast(x, y, names, include_diagnostics=False)
                fit.update(case3f.bayes._loo_metrics(x, y, names, fit))
                fit.update(case3f.bayes._loo_intercept_metrics(y))
                fit["loo_improves_intercept"] = bool(fit["loo_rmse"] < fit["loo_intercept_rmse"])
                fit["validation_note"] = "full dynamic d2/d3 feature pool; fixed-subset LOO only"
                loo_pred = fit["loo_predicted"]
                validation_r2 = fit["loo_r2"]
                validation_rmse = fit["loo_rmse"]
                validation_label = "fixed LOO"
            elif model_name == "traditional_case3f_all_15":
                fit = _fit_all_traditional_features(x, y, names)
                loo_pred = fit["nested_loo_predicted"]
                validation_r2 = fit["nested_loo_r2"]
                validation_rmse = fit["nested_loo_rmse"]
                validation_label = "nested LOO; all 15 fixed"
            else:
                fit = case3f._fit_evidence_fast(x, y, names, include_diagnostics=True)
                loo_pred = fit["nested_loo_predicted"]
                validation_r2 = fit["nested_loo_r2"]
                validation_rmse = fit["nested_loo_rmse"]
                validation_label = "nested LOO"
            fit["candidate_feature_count"] = len(names)
            fit["candidate_features"] = names
            fit["validation_label"] = validation_label
            results["models"][outcome_name][model_name] = fit
            predictions_for_plot[model_name] = {"fit": fit, "predictions": loo_pred}
            metric_rows.append({
                "outcome": outcome_name,
                "model": model_name,
                "candidate_features": len(names),
                "selected_features": ";".join(fit["feature_names"]),
                "in_sample_rmse": fit["rmse"],
                "in_sample_r2": fit["r2_in_sample"],
                "validation": validation_label,
                "validation_rmse": validation_rmse,
                "validation_r2": validation_r2,
                "validation_coverage_95": fit.get("nested_loo_coverage_95", fit.get("loo_coverage_95")),
            })
            for index, row in enumerate(rows):
                prediction_rows.append({
                    "outcome": outcome_name, "model": model_name,
                    "structure_number": row["structure_number"],
                    "structure_id": row["structure_id"], "product": row["product"],
                    "observed": y[index], "loo_prediction": loo_pred[index],
                    "validation": validation_label,
                })
        _plot_outcome(outcome_name, y, [row["product"] for row in rows], predictions_for_plot,
                      OUT / f"case4_bayesian_comparison_{outcome_name}.png")
    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / "case4-bayesian-results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_csv(OUT / "case4-model-comparison.csv", metric_rows)
    _write_csv(OUT / "case4-loo-predictions.csv", prediction_rows)
    report = _write_report(results, rows, metadata)
    results["files"] = {
        "feature_matrix": str(OUT / "case4-feature-matrix.csv"),
        "feature_pool_manifest": str(OUT / "case4-feature-pool-manifest.json"),
        "model_metrics": str(OUT / "case4-model-comparison.csv"),
        "loo_predictions": str(OUT / "case4-loo-predictions.csv"),
        "report": str(report),
        "plots": {name: str(OUT / f"case4_bayesian_comparison_{name}.png") for name in outcomes},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Bayesian fits complete: {results_path}", flush=True)
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_outcome(outcome: str, observed: np.ndarray, labels: list[str], models: dict[str, dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#b23a48", "#6a4c93", "#477a67", "#d18b24"]
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.2), constrained_layout=True, sharex=True, sharey=True)
    predictions = [np.asarray(item["predictions"], dtype=float) for item in models.values()]
    low = min(float(np.min(observed)), *(float(np.min(p)) for p in predictions))
    high = max(float(np.max(observed)), *(float(np.max(p)) for p in predictions))
    padding = max(5.0, 0.08 * (high - low))
    lower, upper = low - padding, high + padding
    for ax, ((model_name, item), color) in zip(axes, zip(models.items(), colors, strict=True), strict=True):
        prediction = np.asarray(item["predictions"], dtype=float)
        fit = item["fit"]
        ax.plot([lower, upper], [lower, upper], color="#777777", ls="--", lw=1.1)
        ax.scatter(observed, prediction, s=74, facecolor="white", edgecolor=color, linewidth=1.8, marker="^")
        for x, y, label in zip(observed, prediction, labels, strict=True):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        validation = fit.get("nested_loo_r2", fit.get("loo_r2"))
        validation_label = fit.get("validation_label", "nested LOO")
        title = model_name.replace("_", " ")
        ax.set_title(f"{title}\nin-sample R²={fit['r2_in_sample']:.2f}; {validation_label} R²={validation:.2f}")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel(f"Observed {outcome.replace('_percent', '').replace('_', ' ')} (%)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("LOO predicted value (%)")
    fig.suptitle(f"Case 4 Bayesian comparison — {outcome.replace('_', ' ')}", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_report(results: dict[str, Any], rows: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
    lines = [
        "# Case 4 Bayesian outcome comparisons",
        "",
        "Exploratory seven-row comparison of all Au–Cu carbene structures 1–7, matched to products 9 and 13–18.",
        "",
        "The common-core D3 reference is the same AuCu scaffold and 15-atom carbene core with R=H at the variable acyl position. It is a constructed reference, not an experimental catalyst. Each complex starts from a 105-snapshot xTB metadynamics trajectory. Strict topology screening against the first snapshot removes frames with undeclared connectivity, metal-coordination, proton-host, or fragment-count changes; retained frame counts are benzoate 29, pivalate 34, phenylacetate 33, para-CF3-benzoate 63, para-OMe-benzoate 48, para-tBu-benzoate 75, and 3,5-di-tBu-benzoate 29. Surviving frames are uniformly weighted after renormalization. The 98-snapshot common-core reference retains 34 frames. No CREST conformer ensembles or Boltzmann populations are used.",
        "",
        "## Experimental response table",
        "",
        "| Structure | Catalyst | Product | Yield (%) | de (%) | cis-ee (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['structure_number']} | {row['catalyst']} | {row['product']} | {row['yield_percent']:.1f} | {row['de_percent']:.1f} | {row['ee_cis_percent']:.1f} |")
    lines += [
        "",
        "Yield is taken from the SI product summaries. de is the absolute major-minus-minor excess from the reported cis:trans ratio. cis-ee is the absolute major-minus-minor excess from the reported cis enantiomer ratio. These transformations preserve the SI's printed ratios. Trans er is retained as metadata where available but is not a complete response across all seven rows.",
        "",
        "## Model comparison",
        "",
        "The dynamic model uses the Case 3f 27-feature schema, with missing or constant descriptors excluded and listed in the feature-pool manifest. The full dynamic model uses all complete, nonconstant numeric d2/d3-derived descriptors. Both map-derived models use the uniformly weighted xTB metadynamics ensembles. The conventional model uses the exact 15-feature traditional pool from Case 3f: static %Vbur and Solid-G plus DBSTEP and Morfeus descriptors. DBSTEP 1.1.0 and Morfeus-ML 0.8.0 descriptors were evaluated on one static xTB-optimized geometry per complex, using Au atom 2 to carbene carbon atom 148 as the axis; they are not trajectory averages.",
        "",
        "For each outcome, the selected traditional model chooses at most three of the 15 descriptors with nested leave-one-out (LOO). A second traditional sensitivity model includes all 15 descriptors while reselecting standardization and prior/noise settings inside each LOO fold. The 27-feature model also uses nested LOO. The high-dimensional full dynamic model reports fixed-subset LOO only: its features were selected using all seven outcomes, so its score is optimistic and not directly comparable to nested scores. The former Case 3d static-map schema is not used as the traditional model in this comparison.",
        "",
        "| Outcome | Model | Features | Selected | In-sample R² | LOO type | LOO R² | LOO RMSE |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for outcome, models in results["models"].items():
        for model_name, fit in models.items():
            validation = fit.get("nested_loo_r2", fit.get("loo_r2"))
            validation_rmse = fit.get("nested_loo_rmse", fit.get("loo_rmse"))
            validation_type = fit.get("validation_label", "nested LOO" if "nested_loo_r2" in fit else "fixed-subset LOO")
            selected = "all 15 Case 3f descriptors (fixed set)" if model_name == "traditional_case3f_all_15" else ", ".join(fit["feature_names"])
            lines.append(f"| {outcome} | {model_name} | {fit['candidate_feature_count']} | {selected} | {fit['r2_in_sample']:.3f} | {validation_type} | {validation:.3f} | {validation_rmse:.2f} |")
    lines += [
        "",
        "## Limits",
        "",
        "There are seven observations and many correlated descriptors. These fits are exploratory and sensitive to individual rows; LOO scores do not establish reliable generalization. The geometries start from published transition-state models and were xTB optimized and sampled by metadynamics; they are not experimentally determined intermediate structures. No mechanistic or causal interpretation is warranted.",
        "",
        "## Outputs",
        "",
        "- `results/case4-feature-matrix.csv`",
        "- `results/case4-feature-pool-manifest.json`",
        "- `results/case4-traditional-per-structure.csv`",
        "- `results/case4-traditional-descriptor-manifest.json`",
        "- `traditional-descriptor-requirements.txt`",
        "- `results/case4-bayesian-results.json`",
        "- `results/case4-model-comparison.csv`",
        "- `results/case4-loo-predictions.csv`",
        "- `results/case4_bayesian_comparison_yield_percent.png`",
        "- `results/case4_bayesian_comparison_de_percent.png`",
        "- `results/case4_bayesian_comparison_ee_cis_percent.png`",
    ]
    path = ANALYSIS / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if not args.fit_only:
        run_maps()
    rows, metadata = build_feature_matrix()
    if not args.maps_only:
        results = fit_models(rows, metadata)
        print(json.dumps({"status": results["status"], "rows": len(rows), "full_features": metadata["full_pool_feature_count"], "core_features_available": len(metadata["case3f_core_variable_in_case4"]), "output": str(OUT)}, indent=2))
    else:
        print(json.dumps({"status": "feature_matrix_ready", "rows": len(rows), "full_features": metadata["full_pool_feature_count"], "core_features_available": len(metadata["case3f_core_variable_in_case4"]), "output": str(OUT / "case4-feature-matrix.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
