"""Run the iron-only Part 3c FeCl porphyrin d2-map/d3-map study.

The publication set is deliberately restricted to FeCl(TDCPP), FeCl(TPP), and
FeCl(TMP).  The supplied XYZ files are starting geometries rather than
conformational ensembles.  New sampling therefore uses a reviewed neutral
high-spin Fe(III) chloride working state (multiplicity 6), preserves the four
Fe--N contacts, and requires a completed conformer ensemble rather than a
trajectory fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dmap import read_xyz_ensemble
from dmap.release import (
    compare_release_outputs,
    create_release_project,
    default_structure_setup,
    execute_release_project,
    read_release_project,
    validate_pair_compatibility,
    write_portable_zip,
)

CHEMICAL_ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = CHEMICAL_ROOT / "3cstructures"
RESULTS = CHEMICAL_ROOT / "part3" / "3c"
DOC_ROOT = CHEMICAL_ROOT.parent / "Doc" / "part3"


def _find_executable(name: str) -> Path:
    candidates = (
        # CREST 3.0.2 on macOS/arm64 crashes in its ANCOPT formatter for
        # open-shell external-xTB calculations.  Keep the publication run on
        # the separately pinned, standalone-xTB CREST 2.12 environment.
        CHEMICAL_ROOT.parent / ".micromamba" / "part3c-crest212" / "bin" / name,
        CHEMICAL_ROOT.parent / "genaisubstrate" / ".chem-env" / "bin" / name,
        CHEMICAL_ROOT.parent / ".micromamba" / "part3c" / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{name} was not found in the existing chemistry environment or the local Part 3c environment"
    )


XTB_EXECUTABLE = _find_executable("xtb")
CREST_EXECUTABLE = _find_executable("crest")

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
LIGANDS = ("tdcpp", "tpp", "tmp")
IRON_PUBLICATION_FILES = {"fecltdcpp.xyz", "fecltpp.xyz", "fecltmp.xyz"}
LIGAND_SOURCE_FILES = {
    "tdcpp": "zntdcpp.xyz",
    "tpp": "zntpp.xyz",
    "ttp": "znttp.xyz",
    "tmp": "zntmp.xyz",
}

# This is intentionally one conservative profile for every structure and pair.
# The direction count is lower than the Part 1 40,962-direction validation
# profile so the first corpus screen remains practical; every project records
# this profile in its JSON and can be regenerated at higher resolution later.
SHARED_PROFILE: dict[str, Any] = {
    "temperature": 298.15,
    "static": {"sphere_radius": 3.5, "spacing": 0.20, "direction_count": 8192},
    "field": {"spacing": 0.25},
    "axial": {"transverse_radius": 3.5, "z_min": 0.0, "z_max": 2.0, "spacing": 0.25},
    "displaced_scan": {
        "z_min": 0.0,
        "z_max": 2.0,
        "spacing": 0.25,
        "sphere_spacing": 0.20,
        "direction_count": 8192,
    },
    "reporting": {
        "enabled": True,
        "z_min": 0.0,
        "z_max": 2.0,
        "z_spacing": 0.50,
        "topographic_spacing": 0.25,
        "ml_fraction_resolutions": [1, 2, 4],
        "office_reports": False,
        "retain_per_frame_topographic_fields": False,
    },
    "vbur_radii_profile": "sambvca_2.1",
    "field_radii_profile": "dmap_vdw_v0.1",
}


@dataclass(frozen=True)
class StructureRecord:
    stem: str
    source: Path
    analysis_input: Path
    metal: str
    metal_key: str
    ligand: str
    center_atom: int
    donor_atoms: tuple[int, ...]
    generated: bool


def _read_geometry(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    ensemble = read_xyz_ensemble(path)
    geometry = ensemble.geometries[0]
    return geometry.elements, np.asarray(geometry.coordinates, dtype=float)


def _write_xyz(path: Path, elements: tuple[str, ...], coordinates: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [str(len(elements)), comment]
    rows.extend(
        f"{element:<3} {coordinate[0]: .10f} {coordinate[1]: .10f} {coordinate[2]: .10f}"
        for element, coordinate in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _metal_index(elements: tuple[str, ...]) -> int:
    indices = [index for index, element in enumerate(elements) if element.capitalize() in TRANSITION_METALS]
    if len(indices) != 1:
        raise ValueError(f"expected exactly one transition metal, found {indices}")
    return indices[0]


def _donor_indices(elements: tuple[str, ...]) -> tuple[int, ...]:
    donors = tuple(index for index, element in enumerate(elements) if element.upper() == "N")
    if len(donors) != 4:
        raise ValueError(f"expected four porphyrin nitrogens, found {len(donors)}")
    return donors


def _canonical_coordinates(
    elements: tuple[str, ...], coordinates: np.ndarray, center_index: int, donors: tuple[int, ...]
) -> np.ndarray:
    """Translate a structure to the metal-centred, porphyrin-plane frame."""
    centered = coordinates - coordinates[center_index]
    donor_coordinates = centered[np.asarray(donors, dtype=int)]
    _, _, right_singular_vectors = np.linalg.svd(
        donor_coordinates - donor_coordinates.mean(axis=0), full_matrices=False
    )
    z_axis = right_singular_vectors[-1]
    pivot = int(np.argmax(np.abs(z_axis)))
    if z_axis[pivot] < 0.0:
        z_axis = -z_axis
    x_axis = centered[donors[0]] - np.dot(centered[donors[0]], z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1.0e-12:
        raise ValueError("first donor does not define a usable secondary frame direction")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return centered @ basis


def _source_ligand(
    path: Path,
) -> tuple[tuple[str, ...], np.ndarray, int, tuple[int, ...]]:
    elements, coordinates = _read_geometry(path)
    center = _metal_index(elements)
    donors = _donor_indices(elements)
    return elements, _canonical_coordinates(elements, coordinates, center, donors), center, donors


def _generated_variant_name(base_stem: str, ligand: str) -> str:
    prefix = re.sub(r"tdcpp$", "", base_stem, flags=re.IGNORECASE)
    return f"{prefix}{ligand}"


def _generate_non_zinc_variants() -> list[dict[str, Any]]:
    """Add TPP/TTP/TMP models for every supplied non-zinc TDCpp structure."""
    ligand_sources = {
        ligand: _source_ligand(STRUCTURES / filename)
        for ligand, filename in LIGAND_SOURCE_FILES.items()
    }
    generated_records: list[dict[str, Any]] = []
    for base in sorted(STRUCTURES.glob("*.xyz")):
        stem = base.stem
        if not stem.lower().endswith("tdcpp") or stem.lower().startswith("zn"):
            continue
        base_elements, base_coordinates = _read_geometry(base)
        base_center = _metal_index(base_elements)
        base_donors = _donor_indices(base_elements)
        base_local = _canonical_coordinates(base_elements, base_coordinates, base_center, base_donors)
        axial_elements = base_elements[77:]
        axial_coordinates = base_local[77:]
        metal = base_elements[base_center]

        for ligand in ("tpp", "ttp", "tmp"):
            source_elements, source_local, source_center, _ = ligand_sources[ligand]
            elements = tuple(
                metal if index == source_center else element
                for index, element in enumerate(source_elements)
            ) + tuple(axial_elements)
            coordinates = np.vstack(
                (
                    np.asarray(
                        [
                            np.zeros(3) if index == source_center else source_local[index]
                            for index in range(len(source_elements))
                        ],
                        dtype=float,
                    ),
                    axial_coordinates,
                )
            )
            target = STRUCTURES / f"{_generated_variant_name(stem, ligand)}.xyz"
            if not target.exists():
                _write_xyz(
                    target,
                    elements,
                    coordinates,
                    f"Part 3c constructed {metal}({ligand.upper()}) from {base.name} axial fragment and {LIGAND_SOURCE_FILES[ligand]}",
                )
            generated_records.append(
                {
                    "file": target.name,
                    "source_non_zinc_tdcpp": base.name,
                    "ligand_source": LIGAND_SOURCE_FILES[ligand],
                    "metal": metal,
                    "ligand": ligand,
                    "construction": "metal-centred porphyrin-frame ligand graft; axial fragment retained from supplied TDCpp geometry",
                    "optimized": False,
                    "sampled": False,
                }
            )

    manifest = STRUCTURES / "part3c-constructed-variants.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "part3c.constructed-variants.v1",
                "purpose": "Non-zinc TPP/TTP/TMP screening coordinate models",
                "records": generated_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return generated_records


def _parse_structure(path: Path, analysis_input: Path) -> StructureRecord:
    elements, _ = _read_geometry(path)
    center = _metal_index(elements)
    donors = _donor_indices(elements)
    stem = path.stem.lower()
    ligand = next((value for value in LIGANDS if stem.endswith(value)), None)
    if ligand is None:
        raise ValueError(f"cannot infer porphyrin ligand from {path.name}")
    metal_key = stem[: -len(ligand)]
    if not metal_key:
        raise ValueError(f"cannot infer metal label from {path.name}")
    metal = elements[center].capitalize()
    return StructureRecord(
        stem=path.stem,
        source=path,
        analysis_input=analysis_input,
        metal=metal,
        metal_key=metal_key,
        ligand=ligand,
        center_atom=center + 1,
        donor_atoms=tuple(index + 1 for index in donors),
        generated=(ligand != "tdcpp" and not path.name.lower().startswith("zn")),
    )


def _prepare_records() -> list[StructureRecord]:
    input_root = RESULTS / "analysis-inputs"
    records: list[StructureRecord] = []
    for source in sorted(STRUCTURES.glob("*.xyz")):
        if source.name.lower() not in IRON_PUBLICATION_FILES:
            continue
        elements, coordinates = _read_geometry(source)
        center = _metal_index(elements)
        donors = _donor_indices(elements)
        canonical = _canonical_coordinates(elements, coordinates, center, donors)
        analysis_input = input_root / f"{source.stem}-canonical.xyz"
        if not analysis_input.exists():
            _write_xyz(
                analysis_input,
                elements,
                canonical,
                f"Part 3c canonical analysis input derived from {source.name}; metal-centred porphyrin frame",
            )
        records.append(_parse_structure(source, analysis_input))

    manifest = RESULTS / "structure-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "part3c.structure-manifest.v1",
                "source_directory": str(STRUCTURES),
                "canonical_analysis_inputs": str(input_root),
                "records": [record.__dict__ for record in records],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def _setup(record: StructureRecord) -> dict[str, Any]:
    elements, coordinates = _read_geometry(record.analysis_input)
    chloride_candidates = tuple(
        index for index, element in enumerate(elements, start=1) if element == "Cl"
    )
    iron_coordinate = coordinates[record.center_atom - 1]
    chloride_distances = sorted(
        (
            float(np.linalg.norm(coordinates[index - 1] - iron_coordinate)),
            index,
        )
        for index in chloride_candidates
    )
    if not chloride_distances or chloride_distances[0][0] >= 3.0:
        raise ValueError(
            f"{record.analysis_input} must contain one Fe-bound chloride within 3.0 A"
        )
    if len(chloride_distances) > 1 and chloride_distances[1][0] < 3.0:
        raise ValueError(f"{record.analysis_input} has ambiguous Fe-bound chlorides")
    chloride_atoms = (chloride_distances[0][1],)
    protected_ligands = (*record.donor_atoms, *chloride_atoms)
    setup = {
        **default_structure_setup(
            record.analysis_input,
            center_atom=record.center_atom,
            reactive_direction=[0.0, 0.0, 1.0],
            secondary_direction=[1.0, 0.0, 0.0],
            alignment_atoms=list(record.donor_atoms),
            charge=0,
            multiplicity=6,
            solvent=None,
            confirmed=True,
            protected_contacts=[
                [record.center_atom, ligand_atom] for ligand_atom in protected_ligands
            ],
            frozen_atoms=[],
            xtb_executable=str(XTB_EXECUTABLE),
            crest_executable=str(CREST_EXECUTABLE),
            threads=4,
        ),
        "chemical_setup_status": "reviewed neutral high-spin Fe(III) chloride working state; multiplicity 6 (S=5/2); state-specific higher-level validation remains required for publication",
        "source_structure": str(record.source),
        "structure_construction": (
            "supplied structure"
            if not record.generated
            else "constructed non-zinc ligand variant; see 3cstructures/part3c-constructed-variants.json"
        ),
        "sampling_provenance": "unrestrained GFN2-xTB tight preoptimization in the fixed high-spin Fe(III) state, followed by a completed CREST super-reduced conformer search with restraints derived from the relaxed four Fe-N contacts and axial Fe-Cl contact; no trajectory fallback",
    }
    setup["sampling"]["quick"] = True
    setup["sampling"]["maximum_reduced"] = True
    setup["sampling"]["constrain_preoptimization"] = False
    setup["sampling"]["allow_trajectory_fallback"] = False
    return setup


def _project_path(kind: str, name: str) -> Path:
    return RESULTS / kind / name / "project.d3map.json"


def _run_project(
    *,
    kind: str,
    name: str,
    reference: StructureRecord,
    extended: StructureRecord | None = None,
) -> Path:
    target = _project_path(kind, name)
    if not target.exists():
        # create_release_project deliberately avoids overwriting an existing
        # directory.  Remove only an empty directory left by an interrupted
        # creation attempt so the requested deterministic path can be used.
        if target.parent.is_dir() and not any(target.parent.iterdir()):
            target.parent.rmdir()
        project = create_release_project(
            kind=kind,
            reference_xyz=reference.analysis_input,
            extended_xyz=None if extended is None else extended.analysis_input,
            output_directory=target.parent,
            reference_setup=_setup(reference),
            extended_setup=None if extended is None else _setup(extended),
            shared=SHARED_PROFILE,
        )
    else:
        project = target
    _refresh_project_provenance(project, reference, extended)
    state_path = project.parent / "run-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        if state.get("status") == "success":
            return project
    if kind == "d3-map":
        _run_cached_d3_project(project, reference, extended)
    else:
        execute_release_project(project, sampling=True)
    return project


def _refresh_project_provenance(
    project: Path,
    reference: StructureRecord,
    extended: StructureRecord | None,
) -> None:
    """Keep resumable project metadata aligned with source classification."""
    payload = _read_json(project)
    records = {"reference": reference}
    if extended is not None:
        records["extended"] = extended
    changed = False
    for role, record in records.items():
        structure = payload.get("structures", {}).get(role)
        if not isinstance(structure, dict):
            continue
        construction = (
            "supplied structure"
            if not record.generated
            else "constructed non-zinc ligand variant; see 3cstructures/part3c-constructed-variants.json"
        )
        if structure.get("structure_construction") != construction:
            structure["structure_construction"] = construction
            changed = True
        if structure.get("source_structure") != str(record.source):
            structure["source_structure"] = str(record.source)
            changed = True
    if changed:
        project.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _run_cached_d3_project(
    project: Path,
    reference: StructureRecord,
    extended: StructureRecord,
) -> None:
    """Compare already-sampled d2 branches without sampling each branch twice."""
    config = read_release_project(project)
    root = project.parent
    role_sources = {
        "reference": _project_path("d2-map", _d2_name(reference)),
        "extended": _project_path("d2-map", _d2_name(extended)),
    }
    for role, source_project in role_sources.items():
        source_root = source_project.parent
        source_results = source_root / "results" / "reference"
        source_stage = source_root / "stages" / "reference"
        if not source_results.is_dir() or not (source_results / "descriptors.json").is_file():
            raise RuntimeError(f"completed d2-map output is missing for {role}: {source_project}")
        target_results = root / "results" / role
        target_stage = root / "stages" / role
        shutil.copytree(source_results, target_results, dirs_exist_ok=True)
        if source_stage.is_dir():
            shutil.copytree(source_stage, target_stage, dirs_exist_ok=True)

    mismatches = validate_pair_compatibility(root, config)
    compare_release_outputs(project, config, mismatches)
    state = {
        "schema": "d3map.run-state.v1",
        "status": "success",
        "sampling_strategy": "reused completed independent d2-map branches",
        "comparison_status": "nonstandard" if mismatches else "standard",
        "compatibility_mismatches": mismatches,
    }
    (root / "run-state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_portable_zip(project)


def _d2_name(record: StructureRecord) -> str:
    return f"{record.metal_key}-{record.ligand}-d2map"


def _d3_name(reference: StructureRecord, extended: StructureRecord) -> str:
    return f"{reference.metal_key}-{reference.ligand}-vs-{extended.ligand}-d3map"


def _d3_pairs(records: list[StructureRecord]) -> list[tuple[StructureRecord, StructureRecord]]:
    by_key = {(record.metal_key, record.ligand): record for record in records}
    reference = by_key[("fecl", "tdcpp")]
    return [
        (reference, by_key[("fecl", "tpp")]),
        (reference, by_key[("fecl", "tmp")]),
    ]


def _run_d2_job(record: StructureRecord) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    name = _d2_name(record)
    print(f"[d2] {record.metal_key} {record.ligand.upper()}", flush=True)
    try:
        project = _run_project(kind="d2-map", name=name, reference=record)
        return _d2_summary(record, project), None
    except Exception as error:  # keep the corpus run auditable and continue  # noqa: BLE001
        failure = {"project": f"d2-map/{name}", "error": str(error)}
        print(f"[d2] {record.metal_key} {record.ligand.upper()} FAILED: {error}", flush=True)
        return None, failure


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _d2_summary(record: StructureRecord, project: Path) -> dict[str, Any]:
    output = project.parent / "results" / "reference"
    descriptors = _read_json(output / "descriptors.json")
    baselines = _read_json(output / "static-baselines.json")["baselines"]
    input_baseline = baselines["input"]
    vbur = descriptors["ensemble_scalar_descriptors"]["vbur_percent"]
    g = descriptors["ensemble_scalar_descriptors"]["g_percent"]
    return {
        "metal": record.metal,
        "metal_key": record.metal_key,
        "ligand": record.ligand,
        "structure": record.source.name,
        "generated": record.generated,
        "frames": len(descriptors["weights"]),
        "analysis_mode": descriptors["analysis_mode"],
        "static_vbur_percent": input_baseline.get("vbur_percent"),
        "static_g_percent": input_baseline.get("g_percent"),
        "d2_mean_vbur_percent": vbur["mean"],
        "d2_mean_g_percent": g["mean"],
        "d2_sd_vbur_percent": vbur["population_standard_deviation"],
        "d2_sd_g_percent": g["population_standard_deviation"],
        "project": str(project.relative_to(RESULTS)),
    }


def _scalar_value(rows: list[dict[str, Any]], descriptor: str, statistic: str) -> dict[str, Any]:
    for row in rows:
        if row["descriptor"] == descriptor and row["statistic"] == statistic:
            return row
    raise KeyError((descriptor, statistic))


def _static_baseline_value(
    rows: list[dict[str, Any]], descriptor: str, baseline: str
) -> dict[str, Any]:
    for row in rows:
        if row["descriptor"] == descriptor and row["baseline"] == baseline:
            return row
    raise KeyError((descriptor, baseline))


def _d3_summary(
    reference: StructureRecord,
    extended: StructureRecord,
    project: Path,
) -> dict[str, Any]:
    comparison = _read_json(project.parent / "results" / "comparison" / "comparison.json")
    rows = comparison["scalar_comparisons"]
    vbur = _scalar_value(rows, "vbur_percent", "mean")
    g = _scalar_value(rows, "g_percent", "mean")
    return {
        "metal": reference.metal,
        "metal_key": reference.metal_key,
        "reference_ligand": reference.ligand,
        "extended_ligand": extended.ligand,
        "static_delta_vbur_percent": _static_baseline_value(
            comparison["static_baseline_comparisons"], "vbur_percent", "input"
        ).get("difference_extended_minus_reference"),
        "static_delta_g_percent": _static_baseline_value(
            comparison["static_baseline_comparisons"], "g_percent", "input"
        ).get("difference_extended_minus_reference"),
        "d3_delta_mean_vbur_percent": vbur["difference_extended_minus_reference"],
        "d3_delta_mean_g_percent": g["difference_extended_minus_reference"],
        "status": comparison["status"],
        "project": str(project.relative_to(RESULTS)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_findings(
    records: list[StructureRecord],
    d2_rows: list[dict[str, Any]],
    d3_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> Path:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    path = DOC_ROOT / "part3c-key-findings.md"
    d2_by_ligand: dict[str, list[dict[str, Any]]] = {ligand: [] for ligand in LIGANDS}
    for row in d2_rows:
        d2_by_ligand[row["ligand"]].append(row)

    lines = [
        "# Part 3c — Fe(III) chloride porphyrin d2-map/d3-map study",
        "",
        "## Scope",
        "",
        "Part 3c is restricted to FeCl(TDCPP), FeCl(TPP), and FeCl(TMP). It applies the maintained d3map d2-map and d3-map probability-field workflows to these three iron porphyrins. Each structure is also evaluated against the SambVca 2.1-compatible buried-volume profile and the Guzei–Wendt method-equivalent Solid-G profile. The result bundles are stored in [`03-chemical-validation/part3/3c`](../../03-chemical-validation/part3/3c).",
        "",
        f"The corpus contains {len(records)} iron structure inputs, {len(d2_rows)} completed d2-map analyses, and {len(d3_rows)} completed d3-map comparisons. FeCl(TDCPP) is the declared d3-map reference; FeCl(TPP) and FeCl(TMP) are the two comparison members.",
        "",
        "## Important interpretation boundary",
        "",
        "The supplied files are single XYZ starting geometries. Each is tightly preoptimized and sampled in the same fixed neutral sextet Fe(III) chloride state. Only a completed CREST conformer ensemble is accepted; trajectory fallback is disabled. The sampling energies therefore support thermodynamic conformer weighting within this declared spin state, but they do not establish the relative stability of alternative spin states.",
        "",
        "The supplied coordinates are first tightly preoptimized without geometric restraints in the fixed sextet state. The four relaxed Fe–porphyrin N distances and relaxed axial Fe–Cl distance then define CREST restraints that protect the intended five-coordinate topology without freezing construction artifacts from the deposited inputs. The workflow uses CREST 2.12 super-reduced search mode with standalone GFN2-xTB 6.7.1 because the available CREST 3.0.2 macOS build fails in its open-shell optimizer. This is ensemble thermodynamics, not a kinetic molecular-dynamics claim; exhaustive-search convergence is not claimed.",
        "",
        "## Shared calculation profile",
        "",
        "- SambVca 2.1-compatible radii for buried volume; d3map van-der-Waals radii for fields and method-equivalent G.",
        "- 3.5 Å sphere, 0.20 Å buried-volume grid, 8,192 angular directions, and 0.25 Å field/topographic grids.",
        "- Metal-centred porphyrin frame: +z is the fitted porphyrin-plane normal and +x points toward the first porphyrin N donor after projection.",
        "- Hydrogen atoms are excluded from steric descriptors; the metal centre is the catalytic origin and is excluded from its own steric atom set.",
        "",
        "## Key findings from the completed screen",
        "",
    ]

    if d2_rows:
        vbur_values = np.asarray([row["static_vbur_percent"] for row in d2_rows], dtype=float)
        g_values = np.asarray([row["static_g_percent"] for row in d2_rows], dtype=float)
        lines.extend(
            [
                f"1. The fixed-input SambVca-compatible `%Vbur` spans {_fmt(vbur_values.min())}–{_fmt(vbur_values.max())}% across the {len(d2_rows)} inputs; method-equivalent G spans {_fmt(g_values.min())}–{_fmt(g_values.max())}%. These are geometric descriptors, not activity or selectivity predictions.",
                f"2. The sampled d2-map means differ from the fixed-input baseline by {_fmt(np.mean([row['d2_mean_vbur_percent'] - row['static_vbur_percent'] for row in d2_rows]))} percentage points for `%Vbur` and {_fmt(np.mean([row['d2_mean_g_percent'] - row['static_g_percent'] for row in d2_rows]))} percentage points for G on average across completed jobs. This is a descriptive sampling shift, not yet a validated chemical effect.",
            ]
        )
        ligand_lines = []
        for ligand in LIGANDS:
            values = d2_by_ligand[ligand]
            if values:
                mean_v = float(np.mean([row["static_vbur_percent"] for row in values]))
                mean_g = float(np.mean([row["static_g_percent"] for row in values]))
                ligand_lines.append(f"{ligand.upper()} mean `%Vbur` {_fmt(mean_v)}%, G {_fmt(mean_g)}%")
        lines.append(
            "3. Iron-ligand results are: "
            + "; ".join(ligand_lines)
            + f". The completed d2 analyses retain {min(row['frames'] for row in d2_rows)}–{max(row['frames'] for row in d2_rows)} CREST conformers per structure with the stored energy-derived weights."
        )
    if d3_rows:
        larger_vbur = max(d3_rows, key=lambda row: abs(float(row["static_delta_vbur_percent"])))
        larger_g = max(d3_rows, key=lambda row: abs(float(row["static_delta_g_percent"])))
        lines.extend(
            [
                f"4. The largest absolute TDCpp-to-ligand input `%Vbur` change in the completed d3-map set is {_fmt(larger_vbur['static_delta_vbur_percent'])} percentage points for {larger_vbur['metal_key']} {larger_vbur['extended_ligand'].upper()} minus TDCpp; the corresponding largest absolute G change is {_fmt(larger_g['static_delta_g_percent'])} percentage points for {larger_g['metal_key']} {larger_g['extended_ligand'].upper()} minus TDCpp.",
                "5. The d3-map bundles store signed extended-minus-reference differences for both the sampled means and the static baselines. These paired differences are the primary Part 3c comparison objects; they should be interpreted together with frame count, effective ensemble size, topology screening, and sampling provenance.",
            ]
        )

    lines.extend(
        [
            "",
            "## d2-map scalar results",
            "",
            "| Metal | Metal key | Ligand | Input `%Vbur` | Input G | d2 mean `%Vbur` | d2 mean G | Frames |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(d2_rows, key=lambda item: (item["metal_key"], LIGANDS.index(item["ligand"]))):
        lines.append(
            f"| {row['metal']} | `{row['metal_key']}` | {row['ligand'].upper()} | {_fmt(row['static_vbur_percent'])} | {_fmt(row['static_g_percent'])} | {_fmt(row['d2_mean_vbur_percent'])} | {_fmt(row['d2_mean_g_percent'])} | {row['frames']} |"
        )

    lines.extend(
        [
            "",
            "## d3-map scalar differences",
            "",
            "Differences are `extended − reference`; the same sign is stored in each comparison bundle.",
            "",
            "| Metal | Metal key | Reference | Extended | Static Δ`%Vbur` | Static ΔG | d3 Δmean `%Vbur` | d3 Δmean G |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(d3_rows, key=lambda item: (item["metal_key"], item["reference_ligand"], item["extended_ligand"])):
        lines.append(
            f"| {row['metal']} | `{row['metal_key']}` | {row['reference_ligand'].upper()} | {row['extended_ligand'].upper()} | {_fmt(row['static_delta_vbur_percent'])} | {_fmt(row['static_delta_g_percent'])} | {_fmt(row['d3_delta_mean_vbur_percent'])} | {_fmt(row['d3_delta_mean_g_percent'])} |"
        )

    lines.extend(["", "## Reproducibility and follow-up", ""])
    lines.append("Every completed project contains its copied input, `project.d3map.json`, d2-map descriptors, static baselines, method-equivalent Solid-G/SambVca scalar outputs, paired difference arrays where applicable, plots, and manifests. The machine-readable corpus summaries are `03-chemical-validation/part3/3c/d2-map-summary.csv` and `d3-map-summary.csv`.")
    lines.append("")
    lines.append("The calculation fixes the experimentally motivated high-spin Fe(III) working state. A manuscript must still state that GFN2-xTB does not validate spin-state ordering; higher-level state-specific single points are a separate validation layer if such a claim is required.")
    if failures:
        lines.extend(["", "## Incomplete jobs", ""])
        for failure in failures:
            lines.append(f"- `{failure['project']}`: {failure['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_run_manifest(
    records: list[StructureRecord], d2_rows: list[dict[str, Any]], d3_rows: list[dict[str, Any]], failures: list[dict[str, str]]
) -> Path:
    path = RESULTS / "run-manifest.json"
    payload = {
        "schema": "part3c.run-manifest.v1",
        "profile": SHARED_PROFILE,
        "source_directory": str(STRUCTURES),
        "record_count": len(records),
        "d2_completed": len(d2_rows),
        "d3_completed": len(d3_rows),
        "failures": failures,
        "d2_summary": "d2-map-summary.csv",
        "d3_summary": "d3-map-summary.csv",
        "findings": str((DOC_ROOT / "part3c-key-findings.md").relative_to(CHEMICAL_ROOT.parent)),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-only", action="store_true")
    parser.add_argument("--d3-only", action="store_true")
    parser.add_argument("--limit", type=int, help="limit each requested job class for a pilot run")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of independent d2 sampling jobs to run concurrently",
    )
    args = parser.parse_args()
    if args.d2_only and args.d3_only:
        parser.error("--d2-only and --d3-only are mutually exclusive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    records = _prepare_records()
    d2_rows: list[dict[str, Any]] = []
    d3_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    if not args.d3_only:
        d2_jobs = records if args.limit is None else records[: args.limit]
        if args.workers == 1:
            for record in d2_jobs:
                row, failure = _run_d2_job(record)
                if row is not None:
                    d2_rows.append(row)
                if failure is not None:
                    failures.append(failure)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending = {executor.submit(_run_d2_job, record): record for record in d2_jobs}
                for future in as_completed(pending):
                    row, failure = future.result()
                    if row is not None:
                        d2_rows.append(row)
                    if failure is not None:
                        failures.append(failure)
        d2_rows.sort(key=lambda row: (row["metal_key"], LIGANDS.index(row["ligand"])))

    if not args.d2_only:
        d3_jobs = _d3_pairs(records)
        if args.limit is not None:
            d3_jobs = d3_jobs[: args.limit]
        for index, (reference, extended) in enumerate(d3_jobs, start=1):
            name = _d3_name(reference, extended)
            print(
                f"[d3 {index}/{len(d3_jobs)}] {reference.metal_key} {reference.ligand.upper()} -> {extended.ligand.upper()}",
                flush=True,
            )
            try:
                project = _run_project(
                    kind="d3-map", name=name, reference=reference, extended=extended
                )
                d3_rows.append(_d3_summary(reference, extended, project))
            except Exception as error:  # keep the corpus run auditable and continue  # noqa: BLE001
                failures.append({"project": f"d3-map/{name}", "error": str(error)})
                print(f"  FAILED: {error}", flush=True)

    _write_csv(RESULTS / "d2-map-summary.csv", d2_rows)
    _write_csv(RESULTS / "d3-map-summary.csv", d3_rows)
    findings = _write_findings(records, d2_rows, d3_rows, failures)
    _write_run_manifest(records, d2_rows, d3_rows, failures)
    print(
        json.dumps(
            {
                "structures": len(records),
                "d2_completed": len(d2_rows),
                "d3_completed": len(d3_rows),
                "failures": len(failures),
                "findings": str(findings),
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
