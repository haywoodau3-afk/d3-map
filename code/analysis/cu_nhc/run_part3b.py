"""Run the Part 3b flexible NHC d2-map/d3-map screen.

The input corpus is the published Cu(I)-NHC series from Zhao et al.
(Angew. Chem. Int. Ed. 2024, DOI 10.1002/anie.202318703).  CCDC CIFs are
converted to one molecule per XYZ input, retaining the deposited geometry and
removing only crystallographic solvent and other independent molecules.  The
analysis then uses the maintained dmap release workflow with GFN2-xTB/CREST
sampling and reports both the SambVca 2.1-compatible buried-volume profile and
the method-equivalent Solid-G profile.

This is a screening workflow.  Cu(I)-NHC complexes are neutral singlets in the
declared setup, and the Cu--C/Cu--Cl contacts are protected during sampling.
Those choices and the vacancy-facing frame convention are written to every
manifest so that a later chemistry-reviewed rerun can be compared directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dmap import DMapError, read_xyz_ensemble
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
REPO_ROOT = CHEMICAL_ROOT.parent
RESULTS = CHEMICAL_ROOT / "part3" / "3b"
CCDC_ROOT = RESULTS / "inputs" / "ccdc"
ANALYSIS_ROOT = RESULTS / "analysis-inputs"
DOC_ROOT = REPO_ROOT / "Doc" / "part3"


def _find_executable(name: str) -> Path:
    candidate = REPO_ROOT / ".micromamba" / "part3c" / "bin" / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"{name} was not found at the user-declared Part 3b environment path: {candidate}"
    )


XTB_EXECUTABLE = _find_executable("xtb")
CREST_EXECUTABLE = _find_executable("crest")


# This intentionally matches the conservative Part 3c screening profile.
# It keeps the first flexible-NHC corpus practical while recording all settings
# needed for a later high-resolution rerun.
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


CCDC_SOURCES = {
    "3a": {
        "file": "CCDC-2270937-POKJAE.cif",
        "ccdc": "2270937",
        "refcode": "POKJAE",
        "ccdc_doi": "10.5517/ccdc.csd.cc2g731w",
        "component": "Cu1",
        "paper_vbur": 52.7,
        "description": "N-Dipp/N-CH2MesMe, 4,5-dimethyl backbone",
    },
    "3b-syn": {
        "file": "CCDC-2270938-POKJEI.cif",
        "ccdc": "2270938",
        "refcode": "POKJEI",
        "ccdc_doi": "10.5517/ccdc.csd.cc2g732x",
        "component": "Cu1",
        "paper_vbur": 49.8,
        "description": "N-Mes/N-CH2MesMe, syn wingtip conformation",
    },
    "3b-anti": {
        "file": "CCDC-2270938-POKJEI.cif",
        "ccdc": "2270938",
        "refcode": "POKJEI",
        "ccdc_doi": "10.5517/ccdc.csd.cc2g732x",
        "component": "Cu2",
        "paper_vbur": 37.3,
        "description": "N-Mes/N-CH2MesMe, anti wingtip conformation",
    },
    "3c": {
        "file": "CCDC-2270939-POKJIM.cif",
        "ccdc": "2270939",
        "refcode": "POKJIM",
        "ccdc_doi": "10.5517/ccdc.csd.cc2g733y",
        "component": "Cu1",
        "paper_vbur": 40.2,
        "description": "published 3c representative independent molecule",
    },
    "3d": {
        "file": "CCDC-1994536-POKPAK.cif",
        "ccdc": "1994536",
        "refcode": "POKPAK",
        "ccdc_doi": "10.5517/ccdc.csd.cc24ygwg",
        "component": "Cu1",
        "paper_vbur": 38.9,
        "description": "published 3d representative independent molecule",
    },
}


@dataclass(frozen=True)
class StructureRecord:
    name: str
    source_cif: Path
    analysis_input: Path
    ccdc: str
    refcode: str
    component: str
    ccdc_doi: str
    description: str
    paper_vbur: float
    center_atom: int
    carbene_atom: int
    donor_atoms: tuple[int, int]
    chloride_atom: int
    atom_count: int
    input_sha256: str


@dataclass(frozen=True)
class CifAtom:
    label: str
    element: str
    fractional: np.ndarray
    occupancy: float


def _number(value: str) -> float:
    match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
    if not match:
        raise ValueError(f"cannot parse CIF number: {value}")
    return float(match.group(0))


def _loop_rows(lines: list[str], start: int) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    cursor = start
    while cursor < len(lines) and lines[cursor].strip().startswith("_"):
        headers.append(lines[cursor].strip().split()[0])
        cursor += 1
    rows: list[list[str]] = []
    width = len(headers)
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if not stripped or stripped.startswith(("loop_", "_", "#")):
            break
        try:
            values = shlex.split(stripped, comments=False)
        except ValueError:
            break
        if len(values) >= width:
            rows.append(values[:width])
        cursor += 1
    return headers, rows


def _read_cif(path: Path) -> tuple[dict[str, float], dict[str, CifAtom], dict[str, set[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cell: dict[str, float] = {}
    for line in lines:
        stripped = line.strip()
        for key in (
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        ):
            if stripped.startswith(key):
                cell[key] = _number(stripped.split()[1])

    atoms: dict[str, CifAtom] = {}
    bonds: dict[str, set[str]] = {}
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers, rows = _loop_rows(lines, index + 1)
        header_index = {header: position for position, header in enumerate(headers)}
        if "_atom_site_label" in header_index and "_atom_site_fract_x" in header_index:
            required = (
                "_atom_site_type_symbol",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_occupancy",
            )
            if not all(key in header_index for key in required):
                continue
            for row in rows:
                label = row[header_index["_atom_site_label"]]
                try:
                    occupancy = _number(row[header_index["_atom_site_occupancy"]])
                except ValueError:
                    occupancy = 1.0
                atoms[label] = CifAtom(
                    label=label,
                    element=row[header_index["_atom_site_type_symbol"]].capitalize(),
                    fractional=np.asarray(
                        [
                            _number(row[header_index["_atom_site_fract_x"]]),
                            _number(row[header_index["_atom_site_fract_y"]]),
                            _number(row[header_index["_atom_site_fract_z"]]),
                        ],
                        dtype=float,
                    ),
                    occupancy=occupancy,
                )
        if "_geom_bond_atom_site_label_1" in header_index:
            second = "_geom_bond_atom_site_label_2"
            symmetry = "_geom_bond_site_symmetry_2"
            if second not in header_index:
                continue
            for row in rows:
                if symmetry in header_index and row[header_index[symmetry]] not in {".", "1_555", "1_555?"}:
                    continue
                first_label = row[header_index["_geom_bond_atom_site_label_1"]]
                second_label = row[header_index[second]]
                bonds.setdefault(first_label, set()).add(second_label)
                bonds.setdefault(second_label, set()).add(first_label)
    if len(cell) != 6 or not atoms or not bonds:
        raise ValueError(f"CIF lacks required cell, atom, or bond data: {path}")
    return cell, atoms, bonds


def _cell_matrix(cell: dict[str, float]) -> np.ndarray:
    a = cell["_cell_length_a"]
    b = cell["_cell_length_b"]
    c = cell["_cell_length_c"]
    alpha = math.radians(cell["_cell_angle_alpha"])
    beta = math.radians(cell["_cell_angle_beta"])
    gamma = math.radians(cell["_cell_angle_gamma"])
    sin_gamma = math.sin(gamma)
    vectors = np.asarray(
        [
            [a, 0.0, 0.0],
            [b * math.cos(gamma), b * sin_gamma, 0.0],
            [
                c * math.cos(beta),
                c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma,
                math.sqrt(
                    max(
                        0.0,
                        c * c
                        - (c * math.cos(beta)) ** 2
                        - (
                            c
                            * (math.cos(alpha) - math.cos(beta) * math.cos(gamma))
                            / sin_gamma
                        )
                        ** 2,
                    )
                ),
            ],
        ],
        dtype=float,
    )
    return vectors


def _unwrap_component(
    atoms: dict[str, CifAtom], bonds: dict[str, set[str]], center: str
) -> list[str]:
    selected: list[str] = []
    fractional: dict[str, np.ndarray] = {center: atoms[center].fractional.copy()}
    pending = [center]
    while pending:
        parent = pending.pop(0)
        if parent not in selected:
            selected.append(parent)
        for child in sorted(bonds.get(parent, ())):
            if child not in atoms or atoms[child].occupancy <= 0.5 or child in fractional:
                continue
            delta = atoms[child].fractional - atoms[parent].fractional
            delta -= np.round(delta)
            fractional[child] = fractional[parent] + delta
            pending.append(child)
    for label in selected:
        atoms[label] = CifAtom(atoms[label].label, atoms[label].element, fractional[label], atoms[label].occupancy)
    return selected


def _canonicalize(
    elements: tuple[str, ...], coordinates: np.ndarray, center: int, donor: int, chloride: int
) -> np.ndarray:
    centered = coordinates - coordinates[center]
    blocked = centered[chloride]
    blocked_norm = np.linalg.norm(blocked)
    if blocked_norm <= 1e-10:
        raise ValueError("Cu--Cl vector is degenerate")
    # +z is the vacancy-facing direction, i.e. opposite the deposited Cu->Cl axis.
    z_axis = -blocked / blocked_norm
    x_axis = centered[donor] - np.dot(centered[donor], z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-10:
        raise ValueError("N donor does not define a secondary frame direction")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return centered @ basis


def _write_xyz(path: Path, elements: tuple[str, ...], coordinates: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [str(len(elements)), comment]
    rows.extend(
        f"{element:<3} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for element, xyz in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _prepare_one(name: str, source: dict[str, Any]) -> StructureRecord:
    cif_path = CCDC_ROOT / source["file"]
    if not cif_path.is_file():
        raise FileNotFoundError(f"missing CCDC source CIF: {cif_path}")
    cell, atoms, bonds = _read_cif(cif_path)
    center_label = source["component"]
    if center_label not in atoms or atoms[center_label].element != "Cu":
        raise ValueError(f"component {center_label} is not a Cu atom in {cif_path.name}")
    selected = _unwrap_component(atoms, bonds, center_label)
    selected_atoms = [atoms[label] for label in selected]
    elements = tuple(atom.element for atom in selected_atoms)
    fractional = np.asarray([atom.fractional for atom in selected_atoms], dtype=float)
    coordinates = fractional @ _cell_matrix(cell)
    center_index = selected.index(center_label)
    cu_neighbors = [label for label in bonds[center_label] if label in selected]
    chloride_labels = [label for label in cu_neighbors if atoms[label].element == "Cl"]
    carbene_labels = [label for label in cu_neighbors if atoms[label].element == "C"]
    if len(chloride_labels) != 1 or len(carbene_labels) != 1:
        raise ValueError(f"expected one Cu--Cl and one Cu--C bond for {name}: {cu_neighbors}")
    chloride_label = chloride_labels[0]
    carbene_label = carbene_labels[0]
    donors = sorted(
        label for label in bonds[carbene_label] if label in selected and atoms[label].element == "N"
    )
    if len(donors) != 2:
        raise ValueError(f"expected two N donors adjacent to carbene {carbene_label}: {donors}")
    canonical = _canonicalize(
        elements,
        coordinates,
        center_index,
        selected.index(donors[0]),
        selected.index(chloride_label),
    )
    target = ANALYSIS_ROOT / f"{name}-canonical.xyz"
    _write_xyz(
        target,
        elements,
        canonical,
        f"Part 3b canonical input from {source['ccdc']} {source['refcode']} {source['component']}; +z vacancy-facing",
    )
    return StructureRecord(
        name=name,
        source_cif=cif_path,
        analysis_input=target,
        ccdc=source["ccdc"],
        refcode=source["refcode"],
        component=center_label,
        ccdc_doi=source["ccdc_doi"],
        description=source["description"],
        paper_vbur=float(source["paper_vbur"]),
        center_atom=center_index + 1,
        carbene_atom=selected.index(carbene_label) + 1,
        donor_atoms=(selected.index(donors[0]) + 1, selected.index(donors[1]) + 1),
        chloride_atom=selected.index(chloride_label) + 1,
        atom_count=len(selected),
        input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


def _prepare_records() -> list[StructureRecord]:
    records = [_prepare_one(name, source) for name, source in CCDC_SOURCES.items()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "part3b.structure-manifest.v1",
        "paper_doi": "10.1002/anie.202318703",
        "ccdc_search_basis": "article DOI via CCDC Access Structures",
        "source_directory": str(CCDC_ROOT),
        "analysis_inputs": str(ANALYSIS_ROOT),
        "excluded_independent_molecules": {
            "3c": ["Cu2", "Cu3"],
            "3d": ["Cu2"],
        },
        "records": [asdict(record) | {"source_cif": str(record.source_cif), "analysis_input": str(record.analysis_input), "donor_atoms": list(record.donor_atoms)} for record in records],
    }
    (RESULTS / "structure-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def _setup(record: StructureRecord) -> dict[str, Any]:
    # Cu(I) d10 [Cu(NHC)Cl] is represented as a neutral singlet.  The Cu--C
    # and Cu--Cl distances are protected; the rest of the NHC remains flexible.
    elements = read_xyz_ensemble(record.analysis_input).elements
    # SambVca-style ligand %Vbur for a metal complex excludes the metal centre
    # and the opposing ancillary chloride from the ligand steric atom set.
    # Keeping Cl in the geometry but out of this set brings the fixed-input 3a
    # value onto the paper's reported 52.7% anchor under the same 3.5 Å sphere.
    steric_atoms = [
        index + 1
        for index, element in enumerate(elements)
        if element != "H" and index + 1 not in {record.center_atom, record.chloride_atom}
    ]
    setup = {
        **default_structure_setup(
            record.analysis_input,
            center_atom=record.center_atom,
            reactive_direction=[0.0, 0.0, 1.0],
            secondary_direction=[1.0, 0.0, 0.0],
            alignment_atoms=[record.center_atom, record.carbene_atom, *record.donor_atoms, record.chloride_atom],
            charge=0,
            multiplicity=1,
            solvent=None,
            confirmed=True,
            protected_contacts=[
                [record.center_atom, record.carbene_atom],
                [record.center_atom, record.chloride_atom],
            ],
            frozen_atoms=[record.center_atom],
            steric_atoms=steric_atoms,
            xtb_executable=str(XTB_EXECUTABLE),
            crest_executable=str(CREST_EXECUTABLE),
            threads=4,
        ),
        "chemical_setup_status": "Cu(I) d10 neutral singlet [Cu(NHC)Cl]; Cu--C and Cu--Cl protected; screening setup",
        "source_structure": str(record.source_cif),
        "source_ccdc_deposition": record.ccdc,
        "source_ccdc_refcode": record.refcode,
        "source_ccdc_doi": record.ccdc_doi,
        "source_paper_doi": "10.1002/anie.202318703",
        "frame_convention": "Cu-centred; +z points opposite deposited Cu->Cl (vacancy-facing); +x points to first N donor projected normal to +z",
        "sampling_provenance": "GFN2-xTB tight preoptimization followed by CREST quick conformational search; trajectory fallback permitted",
    }
    setup["sampling"]["quick"] = True
    setup["sampling"]["allow_trajectory_fallback"] = True
    setup["sampling"]["metadyn_steps"] = 25
    setup["sampling"]["metadyn_time_ps"] = 5.0
    setup["sampling"]["trajectory_stride"] = 10
    return setup


def _d2_name(record: StructureRecord) -> str:
    return f"{record.name}-d2map"


def _d3_name(reference: StructureRecord, extended: StructureRecord) -> str:
    return f"{reference.name}-vs-{extended.name}-d3map"


def _project_path(kind: str, name: str) -> Path:
    return RESULTS / kind / name / "project.d3map.json"


def _run_cached_d3_project(project: Path, reference: StructureRecord, extended: StructureRecord) -> None:
    config = read_release_project(project)
    root = project.parent
    sources = {
        "reference": _project_path("d2-map", _d2_name(reference)),
        "extended": _project_path("d2-map", _d2_name(extended)),
    }
    for role, source_project in sources.items():
        source_root = source_project.parent
        source_results = source_root / "results" / "reference"
        source_stage = source_root / "stages" / "reference"
        if not (source_results / "descriptors.json").is_file():
            raise RuntimeError(f"completed d2-map output is missing for {role}: {source_project}")
        shutil.copytree(source_results, root / "results" / role, dirs_exist_ok=True)
        if source_stage.is_dir():
            shutil.copytree(source_stage, root / "stages" / role, dirs_exist_ok=True)
    mismatches = validate_pair_compatibility(root, config)
    compare_release_outputs(project, config, mismatches)
    state = {
        "schema": "d3map.run-state.v1",
        "status": "success",
        "sampling_strategy": "reused completed independent d2-map branches",
        "comparison_status": "nonstandard" if mismatches else "standard",
        "compatibility_mismatches": mismatches,
    }
    (root / "run-state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_portable_zip(project)


def _run_project(kind: str, name: str, reference: StructureRecord, extended: StructureRecord | None = None) -> Path:
    project = _project_path(kind, name)
    if not project.exists():
        if project.parent.is_dir() and not any(project.parent.iterdir()):
            project.parent.rmdir()
        project = create_release_project(
            kind=kind,
            reference_xyz=reference.analysis_input,
            extended_xyz=None if extended is None else extended.analysis_input,
            output_directory=project.parent,
            reference_setup=_setup(reference),
            extended_setup=None if extended is None else _setup(extended),
            shared=SHARED_PROFILE,
        )
    state_path = project.parent / "run-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        if state.get("status") == "success":
            return project
    if kind == "d3-map":
        if extended is None:
            raise ValueError("d3-map requires an extended structure")
        _run_cached_d3_project(project, reference, extended)
    else:
        execute_release_project(project, sampling=True)
    return project


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _d2_summary(record: StructureRecord, project: Path) -> dict[str, Any]:
    output = project.parent / "results" / "reference"
    descriptors = _read_json(output / "descriptors.json")
    baseline = _read_json(output / "static-baselines.json")["baselines"]["input"]
    vbur = descriptors["ensemble_scalar_descriptors"]["vbur_percent"]
    g = descriptors["ensemble_scalar_descriptors"]["g_percent"]
    return {
        "name": record.name,
        "ccdc": record.ccdc,
        "refcode": record.refcode,
        "paper_vbur_percent": record.paper_vbur,
        "static_vbur_percent": baseline.get("vbur_percent"),
        "static_g_percent": baseline.get("g_percent"),
        "d2_mean_vbur_percent": vbur["mean"],
        "d2_sd_vbur_percent": vbur["population_standard_deviation"],
        "d2_mean_g_percent": g["mean"],
        "d2_sd_g_percent": g["population_standard_deviation"],
        "frames": len(descriptors["weights"]),
        "analysis_mode": descriptors["analysis_mode"],
        "project": str(project.relative_to(RESULTS)),
    }


def _comparison_row(rows: list[dict[str, Any]], descriptor: str, statistic: str) -> dict[str, Any]:
    for row in rows:
        if row["descriptor"] == descriptor and row["statistic"] == statistic:
            return row
    raise KeyError((descriptor, statistic))


def _static_row(rows: list[dict[str, Any]], descriptor: str) -> dict[str, Any]:
    for row in rows:
        if row["baseline"] == "input" and row["descriptor"] == descriptor:
            return row
    raise KeyError(descriptor)


def _d3_summary(reference: StructureRecord, extended: StructureRecord, project: Path) -> dict[str, Any]:
    comparison = _read_json(project.parent / "results" / "comparison" / "comparison.json")
    rows = comparison["scalar_comparisons"]
    vbur = _comparison_row(rows, "vbur_percent", "mean")
    g = _comparison_row(rows, "g_percent", "mean")
    static_vbur = _static_row(comparison["static_baseline_comparisons"], "vbur_percent")
    static_g = _static_row(comparison["static_baseline_comparisons"], "g_percent")
    return {
        "reference": reference.name,
        "extended": extended.name,
        "static_delta_vbur_percent": static_vbur["difference_extended_minus_reference"],
        "static_delta_g_percent": static_g["difference_extended_minus_reference"],
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


def _read_summary_csv(path: Path) -> list[dict[str, Any]]:
    """Read a previously completed summary when running one class only."""
    if not path.is_file():
        return []
    numeric = {
        "paper_vbur_percent",
        "static_vbur_percent",
        "static_g_percent",
        "d2_mean_vbur_percent",
        "d2_sd_vbur_percent",
        "d2_mean_g_percent",
        "d2_sd_g_percent",
        "static_delta_vbur_percent",
        "static_delta_g_percent",
        "d3_delta_mean_vbur_percent",
        "d3_delta_mean_g_percent",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in numeric:
                if key in row:
                    row[key] = None if row[key] == "" else float(row[key])
            if row.get("frames"):
                row["frames"] = int(row["frames"])
            rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_findings(
    records: list[StructureRecord], d2_rows: list[dict[str, Any]], d3_rows: list[dict[str, Any]], failures: list[dict[str, str]]
) -> Path:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    path = DOC_ROOT / "part3b-key-findings.md"
    lines = [
        "# Part 3b — flexible NHC d2-map/d3-map screen",
        "",
        "## Scope",
        "",
        "This screen applies the maintained dmap d2-map and d3-map workflows to the published flexible NHC Cu(I) complexes 3a–3d from [Zhao et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202318703). The deposited crystal structures were retrieved from CCDC Access Structures using the article DOI; the original CIFs and SHA-256 provenance are stored under [`03-chemical-validation/part3/3b/inputs/ccdc`](../../03-chemical-validation/part3/3b/inputs/ccdc).",
        "",
        "The five analysis inputs are 3a, 3b-syn, 3b-anti, 3c, and 3d. The 3b syn/anti pair is retained because the paper reports them separately; 3c and 3d use the first deposited independent molecule while the additional independent molecules are listed in `structure-manifest.json` and are not silently averaged.",
        "",
        "The IMe-derived hidden steric-contribution extension is documented separately in [`part3b-ime-hidden-contribution.md`](part3b-ime-hidden-contribution.md), with its own d2/d3 bundles and machine-readable summaries.",
        "",
        "## Method and conventions",
        "",
        "- GFN2-xTB tight preoptimization followed by CREST 3.0.2 quick sampling, using xTB 6.7.1 from `.micromamba/part3c`; CREST trajectory fallback is enabled if a conventional conformer ensemble cannot be completed.",
        "- Neutral singlet [Cu(NHC)Cl] setup, consistent with a formal Cu(I) d10 complex; Cu–C(carbene) and Cu–Cl contacts are protected, and Cu is frozen during sampling.",
        "- The Cu centre is the origin. The reported +z axis points opposite the deposited Cu→Cl vector (vacancy-facing); +x is defined by the first N donor after projection. This sign convention controls axial/topographic interpretation but not the spherical static `%Vbur` scalar.",
        "- SambVca 2.1-compatible radii are used for `%Vbur`; the Cu centre and opposing ancillary chloride remain in the geometry but are excluded from the ligand steric atom set. dmap van-der-Waals radii are used for fields and the method-equivalent Solid-G value. Every project records both profiles in `project.d3map.json` and its result manifest.",
        "- The first pass uses a 3.5 Å sphere, 8,192 angular directions, 0.20 Å buried-volume spacing, and 0.25 Å field/topographic spacing. These are screening settings, not a claim of exact reproduction of the paper’s Cavallo calculation.",
        "",
        "## Literature anchors",
        "",
        "The article reports deposited-geometry `%Vbur` values of 52.7% (3a), 49.8% (3b-syn), 37.3% (3b-anti), 40.2% (3c), and 38.9% (3d), with IMes and IPr reference values of 37.9% and 47.6%, respectively. The computed values below are dmap/SambVca-compatible outputs under the declared profile and should be compared as method-aware descriptors rather than assumed to be byte-for-byte reproductions of the literature numbers.",
        "",
        "## Key findings from completed calculations",
        "",
    ]
    if d2_rows:
        computed = [row["static_vbur_percent"] for row in d2_rows if row["static_vbur_percent"] is not None]
        if computed:
            lines.append(f"1. The dmap fixed-input SambVca-compatible `%Vbur` values span {_fmt(min(computed))}–{_fmt(max(computed))}% across {len(computed)} completed inputs. The corresponding method-equivalent Solid-G values are reported alongside them in the CSV and project JSON outputs.")
        shifts = [row["d2_mean_vbur_percent"] - row["static_vbur_percent"] for row in d2_rows if row["d2_mean_vbur_percent"] is not None and row["static_vbur_percent"] is not None]
        if shifts:
            lines.append(f"2. Across the completed sampled branches, the mean d2-map `%Vbur` shift from the deposited input is {_fmt(np.mean(shifts))} percentage points; this is a conformational-sampling diagnostic, not an activity prediction.")
        largest = max(d2_rows, key=lambda row: abs(float(row["static_vbur_percent"] - row["paper_vbur_percent"])))
        lines.append(f"3. The largest absolute difference between the dmap fixed-input `%Vbur` and the paper anchor in this set is {_fmt(float(largest['static_vbur_percent']) - float(largest['paper_vbur_percent']))} percentage points for {largest['name']}; this flags profile/frame/input-processing sensitivity that should be revisited in a high-resolution validation pass.")
    if d3_rows:
        largest_v = max(d3_rows, key=lambda row: abs(float(row["static_delta_vbur_percent"])))
        largest_g = max(d3_rows, key=lambda row: abs(float(row["static_delta_g_percent"])))
        lines.extend([
            f"4. In the completed d3-map pairs, the largest absolute static `%Vbur` contrast is {_fmt(largest_v['static_delta_vbur_percent'])} percentage points ({largest_v['extended']} minus {largest_v['reference']}); the largest absolute static Solid-G contrast is {_fmt(largest_g['static_delta_g_percent'])} percentage points ({largest_g['extended']} minus {largest_g['reference']}).",
            "5. d3-map stores signed extended-minus-reference differences for both the sampled means and fixed-input baselines. The d3 rows therefore distinguish a change attributable to deposited geometry from a change that persists after xTB/CREST sampling.",
        ])
    else:
        lines.append("No d3-map pair completed in this run; once the d2 branches are available, the cached d3 comparisons can be generated without resampling.")
    lines.extend([
        "",
        "## d2-map scalar results",
        "",
        "| Input | CCDC | Paper `%Vbur` | dmap input `%Vbur` | dmap input G | d2 mean `%Vbur` | d2 mean G | Frames | Mode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in d2_rows:
        lines.append(f"| {row['name']} | {row['ccdc']} | {_fmt(row['paper_vbur_percent'])} | {_fmt(row['static_vbur_percent'])} | {_fmt(row['static_g_percent'])} | {_fmt(row['d2_mean_vbur_percent'])} | {_fmt(row['d2_mean_g_percent'])} | {row['frames']} | {row['analysis_mode']} |")
    lines.extend([
        "",
        "## d3-map scalar differences",
        "",
        "Differences are `extended − reference`.",
        "",
        "| Reference | Extended | Static Δ`%Vbur` | Static ΔG | d3 Δmean `%Vbur` | d3 Δmean G | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in d3_rows:
        lines.append(f"| {row['reference']} | {row['extended']} | {_fmt(row['static_delta_vbur_percent'])} | {_fmt(row['static_delta_g_percent'])} | {_fmt(row['d3_delta_mean_vbur_percent'])} | {_fmt(row['d3_delta_mean_g_percent'])} | {row['status']} |")
    lines.extend([
        "",
        "## Reproducibility and limitations",
        "",
        "Result bundles are in [`03-chemical-validation/part3/3b`](../../03-chemical-validation/part3/3b). Machine-readable summaries are `d2-map-summary.csv` and `d3-map-summary.csv`; `run-manifest.json` records executable paths, the shared profile, completed jobs, and failures.",
        "",
        "The NHC examples are Cu complexes, not isolated free NHC ligands. GFN2-xTB/CREST treatment of transition-metal coordination is therefore the primary methodological risk. The protected-contact setup is deliberate and auditable, but it does not replace a metal-specific electronic-structure validation. A follow-up should repeat the complete set at the high-resolution profile and inspect the retained Cu–C/Cu–Cl distances, topology screening, and conformer populations before drawing catalytic conclusions.",
    ])
    if failures:
        lines.extend(["", "## Incomplete jobs", ""])
        lines.extend(f"- `{failure['project']}`: {failure['error']}" for failure in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_run_manifest(records: list[StructureRecord], d2_rows: list[dict[str, Any]], d3_rows: list[dict[str, Any]], failures: list[dict[str, str]]) -> Path:
    payload = {
        "schema": "part3b.run-manifest.v1",
        "paper_doi": "10.1002/anie.202318703",
        "profile": SHARED_PROFILE,
        "xtb_executable": str(XTB_EXECUTABLE),
        "crest_executable": str(CREST_EXECUTABLE),
        "record_count": len(records),
        "d2_completed": len(d2_rows),
        "d3_completed": len(d3_rows),
        "failures": failures,
        "d2_summary": "d2-map-summary.csv",
        "d3_summary": "d3-map-summary.csv",
        "findings": str((DOC_ROOT / "part3b-key-findings.md").relative_to(REPO_ROOT)),
        "records": [asdict(record) | {"source_cif": str(record.source_cif), "analysis_input": str(record.analysis_input), "donor_atoms": list(record.donor_atoms)} for record in records],
    }
    path = RESULTS / "run-manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-only", action="store_true")
    parser.add_argument("--d3-only", action="store_true")
    parser.add_argument("--limit", type=int, help="limit each requested job class for a pilot")
    args = parser.parse_args()
    if args.d2_only and args.d3_only:
        parser.error("--d2-only and --d3-only are mutually exclusive")
    records = _prepare_records()
    d2_rows: list[dict[str, Any]] = []
    d3_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if not args.d3_only:
        jobs = records if args.limit is None else records[: args.limit]
        for index, record in enumerate(jobs, start=1):
            name = _d2_name(record)
            print(f"[d2 {index}/{len(jobs)}] {record.name}", flush=True)
            try:
                project = _run_project("d2-map", name, record)
                d2_rows.append(_d2_summary(record, project))
            except (DMapError, KeyError, OSError, RuntimeError, ValueError) as error:
                failures.append({"project": f"d2-map/{name}", "error": str(error)})
                print(f"  FAILED: {error}", flush=True)
    if not args.d2_only:
        by_name = {record.name: record for record in records}
        pairs = [
            (by_name["3a"], by_name["3b-syn"]),
            (by_name["3a"], by_name["3b-anti"]),
            (by_name["3a"], by_name["3c"]),
            (by_name["3a"], by_name["3d"]),
            (by_name["3b-syn"], by_name["3b-anti"]),
        ]
        if args.limit is not None:
            pairs = pairs[: args.limit]
        for index, (reference, extended) in enumerate(pairs, start=1):
            name = _d3_name(reference, extended)
            print(f"[d3 {index}/{len(pairs)}] {reference.name} -> {extended.name}", flush=True)
            try:
                project = _run_project("d3-map", name, reference, extended)
                d3_rows.append(_d3_summary(reference, extended, project))
            except (DMapError, KeyError, OSError, RuntimeError, ValueError) as error:
                failures.append({"project": f"d3-map/{name}", "error": str(error)})
                print(f"  FAILED: {error}", flush=True)
    if args.d3_only:
        d2_rows = _read_summary_csv(RESULTS / "d2-map-summary.csv")
    if args.d2_only:
        d3_rows = _read_summary_csv(RESULTS / "d3-map-summary.csv")
    _write_csv(RESULTS / "d2-map-summary.csv", d2_rows)
    _write_csv(RESULTS / "d3-map-summary.csv", d3_rows)
    findings = _write_findings(records, d2_rows, d3_rows, failures)
    _write_run_manifest(records, d2_rows, d3_rows, failures)
    print(json.dumps({"structures": len(records), "d2_completed": len(d2_rows), "d3_completed": len(d3_rows), "failures": len(failures), "findings": str(findings)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
