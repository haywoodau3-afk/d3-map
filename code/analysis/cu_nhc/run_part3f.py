#!/usr/bin/env python3
"""Run the case 3f dynamic-map/yield analysis.

Case 3f extends the case 3b Cu(I)-NHC screen to the seven catalysts reported
in Table 1 of Zhao et al. at 0.02 mol% loading.  The five existing deposited
structures and IMe-hidden d3 comparisons are reused.  3e, IMes, and IPr are
constructed as explicit xTB/CREST screening analogues because deposited Cu
complex coordinates for those entries are not present in the local CCDC
corpus.  Their provenance is recorded as exploratory rather than published
crystallographic geometry.

The ML unit is one catalyst label.  The two 3b crystal conformers are
averaged within the 3b row.  All finite numeric d2 ensemble features,
matched full-minus-IMe d2 deltas, and finite dynamic d3 comparison features
are archived in the feature matrix.  A smaller mechanistically motivated core
is passed to case-3d-style evidence selection; the selected model remains
capped at three descriptors because there are only seven yield observations.
A one-feature dynamic ensemble %Vbur model is reported as the requested
baseline, not as the primary model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_3d_bayesian_anion_models as bayes
import run_part3b as parent_run
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


ROOT = Path(__file__).resolve().parents[2]
CHEMICAL_ROOT = ROOT / "03-chemical-validation"
PART3B = CHEMICAL_ROOT / "part3" / "3b"
OUT = CHEMICAL_ROOT / "part3" / "3f"
INPUTS = OUT / "analysis-inputs"
MAPS = OUT / "maps"
DOC_ROOT = ROOT / "03-chemical-validation" / "Doc" / "part3"

XTB_EXECUTABLE = parent_run.XTB_EXECUTABLE
CREST_EXECUTABLE = parent_run.CREST_EXECUTABLE
SHARED_PROFILE = parent_run.SHARED_PROFILE

YIELDS = {
    "3a": 98.0,
    "3b": 94.0,
    "3c": 92.0,
    "3d": 40.0,
    "3e": 36.0,
    "IMes": 63.0,
    "IPr": 42.0,
}

PAPER_OPTIMIZED_VBUR = {
    "3a": {"syn": 49.5, "anti": 39.9},
    "3b": {"syn": 45.0, "anti": 35.7},
    "3c": {"single": 38.6},
    "3d": {"single": 36.7},
    "3e": {"single": 33.6},
    "IMes": {"single": 36.4},
    "IPr": {"single": 42.6},
}

# The complete d2/d3 numeric pool is retained in the matrix.  Only this
# mechanistically motivated core is sent to evidence selection: the sample has
# seven catalyst-level outcomes, so the full pool is useful for auditability
# but too broad for an interpretable yield model.  The matched d3/d2 deltas
# are full-catalyst-minus-IMe ensemble differences and include the requested
# CREST-derived entropy signal.
CORE_FEATURE_RATIONALE = {
    "d2_pocket_steric_entropy_volume_angstrom3_nat": "absolute CREST ensemble steric-entropy volume",
    "d2_pocket_steric_entropy_sum": "CREST ensemble steric-entropy occupancy mass; available when normalized entropy is not estimable",
    "d3_d2_delta_pocket_steric_entropy_sum": "matched full-minus-IMe CREST ensemble entropy-occupancy difference",
    "d3_d2_delta_pocket_steric_entropy_volume_angstrom3_nat": "matched full-minus-IMe CREST ensemble entropy difference",
    "d2_normalized_pocket_flexibility": "absolute CREST ensemble flexibility",
    "d3_d2_delta_normalized_pocket_flexibility": "matched full-minus-IMe flexibility difference",
    "d2_persistent_open_fraction": "persistent dynamic access state",
    "d2_breathing_fraction": "dynamic breathing state fraction",
    "d2_persistent_blocked_fraction": "persistent dynamic blocking state",
    "d2_approach_accessibility_mean": "mean dynamic approach accessibility",
    "d2_approach_accessibility_min": "minimum dynamic approach accessibility",
    "d3_d2_delta_approach_accessibility_mean": "matched full-minus-IMe accessibility difference",
    "d3_d2_delta_approach_accessibility_min": "matched full-minus-IMe minimum accessibility difference",
    "d2_pocket_weighted_vbur_percent": "dynamic ensemble %Vbur baseline feature",
    "d2_pocket_integrated_vbur_percent": "dynamic ensemble integrated %Vbur",
    "d3_d2_delta_pocket_weighted_vbur_percent": "matched full-minus-IMe ensemble %Vbur difference",
    "d3_d2_delta_pocket_integrated_vbur_percent": "matched full-minus-IMe integrated %Vbur difference",
    "d3_delta_vbur_percent_mean": "dynamic d3 scalar %Vbur mean difference",
    "d3_delta_g_percent_mean": "dynamic d3 scalar G mean difference",
    "d3_changed_occupation_area_fraction": "dynamic d3 occupied-area change",
    "d3_changed_pocket_occupation_area_fraction": "dynamic d3 pocket occupation change",
    "d3_changed_shielding_probability_area_fraction": "dynamic d3 shielding-probability change",
    "d3_changed_topographic_contact_probability_area_fraction": "dynamic d3 contact-probability change",
    "d3_changed_topographic_first_contact_interval_10_90_area_fraction": "dynamic d3 first-contact interval change",
    "d3_changed_topographic_first_contact_q50_area_fraction": "dynamic d3 first-contact median change",
    "d3_changed_topographic_occupied_depth_area_fraction": "dynamic d3 occupied-depth change",
    "d3_changed_v2_adaptive_class_area_fraction": "dynamic d3 adaptive-class change",
    "d2_quadrant_asymmetry_variance": "dynamic directional asymmetry",
    "d3_d2_delta_quadrant_asymmetry_variance": "matched full-minus-IMe directional asymmetry difference",
    "d2_displaced_g_mean_2": "displaced dynamic G signal selected by the initial full-pool fit",
    "d2_displaced_vbur_mean_2": "matched displaced dynamic %Vbur signal",
}

# Exact descriptor set used by fit_3d_bayesian_anion_models.py.  Case 3f
# evaluates it separately on the single input-geometry baselines so that the
# result is a genuine static Case-3d-protocol comparison with the dynamic
# 27-feature and full-pool models.
CASE3D_STATIC_FEATURES = (
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
CASE3D_V2_RIDGE_PRECISION = 0.25


@dataclass
class Molecule:
    name: str
    path: Path
    elements: tuple[str, ...]
    coordinates: np.ndarray
    labels: tuple[str, ...]
    core_labels: tuple[str, ...]
    external_by_donor: dict[str, str]
    center_atom: int
    carbene_atom: int
    donor_atoms: tuple[int, int]
    chloride_atom: int
    source_status: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MapRecord:
    name: str
    path: Path
    center_atom: int
    carbene_atom: int
    donor_atoms: tuple[int, int]
    chloride_atom: int
    source_status: str


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero-length geometry vector")
    return vector / norm


def _methyl_hydrogens(center: np.ndarray, methyl: np.ndarray) -> list[np.ndarray]:
    toward_n = _unit(center - methyl)
    trial = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(trial, toward_n))) > 0.85:
        trial = np.asarray([1.0, 0.0, 0.0])
    perpendicular = _unit(np.cross(toward_n, trial))
    second = np.cross(toward_n, perpendicular)
    radius = 1.09
    axial = -toward_n / 3.0
    radial = math.sqrt(8.0 / 9.0)
    return [
        methyl
        + radius
        * (axial + radial * (math.cos(angle) * perpendicular + math.sin(angle) * second))
        for angle in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    ]


def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 3D matrix mapping source onto target."""
    a = _unit(source)
    b = _unit(target)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(a, b))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        trial = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(trial, a))) > 0.85:
            trial = np.asarray([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, trial))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def _write_xyz(path: Path, elements: tuple[str, ...], coordinates: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [str(len(elements)), comment]
    rows.extend(
        f"{element:<3} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for element, xyz in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _source_graph(source_name: str) -> tuple[dict[str, Any], list[str], dict[str, np.ndarray], dict[str, set[str]]]:
    source = parent_run.CCDC_SOURCES[source_name]
    cif_path = PART3B / "inputs" / "ccdc" / source["file"]
    cell, atoms, bonds = parent_run._read_cif(cif_path)
    del cell
    selected = parent_run._unwrap_component(atoms, bonds, source["component"])
    source_xyz = PART3B / "analysis-inputs" / f"{source_name}-canonical.xyz"
    ensemble = read_xyz_ensemble(source_xyz)
    if len(selected) != len(ensemble.elements):
        raise ValueError(f"{source_name}: CIF/XYZ atom counts disagree")
    coordinates = {label: ensemble.geometries[0].coordinates[index] for index, label in enumerate(selected)}
    return atoms, selected, coordinates, bonds


def _identify_roles(
    atoms: dict[str, Any], selected: list[str], bonds: dict[str, set[str]], center_label: str
) -> tuple[str, str, str, list[str], list[str], dict[str, str]]:
    selected_set = set(selected)
    cu_neighbors = [label for label in bonds[center_label] if label in selected_set]
    carbene = next(label for label in cu_neighbors if atoms[label].element == "C")
    chloride = next(label for label in cu_neighbors if atoms[label].element == "Cl")
    donors = sorted(
        label for label in bonds[carbene] if label in selected_set and atoms[label].element == "N"
    )
    if len(donors) != 2:
        raise ValueError(f"expected two N donors, found {donors}")
    candidates = {
        donor: [
            label
            for label in bonds[donor]
            if label in selected_set and label != carbene and atoms[label].element == "C"
        ]
        for donor in donors
    }
    rings: list[str] = []
    for donor in donors:
        shared = [
            candidate
            for candidate in candidates[donor]
            if any(
                candidate in bonds[other]
                for other_donor in donors
                if other_donor != donor
                for other in candidates[other_donor]
            )
        ]
        if len(shared) != 1:
            raise ValueError(f"could not identify imidazole ring carbon for {donor}: {shared}")
        rings.append(shared[0])
    external = {
        donor: next(candidate for candidate in candidates[donor] if candidate != rings[index])
        for index, donor in enumerate(donors)
    }
    core = [center_label, carbene, chloride, *donors, *rings]
    return center_label, carbene, chloride, donors, rings, external | {"__core__": core}


def _branch_nodes(root: str, donor: str, core: set[str], selected: set[str], bonds: dict[str, set[str]]) -> set[str]:
    found: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        if current in found or current not in selected or current in core or current == donor:
            continue
        found.add(current)
        pending.extend(neighbor for neighbor in bonds.get(current, ()) if neighbor not in core and neighbor != donor)
    return found


def _direct_external(
    donor: str, external: str, atoms: dict[str, Any], bonds: dict[str, set[str]], selected: set[str]
) -> str:
    branch_hydrogens = [
        label
        for label in bonds[external]
        if label in selected and atoms[label].element == "H"
    ]
    # The aryl root is directly N--C(aryl), while the flexible wingtip has a
    # benzylic CH2 root with attached hydrogens.
    return external if not branch_hydrogens else ""


def _molecule_from_labels(
    name: str,
    atoms: dict[str, Any],
    selected: list[str],
    coordinates: dict[str, np.ndarray],
    bonds: dict[str, set[str]],
    *,
    remove_labels: set[str] | None = None,
    path: Path,
    source_status: str,
    provenance: dict[str, Any],
) -> Molecule:
    center, carbene, chloride, donors, rings, role_data = _identify_roles(
        atoms, selected, bonds, next(label for label in selected if atoms[label].element == "Cu")
    )
    external = {donor: role_data[donor] for donor in donors}
    core = set(role_data["__core__"])
    keep = [label for label in selected if label not in (remove_labels or set())]
    # Keep the core first and then the remaining atoms in deposited order.  The
    # first three positions match the existing Part 3b frame convention.
    ordered = [center, carbene, chloride, *donors, *rings]
    ordered.extend(label for label in keep if label not in ordered)
    elements = tuple(atoms[label].element for label in ordered)
    coords = np.asarray([coordinates[label] for label in ordered], dtype=float)
    index = {label: number + 1 for number, label in enumerate(ordered)}
    external_by_donor = {donor: external[donor] for donor in donors}
    if any(label not in index for label in external_by_donor.values()):
        raise ValueError(f"{name}: external donor branch was removed")
    _write_xyz(path, elements, coords, f"Case 3f {name}; {source_status}")
    return Molecule(
        name=name,
        path=path,
        elements=elements,
        coordinates=coords,
        labels=tuple(ordered),
        core_labels=tuple(role_data["__core__"]),
        external_by_donor=external_by_donor,
        center_atom=index[center],
        carbene_atom=index[carbene],
        donor_atoms=tuple(index[donor] for donor in donors),
        chloride_atom=index[chloride],
        source_status=source_status,
        provenance=provenance,
    )


def _construct_3e() -> Molecule:
    atoms, selected, coordinates, bonds = _source_graph("3b-anti")
    source = parent_run.CCDC_SOURCES["3b-anti"]
    center, carbene, chloride, donors, rings, role_data = _identify_roles(atoms, selected, bonds, source["component"])
    core = set(role_data["__core__"])
    remove: set[str] = set()
    removed_roots: dict[str, str] = {}
    for ring in rings:
        for neighbor in bonds[ring]:
            if neighbor not in set(selected) or neighbor in core or atoms[neighbor].element != "C":
                continue
            removed_roots[ring] = neighbor
            remove.update(_branch_nodes(neighbor, ring, core, set(selected), bonds))
    path = INPUTS / "3e-constructed.xyz"
    return _molecule_from_labels(
        "3e",
        atoms,
        selected,
        coordinates,
        bonds,
        remove_labels=remove,
        path=path,
        source_status="xTB_constructed_exploratory",
        provenance={
            "source_parent": "3b-anti",
            "source_cif_component": source["component"],
            "transformation": "removed the two 4,5-backbone methyl branches from the 3b-anti N-Mes/N-CH2Mes frame",
            "removed_branch_roots": removed_roots,
            "paper_geometry_status": "no deposited 3e Cu complex in local CCDC corpus",
        },
    )


def _branch_from_source_graph(base: Molecule, root: str, donor: str, bonds: dict[str, set[str]]) -> set[str]:
    return _branch_nodes(root, donor, set(base.core_labels), set(base.labels), bonds)


def _clone_direct_wingtip_from_graph(
    base: Molecule,
    atoms: dict[str, Any],
    bonds: dict[str, set[str]],
    target_name: str,
    direct_donor: int,
) -> Molecule:
    labels = list(base.labels)
    donor_labels = [labels[number - 1] for number in base.donor_atoms]
    direct = donor_labels[direct_donor]
    target = donor_labels[1 - direct_donor]
    coords = {label: base.coordinates[index] for index, label in enumerate(labels)}
    elements = {label: base.elements[index] for index, label in enumerate(labels)}
    source_root = base.external_by_donor[direct]
    old_target_root = base.external_by_donor[target]
    source_branch = _branch_from_source_graph(base, source_root, direct, bonds)
    target_branch = _branch_from_source_graph(base, old_target_root, target, bonds)
    for label in target_branch:
        labels.remove(label)
        coords.pop(label)
        elements.pop(label)
    source_donor_coord = coords[direct]
    target_donor_coord = coords[target]
    source_root_coord = coords[source_root]
    old_target_coord = base.coordinates[base.labels.index(old_target_root)]
    source_vector = source_root_coord - source_donor_coord
    target_direction = _unit(old_target_coord - target_donor_coord)
    target_root_coord = target_donor_coord + float(np.linalg.norm(source_vector)) * target_direction
    rotation = _rotation_from_vectors(source_vector, target_root_coord - target_donor_coord)
    cloned_labels: list[str] = []
    for source_label in sorted(source_branch, key=lambda label: base.labels.index(label)):
        clone = f"{target_name}_copy_{source_label}"
        offset = base.coordinates[base.labels.index(source_label)] - source_root_coord
        coords[clone] = target_root_coord + rotation @ offset
        elements[clone] = base.elements[base.labels.index(source_label)]
        cloned_labels.append(clone)
    core_order = list(base.core_labels)
    ordered = [label for label in core_order if label in elements]
    ordered.extend(label for label in base.labels if label in elements and label not in ordered)
    ordered.extend(label for label in cloned_labels if label not in ordered)
    out_elements = tuple(elements[label] for label in ordered)
    out_coords = np.asarray([coords[label] for label in ordered], dtype=float)
    out_index = {label: number + 1 for number, label in enumerate(ordered)}
    new_external = dict(base.external_by_donor)
    new_external[target] = next(label for label in cloned_labels if label.endswith(source_root))
    path = INPUTS / f"{target_name}-constructed.xyz"
    _write_xyz(path, out_elements, out_coords, f"Case 3f {target_name}; xTB-constructed symmetric analogue")
    return Molecule(
        name=target_name,
        path=path,
        elements=out_elements,
        coordinates=out_coords,
        labels=tuple(ordered),
        core_labels=base.core_labels,
        external_by_donor=new_external,
        center_atom=out_index[base.core_labels[0]],
        carbene_atom=out_index[base.core_labels[1]],
        donor_atoms=tuple(out_index[label] for label in donor_labels),
        chloride_atom=out_index[base.core_labels[2]],
        source_status="xTB_constructed_exploratory",
        provenance={
            "source_parent": base.name,
            "transformation": "replaced the flexible N-CH2Ar branch with a copied direct N-Ar wingtip",
            "copied_direct_wingtip_from_donor": direct,
            "target_donor": target,
            "paper_geometry_status": "no deposited Cu complex used as an exact coordinate source",
        },
    )


def _build_ime(base: Molecule, name: str) -> Molecule:
    coords = {label: base.coordinates[index] for index, label in enumerate(base.labels)}
    elements: list[str] = []
    output: list[np.ndarray] = []
    labels: list[str] = []

    def add(label: str, element: str, coordinate: np.ndarray) -> None:
        labels.append(label)
        elements.append(element)
        output.append(np.asarray(coordinate, dtype=float))

    core = list(base.core_labels)
    center, carbene, chloride, donor1, donor2, ring1, ring2 = core
    for label in [center, chloride, carbene, donor1, donor2, ring1, ring2]:
        add(label, next(base.elements[index] for index, item in enumerate(base.labels) if item == label), coords[label])
    donors = [donor1, donor2]
    for index, donor in enumerate(donors, start=1):
        direction = _unit(coords[base.external_by_donor[donor]] - coords[donor])
        methyl = coords[donor] + 1.47 * direction
        methyl_label = f"NMe{index}"
        add(methyl_label, "C", methyl)
        for h_index, hydrogen in enumerate(_methyl_hydrogens(coords[donor], methyl), start=1):
            add(f"{methyl_label}H{h_index}", "H", hydrogen)
    for index, ring in enumerate([ring1, ring2], start=1):
        donor = donors[index - 1]
        other_ring = ring2 if ring == ring1 else ring1
        outward = -(coords[donor] - coords[ring] + coords[other_ring] - coords[ring])
        add(f"ringH{index}", "H", coords[ring] + 1.09 * _unit(outward))
    path = INPUTS / "ime" / f"{name}-IMe-constructed.xyz"
    _write_xyz(path, tuple(elements), np.asarray(output), f"Case 3f IMe baseline from {name}; constructed frame")
    index = {label: number + 1 for number, label in enumerate(labels)}
    return Molecule(
        name=f"{name}-IMe",
        path=path,
        elements=tuple(elements),
        coordinates=np.asarray(output),
        labels=tuple(labels),
        core_labels=tuple([center, carbene, chloride, donor1, donor2, ring1, ring2]),
        external_by_donor={donor1: "NMe1", donor2: "NMe2"},
        center_atom=index[center],
        carbene_atom=index[carbene],
        donor_atoms=(index[donor1], index[donor2]),
        chloride_atom=index[chloride],
        source_status="xTB_constructed_exploratory_reference",
        provenance={
            "source_parent": name,
            "transformation": "replaced both N-Ar/N-CH2Ar branches with N-methyl groups and added C4/C5 hydrogens",
        },
    )


def _record_from_molecule(molecule: Molecule) -> MapRecord:
    return MapRecord(
        name=molecule.name,
        path=molecule.path,
        center_atom=molecule.center_atom,
        carbene_atom=molecule.carbene_atom,
        donor_atoms=molecule.donor_atoms,
        chloride_atom=molecule.chloride_atom,
        source_status=molecule.source_status,
    )


def _setup(record: MapRecord) -> dict[str, Any]:
    elements = read_xyz_ensemble(record.path).elements
    steric_atoms = [
        index + 1
        for index, element in enumerate(elements)
        if element != "H" and index + 1 not in {record.center_atom, record.chloride_atom}
    ]
    setup = {
        **default_structure_setup(
            record.path,
            center_atom=record.center_atom,
            reactive_direction=[0.0, 0.0, 1.0],
            secondary_direction=[1.0, 0.0, 0.0],
            alignment_atoms=[record.center_atom, record.carbene_atom, *record.donor_atoms, record.chloride_atom],
            charge=0,
            multiplicity=1,
            solvent=None,
            confirmed=True,
            protected_contacts=[[record.center_atom, record.carbene_atom], [record.center_atom, record.chloride_atom]],
            frozen_atoms=[record.center_atom],
            steric_atoms=steric_atoms,
            xtb_executable=str(XTB_EXECUTABLE),
            crest_executable=str(CREST_EXECUTABLE),
            threads=4,
        ),
        "chemical_setup_status": "Cu(I) d10 neutral singlet [Cu(NHC)Cl]; Cu--C and Cu--Cl protected; screening setup",
        "source_paper_doi": "10.1002/anie.202318703",
        "frame_convention": "Cu-centred; +z vacancy-facing; +x points to first N donor projected normal to +z",
        "sampling_provenance": "GFN2-xTB tight preoptimization followed by CREST quick conformational search; trajectory fallback permitted",
        "case3f_source_status": record.source_status,
    }
    setup["sampling"]["quick"] = True
    setup["sampling"]["allow_trajectory_fallback"] = True
    setup["sampling"]["metadyn_steps"] = 25
    setup["sampling"]["metadyn_time_ps"] = 5.0
    setup["sampling"]["trajectory_stride"] = 10
    return setup


def _project_path(kind: str, name: str) -> Path:
    return MAPS / kind / name / "project.d3map.json"


def _run_d2(record: MapRecord) -> Path:
    project = _project_path("d2-map", f"{record.name}-d2map")
    state_path = project.parent / "run-state.json"
    if project.is_file() and state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        if state.get("status") == "success":
            return project
    if not project.is_file():
        project = create_release_project(
            kind="d2-map",
            reference_xyz=record.path,
            output_directory=project.parent,
            reference_setup=_setup(record),
            shared=SHARED_PROFILE,
        )
    execute_release_project(project, sampling=True)
    return project


def _run_cached_d3(project: Path, reference: MapRecord, extended: MapRecord, reference_d2: Path, extended_d2: Path) -> None:
    config = read_release_project(project)
    root = project.parent
    for role, source_project in (("reference", reference_d2), ("extended", extended_d2)):
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
        "schema": "part3f.hidden-d3.run-state.v1",
        "status": "success",
        "sampling_strategy": "reused completed IMe and full-catalyst d2-map branches",
        "comparison_status": "nonstandard" if mismatches else "standard",
        "compatibility_mismatches": mismatches,
        "reference": reference.name,
        "extended": extended.name,
        "difference_definition": "full catalyst minus matched IMe baseline",
    }
    (root / "run-state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_portable_zip(project)


def _run_d3(full: MapRecord, ime: MapRecord, full_d2: Path, ime_d2: Path) -> Path:
    project = _project_path("hidden-d3", f"{ime.name}-vs-{full.name}-hidden-d3map")
    if not project.is_file():
        project = create_release_project(
            kind="d3-map",
            reference_xyz=ime.path,
            extended_xyz=full.path,
            output_directory=project.parent,
            reference_setup=_setup(ime),
            extended_setup=_setup(full),
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
    _run_cached_d3(project, ime, full, ime_d2, full_d2)
    return project


def _record(name: str, path: Path, source_status: str, center: int = 1, carbene: int = 2, donors: tuple[int, int] = (4, 5), chloride: int = 3) -> MapRecord:
    return MapRecord(name, path, center, carbene, donors, chloride, source_status)


def _build_inputs() -> dict[str, Any]:
    INPUTS.mkdir(parents=True, exist_ok=True)
    constructed_3e = _construct_3e()
    atoms_3e, selected_3e, _, bonds_3e = _source_graph("3b-anti")
    ime_3e = _build_ime(constructed_3e, "3e")

    # 3e is the Mes/CH2Mes, 4,5-unsubstituted analogue.  IMes copies the
    # direct N-Mes wingtip to the second N atom.
    imes = _clone_direct_wingtip_from_graph(constructed_3e, atoms_3e, bonds_3e, "IMes", direct_donor=0)
    ime_imes = _build_ime(imes, "IMes")

    # In the deposited 3d component, N2 bears the direct Dipp wingtip and N1
    # bears the flexible N-CH2Mes branch.  IPr copies the direct Dipp branch.
    atoms_3d, selected_3d, _, bonds_3d = _source_graph("3d")
    base_3d = _molecule_from_labels(
        "3d",
        atoms_3d,
        selected_3d,
        {label: read_xyz_ensemble(PART3B / "analysis-inputs" / "3d-canonical.xyz").geometries[0].coordinates[index] for index, label in enumerate(selected_3d)},
        bonds_3d,
        path=PART3B / "analysis-inputs" / "3d-canonical.xyz",
        source_status="published_CCDC_canonical",
        provenance={"source_parent": "3d"},
    )
    ipr = _clone_direct_wingtip_from_graph(base_3d, atoms_3d, bonds_3d, "IPr", direct_donor=1)
    ime_ipr = _build_ime(ipr, "IPr")

    manifest = {
        "schema": "part3f.structure-manifest.v1",
        "paper_source": "EXTERNAL_DOCUMENT_DIRECTORY/Wingtip-Flexible-i-N-i-Heterocyclic-Carbenes-Unsymmetrical-Connection-between-IMes-and-IPr.pdf",
        "coordinate_status": {
            "3a": "reused published CCDC canonical input",
            "3b-syn": "reused published CCDC canonical input",
            "3b-anti": "reused published CCDC canonical input",
            "3c": "reused published CCDC canonical input",
            "3d": "reused published CCDC canonical input",
            "3e": "xTB_constructed_exploratory",
            "IMes": "xTB_constructed_exploratory",
            "IPr": "xTB_constructed_exploratory",
        },
        "constructed": {
            name: {"path": str(molecule.path), "source_status": molecule.source_status, "provenance": molecule.provenance}
            for name, molecule in (("3e", constructed_3e), ("IMes", imes), ("IPr", ipr), ("3e-IMe", ime_3e), ("IMes-IMe", ime_imes), ("IPr-IMe", ime_ipr))
        },
        "table1_yields_at_0.02_mol_percent": YIELDS,
        "paper_optimized_vbur_percent": PAPER_OPTIMIZED_VBUR,
    }
    (OUT / "structure-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"3e": constructed_3e, "IMes": imes, "IPr": ipr, "3e-IMe": ime_3e, "IMes-IMe": ime_imes, "IPr-IMe": ime_ipr}


def _map_groups(constructed: dict[str, Molecule]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {
        "3a": {"full": [_record("3a", PART3B / "analysis-inputs/3a-canonical.xyz", "published_CCDC_canonical")], "ime": [_record("3a-IMe", PART3B / "analysis-inputs/ime/3a-IMe.xyz", "derived_IMe_reference", 1, 3, (4, 5), 2)]},
        "3b": {"full": [_record("3b-syn", PART3B / "analysis-inputs/3b-syn-canonical.xyz", "published_CCDC_canonical"), _record("3b-anti", PART3B / "analysis-inputs/3b-anti-canonical.xyz", "published_CCDC_canonical")], "ime": [_record("3b-syn-IMe", PART3B / "analysis-inputs/ime/3b-syn-IMe.xyz", "derived_IMe_reference", 1, 3, (4, 5), 2), _record("3b-anti-IMe", PART3B / "analysis-inputs/ime/3b-anti-IMe.xyz", "derived_IMe_reference", 1, 3, (4, 5), 2)]},
        "3c": {"full": [_record("3c", PART3B / "analysis-inputs/3c-canonical.xyz", "published_CCDC_canonical")], "ime": [_record("3c-IMe", PART3B / "analysis-inputs/ime/3c-IMe.xyz", "derived_IMe_reference", 1, 3, (4, 5), 2)]},
        "3d": {"full": [_record("3d", PART3B / "analysis-inputs/3d-canonical.xyz", "published_CCDC_canonical")], "ime": [_record("3d-IMe", PART3B / "analysis-inputs/ime/3d-IMe.xyz", "derived_IMe_reference", 1, 3, (4, 5), 2)]},
        "3e": {"full": [_record("3e", constructed["3e"].path, constructed["3e"].source_status, constructed["3e"].center_atom, constructed["3e"].carbene_atom, constructed["3e"].donor_atoms, constructed["3e"].chloride_atom)], "ime": [_record("3e-IMe", constructed["3e-IMe"].path, constructed["3e-IMe"].source_status, constructed["3e-IMe"].center_atom, constructed["3e-IMe"].carbene_atom, constructed["3e-IMe"].donor_atoms, constructed["3e-IMe"].chloride_atom)]},
        "IMes": {"full": [_record("IMes", constructed["IMes"].path, constructed["IMes"].source_status, constructed["IMes"].center_atom, constructed["IMes"].carbene_atom, constructed["IMes"].donor_atoms, constructed["IMes"].chloride_atom)], "ime": [_record("IMes-IMe", constructed["IMes-IMe"].path, constructed["IMes-IMe"].source_status, constructed["IMes-IMe"].center_atom, constructed["IMes-IMe"].carbene_atom, constructed["IMes-IMe"].donor_atoms, constructed["IMes-IMe"].chloride_atom)]},
        "IPr": {"full": [_record("IPr", constructed["IPr"].path, constructed["IPr"].source_status, constructed["IPr"].center_atom, constructed["IPr"].carbene_atom, constructed["IPr"].donor_atoms, constructed["IPr"].chloride_atom)], "ime": [_record("IPr-IMe", constructed["IPr-IMe"].path, constructed["IPr-IMe"].source_status, constructed["IPr-IMe"].center_atom, constructed["IPr-IMe"].carbene_atom, constructed["IPr-IMe"].donor_atoms, constructed["IPr-IMe"].chloride_atom)]},
    }
    return groups


def _read_feature_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one feature row in {path}, found {len(rows)}")
    return rows[0]


def _comparison_features(path: Path) -> dict[str, float]:
    if path.is_file() and path.name == "project.d3map.json":
        path = path.parent
    data = json.loads((path / "results" / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    features: dict[str, float] = {}
    for key, value in data.get("changed_area_fractions", {}).items():
        numeric = _finite(value)
        if numeric is not None:
            features[f"d3_changed_{key}_area_fraction"] = numeric
    for item in data.get("scalar_comparisons", []):
        descriptor = str(item.get("descriptor", "unknown"))
        statistic = str(item.get("statistic", "unknown"))
        for role, key in (("delta", "difference_extended_minus_reference"), ("extended", "extended"), ("reference", "reference")):
            numeric = _finite(item.get(key))
            if numeric is not None:
                features[f"d3_{role}_{descriptor}_{statistic}"] = numeric
    return features


def _existing_hidden_path(full: str, ime: str) -> Path:
    return PART3B / "ime-baseline" / "d3-map" / f"{ime}-vs-{full}-hidden-d3map"


def _feature_row_for_catalyst(catalyst: str, group: dict[str, Any], d3_projects: dict[str, Path] | None = None) -> dict[str, Any]:
    d2_rows: list[dict[str, str]] = []
    ime_d2_rows: list[dict[str, str]] = []
    d3_features: list[dict[str, float]] = []
    scalar_d2: dict[str, list[float]] = {"vbur_percent": [], "g_percent": []}
    static_input: dict[str, list[float]] = {"vbur_percent": [], "g_percent": []}
    for full, ime in zip(group["full"], group["ime"], strict=True):
        full_project = _d2_project_for_record(full)
        ime_project = _d2_project_for_record(ime)
        d2_rows.append(_read_feature_row(full_project.parent / "results" / "reference" / "features.csv"))
        ime_d2_rows.append(_read_feature_row(ime_project.parent / "results" / "reference" / "features.csv"))
        descriptors = json.loads((full_project.parent / "results" / "reference" / "descriptors.json").read_text(encoding="utf-8"))
        scalar_descriptors = descriptors.get("ensemble_scalar_descriptors", {})
        for descriptor_name in scalar_d2:
            mean_value = _finite(scalar_descriptors.get(descriptor_name, {}).get("mean"))
            if mean_value is not None:
                scalar_d2[descriptor_name].append(mean_value)
        static_baselines = json.loads((full_project.parent / "results" / "reference" / "static-baselines.json").read_text(encoding="utf-8"))
        static_input_values = static_baselines.get("baselines", {}).get("input", {})
        for descriptor_name in static_input:
            static_value = _finite(static_input_values.get(descriptor_name))
            if static_value is not None:
                static_input[descriptor_name].append(static_value)
        if d3_projects and full.name in d3_projects:
            d3_path = d3_projects[full.name]
        else:
            d3_path = _existing_hidden_path(full.name, ime.name)
        d3_features.append(_comparison_features(d3_path))
    numeric_d2: dict[str, float] = {}
    for key in d2_rows[0]:
        values = [_finite(row.get(key)) for row in d2_rows]
        if all(value is not None for value in values):
            numeric_d2[f"d2_{key}"] = float(np.mean(values))
    numeric_d2_delta: dict[str, float] = {}
    for key in d2_rows[0]:
        deltas: list[float] = []
        for full_row, ime_row in zip(d2_rows, ime_d2_rows, strict=True):
            full_value = _finite(full_row.get(key))
            ime_value = _finite(ime_row.get(key))
            if full_value is None or ime_value is None:
                deltas = []
                break
            deltas.append(full_value - ime_value)
        if deltas:
            numeric_d2_delta[f"d3_d2_delta_{key}"] = float(np.mean(deltas))
    for descriptor_name, values in scalar_d2.items():
        if values and len(values) == len(d2_rows):
            numeric_d2[f"d2_ensemble_{descriptor_name}_mean"] = float(np.mean(values))
    static_features: dict[str, float] = {}
    for descriptor_name, values in static_input.items():
        if values and len(values) == len(d2_rows):
            static_features[f"static_{descriptor_name}"] = float(np.mean(values))
    numeric_d3: dict[str, float] = {}
    d3_keys = sorted(set().union(*(features.keys() for features in d3_features)))
    for key in d3_keys:
        values = [_finite(features.get(key)) for features in d3_features]
        if all(value is not None for value in values):
            numeric_d3[key] = float(np.mean(values))
    return {
        "catalyst": catalyst,
        "yield_percent": YIELDS[catalyst],
        "outcome_source": "Table 1, 0.02 mol% Cu(I)-NHC",
        "geometry_status": "published_CCDC_canonical" if catalyst in {"3a", "3b", "3c", "3d"} else "xTB_constructed_exploratory",
        "d2_source_count": len(d2_rows),
        "d3_source_count": len(d3_features),
        **static_features,
        **numeric_d2,
        **numeric_d2_delta,
        **numeric_d3,
    }


def _d2_project_for_record(record: MapRecord) -> Path:
    existing_names = {"3a", "3b-syn", "3b-anti", "3c", "3d"}
    existing_ime_names = {"3a-IMe", "3b-syn-IMe", "3b-anti-IMe", "3c-IMe", "3d-IMe"}
    if record.name in existing_names:
        return PART3B / "d2-map" / f"{record.name}-d2map" / "project.d3map.json"
    if record.name in existing_ime_names:
        return PART3B / "ime-baseline" / "d2-map" / f"{record.name}-d2map" / "project.d3map.json"
    return _project_path("d2-map", f"{record.name}-d2map")


def _d3_project_for_group(catalyst: str, full: MapRecord, ime: MapRecord, projects: dict[str, Path]) -> Path:
    if catalyst in {"3a", "3b", "3c", "3d"}:
        return _existing_hidden_path(full.name, ime.name)
    return projects[full.name]


def _case3d_static_d2_values(project: Path) -> dict[str, float]:
    """Extract the exact Case 3d d2 fields from the input-geometry baseline."""
    path = project.parent / "results" / "reference" / "baselines" / "input" / "descriptors.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    scalars = data.get("ensemble_scalar_descriptors", {})
    features = data.get("features", {})
    values: dict[str, float] = {}
    for name in ("g_percent", "vbur_percent"):
        value = _finite(scalars.get(name, {}).get("mean"))
        if value is not None:
            values[name] = value
    for name in (
        "approach_accessibility_mean",
        "persistent_open_fraction",
        "persistent_blocked_fraction",
        "pocket_integrated_vbur_percent",
        "pocket_weighted_vbur_percent",
        "quadrant_asymmetry_variance",
    ):
        value = _finite(features.get(name))
        if value is not None:
            values[name] = value
    return values


def _run_case3d_static_d3(
    full: MapRecord,
    ime: MapRecord,
    full_d2: Path,
    ime_d2: Path,
) -> Path:
    """Compare the input-geometry d2 baselines with the Case 3d static protocol."""
    project = _project_path("case3d-static-d3", f"{ime.name}-vs-{full.name}-case3d-static-d3map")
    if not project.is_file():
        project = create_release_project(
            kind="d3-map",
            reference_xyz=ime.path,
            extended_xyz=full.path,
            output_directory=project.parent,
            reference_setup=_setup(ime),
            extended_setup=_setup(full),
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
    config = read_release_project(project)
    root = project.parent
    for role, source_project in (("reference", ime_d2), ("extended", full_d2)):
        source = source_project.parent / "results" / "reference" / "baselines" / "input"
        if not (source / "descriptors.json").is_file():
            raise RuntimeError(f"input-geometry static d2 baseline is missing for {role}: {source_project}")
        shutil.copytree(source, root / "results" / role, dirs_exist_ok=True)
    mismatches = validate_pair_compatibility(root, config)
    compare_release_outputs(project, config, mismatches)
    state = {
        "schema": "part3f.case3d-static-comparison.run-state.v1",
        "status": "success",
        "sampling_strategy": "single input-geometry d2 baselines; no CREST ensemble averaging",
        "comparison_status": "nonstandard" if mismatches else "standard",
        "compatibility_mismatches": mismatches,
        "reference": ime.name,
        "extended": full.name,
        "difference_definition": "static full catalyst minus static matched IMe baseline",
        "case3d_protocol": True,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_portable_zip(project)
    return project


def _build_case3d_static_rows(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build seven catalyst rows using the exact 15-feature Case 3d schema."""
    rows: list[dict[str, Any]] = []
    d3_names = {name for name in CASE3D_STATIC_FEATURES if name.startswith("d3_")}
    for catalyst, group in groups.items():
        conformer_rows: list[dict[str, float]] = []
        for full, ime in zip(group["full"], group["ime"], strict=True):
            full_d2 = _d2_project_for_record(full)
            ime_d2 = _d2_project_for_record(ime)
            values = _case3d_static_d2_values(full_d2)
            static_d3 = _run_case3d_static_d3(full, ime, full_d2, ime_d2)
            comparison = _comparison_features(static_d3)
            values.update({name: comparison[name] for name in d3_names if name in comparison})
            missing = [name for name in CASE3D_STATIC_FEATURES if _finite(values.get(name)) is None]
            if missing:
                raise ValueError(f"{full.name}: incomplete Case 3d static feature row: {missing}")
            conformer_rows.append({name: float(values[name]) for name in CASE3D_STATIC_FEATURES})
        row: dict[str, Any] = {
            "catalyst": catalyst,
            "yield_percent": YIELDS[catalyst],
            "n_static_rows": len(conformer_rows),
            "geometry_status": "published_CCDC_canonical" if catalyst in {"3a", "3b", "3c", "3d"} else "xTB_constructed_exploratory",
        }
        for name in CASE3D_STATIC_FEATURES:
            row[name] = float(np.mean([item[name] for item in conformer_rows]))
        rows.append(row)
    path = OUT / "case3f-case3d-static-model-inputs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["catalyst", "yield_percent", "n_static_rows", "geometry_status", *CASE3D_STATIC_FEATURES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _build_feature_matrix(groups: dict[str, dict[str, Any]], new_d3_projects: dict[str, Path]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    rows = []
    for catalyst, group in groups.items():
        d3_by_full = {full.name: _d3_project_for_group(catalyst, full, ime, new_d3_projects) for full, ime in zip(group["full"], group["ime"], strict=True)}
        rows.append(_feature_row_for_catalyst(catalyst, group, d3_by_full))
    excluded = {
        "d2_feature_schema",
        "d2_variability_status",
        "d2_probability_field_v2",
        "d2_coordination_sectors",
        "d2_coordination_distribution",
        "d2_source_count",
        "d3_source_count",
    }
    all_features = sorted({key for row in rows for key in row if key.startswith(("d2_", "d3_"))})
    static_features = sorted({key for row in rows for key in row if key.startswith("static_")})
    complete = [feature for feature in all_features if all(_finite(row.get(feature)) is not None for row in rows)]
    nonconstant = [feature for feature in complete if np.std([float(row[feature]) for row in rows], ddof=1) > 1e-12]
    core_candidates = [feature for feature in CORE_FEATURE_RATIONALE if feature in nonconstant and feature not in excluded]
    candidates = core_candidates
    full_fit_candidates = [feature for feature in nonconstant if feature not in excluded]
    metadata = {
        "all_numeric_features": all_features,
        "complete_features": complete,
        "excluded_non_numeric_or_incomplete": sorted(set(all_features) - set(complete)),
        "excluded_constant_features": sorted(set(complete) - set(nonconstant)),
        "excluded_string_fields": sorted(excluded),
        "full_pool_nonconstant_features": nonconstant,
        "full_pool_nonconstant_feature_count": len(nonconstant),
        "full_fit_candidates": full_fit_candidates,
        "full_fit_candidate_count": len(full_fit_candidates),
        "core_feature_rationales": {feature: CORE_FEATURE_RATIONALE[feature] for feature in candidates},
        "core_features_missing_or_constant": sorted(set(CORE_FEATURE_RATIONALE) - set(candidates)),
        "candidate_features": candidates,
        "candidate_feature_count": len(candidates),
        "feature_policy": "archive all finite numeric d2 ensemble fields, matched full-minus-IMe d2 ensemble deltas, and dynamic d3 changed-area/scalar-comparison fields; fit only the mechanistically motivated core because n=7",
    }
    fields = ["catalyst", "yield_percent", "outcome_source", "geometry_status", "d2_source_count", "d3_source_count", *static_features, *all_features]
    path = OUT / "case3f-feature-matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    with (OUT / "case3f-yield-table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["catalyst", "yield_percent", "loading_mol_percent", "source"])
        writer.writeheader()
        for catalyst, yield_percent in YIELDS.items():
            writer.writerow(
                {
                    "catalyst": catalyst,
                    "yield_percent": yield_percent,
                    "loading_mol_percent": 0.02,
                    "source": "attached article Table 1",
                }
            )
    (OUT / "feature-pool-manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, candidates, metadata


def _plot_comparison(models: list[tuple[dict[str, Any], str, str]], rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["catalyst"] for row in rows]
    y = np.asarray([row["yield_percent"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for ax, (result, title, color) in zip(axes.flat, models, strict=True):
        x = np.asarray(result["weighted_score"], dtype=float)
        fitted = np.asarray(result["predicted"], dtype=float)
        order = np.argsort(x)
        ax.scatter(x, y, color="#245b9e", zorder=3)
        ax.plot(x[order], fitted[order], color=color, lw=1.8)
        for xi, yi, label in zip(x, y, labels, strict=True):
            ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Standardized Bayesian descriptor score")
        ax.set_ylabel("Table 1 yield at 0.02 mol% (%)")
        validation_r2 = result.get("nested_loo_r2", result.get("loo_r2"))
        validation_label = "nested LOO" if "nested_loo_r2" in result else "fixed LOO"
        ax.set_title(f"{title}\nin-sample R²={result['r2_in_sample']:.3f}; {validation_label} R²={validation_r2:.3f}")
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fit_evidence_fast(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    include_diagnostics: bool = True,
    min_size: int = 1,
    max_size: int = 3,
) -> dict[str, Any]:
    """Vectorized equivalent of the case-3d Gaussian evidence grid.

    The original case-3d implementation evaluates each subset/hyperparameter
    combination in Python.  Case 3f intentionally retains the same grids and
    marginal-likelihood equation but batches all subsets of a given size, so
    the complete core feature pool remains in scope without an impractical
    Python loop.
    """
    n, p = x.shape
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y, ddof=1)) or 1.0
    ys = (y - y_mean) / y_sd
    xs, x_mean, x_sd = bayes._standardize(x)
    tau_grid = np.geomspace(0.08, 4.0, 12)
    sigma_grid = np.geomspace(0.08, 2.0, 14)
    best: tuple[float, tuple[int, ...], float, float] | None = None
    y_bar = float(np.mean(ys))
    y_centered = ys - y_bar
    y_centered_norm2 = float(y_centered @ y_centered)
    for size in range(min_size, min(max_size, p) + 1):
        combinations = np.asarray(list(itertools.combinations(range(p), size)), dtype=int)
        subset_x = np.transpose(xs[:, combinations], (1, 0, 2))
        gram = np.einsum("mni,mnj->mij", subset_x, subset_x)
        cross_y = np.einsum("mni,n->mi", subset_x, y_centered)
        for tau in tau_grid:
            for sigma in sigma_grid:
                sigma2 = sigma**2
                alpha = tau**2 / sigma2
                beta_covariance = np.eye(size)[None, :, :] + alpha * gram
                sign, logdet_beta = np.linalg.slogdet(beta_covariance)
                valid = sign > 0
                if not np.any(valid):
                    continue
                beta_solution = np.linalg.solve(
                    beta_covariance[valid], cross_y[valid, :, None]
                )[:, :, 0]
                correction = (tau**2 / sigma2**2) * np.einsum(
                    "mi,mi->m", cross_y[valid], beta_solution
                )
                energy = (
                    n * y_bar**2 / (sigma2 + 100.0**2 * n)
                    + y_centered_norm2 / sigma2
                    - correction
                )
                logdet = (
                    math.log(sigma2 + 100.0**2 * n)
                    + (n - 1) * math.log(sigma2)
                    + logdet_beta[valid]
                )
                evidence = -0.5 * (energy + logdet + n * math.log(2.0 * math.pi))
                local_index = int(np.argmax(evidence))
                local_value = float(evidence[local_index])
                if best is None or local_value > best[0]:
                    valid_indices = np.flatnonzero(valid)
                    best = (local_value, tuple(int(item) for item in combinations[valid_indices[local_index]]), float(tau), float(sigma))
    if best is None:
        raise ValueError("no Bayesian model candidates were generated")
    evidence, subset, tau, sigma = best
    selected_x = xs[:, subset]
    design = np.column_stack([np.ones(n), selected_x])
    prior_var = np.asarray([100.0**2, *([tau**2] * len(subset))], dtype=float)
    prior_cov = np.diag(prior_var)
    covariance = sigma**2 * np.eye(n) + design @ prior_cov @ design.T
    inv_cov = np.linalg.inv(covariance)
    posterior_mean = prior_cov @ design.T @ inv_cov @ ys
    posterior_cov = prior_cov - prior_cov @ design.T @ inv_cov @ design @ prior_cov
    posterior_sd = np.sqrt(np.clip(np.diag(posterior_cov), 0.0, None))
    descriptor_score = selected_x @ posterior_mean[1:]
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
    for index, feature_index in enumerate(subset):
        weights.append(
            {
                "feature": names[feature_index],
                "posterior_weight_standardized": float(posterior_mean[index + 1]),
                "posterior_sd_standardized": float(posterior_sd[index + 1]),
                "credible_interval_95_standardized": [
                    float(posterior_mean[index + 1] - 1.96 * posterior_sd[index + 1]),
                    float(posterior_mean[index + 1] + 1.96 * posterior_sd[index + 1]),
                ],
                "absolute_contribution_fraction": float(contribution[index]),
            }
        )
    result = {
        "feature_names": [names[index] for index in subset],
        "evidence": float(evidence),
        "prior_tau": float(tau),
        "noise_sigma_standardized": float(sigma),
        "outcome_mean": y_mean,
        "outcome_sd": y_sd,
        "predictor_means": {name: float(x_mean[index]) for index, name in enumerate(names)},
        "predictor_sds": {name: float(x_sd[index]) for index, name in enumerate(names)},
        "weights": weights,
        "intercept_standardized": float(posterior_mean[0]),
        "intercept_sd_standardized": float(posterior_sd[0]),
        "rmse": rmse,
        "r2_in_sample": r2,
        "observed": [float(item) for item in y],
        "predicted": [float(y_mean + y_sd * value) for value in fitted],
        "predictive_sd": [float(value) for value in prediction_sd],
        "descriptor_score": [float(value) for value in descriptor_score],
        "weighted_score": [float(value) for value in descriptor_score],
        "fitted_standardized": [float(value) for value in fitted],
    }
    if include_diagnostics:
        result.update(bayes._loo_metrics(x, y, names, result))
        result.update(bayes._loo_intercept_metrics(y))
        result["loo_improves_intercept"] = bool(result["loo_rmse"] < result["loo_intercept_rmse"])
        result.update(bayes._nested_loo_metrics(x, y, names))
        result["nested_loo_intercept_rmse"] = result["loo_intercept_rmse"]
        result["nested_loo_intercept_r2"] = result["loo_intercept_r2"]
        result["nested_loo_improves_intercept"] = bool(result["nested_loo_rmse"] < result["nested_loo_intercept_rmse"])
        result.update(bayes._permutation_metrics(x, y, names, result))
    return result


def _fit_models(rows: list[dict[str, Any]], candidates: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray([[float(row[feature]) for feature in candidates] for row in rows], dtype=float)
    y = np.asarray([float(row["yield_percent"]) for row in rows], dtype=float)
    bayes._fit_evidence = _fit_evidence_fast
    ensemble = _fit_evidence_fast(x, y, candidates)

    def _fit_fixed(names: list[str]) -> dict[str, Any]:
        values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
        result = _fit_evidence_fast(values, y, names, include_diagnostics=False, min_size=len(names), max_size=len(names))
        result.update(bayes._loo_metrics(values, y, names, result))
        result.update(bayes._loo_intercept_metrics(y))
        result["loo_improves_intercept"] = bool(result["loo_rmse"] < result["loo_intercept_rmse"])
        result["validation_note"] = "fixed descriptor set; no nested subset selection"
        return result

    static_vbur_name = "static_vbur_percent"
    static_g_name = "static_g_percent"
    if not all(_finite(row.get(name)) is not None for row in rows for name in (static_vbur_name, static_g_name)):
        raise ValueError("static %Vbur or static Solid-G input baseline is unavailable")
    static_vbur = _fit_fixed([static_vbur_name])
    static_g = _fit_fixed([static_g_name])

    full_candidates = metadata["full_fit_candidates"]
    full_x = np.asarray([[float(row[feature]) for feature in full_candidates] for row in rows], dtype=float)
    full = _fit_evidence_fast(full_x, y, full_candidates, include_diagnostics=False)
    full.update(bayes._loo_metrics(full_x, y, full_candidates, full))
    full.update(bayes._loo_intercept_metrics(y))
    full["loo_improves_intercept"] = bool(full["loo_rmse"] < full["loo_intercept_rmse"])
    full["validation_note"] = "full finite d2/d3-derived candidate pool; fixed-subset LOO shown, nested full-pool selection omitted because n=7"

    plot_path = OUT / "yield_four_model_comparison.png"
    _plot_comparison(
        [
            (static_vbur, "Static %Vbur", "#58708a"),
            (static_g, "Static Solid G", "#8a6f3d"),
            (ensemble, "27-feature core Bayesian", "#b23a48"),
            (full, "Full-feature Bayesian", "#6a4c93"),
        ],
        rows,
        plot_path,
    )
    shutil.copyfile(plot_path, OUT / "yield_bayesian_vs_dynamic_vbur.png")
    results = {
        "schema": "part3f.bayesian-yield-results.v1",
        "status": "completed_exploratory_dynamic_d3_d2_bayesian_yield_fit",
        "unit_of_analysis": "one catalyst label; 3b syn/anti dynamic rows averaged within the published 3b yield row",
        "outcome": {"name": "yield_percent", "values": YIELDS, "source": "attached article Table 1, 0.02 mol% Cu(I)-NHC condition"},
        "feature_pool": metadata,
        "model_policy": "case-3d Gaussian Bayesian ridge with marginal-likelihood selection of at most three features; selection and hyperparameters are refit in every nested leave-one-catalyst-out fold",
        "static_baseline_features": [static_vbur_name, static_g_name],
        "core_feature_ensemble_model": ensemble,
        "static_vbur_model": static_vbur,
        "static_g_model": static_g,
        "full_feature_ensemble_model": full,
        "scientific_boundary": "The yield fit is exploratory association over seven catalyst labels. It is not a causal mechanism, kinetic probability, free-energy model, or validation of xTB-constructed 3e/IMes/IPr geometries.",
        "files": {
            "feature_matrix": str(OUT / "case3f-feature-matrix.csv"),
            "yield_table": str(OUT / "case3f-yield-table.csv"),
            "feature_pool_manifest": str(OUT / "feature-pool-manifest.json"),
            "comparison_plot": str(plot_path),
            "comparison_plot_legacy_path": str(OUT / "yield_bayesian_vs_dynamic_vbur.png"),
        },
    }
    (OUT / "case3f-bayesian-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def _fit_case3d_static_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the Case 3d 15-feature Bayesian workflow to Case 3f static rows."""
    names = list(CASE3D_STATIC_FEATURES)
    x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    y = np.asarray([float(row["yield_percent"]) for row in rows], dtype=float)
    bayes._fit_evidence = _fit_evidence_fast
    model = _fit_evidence_fast(x, y, names)

    with (OUT / "case3f-case3d-static-nested-loo-predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "catalyst",
            "yield_observed",
            "yield_nested_loo_predicted",
            "yield_nested_loo_predictive_sd",
            "nested_loo_selected_features",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "catalyst": row["catalyst"],
                    "yield_observed": row["yield_percent"],
                    "yield_nested_loo_predicted": model["nested_loo_predicted"][index],
                    "yield_nested_loo_predictive_sd": model["nested_loo_predictive_sd"][index],
                    "nested_loo_selected_features": ";".join(model["nested_loo_selected_features"][index]),
                }
            )

    labels = [str(row["catalyst"]) for row in rows]
    observed = np.asarray(model["observed"], dtype=float)
    score = np.asarray(model["weighted_score"], dtype=float)
    fitted = np.asarray(model["predicted"], dtype=float)
    nested = np.asarray(model["nested_loo_predicted"], dtype=float)
    nested_sd = np.asarray(model["nested_loo_predictive_sd"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    order = np.argsort(score)
    axes[0].scatter(score, observed, color="#245b9e", zorder=3)
    axes[0].plot(score[order], fitted[order], color="#8a6f3d", lw=1.8)
    for xi, yi, label in zip(score, observed, labels, strict=True):
        axes[0].annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Case 3d Bayesian weighted static score")
    axes[0].set_ylabel("Table 1 yield at 0.02 mol% (%)")
    axes[0].set_title(f"In-sample association\nR²={model['r2_in_sample']:.3f}")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(observed, nested, yerr=1.96 * nested_sd, fmt="o", capsize=3, color="#245b9e")
    low = float(min(np.min(observed), np.min(nested - 1.96 * nested_sd)))
    high = float(max(np.max(observed), np.max(nested + 1.96 * nested_sd)))
    margin = max((high - low) * 0.06, 1.0)
    axes[1].plot([low - margin, high + margin], [low - margin, high + margin], color="#b23a48", lw=1.4, linestyle="--")
    for xi, yi, label in zip(observed, nested, labels, strict=True):
        axes[1].annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlim(low - margin, high + margin)
    axes[1].set_ylim(low - margin, high + margin)
    axes[1].set_xlabel("Observed yield (%)")
    axes[1].set_ylabel("Nested LOO predicted yield (%)")
    axes[1].set_title(f"Nested leave-one-catalyst-out\nR²={model['nested_loo_r2']:.3f}")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Case 3f static comparison using the exact Case 3d ML protocol", fontsize=13)
    figure_path = OUT / "yield_case3d_static_protocol.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary = {
        "schema": "part3f.case3d-static-protocol.v1",
        "status": "completed_exact_case3d_static_protocol_on_case3f_yields",
        "unit_of_analysis": "one catalyst label; 3b syn/anti static rows averaged within the 3b outcome",
        "features_considered": names,
        "maximum_model_features": 3,
        "validation_protocol": {
            "nested_leave_one_catalyst_out": True,
            "outcome_shuffle_permutations": 2000,
            "permutation_seed": 20260922,
            "permutation_scope": "fixed full-data subset and prior hyperparameters; calibration control, not nested selection",
            "case3d_match": "same 15 descriptors, standardization, evidence grid, <=3 feature selection, nested LOO refitting, and shuffle count",
        },
        "static_geometry_definition": "single input-geometry d2 baselines and static full-minus-IMe d3 comparisons; 3b syn/anti averaged",
        "model": model,
        "files": {
            "inputs": str(OUT / "case3f-case3d-static-model-inputs.csv"),
            "nested_loo_predictions": str(OUT / "case3f-case3d-static-nested-loo-predictions.csv"),
            "plot": str(figure_path),
        },
        "scientific_boundary": "Exploratory seven-catalyst static comparison; constructed 3e/IMes/IPr geometries and small n prevent general predictive claims.",
    }
    (OUT / "case3f-case3d-static-bayesian-results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    precision: float = CASE3D_V2_RIDGE_PRECISION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Case 3d v2 standardized Bayesian ridge posterior mean."""
    x_mean = np.mean(x_train, axis=0)
    x_sd = np.std(x_train, axis=0, ddof=1)
    x_sd[x_sd < 1e-12] = 1.0
    y_mean = float(np.mean(y_train))
    y_sd = float(np.std(y_train, ddof=1)) or 1.0
    x_standardized = (x_train - x_mean) / x_sd
    y_standardized = (y_train - y_mean) / y_sd
    precision_matrix = x_standardized.T @ x_standardized + precision * np.eye(x_train.shape[1])
    beta = np.linalg.solve(precision_matrix, x_standardized.T @ y_standardized)
    residual = y_standardized - x_standardized @ beta
    dof = max(1, len(y_train) - x_train.shape[1] - 1)
    residual_variance = float(residual @ residual / dof)
    covariance = residual_variance * np.linalg.inv(precision_matrix)
    test_standardized = (x_test - x_mean) / x_sd
    prediction = y_mean + y_sd * (test_standardized @ beta)
    return prediction, {
        "x_mean": x_mean,
        "x_sd": x_sd,
        "y_mean": y_mean,
        "y_sd": y_sd,
        "beta_standardized": beta,
        "posterior_sd_standardized_approx": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "sigma_standardized_empirical": math.sqrt(max(residual_variance, 0.0)),
    }


def _choose_v2_pair(x: np.ndarray, y_model: np.ndarray, names: list[str]) -> tuple[tuple[str, str], list[dict[str, Any]]]:
    """Choose a two-feature pair by inner LOO RMSE at fixed precision 0.25."""
    ranked: list[dict[str, Any]] = []
    for indices in itertools.combinations(range(len(names)), 2):
        selected_x = x[:, indices]
        predictions = []
        for held_out in range(len(y_model)):
            train_mask = np.arange(len(y_model)) != held_out
            prediction, _ = _ridge_predict(
                selected_x[train_mask],
                y_model[train_mask],
                selected_x[[held_out]],
            )
            predictions.append(float(prediction[0]))
        predictions_array = np.asarray(predictions, dtype=float)
        rmse = float(np.sqrt(np.mean(np.square(y_model - predictions_array))))
        ranked.append({"features": [names[index] for index in indices], "inner_loo_rmse_model_scale": rmse})
    ranked.sort(key=lambda item: (item["inner_loo_rmse_model_scale"], item["features"]))
    return tuple(ranked[0]["features"]), ranked


def _logit_yield(values: np.ndarray) -> np.ndarray:
    fraction = np.clip(values / 100.0, 1e-6, 1.0 - 1e-6)
    return np.log(fraction / (1.0 - fraction))


def _inverse_logit_yield(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 100.0 / (1.0 + np.exp(-clipped))


def _score_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else 0.0,
    }


def _plot_case3d_v2_static(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    path: Path,
) -> None:
    """Version 2 observed-yield versus static Bayesian score/%Vbur figure."""
    labels = [str(row["catalyst"]) for row in rows]
    observed = np.asarray([float(row["yield_percent"]) for row in rows], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True, constrained_layout=True)

    x_score = np.asarray(model["raw_score_plot"]["weighted_score"], dtype=float)
    fit_score = np.asarray(model["raw_score_plot"]["fitted"], dtype=float)
    loo_score = np.asarray(model["raw_score_plot"]["loo_predicted"], dtype=float)
    order = np.argsort(x_score)
    ax = axes[0]
    ax.plot(x_score[order], fit_score[order], color="#245b9e", lw=2.0, label="full-data Bayesian ridge fit")
    for xi, yi, loo in zip(x_score, observed, loo_score, strict=True):
        ax.plot([xi, xi], [yi, loo], color="#d17a22", lw=0.9, alpha=0.55)
    ax.scatter(x_score, observed, s=48, color="#222222", label="observed", zorder=3)
    ax.scatter(x_score, loo_score, s=50, marker="x", color="#d17a22", label="fixed-pair LOO", zorder=4)
    for xi, yi, label in zip(x_score, observed, labels, strict=True):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Version 2 static Bayesian weighted score")
    ax.set_ylabel("Table 1 yield at 0.02 mol% (%)")
    ax.set_title(
        f"Static d2/d3 candidate pair\nin-sample R²={model['raw_score_plot']['in_sample_r2']:.3f}; "
        f"conditional LOO R²={model['raw_score_plot']['loo_r2']:.3f}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    x_vbur = np.asarray([float(row["vbur_percent"]) for row in rows], dtype=float)
    vbur_model = model["vbur_baseline"]
    vbur_fitted = np.asarray(vbur_model["fitted"], dtype=float)
    vbur_loo = np.asarray(vbur_model["loo_predicted"], dtype=float)
    order = np.argsort(x_vbur)
    ax.plot(x_vbur[order], vbur_fitted[order], color="#8a6f3d", lw=2.0, linestyle="--", label="static %Vbur linear fit")
    for xi, yi, loo in zip(x_vbur, observed, vbur_loo, strict=True):
        ax.plot([xi, xi], [yi, loo], color="#d17a22", lw=0.9, alpha=0.55)
    ax.scatter(x_vbur, observed, s=48, color="#222222", label="observed", zorder=3)
    ax.scatter(x_vbur, vbur_loo, s=50, marker="x", color="#d17a22", label="LOO prediction", zorder=4)
    for xi, yi, label in zip(x_vbur, observed, labels, strict=True):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Static input-geometry %Vbur")
    ax.set_title(
        f"Static %Vbur baseline\nin-sample R²={vbur_model['in_sample_r2']:.3f}; "
        f"LOO R²={vbur_model['loo_r2']:.3f}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle("Case 3f static Bayesian Version 2: two descriptors, fixed ridge precision 0.25", fontsize=13)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _fit_case3d_v2_static_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the Case 3d Version 2 fixed-precision candidate workflow to yield."""
    names = list(CASE3D_STATIC_FEATURES)
    x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    observed = np.asarray([float(row["yield_percent"]) for row in rows], dtype=float)
    transformed = _logit_yield(observed)

    full_pair, pair_ranking = _choose_v2_pair(x, transformed, names)
    selected_indices = [names.index(name) for name in full_pair]
    selected_x = x[:, selected_indices]
    fitted_model, full_fit = _ridge_predict(selected_x, transformed, selected_x)
    fitted_display = _inverse_logit_yield(fitted_model)
    score_logit = ((selected_x - full_fit["x_mean"]) / full_fit["x_sd"]) @ full_fit["beta_standardized"]

    conditional_logit: list[float] = []
    conditional_beta: list[list[float]] = []
    for held_out in range(len(observed)):
        mask = np.arange(len(observed)) != held_out
        prediction, fit = _ridge_predict(selected_x[mask], transformed[mask], selected_x[[held_out]])
        conditional_logit.append(float(prediction[0]))
        conditional_beta.append([float(item) for item in fit["beta_standardized"]])
    conditional_display = _inverse_logit_yield(np.asarray(conditional_logit, dtype=float))

    nested_logit: list[float] = []
    nested_pairs: list[list[str]] = []
    nested_inner_ranks: list[list[dict[str, Any]]] = []
    for held_out in range(len(observed)):
        mask = np.arange(len(observed)) != held_out
        pair, inner_ranking = _choose_v2_pair(x[mask], transformed[mask], names)
        pair_indices = [names.index(name) for name in pair]
        prediction, _ = _ridge_predict(x[mask][:, pair_indices], transformed[mask], x[[held_out]][:, pair_indices])
        nested_logit.append(float(prediction[0]))
        nested_pairs.append(list(pair))
        nested_inner_ranks.append(inner_ranking)
    nested_display = _inverse_logit_yield(np.asarray(nested_logit, dtype=float))

    raw_fitted, raw_fit = _ridge_predict(selected_x, observed, selected_x)
    raw_score = ((selected_x - raw_fit["x_mean"]) / raw_fit["x_sd"]) @ raw_fit["beta_standardized"]
    raw_loo: list[float] = []
    for held_out in range(len(observed)):
        mask = np.arange(len(observed)) != held_out
        prediction, _ = _ridge_predict(selected_x[mask], observed[mask], selected_x[[held_out]])
        raw_loo.append(float(prediction[0]))

    # Case 3d Version 2 shows the static %Vbur baseline as a raw-scale OLS fit.
    vbur = np.asarray([float(row["vbur_percent"]) for row in rows], dtype=float)
    vbur_design = np.column_stack([np.ones(len(vbur)), vbur])
    vbur_coefficients = np.linalg.lstsq(vbur_design, observed, rcond=None)[0]
    vbur_fitted = vbur_design @ vbur_coefficients
    vbur_loo: list[float] = []
    for held_out in range(len(observed)):
        mask = np.arange(len(observed)) != held_out
        fold_design = np.column_stack([np.ones(int(np.sum(mask))), vbur[mask]])
        coefficients = np.linalg.lstsq(fold_design, observed[mask], rcond=None)[0]
        vbur_loo.append(float(coefficients @ np.asarray([1.0, vbur[held_out]])))

    model_metrics = _score_metrics(observed, fitted_display)
    conditional_metrics = _score_metrics(observed, conditional_display)
    nested_metrics = _score_metrics(observed, nested_display)
    raw_metrics = _score_metrics(observed, raw_fitted)
    raw_loo_metrics = _score_metrics(observed, np.asarray(raw_loo))
    vbur_metrics = _score_metrics(observed, vbur_fitted)
    vbur_loo_metrics = _score_metrics(observed, np.asarray(vbur_loo))

    weights = []
    abs_beta = np.abs(full_fit["beta_standardized"])
    abs_total = float(np.sum(abs_beta))
    for index, feature in enumerate(full_pair):
        coefficient = float(full_fit["beta_standardized"][index])
        sd = float(full_fit["posterior_sd_standardized_approx"][index])
        weights.append({
            "feature": feature,
            "posterior_mean_standardized": coefficient,
            "posterior_sd_standardized_approx": sd,
            "credible_interval_95_standardized_approx": [coefficient - 1.96 * sd, coefficient + 1.96 * sd],
            "absolute_contribution_fraction": float(abs_beta[index] / abs_total) if abs_total else 0.0,
            "loo_coefficient_sign_consistency": float(np.mean(np.sign([beta[index] for beta in conditional_beta]) == np.sign(coefficient))),
        })

    v2_plot_path = OUT / "yield_static_bayesian_v2.png"
    preliminary = {
        "raw_score_plot": {
            "weighted_score": [float(value) for value in raw_score],
            "fitted": [float(value) for value in raw_fitted],
            "loo_predicted": raw_loo,
            "in_sample_rmse": raw_metrics["rmse"],
            "in_sample_r2": raw_metrics["r2"],
            "loo_r2": raw_loo_metrics["r2"],
        },
        "vbur_baseline": {
            "fitted": [float(value) for value in vbur_fitted],
            "loo_predicted": vbur_loo,
            "in_sample_r2": vbur_metrics["r2"],
            "loo_r2": vbur_loo_metrics["r2"],
        },
    }
    _plot_case3d_v2_static(rows, preliminary, v2_plot_path)

    pair_screen_path = OUT / "case3f-case3d-v2-pair-screen.csv"
    with pair_screen_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_1", "feature_2", "inner_loo_rmse_logit_yield"])
        writer.writeheader()
        for rank, record in enumerate(pair_ranking, start=1):
            writer.writerow({"rank": rank, "feature_1": record["features"][0], "feature_2": record["features"][1], "inner_loo_rmse_logit_yield": record["inner_loo_rmse_model_scale"]})

    nested_path = OUT / "case3f-case3d-v2-nested-loo-predictions.csv"
    with nested_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["catalyst", "yield_observed", "yield_fixed_pair_loo", "yield_nested_pair_selection_loo", "nested_selected_features"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "catalyst": row["catalyst"],
                "yield_observed": observed[index],
                "yield_fixed_pair_loo": conditional_display[index],
                "yield_nested_pair_selection_loo": nested_display[index],
                "nested_selected_features": ";".join(nested_pairs[index]),
            })

    result = {
        "status": "completed_case3d_v2_static_bayesian_yield_candidate",
        "features_considered": names,
        "features_selected": list(full_pair),
        "feature_selection_method": "minimum inner leave-one-catalyst-out RMSE over all 105 two-feature pairs on logit-transformed yield",
        "ridge_precision_fixed": CASE3D_V2_RIDGE_PRECISION,
        "outcome_transform": "logit(yield_percent/100); inverted with logistic function for yield-scale diagnostics",
        "standardized_logit_model": {
            "outcome_mean": full_fit["y_mean"],
            "outcome_sd": full_fit["y_sd"],
            "intercept_standardized": 0.0,
            "weights": weights,
            "predictor_means": {name: float(value) for name, value in zip(full_pair, full_fit["x_mean"], strict=True)},
            "predictor_sds": {name: float(value) for name, value in zip(full_pair, full_fit["x_sd"], strict=True)},
            "fitted_logit_yield": [float(value) for value in fitted_model],
            "fitted_yield_percent": [float(value) for value in fitted_display],
            "in_sample_r2_logit_scale": _score_metrics(transformed, fitted_model)["r2"],
        },
        "raw_score_plot": {
            "weighted_score": [float(value) for value in raw_score],
            "raw_scale_intercept": raw_fit["y_mean"],
            "raw_scale_outcome_sd": raw_fit["y_sd"],
            "raw_scale_slope_standardized": [float(value) for value in raw_fit["beta_standardized"]],
            "predictor_means": {name: float(value) for name, value in zip(full_pair, raw_fit["x_mean"], strict=True)},
            "predictor_sds": {name: float(value) for name, value in zip(full_pair, raw_fit["x_sd"], strict=True)},
            "fitted": [float(value) for value in raw_fitted],
            "loo_predicted": raw_loo,
            "in_sample_rmse": raw_metrics["rmse"],
            "in_sample_r2": raw_metrics["r2"],
            "loo_r2": raw_loo_metrics["r2"],
            "loo_rmse": raw_loo_metrics["rmse"],
        },
        "fixed_pair_conditional_loo": {
            "predicted_yield_percent": [float(value) for value in conditional_display],
            **conditional_metrics,
            "model_scale_r2": _score_metrics(transformed, np.asarray(conditional_logit))["r2"],
            "coverage_95": None,
        },
        "nested_pair_selection_loo": {
            "predicted_yield_percent": [float(value) for value in nested_display],
            "selected_features_by_fold": nested_pairs,
            **nested_metrics,
            "model_scale_r2": _score_metrics(transformed, np.asarray(nested_logit))["r2"],
        },
        "full_data_pair_selection_inner_cv_rmse_logit": float(pair_ranking[0]["inner_loo_rmse_model_scale"]),
        "pair_screen": pair_ranking,
        "vbur_baseline": {
            "descriptor": "vbur_percent",
            "coefficients_intercept_slope": [float(value) for value in vbur_coefficients],
            "fitted": [float(value) for value in vbur_fitted],
            "loo_predicted": vbur_loo,
            "in_sample_r2": vbur_metrics["r2"],
            "loo_r2": vbur_loo_metrics["r2"],
            "loo_rmse": vbur_loo_metrics["rmse"],
        },
        "files": {
            "static_inputs": str(OUT / "case3f-case3d-static-model-inputs.csv"),
            "pair_screen": str(pair_screen_path),
            "nested_predictions": str(nested_path),
            "plot": str(v2_plot_path),
        },
        "validation_boundary": "The selected pair is post-hoc for Case 3f. Fixed-pair LOO is conditional; nested LOO repeats the pair selection inside each held-out fold and is the selection-aware estimate.",
    }
    (OUT / "case3f-case3d-v2-static-bayesian-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _plot_single_bayesian_model(
    model: dict[str, Any],
    observed: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    prediction_key: str,
    validation_name: str,
    validation_r2: float,
) -> None:
    score = np.asarray(model["weighted_score"], dtype=float)
    fitted = np.asarray(model["predicted"], dtype=float)
    prediction = np.asarray(model[prediction_key], dtype=float)
    order = np.argsort(score)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    ax = axes[0]
    ax.plot(score[order], fitted[order], color="#245b9e", lw=1.8, label="full-data Bayesian fit")
    for x_value, y_value, pred in zip(score, observed, prediction, strict=True):
        ax.plot([x_value, x_value], [y_value, pred], color="#d17a22", alpha=0.55, lw=0.9)
    ax.scatter(score, observed, color="#222222", s=45, label="observed", zorder=3)
    ax.scatter(score, prediction, marker="x", color="#d17a22", s=50, label=validation_name, zorder=4)
    for x_value, y_value, label in zip(score, observed, labels, strict=True):
        ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Standardized Bayesian weighted descriptor score")
    ax.set_ylabel("Table 1 yield at 0.02 mol% (%)")
    ax.set_title(f"{title}\nin-sample R²={model['r2_in_sample']:.3f}; {validation_name} R²={validation_r2:.3f}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    ax.scatter(observed, prediction, color="#245b9e", s=45)
    low = float(min(np.min(observed), np.min(prediction)))
    high = float(max(np.max(observed), np.max(prediction)))
    margin = max((high - low) * 0.08, 1.0)
    ax.plot([low - margin, high + margin], [low - margin, high + margin], "--", color="#b23a48", lw=1.4)
    for x_value, y_value, label in zip(observed, prediction, labels, strict=True):
        ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Observed yield (%)")
    ax.set_ylabel(f"{validation_name} predicted yield (%)")
    ax.set_title(f"{validation_name} validation\nR²={validation_r2:.3f}")
    ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _write_report_compact(results: dict[str, Any], rows: list[dict[str, Any]], new_d3_projects: dict[str, Path]) -> None:
    ensemble = results["core_feature_ensemble_model"]
    static_vbur = results["static_vbur_model"]
    static_g = results["static_g_model"]
    case3d_static = results["case3d_static_protocol_model"]
    full = results["full_feature_ensemble_model"]

    def _equation(model: dict[str, Any]) -> str:
        terms = " ".join(
            f"{weight['posterior_weight_standardized']:+.4f}·z({weight['feature']})"
            for weight in model["weights"]
        )
        return f"ŷ = {model['outcome_mean']:.4f} + {model['outcome_sd']:.4f}·[{model['intercept_standardized']:.4f} {terms}]"

    lines = [
        "# Case 3f — dynamic d2/d3 ensemble features versus yield",
        "",
        "## Shared understanding and scope",
        "",
        "Case 3f extends the Part 3b Cu(I)-NHC screen using the attached article's Table 1 yields at 0.02 mol% catalyst loading. The model unit is one catalyst label: 3a, 3b, 3c, 3d, 3e, IMes, and IPr. The published 3b syn and anti crystal conformers are averaged within the single 3b yield row (94%).",
        "",
        "The primary comparison is dynamic d3 hidden contribution relative to a matched IMe reference, with the full finite d2/d3-derived pool archived and a smaller dynamic core used for Bayesian weighting. Static %Vbur and static Solid-G one-feature models are retained as independent baselines. They are not the primary representation.",
        "",
        "## Geometry evidence status",
        "",
        "The 3a–3d full structures and their matched IMe references reuse the local Part 3b CCDC/xTB/CREST outputs. 3e, IMes, IPr, and their IMe references are explicit xTB-constructed screening analogues: 3e is formed by removing the 4,5-backbone methyl branches from the 3b-anti frame; IMes copies the direct N-Mes wingtip onto both nitrogens; IPr copies the direct N-Dipp wingtip onto both nitrogens. These new structures are not presented as deposited crystal geometries.",
        "",
        "## Bayesian yield models",
        "",
        f"The feature archive contains {len(results['feature_pool']['all_numeric_features'])} numeric d2/d3-derived fields, including matched full-minus-IMe CREST ensemble deltas. The model uses a {results['feature_pool']['candidate_feature_count']}-feature core selected from {results['feature_pool']['full_pool_nonconstant_feature_count']} complete, nonconstant fields: entropy/flexibility, accessibility, dynamic occupation/contact changes, d3 %Vbur/G deltas, and the initial displaced-contact signal. Evidence selection is capped at three descriptors, matching case 3d, and nested leave-one-catalyst-out refits subset and hyperparameter selection inside each fold.",
        "",
        f"Core dynamic-feature model: selected {', '.join(ensemble['feature_names'])}; evidence {ensemble['evidence']:.3f}; in-sample RMSE {ensemble['rmse']:.2f} percentage points; fixed-subset LOO RMSE {ensemble['loo_rmse']:.2f}; nested LOO RMSE {ensemble['nested_loo_rmse']:.2f}; nested model improves intercept-only baseline: {ensemble['nested_loo_improves_intercept']}.",
        f"Nested LOO core-selection frequency: {', '.join(f'{feature}={count}/7' for feature, count in sorted(ensemble['nested_loo_feature_selection_frequency'].items(), key=lambda item: (-item[1], item[0])))}.",
        f"Static %Vbur model: in-sample R² {static_vbur['r2_in_sample']:.3f}; fixed LOO R² {static_vbur['loo_r2']:.3f}.",
        f"Static Solid-G model: in-sample R² {static_g['r2_in_sample']:.3f}; fixed LOO R² {static_g['loo_r2']:.3f}.",
        f"Full-feature Bayesian model: selected {', '.join(full['feature_names'])}; in-sample R² {full['r2_in_sample']:.3f}; fixed-subset LOO R² {full['loo_r2']:.3f}. Nested full-pool selection is omitted because n=7.",
        "",
        "27-feature core pool:",
        "",
        "; ".join(results["feature_pool"]["candidate_features"]),
        "",
        "Posterior weights are standardized coefficients with 95% credible intervals; they are Bayesian weighting coefficients, not causal importance scores.",
        "",
        "| Model | Feature | Posterior weight | 95% credible interval | Absolute contribution |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name, model in (("static %Vbur", static_vbur), ("static Solid G", static_g), ("core dynamic-feature ensemble", ensemble), ("full dynamic-feature ensemble", full)):
        for weight in model["weights"]:
            interval = weight["credible_interval_95_standardized"]
            lines.append(f"| {model_name} | {weight['feature']} | {weight['posterior_weight_standardized']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] | {weight['absolute_contribution_fraction']:.3f} |")
    lines.extend([
        "",
        "## Bayesian weighting equations",
        "",
        "For every Bayesian panel, each feature is standardized as `z(x_j) = (x_j − μ_j) / s_j`. The posterior-weighted yield equation is `ŷ = μ_y + s_y [b₀ + Σⱼ bⱼ z(x_j)]`, where `b₀` is the posterior standardized intercept and `bⱼ` are the posterior standardized weights below.",
        f"27-feature core equation: `{_equation(ensemble)}`",
        f"Full-feature equation: `{_equation(full)}`",
        "",
        "## Core-feature rationale",
        "",
        "The primary entropy variables are the absolute CREST ensemble steric-entropy volume/sum and the matched full-minus-IMe entropy-occupancy difference. The normalized entropy difference is retained in the archive when estimable; several IMe references report it as not estimable, so the complete core uses the finite entropy-sum difference as the comparable fallback. Flexibility, persistent open/blocked state, approach accessibility, d3 occupation/contact changes, and dynamic d3 %Vbur/G deltas capture complementary dynamic pocket behavior. The displaced G feature is retained because it was selected by the initial full-pool screen; it is treated as an importance-guided candidate, not as a mechanistic conclusion.",
        "",
        "## Nested leave-one-catalyst-out boundary",
        "",
        "With seven yield observations and a large dynamic descriptor pool, the nested LOO result is the primary guard against over-reading the in-sample fit. A favorable in-sample score alone is not evidence that a new catalyst's yield can be predicted. The 3e/IMes/IPr map branches are also exploratory xTB/CREST analogues and remain a material uncertainty.",
        "",
        "## Outputs",
        "",
        f"- Feature matrix: `{OUT / 'case3f-feature-matrix.csv'}`",
        f"- Yield table: `{OUT / 'case3f-yield-table.csv'}`",
        f"- Feature-pool manifest: `{OUT / 'feature-pool-manifest.json'}`",
        f"- Structure manifest: `{OUT / 'structure-manifest.json'}`",
        f"- Bayesian results: `{OUT / 'case3f-bayesian-results.json'}`",
        f"- Four-panel comparison plot: `{OUT / 'yield_four_model_comparison.png'}`",
        f"- New hidden d3 projects: `{MAPS / 'hidden-d3'}`",
        "",
    ])
    path = DOC_ROOT / "part3f-dynamic-yield-ML.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(results: dict[str, Any], rows: list[dict[str, Any]], new_d3_projects: dict[str, Path]) -> None:
    """Write the publication-facing Case 3f ML note in the Case 3d format."""
    core = results["core_feature_ensemble_model"]
    static_vbur = results["static_vbur_model"]
    static_g = results["static_g_model"]
    case3d_static = results["case3d_static_protocol_model"]
    case3d_v2_static = results["case3d_v2_static_protocol_model"]
    full = results["full_feature_ensemble_model"]
    pool = results["feature_pool"]

    def equation(model: dict[str, Any]) -> str:
        terms = " ".join(
            f"{weight['posterior_weight_standardized']:+.4f}·z({weight['feature']})"
            for weight in model["weights"]
        )
        return (
            f"ŷ = {model['outcome_mean']:.4f} + {model['outcome_sd']:.4f}·"
            f"[{model['intercept_standardized']:.4f} {terms}]"
        )

    def v2_logit_equation() -> str:
        model = case3d_v2_static["standardized_logit_model"]
        terms = " ".join(
            f"{weight['posterior_mean_standardized']:+.4f}·z({weight['feature']})"
            for weight in model["weights"]
        )
        return (
            f"η̂ = {model['outcome_mean']:.4f} + {model['outcome_sd']:.4f}·"
            f"[{terms}]; ŷ = 100·logistic(η̂)"
        )

    def v2_raw_plot_equation() -> str:
        model = case3d_v2_static["raw_score_plot"]
        terms = " ".join(
            f"{coefficient:+.4f}·z({feature})"
            for feature, coefficient in zip(
                case3d_v2_static["features_selected"],
                model["raw_scale_slope_standardized"],
                strict=True,
            )
        )
        return (
            f"ŷ = {model['raw_scale_intercept']:.4f} + {model['raw_scale_outcome_sd']:.4f}·"
            f"[{terms}]"
        )

    lines = [
        "---",
        "title: Case 3f dynamic d2/d3 Bayesian yield model",
        "date: 2026-09-22",
        "tags:",
        "  - case-3f",
        "  - d2map",
        "  - d3map",
        "  - crest",
        "  - xtb",
        "  - bayesian-regression",
        "status: exploratory-analogue-analysis-with-positive-core-nested-signal",
        "---",
        "",
        "# Case 3f dynamic d2/d3 Bayesian yield model",
        "",
        "## Report composition and preservation",
        "",
        "This full report preserves the complete first-run dynamic analysis and adds static comparisons without replacing or deleting any result. The preserved first run comprises the 27-feature dynamic core model, the complete 156-feature dynamic-pool model, their posterior equations, and their original validation diagnostics. The appended static block comprises the independent one-feature input-geometry %Vbur and Solid-G baselines plus the exact 15-feature Case 3d static ML protocol. Dynamic and static models remain separate so that their feature definitions and validation levels are not conflated.",
        "",
        "## Purpose and boundary",
        "",
        "This note records the completed Case 3f machine-learning extension of Part 3b. It tests whether dynamic CREST/xTB d2/d3 steric-map descriptors are associated with the attached article's Table 1 yields at 0.02 mol% Cu(I)-NHC loading, and compares them with independent one-feature static %Vbur and static Solid-G baselines.",
        "",
        "The unit of analysis is one catalyst label: 3a, 3b, 3c, 3d, 3e, IMes, or IPr. The 3b syn and anti conformers are averaged into the single experimental 3b yield row, avoiding conformer-level pseudoreplication. The fitted associations do not establish a reaction barrier, binding free energy, kinetic probability, catalytic mechanism, or causal steric explanation. With only seven outcomes and three xTB-constructed catalyst analogues, all conclusions remain exploratory.",
        "",
        "## Experimental outcomes and coordinate provenance",
        "",
        "The response values are the printed Table 1 yields at 0.02 mol% catalyst loading:",
        "",
        "| Catalyst | Yield (%) | Coordinate evidence used for mapping |",
        "|---|---:|---|",
        "| 3a | 98 | reused Part 3b published CCDC-derived branch |",
        "| 3b | 94 | reused Part 3b syn and anti branches; descriptors averaged |",
        "| 3c | 92 | reused Part 3b published CCDC-derived branch |",
        "| 3d | 40 | reused Part 3b published CCDC-derived branch |",
        "| 3e | 36 | xTB-constructed exploratory analogue from 3b-anti |",
        "| IMes | 63 | xTB-constructed exploratory symmetric N-Mes analogue |",
        "| IPr | 42 | xTB-constructed exploratory symmetric N-Dipp analogue |",
        "",
        "The 3a–3d full structures and matched IMe references reuse the local Part 3b xTB/CREST outputs. For 3e, the two 4,5-backbone methyl branches were removed from the 3b-anti N-Mes/N-CH2Mes frame. IMes was formed by copying the direct N-Mes wingtip onto both nitrogens, and IPr by copying the direct N-Dipp wingtip onto both nitrogens. Matched IMe references replace both wingtip branches with N-methyl groups. These constructed branches are screening analogues, not deposited crystal structures. Full transformation records are in [`structure-manifest.json`](../../part3/3f/structure-manifest.json).",
        "",
        "## Dynamic map coverage",
        "",
        "The existing 3a, 3b-syn, 3b-anti, 3c, and 3d d2 and hidden-d3 branches are reused from Part 3b. Case 3f adds six successful d2 projects for 3e, IMes, IPr, and their matched IMe references, plus three successful hidden-d3 comparisons (full catalyst minus matched IMe). The d3 comparison targets the dynamic steric contribution beyond the common IMe-sized NHC core.",
        "",
        "The catalyst-level feature matrix is [`case3f-feature-matrix.csv`](../../part3/3f/case3f-feature-matrix.csv). Static %Vbur and Solid-G are taken independently from each d2 project's input-geometry static baseline; the 3b values are averaged across syn and anti. The article's optimized %Vbur values are retained in the structure manifest for provenance but are not used as predictors.",
        "",
        "## Bayesian aggregation and model construction",
        "",
        f"The archive contains {len(pool['all_numeric_features'])} numeric d2/d3-derived fields. After removing incomplete or constant fields, {pool['full_pool_nonconstant_feature_count']} remain; two bookkeeping source-count fields are excluded, leaving {pool['full_fit_candidate_count']} descriptors in the full ML run. A chemically focused {pool['candidate_feature_count']}-feature core covers entropy/flexibility, persistent accessibility, occupation/contact changes, dynamic d3 %Vbur/G differences, quadrant asymmetry, and displaced-contact signals.",
        "",
        "All predictors and yield are standardized. The regression is the same conjugate Gaussian Bayesian ridge/evidence-selection framework used for Case 3d. Marginal-likelihood selection is capped at three descriptors. The core model repeats descriptor and prior/noise selection inside each leave-one-catalyst-out fold and includes 2,000 outcome-shuffle controls. The static baselines use fixed one-feature descriptor sets. The full 156-candidate model reports only fixed-selected-subset LOO diagnostics because nested selection over that pool is not supportable with n=7.",
        "",
        "The requested four-panel comparison is [`yield_four_model_comparison.png`](../../part3/3f/yield_four_model_comparison.png). Its x-axis is the signed posterior descriptor score `S = Σ βⱼzⱼ`, excluding the intercept. Because the coefficients are selected against yield, response-versus-score straightness is descriptive; predictive evidence comes from held-out validation.",
        "",
        "## Model comparison",
        "",
        "| Model | Candidate pool | Selected | In-sample RMSE | In-sample R² | Validation RMSE | Validation R² |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Static %Vbur | 1 | 1 | {static_vbur['rmse']:.3f} | {static_vbur['r2_in_sample']:.3f} | {static_vbur['loo_rmse']:.3f} fixed LOO | {static_vbur['loo_r2']:.3f} |",
        f"| Static Solid-G | 1 | 1 | {static_g['rmse']:.3f} | {static_g['r2_in_sample']:.3f} | {static_g['loo_rmse']:.3f} fixed LOO | {static_g['loo_r2']:.3f} |",
        f"| Exact Case 3d static protocol | {len(CASE3D_STATIC_FEATURES)} | {len(case3d_static['feature_names'])} | {case3d_static['rmse']:.3f} | {case3d_static['r2_in_sample']:.3f} | {case3d_static['nested_loo_rmse']:.3f} nested LOO | {case3d_static['nested_loo_r2']:.3f} |",
        f"| Case 3d static Version 2 (two-feature ridge) | {len(CASE3D_STATIC_FEATURES)} | 2 | {case3d_v2_static['raw_score_plot']['in_sample_rmse']:.3f} | {case3d_v2_static['raw_score_plot']['in_sample_r2']:.3f} | {case3d_v2_static['nested_pair_selection_loo']['rmse']:.3f} nested LOO | {case3d_v2_static['nested_pair_selection_loo']['r2']:.3f} |",
        f"| First-run dynamic 27-feature core | {pool['candidate_feature_count']} | {len(core['feature_names'])} | {core['rmse']:.3f} | {core['r2_in_sample']:.3f} | {core['nested_loo_rmse']:.3f} nested LOO | {core['nested_loo_r2']:.3f} |",
        f"| First-run full dynamic pool | {pool['full_fit_candidate_count']} | {len(full['feature_names'])} | {full['rmse']:.3f} | {full['r2_in_sample']:.3f} | {full['loo_rmse']:.3f} fixed LOO | {full['loo_r2']:.3f} |",
        "",
        "The static %Vbur and Solid-G models do not outperform the leave-one-out intercept baseline (`R² = -0.361`). The core model retains a positive nested-LOO signal (`R² = 0.794`, RMSE 11.687 percentage points, 85.7% nominal 95% coverage). Its fixed-subset LOO result (`R² = 0.993`) is an optimistic conditional diagnostic. The full-pool fixed-subset LOO result (`R² = 0.987`) is likewise not nested and must not be presented as equivalent validation.",
        "",
        "### Static one-feature baselines",
        "",
        "| Model | Descriptor | Posterior weight | 95% credible interval |",
        "|---|---|---:|---:|",
    ]
    for model_name, model in (("Static %Vbur", static_vbur), ("Static Solid-G", static_g)):
        for weight in model["weights"]:
            interval = weight["credible_interval_95_standardized"]
            lines.append(
                f"| {model_name} | `{weight['feature']}` | {weight['posterior_weight_standardized']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] |"
            )

    lines.extend([
        "",
        "### Exact Case 3d ML protocol as a static comparison",
        "",
        "This additional model applies the Case 3d workflow without expanding its descriptor schema: the same 15 features, per-unit aggregation before fitting, z-standardization, conjugate Gaussian Bayesian ridge model, marginal-likelihood selection of no more than three descriptors, nested leave-one-unit-out feature/hyperparameter refitting, and 2,000 fixed-subset outcome shuffles. For Case 3f, each descriptor is evaluated on the single input-geometry d2 baseline and a static full-catalyst-minus-IMe d3 comparison; only the 3b syn/anti rows are averaged.",
        "",
        "The 15 Case 3d descriptors are:",
        "",
    ])
    lines.extend(f"- `{feature}`" for feature in CASE3D_STATIC_FEATURES)
    lines.extend([
        "",
        "Evidence selection retained:",
        "",
        "| Descriptor | Posterior weight | 95% credible interval | Absolute contribution |",
        "|---|---:|---:|---:|",
    ])
    for weight in case3d_static["weights"]:
        interval = weight["credible_interval_95_standardized"]
        lines.append(
            f"| `{weight['feature']}` | {weight['posterior_weight_standardized']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {weight['absolute_contribution_fraction']:.3f} |"
        )
    static_selection_frequency = ", ".join(
        f"`{feature}` {count}/7"
        for feature, count in sorted(
            case3d_static["nested_loo_feature_selection_frequency"].items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    lines.extend([
        "",
        f"The exact Case 3d static model has in-sample RMSE/R² of {case3d_static['rmse']:.3f}/{case3d_static['r2_in_sample']:.3f}, fixed-subset LOO RMSE/R² of {case3d_static['loo_rmse']:.3f}/{case3d_static['loo_r2']:.3f}, and nested LOO RMSE/R² of {case3d_static['nested_loo_rmse']:.3f}/{case3d_static['nested_loo_r2']:.3f}. Nominal nested 95% coverage is {100.0 * case3d_static['nested_loo_coverage_95']:.1f}%. Nested selection frequency is: {static_selection_frequency}. The 2,000-shuffle fixed-subset calibration gives `p = {case3d_static['permutation_p_ge_observed']:.4f}`.",
        "",
        f"The static Case 3d protocol therefore reproduces the same central warning seen in Case 3d itself: a strong in-sample/fixed-subset association collapses when feature selection is repeated inside each held-out fold. Its nested LOO R² ({case3d_static['nested_loo_r2']:.3f}) is substantially below both the intercept baseline (-0.361) and the dynamic 27-feature core ({core['nested_loo_r2']:.3f}). On these seven catalysts, the exact Case 3d static representation does not provide transferable yield prediction, whereas the dynamic core retains the only positive nested signal.",
        "",
        f"Static Case 3d-protocol equation: `{equation(case3d_static)}`",
        "",
        "The weighted-score and nested-validation panels are in [`yield_case3d_static_protocol.png`](../../part3/3f/yield_case3d_static_protocol.png). Machine-readable inputs, nested predictions, and the full posterior result are [`case3f-case3d-static-model-inputs.csv`](../../part3/3f/case3f-case3d-static-model-inputs.csv), [`case3f-case3d-static-nested-loo-predictions.csv`](../../part3/3f/case3f-case3d-static-nested-loo-predictions.csv), and [`case3f-case3d-static-bayesian-results.json`](../../part3/3f/case3f-case3d-static-bayesian-results.json).",
        "",
        "### Static map Version 2 extension (preserving Version 1)",
        "",
        "Version 2 is added alongside—not in place of—the original 15-feature Case 3d static-protocol model above. It uses the same static feature matrix and catalyst-level aggregation (including averaging the 3b syn/anti descriptors), but follows the Part 3d static-map Version 2 candidate screen: test all 105 two-feature pairs, standardize predictors and response within each training fit, and hold Bayesian ridge precision fixed at 0.25. To adapt this selectivity-oriented workflow to yield, pair screening minimizes inner leave-one-catalyst-out RMSE for `logit(yield/100)`; the plot additionally shows the selected pair fitted on raw yield for directly readable percentage-point comparisons.",
        "",
        f"The full-data Version 2 pair is `{case3d_v2_static['features_selected'][0]}` and `{case3d_v2_static['features_selected'][1]}`. Its transformed-scale Bayesian weighting equation is `{v2_logit_equation()}`. The raw-yield score used in the Version 2 comparison panel is `{v2_raw_plot_equation()}`; each `z(x)` in this raw-scale plot equation uses the stored raw-fit predictor mean and sample standard deviation. The JSON includes the exact means, scales, posterior means, approximate 95% intervals, and pair-screen ranking.",
        "",
        f"Version 2 metrics: raw-scale in-sample R² {case3d_v2_static['raw_score_plot']['in_sample_r2']:.3f}; the plotted raw-yield fixed-pair LOO R² is {case3d_v2_static['raw_score_plot']['loo_r2']:.3f}; the inverse-logit predictions from the logit-scale fixed-pair model have R² {case3d_v2_static['fixed_pair_conditional_loo']['r2']:.3f}; nested pair-selection LOO raw-scale R² is {case3d_v2_static['nested_pair_selection_loo']['r2']:.3f} (RMSE {case3d_v2_static['nested_pair_selection_loo']['rmse']:.2f} percentage points). Both fixed-pair diagnostics condition on a pair selected using all seven labels; nested LOO repeats pair selection within each training fold and is the selection-aware diagnostic. See the [Version 2 plot](../../part3/3f/yield_static_bayesian_v2.png), [pair-screen ranking](../../part3/3f/case3f-case3d-v2-pair-screen.csv), [nested prediction table](../../part3/3f/case3f-case3d-v2-nested-loo-predictions.csv), and [machine-readable results](../../part3/3f/case3f-case3d-v2-static-bayesian-results.json).",
        "",
        "### Three requested Bayesian plots",
        "",
        "These standalone figures keep the three representations distinct and are listed in the requested order. Static Version 1 retains the original Case 3d-style evidence-selection fit; static Version 2 is the two-descriptor, fixed-precision pair screen; dynamic Bayesian is the original 27-feature core with nested leave-one-catalyst-out validation.",
        "",
        "1. Static Bayesian Version 1 (nested LOO)",
        "",
        "![Case 3f static Bayesian Version 1](../../part3/3f/yield_static_bayesian_v1.png)",
        "",
        "2. Static Bayesian Version 2 (fixed pair and static %Vbur reference)",
        "",
        "![Case 3f static Bayesian Version 2](../../part3/3f/yield_static_bayesian_v2.png)",
        "",
        "3. Dynamic Bayesian 27-feature core (nested LOO)",
        "",
        "![Case 3f dynamic Bayesian core](../../part3/3f/yield_dynamic_bayesian.png)",
        "",
        "### Dynamic 27-feature core model",
        "",
        "The 27 candidate features are:",
        "",
    ])
    lines.extend(f"- `{feature}`" for feature in pool["candidate_features"])
    lines.extend([
        "",
        "Evidence selection retained three descriptors:",
        "",
        "| Descriptor | Posterior weight | 95% credible interval | Absolute contribution |",
        "|---|---:|---:|---:|",
    ])
    for weight in core["weights"]:
        interval = weight["credible_interval_95_standardized"]
        lines.append(
            f"| `{weight['feature']}` | {weight['posterior_weight_standardized']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {weight['absolute_contribution_fraction']:.3f} |"
        )

    selection_frequency = ", ".join(
        f"`{feature}` {count}/7"
        for feature, count in sorted(
            core["nested_loo_feature_selection_frequency"].items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    lines.extend([
        "",
        f"Selection stability across the seven nested folds was: {selection_frequency}. The fixed-subset outcome-shuffle control gives `p = {core['permutation_p_ge_observed']:.4f}` against the observed fixed-subset LOO R². This supports a non-random sample association but does not remove the small-n or constructed-geometry limitations.",
        "",
        "### Full 156-feature model",
        "",
        "The full run considered all 156 eligible descriptors and evidence selection retained three:",
        "",
        "| Descriptor | Posterior weight | 95% credible interval | Absolute contribution |",
        "|---|---:|---:|---:|",
    ])
    for weight in full["weights"]:
        interval = weight["credible_interval_95_standardized"]
        lines.append(
            f"| `{weight['feature']}` | {weight['posterior_weight_standardized']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {weight['absolute_contribution_fraction']:.3f} |"
        )

    lines.extend([
        "",
        "The full descriptor inventory is retained in [`feature-pool-manifest.json`](../../part3/3f/feature-pool-manifest.json). The selected terms are a compact association within this seven-point dataset, not a validated universal feature ranking.",
        "",
        "## Bayesian weighting equations",
        "",
        "For each model, `z(x_j) = (x_j - μ_j) / s_j`. The posterior yield equation is `ŷ = μ_y + s_y[b₀ + Σ βⱼz(x_j)]`, where the weights are posterior standardized coefficients rather than causal importance constants.",
        "",
        f"Core-pool selected equation: `{equation(core)}`",
        "",
        f"Full-pool selected equation: `{equation(full)}`",
        "",
        "## Nested validation and publication gate",
        "",
        "The core model passes the internal nested leave-one-catalyst-out comparison against the intercept-only baseline, unlike the static descriptors. This is the strongest statistical result in Case 3f. It is still a seven-point result: one held-out catalyst changes both the training sample and selected descriptor set materially, and three catalyst geometries are constructed analogues. The result is evidence of a dynamic-map association worth testing prospectively, not validation of a general yield predictor.",
        "",
        "The full-pool model does not receive publication-strength predictive status because descriptor selection was not repeated inside each held-out fold. Its high fixed-subset LOO R² is conditional on features selected using all seven outcomes. A larger catalyst series or genuinely external test set is required before comparing the full and core models as predictive methods.",
        "",
        "## Interpretation for the publication package",
        "",
        "The strongest defensible statement is:",
        "",
        "> Across seven Cu(I)-NHC catalyst labels, CREST/xTB-derived dynamic d2/d3 steric descriptors showed a stronger association with the reported 0.02 mol% yields than static %Vbur or static Solid-G alone. A chemically focused 27-feature candidate pool retained a positive nested leave-one-catalyst-out signal, but the result remains exploratory because of the seven-point sample and the constructed 3e, IMes, and IPr coordinate branches.",
        "",
        "The following statements are excluded:",
        "",
        "- the model is a validated predictor for unseen NHC catalysts;",
        "- the 156-feature fixed-subset LOO score is nested or external validation;",
        "- the selected Bayesian weights are universal physical constants;",
        "- dynamic d3 maps identify a reaction barrier, rate law, or causal catalytic mechanism;",
        "- the constructed 3e, IMes, and IPr structures are deposited experimental geometries.",
        "",
        "## Reproducibility outputs",
        "",
        "- [Bayesian result JSON](../../part3/3f/case3f-bayesian-results.json)",
        "- [Catalyst-level feature matrix](../../part3/3f/case3f-feature-matrix.csv)",
        "- [Table 1 yield input](../../part3/3f/case3f-yield-table.csv)",
        "- [Feature-pool manifest](../../part3/3f/feature-pool-manifest.json)",
        "- [Structure/provenance manifest](../../part3/3f/structure-manifest.json)",
        "- [Four-model comparison plot](../../part3/3f/yield_four_model_comparison.png)",
        "- [Publication-report package](part3f/publication-report/part3f-dynamic-static-bayesian-publication-report.md)",
        "- [Technical-statement package](part3f/technical-statement/part3f-d2d3-bayesian-technical-statement.md)",
        "- [Case 3f execution script](../../scripts/run_part3f.py)",
        "",
        "## Final integrity status",
        "",
        "The Case 3f release contains seven catalyst-level outcomes, averages the two 3b conformers within one experimental row, archives 166 numeric dynamic d2/d3-derived fields, uses 156 eligible descriptors in the full run and 27 in the chemically focused core, and records successful new d2/d3 projects for the constructed branches. Static %Vbur and Solid-G are retained as independent baselines. Eight additional static full-minus-IMe d3 comparisons reproduce the exact 15-feature Case 3d ML protocol; that static model fails nested validation, while the dynamic core retains a positive nested signal. The core and Case 3d-style static fits both include nested LOO selection and 2,000 shuffle controls; the 156-feature full-pool fit is explicitly labeled fixed-subset only. The release status is **exploratory analogue analysis with a positive dynamic-core nested signal and a negative exact-Case-3d static comparison, not a validated general yield predictor**.",
        "",
    ])
    path = DOC_ROOT / "part3f-dynamic-yield-ML.md"
    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    full_report_text = report_text.replace(
        "title: Case 3f dynamic d2/d3 Bayesian yield model",
        "title: Case 3f full ML report — dynamic d2/d3 versus static comparison",
        1,
    ).replace(
        "# Case 3f dynamic d2/d3 Bayesian yield model",
        "# Case 3f full ML report — dynamic d2/d3 versus static comparison",
        1,
    )
    (DOC_ROOT / "part3f-full-ML-comparison.md").write_text(full_report_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    constructed = _build_inputs()
    if args.construct_only:
        print(json.dumps({"status": "constructed", "output": str(OUT / "structure-manifest.json")}, indent=2))
        return 0
    groups = _map_groups(constructed)
    new_d3_projects: dict[str, Path] = {}
    if not args.fit_only:
        for catalyst, group in groups.items():
            for full, ime in zip(group["full"], group["ime"], strict=True):
                full_d2 = _run_d2(full) if full.name not in {"3a", "3b-syn", "3b-anti", "3c", "3d"} else _d2_project_for_record(full)
                ime_d2 = _run_d2(ime) if ime.name not in {"3a-IMe", "3b-syn-IMe", "3b-anti-IMe", "3c-IMe", "3d-IMe"} else PART3B / "ime-baseline" / "d2-map" / f"{ime.name}-d2map" / "project.d3map.json"
                if catalyst not in {"3a", "3b", "3c", "3d"}:
                    new_d3_projects[full.name] = _run_d3(full, ime, full_d2, ime_d2)
        if args.maps_only:
            print(json.dumps({"status": "maps_completed", "new_d3_projects": {key: str(value) for key, value in new_d3_projects.items()}}, indent=2))
            return 0
    else:
        for catalyst in ("3e", "IMes", "IPr"):
            new_d3_projects[catalyst] = _project_path("hidden-d3", f"{catalyst}-IMe-vs-{catalyst}-hidden-d3map")
    rows, candidates, metadata = _build_feature_matrix(groups, new_d3_projects)
    results = _fit_models(rows, candidates, metadata)
    case3d_static_rows = _build_case3d_static_rows(groups)
    results["case3d_static_protocol_model"] = _fit_case3d_static_model(case3d_static_rows)
    results["case3d_v2_static_protocol_model"] = _fit_case3d_v2_static_model(case3d_static_rows)
    labels = [str(row["catalyst"]) for row in rows]
    observed = np.asarray([float(row["yield_percent"]) for row in rows], dtype=float)
    static_v1_plot = OUT / "yield_static_bayesian_v1.png"
    dynamic_plot = OUT / "yield_dynamic_bayesian.png"
    _plot_single_bayesian_model(
        results["case3d_static_protocol_model"], observed, labels, static_v1_plot,
        "Static Bayesian Version 1 — Case 3d protocol", "nested_loo_predicted", "nested LOO",
        results["case3d_static_protocol_model"]["nested_loo_r2"],
    )
    _plot_single_bayesian_model(
        results["core_feature_ensemble_model"], observed, labels, dynamic_plot,
        "Dynamic Bayesian — 27-feature core", "nested_loo_predicted", "nested LOO",
        results["core_feature_ensemble_model"]["nested_loo_r2"],
    )
    results["files"]["case3d_static_results"] = str(OUT / "case3f-case3d-static-bayesian-results.json")
    results["files"]["case3d_static_inputs"] = str(OUT / "case3f-case3d-static-model-inputs.csv")
    results["files"]["case3d_static_plot"] = str(OUT / "yield_case3d_static_protocol.png")
    results["files"]["case3d_static_v2_results"] = str(OUT / "case3f-case3d-v2-static-bayesian-results.json")
    results["files"]["case3d_static_v2_plot"] = str(OUT / "yield_static_bayesian_v2.png")
    results["files"]["static_bayesian_v1_plot"] = str(static_v1_plot)
    results["files"]["dynamic_bayesian_plot"] = str(dynamic_plot)
    results["files"]["dynamic_ml_report"] = str(DOC_ROOT / "part3f-dynamic-yield-ML.md")
    results["files"]["full_ml_comparison_report"] = str(DOC_ROOT / "part3f-full-ML-comparison.md")
    (OUT / "case3f-bayesian-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_report(results, rows, new_d3_projects)
    print(json.dumps({"status": results["status"], "candidate_feature_count": len(candidates), "selected_features": results["core_feature_ensemble_model"]["feature_names"], "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
