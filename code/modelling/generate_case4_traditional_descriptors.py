#!/usr/bin/env python3
"""Calculate the Case 3f DBSTEP/Morfeus feature panel for Case 4."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from dbstep.Dbstep import dbstep
from morfeus import BuriedVolume, SASA, SolidAngle, Sterimol


ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "03-chemical-validation" / "4"
MAPS = CASE / "analysis" / "maps" / "d2-map"
OUTPUT = CASE / "analysis" / "results" / "case4-traditional-per-structure.csv"
MANIFEST = CASE / "analysis" / "results" / "case4-traditional-descriptor-manifest.json"
AXIS_ATOM = 148  # 1-based carbene carbon in all seven approved XYZ files
FEATURE_NAMES = [
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


def read_single_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"empty XYZ: {path}")
    atom_count = int(lines[0].strip())
    if len(lines) != atom_count + 2:
        raise ValueError(f"expected one XYZ geometry in {path}, got {len(lines)} lines")
    rows = [line.split() for line in lines[2:]]
    if any(len(row) < 4 for row in rows):
        raise ValueError(f"malformed XYZ atom row in {path}")
    elements = [row[0] for row in rows]
    coordinates = np.asarray([[float(v) for v in row[1:4]] for row in rows], dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"non-finite coordinate in {path}")
    return elements, coordinates


def write_xyz(path: Path, elements: list[str], coordinates: np.ndarray) -> None:
    rows = [str(len(elements)), "Case 4 static xTB-optimized geometry; Case 3f descriptor mask"]
    rows.extend(
        f"{element:<3} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for element, xyz in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def scalar(value: Any) -> float:
    result = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"descriptor returned a non-finite value: {value!r}")
    return result


def evaluate_geometry(
    geometry_path: Path, project_path: Path, temp: Path, structure_id: str
) -> tuple[dict[str, float], dict[str, Any]]:
    elements, coordinates = read_single_xyz(geometry_path)
    config = json.loads(project_path.read_text(encoding="utf-8"))
    settings = config["structures"]["reference"]
    center_atom = int(settings["center_atom"])
    steric_atoms = [int(atom) for atom in settings["steric_atoms"]]
    if center_atom < 1 or center_atom > len(elements):
        raise ValueError(f"{structure_id}: invalid center atom {center_atom}")
    if AXIS_ATOM > len(elements):
        raise ValueError(f"{structure_id}: carbene atom {AXIS_ATOM} is outside the XYZ")
    if elements[center_atom - 1] != "Au" or elements[AXIS_ATOM - 1] != "C":
        raise ValueError(
            f"{structure_id}: expected Au center and carbene C axis, found "
            f"{elements[center_atom - 1]}/{elements[AXIS_ATOM - 1]}"
        )
    ordered_atoms = [
        center_atom,
        AXIS_ATOM,
        *[atom for atom in steric_atoms if atom not in {center_atom, AXIS_ATOM}],
    ]
    if len(set(ordered_atoms)) != len(ordered_atoms):
        raise ValueError(f"{structure_id}: duplicate atoms in descriptor mask")
    if any(atom < 1 or atom > len(elements) for atom in ordered_atoms):
        raise ValueError(f"{structure_id}: steric mask contains an invalid atom index")
    indices = np.asarray(ordered_atoms, dtype=int) - 1
    reduced_elements = [elements[index] for index in indices]
    reduced_coordinates = coordinates[indices]
    reduced_path = temp / f"{structure_id}.xyz"
    write_xyz(reduced_path, reduced_elements, reduced_coordinates)

    # These settings and the reduced center-to-axis geometry match Case 3f.
    dbstep_vbur = dbstep(
        str(reduced_path), atom1=1, volume=True, r=3.5, grid=0.10,
        scalevdw=1.17, quiet=True,
    )
    dbstep_sterimol = dbstep(
        str(reduced_path), atom1=1, atom2=2, sterimol=True,
        measure="classic", scalevdw=1.0, quiet=True,
    )
    buried_volume = BuriedVolume(
        reduced_elements, reduced_coordinates, 1, include_hs=True,
        radius=3.5, radii_type="bondi", radii_scale=1.17, density=0.01,
    )
    solid_angle = SolidAngle(
        reduced_elements, reduced_coordinates, 1,
        radii_type="crc", density=0.01,
    )
    sterimol = Sterimol(
        reduced_elements, reduced_coordinates, 1, 2, radii_type="bondi",
    )
    # Case 3f computes surface area/volume for all retained steric atoms except
    # the center atom; this preserves that convention for the Au-centered case.
    sasa = SASA(reduced_elements[1:], reduced_coordinates[1:], radii_type="crc", density=0.05)
    features = {
        "dbstep_vbur_percent": scalar(dbstep_vbur.bur_vol),
        "dbstep_L_A": scalar(dbstep_sterimol.L),
        "dbstep_Bmin_A": scalar(dbstep_sterimol.Bmin),
        "dbstep_Bmax_A": scalar(dbstep_sterimol.Bmax),
        "morfeus_vbur_percent": float(buried_volume.fraction_buried_volume * 100.0),
        "morfeus_G_percent": scalar(solid_angle.G),
        "morfeus_cone_angle_deg": scalar(solid_angle.cone_angle),
        "morfeus_solid_angle_sr": scalar(solid_angle.solid_angle),
        "morfeus_L_A": scalar(sterimol.L_value),
        "morfeus_B1_A": scalar(sterimol.B_1_value),
        "morfeus_B5_A": scalar(sterimol.B_5_value),
        "morfeus_SASA_A2": scalar(sasa.area),
        "morfeus_SASA_volume_A3": scalar(sasa.volume),
    }
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError(f"{structure_id}: one or more traditional descriptors are non-finite")
    provenance = {
        "geometry": str(geometry_path.relative_to(CASE)),
        "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        "project": str(project_path.relative_to(CASE)),
        "atom_count": len(elements),
        "reduced_atom_count": len(reduced_elements),
        "center_atom_1based": center_atom,
        "center_element": reduced_elements[0],
        "axis_atom_1based": AXIS_ATOM,
        "axis_element": reduced_elements[1],
        "steric_mask_atom_count": len(steric_atoms),
    }
    return features, provenance


def main() -> None:
    projects = sorted(MAPS.glob("*-d2map/project.d3map.json"))
    projects = [path for path in projects if not path.parent.name.startswith("common-core-R-H-reference")]
    if len(projects) != 7:
        raise RuntimeError(f"expected seven full-complex d2-map projects, found {len(projects)}")
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="case4-traditional-descriptors-") as temp_dir:
        temp = Path(temp_dir)
        for project in projects:
            structure_id = project.parent.name.removesuffix("-d2map")
            geometry = project.parent / "results/reference/baselines/input/aligned-ensemble.xyz"
            if not geometry.is_file():
                raise FileNotFoundError(f"{structure_id}: static D2 input geometry missing: {geometry}")
            features, structure_provenance = evaluate_geometry(geometry, project, temp, structure_id)
            rows.append({"structure_id": structure_id, **features})
            provenance[structure_id] = structure_provenance
            print(f"{structure_id}: Case 3f DBSTEP/Morfeus descriptors complete", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["structure_id", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "case4.traditional-descriptors.v1",
        "status": "completed",
        "structure_count": len(rows),
        "feature_names_external": FEATURE_NAMES,
        "case3f_complete_traditional_pool": [
            "static_vbur_percent", "static_g_percent", *FEATURE_NAMES,
        ],
        "packages": {
            "dbstep": importlib.metadata.version("dbstep"),
            "morfeus-ml": importlib.metadata.version("morfeus-ml"),
        },
        "python": platform.python_version(),
        "geometry_policy": "one static xTB-optimized input geometry per complex, read from its d2-map input baseline; no conformer or trajectory averaging",
        "axis_policy": "Au atom 2 to carbene carbon atom 148; indices are 1-based",
        "atom_mask_policy": "D2-map project's steric_atoms plus center and axis atoms, ordered as center then axis; SASA excludes the center atom as in Case 3f",
        "descriptor_settings": {
            "DBSTEP_vbur": {"radius_A": 3.5, "grid_A": 0.10, "vdw_scale": 1.17},
            "DBSTEP_Sterimol": {"measure": "classic", "vdw_scale": 1.0},
            "Morfeus_BuriedVolume": {"radius_A": 3.5, "radii": "Bondi", "radii_scale": 1.17, "density": 0.01, "include_hydrogens": True},
            "Morfeus_SolidAngle": {"radii": "CRC", "density": 0.01},
            "Morfeus_Sterimol": {"radii": "Bondi"},
            "Morfeus_SASA": {"radii": "CRC", "density": 0.05},
        },
        "structures": provenance,
        "output": str(OUTPUT.relative_to(CASE)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "structures": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
