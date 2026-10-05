from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ValidationError
from .io import read_xyz_ensemble
from .project import analyze_project
from .sampling import XtbCrestConfig, read_xyz_comment_energies, run_xtb_crest


# Bump when trajectory-energy interpretation changes.  This invalidates old
# uniform-weight manifests so an existing raw CREST/xTB trajectory is
# reprocessed with the corrected electronic-energy semantics.
ENERGY_HANDOFF_VERSION = "trajectory-energy-v1"


def _write_subset(
    source: Path,
    target: Path,
    stride: int,
    energies: np.ndarray | None = None,
) -> int:
    ensemble = read_xyz_ensemble(source)
    geometries = ensemble.geometries[::stride]
    selected_energies = None if energies is None else np.asarray(energies, dtype=float)[::stride]
    if selected_energies is not None and selected_energies.shape != (len(geometries),):
        raise ValidationError("trajectory energy count does not match sampled frames")
    with target.open("w", encoding="utf-8") as stream:
        for index, geometry in enumerate(geometries):
            source_index = index * stride + 1
            if selected_energies is None:
                comment = f"trajectory frame {source_index}"
            else:
                comment = (
                    f"energy: {selected_energies[index]:.12f} "
                    f"trajectory frame {source_index}"
                )
            stream.write(f"{len(geometry.elements)}\n{comment}\n")
            for element, coordinate in zip(geometry.elements, geometry.coordinates, strict=True):
                stream.write(
                    f"{element} {coordinate[0]:.10f} {coordinate[1]:.10f} {coordinate[2]:.10f}\n"
                )
    return len(geometries)


def run_project(path: str | Path) -> Any:
    """Execute resumable xTB/CREST sampling and the shared downstream analysis."""
    project = Path(path).resolve()
    config = json.loads(project.read_text(encoding="utf-8"))
    sampling = config.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("backend") != "xtb-crest":
        return analyze_project(project)
    source = project.parent / config["input"]
    fingerprint = hashlib.sha256(
        source.read_bytes()
        + json.dumps(sampling, sort_keys=True).encode()
        + ENERGY_HANDOFF_VERSION.encode()
    ).hexdigest()
    stage = project.parent / "stages" / "sampling"
    stage.mkdir(parents=True, exist_ok=True)
    manifest_path = stage / "manifest.json"
    derived_ensemble = stage / "analysis-ensemble.xyz"
    manifest: dict[str, Any] | None = None
    sampled_energies: np.ndarray | None = None
    if manifest_path.is_file():
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            candidate.get("fingerprint") == fingerprint
            and candidate.get("status") == "success"
            and derived_ensemble.is_file()
        ):
            manifest = candidate
    if manifest is None:
        raw_contacts = sampling.get("protected_contacts", [])
        if any(len(pair) != 2 for pair in raw_contacts):
            raise ValidationError("every protected contact requires exactly two atom numbers")
        contacts: tuple[tuple[int, int], ...] = tuple(
            (int(pair[0]), int(pair[1])) for pair in raw_contacts
        )
        result = run_xtb_crest(
            source,
            stage,
            config=XtbCrestConfig(
                charge=int(config.get("charge", 0)),
                multiplicity=int(config.get("multiplicity", 1)),
                frozen_atom_numbers=tuple(config.get("frozen_atoms", [])),
                protected_contacts=contacts,
                solvent=config.get("solvent"),
                threads=int(sampling.get("threads", 1)),
                quick=bool(sampling.get("quick", False)),
                maximum_reduced=bool(sampling.get("maximum_reduced", False)),
                constrain_preoptimization=bool(
                    sampling.get("constrain_preoptimization", True)
                ),
                allow_trajectory_fallback=bool(sampling.get("allow_trajectory_fallback", False)),
                metadyn_steps=int(sampling.get("metadyn_steps", 25)),
                metadyn_time_ps=float(sampling.get("metadyn_time_ps", 5.0)),
                crest_mdlen_ps=(
                    None
                    if sampling.get("crest_mdlen_ps") is None
                    else float(sampling["crest_mdlen_ps"])
                ),
            ),
            xtb_executable=sampling.get("xtb_executable", "xtb"),
            crest_executable=sampling.get("crest_executable", "crest"),
        )
        energy_values = np.asarray(result.electronic_energies_hartree, dtype=float)
        has_trajectory_energies = (
            result.analysis_mode == "trajectory"
            and energy_values.shape == (read_xyz_ensemble(result.ensemble_xyz).size,)
            and np.all(np.isfinite(energy_values))
        )
        stride = (
            int(sampling.get("trajectory_stride", 1)) if result.analysis_mode == "trajectory" else 1
        )
        if stride < 1:
            raise ValidationError("trajectory_stride must be positive")
        frame_count = _write_subset(
            result.ensemble_xyz,
            derived_ensemble,
            stride,
            energy_values if has_trajectory_energies else None,
        )
        analysis_mode = "ensemble" if has_trajectory_energies else result.analysis_mode
        manifest = {
            "status": "success",
            "fingerprint": fingerprint,
            "analysis_mode": analysis_mode,
            "source_analysis_mode": result.analysis_mode,
            "frame_count": frame_count,
            "backend_versions": result.backend_versions,
            "source_ensemble": str(result.ensemble_xyz),
            "trajectory_stride": stride,
        }
        if analysis_mode == "ensemble":
            sampled_energies = energy_values[::stride]
            if result.block_labels is not None:
                manifest["conformer_origin_blocks"] = list(result.block_labels)
            if has_trajectory_energies:
                manifest["energy_provenance"] = (
                    "finite xTB electronic energies recovered from CREST/xTB "
                    "trajectory comments; no CREST conformer-origin table"
                )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif manifest["analysis_mode"] == "ensemble":
        sampled_energies = read_xyz_comment_energies(Path(manifest["source_ensemble"]))
    derived = dict(config)
    derived["input"] = str(derived_ensemble)
    derived["output"] = str((project.parent / config.get("output", "dmap-results")).resolve())
    derived["analysis_mode"] = manifest["analysis_mode"]
    if "conformer_origin_blocks" in manifest:
        derived["replicate_blocks"] = manifest["conformer_origin_blocks"]
    if manifest["analysis_mode"] == "trajectory":
        derived["population"] = {"model": "uniform"}
    configured_population = derived.get("population")
    if (
        manifest["analysis_mode"] == "ensemble"
        and isinstance(configured_population, dict)
        and configured_population.get("model") == "electronic_energy"
        and "energies" not in configured_population
    ):
        if sampled_energies is None:
            raise ValidationError("xTB/CREST ensemble energies are unavailable for population weighting")
        derived["population"] = {
            **configured_population,
            "energies": sampled_energies.tolist(),
            "unit": "hartree",
            "temperature": configured_population.get("temperature", config.get("temperature", 298.15)),
        }
    derived_path = stage / "analysis-project.json"
    derived_path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    return analyze_project(derived_path)
