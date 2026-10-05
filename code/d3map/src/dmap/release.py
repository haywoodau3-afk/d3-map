from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .artifacts import refresh_manifest, write_comparison_reports, write_structure_reports
from .errors import ValidationError
from .io import read_xyz_ensemble
from .project import analyze_project, regenerate_project_report
from .sampling import read_xyz_comment_energies
from .workflow import run_project as run_legacy_project

SCHEMA_VERSION = "1.0"
PROJECT_KINDS = ("d2-map", "d3-map")
STRUCTURE_ROLES = ("reference", "extended")
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

DEFAULT_SHARED: dict[str, Any] = {
    "temperature": 298.15,
    "static": {"sphere_radius": 3.5, "spacing": 0.1, "direction_count": 40_962},
    "field": {"spacing": 0.15},
    "axial": {"transverse_radius": 3.5, "z_min": 0.0, "z_max": 6.0, "spacing": 0.1},
    "displaced_scan": {
        "z_min": 0.0,
        "z_max": 2.0,
        "spacing": 0.1,
        "sphere_spacing": 0.1,
        "direction_count": 40_962,
    },
    "reporting": {
        "enabled": True,
        "z_min": 0.0,
        "z_max": 2.0,
        "z_spacing": 0.1,
        "topographic_spacing": 0.1,
        "ml_fraction_resolutions": [1, 2, 4, 8],
        "office_reports": True,
    },
    "vbur_radii_profile": "sambvca_2.1",
    "field_radii_profile": "dmap_vdw_v0.1",
}

DEFAULT_COMPARISON: dict[str, Any] = {
    "difference_sign": "extended-minus-reference",
    "allow_nonstandard": False,
    "tolerances": {
        "contact_probability": 0.01,
        "first_contact_q50": 0.05,
        "first_contact_interval_10_90": 0.05,
        "occupied_depth": 0.05,
    },
}


@dataclass(frozen=True)
class ReleaseRunResult:
    project: Path
    kind: str
    structure_outputs: dict[str, Path]
    comparison_output: Path | None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read d3map project: {path}") from error
    if not isinstance(value, dict):
        raise ValidationError("d3map project must contain a JSON object")
    return value


def read_release_project(path: str | Path) -> dict[str, Any]:
    project = Path(path).resolve()
    value = _read_json(project)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"project schema_version must be {SCHEMA_VERSION}")
    if value.get("project_kind") not in PROJECT_KINDS:
        raise ValidationError("project_kind must be d2-map or d3-map")
    structures = value.get("structures")
    if not isinstance(structures, dict) or "reference" not in structures:
        raise ValidationError("project requires a reference structure")
    if value["project_kind"] == "d3-map" and "extended" not in structures:
        raise ValidationError("d3-map project requires an extended structure")
    return value


def is_release_project(path: str | Path) -> bool:
    try:
        return _read_json(Path(path).resolve()).get("schema_version") == SCHEMA_VERSION
    except ValidationError:
        return False


def _unique_directory(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    for index in range(2, 10_000):
        numbered = candidate.with_name(f"{candidate.name}-{index}")
        if not numbered.exists():
            return numbered
    raise ValidationError(f"cannot allocate a unique project directory beside {candidate}")


def _normal_structure_config(
    input_name: str,
    *,
    center_atom: int,
    reactive_direction: list[float],
    secondary_direction: list[float],
    alignment_atoms: list[int],
    charge: int,
    multiplicity: int,
    solvent: str | None,
    confirmed: bool,
    protected_contacts: list[list[int]] | None,
    frozen_atoms: list[int] | None,
    steric_atoms: list[int] | None,
    xtb_executable: str | None,
    crest_executable: str | None,
    threads: int,
) -> dict[str, Any]:
    structure: dict[str, Any] = {
        "input": input_name,
        "center_atom": center_atom,
        "origin_atoms": [center_atom],
        "reactive_direction": reactive_direction,
        "secondary_direction": secondary_direction,
        "frame_semantics": "right-handed:+z-reactive:+x-secondary",
        "alignment_atoms": alignment_atoms,
        "frozen_atoms": frozen_atoms or [],
        "charge": charge,
        "multiplicity": multiplicity,
        "solvent": solvent,
        "chemical_setup_confirmed": confirmed,
        "topology": {"policy": "strict"},
        "population": {"model": "electronic_energy", "unit": "hartree"},
        "sampling": {
            "backend": "xtb-crest",
            "threads": threads,
            "protected_contacts": protected_contacts or [],
            "allow_trajectory_fallback": False,
        },
    }
    if steric_atoms is not None:
        structure["steric_atoms"] = steric_atoms
    if xtb_executable:
        structure["sampling"]["xtb_executable"] = xtb_executable
    if crest_executable:
        structure["sampling"]["crest_executable"] = crest_executable
    return structure


def create_release_project(
    *,
    kind: str,
    reference_xyz: str | Path,
    extended_xyz: str | Path | None = None,
    output_directory: str | Path | None = None,
    reference_setup: dict[str, Any],
    extended_setup: dict[str, Any] | None = None,
    shared: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> Path:
    """Create a collision-safe portable d2-map or d3-map project."""
    if kind not in PROJECT_KINDS:
        raise ValidationError("kind must be d2-map or d3-map")
    reference = Path(reference_xyz).resolve()
    if not reference.is_file():
        raise ValidationError(f"reference XYZ not found: {reference}")
    extended = None if extended_xyz is None else Path(extended_xyz).resolve()
    if kind == "d3-map" and (extended is None or not extended.is_file()):
        raise ValidationError("d3-map requires an existing extended XYZ")
    if output_directory is None:
        if kind == "d3-map":
            assert extended is not None
            name = f"{reference.stem}-vs-{extended.stem}-d3map"
        else:
            name = f"{reference.stem}-d2map"
        root = _unique_directory(reference.parent / name)
    else:
        requested = Path(output_directory).resolve()
        root = _unique_directory(requested) if requested.exists() else requested
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    reference_target = inputs / f"reference-{reference.name}"
    shutil.copy2(reference, reference_target)
    structures: dict[str, Any] = {
        "reference": {**reference_setup, "input": str(reference_target.relative_to(root))}
    }
    if extended is not None:
        extended_target = inputs / f"extended-{extended.name}"
        shutil.copy2(extended, extended_target)
        structures["extended"] = {
            **(extended_setup or reference_setup),
            "input": str(extended_target.relative_to(root)),
        }
    config = {
        "schema_version": SCHEMA_VERSION,
        "project_kind": kind,
        "created_with": __version__,
        "created_at": _utc_now(),
        "local_processing_only": True,
        "structures": structures,
        "shared": {**DEFAULT_SHARED, **(shared or {})},
        "comparison": {**DEFAULT_COMPARISON, **(comparison or {})},
        "output": "results",
    }
    project = root / "project.d3map.json"
    project.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return project


def migrate_v03_project(path: str | Path, output: str | Path | None = None) -> Path:
    """Migrate one schema-0.1 project without changing or deleting the source."""
    source = Path(path).resolve()
    old = _read_json(source)
    if old.get("schema_version") != "0.1":
        raise ValidationError("only d-Map schema 0.1 projects can be migrated")
    backup = source.with_suffix(source.suffix + ".v0.3.backup")
    if not backup.exists():
        shutil.copy2(source, backup)
    input_path = (source.parent / old["input"]).resolve()
    setup = {
        key: value
        for key, value in old.items()
        if key
        not in {
            "schema_version", "input", "output", "static", "field", "axial",
            "displaced_scan", "reporting", "temperature", "vbur_radii_profile",
            "field_radii_profile",
        }
    }
    setup.setdefault("chemical_setup_confirmed", False)
    shared = {
        key: old[key]
        for key in (
            "static", "field", "axial", "displaced_scan", "reporting", "temperature",
            "vbur_radii_profile", "field_radii_profile",
        )
        if key in old
    }
    target_dir = Path(output).resolve() if output else source.parent / f"{input_path.stem}-d2map"
    return create_release_project(
        kind="d2-map",
        reference_xyz=input_path,
        output_directory=target_dir,
        reference_setup=setup,
        shared=shared,
    )


def _structure_roles(config: dict[str, Any]) -> tuple[str, ...]:
    return ("reference",) if config["project_kind"] == "d2-map" else STRUCTURE_ROLES


def _is_transition_metal_structure(project_root: Path, structure: dict[str, Any]) -> bool:
    ensemble = read_xyz_ensemble(project_root / structure["input"])
    center = int(structure["center_atom"])
    if center < 1 or center > len(ensemble.elements):
        raise ValidationError("center_atom is outside the input geometry")
    return ensemble.elements[center - 1] in TRANSITION_METALS


def _validate_confirmation(project_root: Path, role: str, structure: dict[str, Any]) -> None:
    if _is_transition_metal_structure(project_root, structure) and not structure.get(
        "chemical_setup_confirmed", False
    ):
        raise ValidationError(
            f"{role} transition-metal chemical setup must be explicitly confirmed before sampling"
        )
    if "charge" not in structure or "multiplicity" not in structure:
        raise ValidationError(f"{role} requires explicit charge and multiplicity")


def _merge_structure_config(
    project_root: Path,
    release_config: dict[str, Any],
    role: str,
    *,
    sampling: bool,
) -> tuple[Path, Path]:
    structure = release_config["structures"][role]
    _validate_confirmation(project_root, role, structure)
    stage_root = project_root / "stages" / role
    stage_root.mkdir(parents=True, exist_ok=True)
    output = project_root / release_config.get("output", "results") / role
    legacy = {
        "schema_version": "0.1",
        **release_config.get("shared", {}),
        **structure,
        "input": str((project_root / structure["input"]).resolve()),
        "output": str(output.resolve()),
    }
    if not sampling:
        legacy.pop("sampling", None)
        population = legacy.get("population", {})
        if population.get("model") in {"electronic_energy", "free_energy"} and not population.get(
            "energies"
        ):
            legacy["population"] = {"model": "uniform"}
    project = stage_root / "analysis-project.json"
    project.write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return project, output


def _normalized(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values)
    if values.shape != (3,) or norm <= 1e-12:
        raise ValidationError("comparison frame directions must be finite nonzero 3-vectors")
    return values / norm


def validate_pair_compatibility(project_root: Path, config: dict[str, Any]) -> list[str]:
    if config["project_kind"] != "d3-map":
        return []
    reference = config["structures"]["reference"]
    extended = config["structures"]["extended"]
    mismatches: list[str] = []
    reference_elements = read_xyz_ensemble(project_root / reference["input"]).elements
    extended_elements = read_xyz_ensemble(project_root / extended["input"]).elements
    if reference_elements[int(reference["center_atom"]) - 1] != extended_elements[
        int(extended["center_atom"]) - 1
    ]:
        mismatches.append("catalytic-centre element")
    if len(reference.get("origin_atoms", [reference["center_atom"]])) != len(
        extended.get("origin_atoms", [extended["center_atom"]])
    ):
        mismatches.append("origin type")
    for key in ("reactive_direction", "secondary_direction"):
        _normalized(reference[key])
        _normalized(extended[key])
    if reference.get("frame_semantics") != extended.get("frame_semantics"):
        mismatches.append("frame semantics")
    shared = config.get("shared", {})
    for key in (
        "temperature", "static", "field", "axial", "displaced_scan", "reporting",
        "vbur_radii_profile", "field_radii_profile",
    ):
        if key not in shared:
            mismatches.append(f"shared.{key}")
    if mismatches and not config.get("comparison", {}).get("allow_nonstandard", False):
        raise ValidationError(
            "standard d3-map comparison is incompatible: " + ", ".join(sorted(set(mismatches)))
        )
    return sorted(set(mismatches))


def _safe_relative(reference: float, difference: float) -> float | None:
    return None if abs(reference) <= 1e-12 else difference / reference


def _safe_relative_array(reference: np.ndarray, difference: np.ndarray) -> np.ndarray:
    relative = np.full(reference.shape, np.nan, dtype=float)
    np.divide(
        difference,
        reference,
        out=relative,
        where=np.isfinite(reference) & np.isfinite(difference) & (np.abs(reference) > 1e-12),
    )
    return relative


def _scalar_comparison(reference: dict[str, Any], extended: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for descriptor in ("vbur_percent", "g_percent"):
        reference_summary = reference["ensemble_scalar_descriptors"][descriptor]
        extended_summary = extended["ensemble_scalar_descriptors"][descriptor]
        for statistic in (
            "mean", "population_standard_deviation", "percentile_5", "percentile_95",
            "percentile_span", "minimum", "maximum", "lowest_energy_value",
        ):
            left = reference_summary.get(statistic)
            right = extended_summary.get(statistic)
            difference = None if left is None or right is None else float(right) - float(left)
            rows.append(
                {
                    "descriptor": descriptor,
                    "statistic": statistic,
                    "reference": left,
                    "extended": right,
                    "difference_extended_minus_reference": difference,
                    "absolute_difference": None if difference is None else abs(difference),
                    "relative_difference": (
                        None if difference is None else _safe_relative(float(left), difference)
                    ),
                }
            )
    return rows


def _static_baseline_comparison(
    reference_output: Path, extended_output: Path
) -> list[dict[str, Any]]:
    reference_path = reference_output / "static-baselines.json"
    extended_path = extended_output / "static-baselines.json"
    if not reference_path.is_file() or not extended_path.is_file():
        return []
    reference = _read_json(reference_path)["baselines"]
    extended = _read_json(extended_path)["baselines"]
    rows: list[dict[str, Any]] = []
    for baseline in ("input", "preoptimized", "lowest-energy"):
        left = reference.get(baseline, {})
        right = extended.get(baseline, {})
        for descriptor in ("vbur_percent", "g_percent"):
            reference_value = left.get(descriptor) if left.get("status") == "available" else None
            extended_value = right.get(descriptor) if right.get("status") == "available" else None
            difference = (
                None
                if reference_value is None or extended_value is None
                else float(extended_value) - float(reference_value)
            )
            relative_difference = None
            if difference is not None and reference_value is not None:
                relative_difference = _safe_relative(float(reference_value), difference)
            rows.append(
                {
                    "baseline": baseline,
                    "descriptor": descriptor,
                    "reference": reference_value,
                    "extended": extended_value,
                    "difference_extended_minus_reference": difference,
                    "absolute_difference": None if difference is None else abs(difference),
                    "relative_difference": relative_difference,
                    "reference_status": left.get("status", "unavailable"),
                    "extended_status": right.get("status", "unavailable"),
                }
            )
    return rows


def _difference_arrays(
    reference_output: Path,
    extended_output: Path,
    comparison: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(reference_output / "axial-fields.npz", allow_pickle=False) as left, np.load(
        extended_output / "axial-fields.npz", allow_pickle=False
    ) as right:
        for axis in ("x", "y", "z"):
            if not np.allclose(left[axis], right[axis], atol=1e-10):
                raise ValidationError(f"paired axial {axis} coordinates differ")
            arrays[f"axial_{axis}"] = left[axis]
        for field in (
            "occupation",
            "entropy",
            "shielding_probability",
            "pocket_occupation",
            "pocket_entropy",
        ):
            if left[field].shape != right[field].shape:
                raise ValidationError(f"paired {field} field shapes differ")
            arrays[f"{field}_reference"] = left[field]
            arrays[f"{field}_extended"] = right[field]
            delta = right[field] - left[field]
            arrays[f"{field}_difference"] = delta
            arrays[f"{field}_absolute_difference"] = np.abs(delta)
            arrays[f"{field}_relative_difference"] = _safe_relative_array(left[field], delta)
            if field in {"occupation", "shielding_probability", "pocket_occupation"}:
                arrays[f"{field}_changed"] = np.isfinite(delta) & (np.abs(delta) > 0.01)
    left_v2 = reference_output / "probability-fields-v2.npz"
    right_v2 = extended_output / "probability-fields-v2.npz"
    if left_v2.is_file() and right_v2.is_file():
        with np.load(left_v2, allow_pickle=False) as left, np.load(
            right_v2, allow_pickle=False
        ) as right:
            if not np.allclose(left["axis"], right["axis"], atol=1e-10):
                raise ValidationError("paired probability-field v2 axes differ")
            arrays["v2_axis"] = left["axis"]
            for field in (
                "occupation",
                "entropy",
                "ensemble_residual_occupation",
                "shielding_probability",
                "ensemble_residual_shielding",
            ):
                if field not in left.files or field not in right.files:
                    continue
                if left[field].shape != right[field].shape:
                    raise ValidationError(f"paired v2 {field} shapes differ")
                delta = right[field] - left[field]
                name = (
                    "residual_double_difference_occupation"
                    if field == "ensemble_residual_occupation"
                    else "residual_double_difference_shielding"
                    if field == "ensemble_residual_shielding"
                    else f"v2_{field}_difference"
                )
                arrays[name] = delta
                arrays[f"{name}_absolute"] = np.abs(delta)
            if "adaptive_mask" in left.files and "adaptive_mask" in right.files:
                arrays["v2_adaptive_class_changed"] = left["adaptive_mask"] != right[
                    "adaptive_mask"
                ]
    left_top = reference_output / "plot-data" / "topographic-fields.npz"
    right_top = extended_output / "plot-data" / "topographic-fields.npz"
    if left_top.is_file() and right_top.is_file():
        tolerances = comparison.get("tolerances", DEFAULT_COMPARISON["tolerances"])
        with np.load(left_top, allow_pickle=False) as left, np.load(
            right_top, allow_pickle=False
        ) as right:
            for coordinate in ("axis", "offsets", "direction_names"):
                if not np.array_equal(left[coordinate], right[coordinate]):
                    raise ValidationError(f"paired topographic {coordinate} values differ")
                arrays[f"topographic_{coordinate}"] = left[coordinate]
            for field in (
                "contact_probability", "first_contact_q50",
                "first_contact_interval_10_90", "occupied_depth",
            ):
                if left[field].shape != right[field].shape:
                    raise ValidationError(f"paired topographic {field} shapes differ")
                delta = right[field] - left[field]
                tolerance = float(tolerances[field])
                arrays[f"topographic_{field}_reference"] = left[field]
                arrays[f"topographic_{field}_extended"] = right[field]
                arrays[f"topographic_{field}_difference"] = delta
                arrays[f"topographic_{field}_absolute_difference"] = np.abs(delta)
                arrays[f"topographic_{field}_relative_difference"] = _safe_relative_array(
                    left[field], delta
                )
                arrays[f"topographic_{field}_changed"] = np.isfinite(delta) & (
                    np.abs(delta) > tolerance
                )
    return arrays


def _write_difference_figures(
    output: Path,
    arrays: dict[str, np.ndarray],
    tolerances: dict[str, float],
) -> list[str]:
    """Render every six-direction topographic difference with tolerance masking."""
    required = {"topographic_axis", "topographic_offsets", "topographic_direction_names"}
    if not required.issubset(arrays):
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = np.asarray(arrays["topographic_axis"], dtype=float)
    offsets = np.asarray(arrays["topographic_offsets"], dtype=float)
    directions = tuple(str(value) for value in arrays["topographic_direction_names"])
    written: list[str] = []
    figure_root = output / "figures" / "topographic-differences"
    for field in (
        "contact_probability",
        "first_contact_q50",
        "first_contact_interval_10_90",
        "occupied_depth",
    ):
        delta = arrays[f"topographic_{field}_difference"]
        changed = arrays[f"topographic_{field}_changed"]
        finite = np.abs(delta[np.isfinite(delta)])
        limit = max(
            float(np.quantile(finite, 0.99)) if finite.size else 0.0,
            float(tolerances[field]),
        )
        for direction_index, direction in enumerate(directions):
            safe_direction = direction.replace("+", "plus-").replace("-", "minus-")
            target = figure_root / field / safe_direction
            target.mkdir(parents=True, exist_ok=True)
            for offset_index, offset in enumerate(offsets):
                values = np.ma.masked_where(
                    ~changed[direction_index, offset_index],
                    delta[direction_index, offset_index],
                )
                figure, axes = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
                axes.set_facecolor("#eceff3")
                image = axes.imshow(
                    values.T,
                    origin="lower",
                    extent=(axis[0], axis[-1], axis[0], axis[-1]),
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="nearest",
                )
                axes.set(
                    title=f"{field.replace('_', ' ')}: {direction} at {offset:.2f} Å",
                    xlabel="local horizontal (Å)",
                    ylabel="local vertical (Å)",
                    aspect="equal",
                )
                figure.colorbar(image, ax=axes, label="extended − reference")
                figure.text(
                    0.01,
                    0.01,
                    f"Grey: |Δ| ≤ {float(tolerances[field]):g}; raw values retained in NPZ",
                    fontsize=8,
                )
                path = target / f"offset-{offset_index:03d}-{offset:+.2f}A.png"
                figure.savefig(path, dpi=160)
                plt.close(figure)
                written.append(str(path.relative_to(output)))
    return written


def _write_v2_difference_figures(
    output: Path, arrays: dict[str, np.ndarray]
) -> list[str]:
    """Render direct and ensemble-residual v2 differences with fixed signed scales."""
    if "v2_axis" not in arrays:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis_values = np.asarray(arrays["v2_axis"], dtype=float)
    fields = (
        ("v2_occupation_difference", "Occupation difference"),
        ("v2_entropy_difference", "Occupancy-entropy difference"),
        ("residual_double_difference_occupation", "Ensemble-residual double difference"),
    )
    available = tuple(item for item in fields if item[0] in arrays)
    if not available:
        return []
    figure, axes = plt.subplots(1, len(available), figsize=(5.3 * len(available), 4.5), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    for axis_object, (key, title) in zip(axes_array, available, strict=True):
        values = np.asarray(arrays[key], dtype=float)
        score = np.nansum(np.abs(values), axis=(0, 1))
        index = int(np.argmax(score))
        slice_values = values[:, :, index]
        finite = np.abs(slice_values[np.isfinite(slice_values)])
        limit = max(float(np.quantile(finite, 0.99)) if finite.size else 0.0, 1e-12)
        image = axis_object.imshow(
            slice_values.T,
            origin="lower",
            extent=(axis_values[0], axis_values[-1], axis_values[0], axis_values[-1]),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis_object.set(
            title=f"{title}\nz = {axis_values[index]:.2f} Å",
            xlabel="x / Å",
            ylabel="y / Å",
            aspect="equal",
        )
        figure.colorbar(image, ax=axis_object, shrink=0.8, label="extended − reference")
    target = output / "figures" / "probability-v2-differences.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return [str(target.relative_to(output))]


def _write_manifest(directory: Path, schema: str, status: str = "success") -> Path:
    artifacts = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema": schema,
        "status": status,
        "software_version": __version__,
        "created_at": _utc_now(),
        "artifacts": artifacts,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def compare_release_outputs(project: Path, config: dict[str, Any], mismatches: list[str]) -> Path:
    root = project.parent
    results = root / config.get("output", "results")
    reference_output = results / "reference"
    extended_output = results / "extended"
    if not reference_output.is_dir() or not extended_output.is_dir():
        raise ValidationError("both d2-map branches must complete before d3-map comparison")
    reference = _read_json(reference_output / "descriptors.json")
    extended = _read_json(extended_output / "descriptors.json")
    output = results / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows = _scalar_comparison(reference, extended)
    baseline_rows = _static_baseline_comparison(reference_output, extended_output)
    arrays = _difference_arrays(reference_output, extended_output, config["comparison"])
    np.savez_compressed(output / "difference-fields.npz", **arrays)  # type: ignore[arg-type]
    figure_files = (
        _write_difference_figures(
            output,
            arrays,
            config["comparison"].get("tolerances", DEFAULT_COMPARISON["tolerances"]),
        )
        if config.get("shared", {}).get("reporting", {}).get("enabled", True)
        else []
    )
    if config.get("shared", {}).get("reporting", {}).get("enabled", True):
        figure_files.extend(_write_v2_difference_figures(output, arrays))
    changed_fractions = {
        key.removesuffix("_changed"): float(value[np.isfinite(value)].mean())
        for key, value in arrays.items()
        if key.endswith("_changed") and value.size
    }
    payload = {
        "schema": "d3map.comparison.v2",
        "software_version": __version__,
        "difference_sign": "extended-minus-reference",
        "status": "nonstandard" if mismatches else "standard",
        "compatibility_mismatches": mismatches,
        "tolerances": config["comparison"].get("tolerances", {}),
        "scalar_comparisons": rows,
        "static_baseline_comparisons": baseline_rows,
        "changed_area_fractions": changed_fractions,
        "difference_figure_files": figure_files,
        "reference_output": "../reference",
        "extended_output": "../extended",
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{row[key] if row[key] is not None else 'not available'}</td>" for key in rows[0]) + "</tr>"
        for row in rows
    )
    headings = "".join(f"<th>{key.replace('_', ' ')}</th>" for key in rows[0])
    warning = (
        f"<p><strong>Non-standard comparison:</strong> {', '.join(mismatches)}</p>"
        if mismatches
        else "<p>All standard d3-map compatibility checks passed.</p>"
    )
    (output / "report.html").write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>d3map comparison</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;color:#18202a}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #d5dbe3;padding:.45rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}th{background:#183153;color:white}</style></head>"
        "<body><h1>d3-map comparison</h1><p>Signed differences are extended minus reference. "
        "Tolerance masking applies only to visual classifications; raw arrays remain complete.</p>"
        f"{warning}<p>{len(figure_files)} tolerance-masked difference maps were generated; "
        "the complete unmasked numerical fields are stored in <code>difference-fields.npz</code>.</p>"
        f"<table><thead><tr>{headings}</tr></thead><tbody>{table_rows}</tbody></table>"
        "<p><a href='../reference/report.html'>Reference d2-map report</a> | "
        "<a href='../extended/report.html'>Extended d2-map report</a></p></body></html>",
        encoding="utf-8",
    )
    _write_manifest(output, "d3map.comparison-manifest.v1")
    return output


def _write_run_state(project: Path, state: dict[str, Any]) -> None:
    path = project.parent / "run-state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_geometry(path: Path, geometry: Any, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(geometry.elements)}\n{comment}\n")
        for element, coordinate in zip(geometry.elements, geometry.coordinates, strict=True):
            stream.write(
                f"{element:<3} {coordinate[0]: .10f} {coordinate[1]: .10f} {coordinate[2]: .10f}\n"
            )


def _generate_static_baselines(
    project_root: Path,
    release_config: dict[str, Any],
    role: str,
    legacy_project: Path,
    output: Path,
    *,
    sampling: bool,
) -> Path:
    """Generate every scientifically available labelled single-geometry baseline."""
    legacy = _read_json(legacy_project)
    stage_root = legacy_project.parent / "baselines"
    source = project_root / release_config["structures"][role]["input"]
    source_ensemble = read_xyz_ensemble(source)
    candidates: list[tuple[str, Any, str]] = [
        ("input", source_ensemble.geometries[0], "user-supplied input geometry")
    ]
    unavailable: dict[str, str] = {}
    if sampling:
        sampling_root = legacy_project.parent / "stages" / "sampling"
        preoptimized = sampling_root / "preoptimized.xyz"
        if preoptimized.is_file():
            candidates.append(
                (
                    "preoptimized",
                    read_xyz_ensemble(preoptimized).geometries[0],
                    "GFN2-xTB preoptimized geometry",
                )
            )
        else:
            unavailable["preoptimized"] = "xTB preoptimization output is unavailable"
        sampling_manifest = sampling_root / "manifest.json"
        if sampling_manifest.is_file():
            manifest = _read_json(sampling_manifest)
            if manifest.get("analysis_mode") == "ensemble":
                ranked_source = Path(manifest["source_ensemble"])
                ranked = read_xyz_ensemble(ranked_source)
                energies = read_xyz_comment_energies(ranked_source)
                index = int(np.argmin(energies))
                candidates.append(
                    (
                        "lowest-energy",
                        ranked.geometries[index],
                        "lowest electronic-energy retained CREST conformer",
                    )
                )
            else:
                unavailable["lowest-energy"] = (
                    "trajectory mode has no isolated-conformer energy ranking"
                )
        else:
            unavailable["lowest-energy"] = "sampling manifest is unavailable"
    else:
        population = legacy.get("population", {})
        energies = population.get("energies")
        if energies is not None and len(energies) == source_ensemble.size:
            candidates.append(
                (
                    "lowest-energy",
                    source_ensemble.geometries[int(np.argmin(np.asarray(energies, dtype=float)))],
                    "lowest-energy imported conformer",
                )
            )
        else:
            unavailable["preoptimized"] = "no integrated xTB preoptimization was requested"
            unavailable["lowest-energy"] = "no conformer energy ranking was supplied"
    records: dict[str, Any] = {}
    for name, geometry, definition in candidates:
        baseline_root = stage_root / name
        baseline_xyz = baseline_root / f"{name}.xyz"
        _write_geometry(baseline_xyz, geometry, definition)
        baseline_output = output / "baselines" / name
        baseline = {
            key: value
            for key, value in legacy.items()
            if key not in {"sampling", "atom_mappings"}
        }
        baseline.update(
            {
                "input": str(baseline_xyz.resolve()),
                "output": str(baseline_output.resolve()),
                "analysis_mode": "static",
                "population": {"model": "uniform"},
            }
        )
        baseline_project = baseline_root / "project.json"
        baseline_project.write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        analyze_project(baseline_project)
        descriptor = _read_json(baseline_output / "descriptors.json")
        records[name] = {
            "status": "available",
            "definition": definition,
            "geometry": str(baseline_xyz.relative_to(project_root)),
            "output": str(baseline_output.relative_to(project_root)),
            "vbur_percent": descriptor["ensemble_scalar_descriptors"]["vbur_percent"]["mean"],
            "g_percent": descriptor["ensemble_scalar_descriptors"]["g_percent"]["mean"],
        }
    for name, reason in unavailable.items():
        records.setdefault(name, {"status": "unavailable", "reason": reason})
    payload = {
        "schema": "d3map.static-baselines.v1",
        "role": role,
        "baselines": records,
    }
    target = output / "static-baselines.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def execute_release_project(path: str | Path, *, sampling: bool) -> ReleaseRunResult:
    project = Path(path).resolve()
    config = read_release_project(project)
    mismatches = validate_pair_compatibility(project.parent, config)
    roles = _structure_roles(config)
    state: dict[str, Any] = {
        "schema": "d3map.run-state.v1",
        "status": "running",
        "started_at": _utc_now(),
        "stages": {},
    }
    _write_run_state(project, state)
    outputs: dict[str, Path] = {}
    try:
        for role in roles:
            legacy_project, output = _merge_structure_config(
                project.parent, config, role, sampling=sampling
            )
            state["stages"][role] = {"status": "running", "started_at": _utc_now()}
            _write_run_state(project, state)
            if sampling:
                run_legacy_project(legacy_project)
            else:
                analyze_project(legacy_project)
            _generate_static_baselines(
                project.parent,
                config,
                role,
                legacy_project,
                output,
                sampling=sampling,
            )
            reporting = config.get("shared", {}).get("reporting", {})
            if reporting.get("enabled", True) and reporting.get("office_reports", True):
                write_structure_reports(output, role=role)
                refresh_manifest(output)
            outputs[role] = output
            state["stages"][role] = {
                "status": "success",
                "completed_at": _utc_now(),
                "output": str(output.relative_to(project.parent)),
            }
            _write_run_state(project, state)
        comparison_output = (
            compare_release_outputs(project, config, mismatches)
            if config["project_kind"] == "d3-map"
            else None
        )
        reporting = config.get("shared", {}).get("reporting", {})
        if comparison_output is not None and reporting.get("enabled", True) and reporting.get(
            "office_reports", True
        ):
            write_comparison_reports(comparison_output)
            refresh_manifest(comparison_output)
    except Exception as error:
        state["status"] = "failed"
        state["failed_at"] = _utc_now()
        state["error"] = str(error)
        _write_run_state(project, state)
        raise
    state["status"] = "success"
    state["completed_at"] = _utc_now()
    if mismatches:
        state["comparison_status"] = "nonstandard"
        state["compatibility_mismatches"] = mismatches
    _write_run_state(project, state)
    write_portable_zip(project)
    return ReleaseRunResult(project, config["project_kind"], outputs, comparison_output)


def regenerate_release_report(path: str | Path) -> ReleaseRunResult:
    project = Path(path).resolve()
    config = read_release_project(project)
    outputs: dict[str, Path] = {}
    for role in _structure_roles(config):
        legacy_project, output = _merge_structure_config(project.parent, config, role, sampling=False)
        regenerate_project_report(legacy_project)
        outputs[role] = output
    mismatches = validate_pair_compatibility(project.parent, config)
    comparison_output = (
        compare_release_outputs(project, config, mismatches)
        if config["project_kind"] == "d3-map"
        else None
    )
    reporting = config.get("shared", {}).get("reporting", {})
    if reporting.get("enabled", True) and reporting.get("office_reports", True):
        for role, output in outputs.items():
            write_structure_reports(output, role=role)
            refresh_manifest(output)
        if comparison_output is not None:
            write_comparison_reports(comparison_output)
            refresh_manifest(comparison_output)
    write_portable_zip(project)
    return ReleaseRunResult(project, config["project_kind"], outputs, comparison_output)


def write_portable_zip(path: str | Path) -> Path:
    project = Path(path).resolve()
    root = project.parent
    target = root / f"{root.name}-portable.zip"
    excluded = {target.name, "machine-preferences.json"}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(root.rglob("*")):
            if item.is_file() and item.name not in excluded:
                archive.write(item, item.relative_to(root.parent))
    return target


def default_structure_setup(
    input_xyz: str | Path,
    *,
    center_atom: int,
    reactive_direction: list[float],
    secondary_direction: list[float],
    alignment_atoms: list[int],
    charge: int = 0,
    multiplicity: int = 1,
    solvent: str | None = None,
    confirmed: bool = False,
    protected_contacts: list[list[int]] | None = None,
    frozen_atoms: list[int] | None = None,
    steric_atoms: list[int] | None = None,
    xtb_executable: str | None = None,
    crest_executable: str | None = None,
    threads: int = 1,
) -> dict[str, Any]:
    """Build the explicit per-structure section used by CLI and desktop setup."""
    return _normal_structure_config(
        Path(input_xyz).name,
        center_atom=center_atom,
        reactive_direction=reactive_direction,
        secondary_direction=secondary_direction,
        alignment_atoms=alignment_atoms,
        charge=charge,
        multiplicity=multiplicity,
        solvent=solvent,
        confirmed=confirmed,
        protected_contacts=protected_contacts,
        frozen_atoms=frozen_atoms,
        steric_atoms=steric_atoms,
        xtb_executable=xtb_executable,
        crest_executable=crest_executable,
        threads=threads,
    )
