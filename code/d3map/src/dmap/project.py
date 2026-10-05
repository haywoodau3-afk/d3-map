from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .alignment import align_ensemble
from .angular import ShieldingResult, analyze_shielding
from .axial import analyze_axial
from .chemistry import perceive_coordination, screen_topology
from .convergence import occupation_grid_convergence, probability_field_grid_convergence
from .errors import ValidationError
from .features import (
    PocketFieldResult,
    analyze_pocket_field,
    angular_fingerprint,
    ensemble_feature_set,
)
from .inorganic import (
    coordination_distribution,
    coordination_sector_shielding,
    coupled_ligand_motion,
    ligand_resolved_fields,
)
from .io import read_xyz_ensemble, remap_ensemble
from .models import AxialGrid, AxialResult, CatalyticFrame, FloatArray
from .populations import boltzmann_weights
from .probability_v2 import (
    ProbabilityFieldV2Result,
    analyze_probability_field_v2,
    analyze_probe_accessibility_v2,
    write_probability_v2_figures,
)
from .radii import radii_for_elements
from .reporting import generate_plot_bundle, regenerate_topographic_outputs
from .static import (
    PROJECTION_DIRECTIONS,
    calculate_buried_volume,
    calculate_displaced_steric_scan,
)


def _one_based_indices(values: list[int], atom_count: int, label: str) -> tuple[int, ...]:
    if any(not isinstance(value, int) or value < 1 or value > atom_count for value in values):
        raise ValidationError(f"{label} must contain one-based atom numbers")
    if len(set(values)) != len(values):
        raise ValidationError(f"{label} cannot contain duplicates")
    return tuple(value - 1 for value in values)


def _read_project(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(
            "legacy project.yml must use the JSON-compatible YAML subset in d3map"
        ) from error
    if not isinstance(data, dict) or data.get("schema_version") != "0.1":
        raise ValidationError("project schema_version must be 0.1")
    return data


def _population_weights(population: dict[str, Any], ensemble_size: int) -> FloatArray:
    model = population.get("model", "uniform")
    if model == "uniform":
        return np.full(ensemble_size, 1.0 / ensemble_size)
    if model == "explicit":
        values = np.asarray(population.get("weights", []), dtype=float)
        if values.shape != (ensemble_size,) or np.any(values < 0) or values.sum() <= 0:
            raise ValidationError("explicit population requires one non-negative weight per frame")
        return np.asarray(values / values.sum(), dtype=np.float64)
    if model in ("free_energy", "electronic_energy"):
        return boltzmann_weights(
            np.asarray(population.get("energies", []), dtype=float),
            temperature=float(population["temperature"]),
            energy_kind=model,
            unit=population.get("unit", "kcal/mol"),
            degeneracies=(
                None
                if "degeneracies" not in population
                else np.asarray(population["degeneracies"], dtype=float)
            ),
        ).weights
    raise ValidationError(f"unsupported population model: {model}")


def _relative_population_energies_kcal(
    population: dict[str, Any], valid_indices: np.ndarray
) -> FloatArray | None:
    if population.get("model") not in {"free_energy", "electronic_energy"}:
        return None
    values = np.asarray(population.get("energies", []), dtype=float)[valid_indices]
    factors = {"kcal/mol": 1.0, "kJ/mol": 1.0 / 4.184, "hartree": 627.5094740631}
    unit = population.get("unit", "kcal/mol")
    if unit not in factors:
        raise ValidationError(f"unsupported energy unit: {unit}")
    converted = values * factors[unit]
    return np.asarray(converted - converted.min(), dtype=float)


def _weighted_quantile(values: FloatArray, weights: FloatArray, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = min(len(cumulative[cumulative < probability]), len(values) - 1)
    return float(sorted_values[index])


def _weighted_summary(
    values: list[float], weights: FloatArray, lowest_energy_index: int | None
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    mean = float(np.dot(weights, data))
    standard_deviation = (
        None if len(data) == 1 else float(np.sqrt(np.dot(weights, (data - mean) ** 2)))
    )
    percentile_5 = _weighted_quantile(data, weights, 0.05)
    percentile_95 = _weighted_quantile(data, weights, 0.95)
    return {
        "values": data.tolist(),
        "mean": mean,
        "population_standard_deviation": standard_deviation,
        "variability_status": "not_estimable" if len(data) == 1 else "estimated",
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "percentile_span": percentile_95 - percentile_5,
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "lowest_energy_value": (
            None if lowest_energy_index is None else float(data[lowest_energy_index])
        ),
    }


def _write_artifacts(
    output: Path,
    result: AxialResult,
    shielding: ShieldingResult,
    pocket: PocketFieldResult,
    features: dict[str, Any],
    ensemble: Any,
    metadata: dict[str, Any],
    probability_v2: ProbabilityFieldV2Result,
) -> None:
    if output.exists() and not output.is_dir():
        raise ValidationError(f"output location is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "axial-fields.npz",
        x=result.x,
        y=result.y,
        z=result.z,
        occupation=result.occupation,
        entropy=result.entropy,
        angular_directions=shielding.directions,
        shielding_probability=shielding.shielding_probability,
        shielding_per_geometry=shielding.per_geometry_shielding,
        pocket_axis=pocket.axis,
        pocket_occupation=pocket.occupation_probability,
        pocket_entropy=pocket.entropy,
        pocket_domain=pocket.domain_mask,
        pocket_per_geometry_occupation=pocket.per_geometry_occupation,
    )
    np.savez_compressed(
        output / "probability-fields-v2.npz",
        schema=np.asarray("d3map.fields.v2"),
        axis=pocket.axis,
        domain=pocket.domain_mask,
        occupation=pocket.occupation_probability,
        entropy=pocket.entropy,
        **probability_v2.arrays,
    )
    summaries = [asdict(summary) for summary in result.summaries]
    descriptors = {**metadata, "axial_summaries": summaries}
    (output / "descriptors.json").write_text(
        json.dumps(descriptors, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "axial-profile.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    flat_features: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                flat_features[f"{key}_{index + 1}"] = item
        else:
            flat_features[key] = value
    with (output / "features.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_features))
        writer.writeheader()
        writer.writerow(flat_features)
    with (output / "aligned-ensemble.xyz").open("w", encoding="utf-8") as stream:
        for index, geometry in enumerate(ensemble.geometries):
            stream.write(f"{len(geometry.elements)}\naligned d3map frame {index + 1}\n")
            for element, coordinate in zip(geometry.elements, geometry.coordinates, strict=True):
                stream.write(
                    f"{element:<3} {coordinate[0]: .10f} {coordinate[1]: .10f} {coordinate[2]: .10f}\n"
                )
    report_data = _report_data(
        output,
        x=result.x,
        y=result.y,
        z=result.z,
        occupation=result.occupation,
        entropy=result.entropy,
    )
    (output / "report.html").write_text(_interactive_report(report_data), encoding="utf-8")
    (output / "axial-profile.svg").write_text(_profile_svg(result), encoding="utf-8")
    _refresh_manifest(output, metadata["software_version"], metadata["input_sha256"])


def _refresh_manifest(output: Path, software_version: str, input_sha256: str) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema": "d3map.stage-manifest.v1",
        "status": "success",
        "software_version": software_version,
        "input_sha256": input_sha256,
        "artifacts": {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _report_data(
    output: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    occupation: np.ndarray,
    entropy: np.ndarray,
) -> str:
    payload: dict[str, Any] = {
        "axial": {
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
            "occupation": np.nan_to_num(occupation, nan=-1.0).tolist(),
            "entropy": np.nan_to_num(entropy, nan=-1.0).tolist(),
        }
    }
    topographic_path = output / "plot-data" / "topographic-fields.npz"
    if topographic_path.is_file():
        with np.load(topographic_path, allow_pickle=False) as stored:
            if "schema" in stored.files and str(stored["schema"]) in {
                "dmap.topographic-fields.v0.3",
                "d3map.topographic-fields.v1",
            }:
                payload["topographic"] = {
                    "axis": stored["axis"].tolist(),
                    "directions": stored["direction_names"].tolist(),
                    "offsets": stored["offsets"].tolist(),
                    "horizontal": stored["plane_horizontal_axes"].tolist(),
                    "vertical": stored["plane_vertical_axes"].tolist(),
                    "sphereRadius": float(stored["sphere_radius"]),
                    "layers": {
                        "contact_probability": np.nan_to_num(
                            stored["contact_probability"], nan=-999.0
                        ).tolist(),
                        "first_contact_median": np.nan_to_num(
                            stored["first_contact_q50"], nan=-999.0
                        ).tolist(),
                        "first_contact_interval_10_90": np.nan_to_num(
                            stored["first_contact_interval_10_90"], nan=-999.0
                        ).tolist(),
                        "occupied_depth": np.nan_to_num(
                            stored["occupied_depth"], nan=-999.0
                        ).tolist(),
                    },
                }
    return json.dumps(payload, separators=(",", ":"))


def _profile_svg(result: AxialResult) -> str:
    width, height, pad = 760, 420, 50
    z_min, z_max = float(result.z.min()), float(result.z.max())
    span = max(z_max - z_min, 1.0)
    colours = {
        "open_fraction": "#2ca25f",
        "breathing_fraction": "#fec44f",
        "blocked_fraction": "#de2d26",
    }
    lines = []
    for key, colour in colours.items():
        points = []
        for summary in result.summaries:
            px = pad + (summary.z - z_min) / span * (width - 2 * pad)
            py = height - pad - getattr(summary, key) * (height - 2 * pad)
            points.append(f"{px:.2f},{py:.2f}")
        lines.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="3" points="{" ".join(points)}"/>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><path d="M {pad} {pad} V {height - pad} H {width - pad}" fill="none" stroke="#222"/>{"".join(lines)}<text x="{width / 2}" y="{height - 8}" text-anchor="middle">position along reactive +z / Å</text><text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">area fraction</text></svg>'''


def _interactive_report(report_data: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>d3map ensemble analysis</title>
<style>body{{font:16px system-ui;max-width:1180px;margin:2rem auto;color:#18202a}}
canvas{{border:1px solid #aab2bd;background:#eee;max-width:100%;height:auto}}
.row{{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin:.8rem 0}}
label{{font-weight:600}}select,input{{font:inherit}}.panel{{padding:1rem 1.2rem;border:1px solid #d8dde5;border-radius:10px;margin:1.2rem 0}}
.legend{{height:14px;width:560px;max-width:100%;background:linear-gradient(90deg,#313695,#ffffbf,#a50026);border:1px solid #888}}
.muted{{color:#596575}}output{{font-variant-numeric:tabular-nums}}</style></head>
<body><h1>d3map ensemble analysis</h1>
<section class="panel"><h2>Direction- and distance-resolved topographic diagnostic</h2>
<p>Contact frequency, conditional first-contact geometry, conformational spread, and occupied ray depth are shown as separate quantities. Views look from the named signed direction toward the catalytic origin.</p>
<div id="topographic-controls" class="row"><label for="direction">direction</label><select id="direction"></select>
<label for="offset">outward distance</label><input id="offset" type="range" min="0" value="0"><output id="offset-value"></output>
<label for="topographic-layer">layer</label><select id="topographic-layer">
<option value="contact_probability">contact probability</option><option value="first_contact_median">conditional median first contact</option>
<option value="first_contact_interval_10_90">conditional 10–90% interval</option><option value="occupied_depth">population-weighted occupied depth</option>
</select></div><canvas id="topographic-map" width="640" height="640"></canvas><div id="topographic-legend" class="legend"></div>
<p id="topographic-scale" class="muted"></p><p id="topographic-axes" class="muted"></p>
<p><a href="plot-data/topographic-analysis.md">Open the descriptive topographic analysis</a></p></section>
<section class="panel"><h2>d2-map axial occupation field</h2><p>Occupation probability along the confirmed reactive axis.</p>
<div class="row"><label for="axial-layer">layer</label><select id="axial-layer"><option value="occupation">occupation</option><option value="entropy">entropy</option><option value="persistence">persistence</option></select>
<label for="slice">z slice</label><input id="slice" type="range" min="0"><output id="z"></output></div>
<canvas id="axial-map" width="560" height="560"></canvas></section>
<script>const d={report_data},a=d.axial;
function clamp(v){{return Math.max(0,Math.min(1,v))}}function mix(c1,c2,t){{return c1.map((v,i)=>Math.round(v+(c2[i]-v)*t))}}
function colour(q,layer){{q=clamp(q);let c;if(layer==='first_contact_median'){{c=q<.5?mix([49,54,149],[255,255,191],q*2):mix([255,255,191],[165,0,38],(q-.5)*2)}}else if(layer==='contact_probability'){{c=mix([68,1,84],[253,231,37],q)}}else if(layer==='occupied_depth'){{c=mix([0,0,4],[252,253,191],q)}}else{{c=mix([0,34,78],[255,255,204],q)}}return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`}}
function paint(canvas,values,layer,min,max){{const ctx=canvas.getContext('2d'),w=values.length,h=values[0].length;ctx.clearRect(0,0,canvas.width,canvas.height);for(let i=0;i<w;i++)for(let j=0;j<h;j++){{const p=values[i][j];ctx.fillStyle=p<=-998?'#eee':colour((p-min)/(max-min),layer);ctx.fillRect(i*canvas.width/w,(h-1-j)*canvas.height/h,canvas.width/w+1,canvas.height/h+1)}}}}
const slider=document.querySelector('#slice'),out=document.querySelector('#z'),axialLayer=document.querySelector('#axial-layer'),axialCanvas=document.querySelector('#axial-map');slider.max=a.z.length-1;
function drawAxial(){{const k=Number(slider.value),name=axialLayer.value,values=name==='entropy'?a.entropy[k]:a.occupation[k];out.value=`${{a.z[k].toFixed(2)}} Å`;let shown=values.map(row=>row.map(p=>p<0?-999:p));if(name==='persistence')shown=values.map(row=>row.map(p=>p<0?-999:(p<=.1?0:(p>=.9?1:.5))));paint(axialCanvas,shown,name,0,name==='entropy'?.693147:1)}}
slider.addEventListener('input',drawAxial);axialLayer.addEventListener('change',drawAxial);drawAxial();
const t=d.topographic,section=document.querySelector('#topographic-controls');if(!t){{section.parentElement.innerHTML='<h2>Topographic diagnostic unavailable</h2><p>Regenerate this project with the current reporter.</p>'}}else{{
const direction=document.querySelector('#direction'),offset=document.querySelector('#offset'),offsetValue=document.querySelector('#offset-value'),topLayer=document.querySelector('#topographic-layer'),topCanvas=document.querySelector('#topographic-map'),scale=document.querySelector('#topographic-scale'),axes=document.querySelector('#topographic-axes'),legend=document.querySelector('#topographic-legend');
t.directions.forEach((name,index)=>direction.add(new Option(name,index)));offset.max=t.offsets.length-1;
function drawTopographic(){{const di=Number(direction.value),oi=Number(offset.value),layer=topLayer.value,R=t.sphereRadius;let min=0,max=1;if(layer==='first_contact_median'){{min=-R;max=R;legend.style.background='linear-gradient(90deg,#313695,#ffffbf,#a50026)'}}else if(layer==='contact_probability'){{legend.style.background='linear-gradient(90deg,#440154,#fde725)'}}else{{max=2*R;legend.style.background=layer==='occupied_depth'?'linear-gradient(90deg,#000004,#fcfdbf)':'linear-gradient(90deg,#00224e,#ffffcc)'}}paint(topCanvas,t.layers[layer][di][oi],layer,min,max);offsetValue.value=`${{t.offsets[oi].toFixed(2)}} Å`;scale.textContent=`Fixed scale: ${{min.toFixed(1)}} to ${{max.toFixed(1)}}${{layer==='contact_probability'?'':' Å'}}`;axes.textContent=`Horizontal: catalytic ${{t.horizontal[di]}}; vertical: catalytic ${{t.vertical[di]}}; view from ${{t.directions[di]}} toward origin.`}}
[direction,offset,topLayer].forEach(control=>control.addEventListener('input',drawTopographic));drawTopographic()}}</script></body></html>"""


def regenerate_project_report(path: str | Path) -> Path:
    """Regenerate visual reporting exclusively from persisted analysis artifacts."""
    project_path = Path(path).resolve()
    config = _read_project(project_path)
    output = project_path.parent / config.get("output", "dmap-results")
    if not output.is_dir():
        raise ValidationError(f"analysis output directory not found: {output}")
    axial_path = output / "axial-fields.npz"
    descriptors_path = output / "descriptors.json"
    if not axial_path.is_file() or not descriptors_path.is_file():
        raise ValidationError("report regeneration requires existing axial fields and descriptors")
    regenerate_topographic_outputs(output)
    with np.load(axial_path, allow_pickle=False) as axial:
        report_data = _report_data(
            output,
            x=axial["x"],
            y=axial["y"],
            z=axial["z"],
            occupation=axial["occupation"],
            entropy=axial["entropy"],
        )
    report_path = output / "report.html"
    report_path.write_text(_interactive_report(report_data), encoding="utf-8")
    descriptors = json.loads(descriptors_path.read_text(encoding="utf-8"))
    _refresh_manifest(output, descriptors["software_version"], descriptors["input_sha256"])
    return report_path


def analyze_project(path: str | Path) -> AxialResult:
    """Run the external-ensemble d2-map workflow from a persisted project."""
    project_path = Path(path).resolve()
    config = _read_project(project_path)
    root = project_path.parent
    input_path = root / config["input"]
    ensemble = read_xyz_ensemble(input_path)
    if "atom_mappings" in config:
        ensemble = remap_ensemble(ensemble, config["atom_mappings"])
    original_size = ensemble.size
    atom_count = len(ensemble.elements)
    center_atom = int(config["center_atom"])
    center_index = _one_based_indices([center_atom], atom_count, "center_atom")[0]
    origin_indices = _one_based_indices(
        list(config.get("origin_atoms", [center_atom])), atom_count, "origin_atoms"
    )
    if len(origin_indices) not in {1, 2}:
        raise ValidationError("origin_atoms must contain one atom or a two-centre midpoint")
    alignment_values = list(config.get("alignment_atoms", []))
    if not alignment_values:
        raise ValidationError("alignment_atoms must identify an invariant catalytic core")
    alignment_indices = _one_based_indices(alignment_values, atom_count, "alignment_atoms")

    topology = config.get("topology", {"policy": "strict"})
    screened = screen_topology(
        ensemble,
        center_index=center_index,
        policy=topology.get("policy", "strict"),
        allowed_codes=tuple(topology.get("allowed_codes", [])),
    )
    population = config.get("population", {"model": "uniform"})
    analysis_mode = config.get("analysis_mode", "static" if original_size == 1 else "ensemble")
    if analysis_mode not in {"static", "ensemble", "trajectory"}:
        raise ValidationError("analysis_mode must be static, ensemble, or trajectory")
    if analysis_mode == "static" and original_size != 1:
        raise ValidationError("static analysis requires exactly one geometry")
    if analysis_mode == "trajectory" and population.get("model") in {
        "free_energy",
        "electronic_energy",
    }:
        raise ValidationError("trajectory mode cannot use isolated-conformer Boltzmann weights")
    original_weights = _population_weights(population, original_size)
    valid_indices = np.asarray(screened.valid_indices, dtype=int)
    weights = original_weights[valid_indices]
    retained_population = float(weights.sum())
    minimum_retained = float(topology.get("minimum_retained_population", 0.0))
    if retained_population < minimum_retained:
        raise ValidationError(
            f"topology quarantine retained population {retained_population:.6f}, "
            f"below required {minimum_retained:.6f}"
        )
    weights = weights / weights.sum()
    ensemble = screened.ensemble
    alignment_metadata: dict[str, Any]
    if ensemble.size > 1:
        aligned = align_ensemble(
            ensemble,
            atom_indices=alignment_indices,
            center_index=center_index if len(origin_indices) == 1 else None,
        )
        ensemble = aligned.ensemble
        alignment_metadata = {
            "atom_numbers": [index + 1 for index in alignment_indices],
            "reference_frame": 1,
            "diagnostics": [
                {
                    "rmsd": item.rmsd,
                    "determinant": item.determinant,
                    "rotation": item.rotation.tolist(),
                    "translation": item.translation.tolist(),
                }
                for item in aligned.diagnostics
            ],
        }
    else:
        alignment_metadata = {
            "atom_numbers": [index + 1 for index in alignment_indices],
            "reference_frame": 1,
            "diagnostics": [],
        }

    steric_values = config.get("steric_atoms")
    hydrogen_policy = str(config.get("hydrogen_policy", "included"))
    if hydrogen_policy not in {"included", "excluded"}:
        raise ValidationError("hydrogen_policy must be included or excluded")
    if steric_values is None:
        steric_indices = tuple(
            index
            for index, element in enumerate(ensemble.elements)
            if index not in origin_indices and (hydrogen_policy == "included" or element != "H")
        )
    else:
        selected = _one_based_indices(list(steric_values), atom_count, "steric_atoms")
        if any(index in selected for index in origin_indices):
            raise ValidationError("a catalytic origin atom cannot be in the steric atom set")
        steric_indices = selected
    frozen_atoms = list(config.get("frozen_atoms", []))
    _one_based_indices(frozen_atoms, atom_count, "frozen_atoms")

    reference = ensemble.geometries[0]
    coordination = perceive_coordination(reference, center_index)
    if "reactive_direction" in config:
        reactive_direction = np.asarray(config["reactive_direction"], dtype=float)
        direction_source = "user_confirmed"
    else:
        reactive_direction = coordination.vacancy_directions[0]
        direction_source = "coordination_ranked_proposal"
    if "secondary_direction" in config:
        secondary_direction = np.asarray(config["secondary_direction"], dtype=float)
    elif coordination.donor_indices:
        secondary_direction = (
            reference.coordinates[coordination.donor_indices[0]]
            - reference.coordinates[center_index]
        )
    else:
        secondary_direction = np.array([1.0, 0.0, 0.0])
    frame = CatalyticFrame.from_directions(
        origin=reference.coordinates[np.asarray(origin_indices, dtype=int)].mean(axis=0),
        z_direction=reactive_direction,
        x_hint=secondary_direction,
    )
    axial_values = config.get("axial", {})
    grid = AxialGrid(
        transverse_radius=float(axial_values.get("transverse_radius", 3.5)),
        z_min=float(axial_values.get("z_min", 0.0)),
        z_max=float(axial_values.get("z_max", 6.0)),
        spacing=float(axial_values.get("spacing", 0.1)),
    )
    energy_values = population.get("energies")
    lowest_energy_index = (
        None
        if energy_values is None
        else int(np.argmin(np.asarray(energy_values, dtype=float)[valid_indices]))
    )
    radii = radii_for_elements(
        [ensemble.elements[index] for index in steric_indices],
        config.get("radii_overrides"),
        profile=config.get("field_radii_profile", "dmap_vdw_v0.1"),
    )
    vbur_radii = radii_for_elements(
        [ensemble.elements[index] for index in steric_indices],
        config.get("vbur_radii_overrides"),
        profile=config.get("vbur_radii_profile", "sambvca_2.1"),
    )
    result = analyze_axial(
        ensemble,
        frame=frame,
        grid=grid,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
    )
    static_values = config.get("static", {})
    sphere_radius = float(static_values.get("sphere_radius", 3.5))
    static_spacing = float(static_values.get("spacing", 0.1))
    direction_count = int(static_values.get("direction_count", 40_962))
    vbur_values: list[float] = []
    shielding = analyze_shielding(
        ensemble,
        frame=frame,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
        direction_count=direction_count,
    )
    g_values = shielding.per_geometry_g_percent.tolist()
    for geometry in ensemble.geometries:
        vbur_values.append(
            calculate_buried_volume(
                geometry,
                frame=frame,
                steric_atom_indices=steric_indices,
                radii=vbur_radii,
                sphere_radius=sphere_radius,
                spacing=static_spacing,
            ).percent_buried
        )
    field_values = config.get("field", {})
    pocket = analyze_pocket_field(
        ensemble,
        frame=frame,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
        sphere_radius=sphere_radius,
        spacing=float(field_values.get("spacing", max(static_spacing, 0.15))),
    )
    energy_values_kcal = _relative_population_energies_kcal(population, valid_indices)
    replicate_blocks = config.get("replicate_blocks")
    retained_blocks = None
    if replicate_blocks is not None:
        if not isinstance(replicate_blocks, list) or len(replicate_blocks) != original_size:
            raise ValidationError("replicate_blocks must provide one label per input geometry")
        retained_blocks = tuple(str(replicate_blocks[index]) for index in valid_indices)
    probability_v2 = analyze_probability_field_v2(
        pocket,
        weights=weights,
        energies_kcal=energy_values_kcal,
        block_labels=retained_blocks,
    )
    baseline_index = lowest_energy_index if lowest_energy_index is not None else 0
    baseline_kind = "lowest-energy-retained" if lowest_energy_index is not None else "reference-frame"
    baseline_occupation = pocket.per_geometry_occupation[baseline_index].astype(float)
    baseline_occupation[~pocket.domain_mask] = np.nan
    probability_v2.arrays["baseline_occupation"] = baseline_occupation
    probability_v2.arrays["ensemble_residual_occupation"] = (
        pocket.occupation_probability - baseline_occupation
    )
    probability_v2.features["ensemble_residual"] = {
        "baseline": baseline_kind,
        "baseline_frame_number": baseline_index + 1,
        "status": "primary" if lowest_energy_index is not None else "diagnostic",
    }
    shielding_probability = shielding.shielding_probability
    shielding_entropy = np.zeros_like(shielding_probability)
    mixed_shielding = (shielding_probability > 0.0) & (shielding_probability < 1.0)
    shielding_entropy[mixed_shielding] = -(
        shielding_probability[mixed_shielding] * np.log(shielding_probability[mixed_shielding])
        + (1.0 - shielding_probability[mixed_shielding])
        * np.log(1.0 - shielding_probability[mixed_shielding])
    )
    baseline_shielding = shielding.per_geometry_shielding[baseline_index].astype(float)
    probability_v2.arrays.update(
        {
            "shielding_directions": shielding.directions,
            "shielding_probability": shielding_probability,
            "shielding_entropy": shielding_entropy,
            "shielding_per_geometry": shielding.per_geometry_shielding,
            "baseline_shielding": baseline_shielding,
            "ensemble_residual_shielding": shielding_probability - baseline_shielding,
        }
    )
    shielding_classes = {
        "persistent_open_fraction": float(np.mean(shielding_probability <= 0.1)),
        "adaptive_fraction": float(
            np.mean((shielding_probability > 0.1) & (shielding_probability < 0.9))
        ),
        "persistent_shielded_fraction": float(np.mean(shielding_probability >= 0.9)),
    }
    probability_v2.features["shielding"] = {
        **shielding_classes,
        "persistent_open_solid_angle_steradian": 4.0
        * float(np.pi)
        * shielding_classes["persistent_open_fraction"],
        "adaptive_solid_angle_steradian": 4.0
        * float(np.pi)
        * shielding_classes["adaptive_fraction"],
        "persistent_shielded_solid_angle_steradian": 4.0
        * float(np.pi)
        * shielding_classes["persistent_shielded_fraction"],
        "mean_binary_entropy_nat": (
            None if ensemble.size == 1 else float(np.mean(shielding_entropy))
        ),
    }
    if energy_values_kcal is not None:
        centered_energy = energy_values_kcal - float(np.dot(weights, energy_values_kcal))
        shielding_energy_covariance = np.tensordot(
            weights * centered_energy,
            shielding.per_geometry_shielding.astype(float),
            axes=(0, 0),
        )
        probability_v2.arrays["shielding_energy_covariance_kcal_per_mol"] = (
            shielding_energy_covariance
        )
    transformed = tuple(frame.transform(geometry.coordinates) for geometry in ensemble.geometries)
    accessibility_config = config.get("accessibility", {})
    probe_accessibility = analyze_probe_accessibility_v2(
        axis=pocket.axis,
        domain=pocket.domain_mask,
        frame_coordinates=transformed,
        weights=weights,
        steric_atom_indices=steric_indices,
        radii=radii,
        elements=ensemble.elements,
        probe_radii=tuple(
            accessibility_config.get(
                "probe_radii", [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            )
        ),
        target_distances=tuple(
            accessibility_config.get("target_distances", [0.5, 1.0, 1.5, 2.0])
        ),
    )
    probability_v2.features["probe_accessibility"] = probe_accessibility.features
    probability_v2.arrays.update(probe_accessibility.arrays)
    probability_v2.features.update(
        {
            "analysis_mode": analysis_mode,
            "source_probability_semantics": population.get("model", "uniform"),
        }
    )
    features = ensemble_feature_set(
        pocket,
        weights=weights,
        frame_coordinates=transformed,
        steric_atom_indices=steric_indices,
        radii=radii,
        elements=ensemble.elements,
        probe_radius=float(accessibility_config.get("probe_radius", 1.2)),
    )
    features["probability_field_v2"] = probability_v2.features
    features["angular_fingerprint_12"] = angular_fingerprint(
        shielding.directions, shielding.shielding_probability
    )
    features["coordination_sectors"] = coordination_sector_shielding(
        shielding.directions, shielding.shielding_probability
    )
    coordination_weights, coordination_environments = coordination_distribution(
        ensemble, center_index, weights
    )
    features["coordination_distribution"] = coordination_weights
    ligand_groups: dict[str, tuple[int, ...]] = {}
    for label, atom_numbers in config.get("ligand_groups", {}).items():
        ligand_groups[str(label)] = _one_based_indices(
            list(atom_numbers), atom_count, f"ligand_groups.{label}"
        )
    if ligand_groups:
        resolved = ligand_resolved_fields(
            ensemble,
            frame=frame,
            weights=weights,
            ligand_atom_indices=ligand_groups,
            radii=radii,
            sphere_radius=sphere_radius,
            spacing=float(field_values.get("spacing", max(static_spacing, 0.15))),
        )
        features["ligand_resolved_vbur_percent"] = {
            label: field.weighted_buried_percent for label, field in resolved.items()
        }
    if len(ligand_groups) >= 2:
        labels, correlation = coupled_ligand_motion(transformed, ligand_groups, weights)
        features["coupled_motion_labels"] = list(labels)
        features["coupled_motion_correlation"] = correlation.tolist()
    scan_values = config.get("displaced_scan", {})
    offsets = np.arange(
        float(scan_values.get("z_min", 0.0)),
        float(scan_values.get("z_max", 2.0)) + 0.5 * float(scan_values.get("spacing", 0.1)),
        float(scan_values.get("spacing", 0.1)),
    )
    scans = [
        calculate_displaced_steric_scan(
            geometry,
            frame=frame,
            offsets=offsets.tolist(),
            steric_atom_indices=steric_indices,
            radii=vbur_radii,
            g_radii=radii,
            sphere_radius=sphere_radius,
            spacing=float(scan_values.get("sphere_spacing", static_spacing)),
            direction_count=int(scan_values.get("direction_count", direction_count)),
        )
        for geometry in ensemble.geometries
    ]
    displaced_vbur = np.stack([scan.vbur_percent for scan in scans])
    displaced_g = np.stack([scan.g_percent for scan in scans])
    features["displaced_vbur_mean"] = (weights @ displaced_vbur).tolist()
    features["displaced_g_mean"] = (weights @ displaced_g).tolist()
    convergence: list[dict[str, Any]] | None = None
    probability_convergence: list[dict[str, Any]] | None = None
    if "convergence" in config:
        convergence_points = occupation_grid_convergence(
            ensemble,
            frame=frame,
            weights=weights,
            steric_atom_indices=steric_indices,
            radii=radii,
            sphere_radius=sphere_radius,
            spacings=tuple(
                float(value) for value in config["convergence"].get("spacings", [0.3, 0.2, 0.15])
            ),
        )
        convergence = [asdict(point) for point in convergence_points]
        probability_convergence = [
            asdict(point)
            for point in probability_field_grid_convergence(
                ensemble,
                frame=frame,
                weights=weights,
                steric_atom_indices=steric_indices,
                radii=radii,
                sphere_radius=sphere_radius,
                spacings=tuple(
                    float(value)
                    for value in config["convergence"].get("spacings", [0.3, 0.2, 0.15])
                ),
            )
        ]
    metadata = {
        "schema_version": "0.1",
        "software_version": __version__,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "analysis_mode": analysis_mode,
        "center_atom": center_atom,
        "origin_atoms": [index + 1 for index in origin_indices],
        "frozen_atoms": frozen_atoms,
        "steric_atoms": [index + 1 for index in steric_indices],
        "hydrogen_policy": hydrogen_policy,
        "population_model": population.get("model", "uniform"),
        "weights": weights.tolist(),
        "effective_ensemble_size": float(1.0 / np.sum(weights * weights)),
        "temperature": population.get("temperature", config.get("temperature")),
        "population_accounting": {
            "valid": float(original_weights[valid_indices].sum()),
            "quarantined": float(original_weights[list(screened.quarantined_indices)].sum())
            if screened.quarantined_indices
            else 0.0,
            "missing": 0.0,
        },
        "weighted_quantile_convention": "inverse_cdf_left_v0.1",
        "reactive_direction": reactive_direction.tolist(),
        "secondary_direction": secondary_direction.tolist(),
        "catalytic_frame": {
            "origin": frame.origin.tolist(),
            "basis": frame.basis.tolist(),
            "positive_z_semantics": direction_source,
        },
        "coordination": {
            "coordination_number": coordination.coordination_number,
            "donor_atom_numbers": [index + 1 for index in coordination.donor_indices],
            "geometry": coordination.geometry,
            "angle_rmsd_degrees": coordination.angle_rmsd_degrees,
            "vacancy_directions": coordination.vacancy_directions.tolist(),
            "vacancy_scores_degrees": coordination.vacancy_scores_degrees.tolist(),
            "ensemble_distribution": coordination_weights,
            "per_frame": [
                {
                    "coordination_number": item.coordination_number,
                    "geometry": item.geometry,
                    "angle_rmsd_degrees": item.angle_rmsd_degrees,
                }
                for item in coordination_environments
            ],
        },
        "topology_screening": {
            "policy": topology.get("policy", "strict"),
            "valid_frame_numbers": [index + 1 for index in screened.valid_indices],
            "quarantined_frame_numbers": [index + 1 for index in screened.quarantined_indices],
            "findings": [asdict(finding) for finding in screened.findings],
        },
        "alignment": alignment_metadata,
        "axial_grid": asdict(grid),
        "static_profile": {
            "sphere_radius": sphere_radius,
            "spacing": static_spacing,
            "direction_count": direction_count,
            "vbur_radii_profile": config.get("vbur_radii_profile", "sambvca_2.1"),
            "g_radii_profile": config.get("field_radii_profile", "dmap_vdw_v0.1"),
        },
        "topographic_projection": {
            "directions": list(PROJECTION_DIRECTIONS),
            "direction_vectors": [
                frame.basis[:, 2].tolist(),
                (-frame.basis[:, 2]).tolist(),
                frame.basis[:, 0].tolist(),
                (-frame.basis[:, 0]).tolist(),
                frame.basis[:, 1].tolist(),
                (-frame.basis[:, 1]).tolist(),
            ],
            "offsets_are_positive_distance_along_each_direction": True,
        },
        "field_profile": {
            "radii_profile": config.get("field_radii_profile", "dmap_vdw_v0.1"),
            "hydrogen_policy": hydrogen_policy,
            "smoothing": "none",
        },
        "ensemble_scalar_descriptors": {
            "vbur_percent": _weighted_summary(vbur_values, weights, lowest_energy_index),
            "g_percent": _weighted_summary(g_values, weights, lowest_energy_index),
        },
        "canonical_field_identity": {
            "weighted_vbur_percent": pocket.weighted_buried_percent,
            "integrated_occupation_percent": pocket.integrated_buried_percent,
            "absolute_error_percent": abs(
                pocket.weighted_buried_percent - pocket.integrated_buried_percent
            ),
        },
        "displaced_scan": {
            "semantic": "translated_origin_vbur_and_g_not_axial_plane_occupation",
            "offsets_angstrom": offsets.tolist(),
            "vbur_percent_mean": (weights @ displaced_vbur).tolist(),
            "g_percent_mean": (weights @ displaced_g).tolist(),
        },
        "features": features,
        "grid_convergence": convergence,
        "probability_field_v2_convergence": probability_convergence,
    }
    output = root / config.get("output", "dmap-results")
    reporting = config.get("reporting", {})
    # Persist the numerical v2 summary even when expensive figure/report
    # rendering is disabled.  The summary is a scientific data product, not
    # merely a presentation artifact; leaving an older uniform/trajectory
    # summary beside newly energy-weighted fields creates a misleading mixed
    # state after a repair or batch reanalysis.
    output.mkdir(parents=True, exist_ok=True)
    (output / "probability-v2-summary.json").write_text(
        json.dumps(probability_v2.features, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if reporting.get("enabled", True):
        generate_plot_bundle(
            output,
            result=result,
            shielding=shielding,
            pocket=pocket,
            metadata=metadata,
            ensemble=ensemble,
            frame=frame,
            weights=weights,
            steric_atom_indices=steric_indices,
            topographic_radii=vbur_radii,
            displaced_offsets=offsets,
            displaced_vbur=displaced_vbur,
            displaced_g=displaced_g,
            sphere_radius=sphere_radius,
            topographic_spacing=float(reporting.get("topographic_spacing", 0.1)),
            reporting=reporting,
        )
        write_probability_v2_figures(output, pocket, probability_v2)
    _write_artifacts(
        output,
        result,
        shielding,
        pocket,
        features,
        ensemble,
        metadata,
        probability_v2,
    )
    return result
