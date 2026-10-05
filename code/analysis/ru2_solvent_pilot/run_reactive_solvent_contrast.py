#!/usr/bin/env python3
"""Sequential 1a-allenylidene/4d/OTs dielectric pilots and compatible D2/D3 maps.

Separate from the prepared 5c-4e/OMs solvent experiment. No legacy outputs
or the accepted unoptimized review structure are modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import run_solvent_contrast as engine

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OUT = BASE / "reactive-1a-solvent-contrast"
PREVIEW = ROOT / "03-chemical-validation/3dstructures/reactive-1a-4d-ots-preview"
SOURCE = PREVIEW / "assembly-unoptimized.xyz"
STATE_DATA = engine.read(PREVIEW / "assembly-connectivity.json")
META = STATE_DATA["metadata"]
STATES = engine.STATES
PROTOCOL = dict(engine.PROTOCOL, bootstrap_time_ps=.02, restraints="seven Ru-core distances, native force constant 0.25",
                velocity_policy="native fixed-random bootstrap per state; rescale saved velocities using actual native MD degrees of freedom; paired protocol, not independent replicates")
PAIRS = [(1, 120, "Ru-Ru"), (1, 122, "Ru1-S1"), (1, 123, "Ru1-S2"),
         (120, 122, "Ru2-S1"), (120, 123, "Ru2-S2"), (120, 124, "Ru2-Cl"), (1, 121, "Ru1-C_alpha")]
engine.OUT, engine.SOURCE, engine.PROTOCOL = OUT, SOURCE, PROTOCOL


def manifest():
    OUT.mkdir(exist_ok=True)
    p = OUT / "run-manifest.json"
    if p.exists():
        data = engine.read(p)
        if data["protocol"] != PROTOCOL or data["source_sha256"] != engine.digest(SOURCE):
            raise RuntimeError("Changed experiment input/protocol; create a new version")
        return data
    return dict(schema="case3d.reactive-1a.dielectric-pilot.v1", created_at=engine.now(), status="prepared",
                chemical_state="constructed substrate-1a-derived Ru-allenylidene + 4d + OTs; net neutral working singlet",
                source=str(SOURCE.relative_to(ROOT)), source_sha256=engine.digest(SOURCE),
                construction_provenance_sha256=engine.digest(PREVIEW / "provenance.json"),
                xtb_version=engine.original.xTB_version(), atom_count=241, protocol=PROTOCOL,
                states={s: dict(status="pending", epsilon=e) for s, e in STATES.items()},
                interpretation="One short explicitly biased block per dielectric. Sampled occupation, not equilibrium probabilities, reaction barriers, ee or a complete solvent model.")


def constraints():
    _, xyz, _ = engine.original.parse_xyz(SOURCE)
    rows = ["$constrain", "  force constant=0.25"]
    for a, b, _ in PAIRS:
        rows.append(f"  distance: {a}, {b}, {np.linalg.norm(np.array(xyz[a-1])-xyz[b-1]):.8f}")
    return "\n".join([*rows, "$end", ""])


def md_input(bootstrap=False):
    heavy = [i + 1 for i, a in enumerate(STATE_DATA["atoms"]) if a["element"] != "H"]
    time, dump = (.02, 2.) if bootstrap else (1., 20.)
    text = constraints() + ("$samerand\n$md\n"
           f"  time={time}\n  step=2.0\n  dump={dump}\n  sdump=100.0\n  temp=298.15\n"
           f"  nvt=true\n  restart={'false' if bootstrap else 'true'}\n"
           "  hmass=4\n  shake=2\n  sccacc=1.0\n$end\n")
    if not bootstrap:
        text += ("$metadyn\n  save=25\n  kpush=0.001\n  alp=1.0\n  ramp=0.03\n"
                 f"  atoms: {','.join(map(str,heavy))}\n$end\n")
    return text


def prepare(state):
    folder = OUT / state
    folder.mkdir(exist_ok=True)
    for name, text in [("input.xyz", SOURCE.read_text()), ("preopt.inp", constraints()),
                       ("bootstrap.inp", md_input(True)), ("metadyn.inp", md_input())]:
        engine.immutable(folder / name, text)
    return folder


def rescaled_restart(folder, elements, bootstrap_log):
    nfree = int(re.search(r"# deg\. of freedom\s*:\s*(\d+)", bootstrap_log).group(1))
    rows = (folder / "bootstrap.mdrestart").read_text().splitlines()
    values = np.array([[float(t.replace("D", "E")) for t in row.split()] for row in rows[1:]])
    if values.shape != (241, 6):
        raise ValueError("Native restart must contain exactly 241 coordinate/velocity rows")
    masses = {"H":4., "C":12.011, "N":14.007, "O":15.999, "F":18.998403163,
              "S":32.06, "Cl":35.45, "Ru":101.07}
    kinetic = .5 * np.sum(np.array([masses[e] for e in elements])[:,None] * 1822.888486 * values[:,3:]**2)
    temperature = 2 * kinetic / (nfree * 3.166811563e-6)
    values[:,3:] *= math.sqrt(298.15 / temperature)
    text = " -1.0\n" + "".join("".join(f"{v:22.14E}".replace("E","D") for v in row) + "\n" for row in values)
    return text, dict(native_degrees_of_freedom=nfree, bootstrap_final_velocity_temperature_K=temperature,
                      velocity_scale=math.sqrt(298.15 / temperature), restart_coordinates="native final 20 fs bootstrap geometry")


def run_one(state, data):
    folder = prepare(state)
    info = data["states"][state]
    if info["status"] == "success":
        assert engine.digest(folder / "solvent-md.xtb.trj") == info["trajectory_sha256"]
        return
    elements, _, _ = engine.original.parse_xyz(SOURCE)
    eps = str(STATES[state])
    common = [str(engine.XTB), "input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--cosmo", eps, "--parallel", "4"]
    info.update(status="running", started_at=engine.now(), input_sha256=engine.digest(folder / "input.xyz"))
    data["status"] = "running"
    engine.write(OUT / "run-manifest.json", data)
    log = engine.command(folder, "define", common + ["--metadyn", "25", "--input", "metadyn.inp", "--define"])
    if "not recognized" in log.lower():
        raise RuntimeError("Unsupported input option; no production run permitted")
    engine.command(folder, "singlepoint", common + ["--sp"])
    engine.command(folder, "preoptimization", common + ["--opt", "normal", "--input", "preopt.inp"])
    optimized = folder / "xtbopt.xyz"
    if engine.original.parse_xyz(optimized)[0] != elements:
        raise RuntimeError("Optimization changed atom identities/order")
    engine.original.copy_input(optimized, folder / "preoptimized.xyz")
    base = [str(engine.XTB), "preoptimized.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--cosmo", eps, "--parallel", "4"]
    log = engine.command(folder, "bootstrap", base + ["--md", "--input", "bootstrap.inp", "--namespace", "bootstrap"])
    if "normal exit of md()" not in log:
        raise RuntimeError("20 fs bootstrap did not complete")
    text, velocity_info = rescaled_restart(folder, elements, log)
    engine.immutable(folder / "initial-mdrestart", text)
    if not (folder / "metadynamics.complete.json").exists():
        engine.original.copy_input(folder / "initial-mdrestart", folder / "solvent-md.mdrestart")
    info.update(velocity_initialization=velocity_info, preoptimized_sha256=engine.digest(optimized))
    args = base + ["--metadyn", "25", "--input", "metadyn.inp", "--namespace", "solvent-md"]
    info["metadynamics_command"] = args
    engine.write(OUT / "run-manifest.json", data)
    log = engine.command(folder, "metadynamics", args)
    if not re.search(r"kpush\s*:\s*0\.001", log) or "adding snapshot to metadynamics bias" not in log:
        raise RuntimeError("Explicit nonzero RMSD bias/deposition not confirmed")
    if "RESTART" not in log or "normal exit of md()" not in log:
        raise RuntimeError("Native restart or complete MD missing")
    summary = engine.original.validate_trajectory(folder / "solvent-md.xtb.trj", elements, PAIRS)
    # Native 6.7.1 writes before integration at k=1,11,...,491: t=0..0.98 ps.
    # The final 1.00 ps geometry is in mdrestart, not an extra trajectory frame.
    if summary["frame_count"] != 50:
        raise RuntimeError(f"Expected 50 native frames for 500 steps: {summary['frame_count']}")
    temp = float(re.search(r"^\s*T\s*:\s*([\d.Ee+-]+)", log, re.M).group(1))
    warnings = []
    if abs(temp / 298.15 - 1) > .02:
        warnings.append("Average temperature differs from target by >2%")
    if "thermostating problem" in log:
        warnings.append("Native thermostating problem warning")
    info.update(status="success", completed_at=engine.now(), trajectory_sha256=engine.digest(folder / "solvent-md.xtb.trj"),
                trajectory_validation=summary, average_temperature_K=temp,
                saved_frame_times_ps=[i*.02 for i in range(50)],
                diagnostic_note="Corrected initial 51-frame assumption against native 6.7.1 dump loop: 50 pre-integration frames, 0.00..0.98 ps; no dynamics rerun or settings changed",
                bias_deposition_count=log.count("adding snapshot to metadynamics bias"),
                scientific_qc_status="exploratory_with_warnings" if warnings else "exploratory_single_short_biased_block",
                sampling_warnings=warnings)
    engine.write(folder / "trajectory-summary.json", info)
    engine.write(OUT / "run-manifest.json", data)


def build_maps(data):
    common = engine.read_xyz_ensemble(SOURCE).geometries[0]
    template = ROOT / "03-chemical-validation/part3/3d/stage-a/published_5c_4e_I/alpb-ccl4/project.d3map.json"
    config = engine.read_release_project(template)
    shared = copy.deepcopy(config["shared"])
    shared["reporting"]["office_reports"] = False
    masks = {"full": [i + 1 for i in range(241) if i + 1 != 186],
             "external": [i + 1 for i, a in enumerate(STATE_DATA["atoms"]) if a["component"] != "1a_ligand"]}
    projects = {}
    for state in STATES:
        raw = OUT / state / "solvent-md.xtb.trj"
        assert engine.digest(raw) == data["states"][state]["trajectory_sha256"]
        ensemble = engine.read_xyz_ensemble(raw)
        keep = [i for i in range(ensemble.size) if i * .02 >= .2 - 1e-12]
        prepared = OUT / state / "map-input.xyz"
        text = ""
        for k, geom in enumerate([common] + [ensemble.geometries[i] for i in keep]):
            label = "common zero-weight spatial anchor" if k == 0 else f"{state} native frame {keep[k-1]+1} at {keep[k-1]*.02:.2f} ps"
            text += str(len(geom.elements)) + "\n" + label + "\n" + "".join(
                f"{e:3s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n" for e,p in zip(geom.elements, geom.coordinates))
        engine.immutable(prepared, text)
        for mask, ids in masks.items():
            setup = copy.deepcopy(config["structures"]["reference"])
            for key in ["input", "sampling", "case3d"]:
                setup.pop(key, None)
            setup.update(analysis_mode="trajectory", population=dict(model="explicit", weights=[0.] + [1.] * len(keep)),
                         steric_atoms=ids, solvent=None, topology=dict(policy="strict", minimum_retained_population=.9),
                         origin_atoms=[186], alignment_atoms=[1,120,122,123,124,121,185,186],
                         reactive_direction=(common.coordinates[185] - common.coordinates[184]).tolist(),
                         secondary_direction=(common.coordinates[119] - common.coordinates[0]).tolist(),
                         charge=0, multiplicity=1, chemical_setup_confirmed=True,
                         chemical_setup_status="constructed 1a-allenylidene/4d/OTs; working net-neutral singlet; seven-distance restrained Ru core; explicitly biased pilot",
                         case3d=dict(state=state, epsilon=STATES[state], mask=mask,
                                     chemical_state="1a-derived Ru-allenylidene + 4d + OTs",
                                     sampling="uniform saved-frame visitation after 0.2 ps discard; RMSD-biased, not equilibrium",
                                     frame_anchor="identical unoptimized preview geometry with exactly zero statistical weight",
                                     external_mask="Ru/Cp*/SMe/Cl + 4d + OTs only; excludes the entire 1a-derived organic ligand"))
            project = OUT / "maps/d2" / f"{state}-{mask}" / "project.d3map.json"
            if not project.exists():
                engine.create_release_project(kind="d2-map", reference_xyz=prepared, output_directory=project.parent,
                                              reference_setup=setup, shared=shared)
            statefile = project.parent / "run-state.json"
            if not statefile.exists() or engine.read(statefile).get("status") != "success":
                print(f"Starting D2: {state}, {mask}", flush=True)
                engine.execute_release_project(project, sampling=False)
            projects[(state, mask)] = project
    engine.cached_maps.MAPS = OUT / "maps"
    differences = {}
    for mask in masks:
        project = engine.cached_maps.build_d3(f"dcm-minus-ccl4-{mask}", projects[("ccl4-like", mask)], projects[("dcm-like", mask)],
                 f"DCM-like minus CCl4-like; same 1a-allenylidene/4d/OTs chemical identity; mask={mask}; explicit RMSD-biased visitation; no TS or equilibrium interpretation")
        # The generic cache helper predates this explicitly nonzero-bias experiment.
        p = project.parent / "run-state.json"
        info = engine.read(p)
        info["sampling_strategy"] = "cached uniformly weighted explicitly RMSD-biased 1a/4d/OTs frames; kpush=0.001; no map-time resampling; not equilibrium"
        info["schema"] = "case3d.reactive-1a.dielectric-d3.v1"
        engine.write(p, info)
        engine.cached_maps.write_portable_zip(project)
        differences[mask] = str(project.relative_to(OUT))
    engine.write(OUT / "maps/map-index.json", dict(status="success", d2_projects={f"{s}/{m}":str(p.relative_to(OUT)) for (s,m),p in projects.items()},
                 d3_projects=differences, completed_at=engine.now()))
    data["status"] = "maps_complete"
    engine.write(OUT / "run-manifest.json", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--state", choices=list(STATES), help="run one state only; no maps until both complete")
    args = parser.parse_args()
    data = manifest()
    try:
        if args.prepare_only:
            for state in STATES:
                prepare(state)
        else:
            if not args.maps_only:
                for state in ([args.state] if args.state else STATES):
                    run_one(state, data)
            if not args.state:
                if any(info["status"] != "success" for info in data["states"].values()):
                    raise RuntimeError("Both trajectories must complete before mapping")
                build_maps(data)
    except (Exception, KeyboardInterrupt) as err:
        data.update(status="interrupted" if isinstance(err, KeyboardInterrupt) else "failed", error=str(err), updated_at=engine.now())
        engine.write(OUT / "run-manifest.json", data)
        raise
    data["updated_at"] = engine.now()
    engine.write(OUT / "run-manifest.json", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
