from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._version import __version__
from .errors import DMapError
from .io import read_xyz_ensemble
from .project import analyze_project, regenerate_project_report
from .release import (
    create_release_project,
    default_structure_setup,
    execute_release_project,
    is_release_project,
    migrate_v03_project,
    regenerate_release_report,
)
from .sampling import executable_version
from .ui import write_setup_viewer
from .workflow import run_project


def _atom_numbers(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("atom numbers must be comma-separated integers") from error


def _parser(prog: str = "d3map") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    new = subparsers.add_parser("new", help="create a portable d2-map or d3-map project")
    new.add_argument("--mode", choices=("d2-map", "d3-map"))
    new.add_argument("--reference")
    new.add_argument("--extended")
    new.add_argument("--output")
    new.add_argument("--center", type=int)
    new.add_argument("--extended-center", type=int)
    new.add_argument("--reactive-direction", type=float, nargs=3)
    new.add_argument("--extended-reactive-direction", type=float, nargs=3)
    new.add_argument("--secondary-direction", type=float, nargs=3)
    new.add_argument("--extended-secondary-direction", type=float, nargs=3)
    new.add_argument("--alignment-atoms", type=_atom_numbers)
    new.add_argument("--extended-alignment-atoms", type=_atom_numbers)
    new.add_argument("--freeze-atoms", type=_atom_numbers, default=[])
    new.add_argument("--extended-freeze-atoms", type=_atom_numbers)
    new.add_argument("--steric-atoms", type=_atom_numbers)
    new.add_argument("--extended-steric-atoms", type=_atom_numbers)
    new.add_argument("--charge", type=int, default=0)
    new.add_argument("--extended-charge", type=int)
    new.add_argument("--multiplicity", type=int, default=1)
    new.add_argument("--extended-multiplicity", type=int)
    new.add_argument("--solvent")
    new.add_argument("--extended-solvent")
    new.add_argument("--protect", action="append", default=[], metavar="A-B")
    new.add_argument("--extended-protect", action="append", metavar="A-B")
    new.add_argument("--xtb")
    new.add_argument("--crest")
    new.add_argument("--threads", type=int, default=1)
    new.add_argument("--confirm-chemical-setup", action="store_true")
    prepare = subparsers.add_parser("prepare", help="write a reproducible project configuration")
    prepare.add_argument("xyz")
    prepare.add_argument("--center", type=int, required=True)
    prepare.add_argument("--reactive-direction", type=float, nargs=3, required=True)
    prepare.add_argument("--secondary-direction", type=float, nargs=3, required=True)
    prepare.add_argument("--alignment-atoms", type=_atom_numbers, required=True)
    prepare.add_argument("--freeze-atoms", type=_atom_numbers, default=[])
    prepare.add_argument("--steric-atoms", type=_atom_numbers)
    prepare.add_argument("--charge", type=int, default=0)
    prepare.add_argument("--multiplicity", type=int, default=1)
    prepare.add_argument("--solvent")
    prepare.add_argument("--temperature", type=float, default=298.15)
    prepare.add_argument("--output", default="project.yml")
    analyze = subparsers.add_parser("analyze", help="analyze an external ensemble project")
    analyze.add_argument("project")
    run = subparsers.add_parser("run", help="run configured sampling and downstream analysis")
    run.add_argument("project")
    report = subparsers.add_parser("report", help="regenerate plots from stored analysis artifacts")
    report.add_argument("project")
    batch = subparsers.add_parser("batch", help="run projects listed one-per-line")
    batch.add_argument("dataset")
    migrate = subparsers.add_parser("migrate", help="migrate a v0.3 project to d3map 1.0")
    migrate.add_argument("project")
    migrate.add_argument("--output")
    subparsers.add_parser("gui", help="open the d3map desktop application")
    subparsers.add_parser("doctor", help="report optional computational backends")
    return parser


def _contacts(values: list[str] | None) -> list[list[int]]:
    contacts: list[list[int]] = []
    for value in values or []:
        try:
            first, second = (int(item) for item in value.split("-", 1))
        except ValueError as error:
            raise DMapError("protected contacts must use A-B one-based atom numbers") from error
        contacts.append([first, second])
    return contacts


def _prompt(value: str | None, label: str) -> str:
    return value if value is not None else input(f"{label}: ").strip()


def _new(arguments: argparse.Namespace) -> Path:
    mode = _prompt(arguments.mode, "Mode (d2-map or d3-map)")
    if mode not in {"d2-map", "d3-map"}:
        raise DMapError("mode must be d2-map or d3-map")
    reference = _prompt(arguments.reference, "Reference XYZ")
    extended = arguments.extended
    if mode == "d3-map":
        extended = _prompt(extended, "Extended XYZ")
    required = {
        "center": arguments.center,
        "reactive-direction": arguments.reactive_direction,
        "secondary-direction": arguments.secondary_direction,
        "alignment-atoms": arguments.alignment_atoms,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise DMapError("new requires explicit " + ", ".join(missing))
    reference_setup = default_structure_setup(
        reference,
        center_atom=arguments.center,
        reactive_direction=list(arguments.reactive_direction),
        secondary_direction=list(arguments.secondary_direction),
        alignment_atoms=arguments.alignment_atoms,
        charge=arguments.charge,
        multiplicity=arguments.multiplicity,
        solvent=arguments.solvent,
        confirmed=arguments.confirm_chemical_setup,
        protected_contacts=_contacts(arguments.protect),
        frozen_atoms=arguments.freeze_atoms,
        steric_atoms=arguments.steric_atoms,
        xtb_executable=arguments.xtb,
        crest_executable=arguments.crest,
        threads=arguments.threads,
    )
    extended_setup = None
    if mode == "d3-map":
        extended_setup = default_structure_setup(
            extended,
            center_atom=arguments.extended_center or arguments.center,
            reactive_direction=list(
                arguments.extended_reactive_direction or arguments.reactive_direction
            ),
            secondary_direction=list(
                arguments.extended_secondary_direction or arguments.secondary_direction
            ),
            alignment_atoms=arguments.extended_alignment_atoms or arguments.alignment_atoms,
            charge=(
                arguments.charge
                if arguments.extended_charge is None
                else arguments.extended_charge
            ),
            multiplicity=(
                arguments.multiplicity
                if arguments.extended_multiplicity is None
                else arguments.extended_multiplicity
            ),
            solvent=(
                arguments.solvent
                if arguments.extended_solvent is None
                else arguments.extended_solvent
            ),
            confirmed=arguments.confirm_chemical_setup,
            protected_contacts=_contacts(arguments.extended_protect or arguments.protect),
            frozen_atoms=arguments.extended_freeze_atoms or arguments.freeze_atoms,
            steric_atoms=arguments.extended_steric_atoms or arguments.steric_atoms,
            xtb_executable=arguments.xtb,
            crest_executable=arguments.crest,
            threads=arguments.threads,
        )
    return create_release_project(
        kind=mode,
        reference_xyz=reference,
        extended_xyz=extended,
        output_directory=arguments.output,
        reference_setup=reference_setup,
        extended_setup=extended_setup,
    )


def _doctor() -> None:
    print(f"numpy: {np.__version__}")
    for executable in ("xtb", "crest"):
        location = shutil.which(executable)
        version = executable_version(location).splitlines()[0] if location else None
        print(
            f"{executable}: {location or 'not found (optional for external-ensemble analysis)'}{f' [{version}]' if version else ''}"
        )


def _prepare(arguments: argparse.Namespace) -> None:
    xyz = Path(arguments.xyz).resolve()
    ensemble = read_xyz_ensemble(xyz)
    elements = ensemble.elements
    selected = {
        "center": [arguments.center],
        "alignment": arguments.alignment_atoms,
        "frozen": arguments.freeze_atoms,
    }
    if arguments.steric_atoms is not None:
        selected["steric"] = arguments.steric_atoms
    for label, atom_numbers in selected.items():
        if any(number < 1 or number > len(elements) for number in atom_numbers):
            raise DMapError(f"{label} atom number is out of range")
        if any(elements[number - 1] == "H" for number in atom_numbers):
            raise DMapError(f"hydrogen atoms cannot be selected as {label} atoms")

    project = Path(arguments.output).resolve()
    project.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "0.1",
        "input": os.path.relpath(xyz, project.parent),
        "center_atom": arguments.center,
        "reactive_direction": arguments.reactive_direction,
        "secondary_direction": arguments.secondary_direction,
        "alignment_atoms": arguments.alignment_atoms,
        "frozen_atoms": arguments.freeze_atoms,
        "charge": arguments.charge,
        "multiplicity": arguments.multiplicity,
        "solvent": arguments.solvent,
        "temperature": arguments.temperature,
        "population": {"model": "uniform"},
        "axial": {
            "transverse_radius": 3.5,
            "z_min": 0.0,
            "z_max": 6.0,
            "spacing": 0.1,
        },
        "output": "dmap-results",
    }
    if arguments.steric_atoms is not None:
        config["steric_atoms"] = arguments.steric_atoms
    project.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_setup_viewer(xyz, project.with_suffix(".setup.html"))


def main(argv: Sequence[str] | None = None, *, prog: str = "d3map") -> int:
    arguments = _parser(prog).parse_args(argv)
    try:
        if arguments.command == "new":
            project = _new(arguments)
            print(project)
        elif arguments.command == "prepare":
            _prepare(arguments)
        elif arguments.command == "doctor":
            _doctor()
        elif arguments.command == "analyze":
            if is_release_project(arguments.project):
                execute_release_project(arguments.project, sampling=False)
            else:
                analyze_project(arguments.project)
        elif arguments.command == "report":
            if is_release_project(arguments.project):
                regenerate_release_report(arguments.project)
            else:
                regenerate_project_report(arguments.project)
        elif arguments.command == "run":
            if is_release_project(arguments.project):
                execute_release_project(arguments.project, sampling=True)
            else:
                run_project(arguments.project)
        elif arguments.command == "batch":
            dataset = Path(arguments.dataset).resolve()
            for line in dataset.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    item = dataset.parent / line.strip()
                    if is_release_project(item):
                        execute_release_project(item, sampling=True)
                    else:
                        run_project(item)
        elif arguments.command == "migrate":
            print(migrate_v03_project(arguments.project, arguments.output))
        elif arguments.command == "gui":
            from .app import main as app_main

            return app_main()
    except (DMapError, KeyError, TypeError, ValueError) as error:
        print(f"{prog}: {error}", file=sys.stderr)
        return 2
    return 0


def legacy_main(argv: Sequence[str] | None = None) -> int:
    print(
        "dmap is deprecated and will be removed in d3map 2.0; use d3map instead",
        file=sys.stderr,
    )
    return main(argv, prog="dmap")
