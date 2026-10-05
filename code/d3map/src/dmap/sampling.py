from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import DMapError, ValidationError
from .io import read_xyz_ensemble


@dataclass(frozen=True)
class XtbCrestConfig:
    charge: int = 0
    multiplicity: int = 1
    frozen_atom_numbers: tuple[int, ...] = ()
    protected_contacts: tuple[tuple[int, int], ...] = ()
    solvent: str | None = None
    threads: int = 1
    quick: bool = False
    maximum_reduced: bool = False
    constrain_preoptimization: bool = True
    allow_trajectory_fallback: bool = False
    metadyn_steps: int = 25
    metadyn_time_ps: float = 5.0
    crest_mdlen_ps: float | None = None

    def __post_init__(self) -> None:
        if self.multiplicity < 1:
            raise ValidationError("multiplicity must be positive")
        if self.threads < 1:
            raise ValidationError("threads must be positive")
        if self.metadyn_steps < 1:
            raise ValidationError("metadyn_steps must be positive")
        if self.metadyn_time_ps <= 0.0:
            raise ValidationError("metadyn_time_ps must be positive")
        if self.crest_mdlen_ps is not None and self.crest_mdlen_ps <= 0.0:
            raise ValidationError("crest_mdlen_ps must be positive when provided")
        if any(number < 1 for number in self.frozen_atom_numbers):
            raise ValidationError("frozen atom numbers use positive one-based indexing")
        if len(set(self.frozen_atom_numbers)) != len(self.frozen_atom_numbers):
            raise ValidationError("frozen atom numbers must be distinct")
        if any(a < 1 or b < 1 or a == b for a, b in self.protected_contacts):
            raise ValidationError("protected contacts require distinct positive atom numbers")


@dataclass(frozen=True)
class XtbCrestResult:
    preoptimized_xyz: Path
    ensemble_xyz: Path
    electronic_energies_hartree: np.ndarray
    xtb_log: Path
    crest_log: Path
    analysis_mode: str = "ensemble"
    backend_versions: dict[str, str] | None = None
    block_labels: tuple[str, ...] | None = None


def _resolve_executable(value: str | Path, label: str) -> str:
    candidate = str(value)
    resolved = shutil.which(candidate) if os.sep not in candidate else candidate
    if resolved is None or not Path(resolved).is_file():
        raise DMapError(f"{label} executable not found: {candidate}")
    return resolved


def _write_protected_core(
    path: Path,
    atom_numbers: tuple[int, ...],
    contacts: tuple[tuple[int, int], ...],
    coordinates: np.ndarray,
) -> None:
    if not atom_numbers and not contacts:
        path.write_text("$end\n", encoding="utf-8")
        return
    rows = ["$constrain", "  force constant=0.25"]
    if atom_numbers:
        rows.append(f"  atoms: {','.join(str(number) for number in atom_numbers)}")
    for first, second in contacts:
        distance = float(np.linalg.norm(coordinates[first - 1] - coordinates[second - 1]))
        rows.append(f"  distance: {first}, {second}, {distance:.8f}")
    rows.append("$end")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def executable_version(executable: str | Path) -> str:
    resolved = _resolve_executable(executable, str(executable))
    completed = subprocess.run(
        [resolved, "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return next(
        (line for line in lines if "version" in line.lower()), lines[0] if lines else "unknown"
    )


def _run(command: list[str], *, cwd: Path, log: Path, environment: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise DMapError(
            f"external calculation failed with exit code {completed.returncode}; see {log}"
        )


def read_xyz_comment_energies(path: str | Path) -> np.ndarray:
    """Read one Hartree energy from each XYZ comment in a CREST-style ensemble.

    CREST conformer files conventionally put the energy first (``-123.4``),
    whereas xTB trajectory comments use the explicit form
    ``energy: -123.4 gnorm: ...``.  Both are valid electronic-energy
    annotations and must be treated identically so trajectory-derived
    ensembles are not silently downgraded to uniform populations.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cursor = 0
    energies: list[float] = []
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor == len(lines):
            break
        try:
            atom_count = int(lines[cursor])
            comment = lines[cursor + 1]
            match = re.search(
                r"(?:^|\s)(?:energy\s*:\s*)?"
                r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
                r"(?:\s|$)",
                comment,
            )
            if match is None:
                raise ValueError("no numeric electronic energy in XYZ comment")
            energies.append(float(match.group(1)))
        except (IndexError, ValueError) as error:
            raise ValidationError(
                "CREST ensemble lacks a numeric energy in an XYZ comment"
            ) from error
        cursor += atom_count + 2
    values = np.asarray(energies, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValidationError("CREST ensemble contains no finite electronic energies")
    return values


def read_crest_conformer_origins(
    path: str | Path, expected_count: int
) -> tuple[str, ...] | None:
    """Read final CREGEN origin labels for retained conformers when available.

    CREST 2.x prints one eight-column row for each unique conformer.  Earlier
    iteration tables may also occur in the log, so the final ``expected_count``
    rows are the provenance labels corresponding to ``crest_conformers.xyz``.
    A missing or changed table format is treated as unavailable provenance,
    not as a failed electronic-structure calculation.
    """
    if expected_count < 1:
        raise ValidationError("expected_count must be positive")
    rows: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if (
            len(tokens) == 8
            and tokens[0].isdigit()
            and tokens[5].isdigit()
            and tokens[6].isdigit()
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]*", tokens[7])
        ):
            try:
                float(tokens[1])
                float(tokens[2])
                float(tokens[3])
                float(tokens[4])
            except ValueError:
                continue
            rows.append(tokens[7])
    if len(rows) < expected_count:
        return None
    return tuple(rows[-expected_count:])


def _trajectory_energies(path: Path, frame_count: int) -> np.ndarray:
    """Return finite trajectory energies when xTB annotated every frame.

    Older fallback trajectories may contain only a descriptive comment.  Those
    remain valid geometric trajectories, but are explicitly marked as
    energy-unavailable rather than being assigned fabricated populations.
    """
    try:
        values = read_xyz_comment_energies(path)
    except ValidationError:
        return np.full(frame_count, np.nan, dtype=float)
    if values.shape != (frame_count,) or not np.all(np.isfinite(values)):
        return np.full(frame_count, np.nan, dtype=float)
    return values


def run_xtb_crest(
    input_xyz: str | Path,
    output_directory: str | Path,
    *,
    config: XtbCrestConfig,
    xtb_executable: str | Path = "xtb",
    crest_executable: str | Path = "crest",
) -> XtbCrestResult:
    """Pre-optimize one XYZ with GFN2-xTB, then run a protected CREST search."""
    source = Path(input_xyz).resolve()
    ensemble = read_xyz_ensemble(source)
    if ensemble.size != 1:
        raise ValidationError("xTB/CREST sampling requires a single starting XYZ geometry")
    atom_count = len(ensemble.elements)
    protected_numbers = tuple(number for pair in config.protected_contacts for number in pair)
    if any(number > atom_count for number in config.frozen_atom_numbers + protected_numbers):
        raise ValidationError("a frozen atom number is outside the input geometry")

    xtb = _resolve_executable(xtb_executable, "xTB")
    crest = _resolve_executable(crest_executable, "CREST")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    starting_xyz = output / "input.xyz"
    shutil.copyfile(source, starting_xyz)
    constraints = output / "constraints.inp"
    _write_protected_core(
        constraints,
        config.frozen_atom_numbers,
        config.protected_contacts,
        ensemble.geometries[0].coordinates,
    )

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(config.threads)
    environment["MKL_NUM_THREADS"] = str(config.threads)
    environment["PATH"] = f"{Path(xtb).parent}{os.pathsep}{environment.get('PATH', '')}"
    uhf = config.multiplicity - 1
    common = ["--chrg", str(config.charge), "--uhf", str(uhf)]
    solvent = [] if config.solvent is None else ["--alpb", config.solvent]

    xtb_log = output / "xtb-preopt.log"
    # On a repair pass, a prior CREST attempt may already have produced both
    # the validated preoptimized geometry and a finite-energy trajectory.  Do
    # not launch another expensive preoptimization: the raw trajectory is the
    # immutable sampling artifact and its energies are re-read below.
    preoptimized = output / "preoptimized.xyz"
    fallback = output / "crest_dynamics.trj"
    if config.allow_trajectory_fallback and preoptimized.is_file() and fallback.is_file():
        try:
            sampled = read_xyz_ensemble(fallback)
        except ValidationError:
            sampled = None
        if sampled is not None:
            return XtbCrestResult(
                preoptimized,
                fallback,
                _trajectory_energies(fallback, sampled.size),
                xtb_log,
                output / "crest.log",
                "trajectory",
                {"xtb": executable_version(xtb), "crest": executable_version(crest)},
            )
    preoptimization_constraints = (
        ["--input", constraints.name] if config.constrain_preoptimization else []
    )
    _run(
        [
            xtb,
            starting_xyz.name,
            "--gfn",
            "2",
            "--opt",
            "tight",
            *common,
            *solvent,
            *preoptimization_constraints,
            "--parallel",
            str(config.threads),
        ],
        cwd=output,
        log=xtb_log,
        environment=environment,
    )
    generated_preoptimized = output / "xtbopt.xyz"
    if not generated_preoptimized.is_file():
        raise DMapError(f"xTB completed without writing {generated_preoptimized}")
    preoptimized = output / "preoptimized.xyz"
    shutil.copyfile(generated_preoptimized, preoptimized)
    if not config.constrain_preoptimization:
        relaxed = read_xyz_ensemble(preoptimized)
        _write_protected_core(
            constraints,
            config.frozen_atom_numbers,
            config.protected_contacts,
            relaxed.geometries[0].coordinates,
        )

    # A prior CREST attempt may have produced a valid trajectory even when it
    # did not produce a conformer ensemble.  Reuse that artifact on resume so
    # a repair pass does not launch the expensive CREST search again.
    fallback = output / "crest_dynamics.trj"
    if config.allow_trajectory_fallback and fallback.is_file():
        try:
            sampled = read_xyz_ensemble(fallback)
        except ValidationError:
            pass
        else:
            return XtbCrestResult(
                preoptimized,
                fallback,
                _trajectory_energies(fallback, sampled.size),
                xtb_log,
                output / "crest.log",
                "trajectory",
                {"xtb": executable_version(xtb), "crest": executable_version(crest)},
            )

    crest_command = [
        crest,
        preoptimized.name,
        "--gfn2",
        *common,
        *solvent,
        "-T",
        str(config.threads),
        "-xnam",
        xtb,
    ]
    if config.frozen_atom_numbers or config.protected_contacts:
        crest_command.extend(["--cinp", constraints.name])
    if config.quick:
        crest_command.append("--mquick" if config.maximum_reduced else "--quick")
    if config.crest_mdlen_ps is not None:
        crest_command.extend(["--mdlen", str(config.crest_mdlen_ps)])

    crest_log = output / "crest.log"
    try:
        _run(crest_command, cwd=output, log=crest_log, environment=environment)
        ensemble_candidates = (
            output / "crest_conformers.xyz",
            output / "crest_ensemble.xyz",
        )
        if not any(
            candidate.is_file() and candidate.stat().st_size > 0
            for candidate in ensemble_candidates
        ):
            raise DMapError(
                "CREST completed without writing a recognized conformer ensemble"
            )
    except DMapError as crest_error:
        fallback = output / "crest_dynamics.trj"
        if config.allow_trajectory_fallback and not fallback.is_file():
            metadyn_log = output / "xtb-metadyn-fallback.log"
            metadyn_input = output / "xtb-metadyn.inp"
            metadyn_input.write_text(
                constraints.read_text(encoding="utf-8")
                + "\n$md\n"
                + f"  time={config.metadyn_time_ps:.3f}\n"
                + "  dump=50.0\n"
                + "  step=4.0\n"
                + "  hmass=4\n"
                + "$end\n",
                encoding="utf-8",
            )
            metadyn_command = [
                xtb,
                preoptimized.name,
                "--gfn",
                "2",
                "--metadyn",
                str(config.metadyn_steps),
                *common,
                *solvent,
                "--input",
                metadyn_input.name,
                "--parallel",
                str(config.threads),
                "--namespace",
                "dmap-md",
            ]
            try:
                _run(
                    metadyn_command,
                    cwd=output,
                    log=metadyn_log,
                    environment=environment,
                )
            except DMapError as metadyn_error:
                raise DMapError(
                    f"CREST failed ({crest_error}); xTB metadynamics fallback failed "
                    f"({metadyn_error})"
                ) from metadyn_error
            trajectory_candidates = sorted(
                candidate
                for candidate in output.glob("*.trj*")
                if candidate.is_file() and candidate.name != fallback.name
            )
            for candidate in trajectory_candidates:
                try:
                    read_xyz_ensemble(candidate)
                except ValidationError:
                    continue
                shutil.copyfile(candidate, fallback)
                break
        if not config.allow_trajectory_fallback or not fallback.is_file():
            raise
        sampled = read_xyz_ensemble(fallback)
        return XtbCrestResult(
            preoptimized,
            fallback,
            _trajectory_energies(fallback, sampled.size),
            xtb_log,
            crest_log,
            "trajectory",
            {"xtb": executable_version(xtb), "crest": executable_version(crest)},
        )
    ensemble_candidates = (
        output / "crest_conformers.xyz",
        # CREST 3.0.2 may use its native output name even when the wrapper
        # requested the standard conformer search path.
        output / "crest_ensemble.xyz",
    )
    ensemble_xyz = next(
        (candidate for candidate in ensemble_candidates if candidate.is_file() and candidate.stat().st_size > 0),
        None,
    )
    if ensemble_xyz is None:
        fallback = output / "crest_dynamics.trj"
        if config.allow_trajectory_fallback and fallback.is_file():
            try:
                sampled = read_xyz_ensemble(fallback)
            except ValidationError:
                sampled = None
            if sampled is not None:
                return XtbCrestResult(
                    preoptimized,
                    fallback,
                    _trajectory_energies(fallback, sampled.size),
                    xtb_log,
                    crest_log,
                    "trajectory",
                    {"xtb": executable_version(xtb), "crest": executable_version(crest)},
                )
        raise DMapError(
            "CREST completed without writing a recognized conformer ensemble: "
            + ", ".join(str(candidate) for candidate in ensemble_candidates)
        )
    sampled = read_xyz_ensemble(ensemble_xyz)
    energies = read_xyz_comment_energies(ensemble_xyz)
    if sampled.size != len(energies):
        raise ValidationError("CREST geometry and energy counts differ")
    block_labels = read_crest_conformer_origins(crest_log, sampled.size)
    return XtbCrestResult(
        preoptimized,
        ensemble_xyz,
        energies,
        xtb_log,
        crest_log,
        "ensemble",
        {"xtb": executable_version(xtb), "crest": executable_version(crest)},
        block_labels,
    )
