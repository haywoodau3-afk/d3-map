from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .errors import DMapError, ValidationError


@dataclass(frozen=True)
class RefinementSelection:
    selected_indices: tuple[int, ...]
    relative_energies_kcal_mol: NDArray[np.float64]
    window_kcal_mol: float


def select_refinement_candidates(
    energies: NDArray[np.float64], *, unit: str = "hartree", window_kcal_mol: float = 6.0
) -> RefinementSelection:
    values = np.asarray(energies, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValidationError("refinement energies must be a non-empty finite vector")
    if window_kcal_mol <= 0:
        raise ValidationError("refinement window must be positive")
    factors = {"hartree": 627.509474, "kcal/mol": 1.0, "kj/mol": 0.239005736}
    if unit not in factors:
        raise ValidationError("energy unit must be hartree, kcal/mol, or kj/mol")
    relative = (values - values.min()) * factors[unit]
    selected = tuple(int(index) for index in np.flatnonzero(relative <= window_kcal_mol + 1e-12))
    return RefinementSelection(selected, relative, window_kcal_mol)


def read_energy_table(
    path: str | Path, *, expected_count: int | None = None
) -> NDArray[np.float64]:
    """Read a CSV energy column for user-supplied or higher-level refinement results."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    try:
        values = np.asarray([float(row["energy"]) for row in rows], dtype=float)
    except (KeyError, ValueError) as error:
        raise ValidationError("energy table must contain a numeric 'energy' column") from error
    if expected_count is not None and values.shape != (expected_count,):
        raise ValidationError("energy table row count does not match the ensemble")
    if not np.all(np.isfinite(values)):
        raise ValidationError("energy table contains non-finite values")
    return values


def run_external_refinement(
    command: tuple[str, ...], *, working_directory: str | Path, log_name: str = "refinement.log"
) -> Path:
    """Run an explicitly configured higher-level backend without shell interpolation."""
    if not command:
        raise ValidationError("external refinement command cannot be empty")
    root = Path(working_directory)
    root.mkdir(parents=True, exist_ok=True)
    log = root / log_name
    completed = subprocess.run(
        list(command),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise DMapError(
            f"external refinement failed with exit code {completed.returncode}; see {log}"
        )
    return log
