from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import ValidationError
from .models import Ensemble, Geometry


def read_xyz_ensemble(path: str | Path) -> Ensemble:
    """Read one or more concatenated XYZ geometries without changing atom order."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValidationError(f"cannot read XYZ file: {source}") from error

    cursor = 0
    geometries: list[Geometry] = []
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor == len(lines):
            break
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as error:
            raise ValidationError(f"expected XYZ atom count on line {cursor + 1}") from error
        if atom_count <= 0:
            raise ValidationError("XYZ atom count must be positive")
        if cursor + atom_count + 1 >= len(lines):
            raise ValidationError("XYZ frame is truncated")
        cursor += 2
        elements: list[str] = []
        coordinates: list[list[float]] = []
        for _ in range(atom_count):
            fields = lines[cursor].split()
            if len(fields) < 4:
                raise ValidationError(f"XYZ atom line {cursor + 1} has fewer than four fields")
            try:
                coordinate = [float(value) for value in fields[1:4]]
            except ValueError as error:
                raise ValidationError(f"invalid coordinate on XYZ line {cursor + 1}") from error
            elements.append(fields[0])
            coordinates.append(coordinate)
            cursor += 1
        geometries.append(Geometry(tuple(elements), np.asarray(coordinates, dtype=float)))

    if not geometries:
        raise ValidationError("XYZ file contains no geometries")
    return Ensemble(tuple(geometries))


def remap_ensemble(ensemble: Ensemble, mappings: list[list[int]]) -> Ensemble:
    """Apply explicit one-based source indices so every frame matches reference ordering."""
    if len(mappings) != ensemble.size:
        raise ValidationError("atom_mappings must contain one mapping per frame")
    atom_count = len(ensemble.elements)
    expected = set(range(1, atom_count + 1))
    geometries: list[Geometry] = []
    reference_elements = ensemble.elements
    for geometry, mapping in zip(ensemble.geometries, mappings, strict=True):
        if set(mapping) != expected or len(mapping) != atom_count:
            raise ValidationError("each atom mapping must be a complete one-based permutation")
        indices = np.asarray(mapping, dtype=int) - 1
        elements = tuple(geometry.elements[index] for index in indices)
        if elements != reference_elements:
            raise ValidationError("atom mapping does not reproduce reference element ordering")
        geometries.append(Geometry(elements, geometry.coordinates[indices]))
    return Ensemble(tuple(geometries))
