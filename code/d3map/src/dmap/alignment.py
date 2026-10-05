from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError
from .models import Ensemble, Geometry


@dataclass(frozen=True)
class AlignmentDiagnostic:
    rmsd: float
    determinant: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class AlignedEnsemble:
    ensemble: Ensemble
    diagnostics: tuple[AlignmentDiagnostic, ...]


def align_ensemble(
    ensemble: Ensemble,
    *,
    atom_indices: Sequence[int],
    reference_index: int = 0,
    center_index: int | None = None,
) -> AlignedEnsemble:
    """Proper-rotation Kabsch alignment on a user-confirmed invariant atom set."""
    indices = np.asarray(tuple(atom_indices), dtype=int)
    if indices.ndim != 1 or indices.size < 3 or len(set(indices.tolist())) != indices.size:
        raise ValidationError("alignment requires at least three distinct atom indices")
    if np.any(indices < 0) or np.any(indices >= len(ensemble.elements)):
        raise ValidationError("alignment atom index is out of range")
    if reference_index < 0 or reference_index >= ensemble.size:
        raise ValidationError("alignment reference index is out of range")
    if center_index is not None and (center_index < 0 or center_index >= len(ensemble.elements)):
        raise ValidationError("catalytic center index is out of range")

    reference = ensemble.geometries[reference_index].coordinates
    reference_anchors = reference[indices]
    reference_origin = (
        reference_anchors.mean(axis=0) if center_index is None else reference[center_index]
    )
    reference_centered = reference_anchors - reference_origin
    if np.linalg.matrix_rank(reference_centered) < 2:
        raise ValidationError("alignment anchors are coincident or collinear")

    geometries: list[Geometry] = []
    diagnostics: list[AlignmentDiagnostic] = []
    for geometry in ensemble.geometries:
        mobile_anchors = geometry.coordinates[indices]
        mobile_origin = (
            mobile_anchors.mean(axis=0)
            if center_index is None
            else geometry.coordinates[center_index]
        )
        mobile_centered = mobile_anchors - mobile_origin
        covariance = mobile_centered.T @ reference_centered
        left, _, right_transpose = np.linalg.svd(covariance)
        rotation = left @ right_transpose
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_transpose
        transformed = (geometry.coordinates - mobile_origin) @ rotation + reference_origin
        residual = transformed[indices] - reference_anchors
        rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        translation = reference_origin - mobile_origin @ rotation
        geometries.append(Geometry(geometry.elements, transformed))
        diagnostics.append(
            AlignmentDiagnostic(rmsd, float(np.linalg.det(rotation)), rotation, translation)
        )
    return AlignedEnsemble(Ensemble(tuple(geometries)), tuple(diagnostics))
