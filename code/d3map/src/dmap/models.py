from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Geometry:
    elements: tuple[str, ...]
    coordinates: FloatArray

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.shape != (len(self.elements), 3):
            raise ValidationError("coordinates must have shape (atom_count, 3)")
        if not np.all(np.isfinite(coordinates)):
            raise ValidationError("coordinates must be finite")
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True)
class Ensemble:
    geometries: tuple[Geometry, ...]

    def __post_init__(self) -> None:
        if not self.geometries:
            raise ValidationError("ensemble must contain at least one geometry")
        elements = self.geometries[0].elements
        if any(geometry.elements != elements for geometry in self.geometries[1:]):
            raise ValidationError("all ensemble geometries must use identical atom ordering")

    @property
    def elements(self) -> tuple[str, ...]:
        return self.geometries[0].elements

    @property
    def size(self) -> int:
        return len(self.geometries)


@dataclass(frozen=True)
class CatalyticFrame:
    origin: FloatArray
    basis: FloatArray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        basis = np.asarray(self.basis, dtype=float)
        if origin.shape != (3,) or basis.shape != (3, 3):
            raise ValidationError("catalytic frame requires a 3-vector origin and 3x3 basis")
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-10):
            raise ValidationError("catalytic frame basis must be orthonormal")
        if np.linalg.det(basis) <= 0.0:
            raise ValidationError("catalytic frame must be a proper right-handed rotation")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "basis", basis)

    @classmethod
    def identity(cls) -> CatalyticFrame:
        return cls(np.zeros(3), np.eye(3))

    @classmethod
    def from_directions(
        cls,
        *,
        origin: FloatArray,
        z_direction: FloatArray,
        x_hint: FloatArray,
    ) -> CatalyticFrame:
        z_axis = np.asarray(z_direction, dtype=float)
        x_candidate = np.asarray(x_hint, dtype=float)
        if z_axis.shape != (3,) or x_candidate.shape != (3,):
            raise ValidationError("frame directions must be three-dimensional vectors")
        z_norm = np.linalg.norm(z_axis)
        if z_norm <= 1e-12:
            raise ValidationError("reactive z direction cannot be zero")
        z_axis = z_axis / z_norm
        x_axis = x_candidate - np.dot(x_candidate, z_axis) * z_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm <= 1e-12:
            raise ValidationError("secondary direction cannot be parallel to reactive z direction")
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        return cls(np.asarray(origin, dtype=float), np.column_stack((x_axis, y_axis, z_axis)))

    def transform(self, coordinates: FloatArray) -> FloatArray:
        return (np.asarray(coordinates, dtype=float) - self.origin) @ self.basis


@dataclass(frozen=True)
class AxialGrid:
    transverse_radius: float = 3.5
    z_min: float = 0.0
    z_max: float = 6.0
    spacing: float = 0.1

    def __post_init__(self) -> None:
        if self.transverse_radius <= 0.0:
            raise ValidationError("transverse radius must be positive")
        if self.spacing <= 0.0:
            raise ValidationError("grid spacing must be positive")
        if self.z_max < self.z_min:
            raise ValidationError("z_max must be greater than or equal to z_min")
        transverse_steps = 2.0 * self.transverse_radius / self.spacing
        axial_steps = (self.z_max - self.z_min) / self.spacing
        if not np.isclose(transverse_steps, round(transverse_steps), atol=1e-10):
            raise ValidationError("grid spacing must divide the transverse diameter exactly")
        if not np.isclose(axial_steps, round(axial_steps), atol=1e-10):
            raise ValidationError("grid spacing must divide the axial range exactly")


@dataclass(frozen=True)
class AxialSliceSummary:
    z: float
    mean_occupation: float
    open_fraction: float
    breathing_fraction: float
    blocked_fraction: float
    mean_entropy: float
    integrated_entropy: float
    quadrant_mean_occupation: tuple[float, float, float, float]
    largest_open_region_fraction: float
    largest_open_region_centroid_x: float | None
    largest_open_region_centroid_y: float | None
    largest_open_region_anisotropy: float | None


@dataclass(frozen=True)
class AxialResult:
    x: FloatArray
    y: FloatArray
    z: FloatArray
    occupation: FloatArray
    entropy: FloatArray
    summaries: tuple[AxialSliceSummary, ...]
