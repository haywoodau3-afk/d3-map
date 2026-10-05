from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError
from .models import CatalyticFrame, Geometry

PROJECTION_DIRECTIONS = ("+z", "-z", "+x", "-x", "+y", "-y")


@dataclass(frozen=True)
class BuriedVolumeResult:
    sampled_total: float
    sampled_buried: float
    percent_buried: float
    sphere_radius: float
    spacing: float


@dataclass(frozen=True)
class SolidAngleResult:
    solid_angle: float
    g_percent: float
    exposed_percent: float
    direction_count: int


@dataclass(frozen=True)
class TopographicMapResult:
    axis: np.ndarray
    top_surface: np.ndarray
    bottom_surface: np.ndarray
    sphere_radius: float
    spacing: float


@dataclass(frozen=True)
class AxialTopographicMapsResult:
    offsets: np.ndarray
    axis: np.ndarray
    top_surfaces: np.ndarray
    bottom_surfaces: np.ndarray
    sphere_radius: float
    spacing: float
    direction_names: tuple[str, ...]
    direction_vectors: np.ndarray


@dataclass(frozen=True)
class DisplacedStericScanResult:
    offsets: np.ndarray
    vbur_percent: np.ndarray
    g_percent: np.ndarray


def _selected_atoms(
    geometry: Geometry,
    frame: CatalyticFrame,
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(tuple(steric_atom_indices), dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(geometry.elements)):
        raise ValidationError("steric atom indices must identify at least one valid atom")
    try:
        atom_radii = np.array([radii[geometry.elements[index]] for index in indices])
    except KeyError as error:
        raise ValidationError(f"no radius configured for element: {error.args[0]}") from error
    if np.any(atom_radii <= 0.0):
        raise ValidationError("atomic radii must be positive")
    return frame.transform(geometry.coordinates)[indices], atom_radii


def calculate_buried_volume(
    geometry: Geometry,
    *,
    frame: CatalyticFrame,
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    sphere_radius: float = 3.5,
    spacing: float = 0.1,
) -> BuriedVolumeResult:
    """Sample hard-sphere buried volume independently of the Part 1 runtime."""
    if sphere_radius <= 0.0 or spacing <= 0.0:
        raise ValidationError("sphere radius and spacing must be positive")
    coordinates, atom_radii = _selected_atoms(geometry, frame, steric_atom_indices, radii)
    count = int(2.0 * sphere_radius / spacing + 1)
    axis = -sphere_radius + np.arange(count, dtype=float) * spacing
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    inside = np.sum(points * points, axis=1) <= sphere_radius**2 + 0.0001 * spacing**2
    sphere_points = points[inside]
    occupied = np.zeros(len(sphere_points), dtype=bool)
    for atom, radius in zip(coordinates, atom_radii, strict=True):
        delta = sphere_points - atom
        occupied |= np.sum(delta * delta, axis=1) < radius * radius
    total = float(len(sphere_points) * spacing**3)
    buried = float(np.count_nonzero(occupied) * spacing**3)
    percent = 100.0 * buried / total if total else 0.0
    return BuriedVolumeResult(total, buried, percent, sphere_radius, spacing)


def calculate_topographic_map(
    geometry: Geometry,
    *,
    frame: CatalyticFrame,
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    sphere_radius: float = 3.5,
    spacing: float = 0.1,
) -> TopographicMapResult:
    """Project first and last occupied grid points along the catalytic z axis."""
    if sphere_radius <= 0.0 or spacing <= 0.0:
        raise ValidationError("sphere radius and spacing must be positive")
    coordinates, atom_radii = _selected_atoms(geometry, frame, steric_atom_indices, radii)
    count = int(2.0 * sphere_radius / spacing + 1)
    axis = -sphere_radius + np.arange(count, dtype=float) * spacing
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    inside = xx * xx + yy * yy + zz * zz <= sphere_radius**2 + 0.0001 * spacing**2
    occupied = np.zeros_like(inside)
    for atom, radius in zip(coordinates, atom_radii, strict=True):
        distance_squared = (xx - atom[0]) ** 2 + (yy - atom[1]) ** 2 + (zz - atom[2]) ** 2
        occupied |= inside & (distance_squared < radius * radius)

    any_occupied = occupied.any(axis=2)
    top = np.max(np.where(occupied, zz, -np.inf), axis=2)
    bottom = np.min(np.where(occupied, zz, np.inf), axis=2)
    top = np.where(any_occupied, top, -2.0 * sphere_radius)
    bottom = np.where(any_occupied, bottom, 2.0 * sphere_radius)
    return TopographicMapResult(axis, top, bottom, sphere_radius, spacing)


def calculate_axial_topographic_maps(
    geometry: Geometry,
    *,
    frame: CatalyticFrame,
    offsets: Sequence[float],
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    sphere_radius: float = 3.5,
    spacing: float = 0.1,
    directions: Sequence[str] = PROJECTION_DIRECTIONS,
) -> AxialTopographicMapsResult:
    """Repeat the SambVca-style projection along each requested signed frame axis."""
    offset_values = np.asarray(tuple(offsets), dtype=float)
    if offset_values.ndim != 1 or offset_values.size == 0:
        raise ValidationError("axial projection offsets must be a non-empty sequence")
    if not np.all(np.isfinite(offset_values)):
        raise ValidationError("axial projection offsets must be finite")
    direction_names = tuple(directions)
    if not direction_names:
        raise ValidationError("axial projection directions must be non-empty")
    if len(set(direction_names)) != len(direction_names):
        raise ValidationError("axial projection directions cannot contain duplicates")
    unknown = tuple(name for name in direction_names if name not in PROJECTION_DIRECTIONS)
    if unknown:
        raise ValidationError(
            f"unsupported axial projection direction(s): {', '.join(unknown)}"
        )

    top_surfaces = []
    bottom_surfaces = []
    axis: np.ndarray | None = None
    direction_vectors = []
    for direction_name in direction_names:
        local_basis = _projection_basis(frame, direction_name)
        direction_vectors.append(local_basis[:, 2])
        direction_top_surfaces = []
        direction_bottom_surfaces = []
        for offset in offset_values:
            local_frame = CatalyticFrame(
                frame.origin + float(offset) * local_basis[:, 2], local_basis
            )
            projection = calculate_topographic_map(
                geometry,
                frame=local_frame,
                steric_atom_indices=steric_atom_indices,
                radii=radii,
                sphere_radius=sphere_radius,
                spacing=spacing,
            )
            axis = projection.axis
            direction_top_surfaces.append(projection.top_surface)
            direction_bottom_surfaces.append(projection.bottom_surface)
        top_surfaces.append(np.stack(direction_top_surfaces))
        bottom_surfaces.append(np.stack(direction_bottom_surfaces))

    if axis is None:  # pragma: no cover - guarded by the non-empty validation above
        raise ValidationError("axial projection offsets must be non-empty")
    return AxialTopographicMapsResult(
        offset_values,
        axis,
        np.stack(top_surfaces),
        np.stack(bottom_surfaces),
        sphere_radius,
        spacing,
        direction_names,
        np.stack(direction_vectors),
    )


def _projection_basis(frame: CatalyticFrame, direction: str) -> np.ndarray:
    """Return a right-handed local basis whose positive z points in ``direction``."""
    x_axis, y_axis, z_axis = frame.basis.T
    bases = {
        "+z": (x_axis, y_axis, z_axis),
        "-z": (x_axis, -y_axis, -z_axis),
        "+x": (y_axis, z_axis, x_axis),
        "-x": (y_axis, -z_axis, -x_axis),
        "+y": (z_axis, x_axis, y_axis),
        "-y": (z_axis, -x_axis, -y_axis),
    }
    return np.column_stack(bases[direction])


def calculate_displaced_steric_scan(
    geometry: Geometry,
    *,
    frame: CatalyticFrame,
    offsets: Sequence[float],
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    g_radii: Mapping[str, float] | None = None,
    sphere_radius: float = 3.5,
    spacing: float = 0.1,
    direction_count: int = 40_962,
) -> DisplacedStericScanResult:
    """Calculate separately labelled Vbur(z;R) and G(z) translated along local +z."""
    values = np.asarray(tuple(offsets), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValidationError("displaced scan offsets must be a non-empty finite sequence")
    direction = frame.basis[:, 2]
    vbur, g = [], []
    for offset in values:
        local = CatalyticFrame(frame.origin + float(offset) * direction, frame.basis)
        vbur.append(
            calculate_buried_volume(
                geometry,
                frame=local,
                steric_atom_indices=steric_atom_indices,
                radii=radii,
                sphere_radius=sphere_radius,
                spacing=spacing,
            ).percent_buried
        )
        g.append(
            calculate_solid_angle(
                geometry,
                frame=local,
                steric_atom_indices=steric_atom_indices,
                radii=radii if g_radii is None else g_radii,
                direction_count=direction_count,
            ).g_percent
        )
    return DisplacedStericScanResult(values, np.asarray(vbur), np.asarray(g))


def equal_area_directions(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def angular_mask(
    coordinates: np.ndarray, atom_radii: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    blocked = np.zeros(len(directions), dtype=bool)
    for atom, radius in zip(coordinates, atom_radii, strict=True):
        distance_squared = float(np.dot(atom, atom))
        if distance_squared <= radius * radius:
            blocked[:] = True
            break
        forward = directions @ atom
        perpendicular_squared = distance_squared - forward * forward
        blocked |= (forward > 0.0) & (perpendicular_squared < radius * radius)
    return blocked


def calculate_solid_angle(
    geometry: Geometry,
    *,
    frame: CatalyticFrame,
    steric_atom_indices: Sequence[int],
    radii: Mapping[str, float],
    direction_count: int = 40_962,
) -> SolidAngleResult:
    """Calculate the union angular shadow on a deterministic equal-area grid."""
    if direction_count < 100:
        raise ValidationError("direction_count must be at least 100")
    coordinates, atom_radii = _selected_atoms(geometry, frame, steric_atom_indices, radii)
    directions = equal_area_directions(direction_count)
    blocked = angular_mask(coordinates, atom_radii, directions)
    fraction = float(np.mean(blocked))
    return SolidAngleResult(
        4.0 * np.pi * fraction,
        100.0 * fraction,
        100.0 * (1 - fraction),
        direction_count,
    )
