from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .errors import ValidationError
from .models import Ensemble, Geometry
from .static import equal_area_directions

# Pyykko-like single-bond radii, deliberately conservative for topology diagnostics.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Zn": 1.22,
    "Mn": 1.39,
    "Cr": 1.39,
    "Mo": 1.54,
    "W": 1.62,
    "Ti": 1.60,
    "Zr": 1.75,
    "Hf": 1.70,
    "V": 1.53,
    "Re": 1.51,
}
METALS = frozenset(
    {
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Zr",
        "Mo",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Hf",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
    }
)


@dataclass(frozen=True)
class CoordinationEnvironment:
    center_index: int
    donor_indices: tuple[int, ...]
    coordination_number: int
    geometry: str
    angle_rmsd_degrees: float | None
    vacancy_directions: NDArray[np.float64]
    vacancy_scores_degrees: NDArray[np.float64]


@dataclass(frozen=True)
class TopologyFinding:
    frame_index: int
    code: str
    atom_numbers: tuple[int, ...]
    message: str


@dataclass(frozen=True)
class ScreenedEnsemble:
    ensemble: Ensemble
    valid_indices: tuple[int, ...]
    quarantined_indices: tuple[int, ...]
    findings: tuple[TopologyFinding, ...]


def _radius(element: str) -> float:
    return COVALENT_RADII.get(element, 0.77)


def connectivity(geometry: Geometry, *, scale: float = 1.25) -> frozenset[tuple[int, int]]:
    """Infer ordinary covalent connectivity; metal contacts are handled separately."""
    edges: set[tuple[int, int]] = set()
    for i in range(len(geometry.elements)):
        for j in range(i + 1, len(geometry.elements)):
            if geometry.elements[i] in METALS or geometry.elements[j] in METALS:
                continue
            cutoff = scale * (_radius(geometry.elements[i]) + _radius(geometry.elements[j]))
            distance = float(np.linalg.norm(geometry.coordinates[i] - geometry.coordinates[j]))
            if 0.35 < distance <= cutoff:
                edges.add((i, j))
    return frozenset(edges)


def coordination_contacts(
    geometry: Geometry, center_index: int, *, scale: float = 1.35
) -> tuple[int, ...]:
    if center_index < 0 or center_index >= len(geometry.elements):
        raise ValidationError("center index is out of range")
    center = geometry.coordinates[center_index]
    center_element = geometry.elements[center_index]
    contacts = []
    for index, (element, coordinate) in enumerate(zip(geometry.elements, geometry.coordinates)):
        if index == center_index or element == "H":
            continue
        cutoff = scale * (_radius(center_element) + _radius(element))
        if float(np.linalg.norm(coordinate - center)) <= cutoff:
            contacts.append(index)
    return tuple(contacts)


_ANGLE_SIGNATURES: dict[int, dict[str, tuple[float, ...]]] = {
    2: {"linear": (180.0,), "bent": (120.0,)},
    3: {"trigonal_planar": (120.0, 120.0, 120.0), "t_shaped": (90.0, 90.0, 180.0)},
    4: {"tetrahedral": (109.47,) * 6, "square_planar": (90.0,) * 4 + (180.0,) * 2},
    5: {
        "trigonal_bipyramidal": (90.0,) * 6 + (120.0,) * 3 + (180.0,),
        "square_pyramidal": (90.0,) * 8 + (180.0,) * 2,
    },
    6: {"octahedral": (90.0,) * 12 + (180.0,) * 3},
}


def perceive_coordination(
    geometry: Geometry, center_index: int, *, vacancy_count: int = 3
) -> CoordinationEnvironment:
    donors = coordination_contacts(geometry, center_index)
    center = geometry.coordinates[center_index]
    vectors = (
        geometry.coordinates[np.asarray(donors, dtype=int)] - center if donors else np.empty((0, 3))
    )
    if len(vectors):
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    observed = sorted(
        np.degrees(np.arccos(np.clip(float(np.dot(vectors[i], vectors[j])), -1.0, 1.0)))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    )
    candidates = _ANGLE_SIGNATURES.get(len(donors), {})
    ranked = []
    for name, signature in candidates.items():
        expected = np.asarray(sorted(signature), dtype=float)
        if expected.shape == np.asarray(observed).shape:
            ranked.append((float(np.sqrt(np.mean((np.asarray(observed) - expected) ** 2))), name))
    geometry_name = min(ranked)[1] if ranked else f"coordination_{len(donors)}"
    rmsd = min(ranked)[0] if ranked else None

    sphere = equal_area_directions(4096)
    if len(vectors):
        nearest_angle = np.min(
            np.degrees(np.arccos(np.clip(sphere @ vectors.T, -1.0, 1.0))), axis=1
        )
    else:
        nearest_angle = np.full(len(sphere), 180.0)
    selected: list[int] = []
    for index in np.argsort(nearest_angle)[::-1]:
        if all(float(np.dot(sphere[index], sphere[other])) < 0.85 for other in selected):
            selected.append(int(index))
        if len(selected) == vacancy_count:
            break
    return CoordinationEnvironment(
        center_index,
        donors,
        len(donors),
        geometry_name,
        rmsd,
        sphere[selected],
        nearest_angle[selected],
    )


def screen_topology(
    ensemble: Ensemble,
    *,
    center_index: int,
    policy: Literal["strict", "coordination-flexible", "custom"] = "strict",
    allowed_codes: tuple[str, ...] = (),
) -> ScreenedEnsemble:
    """Compare every frame with frame zero and quarantine undeclared chemical changes."""
    if policy not in {"strict", "coordination-flexible", "custom"}:
        raise ValidationError("topology policy must be strict, coordination-flexible, or custom")
    reference = ensemble.geometries[0]
    reference_edges = connectivity(reference)
    reference_contacts = set(coordination_contacts(reference, center_index))
    reference_h_hosts = _hydrogen_hosts(reference)
    findings: list[TopologyFinding] = []
    invalid: set[int] = set()
    for frame_index, geometry in enumerate(ensemble.geometries[1:], start=1):
        current_edges = connectivity(geometry)
        for edge in sorted(reference_edges - current_edges):
            findings.append(
                TopologyFinding(
                    frame_index,
                    "covalent_bond_lost",
                    tuple(i + 1 for i in edge),
                    "covalent bond lost",
                )
            )
        for edge in sorted(current_edges - reference_edges):
            findings.append(
                TopologyFinding(
                    frame_index,
                    "covalent_bond_formed",
                    tuple(i + 1 for i in edge),
                    "covalent bond formed",
                )
            )
        contacts = set(coordination_contacts(geometry, center_index))
        for index in sorted(reference_contacts - contacts):
            findings.append(
                TopologyFinding(
                    frame_index,
                    "metal_ligand_dissociation",
                    (center_index + 1, index + 1),
                    "metal-ligand contact lost",
                )
            )
        for index in sorted(contacts - reference_contacts):
            findings.append(
                TopologyFinding(
                    frame_index,
                    "metal_ligand_association",
                    (center_index + 1, index + 1),
                    "metal-ligand contact formed",
                )
            )
        hosts = _hydrogen_hosts(geometry)
        for hydrogen, host in reference_h_hosts.items():
            if hosts.get(hydrogen) != host:
                findings.append(
                    TopologyFinding(
                        frame_index,
                        "proton_transfer",
                        (hydrogen + 1,),
                        "nearest heavy-atom host changed",
                    )
                )
        if _fragment_count(
            len(geometry.elements), current_edges, center_index, contacts
        ) != _fragment_count(
            len(reference.elements), reference_edges, center_index, reference_contacts
        ):
            findings.append(
                TopologyFinding(
                    frame_index, "fragmentation_change", (), "molecular fragment count changed"
                )
            )
    flexible = {"metal_ligand_dissociation", "metal_ligand_association"}
    allowed = set(allowed_codes) | (flexible if policy == "coordination-flexible" else set())
    for finding in findings:
        if finding.code not in allowed:
            invalid.add(finding.frame_index)
    valid = tuple(i for i in range(ensemble.size) if i not in invalid)
    if not valid:
        raise ValidationError("topology screening quarantined every frame")
    return ScreenedEnsemble(
        Ensemble(tuple(ensemble.geometries[i] for i in valid)),
        valid,
        tuple(sorted(invalid)),
        tuple(findings),
    )


def _hydrogen_hosts(geometry: Geometry) -> dict[int, int | None]:
    hosts: dict[int, int | None] = {}
    for index, element in enumerate(geometry.elements):
        if element != "H":
            continue
        distances = np.linalg.norm(geometry.coordinates - geometry.coordinates[index], axis=1)
        choices = [i for i, candidate in enumerate(geometry.elements) if candidate != "H"]
        host = min(choices, key=lambda i: distances[i]) if choices else None
        hosts[index] = host if host is not None and distances[host] <= 1.35 else None
    return hosts


def _fragment_count(
    atom_count: int, edges: frozenset[tuple[int, int]], center: int, contacts: set[int]
) -> int:
    adjacency: list[set[int]] = [set() for _ in range(atom_count)]
    for left, right in set(edges) | {(center, item) for item in contacts}:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[int] = set()
    count = 0
    for start in range(atom_count):
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            for item in adjacency[stack.pop()]:
                if item not in seen:
                    seen.add(item)
                    stack.append(item)
    return count
