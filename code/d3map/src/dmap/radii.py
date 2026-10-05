from __future__ import annotations

from collections.abc import Mapping, Sequence

from .errors import ValidationError

# Versioned, conservative van-der-Waals defaults for the initial heavy-atom axial profile.
VDW_RADII_ANGSTROM: dict[str, float] = {
    # Hydrogen is included so the public axial/field routines honor the
    # default ``hydrogen_policy=\"included\"`` without a profile lookup error.
    "H": 1.20,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Si": 2.10,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "As": 1.85,
    "Se": 1.90,
    "Br": 1.85,
    "I": 1.98,
}

# SambVca 2.1 compatibility values (already scaled and rounded by that distribution).
SAMBVCA_21_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.28,
    "B": 2.25,
    "C": 1.99,
    "N": 1.81,
    "O": 1.78,
    "F": 1.72,
    "Si": 2.46,
    "P": 2.11,
    "S": 2.11,
    "Cl": 2.05,
    "As": 2.16,
    "Se": 2.22,
    "Br": 2.16,
    "I": 2.32,
    "Ti": 2.47,
    "V": 2.42,
    "Cr": 2.41,
    "Mn": 2.40,
    "Fe": 2.39,
    "Co": 2.34,
    "Ni": 1.91,
    "Cu": 1.64,
    "Zn": 1.63,
    "Zr": 2.61,
    "Mo": 2.54,
    "Ru": 2.49,
    "Rh": 2.46,
    "Pd": 1.91,
    "Ag": 2.01,
    "Hf": 2.61,
    "W": 2.55,
    "Re": 2.53,
    "Os": 2.53,
    "Ir": 2.49,
    "Pt": 2.01,
    "Au": 1.94,
}


def radii_for_elements(
    elements: Sequence[str],
    overrides: Mapping[str, float] | None = None,
    *,
    profile: str = "dmap_vdw_v0.1",
) -> dict[str, float]:
    profiles = {
        "dmap_vdw_v0.1": VDW_RADII_ANGSTROM,
        "sambvca_2.1": SAMBVCA_21_RADII_ANGSTROM,
    }
    if profile not in profiles:
        raise ValidationError(f"unknown radii profile: {profile}")
    values = dict(profiles[profile])
    if overrides:
        values.update({element: float(radius) for element, radius in overrides.items()})
    missing = sorted({element for element in elements if element != "H" and element not in values})
    if missing:
        joined = ", ".join(missing)
        raise ValidationError(f"no radius configured for element(s) in {profile}: {joined}")
    return values
