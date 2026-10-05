# Case 3d: reactive 1a/4d/OTs dielectric pilot

## Scope and chemical identity

This is a separate experiment from the earlier bound-I 5c–4e/OMs demonstration. It uses the user-reviewed, constructed **substrate-1a-derived Ru–allenylidene + chiral donor 4d + OTs** assembly (241 atoms). The starting XYZ is preserved in [the preview folder](../../../../3dstructures/reactive-1a-4d-ots-preview/README.md); its construction provenance, formula, atom roles and stereochemical checks remain there. Coordinates were constructed from the published analogue, not obtained from a published DFT calculation of this exact assembly.

The net-neutral, closed-shell GFN2-xTB assignment is a working model, not a validated electronic-state determination for the diruthenium complex. The allenylidene precursor is relevant to the cyclization, but **is not a transition state**. No DFT, reaction-coordinate search, barrier estimate or ee/dr regression is performed here.

## Matched sequential protocol

1. Validate the native xTB input parser and calculate a single-point energy.
2. Optimize separately with GFN2-xTB in ddCOSMO at dielectric constants **2.24** and **9.08**, respectively. These are CCl₄-like and DCM-like dielectric surrogates, not complete solvent models or ALPB calculations.
3. Apply the same seven harmonic core-distance restraints (native force constant 0.25) in optimization and dynamics: Ru–Ru, all four Ru–bridging-S distances, Ru2–Cl and Ru1–Cα. Targets are identical and come from the reviewed starting geometry. There is no donor–anion or substrate-approach restraint.
4. Generate native velocities in a 20 fs fixed-random bootstrap, using a 2 fs integration step, hydrogen mass 4 amu, native SHAKE setting 2 and NVT at 298.15 K. Rescale the final bootstrap velocities to the target temperature using the **native logged degrees of freedom**, and retain the native final bootstrap coordinates. This bootstrap is preparation, not demonstrated equilibration.
5. Run one **1 ps explicitly RMSD-biased metadynamics pilot per dielectric**, sequentially, with the same numerical settings: `kpush=0.001` Eh, `alp=1.0`, `ramp=0.03`, maximum 25 bias structures, heavy-atom RMSD bias, nominal deposition interval 100 fs and saved geometries every 20 fs. Four CPU threads are used. The native log must confirm nonzero bias, deposition and normal MD completion; an emergency exit is a failure even if xTB also prints its general termination message.
6. The native 6.7.1 loop writes 50 pre-integration frames at 0.00–0.98 ps for this 500-step run; the final 1.00 ps geometry is retained in the restart, not an additional trajectory frame. Discard the first 0.2 ps and use uniform saved-frame weights. Energies in the biased trajectories are **not** used as Boltzmann weights. The states have a paired initialization protocol, not independent replicates.
7. Generate matched D2 maps in two masks: full assembly (excluding the Cγ origin atom), and external environment (Ru core + 4d + OTs, excluding the entire 1a-derived organic ligand). The latter separates environmental shielding from substrate-tether motion; it is a conditional mask of the same geometries, not an independently relaxed subsystem.
8. Construct signed D3 differences **DCM-like minus CCl₄-like** for each mask. Use the same grid, radius profiles, Cγ origin and core-aligned frame. The common reviewed geometry is a **zero-weight spatial anchor** in both inputs and contributes no occupation. The reactive +z direction is Cβ→Cγ; the secondary direction is Ru1→Ru2.
9. Compare the spatial occupation/accessibility changes with ensemble %Vbur. %Vbur uses the inherited SambVca 2.1 radius profile; spatial maps use the inherited d-Map profile. They are matched within each descriptor comparison but their integrals are not numerically interchangeable.

## Integrity and interpretation

Inputs are immutable once prepared. Completed stages record command lines and log hashes; partial-stage logs are retained and cannot be silently overwritten. The run manifest records progress, output hashes, trajectory count, temperature and restraint diagnostics. Mapping requires strict topology screening with at least 90% retained statistical weight and standard D3 compatibility. A failed gate must be investigated, not relaxed to obtain a desired solvent trend.

The pilot answers: **where does sampled steric occupation change when the dielectric changes for this specific constructed, core-restrained chemical state?** It does not establish equilibrium probabilities, solvent free energies, kinetics, transition-state discrimination, experimental ee or sampling convergence. Raw metadynamics visitation is biased. Any first/second-half stability diagnostic is descriptive, not independent validation. Agreement or disagreement with experimental selectivity is not a quantitative prediction.

Execution and results: [run-manifest.json](run-manifest.json). The driver is [run_reactive_solvent_contrast.py](../run_reactive_solvent_contrast.py). Only this driver targets the accepted reactive assembly; the older solvent driver targets the 5c–4e/OMs analogue.

## Sources

- [Ovian et al., Nature (2023)](https://doi.org/10.1038/s41586-023-05804-3): experimental and mechanistic context. The exact constructed assembly is not claimed to be its published DFT model.
- [xTB 6.7.1 command manual](https://github.com/grimme-lab/xtb/blob/v6.7.1/man/xtb.1.adoc), [native input manual](https://github.com/grimme-lab/xtb/blob/v6.7.1/man/xcontrol.7.adoc), and [native dynamics implementation](https://github.com/grimme-lab/xtb/blob/v6.7.1/src/dynamic.f90): dielectric option, metadynamics settings, restart and temperature handling.
- Molecular-dynamics and scientific-critical-thinking guidance from the downloaded Scientific Agent Skills informed preparation checks and the bounded interpretation; [Kassis et al. (2026)](https://doi.org/10.48550/arXiv.2609.00065).
