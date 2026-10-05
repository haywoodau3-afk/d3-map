# d3map feature contracts

`features.csv` uses semantic identifier `d3map.features.v1`. It contains ensemble dVbur and dG summaries, occupation persistence, breathing fraction, pocket steric entropy, quadrant asymmetry, radial occupation bins, coordination-frame angular sectors, an angular fingerprint, approach accessibility, coordination-state populations, and translated-origin profiles.

The canonical fields are unsmoothed hard-sphere occupation probability and ray-union shielding probability. Their numerical integrals reproduce like-profile population-weighted Vbur and G. SambVca-compatible scalar Vbur and topographic projections use `sambvca_2.1`; canonical occupation and G use `dmap_vdw_v0.1`. Radii-profile differences are always disclosed.

All topographic output resolves `+x`, `-x`, `+y`, `-y`, `+z`, and `-z`. Each direction and offset keeps contact probability, conditional median first-contact depth, conditional 10–90% interval, and population-weighted occupied depth separate. The multiscale ML tables pool raw numerical fields rather than image colours and preserve missing-domain support through valid-fraction columns.

In d3-map mode, comparison output retains each reference and extended field plus signed, absolute, and safe-relative differences and tolerance-state masks. Relative values are missing where the reference is zero. Static, ensemble, trajectory, and paired-difference rows carry distinct semantics and must not be pooled without an explicit model.

## Probability-field v2

`d3map.features.v2` and `d3map.fields.v2` preserve the canonical probability fields while adding physical-volume and morphology outputs:

- local steric occupancy entropy integrated with the actual voxel volume in `Å³ nat`;
- normalized pocket flexibility and explicit `not_estimable` status for one-geometry inputs;
- the complete accessibility-persistence curve plus `0.10`, `0.50`, and `0.90` landmarks;
- absolute persistent-open, adaptive, and persistent-excluded volumes;
- signed `+x/-x/+y/-y/+z/-z` volumes, extents, centroids, principal axes, and anisotropy;
- geometry-level point, straight radial, and connected probe accessibility aggregated with declared population weights;
- energy-conditioned occupation, energy--occupation covariance, and conditional energy contrasts when compatible conformer energies exist;
- ensemble-minus-static residuals and d3 residual double differences;
- compressed per-geometry occupation, shielding, and accessibility events for reweighting and audit.

Connectivity is evaluated per geometry before population aggregation. Thresholding an average occupation field is never used to claim a physical channel. Legacy raw voxel entropy sums and uniformly averaged accessibility values remain explicitly named compatibility outputs.

Frozen-data upgrades use only stored coordinates, weights, frames, atom selections, and radii profiles. They do not launch xTB, CREST, MD, optimization, or energy calculations, and their conclusions retain the evidence status supported by the source data.
