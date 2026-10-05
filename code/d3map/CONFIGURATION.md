# d3map project configuration

Release projects use schema `1.0` in `project.json`. Atom numbers are one-based. `project_kind` is `d2-map` or `d3-map`; `structures.reference` is always present and `structures.extended` is required for d3-map.

Each structure records its copied input, charge, multiplicity, centre and origin atoms, alignment atoms, reactive and secondary directions, protected contacts, frozen atoms, steric atoms, solvent, topology policy, sampling settings, and explicit chemical-review confirmation. Transition-metal execution is rejected until confirmation is recorded. Hydrogen atoms remain available to electronic-structure and topology calculations but are excluded from ordinary centre and steric selectors.

`shared` defines temperature and the common static, occupation-field, axial, displaced-scan, reporting, and radii profiles. A standard d3-map requires compatible centre element, origin type, frame semantics, grids, domains, offsets, temperature, radii, thresholds, and probe semantics. `comparison.allow_nonstandard: true` permits an expert override and permanently labels the result non-standard.

`comparison.difference_sign` is fixed to `extended-minus-reference`. Its default map tolerances are:

```json
{
  "contact_probability": 0.01,
  "first_contact_q50": 0.05,
  "first_contact_interval_10_90": 0.05,
  "occupied_depth": 0.05
}
```

Tolerance masks affect figures and changed-area classifications only. Signed, absolute, and safe-relative numerical arrays remain complete in `difference-fields.npz`.

The sampling backend is `xtb-crest`. Per-structure settings may specify executable paths, thread count, quick mode, protected contacts, Cartesian freezes, and explicit trajectory fallback consent. Executable locations may be stored in desktop preferences; portable ZIPs exclude those machine-local preferences.

Legacy schema `0.1` remains readable by the numerical engine. Use `d3map migrate OLD_PROJECT DESTINATION` to create a release project; the source is copied to a `.v0.3.backup` file before migration.
