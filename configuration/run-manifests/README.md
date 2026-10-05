# Run manifests and provenance

Source manifests are included alongside a repository-level summary and SHA-256 file inventory. Existing records sometimes include unrecorded fields or legacy path assumptions. Values copied from source manifests retain their provenance; fields not present in the source are identified in each system manifest. Before a public tag, regenerate the checksum file after final edits and replace any remaining source-workspace paths with repository-relative paths.
