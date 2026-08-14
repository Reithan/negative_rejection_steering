# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-08-14

### Fixed
- **Flow-family models now use a dedicated FLOW prediction/operation space** (#37). Flow-matching models (e.g. WAN21, Flux) were previously misclassified under the wrong prediction-type assumption, so NRS applied incorrect guidance geometry and underperformed on them. Reclassification corrects this: NRS now applies proper guidance for flow-family models.
- Prediction-type detection for WAN / RES4LYF samplers (#30).

### Added
- Pack-aware per-stream NRS routing with a degeneracy tripwire (#36): NRS now unpacks multi-stream packed latents (e.g. MiniMax H3 audio+video) and applies the geometry per stream on the real channel axis, instead of collapsing to a silent no-op on the flat packed latent.
- `__version__` string plus a patch-time log line announcing the version and detected prediction type; platform-agnostic GitHub issue template (#39).
- CI: full test suite with a >90% branch-coverage gate on diffs (#38); version-increment check in the publish workflow (#33); git hooks and development infrastructure (#32).

### Changed
- Cleanup: removed a mangled guard and dead operation-space code paths; added prediction-type detection tests (#34).
- Resolved Node.js 20 deprecation warnings in GitHub Actions.

### Notes
- As a result of the flow-family fix above, output changes for existing flow-model users — retune Skew/Stretch/Squash for these models. The `pre-flow` git tag rolls back to the prior (incorrect) behavior if needed.
