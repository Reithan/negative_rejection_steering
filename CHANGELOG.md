# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-08-14

### Fixed
- Flow-family (flow-matching) models now use a dedicated FLOW prediction/operation space (#37) so NRS applies the correct guidance geometry to them. Previously these models were misclassified, causing NRS to operate on an incorrect prediction-type assumption and underperform. This is the headline fix in 1.0.0.
- Pack-aware per-stream NRS routing with a degeneracy tripwire (#36): NRS now unpacks multi-stream packed latents (e.g. MiniMax H3 audio+video) and applies the geometry per stream on the real channel axis, instead of collapsing to a silent no-op on the flat packed latent.
- Prediction-type detection for WAN / RES4LYF samplers (#30).
- Removed a mangled guard and dead operation-space code paths; added prediction-type detection tests (#34).
- Resolved Node.js 20 deprecation warnings in GitHub Actions.

### Added
- `__version__` string plus a patch-time log line announcing the version and detected prediction type; platform-agnostic GitHub issue template (#39).
- CI: full test suite with a >90% branch-coverage gate on diffs (#38); version-increment check in the publish workflow (#33); git hooks and development infrastructure (#32).

### Changed / Upgrade notes
- Because flow-family models now use the correct FLOW space, NRS output for these models changes (for the better). Existing users of flow-matching models should retune Skew/Stretch/Squash. The `pre-flow` git tag preserves the prior behavior if a rollback is needed.
