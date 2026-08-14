# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-08-14

### Breaking
- Flow-family models are now reclassified to a dedicated FLOW prediction/operation space (#37), which changes NRS guidance behavior for flow-matching models. Retune Skew/Stretch/Squash for these models. The `pre-flow` git tag is available to roll back to the prior behavior.

### Added
- Pack-aware per-stream NRS routing with a degeneracy tripwire (#36): NRS now unpacks multi-stream packed latents (e.g. MiniMax H3 audio+video) and applies the geometry per stream on the real channel axis, instead of collapsing to a silent no-op on the flat packed latent.
- `__version__` string plus a patch-time log line announcing the version and detected prediction type; platform-agnostic GitHub issue template (#39).
- CI: full test suite with a >90% branch-coverage gate on diffs (#38); version-increment check in the publish workflow (#33); git hooks and development infrastructure (#32).

### Fixed / Changed
- Prediction-type detection for WAN / RES4LYF samplers (#30).
- Cleanup: removed a mangled guard and dead operation-space code paths; added prediction-type detection tests (#34).
- Resolved Node.js 20 deprecation warnings in GitHub Actions.
