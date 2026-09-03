# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - Unreleased

### Added

- **X0 (sample) prediction support** (#44): x0-prediction models are now handled through the shared v-prediction-space path, alongside EPS and v-pred.

### Fixed

- **Correct v-space conversion for all variance-preserving parameterizations** (#44). The sampler hook delivers `cond`/`uncond` as `x - x0` for EPS, v-pred, and x0 alike, so NRS now recovers the true velocity `v = (cond - A)/factor` and runs its geometry in v-prediction space, then inverts exactly on return. This replaces the prior EPS-only affine, which operated on an incorrect input-space assumption. Flow-matching (FLOW/CONST) models remain operated natively — their prediction is already a pure scalar multiple of the velocity, so no conversion is applied.

### Changed / Upgrade notes

- **Default parameters changed** 2/5/0.75 → **2/4/0.5** (Skew/Stretch/Squash) in both the ComfyUI node and the A1111-family (Forge/reForge/Forge Neo) script.
- **v-prediction models now run the v-space conversion** instead of operating on the raw guidance. For typical config ranges the output change is expected to be minimal (verified on EPS; v-pred/x0 are math-validated but **not yet image-validated** — spot-check and retune if needed).
- **Reproducibility note:** the same seed + config may produce a slightly different image than 1.0.0 because of the corrected v-space handling and the new defaults.

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
- Declared `requires-python` (>=3.10) so dependency locking is deterministic across environments.

### Changed / Upgrade notes

- Because flow-family models now use the correct FLOW space, NRS output for these models changes (for the better). Existing users of flow-matching models should retune Skew/Stretch/Squash. The `pre-flow` git tag preserves the prior behavior if a rollback is needed.
