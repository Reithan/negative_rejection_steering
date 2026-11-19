# Experiment: decoupled squash reference (P2)

Status: EXPERIMENTAL (P2 A/B). Not for release; remove or promote before merging.

## What the switch does

`NRS_SQUASH_REF` is an environment variable read on every guidance call.

- unset or `v`: shipped behavior. Skew, stretch and squash all run in v-space, and
  squash restores the length of the v-space cond.
- `raw`: skew and stretch still run in v-space. The skewed result is converted back
  to `x - x0` form with no squash applied, then squash is applied there:

  ```
  skewed_h = finalize(skewed_v)
  out = skewed_h * ((1 - squash) + squash * |cond| / |skewed_h|)
  ```

  where `cond` is the raw hook vector `x - x0_c`. Both lengths are measured in the
  same space (mixing spaces was the 0.6.0 bug).
- any other value: logs a warning once and uses the default.

For FLOW models the conversion is the identity, so `raw` is bit-identical to the default.

## Why

- At squash 1.0, `raw` returns a vector with exactly the sampler's own `|x - x0_c|` at every
  pixel on every model type, so the README's speed-limit text becomes true again.
- At high sigma `|cond|` in `x - x0` space is long, so squash leaves skew alone in early
  steps, as it did before 1.1.0.

## Risk

1.1.0's v-space squash was reported as burn-resistant at the failure edge. If that benefit
comes from squash acting on skew early, `raw` gives it up. The A/B has to check for this.

## A/B protocol

Use a fixed seed, the same prompt and the same model for every pair of runs, and set
`NRS_GEOMETRY_LOG=<file>.jsonl` so each run records per-step statistics. Run each setting
once with `NRS_SQUASH_REF` unset and once with `NRS_SQUASH_REF=raw`.

Settings (skew/stretch/squash):

1. 2/4/0.5 (the defaults).
2. 10/2/0.5 and 10/2/1.0 (high skew, squash 0.5 and 1.0).
3. The settings where 1.0.0 burned.

Compare images for burn at the failure edge, and compare `squash_scale_mean/min/p05`
against `squash_scale_raw_mean/min/p05` in the logs (the `squash_ref` field says which
mode produced each line). Suggested models: the EPS checkpoint, one v-pred checkpoint, one
flow model (expect identical output), and H3.
