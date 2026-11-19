# Experiment: decoupled squash reference (P2)

Status: EXPERIMENTAL (P2 A/B). Not for release; remove or promote before merging.

## What the switch does

`NRS_SQUASH_REF` is an environment variable read on every guidance call.

- unset or `v`: shipped behavior. Skew, stretch and squash all run in v-space, and
  squash rescales the whole guided vector to the length of the v-space cond.
- `guidance`: skew and stretch are unchanged. Squash then limits only the *push* NRS adds
  on top of cond, towards the length of `cond - uncond`. Cond itself is left alone:

  ```
  delta = cond - uncond
  push  = stretch * delta_par + skew * delta_perp      # what NRS adds on top of cond
  scale = (1 - squash) + squash * |delta| / |push|
  out   = cond + scale * push
  ```

  The code computes this in v-space (`push = skewed - nrs_cond`). Both `push` and `delta`
  are differences, so the affine v-space conversion cancels in the ratio. The result
  doesn't depend on the reference point and is the same idea for VP and FLOW models.
- any other value (including the retired `raw`): logs a warning once and uses the default.

What to expect from the numbers:

- Squash 0 matches the default (up to float rounding).
- Squash 1 keeps the push's direction but gives it the length of `cond - uncond`. That is
  as much push as CFG 2 (skew = stretch = 1) would give, just pointed where skew/stretch say.
- At high skew the push is much longer than `|delta|`, so the scale is roughly
  `1 - squash`. For example, 10/2 at squash 0.5 is close to 5/1 in strength, with 10/2's
  direction.

## Why

The shipped squash rescales `p - x0` as a whole. With p near 0 at high sigma, that shrinks
the base prediction towards p as well as the guidance, mostly in early steps
(`squash_scale_mean` about 0.2 early at squash 1 in the P2 logs). The working hypothesis is
that this base shrinkage, not the limit on the guidance, causes squash burn (overdetailed,
contrasty "microdetail" look). `guidance` mode limits only the guidance, so the A/B tells
the two apart:

- Detail boost kept, squash burn gone: the burn came from the base shrinkage. `guidance`
  is a candidate to ship.
- Detail boost and burn both gone: both came from the base shrinkage. Keep `v`, and
  document squash as a detail/contrast control rather than a guidance limit.
- Burn kept: squash burn comes from limiting the guidance itself.

## Retired variant: `raw`

The first P2 variant squashed against the length of the raw hook vector `x - x0_c`. In the
2026-11-19 A/B (v-pred ztSNR model, Euler a, 24 steps), its scale stayed at 0.97–1.00 in
every sigma band even at squash 1, so it barely squashed. Burn went down, but squash's
detail effect also disappeared. It also measures the wrong thing: squash is meant to limit
the guidance, not the size of the model's ordinary denoising step. It was removed in favour
of `guidance`.

## A/B protocol

Use a fixed seed, the same prompt and the same model for every pair of runs, and set
`NRS_GEOMETRY_LOG=<file>.jsonl` so each run records per-step statistics. Run each setting
once with `NRS_SQUASH_REF` unset and once with `NRS_SQUASH_REF=guidance`.

Settings (skew/stretch/squash):

1. 2/4/0.5 (the defaults).
2. 10/2/0.5, 10/2/0.75 and 10/2/1.0 (high skew across the squash range).
3. A setting at the skew/stretch burn edge (about 15–20) with squash 0.5, to check that
   `guidance` still holds back skew/stretch burn.

Compare images for squash burn (overdetail, contrast), the detail boost, and skew/stretch
burn (oversaturation, shapes turning abstract). In the logs, `squash_ref` says which mode
produced each line. `squash_scale_mean/min/p05` is always the shipped v-space scale, and
`guidance` lines also carry `squash_scale_guidance_mean/min/p05`, the scale actually
applied to the push. Suggested models: the EPS checkpoint, one v-pred checkpoint, one flow
model, and H3.
