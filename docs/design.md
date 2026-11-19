# NRS design note

This note is for contributors who change the guidance math in `NRS/nodes_NRS.py`. It records what the sampler hands to NRS, the one design variable that every shipped version has chosen differently, and the rules for changing it.

Regression tests for the identities below live in `tests/test_reference_point.py` (added in a separate PR).

## The hook contract

NRS runs inside the sampler's cfg hook. For every prediction type, the hook receives:

```
cond   = x - x0_c
uncond = x - x0_u
```

`x` is the noisy latent. `x0_c` and `x0_u` are the denoised predictions for the conditional and unconditional passes. The hook returns a vector `r`, and the sampler treats `x - r` as the guided denoised prediction.

The source is `cfg_function` in ComfyUI `comfy/samplers.py`. Before it calls the hook, ComfyUI has already converted the raw model output to `x0`. Forge and reForge build the same arguments.

Consequence: EPS, V and X0 stop mattering before NRS runs. The two vectors have the same meaning for all three. Do not apply prediction-type formulas (eps to v, v to eps, and so on) to them. They are not eps or v outputs. They are `x - x0`. Applying those formulas to them produced the 0.5.0 and 0.6.0 conversions, which have no principled reading.

## The reference-point form

Every conversion NRS has shipped is an invertible affine map of `cond` and `uncond`. Each one is the same as picking a reference point `p = λ(σ)·x` and running the unchanged geometry on the vectors `p - x0_c` and `p - x0_u`. The geometry is scale-invariant, so only `p` matters, not the scale of the vectors.

Define `Δ = x0_c - x0_u`. The axis is the direction of `p - x0_c`. At each pixel, across the channel dimension (`dim=1`), split `Δ` into:

- `Δ∥`, the part along the axis.
- `Δ⊥`, the part perpendicular to it.

Then:

```
guided x0 = (1 - s)·p + s·(x0_c + stretch·Δ∥ + skew·Δ⊥)

s = squash_scale
  = (1 - squash) + squash·|p - x0_c| / |p - x0_c - stretch·Δ∥ - skew·Δ⊥|
```

In `_apply_guidance`, `proj_diff` is `-Δ∥`, `u_rej_c` is `Δ⊥`, and `x_final` is `s·(p - x0_c - stretch·Δ∥ - skew·Δ⊥)`.

### Consequences

1. At squash 0, `s = 1` and `p` only chooses the axis. With skew = stretch = w - 1, the result is exactly CFG at scale w in exact arithmetic. This holds for every prediction type and every version. In float32 the two differ by rounding, about 1e-5 at high sigma, well below the effect of a 0.001 change in CFG scale (`0.001·|x0_c - x0_u|`, about 3e-4 on synthetic latents). The model amplifies small perturbations, so generations are near-identical but not pixel-identical. How much changes depends on the model, prompt, sampler and settings. Small details change most, but small composition changes can also appear. In one Forge-Neo test (Euler a, 24 steps) the visible difference was roughly the same as, or smaller than, changing CFG by 0.001 (a subjective comparison).
2. With skew 0, squash removes its own fraction of the stretch push. The effective stretch is `stretch·(1 - squash)`, at every sigma, on each pixel where the push does not overshoot, that is where `|p - x0_c| > stretch·(Δ·e)` (e is the unit vector along `p - x0_c`). Where the push does overshoot, `p - x0_c - stretch·Δ∥` flips direction, its length becomes `stretch·(Δ·e) - |p - x0_c|`, and the identity no longer holds. `tests/test_reference_point.py` checks it on the non-overshooting pixels only.
3. Whatever squash removes is replaced by `p`. So the reference point decides what squash does to the image.

## The current choice

```
λ = 1/(1 + σ²)   for EPS, V, X0 and UNKNOWN
λ = 1            for FLOW
```

For the VP case, `_convert_to_v_space` computes:

```
A      = x - x/(1 + σ²)          (= x·σ²/(1 + σ²))
factor = σ / sqrt(1 + σ²)
out    = (cond - A) / factor
       = (x - x0 - x + x/(1 + σ²)) / factor
       = (p - x0) / factor        with p = x/(1 + σ²)
```

So `p = x/(1 + σ²)` and the vector is the true VP velocity. The `factor` only rescales, which the geometry ignores. `x/(1 + σ²)` is the denoised estimate for unit-variance Gaussian data, so each vector measures how far a prediction departs from that baseline. `_finalize_from_v_space` inverts the map exactly: `A + x_final·factor`.

For FLOW, `x - x0 = σ·out` is already a scalar multiple of the model's velocity. The map is the identity, which is `p = x`.

`_is_vp` decides the branch. Unknown prediction types log a warning and take the VP path.

## Version history

This is history, not a menu. Each row is what the reference point was, not a recommendation.

| Version | Commit | EPS models | v-pred models |
|---|---|---|---|
| 0.4.x (Mar 2025) | tag `0.4.5` | `x/(1+σ²)` | `x/(1+σ²)` |
| 0.5.x (Jul 2025) | `50ddd2a` | `x` | `x/(1+σ²)` |
| 0.6.0 to 1.0.0 | `60b5127` | `x·(1 + σ/√(1+σ²))` | `x` |
| 1.1.0 | `1be05ad` | `x/(1+σ²)` | `x/(1+σ²)` |

Notes:

- Flow models ran the EPS formula with a flow sigma until 1.0.0 (`c526ec9`). They have used `p = x` since.
- 0.4.x took dot products over the last tensor axis. They have been over the channel axis (`dim=1`) since 0.6.0.
- The 0.6.0 EPS reference point came from applying an eps-to-v formula to a vector that was already `x - x0`.
- No version was numerically broken. Each map inverts exactly. The versions differ only in geometry.
- The squash code has not changed since `3d8827f`. The 1.1.0 behavior change comes from `_convert_to_v_space`, which changed the length of the vector squash works on. At high sigma, 1.1.0 vectors are much shorter than the 0.6.0 ones, so squash now acts on skew at every step, not only in late steps. The axis also moved: at high sigma it is now the predicted image `x0_c`, not the per-pixel noise direction.

## Rejected: decoupled squash reference (P2)

Squash is meant to keep the direction that skew and stretch produce and pull the length back towards the length of the positive prediction. The shipped squash does this on the v-space vectors, so its target is the positive prompt's own velocity length. In November 2026 two alternatives were tried on draft PR #53 (closed, not merged), behind an `NRS_SQUASH_REF` switch:

- `raw`: squash against the length of the hook vector `x - x0_c` instead. For VP models that vector is the predicted noise scaled by σ. It is nearly the same length for both prompts, so the squash scale stayed at 0.97–1.00 in every sigma band even at squash 1. Burn went down, but squash's detail effect also disappeared. Measured on one v-pred ztSNR model in Forge-Neo (Euler a, 24 steps).
- `guidance`: shrink only the part skew and stretch add on top of cond, towards `|cond - uncond|`. Cond itself was left alone. This pulls the direction back towards plain cond, which defeats the point of squash. It was reverted before any image test.

The shipped v-space squash stays. For reference, at squash 1 on that model it scaled the result by about 0.2 at σ > 10, 0.45–0.7 at 1 < σ ≤ 10 and 0.85–0.96 at σ ≤ 1. Squash acts mostly in the early, high-noise steps.

## Rules for changing the geometry

- Measure the squash lengths in one space. Compute `cond_len` and `nrs_len` from vectors in the same space. 0.6.0 mixed spaces here, and `3d8827f` fixed it.
- Keep both entry points working. The ComfyUI node and the A1111/Forge script both go through `NRS().patch`. The script calls it directly in `scripts/negative_rejection_steering_script.py`.
- Handle every prediction type: EPS, V, X0, FLOW (which covers the CONST families) and UNKNOWN. Handle packed multi-stream latents too. `_apply_guidance` runs on one unpacked stream at a time.
- Write tests against real torch, not stubs. Follow `tests/test_pack_split.py`.
- Any change to output needs a version bump and a `CHANGELOG.md` entry. Say which models and which settings change, and by how much.
- State the new `λ(σ)` or the new axis rule in the PR, and add a row to the table above.
- Do not describe a change as moving between eps-space and v-space. Describe it as a change to `p`, or to what the geometry does with `p`.
