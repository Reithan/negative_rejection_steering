# Example grid specs

Specs for regenerating the NRS example grids (`skew_array.png`, `stretch_array.png`, `squash_matrix.png`) on the current release. Each grid below is a set of values to press go on; nothing here has been run yet.

The sweeps sit around the default (Skew 2, Stretch 4, Squash 0.5) with a plain CFG row beside them. Balanced NRS settings behave close to CFG. At Squash 0, Skew = Stretch = w − 1 is the NRS counterpart of CFG at scale w: the same math, giving near-identical but not pixel-identical images. More generally, when Skew equals Stretch × (1 − Squash), the result behaves roughly like CFG at scale Skew + 1 (the default 2/4/0.5 is roughly comparable to CFG 3). The CFG row shows what NRS adds beyond a plain guidance scale.

## Record for every grid

Write these down for each grid set so the outputs can be compared later:

- NRS version (`__version__` in `NRS/nodes_NRS.py`)
- Model (checkpoint name and hash) and prediction type (EPS, V, X0, or FLOW)
- Sampler and scheduler
- Steps
- Seed (fixed across all cells)
- Resolution
- Positive prompt and negative prompt

Recommended model set: run the full set on an EPS (SDXL-family) checkpoint, a v-prediction checkpoint, and a flow-matching model. EPS, v-pred and x0 models share one NRS path and flow models run natively, but the families differ in training and latent channel count, so the same numbers can look different across them.

## Grid 1: CFG reference row (NRS disabled)

- **Purpose:** baseline for the NRS grids. Shows what plain CFG does at the same scales.
- **Fixed:** NRS off ("Enable NRS" unchecked), everything else from the record list.
- **Swept:** CFG scale 1, 3, 5, 7.

A1111/Forge XYZ plot:

- X type: `CFG Scale`
- X values: `1,3,5,7`

ComfyUI: bypass (or disconnect) the NRS node so the model feeds the KSampler directly. Set the KSampler CFG to 1, 3, 5, 7.

## Grid 2: Skew sweep

- **Purpose:** how Skew changes composition and geometry at fixed Stretch and Squash.
- **Fixed:** Stretch 4, Squash 0.5, NRS enabled.
- **Swept:** Skew 0, 1, 2, 4, 8.

A1111/Forge XYZ plot:

- X type: `(NRS) Skew`
- X values: `0,1,2,4,8`
- Set the UI sliders to Stretch 4, Squash 0.5, and tick "Enable NRS".

ComfyUI: set the NRS node's `stretch` to 4 and `squash` to 0.5; vary `skew`.

## Grid 3: Stretch sweep

- **Purpose:** how Stretch changes color and texture intensity at fixed Skew and Squash.
- **Fixed:** Skew 2, Squash 0.5, NRS enabled.
- **Swept:** Stretch 0, 2, 4, 8, 12.

A1111/Forge XYZ plot:

- X type: `(NRS) Stretch`
- X values: `0,2,4,8,12`
- Set the UI sliders to Skew 2, Squash 0.5, and tick "Enable NRS".

ComfyUI: set `skew` to 2 and `squash` to 0.5; vary `stretch`.

## Grid 4: Squash sweep (two rows)

- **Purpose:** how Squash softens Skew and Stretch. Run at two strengths so the result is not tied to one operating point.
- **Fixed:** NRS enabled.
- **Swept:** Squash 0, 0.25, 0.5, 0.75, 1.0.
  - 4a: Skew 2, Stretch 4.
  - 4b: Skew 6, Stretch 8.

A1111/Forge XYZ plot (run once per row, changing only the UI sliders):

- X type: `(NRS) Squash`
- X values: `0,0.25,0.5,0.75,1`
- 4a: UI sliders Skew 2, Stretch 4.
- 4b: UI sliders Skew 6, Stretch 8.

ComfyUI: for 4a set `skew` 2 and `stretch` 4; for 4b set `skew` 6 and `stretch` 8. Vary `squash`.

## Grid 5 (optional): legacy single-knob arrays

Kept from earlier versions. Only keep these in the set if they show something the sweeps above do not. The maintainer decides at regeneration time.

- **5a, Skew from zero:** Skew 0 to 10 in steps of 1, Stretch 0, Squash 0.
  - A1111/Forge X type `(NRS) Skew`, X values `0,1,2,3,4,5,6,7,8,9,10`.
- **5b, Stretch from zero:** Stretch 0 to 10 in steps of 1, Skew 0, Squash 0.
  - A1111/Forge X type `(NRS) Stretch`, X values `0,1,2,3,4,5,6,7,8,9,10`.

In both, set the other two knobs in the UI as stated and tick "Enable NRS".

## A1111/Forge support

`scripts/negative_rejection_steering_script.py` registers `(NRS) Enabled`, `(NRS) Skew`, `(NRS) Stretch`, and `(NRS) Squash` as XYZ axes when the built-in XYZ Plot script is loaded (`make_axis_on_xyz_grid`). The axes are only added if the XYZ Plot script is found; if it is missing, set the sliders by hand and run one image per value. Forge has not been checked against this code path.

UI labels, for setting fixed values by hand: "Enable NRS", "NRS Skew Scale", "NRS Stretch Scale", "NRS Squash Multiplier".

## ComfyUI node inputs

The NRS node (`advanced/model`) takes a MODEL input and outputs a MODEL. Inputs: `model`, `skew`, `stretch`, `squash`. The KSampler CFG is ignored while NRS is in the chain.

- Wire: Model → NRS node → KSampler.
- Vary only the widget named in each grid; hold the others at the values listed.
- For the CFG row, bypass the NRS node and vary the KSampler CFG.

## Checklist before pressing go

- [ ] NRS version recorded (`__version__`)
- [ ] Model and prediction type recorded
- [ ] Sampler, scheduler, steps, seed, resolution recorded
- [ ] Prompt and negative prompt recorded
- [ ] Grid 1 CFG row run with NRS off
- [ ] Grids 2 to 4 run with NRS on and the fixed values set
- [ ] Repeated on the EPS, v-pred, and flow checkpoints
