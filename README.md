# Negative Rejection Steering
NRS seeks to replace the 'naive' linear interpolation of Classifier Free Guidance with a more nuanced and composable steering of the generation process with better mathematical basis.

#### _**TL;DR**_:
1. CFG is a bad 'knob'
2. NRS replaces CFG with 3 new knobs.
3. NRS lets you to create cooler outputs than CFG.

> [!TIP]
> Skip to the [Beginner How-To](#beginner-how-to) if you want to just get started.

### Math Demonstration
<details>
<summary>Expand for explanation of algorithm</summary>
<img align="right" src="Examples/NRS_graph.png" alt="Graph of NRS vs CFG" style="width: 40%; float: right;">

### NRS is Applied in Three Steps:
0. ***V-Space**: Optional pre-NRS step* If the model is not using v-prediction, we transform the EPS `cond` and `uncond` into v-prediction space before continuing, then revert to eps-space before return.
1. **Skewing**: The conditioned output tensor is skewed away from the direction of the rejection of the unconditioned tensor on the conditioned tensor. This lengthens the tensor in a direction perpendicular to its direction without affecting the positive guidance. The tensor is displaced by the rejection multiplied by the Skew parameter.[^1]
2. **Stretching**: The skewed tensor is stretched towards the direction of the original conditioned tensor based on its difference from the projection of uncond on cond. The stretch is multiplied by the Stretch parameter.[^1]
3. **Squashing**: The skewed and stretched tensor is rescaled towards the original length of the conditioned tensor. 100% squashing outputs the original length of the conditioned tensor simply 'steered' towards the skewed & squashed version's direction.[^1]
[^1]: All operations are done per feature across the step's batch, width, and height.

[Interactive Graph on Math3D.org](https://www.math3d.org/aTJW4UZtCh)
</details>

## Examples of NRS Effects
**Skew**
![Skew Example](Examples/skew_array.png)
**Stretch**
![Stretch Example](Examples/stretch_array.png)
**Squash**
![Squash Example](Examples/squash_matrix.png)
<details>
<summary><small>Generation details for reproduction</small></summary>

| Prompt     | |
| ---------- | --- |
| Tool       | [Stable Diffusion WebUI reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) |  
| Sampler    | DPM++ 2M |
| Scheduler  | Align Your Steps |
| Steps      | 25 |
| Dimensions | 912 x 624 |
| Seed       | `1334103348` |
| Model      | [Lobotomized Mix v1.5](https://civitai.com/models/1144932) |
| Embeddings | [Lazy Embeddings for ALL illustrious NoobAI...](https://civitai.com/models/1302719), [Smooth Embeddings](https://civitai.com/models/1065154) |
| Positive   | lazypos, [Smooth_Quality\|SmoothNoob_Quality], BREAK<br>very awa, masterpiece, best quality, year 2024, newest, highres, absurdres,<br>1girl, samurai archer, cyberpunk cityscape, rain-soaked rooftop, neon reflection puddles, volumetric mist,<br>photorealistic, digital art,<br>dramatic rim lighting, shallow depth of field, low angle viewpoint |
| Negative   | lazyloli, lazynsfw, BREAK<br>lazyhand, SmoothNegative_Hands-neg, BREAK<br>[Smooth_Negative-neg\|SmoothNoob_Negative-neg], BREAK<br>lowres, worst quality, worst aesthetic, bad quality, jpeg artifacts, scan artifacts,<br>blurry, deformed anatomy, bad hands, extra fingers, missing fingers, mutated hands,<br>watermark, logo, text, nsfw |
</details>

### Explanation of Effects
#### Skew
**Skew** changes the 'direction' of your generation, altering the image generation to 'steer' away from negative prompt elements as they conflict with your positive prompt. Increasing Skew will change scene composition, geometry, and scene elements to ensure that the final image aligns with the intention of your prompt pair.
#### Stretch
**Stretch** changes the intensity of generated elements that align more with your positive prompt than the negative. This 'hits the gas' on any elements that are more strongly aligned with your positive prompt than your negative, and 'hit the brakes' on the opposite.
#### Squash
**Squash** is the speed limit. At 0.0 Squash, each diffusion step receives the full intensity you set from Skew and Stretch, while 1.0 Squash ensures each step has only the original step size output by the model. This setting has no effect unless you have a non-zero Skew value. Squash will 'soften' the effects of Skew and Stretch as it's raised, but the 'removed' Skew and Stretch intensity is replaced by enhanced micro-detailing and 'burn'. Squash should generally be left low and used as a 'finishing' step after dialing in a decent Skew and Stretch value.

## Beginner How-To
1. Set Skew to your normal CFG Scale setting and Stretch to 1/2 your normal CFG Scale. Set Squash to 0.0.<br>
*Alternatively, try starting at 1/1/0.0 to get a baseline.*
2. Test some outputs. Results should be similar in quality to CFG.
3. Adjust Skew to change the intensity of your outputs adherence to your positive and negative prompts. This primarily effects composition of the output.
4. Adjust Stretch to intensify your positive prompt's aspects and colors where they differ from the negative prompt. This primarily effects color and texture.
5. Adjust Squash to soften Skew and Stretch's effects. The intensity removed from Skew and Stretch will generally become additional micro-detailing and elements.

> [!TIP] 
> You can experiment with negative values for each setting as well. This can be useful to understand how the model interpreting your negative prompt.

> [!WARNING] 
> Don't set NRS values to negatives if there are things in your negative prompt you **actually** don't want to see.

## Submitted User Examples
| User | CFG | NRS |
| --- | --- | --- |
| Mohnjiles from StabilityMatrix | ![CFG Example](Examples/mohnjiles_cfg.png) | ![NRS Example](Examples/mohnjiles_nrs.png) |
