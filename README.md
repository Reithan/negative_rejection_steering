# Negative Rejection Steering
NRS seeks to replace the 'naive' linear interpolation of Classifier Free Guidance with a more nuanced and composable steering of the generation process with better mathematical basis.

#### _**TL;DR**_:
1. CFG is a bad 'knob'
2. NRS replaces CFG with 3 new knobs.
3. NRS lets you to create cooler outputs than CFG.

### Math Demonstration
<details>
<summary>Expand for explanation of algorithm</summary>
<img align="right" src="https://github.com/user-attachments/assets/01fabaff-8499-45f6-adad-d54b2c2fb7f1" alt="Graph of NRS vs CFG" style="width: 40%; float: right;">

### NRS is Applied in Three Steps:
1. **Skewing**: The conditioned output tensor is skewed away from the direction of the rejection of the unconditioned tensor on the conditioned tensor. This lengthens the tensor in a direction perpendicular to its direction without affecting the positive guidance. The tensor is displaced by the rejection multiplied by the Skew parameter.
2. **Stretching**: The skewed tensor is stretched towards the direction of the original conditioned tensor based on its difference from the projection of uncond on cond. The stretch is multiplied by the Stretch parameter.
3. **Squashing**: The skewed and stretched tensor is rescaled towards the original length of the conditioned tensor. 100% squashing outputs the original length of the conditioned tensor simply 'steered' towards the skewed & squashed version's direction.

[Interactive Graph on Math3D.org](https://www.math3d.org/aTJW4UZtCh)
</details>

## Parameters
Skew and Stretch are roughly similar to CFG, but decomposed, with `Stretch + 2 * Skew = 2 * CFG`, roughly.
Meaning, if you want to 'replicate' a simliar effect for a given CFG setting, you should set Skew equal to CFG, and Stretch to 1/2 CFG.
Squash should initially be set to 0%, then adjusted based on 'burn' of output.

- **Skew** changes the 'direction' of generation, which should result in changes to the content and composition of the image.
- **Stretch** changes to 'amplification' of generation, which should result in stronger prompt representation.
- **Squash** 'normalizes' the resulting guidance back towards the original amplitude. This results in a removal of 'burn-in' and artifacting of the output, transforming these defects into alternative guidance.

## Parameter Precision and Advanced Controls

Recent updates have introduced finer control over NRS parameters and new advanced options for more precise tuning.

### Parameter Precision Update

The core parameters—**Skew**, **Stretch**, and **Squash**—now support more granular adjustments. For instance, in environments like ComfyUI, their step value has been decreased from `0.01` to `0.001`. This allows for very fine-tuned control, which can be particularly useful when seeking subtle changes in image composition or artifact reduction.

### Advanced Parameters

These parameters offer deeper control over the NRS algorithm's components. It's generally recommended to start with the default values and adjust them cautiously once you are familiar with the effects of the primary Skew, Stretch, and Squash parameters.

*   **Rejection Strength (`rejection_strength`)**
    *   **Purpose**: Modulates the intensity of the rejection vector (`u_rej_c`) used in the Skewing step. This vector represents the aspects of the unconditioned tensor that are orthogonal to the conditioned tensor.
    *   **Default**: `1.0`
    *   **Range**: `0.0` to `5.0`
    *   **Usage**: Increasing this value amplifies the effect of the `skew` parameter by strengthening the influence of the rejection component. If you find the `skew` effect too subtle or too strong at its default, you can fine-tune it here. For example, a higher `rejection_strength` will make the `skew` parameter more sensitive.

*   **Projection Difference Strength (`projection_diff_strength`)**
    *   **Purpose**: Modulates the intensity of the projection difference vector (`proj_diff`) used in the Stretching step. This vector (`cond - u_on_c`) captures how much the conditioned tensor already diverges from the unconditioned tensor in its own direction.
    *   **Default**: `1.0`
    *   **Range**: `0.0` to `5.0`
    *   **Usage**: This parameter scales the `proj_diff` component before it's multiplied by the main `stretch` parameter. If `stretch` seems to be affecting the image too aggressively or not enough, adjusting `projection_diff_strength` can provide a more calibrated response. It allows you to control how much of the "difference" aspect is factored into the stretch.

*   **Squash Target Length Factor (`squash_target_length_factor`)**
    *   **Purpose**: Adjusts the target length for the Squashing step. Instead of always squashing towards the original length of the conditioned tensor, this factor allows you to aim for a length that is a multiple of the original (e.g., 0.8x or 1.2x the original length).
    *   **Default**: `1.0` (targets the original conditioned tensor's length)
    *   **Range**: `0.5` to `2.0`
    *   **Usage**: If you find that squashing to the exact original length is too restrictive or not aggressive enough in controlling "burn-in," you can use this factor. A value less than 1.0 will squash towards a shorter vector, potentially reducing energy/burn-in more. A value greater than 1.0 will allow for a slightly longer vector than the original `cond` after squashing, which might be useful if the default squashing feels too dampening.

*   **NRS Effect Blend (`nrs_effect_blend`)**
    *   **Purpose**: Linearly interpolates between the original conditioned tensor (after initial rescaling for v-prediction models) and the fully processed NRS tensor. This allows you to "blend" the NRS effect with the standard CFG-like guidance.
    *   **Default**: `1.0` (applies 100% of the NRS effect)
    *   **Range**: `0.0` to `1.0`
    *   **Usage**: If the full NRS effect is too strong or introduces undesired changes, you can reduce this value to mix it with the more standard guidance. A value of `0.0` would effectively bypass the NRS modifications (Stretch, Skew, Squash), while `0.5` would be an even mix. This provides a global way to temper the overall NRS adjustments.

## Beginner How-To
1. Set Skew to your normal CFG Scale setting and Stretch to 1/2 your normal CFG Scale.
2. Set Squash to 0.0.
3. Test some outputs. Results should be similar in quality to CFG.
4. Adjust Skew up/down to change content and composition.
5. Adjust Stretch up/down to change strength of positive prompt aspects and colors.
6. Adjust Squash up to remove artifacts and color burn (these will tend to be replaced by additional details and elements).

**Tip**: You can experiment with negative values for Skew and Stretch as well, to see how the model is interpeting your negative prompt.

## Examples
| User | CFG | NRS |
|---|---|---|
| Mohnjiles from StabilityMatrix | ![CFG Example](Examples/mohnjiles_cfg.png) | ![NRS Example](Examples/mohnjiles_nrs.png) |
