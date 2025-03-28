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
