# Negative Rejection Steering (NRS) Documentation

## 1. Core Concept: What is Negative Rejection Steering?

Negative Rejection Steering (NRS) is an advanced technique used in image generation models to guide the creative process. It offers a more nuanced and composable alternative to the traditional linear interpolation method of Classifier Free Guidance (CFG).

The core idea behind NRS is to provide more precise control over how the generation process adheres to positive prompts while diverging from negative ones. Instead of a single "guidance scale" knob like CFG, NRS introduces multiple parameters that allow for finer adjustments to the image output.

### Advantages over Classifier Free Guidance (CFG):

*   **Decomposed Controls**: CFG typically offers a single scale to control guidance. NRS breaks this down into multiple parameters (Skew, Stretch, Squash, and Normalize Strength), offering more granular control over different aspects of the guidance.
*   **Improved Nuance**: By separating these controls, NRS allows for more sophisticated steering of the generation. For example, one can adjust the content and composition (Skew) independently of the prompt adherence strength (Stretch).
*   **Better Mathematical Basis**: NRS is designed with a more explicit mathematical foundation for how it steers the diffusion process, aiming for more predictable and intuitive results.
*   **Enhanced Output Quality**: With finer control, users can often achieve cooler, more detailed, or more artistically specific outputs that might be difficult to obtain with CFG alone, potentially reducing artifacts or "burn-in" effects.

NRS aims to empower users with more expressive capabilities in guiding image synthesis.

## 2. NRS Parameters (v0.4.5 Logic)

NRS replaces the single CFG scale with a set of distinct parameters. The behavior described here is based on the "v0.4.5" logic found in the `NRS/nodes_NRS.py` implementation. In this version:

1.  The conditioned tensor (`cond`) is first amplified based on its difference from the projection of the unconditioned tensor (`uncond`) onto it. This is controlled by the `Stretch` parameter.
2.  The resulting tensor is then skewed (steered) away from the direction of the rejection of `uncond` on `cond`. This is controlled by the `Skew` parameter.
3.  Finally, the length of this modified tensor is rescaled back towards the original length of `cond`, influenced by the `Squash` parameter.

Let's break down each parameter:

### Skew

*   **Purpose**: `Skew` primarily influences the *direction* of the generation process. It changes the content and composition of the image by steering the guidance.
*   **Mechanism (v0.4.5)**: After the initial stretch, the `Skew` parameter controls how much the tensor is displaced (skewed) in the direction perpendicular to the original `cond` tensor, specifically using the rejection of `uncond` on `cond` (`u_rej_c`). A higher `Skew` value means a stronger push away from what the model might generate based on the `uncond` (negative prompt) influence in that perpendicular aspect.
*   **Effect**: Modifying `Skew` can lead to significant changes in the elements depicted, their arrangement, and the overall artistic style. The README suggests that `Skew` is roughly comparable to the full CFG scale in some respects (e.g., `Skew = CFG_Scale`).

### Stretch

*   **Purpose**: `Stretch` primarily controls the *amplification* or strength of the positive prompt's representation in the image.
*   **Mechanism (v0.4.5)**: `Stretch` adjusts how much the `cond` tensor is amplified based on its difference from the projection of `uncond` on `cond` (`proj_diff = cond - u_on_c`). A higher `Stretch` value increases the magnitude of this amplification, making the features related to the positive prompt more pronounced.
*   **Effect**: Increasing `Stretch` tends to result in stronger adherence to the positive prompt, potentially leading to more vibrant colors and more defined features described in the prompt. The README suggests `Stretch` could be initially set to `0.5 * CFG_Scale`.

### Squash

*   **Purpose**: `Squash` is used to *normalize* the final guidance vector's magnitude, specifically by rescaling it towards the original length of the conditioned tensor (`cond`). This can help in reducing "burn-in" artifacts and oversaturation that might occur from aggressive Skew or Stretch values.
*   **Mechanism (v0.4.5)**: After skewing and stretching, the `Squash` parameter determines how much the resulting tensor's length is interpolated back towards the original length of `cond`. A `Squash` of 0.0 means no length normalization (relative to this step), while a `Squash` of 1.0 means the final vector will have the same length as the original `cond`, but its direction will be the result of the Skew and Stretch operations.
*   **Effect**: Increasing `Squash` can help to mitigate artifacts like excessive contrast or color burn, transforming these potential defects into more nuanced details and alternative interpretations of the prompt. It's often adjusted after finding good Skew/Stretch values, starting from 0.0 and increasing as needed.

## 3. Latest Change: Adaptive Normalization and `normalize_strength`

A recent key addition to NRS is the introduction of **Adaptive Normalization**, controlled by the `normalize_strength` parameter. This feature is available in both `NRS` and `NRSEpsilon` nodes.

### Adaptive Normalization

Adaptive Normalization is a technique applied *after* the core Skew, Stretch, and Squash operations. Its purpose is to further refine the generated tensor (`x_final`) by aligning its statistical properties (mean and variance) more closely with those of the original conditioned tensor (`cond`).

The process involves:
1.  Calculating the mean and variance of the `cond` tensor across its embedding dimensions.
2.  Calculating the mean and variance of the `x_final` tensor (output of Skew/Stretch/Squash).
3.  Normalizing `x_final` (by subtracting its mean and dividing by its standard deviation).
4.  Denormalizing this result using the mean and variance of `cond`.
5.  The final output is then an interpolation between the original `x_final` (after Skew/Stretch/Squash) and this adaptively normalized version, controlled by `normalize_strength`.

### `normalize_strength` Parameter

*   **Purpose**: This parameter controls the intensity of the Adaptive Normalization effect.
*   **Range**: Typically `0.0` to `1.0`.
    *   `0.0`: Adaptive Normalization is effectively off. The output is purely the result of Skew, Stretch, and Squash.
    *   `1.0`: The output is fully replaced by the adaptively normalized version (aligned with `cond`'s statistics).
    *   Values in between blend the two.
*   **Default Value**:
    *   In the `NRS` node, the default for `normalize_strength` is `0.0`.
    *   In the `NRSEpsilon` node, the default for `normalize_strength` is `1.0`.
*   **Effect**:
    *   Applying adaptive normalization can help in producing images that have a more consistent statistical profile with what the model expects for conditioned outputs.
    *   It may lead to improved coherence, better preservation of certain details from the `cond` guidance, or a reduction in unexpected tonal shifts or artifacts that might arise from aggressive steering.
    *   The process has been refined to be gentler: the adjustment to the signal's standard deviation is now clamped. This means it won't attempt to force the statistics to match if the difference is too large (e.g., the target standard deviation will not be more than a factor of K=10 times different from the original). This makes the normalization more robust against potential artifacts from drastic statistical shifts.
    *   The visual impact can vary depending on the model and other parameters, so experimentation is encouraged. For instance, if high Skew/Stretch values introduce undesirable color casts or intensity imbalances, `normalize_strength` might help to temper these effects by pulling the output's characteristics back towards the `cond` baseline.

## 4. `NRS` Node vs. `NRSEpsilon` Node

The implementation provides two ComfyUI nodes: `NRS` and `NRSEpsilon`. While both apply the same core v0.4.5 steering logic (Skew, Stretch, Squash) and Adaptive Normalization, they differ in how they process the inputs and what they output, suggesting different use cases.

### `NRS` Node

*   **Input Processing**: This node takes `model`, `skew`, `stretch`, `squash`, and `normalize_strength` as inputs. Internally, its `patch` method sets up an `nrs` function that expects arguments including `cond`, `uncond`, `sigma` (noise level), and `input` (original noisy latent, `x_orig`).
*   **Pre-Steering Calculations**: Before applying the Skew/Stretch/Squash logic, it performs calculations involving `x_orig` and `sigma` to derive internal representations of `cond` and `uncond`. This suggests it's designed to work within a sampling process where these adjustments based on the current noise level are necessary.
    *   `x = x_orig / (sigma * sigma + 1.0)`
    *   `cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)`
    *   `uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)`
*   **Output Calculation**: After the NRS steering and adaptive normalization (resulting in `x_final`), it calculates the return value as:
    `x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)`. This formula is typical for applying guided noise to an original latent in a diffusion step.
*   **Default `normalize_strength`**: `0.0`.
*   **Intended Use**: Likely intended as a general-purpose NRS application within a standard diffusion sampling loop where the model's noise prediction needs to be steered and then applied back to the noisy latents, taking `sigma` into account.

### `NRSEpsilon` Node

*   **Input Processing**: Similar inputs: `model`, `skew`, `stretch`, `squash`, and `normalize_strength`. Its `patch` method sets up an `nrs` function that expects `cond` and `uncond` directly.
*   **Pre-Steering Calculations**: It does *not* perform the `sigma`-based adjustments to `cond` and `uncond` seen in the `NRS` node. It uses the provided `cond` and `uncond` more directly for the steering logic.
*   **Output Calculation**: The `nrs` function directly returns the `x_final` (the result of Skew/Stretch/Squash and Adaptive Normalization). It does not perform the final recombination with `x_orig` and `sigma` that the `NRS` node does.
*   **Default `normalize_strength`**: `1.0`.
*   **Intended Use**: The name "Epsilon" suggests this node is designed to work more directly with noise predictions (epsilon). It's likely suitable for scenarios where:
    *   The `cond` and `uncond` inputs are already the model's direct noise predictions.
    *   The sampling framework handles the application of this steered noise (epsilon) to the latents separately.
    *   You want to apply NRS directly to epsilon values before they are used by the sampler's stepping function.

### When to Use Which?

*   Choose the **`NRS`** node for most standard ComfyUI workflows where you are patching a model and expect the steering to be integrated into a typical sampling process that handles noise application.
*   Choose the **`NRSEpsilon`** node if you are working with a workflow where you need to manipulate the raw noise (epsilon) predictions directly, and the sampler or subsequent nodes will handle the scaling and application of this steered epsilon. This might be more common in custom or advanced sampling setups.

The different default for `normalize_strength` (`0.0` for `NRS`, `1.0` for `NRSEpsilon`) is also a key practical difference to be aware of when switching between them.

## 5. Practical Usage Guide

This guide helps you get started with Negative Rejection Steering and provides tips for experimenting with its parameters to achieve desired image outcomes. These parameters are typically available as sliders in the ComfyUI interface when using the NRS nodes, thanks to the integration script (`scripts/negative_rejection_steering_script.py`).

### Initial Setup (Based on README suggestions)

1.  **Node Selection**: For most general use cases, start with the `NRS` node. If you have a specific need to manipulate epsilon (noise) values directly, consider `NRSEpsilon` (see section 4 for differences).
2.  **Baseline Settings (Approximating CFG)**:
    *   Set **`Skew`** to your usual CFG Scale value (e.g., if you use CFG 7.0, try `Skew` 7.0).
    *   Set **`Stretch`** to approximately half of your usual CFG Scale (e.g., if you use CFG 7.0, try `Stretch` 3.5).
        *   The README mentions a rough equivalence: `Stretch + 2 * Skew = 2 * CFG`. So, for `Skew = CFG`, `Stretch` would be `0`. However, the beginner guide suggests `Stretch = 0.5 * CFG`. This implies some flexibility. Starting with `Stretch` around `0.5 * Skew` is a reasonable approach.
    *   Set **`Squash`** to `0.0` initially.
    *   Set **`normalize_strength`** to its default for the chosen node (`0.0` for `NRS`, `1.0` for `NRSEpsilon`). If using the `NRS` node, start with `0.0`.

3.  **Generate Test Images**: Produce a few images with these baseline settings. The results should be somewhat comparable in quality and adherence to your prompt as you'd get with CFG.

### Experimenting and Fine-Tuning

Once you have a baseline, start adjusting the parameters:

*   **Adjusting `Skew`**:
    *   **Effect**: Changes the content, composition, and artistic style.
    *   **Action**: Increase `Skew` to steer the image more strongly, potentially introducing more diverse elements or a more stylized look. Decrease it for less aggressive steering.
    *   **Tip**: You can try negative `Skew` values to see how the model interprets steering *towards* the negative prompt's rejection characteristics, which can sometimes lead to interesting creative effects.

*   **Adjusting `Stretch`**:
    *   **Effect**: Amplifies or dampens the positive prompt's representation, affecting color intensity and feature definition.
    *   **Action**: Increase `Stretch` for stronger prompt adherence and more vivid outputs. Decrease it if the image feels over-processed or too literal.
    *   **Tip**: Negative `Stretch` values can also be experimented with, which might invert or subdue the positive prompt's influence in unusual ways.

*   **Adjusting `Squash`**:
    *   **Effect**: Normalizes the guidance magnitude, helping to reduce "burn-in," artifacts, and oversaturation.
    *   **Action**: If your images (especially with higher `Skew` or `Stretch`) show excessive contrast or color burn, gradually increase `Squash` (e.g., in steps of 0.1 or 0.25). This will rescale the steered vector towards the original `cond` length, often transforming artifacts into more subtle details or textural elements.
    *   **Tip**: `Squash` is best tuned *after* you've found interesting Skew/Stretch combinations.

*   **Adjusting `normalize_strength`**:
    *   **Effect**: Controls the adaptive normalization, aligning the output's statistical properties (mean, variance) with the `cond` tensor. Can improve coherence or temper extreme effects.
    *   **Action**:
        *   If using the `NRS` node (default `normalize_strength = 0.0`), try increasing it if your image feels tonally imbalanced or has artifacts not addressed by `Squash`.
        *   If using `NRSEpsilon` (default `normalize_strength = 1.0`), try decreasing it if you want less of this statistical alignment and more of the raw steered output.
    *   **Tip**: The impact of `normalize_strength` can be subtle or significant depending on the other parameters and the model. Experiment by toggling it between 0.0, 0.5, and 1.0 to see its effect. It can be particularly useful for taming "rogue" generations where colors or intensities become extreme.

### General Tips for Practical Use

*   **Iterative Refinement**: NRS encourages an iterative process. Make small adjustments to one parameter at a time to understand its impact.
*   **Prompting**: Clear and well-defined positive and negative prompts are still crucial. NRS provides more control over how these prompts are interpreted, but it doesn't replace good prompting.
*   **Sampler and Step Count**: The choice of sampler and the number of sampling steps can also interact with NRS settings. What works well for one sampler might need tweaking for another.
*   **XYZ Plotting**: Use tools like an XYZ plot (often available in UIs like ComfyUI or A1111 WebUI) to systematically test ranges of Skew, Stretch, Squash, and `normalize_strength` to discover interesting combinations and their effects on your images. The script `scripts/negative_rejection_steering_script.py` explicitly adds NRS parameters to such an XYZ grid if available.
*   **Subjectivity**: The "best" settings are subjective and depend on the desired artistic outcome. Don't be afraid to try unconventional values once you understand the basics.

## 6. Examples

Visual examples are often the best way to understand the impact of different NRS settings.

### Existing Examples

The `Examples/` directory in this repository contains some image comparisons:

*   **`Examples/mohnjiles_cfg.png`**: An image generated using standard Classifier Free Guidance.
*   **`Examples/mohnjiles_nrs.png`**: A comparable image generated using Negative Rejection Steering.

These examples, also featured in the main `README.md`, showcase the potential differences in output quality and style that NRS can offer.

### Showcasing `normalize_strength`

To fully appreciate the effect of the new `normalize_strength` parameter, it would be beneficial to generate a series of images where `Skew`, `Stretch`, and `Squash` are kept constant, while `normalize_strength` is varied (e.g., 0.0, 0.5, 1.0). This would help illustrate how it modifies the output, potentially:

*   Tempering "burn-in" or excessive contrast.
*   Subtly shifting color balance or tonal range.
*   Improving overall coherence or detail preservation.

Users are encouraged to create their own comparison sets using the XYZ plotting tools mentioned in the Practical Usage Guide to best understand how `normalize_strength` interacts with other parameters and their specific prompts.
