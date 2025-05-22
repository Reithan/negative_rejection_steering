Negative Rejection Steering (NRS) is a technique used in diffusion-based image generation as an alternative to the more common Classifier Free Guidance (CFG). While CFG provides a single "knob" for controlling the generation process, NRS aims to offer a more nuanced and composable approach. It replaces CFG's linear interpolation with a multi-faceted steering mechanism, giving users finer control over the final output. This is achieved by decomposing the guidance into three distinct parameters: Skew, Stretch, and Squash, which allow for more precise adjustments to the content, composition, and overall fidelity of the generated image, moving beyond the simpler amplification provided by CFG. The goal is to provide a better mathematical basis for guiding the diffusion process, leading to potentially higher quality and more creatively directed results.

## Core Components and Parameters

NRS operates through three primary parameters that offer granular control over the image generation:

*   **Skew**: This parameter adjusts the 'direction' of the generation. Modifying Skew can lead to changes in the content and composition of the resulting image.
    *   Default value: `4.0`
    *   Typical range: `-30.0` to `30.0`

*   **Stretch**: This parameter controls the 'amplification' of the generation process. Increasing Stretch typically results in a stronger representation of the positive prompt aspects and more vibrant colors.
    *   Default value: `2.0`
    *   Typical range: `-30.0` to `30.0`

*   **Squash**: This parameter 'normalizes' the resulting guidance back towards the original amplitude of the conditioned tensor. It is primarily used to remove 'burn-in' artifacts and excessive color saturation that can occur with high Skew or Stretch values. These defects are often transformed into alternative details and elements in the image.
    *   Default value: `0.0`
    *   Typical range: `0.0` to `1.0` (representing 0% to 100% squashing)

The README suggests that for users familiar with CFG, a starting point for NRS could be setting Skew equal to their usual CFG scale and Stretch to half of that value, with Squash initially at 0.0. Negative values for Skew and Stretch can also be explored to understand how the model interprets the negative prompt.

## Mathematical Intuition (v0.4.5 Implementation)

The 'v0.4.5' implementation of NRS, as seen in `NRS/nodes_NRS.py`, applies these parameters through a series of vector operations on the conditioned (`cond`) and unconditioned (`uncond`) tensors. Here's a breakdown:

1.  **Calculating Key Vectors**:
    *   **Projection of `uncond` onto `cond` (`u_on_c`)**:
        *   `u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)` (Dot product)
        *   `c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)` (Squared magnitude of `cond`)
        *   `u_on_c = (u_dot_c / c_dot_c) * cond`
        *   This vector, `u_on_c`, represents the component of the unconditioned guidance that lies in the direction of the conditioned guidance.

    *   **Rejection of `uncond` from `cond` (`u_rej_c`)**:
        *   `u_rej_c = uncond - u_on_c`
        *   This vector, `u_rej_c`, is crucial for the Skew operation. It is perpendicular (orthogonal) to the `cond` tensor and represents the aspects of the `uncond` tensor that are not aligned with `cond`. Steering with or against this vector allows modification of the output in directions independent of the primary positive conditioning.

    *   **Projection Difference (`proj_diff`)**:
        *   `proj_diff = cond - u_on_c`
        *   This vector measures the difference between the `cond` tensor and the component of `uncond` that is aligned with `cond`. It effectively shows how `cond` already diverges from `uncond`'s influence along `cond`'s own direction. A larger `proj_diff` means `cond` is already quite distinct from `uncond` in its primary direction.

2.  **Applying NRS Parameters**: The NRS algorithm then modifies the `cond` tensor in three conceptual steps, aligning with the README's description:

    *   **Stretching**:
        *   `stretched = cond + (stretch * proj_diff)`
        *   The conditioned tensor `cond` is amplified by adding a scaled version of `proj_diff`. The `stretch` parameter controls this amplification. A positive `stretch` value pushes the `cond` tensor further away from `u_on_c` (the part of `uncond` that was aligned with `cond`), effectively boosting the unique aspects of `cond` relative to `uncond` along `cond`'s original direction. This corresponds to the "Stretching" step in the README, aimed at strengthening the positive prompt's representation.

    *   **Skewing**:
        *   `skewed = stretched - skew * u_rej_c`
        *   The `stretched` tensor is then adjusted by subtracting `u_rej_c` (the rejection vector) scaled by the `skew` parameter. Because `u_rej_c` is perpendicular to the original `cond` vector, this operation "skews" or steers the tensor in a direction orthogonal to `cond`. A positive `skew` value moves the tensor away from the direction of `u_rej_c`,cję_rej_c`, effectively reducing the influence of the unconditioned prompt's off-axis components. This is the "Skewing" step from the README, designed to alter content and composition by navigating away from `uncond`'s influence in a manner that is perpendicular to `cond`.

    *   **Squashing**:
        *   `cond_len = c_dot_c ** 0.5` (Original length of `cond`)
        *   `sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)` (Squared length of the `skewed` tensor)
        *   `squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)`
        *   `x_final = skewed * squash_scale`
        *   This final step rescales the `skewed` tensor. The `squash` parameter determines how close the final vector's length will be to the original `cond` tensor's length. A `squash` of 0.0 leaves the length of `skewed` unchanged, while a `squash` of 1.0 attempts to restore the length of the original `cond` tensor. This "Squashing" step, as described in the README, normalizes the amplitude of the guided vector, helping to prevent artifacts and "burn-in" by ensuring the final guidance signal isn't excessively strong.

The order of operations in the 'v0.4.5' code (calculating `stretched`, then `skewed`, then applying `squash_scale`) directly implements these three conceptual steps. This approach allows for a decomposed and more intuitive control over the guidance process compared to a single CFG scale.

## Integration into Image Generation Pipelines

Negative Rejection Steering can be integrated into image generation pipelines, such as those used by Web UIs like A1111/Forge or node-based systems like ComfyUI.

**1. Script-Based Integration (e.g., A1111/Forge WebUI):**

The `scripts/negative_rejection_steering_script.py` file provides an example of how NRS can be integrated as a script.

*   **`NRSScript` Class**: This class inherits from `scripts.Script` (a common base class for extensions in such UIs). It's responsible for:
    *   Defining the title and visibility of the script in the UI.
    *   Storing the NRS parameter values (enabled, skew, stretch, squash).

*   **UI Elements**: The `ui` method uses the Gradio library (`gr`) to create the user interface components:
    *   An accordion section to group the NRS settings.
    *   A checkbox (`gr.Checkbox`) to enable or disable NRS.
    *   Sliders (`gr.Slider`) for the `Skew`, `Stretch`, and `Squash` parameters, allowing users to adjust their values within defined ranges and steps.
    *   These UI elements are linked to the script's internal variables.

*   **`process_before_every_sampling` Method**: This is a critical method that hooks into the image generation lifecycle.
    *   It's called before each sampling step.
    *   It first retrieves the NRS parameter values set by the user through the UI or from other sources like XYZ plot configurations.
    *   Crucially, it **clones the U-Net model** (`unet = p.sd_model.forge_objects.unet.clone()`). This is important to ensure that the original model remains unchanged and that NRS modifications are applied to a fresh copy for each generation or if NRS is disabled.
    *   If NRS is enabled, it instantiates the `NRS` class (from `NRS/nodes_NRS.py`) and calls its `patch` method. The `patch` method modifies the U-Net's sampler CFG function to incorporate the NRS logic with the current Skew, Stretch, and Squash values.
    *   The modified U-Net is then assigned back to the processing pipeline (`p.sd_model.forge_objects.unet = unet`).
    *   If NRS is disabled, the script ensures the original, unpatched U-Net is used.
    *   It also updates `p.extra_generation_params` to include the NRS settings, so they are saved in the image metadata.

**2. Node-Based Integration (e.g., ComfyUI):**

The `NRS/nodes_NRS.py` file itself is structured to facilitate integration into node-based image generation systems like ComfyUI.

*   **`NRS` Class as a Node**: The `NRS` class in `nodes_NRS.py` is designed like a custom node.
    *   `INPUT_TYPES`: This class method defines the inputs the node expects (the model, skew, stretch, and squash parameters with their types, defaults, and ranges).
    *   `RETURN_TYPES`: Defines what the node outputs (a modified "MODEL").
    *   `FUNCTION = "patch"`: Specifies that the `patch` method is the core logic of this node.
    *   `CATEGORY = "advanced/model"`: Determines where the node will appear in the UI of a node-based system.

*   **`NODE_CLASS_MAPPINGS`**:
    *   `NODE_CLASS_MAPPINGS = { "NRS": NRS }`
    *   This dictionary is a standard convention in ComfyUI and similar systems. It maps a user-friendly name for the node ("NRS") to the actual class (`NRS`) that implements its functionality. This allows the system to discover and register the custom node, making NRS available as a draggable component in a visual workflow.

This dual approach—a script for direct UI integration and a node class for modular pipeline construction—allows NRS to be used in a variety of image generation setups, providing flexibility for different user preferences and workflows.

## Conclusion and Getting Started

Negative Rejection Steering (NRS) offers a sophisticated alternative to traditional Classifier Free Guidance by providing three distinct "knobs"—Skew, Stretch, and Squash—for more granular control over the image generation process. This decomposed approach allows for nuanced adjustments to content, composition, color representation, and artifact reduction, ultimately enabling the creation of more refined and "cooler outputs," as highlighted in the project's README. By moving beyond a single guidance scale, NRS empowers users to steer their creations with greater precision and artistic intent.

For those new to NRS and looking for practical first steps, the **"Beginner How-To"** section in the main `README.md` file of this repository is an excellent starting point. It provides clear guidance on how to initially set the Skew, Stretch, and Squash parameters based on familiar CFG settings and then how to adjust them to explore their unique effects on the output. Experimenting with these parameters is key to unlocking the full potential of NRS.
