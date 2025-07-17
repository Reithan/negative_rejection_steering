# Negative Rejection Steering (NRS) Documentation

## 1. Introduction

### 1.1. What is Negative Rejection Steering?
Negative Rejection Steering (NRS) is an advanced technique for guiding the output of generative AI models, offering a more nuanced and powerful alternative to traditional methods like Classifier-Free Guidance (CFG). Where CFG provides a single knob to control prompt adherence, NRS decomposes this control into distinct parameters: **Skew**, **Stretch**, and **Squash**. This allows for fine-grained adjustments to the content, composition, and quality of the generated output.

### 1.2. Advantages over CFG
*   **Decomposed Control**: Instead of a single guidance scale, NRS provides separate controls for steering the *direction* (`Skew`), *amplifying* the prompt (`Stretch`), and *normalizing* the output (`Squash`).
*   **Enhanced Creativity**: By manipulating these parameters, users can explore a wider creative space, achieving unique artistic styles and compositions.
*   **Improved Quality**: NRS can help mitigate common artifacts like "burn-in" and oversaturation by transforming them into desirable details.

### 1.3. Core Parameters
*   **Skew**: Influences the *content and composition* of the image by steering the generation process away from the negative prompt's influence.
*   **Stretch**: Controls the *strength and amplification* of the positive prompt, affecting color and feature intensity.
*   **Squash**: *Normalizes the output's magnitude*, reducing artifacts and "burn-in."

---

## 2. The NRS Nodes

This implementation of NRS provides three distinct ComfyUI nodes.

### 2.1. `NRS`
This is the standard, general-purpose NRS node. It is designed to work within a typical diffusion sampling loop. Before applying NRS, it recalculates `cond` and `uncond` based on the original input (`x_orig`) and the current noise level (`sigma`). This makes it compatible with samplers that expect v-prediction-like inputs.

*   **Processing Steps**:
    1.  Receives `cond`, `uncond`, `sigma`, and `x_orig` from the sampler.
    2.  Re-derives `cond` and `uncond` based on `sigma`.
    3.  Applies the selected NRS version (`v0.4.5` by default) to the re-derived tensors.
    4.  Applies Adaptive Normalization.
    5.  Returns the final steered tensor, which the sampler then uses to guide the diffusion step.
*   **Inputs**: `model`, `version`, `skew`, `stretch`, `squash`, `normalize_strength`, `adaptive_k`
*   **Default `normalize_strength`**: `0.0`
*   **Use Case**: Recommended for most standard workflows in ComfyUI.

### 2.2. `NRSEpsilon`
This node is designed to work directly with the model's noise predictions (epsilon, `eps`). It applies the NRS logic directly to the `cond` and `uncond` tensors it receives, without the `sigma`-based adjustments that the `NRS` node performs.

*   **Processing Steps**:
    1.  Receives `cond` and `uncond` from the sampler (these are typically the raw `eps` predictions).
    2.  Applies the `v0.4.5` NRS algorithm directly to these tensors.
    3.  Applies Adaptive Normalization.
    4.  Returns the final steered `eps` tensor.
*   **Inputs**: `model`, `skew`, `stretch`, `squash`, `normalize_strength`, `adaptive_k`
*   **Default `normalize_strength`**: `1.0`
*   **Use Case**: Advanced workflows where you need to manipulate the raw `epsilon` predictions before they are used by the sampler.

### 2.3. `NRSFDG`
This node combines NRS with Frequency Domain Guidance (FDG). FDG applies guidance at different frequency levels of the image, allowing for more detailed control over textures and structures. The FDG is applied only during the initial steps of the sampling process, after which the standard NRS guidance takes over.

*   **Processing Steps**:
    1.  For the first `fdg_steps` of the sampling process, it applies `laplacian_guidance` to `cond` and `uncond`.
    2.  For all steps, it performs the same `sigma`-based adjustments as the `NRS` node.
    3.  It then applies the `v0.4.5` NRS algorithm.
    4.  Applies Adaptive Normalization.
    5.  Returns the final steered tensor.
*   **Inputs**: All `NRS` inputs, plus `guidance_scale_high`, `guidance_scale_low`, `levels`, `fdg_steps`.
*   **Use Case**: Advanced workflows where you want to combine the steering capabilities of NRS with the frequency-based control of FDG for enhanced detail and texture.

---

## 3. Advanced Features

### 3.1. Adaptive Normalization
Adaptive Normalization is a feature that refines the output of the NRS calculations by aligning its statistical properties (mean and variance) with those of the original `cond` tensor. This can improve coherence and reduce artifacts introduced by aggressive steering.

The process works as follows:
1.  The mean and variance of the `cond` tensor and the NRS-steered tensor (`x_final`) are calculated.
2.  `x_final` is normalized by subtracting its mean and dividing by its standard deviation.
3.  This normalized tensor is then denormalized using the mean and variance of the original `cond` tensor.
4.  The final output is an interpolation between the original `x_final` and this newly normalized version.

*   **`normalize_strength`**: Controls the strength of this interpolation. At `0.0`, Adaptive Normalization is off. At `1.0`, the output is fully replaced by the statistically-aligned version. The default is `0.0` for the `NRS` node and `1.0` for the `NRSEpsilon` node.
*   **`adaptive_k`**: A clamping factor that limits how much the standard deviation of the output can be changed during normalization. For example, a `k` of 10 means the target standard deviation will be at most 10x or at least 0.1x of the original. This prevents extreme adjustments and improves stability. The default is `10.0`.

### 3.2. NRS Algorithm Versions
This implementation includes several versions of the NRS algorithm, each with a slightly different mathematical approach. The recommended and most recent version is **`v0.4.5`**. The other versions are available for experimentation and backward compatibility. Below is a brief overview of their evolution.

*   **`v1`**: The original implementation. It displaces the conditional tensor by the rejection of the unconditional tensor and then applies squash and stretch.
*   **`v2`**: Introduces clamping on the unconditional projection magnitude to prevent over-correction and uses a different method for stretching.
*   **`v3`**: A different approach to stretching, where the stretch scale is calculated based on the lengths of the conditional tensor and its projection of the unconditional tensor.
*   **`v4`**: A version by DGSpitzer. It uses a different approach to squash and stretch, with stretch being influenced by the magnitude of the rejection.
*   **`v0.4.1`**: Separates the stretch and skew operations. First, it stretches the conditional tensor, then skews the result.
*   **`v0.4.2`**: A new stretching formula based on the absolute difference in length between the conditional tensor and its projection of the unconditional tensor.
*   **`v0.4.3`**: Similar to `v0.4.2`, but removes the absolute value from the length difference calculation, allowing for negative stretching.
*   **`v0.4.4`**: Stretches based on the length of the difference between the conditional tensor and its projection of the unconditional tensor.
*   **`v0.4.5`**: The current recommended version. It simplifies the `v0.4.4` logic for a more direct and stable implementation of stretching and skewing.

---

## 4. Practical Usage Guide

### 4.1. Initial Setup
1.  **Start with the `NRS` node** for most workflows.
2.  **Set `version` to `v0.4.5`**, which is the latest and recommended algorithm.
3.  **Establish a baseline** by approximating your usual CFG settings:
    *   Set **`Skew`** to your typical CFG scale (e.g., `7.0`).
    *   Set **`Stretch`** to half of your `Skew` value (e.g., `3.5`).
    *   Set **`Squash`** to `0.0`.
    *   Leave **`normalize_strength`** at its default of `0.0` for the `NRS` node.
4.  Generate a test image. The result should be comparable to what you would expect from a standard CFG setup.

### 4.2. Fine-Tuning Workflow
Once you have a baseline, adjust the parameters iteratively:
*   **To change content and composition**, adjust **`Skew`**. Increasing it will steer the image more strongly.
*   **To change color intensity and prompt adherence**, adjust **`Stretch`**. Increasing it will amplify the prompt's influence.
*   **If you see "burn-in" or oversaturation**, increase **`Squash`** in small increments (e.g., `0.1` to `0.25`). This will normalize the output and often transform artifacts into details.
*   **If the image feels tonally imbalanced**, increase **`normalize_strength`**. This can help to correct for extreme color shifts or contrast introduced by high `Skew` or `Stretch` values.

### 4.3. Using the Script (for A1111/Forge)
The `scripts/negative_rejection_steering_script.py` provides a user interface for controlling NRS parameters within the A1111/Forge web UI.
*   Enable the script in the "Scripts" section of the UI.
*   The NRS parameters (`Skew`, `Stretch`, `Squash`, and `normalize_strength`) will appear as sliders.
*   The script also adds these parameters to the XYZ plot feature, which is highly recommended for systematically testing different combinations and discovering their effects.

---

## 5. Examples
The `Examples/` directory contains images demonstrating the difference between CFG and NRS. Users are encouraged to generate their own comparisons using the XYZ plot to explore the effects of different parameter combinations.
