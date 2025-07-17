import logging
import torch
import kornia
import math
from kornia.geometry.transform import build_laplacian_pyramid

def project(v0: torch.Tensor, v1: torch.Tensor):
    """Projects tensor v0 onto v1 and returns parallel and orthogonal components."""
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def build_image_from_pyramid(pyramid):
    """Reconstructs image from laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = kornia.geometry.pyrup(img) + pyramid[i]
    return img

def laplacian_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale=[1.0, 1.0],
    parallel_weights=None,
):
    """Applies laplacian guidance using laplacian pyramids."""
    levels = len(guidance_scale)
    if parallel_weights is None:
        parallel_weights = [1.0] * levels
    original_size = pred_cond.shape[-2:]
    pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels)
    pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)
    pred_guided_pyramid = []

    parameters = zip(
        pred_cond_pyramid,
        pred_uncond_pyramid,
        guidance_scale,
        parallel_weights
    )


    for idx, (p_cond, p_uncond, scale, par_weight) in enumerate(parameters):
        """Crop the padding area added by build_laplacian_pyramid"""
        level_size = (original_size[0] // (2 ** idx), original_size[1] // (2 ** idx))
        p_cond = p_cond[..., :level_size[0], :level_size[1]]
        p_uncond = p_uncond[..., :level_size[0], :level_size[1]]
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project(diff, p_cond)
        diff = par_weight * diff_parallel + diff_orthogonal
        p_guided = p_cond + (scale - 1) * diff
        pred_guided_pyramid.append(p_guided)
    pred_guided = build_image_from_pyramid(pred_guided_pyramid)

    return pred_guided.to(pred_cond.dtype)


def create_linear_guidance_scale(high_scale, low_scale, levels):
    """Creates linearly interpolated guidance scale array."""
    if levels == 1:
        return [high_scale]

    """Linear interpolation of guidance between levels."""
    scales = torch.linspace(high_scale, low_scale, levels).tolist()
    return scales

# Helper function for Adaptive Normalization
def _adaptive_normalize(x_final, cond, normalize_strength, epsilon_norm=1e-6):
    if normalize_strength > 0:
        dims_to_reduce = tuple(range(2, cond.ndim))

        mean_cond = torch.mean(cond, dim=dims_to_reduce, keepdim=True)
        var_cond = torch.var(cond, dim=dims_to_reduce, unbiased=False, keepdim=True)
        std_cond = torch.sqrt(var_cond + epsilon_norm)

        mean_x_final_orig = torch.mean(x_final, dim=dims_to_reduce, keepdim=True)
        var_x_final_orig = torch.var(x_final, dim=dims_to_reduce, unbiased=False, keepdim=True)
        std_x_final_orig = torch.sqrt(var_x_final_orig + epsilon_norm)

        # K defines the maximum factor by which the standard deviation can be changed.
        # A K of 10 means the target std dev will be at most 10x or at least 0.1x of the original.
        K = 10.0

        # Calculate the clamped target standard deviation
        target_std = torch.clamp(std_cond, min=std_x_final_orig / K, max=std_x_final_orig * K)

        normalized_x_final_temp = (x_final - mean_x_final_orig) / std_x_final_orig # std_x_final_orig already has epsilon
        denormalized_x_final = normalized_x_final_temp * target_std + mean_cond

        x_final = (1.0 - normalize_strength) * x_final + normalize_strength * denormalized_x_final
    return x_final

class NRS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "skew": ("FLOAT", {"default": 4.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                              "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                              "squash": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              "normalize_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, skew, stretch, squash, normalize_strength):
        def nrs(args, normalize_strength): # Added normalize_strength here
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            logging.debug(f"NRS.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}, Normalize: {normalize_strength}") # Added Normalize to log

            #rescale cfg has to be done on v-pred model output
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            logging.debug(f"NRS.nrs: generated cond and uncond")

            x_final = None
            match "v0.4.5":
                case "v1":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c = (u_dot_c / c_dot_c) * cond
                    u_rej_c = uncond - u_on_c
                    displaced = (cond - skew * u_rej_c)
                    logging.debug(f"NRS.nrs: displaced")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)
                    squashed = displaced * squash_scale
                    logging.debug(f"NRS.nrs: squashed")

                    # stretch turned vector towards cond based on stretch scale
                    sq_dot_c = torch.sum(squashed * cond, dim=-1, keepdim=True)
                    sq_on_c = (sq_dot_c / c_dot_c) * cond
                    x_final = squashed + sq_on_c * stretch
                    logging.debug(f"NRS.nrs: final")
                case "v2":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    displaced = cond + stretch * (cond - torch.clamp(u_dot_c / c_dot_c, min=0, max=1) * cond) - skew * u_rej_c
                    logging.debug(f"NRS.nrs: displaced & stretched")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)
                    x_final = displaced * squash_scale
                    logging.debug(f"NRS.nrs: final")
                case "v3":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    displaced = (cond - skew * u_rej_c)
                    logging.debug(f"NRS.nrs: displaced")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)

                    # stretch vector towards 2*len(cond) - len(u_on_c)
                    c_len = c_dot_c ** 0.5
                    stretch_scale = (1 - stretch) + stretch * (2 * c_len - u_on_c_mag)/c_len

                    x_final = displaced * squash_scale * stretch_scale
                    logging.debug(f"NRS.nrs: final")
                case "v4":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    x_final = (cond - squash * u_rej_c + stretch * cond * ((rej_dor_rej/c_dot_c) ** 0.5))
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.1":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    stretched = cond + stretch * cond * ((rej_dor_rej/c_dot_c) ** 0.5)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/sk_dot_sk) ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.2":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    proj_len = torch.sum(u_on_c * u_on_c, dim=-1, keepdim=True) ** 0.5
                    cond_len = c_dot_c ** 0.5
                    stretched = cond * (1 + stretch * torch.abs(cond_len - proj_len) / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.3":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    proj_len = torch.sum(u_on_c * u_on_c, dim=-1, keepdim=True) ** 0.5
                    cond_len = c_dot_c ** 0.5
                    stretched = cond * (1 + stretch * (cond_len - proj_len) / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.4":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    cond_len = c_dot_c ** 0.5
                    proj_diff = cond - u_on_c
                    proj_diff_len = torch.sum(proj_diff * proj_diff, dim=-1, keepdim=True) ** 0.5
                    stretched = cond * (1 + stretch * proj_diff_len / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.5":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    cond_len = c_dot_c ** 0.5
                    proj_diff = cond - u_on_c

                    # Amplify Cond based on length compared to projection of uncond
                    stretched = cond + (stretch * proj_diff)

                    # Skew/Steer Conf based on rejection of uncond on cond
                    skewed = stretched - skew * u_rej_c

                    # Squash final length back down to original length of cond
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale

            # Adaptive Normalization Logic
            x_final = _adaptive_normalize(x_final, cond, normalize_strength)

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
        
        m = model.clone()
        # Use a lambda to pass normalize_strength to the nrs function
        m.set_model_sampler_cfg_function(lambda args: nrs(args, normalize_strength), True)
        return (m, )

class NRSEpsilon:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skew": ("FLOAT", {"default": 4.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                "squash": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, skew, stretch, squash, normalize_strength):
        # It's important to import torch and logging if they are not already top-level in the file.
        # Assuming 'torch' and 'logging' are available in the scope from file-level imports.

        def nrs(args):
            cond = args["cond"]
            uncond = args["uncond"]
            # sigma = args["sigma"] # Not used here
            # x_orig = args["input"] # Not used here

            # Use logging if available, otherwise this line would need to be conditional or removed.
            # logging.debug(f"NRSEpsilon.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}, Normalize: {normalize_strength}")

            x_final = None

            # v0.4.5 steering logic (embedded directly)
            u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
            c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)

            epsilon_steering = 1e-6
            u_on_c_mag = (u_dot_c / (c_dot_c + epsilon_steering))

            u_on_c = u_on_c_mag * cond
            u_rej_c = uncond - u_on_c

            cond_len_sq = c_dot_c
            cond_len = torch.sqrt(cond_len_sq + epsilon_steering)

            proj_diff = cond - u_on_c

            stretched = cond + (stretch * proj_diff)
            skewed = stretched - skew * u_rej_c

            sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
            sk_dot_sk_sqrt = torch.sqrt(sk_dot_sk + epsilon_steering)

            squash_scale = (1 - squash) + squash * (cond_len / (sk_dot_sk_sqrt + epsilon_steering))
            x_final = skewed * squash_scale

            # Adaptive Normalization Logic
            x_final = _adaptive_normalize(x_final, cond, normalize_strength)

            return x_final

        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m,)


class NRSFDG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skew": ("FLOAT", {"default": 4.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                "squash": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale_high": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "guidance_scale_low": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "levels": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                    "step": 1
                }),
                "fdg_steps": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 50,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, skew, stretch, squash, normalize_strength, guidance_scale_high, guidance_scale_low, levels, fdg_steps):
        guidance_scale = create_linear_guidance_scale(guidance_scale_high, guidance_scale_low, levels)
        parallel_weights = [1.0] * (levels)

        def nrs_fdg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            x_orig = args["input"]
            sample_sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            step_limits = fdg_steps

            if uncond is not None:
                if step_limits >= (len(sample_sigmas) - 1):
                    step_limits = len(sample_sigmas) - 1
                if sigma.item() > sample_sigmas[step_limits].item():
                    guided = laplacian_guidance(
                        cond,
                        uncond,
                        guidance_scale,
                        parallel_weights
                    )
                    cond = guided

            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))

            #rescale cfg has to be done on v-pred model output
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

            u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
            c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
            u_on_c_mag = (u_dot_c / c_dot_c)
            u_on_c = u_on_c_mag * cond
            u_rej_c = uncond - u_on_c
            cond_len = c_dot_c ** 0.5
            proj_diff = cond - u_on_c

            # Amplify Cond based on length compared to projection of uncond
            stretched = cond + (stretch * proj_diff)

            # Skew/Steer Conf based on rejection of uncond on cond
            skewed = stretched - skew * u_rej_c

            # Squash final length back down to original length of cond
            sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
            squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
            x_final = skewed * squash_scale

            # Adaptive Normalization Logic
            x_final = _adaptive_normalize(x_final, cond, normalize_strength)

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

        m = model.clone()
        m.set_model_sampler_cfg_function(nrs_fdg, True)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
    "NRSEpsilon": NRSEpsilon,
    "NRSFDG": NRSFDG,
}