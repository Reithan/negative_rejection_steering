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

class FDGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
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
    
    def patch(self, model, guidance_scale_high, guidance_scale_low, levels, fdg_steps):
        
        """Create guidance scales and parallel weights arrays"""
        guidance_scale = create_linear_guidance_scale(guidance_scale_high, guidance_scale_low, levels)
        parallel_weights = [1.0] * (levels)
        def fdg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = guidance_scale_high if math.isclose(args["cond_scale"], 1.0) else args["cond_scale"]
            sample_sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            sigma = args["sigma"]
            step_limits = fdg_steps
            """Use CFG after limited FDG application for early steps."""
            if uncond is not None:
                if step_limits >= (len(sample_sigmas) - 1):
                    step_limits = len(sample_sigmas) - 1
                if sigma.item() > sample_sigmas[step_limits].item():
                    return laplacian_guidance(
                        cond,
                        uncond,
                        guidance_scale,
                        parallel_weights
                    )
                else: 
                    cond = uncond + (cond - uncond) * cond_scale
                    return cond
            else:
                return cond
        
        m = model.clone()
        m.set_model_sampler_cfg_function(fdg_function, disable_cfg1_optimization=True)
        return (m,)

