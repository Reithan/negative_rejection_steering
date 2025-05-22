import logging
import torch

class NRS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "skew": ("FLOAT", {"default": 4.0, "min": -30.0, "max": 30.0, "step": 0.001}),
                              "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.001}),
                              "squash": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              # Modulates the intensity of the rejection vector (u_rej_c) used in Skewing.
                              "rejection_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                              # Modulates the intensity of the projection difference vector (proj_diff) used in Stretching.
                              "projection_diff_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                              # Adjusts the target length for Squashing, as a factor of original cond_len.
                              "squash_target_length_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                              # Blends the final NRS effect with the original (rescaled) conditioned tensor. 1.0 is full NRS.
                              "nrs_effect_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, skew, stretch, squash, rejection_strength, projection_diff_strength, squash_target_length_factor, nrs_effect_blend):
        def nrs(args, sk, st, sq, rs, pds, stlf, neb):
            cond_input = args["cond"]
            uncond_input = args["uncond"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond_input.ndim - 1))
            x_orig = args["input"]

            logging.debug(f"NRS.nrs: Skew: {sk}, Stretch: {st}, Squash: {sq}, RejStr: {rs}, ProjDiffStr: {pds}, SquashTargetLenFactor: {stlf}, NRSEffectBlend: {neb}")

            #rescale cfg has to be done on v-pred model output
            x = x_orig / (sigma * sigma + 1.0)
            cond = ((x - (x_orig - cond_input)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            uncond = ((x - (x_orig - uncond_input)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
            logging.debug(f"NRS.nrs: generated cond and uncond")

            x_final = None
            match "v0.4.5":
                case "v1":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c = (u_dot_c / c_dot_c) * cond
                    u_rej_c = uncond - u_on_c
                    displaced = (cond - sk * u_rej_c)
                    logging.debug(f"NRS.nrs: displaced")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * ((c_dot_c/d_len_sq) ** 0.5)
                    squashed = displaced * squash_scale
                    logging.debug(f"NRS.nrs: squashed")

                    # stretch turned vector towards cond based on stretch scale
                    sq_dot_c = torch.sum(squashed * cond, dim=-1, keepdim=True)
                    sq_on_c = (sq_dot_c / c_dot_c) * cond
                    x_final = squashed + sq_on_c * st
                    logging.debug(f"NRS.nrs: final")
                case "v2":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    displaced = cond + st * (cond - torch.clamp(u_dot_c / c_dot_c, min=0, max=1) * cond) - sk * u_rej_c
                    logging.debug(f"NRS.nrs: displaced & stretched")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * ((c_dot_c/d_len_sq) ** 0.5)
                    x_final = displaced * squash_scale
                    logging.debug(f"NRS.nrs: final")
                case "v3":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    displaced = (cond - sk * u_rej_c)
                    logging.debug(f"NRS.nrs: displaced")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * ((c_dot_c/d_len_sq) ** 0.5)

                    # stretch vector towards 2*len(cond) - len(u_on_c)
                    c_len = c_dot_c ** 0.5
                    stretch_scale = (1 - st) + st * (2 * c_len - u_on_c_mag)/c_len

                    x_final = displaced * squash_scale * stretch_scale
                    logging.debug(f"NRS.nrs: final")
                case "v4":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    x_final = (cond - sq * u_rej_c + st * cond * ((rej_dor_rej/c_dot_c) ** 0.5))
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.1":
                    u_dot_c = torch.sum(uncond * cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(cond * cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * cond
                    u_rej_c = uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    stretched = cond + st * cond * ((rej_dor_rej/c_dot_c) ** 0.5)
                    skewed = stretched - sk * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * ((c_dot_c/sk_dot_sk) ** 0.5)
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
                    stretched = cond * (1 + st * torch.abs(cond_len - proj_len) / cond_len)
                    skewed = stretched - sk * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * cond_len / (sk_dot_sk ** 0.5)
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
                    stretched = cond * (1 + st * (cond_len - proj_len) / cond_len)
                    skewed = stretched - sk * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * cond_len / (sk_dot_sk ** 0.5)
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
                    stretched = cond * (1 + st * proj_diff_len / cond_len)
                    skewed = stretched - sk * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - sq) + sq * cond_len / (sk_dot_sk ** 0.5)
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

                    # Amplify Cond based on length compared to projection of uncond, modulated by projection_diff_strength (pds)
                    stretched = cond + (st * (proj_diff * pds))

                    # Skew/Steer Conf based on rejection of uncond on cond, modulated by rejection_strength (rs)
                    skewed = stretched - sk * (u_rej_c * rs)

                    # Squash final length back down towards a target length derived from original cond_len and squash_target_length_factor (stlf)
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    target_cond_len = cond_len * stlf # Apply factor to target length
                    squash_scale = (1 - sq) + sq * target_cond_len / (sk_dot_sk ** 0.5)
                    x_final_nrs_core = skewed * squash_scale
                    
                    # Blend NRS effect with original (rescaled) cond using nrs_effect_blend (neb)
                    x_final = (1 - neb) * cond + neb * x_final_nrs_core


            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
        
        m = model.clone()
        m.set_model_sampler_cfg_function(lambda args: nrs(args, skew, stretch, squash, rejection_strength, projection_diff_strength, squash_target_length_factor, nrs_effect_blend), True)
        return (m, )

NODE_CLASS_MAPPINGS = {
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

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
        
        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
}