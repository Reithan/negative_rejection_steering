import logging
import torch

class NRS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "skew": ("FLOAT", {"default": 4.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                              "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                              "squash": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, skew, stretch, squash):
        def nrs(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            logging.debug(f"NRS.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}")

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

            return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)
        
        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m, )

class NRSEpsilon:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "skew": ("FLOAT", {"default": 4.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                              "stretch": ("FLOAT", {"default": 2.0, "min": -30.0, "max": 30.0, "step": 0.01}),
                              "squash": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, skew, stretch, squash):
        def nrs(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]

            logging.debug(f"NRS.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}")

            #rescale cfg has to be done on v-pred model output
            x = x_orig # Changed for EpsilonPred
            cond = args["cond"]
            uncond = args["uncond"]
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

            return x_final # Changed for EpsilonPred

        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
    "NRSEpsilon": NRSEpsilon,
}