import inspect
import logging
import torch
from enum import Enum, auto
from typing import Any


class PredictionType(Enum):
    EPS     = auto()   # ε-prediction
    V       = auto()   # v-prediction
    X0      = auto()   # x₀-prediction
    UNKNOWN = auto()   # couldn’t detect / new scheduler

_RAW_TO_ENUM = {
    "eps":          PredictionType.EPS,
    "epsilon":      PredictionType.EPS,
    "v":            PredictionType.V,
    "v_prediction": PredictionType.V,
    "x0":           PredictionType.X0,
    "sample":       PredictionType.X0,
}


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

    def _get_pred_type(self, model) -> PredictionType:
        """
        In order to support Comfy, Forge, and possibly other models
        and various loaders.
        Walk common wrappers until we find something that looks like a
        prediction-type flag, then map it to the enum.
        Defaults to EPS if all else fails.
        """
        def _canon(p):
            if p is None:
                return ""
            if isinstance(p, bytes):
                p = p.decode(errors="ignore")
            if isinstance(p, Enum):
                p = p.name
            return str(p).strip().lower()

        # Breadth-first search through a few well-known wrappers.
        queue = [model]
        visited = set()

        while queue:
            obj = queue.pop(0)
            if id(obj) in visited:
                continue
            visited.add(id(obj))

            # 1) direct hit on this object ---------------------------------
            p = _canon(getattr(obj, "parameterization", None))             # k-diffusion
            if p:
                return _RAW_TO_ENUM.get(p, PredictionType.UNKNOWN)

            p = _canon(getattr(getattr(obj, "config", None), "prediction_type", None))  # diffusers
            if p:
                return _RAW_TO_ENUM.get(p, PredictionType.UNKNOWN)

            p = _canon(getattr(obj, "prediction_type", None))              # rare misc
            if p:
                return _RAW_TO_ENUM.get(p, PredictionType.UNKNOWN)

            # 2) enqueue child containers we care about -------------------
            for attr in ("model_sampling", "model", "diffusion_model", "scheduler"):
                child = getattr(obj, attr, None)
                if child is not None:
                    queue.append(child)

        # 3) default ------------------------------------------------------
        return PredictionType.EPS

   
    def _pre_scale_conditioning(self, x_orig, sigma, cond, uncond):
        x_div = None
        eps_cond = cond
        eps_uncond = uncond
        if self.__pred_type == PredictionType.V:
            # v → ε conversion
            logging.debug(f"NRS._pre_scale_conditioning: generating x_div, cond, and uncond for v-pred")
            sigma2_1 = (sigma ** 2 + 1.0)
            x_div = x_orig / sigma2_1
            root = sigma2_1.sqrt()

            eps_cond = ((x_div - (x_orig - cond)) * root) / (sigma)
            eps_uncond = ((x_div - (x_orig - uncond)) * root) / (sigma)
        elif self.__pred_type == PredictionType.EPS:
            logging.debug(f"NRS._pre_scale_conditioning: already in eps, no pre-scale needed")
            pass  # already in ε space
        elif self.__pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._pre_scale_conditioning: x0-prediction not supported yet.")
        else:
            raise RuntimeError("NRS._pre_scale_conditioning: Could not determine prediction type for this model.")
        
        return x_div, eps_cond, eps_uncond

    def _post_scale_conditioning(self, x_orig, x_div, x_final, sigma):
        if self.__pred_type == PredictionType.V:
            # ε → v conversion
            root = (sigma ** 2 + 1).sqrt()

            logging.debug(f"NRS._post_scale_conditioning: generating cfg_result for v-pred")
            return x_orig - (x_div - x_final * sigma / root)
        elif self.__pred_type == PredictionType.EPS:
            # already in ε space
            logging.debug(f"NRS._post_scale_conditioning: already in eps, no post-scale needed")
            return x_final
        elif self.__pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._post_scale_conditioning: x0-prediction not supported yet.")
        else:
            raise RuntimeError("NRS._post_scale_conditioning: Could not determine prediction type for this model.")
        
    def patch(self, model, skew, stretch, squash):
        self.__pred_type = self._get_pred_type(model) if not hasattr(self, "__pred_type") else self.__pred_type

        def nrs(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            x_orig = args["input"]
            self.__pred_type = self.__pred_type if self.__pred_type is not None else self._get_pred_type(model)

            logging.debug(f"NRS.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}")
            
            x_div, eps_cond, eps_uncond = self._pre_scale_conditioning(x_orig, sigma, cond, uncond)

            x_final = None
            match "v0.5.0":
                case "v1":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c = (u_dot_c / c_dot_c) * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    displaced = (eps_cond - skew * u_rej_c)
                    logging.debug(f"NRS.nrs: displaced")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)
                    squashed = displaced * squash_scale
                    logging.debug(f"NRS.nrs: squashed")

                    # stretch turned vector towards cond based on stretch scale
                    sq_dot_c = torch.sum(squashed * eps_cond, dim=-1, keepdim=True)
                    sq_on_c = (sq_dot_c / c_dot_c) * eps_cond
                    x_final = squashed + sq_on_c * stretch
                    logging.debug(f"NRS.nrs: final")
                case "v2":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    displaced = eps_cond + stretch * (eps_cond - torch.clamp(u_dot_c / c_dot_c, min=0, max=1) * eps_cond) - skew * u_rej_c
                    logging.debug(f"NRS.nrs: displaced & stretched")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)
                    x_final = displaced * squash_scale
                    logging.debug(f"NRS.nrs: final")
                case "v3":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    displaced = (eps_cond - skew * u_rej_c)
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
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    x_final = (eps_cond - squash * u_rej_c + stretch * eps_cond * ((rej_dor_rej/c_dot_c) ** 0.5))
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.1":
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    stretched = eps_cond + stretch * eps_cond * ((rej_dor_rej/c_dot_c) ** 0.5)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/sk_dot_sk) ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.2":
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    proj_len = torch.sum(u_on_c * u_on_c, dim=-1, keepdim=True) ** 0.5
                    cond_len = c_dot_c ** 0.5
                    stretched = eps_cond * (1 + stretch * torch.abs(cond_len - proj_len) / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.3":
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    proj_len = torch.sum(u_on_c * u_on_c, dim=-1, keepdim=True) ** 0.5
                    cond_len = c_dot_c ** 0.5
                    stretched = eps_cond * (1 + stretch * (cond_len - proj_len) / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.4":
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    cond_len = c_dot_c ** 0.5
                    proj_diff = eps_cond - u_on_c
                    proj_diff_len = torch.sum(proj_diff * proj_diff, dim=-1, keepdim=True) ** 0.5
                    stretched = eps_cond * (1 + stretch * proj_diff_len / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.5":
                    u_dot_c = torch.sum(eps_uncond * eps_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(eps_cond * eps_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * eps_cond
                    u_rej_c = eps_uncond - u_on_c
                    cond_len = c_dot_c ** 0.5
                    proj_diff = eps_cond - u_on_c

                    # Amplify Cond based on length compared to projection of uncond
                    stretched = eps_cond + (stretch * proj_diff)

                    # Skew/Steer Conf based on rejection of uncond on cond
                    skewed = stretched - skew * u_rej_c

                    # Squash final length back down to original length of cond
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                case "v0.5.0":
                    def _dot(a, b):
                        return (a*b).flatten(1).sum(dim=1, keepdim=True) # [B,1]
                    def _nrm2(v):
                        return _dot(v, v)
                    
                    eps = eps = torch.finfo(eps_cond.dtype).eps
                    c_dot_c = _nrm2(eps_cond) + eps # [B,1]
                    u_dot_c = _dot(eps_uncond, eps_cond) # [B,1]

                    u_on_c = (u_dot_c / c_dot_c).unsqueeze(-1) * eps_cond # [B,1,1,1] * [B,C,H,W]
                    u_rej_c = eps_uncond - u_on_c
                    proj_diff = eps_cond - u_on_c

                    
                    # Amplify Cond based on length compared to projection of uncond
                    stretched = eps_cond + (stretch * proj_diff)

                    # Skew/Steer Conf based on rejection of uncond on cond
                    skewed = stretched - skew * u_rej_c

                    # Squash final length back down to original length of cond
                    cond_len = torch.sqrt(c_dot_c) # [B,1]
                    nrs_len = torch.sqrt(_nrm2(skewed) + eps) # [B,1]

                    squash_scale = (1 - squash) + squash * (cond_len / nrs_len)
                    x_final = skewed * squash_scale.unsqueeze(-1)

            return self._post_scale_conditioning(x_orig, x_div, x_final, sigma)
        
        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
}