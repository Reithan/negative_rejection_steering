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
        queue, seen = [model], set()

        while queue:
            obj = queue.pop(0)

            # 1) direct hit on this object ---------------------------------
            for attr in ("model_type", "prediction_type", "parameterization"):
                p = _canon(getattr(obj, attr, None))
                if p:
                    return _RAW_TO_ENUM.get(p, PredictionType.UNKNOWN)

            # 2) enqueue child containers we care about -------------------
            for attr in ("model", "diffusion_model", "config", "scheduler", "inner_model", "model_sampling"):
                child = getattr(obj, attr, None)
                if child is not None and id(child) not in seen:
                    seen.add(id(child))
                    queue.append(child)

        # 3) default ------------------------------------------------------
        return PredictionType.UNKNOWN
   
    def _convert_to_eps_space(self, x_orig, sig_root, sigma, cond, uncond):
        x_div = None
        eps_cond = cond
        eps_uncond = uncond
        if self.__pred_type == PredictionType.V:
            # v → ε conversion
            logging.debug(f"NRS._convert_to_eps_space: generating x_div, eps_cond, and eps_uncond for v-pred")
            x_div = x_orig / (sigma ** 2 + 1)

            eps_cond = ((x_div - (x_orig - cond)) * sig_root) / (sigma)
            eps_uncond = ((x_div - (x_orig - uncond)) * sig_root) / (sigma)
        elif self.__pred_type == PredictionType.EPS:
            logging.debug(f"NRS._convert_to_eps_space: already in eps, no pre-scale needed")
            pass  # already in ε space
        elif self.__pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._convert_to_eps_space: x0-prediction not supported yet.")
        else:
            raise RuntimeError("NRS._convert_to_eps_space: Could not determine prediction type for this model.")
        
        return x_div, eps_cond, eps_uncond

    def _finalize_from_eps_space(self, x_orig, x_div, x_final, sig_root, sigma):
        nrs_result = x_final
        if self.__pred_type == PredictionType.V:
            # ε → v conversion
            logging.debug(f"NRS._finalize_from_eps_space: generating cfg_result for v-pred")
            nrs_result = x_orig - (x_div - x_final * sigma / sig_root)
        elif self.__pred_type == PredictionType.EPS:
            # already in ε space
            logging.debug(f"NRS._finalize_from_eps_space: already in eps, no post-scale needed")
            pass
        elif self.__pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._finalize_from_eps_space: x0-prediction not supported yet.")
        else:
            raise RuntimeError("NRS._finalize_from_eps_space: Could not determine prediction type for this model.")
        return nrs_result
      
    def _convert_to_v_space(self, x_orig, sig_root, sigma, cond, uncond):
        x_div = None
        v_cond = cond
        v_uncond = uncond
        if self.__pred_type == PredictionType.V:
            logging.debug(f"NRS._convert_to_v_space: already in v, no pre-scale needed")
            pass  # already in v space
        elif self.__pred_type == PredictionType.EPS:
            # ε → v conversion
            logging.debug(f"NRS._convert_to_v_space: generating x_div, v_cond, and v_uncond for eps")
            x_div = x_orig / (sigma ** 2 + 1)
            factor = sigma / sig_root

            v_cond = x_orig - (x_div - cond * factor)
            v_uncond = x_orig - (x_div - uncond * factor)
        elif self.__pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._convert_to_v_space: x0-prediction not supported yet.")
        else:
            raise RuntimeError("NRS._convert_to_v_space: Could not determine prediction type for this model.")
        
        return x_div, v_cond, v_uncond

    def _finalize_from_v_space(self, x_orig, x_div, x_final, sig_root, sigma):
        nrs_result = x_final
        if self.__pred_type == PredictionType.V:
            # already in v space
            logging.debug(f"NRS._finalize_from_v_space: already in v, no post-scale needed")
            pass
        elif self.__pred_type == PredictionType.EPS:
            # v → ε conversion
            logging.debug(f"NRS._finalize_from_v_space: generating cfg_result for eps")
            nrs_result = (x_div - (x_orig - x_final)) * (sig_root / sigma)
        elif self.__pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._finalize_from_v_space: x0-prediction not supported yet.")
        else:
            raise RuntimeError("NRS._finalize_from_v_space: Could not determine prediction type for this model.")
        return nrs_result
        
    def patch(self, model, skew, stretch, squash):
        self.__pred_type = self._get_pred_type(model) if not hasattr(self, "__pred_type") else self.__pred_type
        self.__OPERATION_SPACE = PredictionType.V

        def nrs(args):
            logging.debug(f"NRS.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}")
            # self.__pred_type = self.__pred_type if self.__pred_type is not None else self._get_pred_type(model)
            cond = args["cond"]
            uncond = args["uncond"]
            x_orig = args["input"]
            
            sigma = args["sigma"]
            sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
            sig_root = (sigma ** 2 + 1).sqrt()
            
            nrs_cond, nrs_uncond = None, None
            match self.__OPERATION_SPACE:
                case PredictionType.V:
                    x_div, nrs_cond, nrs_uncond = self._convert_to_v_space(x_orig, sig_root, sigma, cond, uncond)
                case PredictionType.EPS:
                    x_div, nrs_cond, nrs_uncond = self._convert_to_eps_space(x_orig, sig_root, sigma, cond, uncond)
                case PredictionType.X0:
                    raise RuntimeError("NRS.nrs: x0-prediction not supported yet.")
                case PredictionType.UNKNOWN:
                    raise RuntimeError("NRS.nrs: Could not determine prediction type for this operation.")
                case _:
                    raise RuntimeError("NRS.nrs: Invalid PredictionType used.")

            x_final = None
            match "v0.6.0":
                case "v1":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c = (u_dot_c / c_dot_c) * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    displaced = (nrs_cond - skew * u_rej_c)
                    logging.debug(f"NRS.nrs: displaced")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)
                    squashed = displaced * squash_scale
                    logging.debug(f"NRS.nrs: squashed")

                    # stretch turned vector towards cond based on stretch scale
                    sq_dot_c = torch.sum(squashed * nrs_cond, dim=-1, keepdim=True)
                    sq_on_c = (sq_dot_c / c_dot_c) * nrs_cond
                    x_final = squashed + sq_on_c * stretch
                    logging.debug(f"NRS.nrs: final")
                case "v2":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    displaced = nrs_cond + stretch * (nrs_cond - torch.clamp(u_dot_c / c_dot_c, min=0, max=1) * nrs_cond) - skew * u_rej_c
                    logging.debug(f"NRS.nrs: displaced & stretched")

                    # squash displaced vector towards len(cond) based on squash scale
                    d_len_sq = torch.sum(displaced * displaced, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/d_len_sq) ** 0.5)
                    x_final = displaced * squash_scale
                    logging.debug(f"NRS.nrs: final")
                case "v3":
                    # displace cond by rejection of uncond on cond
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    displaced = (nrs_cond - skew * u_rej_c)
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
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    x_final = (nrs_cond - squash * u_rej_c + stretch * nrs_cond * ((rej_dor_rej/c_dot_c) ** 0.5))
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.1":
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    rej_dor_rej = torch.sum(u_rej_c * u_rej_c, dim=-1, keepdim=True)
                    stretched = nrs_cond + stretch * nrs_cond * ((rej_dor_rej/c_dot_c) ** 0.5)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * ((c_dot_c/sk_dot_sk) ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.2":
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    proj_len = torch.sum(u_on_c * u_on_c, dim=-1, keepdim=True) ** 0.5
                    cond_len = c_dot_c ** 0.5
                    stretched = nrs_cond * (1 + stretch * torch.abs(cond_len - proj_len) / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.3":
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    proj_len = torch.sum(u_on_c * u_on_c, dim=-1, keepdim=True) ** 0.5
                    cond_len = c_dot_c ** 0.5
                    stretched = nrs_cond * (1 + stretch * (cond_len - proj_len) / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.4":
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    cond_len = c_dot_c ** 0.5
                    proj_diff = nrs_cond - u_on_c
                    proj_diff_len = torch.sum(proj_diff * proj_diff, dim=-1, keepdim=True) ** 0.5
                    stretched = nrs_cond * (1 + stretch * proj_diff_len / cond_len)
                    skewed = stretched - skew * u_rej_c
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                    logging.debug(f"NRS.nrs: displaced")
                case "v0.4.5":
                    u_dot_c = torch.sum(nrs_uncond * nrs_cond, dim=-1, keepdim=True)
                    c_dot_c = torch.sum(nrs_cond * nrs_cond, dim=-1, keepdim=True)
                    u_on_c_mag = (u_dot_c / c_dot_c)
                    u_on_c = u_on_c_mag * nrs_cond
                    u_rej_c = nrs_uncond - u_on_c
                    cond_len = c_dot_c ** 0.5
                    proj_diff = nrs_cond - u_on_c

                    # Amplify Cond based on length compared to projection of uncond
                    stretched = nrs_cond + (stretch * proj_diff)

                    # Skew/Steer Conf based on rejection of uncond on cond
                    skewed = stretched - skew * u_rej_c

                    # Squash final length back down to original length of cond
                    sk_dot_sk = torch.sum(skewed * skewed, dim=-1, keepdim=True)
                    squash_scale = (1 - squash) + squash * cond_len / (sk_dot_sk ** 0.5)
                    x_final = skewed * squash_scale
                case "v0.5.0":
                    def _dot(a, b):
                        return (a*b).flatten(2).sum(dim=2, keepdim=True) # [B,C,W,H] => [B,C,1]

                    def _nrm2(v):
                        return _dot(v, v)

                    eps = torch.finfo(nrs_cond.dtype).eps
                    c_dot_c = _nrm2(nrs_cond) + eps # [B,1]
                    u_dot_c = _dot(nrs_uncond, nrs_cond) # [B,1]

                    u_on_c = (u_dot_c / c_dot_c).unsqueeze(-1) * nrs_cond # [B,1,1,1] * [B,C,H,W]
                    
                    # Amplify Cond based on length compared to projection of uncond
                    proj_diff = nrs_cond - u_on_c
                    stretched = nrs_cond + (stretch * proj_diff)

                    # Skew/Steer Conf based on rejection of uncond on cond
                    u_rej_c = nrs_uncond - u_on_c
                    skewed = stretched - (skew * u_rej_c)

                    # Squash final length back down to original length of cond
                    cond_len = torch.sqrt(c_dot_c) # [B,1]
                    nrs_len = torch.sqrt(_nrm2(skewed)) + eps # [B,1]

                    squash_scale = (1 - squash) + (squash * (cond_len / nrs_len))
                    x_final = skewed * squash_scale.unsqueeze(-1)
                case "v0.6.0":
                    def _dot(a, b):
                        return (a*b).sum(dim=1, keepdim=True) # [B,C,W,H] => [B,1,W,H]

                    def _nrm2(v):
                        return _dot(v, v)

                    eps = torch.finfo(nrs_cond.dtype).eps
                    c_dot_c = _nrm2(nrs_cond) + eps # [B,1]
                    u_dot_c = _dot(nrs_uncond, nrs_cond) # [B,1]

                    u_on_c = (u_dot_c / c_dot_c) * nrs_cond # [B,1,1,1] * [B,C,H,W]
                    
                    # Amplify Cond based on length compared to projection of uncond
                    proj_diff = nrs_cond - u_on_c
                    stretched = nrs_cond + (stretch * proj_diff)

                    # Skew/Steer Conf based on rejection of uncond on cond
                    u_rej_c = nrs_uncond - u_on_c
                    skewed = stretched - (skew * u_rej_c)

                    # Squash final length back down to original length of cond
                    cond_len = cond.norm(dim=1, keepdim=True)
                    nrs_len = skewed.norm(dim=1, keepdim=True)

                    squash_scale = (1 - squash) + (squash * (cond_len / nrs_len))
                    x_final = skewed * squash_scale

            match self.__OPERATION_SPACE:
                case PredictionType.V:
                    return self._finalize_from_v_space(x_orig, x_div, x_final, sig_root, sigma)
                case PredictionType.EPS:
                    return self._finalize_from_eps_space(x_orig, x_div, x_final, sig_root, sigma)
                case PredictionType.X0:
                    raise RuntimeError("NRS.nrs: x0-prediction not supported yet.")
                case PredictionType.UNKNOWN:
                    raise RuntimeError("NRS.nrs: Could not determine prediction type for this operation.")
                case _:
                    raise RuntimeError("NRS.nrs: Invalid PredictionType used.")
        
        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
}