import logging
import math
from enum import Enum, auto

import torch

try:
    import comfy.utils as _comfy_utils
except Exception:
    _comfy_utils = None


def _unpack_latents(combined, latent_shapes):
    """Split a flat packed latent [B, 1, N] back into its per-stream tensors.

    Mirrors comfy.utils.unpack_latents: for each shape in latent_shapes, take
    math.prod(shape[1:]) elements off the last dim and reshape that [B, 1, n]
    slice back to `shape`.
    """
    streams = []
    offset = 0
    for shape in latent_shapes:
        n = math.prod(shape[1:])
        chunk = combined[:, :, offset : offset + n]
        streams.append(chunk.reshape(shape))
        offset += n
    return streams


def _pack_latents(streams):
    """Pack a list of per-stream tensors [B, C, ...] into a flat [B, 1, N] tensor.

    Mirrors comfy.utils.pack_latents: each stream is reshaped to (B, 1, -1)
    and concatenated on the last dim.
    """
    flat = [s.reshape(s.shape[0], 1, -1) for s in streams]
    return torch.cat(flat, dim=-1)


# fmt: off
class PredictionType(Enum):
    EPS     = auto()   # ε-prediction
    V       = auto()   # v-prediction
    X0      = auto()   # x₀-prediction
    FLOW    = auto()   # flow-matching / velocity — operated natively, no VP conversion
    UNKNOWN = auto()   # couldn’t detect / new scheduler


_RAW_TO_ENUM = {
    "eps":          PredictionType.EPS,
    "epsilon":      PredictionType.EPS,
    "flux":         PredictionType.EPS,
    "chroma":       PredictionType.EPS,
    "flow":         PredictionType.EPS,  # FLOW models (WAN, etc.) are EPS-compatible
    "wan":          PredictionType.EPS,  # WAN21 is FLOW-based
    "const":        PredictionType.EPS,  # CONST prediction class used in FLOW models
    "v":            PredictionType.V,
    "v_prediction": PredictionType.V,
    "x0":           PredictionType.X0,
    "sample":       PredictionType.X0,
}
# fmt: on


class NRS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input model to apply NRS to"}),
                "skew": (
                    "FLOAT",
                    {
                        "default": 2.00,
                        "min": -30.0,
                        "max": 30.0,
                        "step": 0.01,
                        "tooltip": "Changes the 'direction' of generation, steering away from negative prompt elements. Start with CFG/2.",
                    },
                ),
                "stretch": (
                    "FLOAT",
                    {
                        "default": 5.00,
                        "min": -30.0,
                        "max": 30.0,
                        "step": 0.01,
                        "tooltip": "Intensifies positive prompt elements. Start with your normal CFG value.",
                    },
                ),
                "squash": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Softens Skew/Stretch effects, adding micro-detailing. Keep low initially.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    DESCRIPTION = "Negative Rejection Steering (NRS) replaces CFG with more nuanced guidance. IMPORTANT: Set your KSampler CFG to any value (it will be ignored). Connect your model through this node before sampling."

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
                    pred_type = _RAW_TO_ENUM.get(p, PredictionType.UNKNOWN)
                    if pred_type != PredictionType.UNKNOWN:
                        logging.debug(
                            f"NRS._get_pred_type: Found prediction type '{p}' from attribute '{attr}' -> {pred_type}"
                        )
                        return pred_type

            # 2) enqueue child containers we care about -------------------
            for attr in ("model", "diffusion_model", "config", "scheduler", "inner_model", "model_sampling"):
                child = getattr(obj, attr, None)
                if child is not None and id(child) not in seen:
                    seen.add(id(child))
                    queue.append(child)

        # 3) enhanced detection for FLOW models (WAN, Flux, etc.) -------
        try:
            # Check model_sampling class type for FLOW models
            if hasattr(model, "model_sampling") and model.model_sampling is not None:
                sampling_class_name = type(model.model_sampling).__name__.lower()
                logging.debug(f"NRS._get_pred_type: Found model_sampling class: {sampling_class_name}")

                # CONST class is used by FLOW models (WAN21, Flux, etc.)
                if "const" in sampling_class_name:
                    logging.debug("NRS._get_pred_type: Detected FLOW model via CONST sampling class -> EPS")
                    return PredictionType.EPS
                elif "v_prediction" in sampling_class_name:
                    logging.debug("NRS._get_pred_type: Detected V-prediction model via sampling class -> V")
                    return PredictionType.V
                elif "eps" in sampling_class_name:
                    logging.debug("NRS._get_pred_type: Detected EPS model via sampling class -> EPS")
                    return PredictionType.EPS

            # Check model.model.model_type enum for newer models
            if hasattr(model, "model") and hasattr(model.model, "model_type"):
                model_type_str = _canon(str(model.model.model_type))
                logging.debug(f"NRS._get_pred_type: Found model.model.model_type: {model_type_str}")

                if "flow" in model_type_str or "flux" in model_type_str:
                    logging.debug("NRS._get_pred_type: Detected FLOW/Flux model via model_type -> EPS")
                    return PredictionType.EPS
                elif "v_prediction" in model_type_str:
                    logging.debug("NRS._get_pred_type: Detected V-prediction model via model_type -> V")
                    return PredictionType.V
                elif "eps" in model_type_str:
                    logging.debug("NRS._get_pred_type: Detected EPS model via model_type -> EPS")
                    return PredictionType.EPS

        except Exception as e:
            logging.debug(f"NRS._get_pred_type: Exception during enhanced detection: {e}")

        # 4) safe default (matches docstring promise) --------------------
        logging.warning("NRS._get_pred_type: Could not determine prediction type for model. Using EPS as fallback.")
        logging.debug(
            f"NRS._get_pred_type: Model structure: {[attr for attr in dir(model) if not attr.startswith('_')]}"
        )
        return PredictionType.EPS

    def _convert_to_v_space(self, x_orig, sig_root, sigma, cond, uncond, pred_type):
        x_div = None
        v_cond = cond
        v_uncond = uncond
        if pred_type in (PredictionType.V, PredictionType.FLOW):
            logging.debug("NRS._convert_to_v_space: already in v/flow, no pre-scale needed")
            pass  # already in v space / flow-matching operated natively
        elif pred_type == PredictionType.EPS:
            # ε → v conversion
            logging.debug("NRS._convert_to_v_space: generating x_div, v_cond, and v_uncond for eps")
            x_div = x_orig / (sigma**2 + 1)
            factor = sigma / sig_root

            v_cond = x_orig - (x_div - cond * factor)
            v_uncond = x_orig - (x_div - uncond * factor)
        elif pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._convert_to_v_space: x0-prediction not supported yet.")
        else:
            # Fallback: treat UNKNOWN as EPS and convert to V-space
            logging.warning(f"NRS._convert_to_v_space: Unknown prediction type {pred_type}, treating as EPS")
            logging.debug("NRS._convert_to_v_space: generating x_div, v_cond, and v_uncond for eps (fallback)")
            x_div = x_orig / (sigma**2 + 1)
            factor = sigma / sig_root
            v_cond = x_orig - (x_div - cond * factor)
            v_uncond = x_orig - (x_div - uncond * factor)

        return x_div, v_cond, v_uncond

    def _finalize_from_v_space(self, x_orig, x_div, x_final, sig_root, sigma, pred_type):
        nrs_result = x_final
        if pred_type in (PredictionType.V, PredictionType.FLOW):
            # already in v space / flow-matching operated natively
            logging.debug("NRS._finalize_from_v_space: already in v/flow, no post-scale needed")
            pass
        elif pred_type == PredictionType.EPS:
            # v → ε conversion
            logging.debug("NRS._finalize_from_v_space: generating cfg_result for eps")
            nrs_result = (x_div - (x_orig - x_final)) * (sig_root / sigma)
        elif pred_type == PredictionType.X0:
            raise NotImplementedError("NRS._finalize_from_v_space: x0-prediction not supported yet.")
        else:
            # Fallback: treat UNKNOWN as EPS and convert from V-space
            logging.warning(f"NRS._finalize_from_v_space: Unknown prediction type {pred_type}, treating as EPS")
            logging.debug("NRS._finalize_from_v_space: generating cfg_result for eps (fallback)")
            nrs_result = (x_div - (x_orig - x_final)) * (sig_root / sigma)
        return nrs_result

    def _apply_guidance(self, x_orig, cond, uncond, sigma, skew, stretch, squash, pred_type):
        """Run the NRS geometry pipeline on a single (already-unpacked, channels-first) stream."""
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        sig_root = (sigma**2 + 1).sqrt()

        # V and FLOW models are operated natively (identity); EPS models are converted to v-space.
        x_div, nrs_cond, nrs_uncond = self._convert_to_v_space(x_orig, sig_root, sigma, cond, uncond, pred_type)

        def _dot(a, b):
            return (a * b).sum(dim=1, keepdim=True)  # [B,C,W,H] => [B,1,W,H]

        def _nrm2(v):
            return _dot(v, v)

        eps = torch.finfo(nrs_cond.dtype).eps
        c_dot_c = _nrm2(nrs_cond) + eps  # [B,1,W,H]
        u_dot_c = _dot(nrs_uncond, nrs_cond)  # [B,1,W,H]
        u_on_c = (u_dot_c / c_dot_c) * nrs_cond  # [B,1,W,H] * [B,C,H,W]

        # Amplify Cond based on length compared to projection of uncond
        proj_diff = nrs_cond - u_on_c
        stretched = nrs_cond + (stretch * proj_diff)

        # Skew/Steer Conf based on rejection of uncond on cond
        u_rej_c = nrs_uncond - u_on_c
        skewed = stretched - (skew * u_rej_c)

        # Squash final length back down to original length of cond
        cond_len = nrs_cond.norm(dim=1, keepdim=True)
        nrs_len = skewed.norm(dim=1, keepdim=True) + eps

        squash_scale = (1 - squash) + (squash * (cond_len / nrs_len))
        x_final = skewed * squash_scale

        return self._finalize_from_v_space(x_orig, x_div, x_final, sig_root, sigma, pred_type)

    def patch(self, model, skew, stretch, squash):
        pred_type = self._get_pred_type(model)
        warned = {"done": False}

        def nrs(args):
            logging.debug(f"NRS.nrs: Skew: {skew}, Stretch: {stretch}, Squash: {squash}")
            cond = args["cond"]
            uncond = args["uncond"]
            x_orig = args["input"]
            sigma = args["sigma"]

            shapes = getattr(args["model"], "latent_shapes", None)
            if shapes and len(shapes) > 1:
                if _comfy_utils is not None and hasattr(_comfy_utils, "unpack_latents"):
                    cond_streams = _comfy_utils.unpack_latents(cond, shapes)
                    uncond_streams = _comfy_utils.unpack_latents(uncond, shapes)
                    x_streams = _comfy_utils.unpack_latents(x_orig, shapes)
                else:
                    cond_streams = _unpack_latents(cond, shapes)
                    uncond_streams = _unpack_latents(uncond, shapes)
                    x_streams = _unpack_latents(x_orig, shapes)
            else:
                cond_streams, uncond_streams, x_streams = [cond], [uncond], [x_orig]

            if not warned["done"]:
                for stream in x_streams:
                    if stream.shape[1] == 1:
                        logging.warning(
                            f"NRS.nrs: routed stream has a singleton reduction axis {tuple(stream.shape)}; "
                            "NRS geometry (dot/proj/skew) will degenerate to a no-op on this stream."
                        )
                        warned["done"] = True
                        break

            results = [
                self._apply_guidance(
                    x_streams[i], cond_streams[i], uncond_streams[i], sigma, skew, stretch, squash, pred_type
                )
                for i in range(len(cond_streams))
            ]

            if len(results) == 1:
                return results[0]

            if _comfy_utils is not None and hasattr(_comfy_utils, "pack_latents"):
                return _comfy_utils.pack_latents(results)[0]
            return _pack_latents(results)

        m = model.clone()
        m.set_model_sampler_cfg_function(nrs, True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "NRS": NRS,
}
