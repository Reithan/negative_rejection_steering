# ruff: noqa: N999 -- filename predates NRS/__init__.py; mixed-case "nodes_NRS"
# only became checkable once NRS became a regular (non-namespace) package here.
# Renaming it is out of scope (would break existing imports); pyproject.toml's
# per-file-ignores already carve out N802/N804 for this same file.
import json
import logging
import math
import os
from enum import Enum, auto

import torch

try:
    import comfy.utils as _comfy_utils
except Exception:
    _comfy_utils = None

# Must be bumped together with the `version` field in pyproject.toml at release time.
__version__ = "1.1.1"


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


_GEOMETRY_ENV = "NRS_GEOMETRY_LOG"
_GEOMETRY_KEYS = (
    "parallel_share",
    "frac_backward",
    "squash_scale_mean",
    "squash_scale_min",
    "squash_scale_p05",
    "skew_push_median",
    "stretch_push_median",
)


def _geometry_target():
    """Return None (disabled), "" (log via logging) or a file path, from NRS_GEOMETRY_LOG."""
    val = os.environ.get(_GEOMETRY_ENV, "")
    if not val or val == "0":
        return None
    if "/" in val or os.sep in val or val.endswith(".jsonl"):
        return val
    return ""


def _geometry_stats(cond_v, uncond_v, skew, stretch, squash):
    """Diagnostic geometry statistics as a float32 tensor of len(_GEOMETRY_KEYS) (stays on device).

    Uses the same formulas as NRS._apply_guidance, on already-converted guidance vectors.
    """
    cond_v = cond_v.float()
    uncond_v = uncond_v.float()
    eps = torch.finfo(cond_v.dtype).eps
    c_dot_c = (cond_v * cond_v).sum(dim=1, keepdim=True) + eps
    u_dot_c = (uncond_v * cond_v).sum(dim=1, keepdim=True)
    ratio = u_dot_c / c_dot_c
    proj_diff = cond_v - ratio * cond_v
    u_rej_c = uncond_v - ratio * cond_v
    skewed = cond_v + stretch * proj_diff - skew * u_rej_c
    cond_len = cond_v.norm(dim=1, keepdim=True)
    scale = (1 - squash) + squash * (cond_len / (skewed.norm(dim=1, keepdim=True) + eps))
    pd2 = (proj_diff * proj_diff).sum()
    ur2 = (u_rej_c * u_rej_c).sum()
    flat = scale.flatten()
    k = max(1, math.ceil(0.05 * flat.numel()))
    clen = cond_len + eps
    return torch.stack(
        [
            pd2 / (pd2 + ur2 + eps),
            (ratio > 1).float().mean(),
            flat.mean(),
            flat.min(),
            flat.kthvalue(k).values,
            (abs(skew) * u_rej_c.norm(dim=1, keepdim=True) / clen).flatten().median(),
            (abs(stretch) * proj_diff.norm(dim=1, keepdim=True) / clen).flatten().median(),
        ]
    )


_geometry_failed_warned = False


def _json_safe(v):
    """Map non-finite floats to None (recursively) so every line is strict JSON."""
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _emit_geometry(record, target):
    line = json.dumps(_json_safe(record), allow_nan=False)
    if target:
        try:
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as e:
            logging.warning(f"NRS geometry log: cannot write {target}: {e}")
    else:
        logging.getLogger("NRS.geometry").info(f"NRS_GEOMETRY {line}")


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
    "flux":         PredictionType.FLOW,
    "chroma":       PredictionType.FLOW,
    "flow":         PredictionType.FLOW,  # FLOW models (WAN, etc.) operated natively
    "wan":          PredictionType.FLOW,  # WAN21 is FLOW-based
    "const":        PredictionType.FLOW,  # CONST prediction class used in FLOW models
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
                        "tooltip": "Changes the 'direction' of generation, steering away from negative prompt elements. Skew = Stretch = CFG-1 with Squash 0 reproduces CFG exactly.",
                    },
                ),
                "stretch": (
                    "FLOAT",
                    {
                        "default": 4.00,
                        "min": -30.0,
                        "max": 30.0,
                        "step": 0.01,
                        "tooltip": "Intensifies positive prompt elements. Skew = Stretch = CFG-1 with Squash 0 reproduces CFG exactly.",
                    },
                ),
                "squash": (
                    "FLOAT",
                    {
                        "default": 0.50,
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
                    logging.debug("NRS._get_pred_type: Detected FLOW model via CONST sampling class -> FLOW")
                    return PredictionType.FLOW
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
                    logging.debug("NRS._get_pred_type: Detected FLOW/Flux model via model_type -> FLOW")
                    return PredictionType.FLOW
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

    def _is_vp(self, pred_type):
        """VP (variance-preserving) parameterizations converted to v-space: EPS, V, X0.

        UNKNOWN (and any unhandled type) falls back to VP/v-space. FLOW/CONST is the only
        parameterization operated natively (see _convert_to_v_space).
        """
        if pred_type in (PredictionType.EPS, PredictionType.V, PredictionType.X0):
            return True
        if pred_type == PredictionType.FLOW:
            return False
        logging.warning(f"NRS: unknown prediction type {pred_type}, treating as VP (v-space)")
        return True

    def _convert_to_v_space(self, x_orig, sig_root, sigma, cond, uncond, pred_type):
        """Convert the (x - x0) guidance vectors into v-prediction space before the NRS geometry.

        The sampler hook delivers cond/uncond as `x - x0` for every parameterization (the
        model's raw output is converted to a denoised x0 before NRS sees it), so the true
        velocity is recovered the same way regardless of EPS/V/X0:
            v = (cond - A)/factor = (x/(sigma^2+1) - x0) * sig_root/sigma
        with A = x*sigma^2/(sigma^2+1), factor = sigma/sqrt(sigma^2+1).

        FLOW/CONST is operated natively: `x - x0 = sigma*out` is a pure scalar multiple of
        the model's velocity (no additive offset), and the NRS geometry is scale-invariant,
        so identity already runs on the native prediction. There is no VP v-space for
        flow-matching (its sigma is a [0,1] flow time, not a VP karras sigma).
        """
        if not self._is_vp(pred_type):
            logging.debug("NRS._convert_to_v_space: flow/const operated natively (identity)")
            return cond, uncond

        logging.debug("NRS._convert_to_v_space: converting VP prediction to v-space")
        factor = sigma / sig_root
        a_off = x_orig - x_orig / (sigma**2 + 1)  # A = x*sigma^2/(sigma^2+1)
        return (cond - a_off) / factor, (uncond - a_off) / factor

    def _finalize_from_v_space(self, x_orig, x_final, sig_root, sigma, pred_type):
        """Invert _convert_to_v_space so the hook returns `x - x0_final`. Round-trips exactly."""
        if not self._is_vp(pred_type):
            logging.debug("NRS._finalize_from_v_space: flow/const operated natively (identity)")
            return x_final

        factor = sigma / sig_root
        a_off = x_orig - x_orig / (sigma**2 + 1)
        return a_off + x_final * factor

    def _apply_guidance(self, x_orig, cond, uncond, sigma, skew, stretch, squash, pred_type, stream=0):
        """Run the NRS geometry pipeline on a single (already-unpacked, channels-first) stream."""
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        sig_root = (sigma**2 + 1).sqrt()

        # Convert (x - x0) guidance into v-space for all VP parameterizations (EPS/V/X0);
        # FLOW/CONST runs natively.
        nrs_cond, nrs_uncond = self._convert_to_v_space(x_orig, sig_root, sigma, cond, uncond, pred_type)

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

        target = _geometry_target()
        if target is not None:
            try:
                self._log_geometry(
                    target,
                    (x_orig, cond, uncond, nrs_cond, nrs_uncond, sigma),
                    (skew, stretch, squash),
                    pred_type,
                    stream,
                )
            except Exception as e:  # diagnostics must never abort a generation
                global _geometry_failed_warned
                if not _geometry_failed_warned:
                    _geometry_failed_warned = True
                    logging.warning(f"NRS geometry log: stats failed, skipping ({type(e).__name__}: {e})")

        return self._finalize_from_v_space(x_orig, x_final, sig_root, sigma, pred_type)

    def _log_geometry(self, target, tensors, knobs, pred_type, stream):
        """Emit one JSON line of geometry stats (opt-in via NRS_GEOMETRY_LOG). Never touches the output."""
        x_orig, cond, uncond, nrs_cond, nrs_uncond, sigma = tensors
        skew, stretch, squash = knobs
        n = len(_GEOMETRY_KEYS)
        with torch.no_grad():
            parts = [_geometry_stats(nrs_cond, nrs_uncond, skew, stretch, squash)]
            names = []
            if self._is_vp(pred_type):
                # Reference point p = lam*x: geometry vectors are p - x0 = (lam - 1)*x + (x - x0).
                x = x_orig.float()
                lams = (
                    ("lam_1", torch.ones_like(sigma)),
                    ("lam_old_eps", 1 + sigma / (sigma**2 + 1).sqrt()),
                )
                for name, lam in lams:
                    off = (lam.float() - 1) * x
                    parts.append(_geometry_stats(off + cond.float(), off + uncond.float(), skew, stretch, squash))
                    names.append(name)
            parts.append(sigma.flatten()[:1].float())
            vals = torch.cat(parts).cpu().tolist()  # single device->host transfer
        record = {
            "version": __version__,
            "sigma": vals[-1],
            "pred_type": pred_type.name,
            "stream": stream,
            "skew": float(skew),
            "stretch": float(stretch),
            "squash": float(squash),
            **dict(zip(_GEOMETRY_KEYS, vals[:n])),
        }
        if names:
            record["alt"] = {
                nm: dict(zip(_GEOMETRY_KEYS, vals[n * (i + 1) : n * (i + 2)])) for i, nm in enumerate(names)
            }
        _emit_geometry(record, target)

    def patch(self, model, skew, stretch, squash):
        pred_type = self._get_pred_type(model)
        logging.info(f"NRS v{__version__}: prediction type detected -> {pred_type.name}")
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
                    x_streams[i],
                    cond_streams[i],
                    uncond_streams[i],
                    sigma,
                    skew,
                    stretch,
                    squash,
                    pred_type,
                    stream=i,
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
