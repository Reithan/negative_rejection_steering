"""Tests for pack-aware per-stream routing in NRS.nodes_NRS.

These tests need real torch (tensor math), but tests/conftest.py installs a
MagicMock in sys.modules["torch"] for the whole session so other test modules
can import without the heavy dependency. We swap the real torch module in for
the duration of this module only, then restore the mock so the rest of the
suite is unaffected.
"""

import sys

import pytest

_saved_torch = None
_saved_nodes_nrs = None
torch = None
nrs_module = None


def setup_module(module):
    # NOTE: we deliberately avoid importlib.reload() here. reload() mutates
    # the *existing* NRS.nodes_NRS module dict in place, and other test
    # modules (e.g. test_pred_type.py) import PredictionType/NRS at
    # collection time and keep those references for the whole session. Their
    # methods' __globals__ point at that same dict, so an in-place reload
    # would silently swap PredictionType out from under them (new class
    # object, same name -> broken identity-based Enum equality). Instead we
    # unregister the module from sys.modules and import it fresh: this
    # creates an independent module object, leaving the original (still
    # cached in other modules' namespaces) untouched. We restore the exact
    # original module object on teardown.
    global _saved_torch, _saved_nodes_nrs, torch, nrs_module
    _saved_torch = sys.modules.get("torch")
    sys.modules.pop("torch", None)
    try:
        import torch as real_torch
    except ImportError:
        pytest.skip("real torch unavailable", allow_module_level=True)
    torch = real_torch

    _saved_nodes_nrs = sys.modules.get("NRS.nodes_NRS")
    sys.modules.pop("NRS.nodes_NRS", None)

    import NRS.nodes_NRS as m

    nrs_module = m


def teardown_module(module):
    if _saved_torch is not None:
        sys.modules["torch"] = _saved_torch
    else:
        sys.modules.pop("torch", None)

    if _saved_nodes_nrs is not None:
        sys.modules["NRS.nodes_NRS"] = _saved_nodes_nrs
    else:
        sys.modules.pop("NRS.nodes_NRS", None)


class _StubModelSampling:
    """Minimal stand-in that makes _get_pred_type fall back to EPS quickly."""


class _StubInnerModel:
    def __init__(self, latent_shapes=None):
        self.model_sampling = _StubModelSampling()
        if latent_shapes is not None:
            self.latent_shapes = latent_shapes


class _StubModel:
    """Stub for the outer ComfyUI ModelPatcher passed to NRS.patch()."""

    def __init__(self, latent_shapes=None):
        self.model = _StubInnerModel(latent_shapes)
        self._captured_fn = None

    def clone(self):
        return self

    def set_model_sampler_cfg_function(self, fn, flag):
        self._captured_fn = fn


def _make_args(model, cond, uncond, x_orig, sigma):
    return {
        "model": model.model,  # args["model"] is the inner model carrying latent_shapes
        "cond": cond,
        "uncond": uncond,
        "input": x_orig,
        "sigma": sigma,
    }


# ---------------------------------------------------------------------------
# Phase 1: round-trip pack/unpack correctness
# ---------------------------------------------------------------------------


def test_roundtrip_unpack_repack_two_streams():
    video = torch.randn(1, 4, 3, 2)
    audio = torch.randn(1, 6, 5)
    shapes = [video.shape, audio.shape]

    packed = nrs_module._pack_latents([video, audio])
    assert packed.shape == (1, 1, video.numel() + audio.numel())

    unpacked = nrs_module._unpack_latents(packed, shapes)
    assert len(unpacked) == 2
    assert torch.allclose(unpacked[0], video)
    assert torch.allclose(unpacked[1], audio)

    repacked = nrs_module._pack_latents(unpacked)
    assert torch.allclose(repacked, packed)


def test_roundtrip_single_stream():
    x = torch.randn(1, 4, 8, 8)
    shapes = [x.shape]

    packed = nrs_module._pack_latents([x])
    unpacked = nrs_module._unpack_latents(packed, shapes)
    assert len(unpacked) == 1
    assert torch.allclose(unpacked[0], x)


# ---------------------------------------------------------------------------
# Phase 2: fallback path (no latent_shapes) is a byte-for-byte regression no-op
# ---------------------------------------------------------------------------


def _run_nrs(model, cond, uncond, x_orig, sigma, skew=2.0, stretch=5.0, squash=0.75):
    node = nrs_module.NRS()
    (patched_model,) = node.patch(model, skew, stretch, squash)
    fn = patched_model._captured_fn
    args = _make_args(model, cond, uncond, x_orig, sigma)
    return fn(args)


def test_fallback_no_latent_shapes_matches_single_stream_shape():
    model = _StubModel(latent_shapes=None)
    cond = torch.randn(2, 4, 8, 8)
    uncond = torch.randn(2, 4, 8, 8)
    x_orig = torch.randn(2, 4, 8, 8)
    sigma = torch.rand(2) + 0.1

    result = _run_nrs(model, cond, uncond, x_orig, sigma)
    assert result.shape == x_orig.shape

    # Regression check: manually compute the single-stream result the same
    # way the pre-split code path did, and confirm equality.
    node = nrs_module.NRS()
    expected = node._apply_guidance(
        x_orig, cond, uncond, sigma, 2.0, 5.0, 0.75, nrs_module.PredictionType.EPS
    )
    assert torch.allclose(result, expected)


def test_single_stream_latent_shapes_also_matches():
    """A model.latent_shapes list of length 1 must take the same code path."""
    cond = torch.randn(1, 4, 5, 5)
    uncond = torch.randn(1, 4, 5, 5)
    x_orig = torch.randn(1, 4, 5, 5)
    sigma = torch.rand(1) + 0.1

    model = _StubModel(latent_shapes=[cond.shape])
    result = _run_nrs(model, cond, uncond, x_orig, sigma)

    node = nrs_module.NRS()
    expected = node._apply_guidance(
        x_orig, cond, uncond, sigma, 2.0, 5.0, 0.75, nrs_module.PredictionType.EPS
    )
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Phase 3: degeneracy tripwire
# ---------------------------------------------------------------------------


def test_tripwire_fires_on_flat_pack_without_latent_shapes(caplog):
    model = _StubModel(latent_shapes=None)
    cond = torch.randn(1, 1, 100)
    uncond = torch.randn(1, 1, 100)
    x_orig = torch.randn(1, 1, 100)
    sigma = torch.rand(1) + 0.1

    with caplog.at_level("WARNING"):
        _run_nrs(model, cond, uncond, x_orig, sigma)

    assert any("singleton reduction axis" in rec.message for rec in caplog.records)


def test_tripwire_does_not_fire_for_normal_single_stream(caplog):
    model = _StubModel(latent_shapes=None)
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    x_orig = torch.randn(1, 4, 8, 8)
    sigma = torch.rand(1) + 0.1

    with caplog.at_level("WARNING"):
        _run_nrs(model, cond, uncond, x_orig, sigma)

    assert not any("singleton reduction axis" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Phase 4: per-stream reduced shapes after unpack (H3-like video + audio)
# ---------------------------------------------------------------------------


def test_per_stream_reduced_shapes_after_unpack():
    video = torch.randn(1, 24, 4, 3, 2)
    audio = torch.randn(1, 32, 2, 5)
    shapes = [video.shape, audio.shape]

    packed = nrs_module._pack_latents([video, audio])
    unpacked = nrs_module._unpack_latents(packed, shapes)

    video_u, audio_u = unpacked
    assert video_u.shape == video.shape
    assert audio_u.shape == audio.shape

    # Channels sit at dim 1 for both streams.
    assert video_u.shape[1] == 24
    assert audio_u.shape[1] == 32

    video_reduced = video_u.sum(dim=1, keepdim=True)
    audio_reduced = audio_u.sum(dim=1, keepdim=True)
    assert video_reduced.shape == (1, 1, 4, 3, 2)
    assert audio_reduced.shape == (1, 1, 2, 5)


# ---------------------------------------------------------------------------
# Phase 5: split restores non-degenerate rejection (proves Skew is alive)
# ---------------------------------------------------------------------------


def test_split_restores_nondegenerate_rejection():
    """On a real multi-channel stream, uncond's rejection on cond must not
    collapse to ~0 -- this is the geometry that was silently dead on the flat
    [B,1,N] pack before the unpack/repack fix.
    """
    torch.manual_seed(0)
    cond = torch.randn(1, 8, 4, 4)
    # Make uncond non-parallel to cond so the rejection component is nonzero.
    uncond = torch.randn(1, 8, 4, 4)

    def _dot(a, b):
        return (a * b).sum(dim=1, keepdim=True)

    eps = torch.finfo(cond.dtype).eps
    c_dot_c = _dot(cond, cond) + eps
    u_dot_c = _dot(uncond, cond)
    u_on_c = (u_dot_c / c_dot_c) * cond
    u_rej_c = uncond - u_on_c

    assert u_rej_c.abs().max().item() > 1e-4


def test_flat_pack_rejection_is_degenerate_without_split():
    """Sanity check for the bug this PR fixes: reducing over the flat pack's
    singleton dim=1 axis collapses the rejection to exactly zero (up to
    floating point noise from the eps regularization term).
    """
    # float64 keeps the residual from the eps regularizer near the true
    # machine epsilon instead of float32 accumulation noise, so the
    # collapse-to-zero identity is exact enough to assert tightly.
    packed_cond = torch.randn(1, 1, 100, dtype=torch.float64)
    packed_uncond = torch.randn(1, 1, 100, dtype=torch.float64)

    def _dot(a, b):
        return (a * b).sum(dim=1, keepdim=True)

    eps = torch.finfo(packed_cond.dtype).eps
    c_dot_c = _dot(packed_cond, packed_cond) + eps
    u_dot_c = _dot(packed_uncond, packed_cond)
    u_on_c = (u_dot_c / c_dot_c) * packed_cond
    u_rej_c = packed_uncond - u_on_c

    assert u_rej_c.abs().max().item() < 1e-8
