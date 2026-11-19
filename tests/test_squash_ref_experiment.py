"""Experimental NRS_SQUASH_REF=guidance switch: squash only the NRS push, towards |cond - uncond|."""

import json
import logging
import sys

import pytest

_saved_torch = None
_saved_nodes_nrs = None
torch = None
nrs_module = None


def setup_module(module):
    # See tests/test_pack_split.py: no importlib.reload(); import a fresh module object and
    # restore the original on teardown. Real torch is cached because it can't be re-imported.
    global _saved_torch, _saved_nodes_nrs, torch, nrs_module
    _saved_torch = sys.modules.get("torch")
    sys.modules.pop("torch", None)
    real_torch = sys.modules.get("_nrs_real_torch")
    if real_torch is None:
        try:
            import torch as real_torch
        except ImportError:
            pytest.skip("real torch unavailable", allow_module_level=True)
        sys.modules["_nrs_real_torch"] = real_torch
    else:
        sys.modules["torch"] = real_torch
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


ENV = "NRS_SQUASH_REF"
LOG_ENV = "NRS_GEOMETRY_LOG"
ALL_PREDS = ["EPS", "V", "X0", "UNKNOWN", "FLOW"]
VP_PREDS = ["EPS", "V", "X0"]


def _inputs(dtype=None):
    g = torch.Generator().manual_seed(1)
    shape = (2, 4, 8, 8)
    x, cond, uncond = (torch.randn(shape, generator=g) for _ in range(3))
    sigma = torch.tensor([0.7, 1.3])
    if dtype is not None:
        x, cond, uncond, sigma = (t.to(dtype) for t in (x, cond, uncond, sigma))
    return x, cond, uncond, sigma


def _run(pred, squash, mode, monkeypatch, dtype=None, skew=2.0, stretch=4.0):
    if mode is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, mode)
    x, cond, uncond, sigma = _inputs(dtype)
    pt = getattr(nrs_module.PredictionType, pred)
    return nrs_module.NRS()._apply_guidance(x, cond, uncond, sigma, skew, stretch, squash, pt)


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_default_and_v_bit_identical_to_unset(monkeypatch, pred):
    base = _run(pred, 0.5, None, monkeypatch)
    assert torch.equal(base, _run(pred, 0.5, "v", monkeypatch))
    assert torch.equal(base, _run(pred, 0.5, "", monkeypatch))


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_squash_zero_guidance_equals_default(monkeypatch, pred):
    base = _run(pred, 0.0, None, monkeypatch)
    alt = _run(pred, 0.0, "guidance", monkeypatch)
    assert torch.allclose(base, alt, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_guidance_squash_one_push_has_delta_length_float64(monkeypatch, pred):
    # In x - x0 form the guided result minus cond is the (affinely scaled) push; at squash 1 its
    # length equals |cond - uncond| per pixel, independent of the reference point.
    x, cond, uncond, _ = _inputs(torch.float64)
    out = _run(pred, 1.0, "guidance", monkeypatch, dtype=torch.float64)
    push_len = (out - cond).norm(dim=1)
    delta_len = (cond - uncond).norm(dim=1)
    assert torch.allclose(push_len, delta_len, rtol=1e-6)


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_guidance_keeps_push_direction(monkeypatch, pred):
    x, cond, _, _ = _inputs(torch.float64)
    zero = _run(pred, 0.0, "guidance", monkeypatch, dtype=torch.float64)
    one = _run(pred, 1.0, "guidance", monkeypatch, dtype=torch.float64)
    cos = torch.nn.functional.cosine_similarity(zero - cond, one - cond, dim=1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-9)


def test_guidance_differs_from_default(monkeypatch):
    base = _run("EPS", 1.0, None, monkeypatch)
    alt = _run("EPS", 1.0, "guidance", monkeypatch)
    assert not torch.allclose(base, alt)


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_guidance_zero_push_returns_cond(monkeypatch, pred):
    x, cond, _, _ = _inputs(torch.float64)
    out = _run(pred, 1.0, "guidance", monkeypatch, dtype=torch.float64, skew=0.0, stretch=0.0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, cond, atol=1e-9)


@pytest.mark.parametrize("mode", ["bogus", "raw"])
def test_unknown_value_warns_once_and_uses_default(monkeypatch, caplog, mode):
    nrs_module._squash_ref_warned.clear()
    base = _run("EPS", 0.5, None, monkeypatch)
    with caplog.at_level(logging.WARNING):
        a = _run("EPS", 0.5, mode, monkeypatch)
        b = _run("EPS", 0.5, mode, monkeypatch)
    warnings = [r for r in caplog.records if "NRS_SQUASH_REF" in r.getMessage()]
    assert len(warnings) == 1
    assert torch.equal(base, a)
    assert torch.equal(base, b)


def _logged(pred, mode, monkeypatch, caplog):
    monkeypatch.setenv(LOG_ENV, "1")
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        _run(pred, 0.5, mode, monkeypatch)
    recs = [r for r in caplog.records if r.name == "NRS.geometry"]
    assert len(recs) == 1
    return json.loads(recs[0].getMessage().split(" ", 1)[1])


def test_log_fields_default_mode(monkeypatch, caplog):
    rec = _logged("EPS", None, monkeypatch, caplog)
    assert rec["squash_ref"] == "v"
    assert not any(k.startswith("squash_scale_guidance") for k in rec)
    assert "squash_scale_mean" in rec


@pytest.mark.parametrize("pred", ["EPS", "FLOW"])
def test_log_fields_guidance_mode(monkeypatch, caplog, pred):
    rec = _logged(pred, "guidance", monkeypatch, caplog)
    assert rec["squash_ref"] == "guidance"
    lo, mean = rec["squash_scale_guidance_min"], rec["squash_scale_guidance_mean"]
    assert lo <= rec["squash_scale_guidance_p05"] + 1e-6
    assert lo <= mean + 1e-6
    assert "squash_scale_mean" in rec  # existing v-space fields are unchanged
    if pred == "EPS":
        assert set(rec["alt"]) == {"lam_1", "lam_old_eps"}
