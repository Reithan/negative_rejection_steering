"""Experimental NRS_SQUASH_REF=raw switch: squash against the raw x - x0 hook vector length."""

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


def _run(pred, squash, mode, monkeypatch, dtype=None):
    if mode is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, mode)
    g = torch.Generator().manual_seed(1)
    shape = (2, 4, 8, 8)
    x, cond, uncond = (torch.randn(shape, generator=g) for _ in range(3))
    sigma = torch.tensor([0.7, 1.3])
    if dtype is not None:
        x, cond, uncond, sigma = (t.to(dtype) for t in (x, cond, uncond, sigma))
    pt = getattr(nrs_module.PredictionType, pred)
    out = nrs_module.NRS()._apply_guidance(x, cond, uncond, sigma, 2.0, 4.0, squash, pt)
    return out, cond


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_default_and_v_bit_identical_to_unset(monkeypatch, pred):
    base, _ = _run(pred, 0.5, None, monkeypatch)
    v, _ = _run(pred, 0.5, "v", monkeypatch)
    empty, _ = _run(pred, 0.5, "", monkeypatch)
    assert torch.equal(base, v)
    assert torch.equal(base, empty)


def test_flow_raw_equals_default(monkeypatch):
    base, _ = _run("FLOW", 0.5, None, monkeypatch)
    raw, _ = _run("FLOW", 0.5, "raw", monkeypatch)
    assert torch.equal(base, raw)


@pytest.mark.parametrize("pred", ["EPS", "V", "X0"])
def test_raw_squash_one_restores_cond_length_float64(monkeypatch, pred):
    out, cond = _run(pred, 1.0, "raw", monkeypatch, dtype=torch.float64)
    assert torch.allclose(out.norm(dim=1), cond.norm(dim=1), rtol=1e-6)


def test_raw_differs_from_default_for_vp(monkeypatch):
    base, _ = _run("EPS", 1.0, None, monkeypatch)
    raw, _ = _run("EPS", 1.0, "raw", monkeypatch)
    assert not torch.allclose(base, raw)


@pytest.mark.parametrize("pred", ALL_PREDS)
def test_squash_zero_raw_equals_default(monkeypatch, pred):
    base, _ = _run(pred, 0.0, None, monkeypatch)
    raw, _ = _run(pred, 0.0, "raw", monkeypatch)
    assert torch.allclose(base, raw, rtol=1e-4, atol=1e-5)


def test_unknown_value_warns_once_and_uses_default(monkeypatch, caplog):
    nrs_module._squash_ref_warned.clear()
    base, _ = _run("EPS", 0.5, None, monkeypatch)
    with caplog.at_level(logging.WARNING):
        a, _ = _run("EPS", 0.5, "bogus", monkeypatch)
        b, _ = _run("EPS", 0.5, "bogus", monkeypatch)
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
    assert not any(k.startswith("squash_scale_raw") for k in rec)
    assert "squash_scale_mean" in rec


def test_log_fields_raw_mode(monkeypatch, caplog):
    rec = _logged("EPS", "raw", monkeypatch, caplog)
    assert rec["squash_ref"] == "raw"
    lo, mean = rec["squash_scale_raw_min"], rec["squash_scale_raw_mean"]
    assert lo <= rec["squash_scale_raw_p05"] + 1e-6
    assert lo <= mean + 1e-6
    assert set(rec["alt"]) == {"lam_1", "lam_old_eps"}
    assert "squash_scale_mean" in rec  # existing v-space fields are unchanged


def test_log_fields_raw_mode_flow_matches_v_scale(monkeypatch, caplog):
    rec = _logged("FLOW", "raw", monkeypatch, caplog)
    assert rec["squash_scale_raw_mean"] == pytest.approx(rec["squash_scale_mean"], rel=1e-5)
    assert rec["squash_scale_raw_min"] == pytest.approx(rec["squash_scale_min"], rel=1e-5)
