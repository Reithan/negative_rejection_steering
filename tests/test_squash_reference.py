"""Experimental NRS_SQUASH_REF=raw switch: squash against the raw x - x0 hook vector length."""

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


def _run(pred, squash, mode, monkeypatch):
    if mode:
        monkeypatch.setenv(ENV, mode)
    else:
        monkeypatch.delenv(ENV, raising=False)
    g = torch.Generator().manual_seed(1)
    shape = (2, 4, 8, 8)
    x, cond, uncond = (torch.randn(shape, generator=g) for _ in range(3))
    sigma = torch.tensor([0.7, 1.3])
    pt = getattr(nrs_module.PredictionType, pred)
    out = nrs_module.NRS()._apply_guidance(x, cond, uncond, sigma, 2.0, 4.0, squash, pt)
    return out, cond


def test_flow_identical_to_default(monkeypatch):
    base, _ = _run("FLOW", 0.5, None, monkeypatch)
    raw, _ = _run("FLOW", 0.5, "raw", monkeypatch)
    assert torch.equal(base, raw)


def test_vp_default_unchanged_when_switch_unset_or_other(monkeypatch):
    base, _ = _run("EPS", 0.5, None, monkeypatch)
    other, _ = _run("EPS", 0.5, "v", monkeypatch)
    assert torch.equal(base, other)


@pytest.mark.parametrize("pred", ["EPS", "V", "X0"])
def test_raw_squash_one_restores_cond_length(monkeypatch, pred):
    out, cond = _run(pred, 1.0, "raw", monkeypatch)
    assert torch.allclose(out.norm(dim=1), cond.norm(dim=1), rtol=1e-4, atol=1e-5)


def test_raw_differs_from_default_for_vp(monkeypatch):
    base, _ = _run("EPS", 1.0, None, monkeypatch)
    raw, _ = _run("EPS", 1.0, "raw", monkeypatch)
    assert not torch.allclose(base, raw)


def test_raw_squash_zero_matches_default(monkeypatch):
    base, _ = _run("EPS", 0.0, None, monkeypatch)
    raw, _ = _run("EPS", 0.0, "raw", monkeypatch)
    assert torch.allclose(base, raw, rtol=1e-4, atol=1e-5)
