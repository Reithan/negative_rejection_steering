"""Tests for the opt-in NRS_GEOMETRY_LOG diagnostics in NRS.nodes_NRS.

Real torch is needed; tests/conftest.py installs a MagicMock torch for the whole
session, so we swap the real module in for this module only (same pattern as
tests/test_pack_split.py).
"""

import json
import logging
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
    # Real torch cannot be re-imported once popped from sys.modules (C-extension
    # re-registration fails), so the first module to load it caches it for the rest.
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


ENV = "NRS_GEOMETRY_LOG"
BASE_KEYS = {
    "version",
    "sigma",
    "pred_type",
    "stream",
    "skew",
    "stretch",
    "squash",
    "parallel_share",
    "frac_backward",
    "squash_scale_mean",
    "squash_scale_min",
    "squash_scale_p05",
    "skew_push_median",
    "stretch_push_median",
}


def _inputs(shape=(2, 4, 8, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    cond = torch.randn(shape, generator=g)
    uncond = torch.randn(shape, generator=g)
    x = torch.randn(shape, generator=g)
    sigma = torch.tensor([0.7, 1.3])
    return x, cond, uncond, sigma


def _guide(pred_name):
    x, cond, uncond, sigma = _inputs()
    node = nrs_module.NRS()
    pt = getattr(nrs_module.PredictionType, pred_name)
    return node._apply_guidance(x, cond, uncond, sigma, 2.0, 4.0, 0.5, pt)


def _records(caplog):
    return [r for r in caplog.records if r.name == "NRS.geometry"]


def test_off_by_default(monkeypatch, caplog):
    monkeypatch.delenv(ENV, raising=False)
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        _guide("EPS")
    assert _records(caplog) == []


def test_zero_is_off(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "0")
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        _guide("EPS")
    assert _records(caplog) == []


def test_on_emits_one_parseable_record(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        _guide("EPS")
    recs = _records(caplog)
    assert len(recs) == 1
    msg = recs[0].getMessage()
    assert msg.startswith("NRS_GEOMETRY ")
    rec = json.loads(msg[len("NRS_GEOMETRY ") :])
    assert BASE_KEYS <= set(rec)
    assert rec["version"] == nrs_module.__version__
    assert rec["pred_type"] == "EPS"
    assert rec["stream"] == 0
    assert rec["sigma"] == pytest.approx(0.7)
    assert (rec["skew"], rec["stretch"], rec["squash"]) == (2.0, 4.0, 0.5)
    assert 0.0 <= rec["parallel_share"] <= 1.0
    assert 0.0 <= rec["frac_backward"] <= 1.0
    assert set(rec["alt"]) == {"lam_1", "lam_old_eps"}
    for alt in rec["alt"].values():
        assert set(alt) == BASE_KEYS - {"version", "sigma", "pred_type", "stream", "skew", "stretch", "squash"}
        assert 0.0 <= alt["parallel_share"] <= 1.0


@pytest.mark.parametrize("pred", ["EPS", "V", "X0", "UNKNOWN", "FLOW"])
def test_output_bit_identical_on_vs_off(monkeypatch, pred):
    monkeypatch.delenv(ENV, raising=False)
    off = _guide(pred)
    monkeypatch.setenv(ENV, "1")
    on = _guide(pred)
    assert torch.equal(off, on)


def test_alt_present_for_vp_absent_for_flow(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        _guide("V")
        _guide("FLOW")
    recs = [json.loads(r.getMessage().split(" ", 1)[1]) for r in _records(caplog)]
    assert "alt" in recs[0]
    assert recs[1]["pred_type"] == "FLOW"
    assert "alt" not in recs[1]


@pytest.mark.parametrize("target_kind", ["jsonl_suffix", "path_sep"])
def test_file_mode_appends_jsonl(monkeypatch, caplog, tmp_path, target_kind):
    path = tmp_path / ("geo.jsonl" if target_kind == "jsonl_suffix" else "geo.log")
    monkeypatch.setenv(ENV, str(path))
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        _guide("EPS")
        _guide("FLOW")
    assert _records(caplog) == []
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["pred_type"] for line in lines] == ["EPS", "FLOW"]


def test_file_mode_unwritable_path_warns_and_continues(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / "missing_dir" / "geo.jsonl"))
    with caplog.at_level(logging.WARNING):
        out = _guide("EPS")
    assert out is not None
    assert any("cannot write" in r.getMessage() for r in caplog.records)


def test_parallel_share_extremes(monkeypatch, caplog):
    # uncond == cond -> no orthogonal difference at all -> share 0
    monkeypatch.setenv(ENV, "1")
    x, cond, _, sigma = _inputs()
    node = nrs_module.NRS()
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        node._apply_guidance(x, cond, cond.clone(), sigma, 2.0, 4.0, 0.5, nrs_module.PredictionType.FLOW)
    rec = json.loads(_records(caplog)[0].getMessage().split(" ", 1)[1])
    assert rec["parallel_share"] == pytest.approx(0.0, abs=1e-4)


# --- packed multi-stream routing -------------------------------------------


class _Sampling:
    pass


class _Inner:
    def __init__(self, latent_shapes=None):
        self.model_sampling = _Sampling()
        if latent_shapes is not None:
            self.latent_shapes = latent_shapes


class _Model:
    def __init__(self, latent_shapes=None):
        self.model = _Inner(latent_shapes)
        self.fn = None

    def clone(self):
        return self

    def set_model_sampler_cfg_function(self, fn, flag):
        self.fn = fn


def test_multi_stream_emits_per_stream_indices(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    video_shape, audio_shape = (1, 4, 3, 2), (1, 6, 5)
    shapes = [torch.Size(video_shape), torch.Size(audio_shape)]
    model = _Model(shapes)
    (patched,) = nrs_module.NRS().patch(model, 2.0, 4.0, 0.5)
    pack = nrs_module._pack_latents
    streams = [[torch.randn(s) for s in (video_shape, audio_shape)] for _ in range(3)]
    args = {
        "model": model.model,
        "cond": pack(streams[0]),
        "uncond": pack(streams[1]),
        "input": pack(streams[2]),
        "sigma": torch.tensor([0.9]),
    }
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        patched.fn(args)
    recs = [json.loads(r.getMessage().split(" ", 1)[1]) for r in _records(caplog)]
    assert [r["stream"] for r in recs] == [0, 1]


def test_single_stream_via_patch_uses_stream_zero(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    model = _Model(None)
    (patched,) = nrs_module.NRS().patch(model, 2.0, 4.0, 0.5)
    x, cond, uncond, sigma = _inputs()
    args = {"model": model.model, "cond": cond, "uncond": uncond, "input": x, "sigma": sigma}
    with caplog.at_level(logging.INFO, logger="NRS.geometry"):
        patched.fn(args)
    recs = [json.loads(r.getMessage().split(" ", 1)[1]) for r in _records(caplog)]
    assert [r["stream"] for r in recs] == [0]
