"""Regression tests pinning the geometry identities of NRS._apply_guidance.

Every conversion NRS has shipped is equivalent to running the unchanged
skew/stretch/squash geometry on the vectors ``p - x0_c`` and ``p - x0_u``
for a reference point ``p = lam(sigma) * x`` (see the 1.1.0 review, "The
reference-point form"). These tests port that numpy reference to torch and
check ``_apply_guidance`` against it, so any change to the reference point
(for example P2 in the review) shows up as a deliberate test change.

Real torch is needed for tensor math, but tests/conftest.py mocks torch for
the session, so we swap the real module in for this module only.
"""

import logging
import sys

import pytest

_REAL_TORCH_KEY = "_nrs_real_torch"
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
    # Real torch can only be initialised once per process, so the first test
    # module to import it parks it under _REAL_TORCH_KEY for later ones to reuse.
    real_torch = sys.modules.get(_REAL_TORCH_KEY)
    if real_torch is None:
        try:
            import torch as real_torch
        except ImportError:
            pytest.skip("real torch unavailable", allow_module_level=True)
        sys.modules[_REAL_TORCH_KEY] = real_torch
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


KNOBS = [(2, 4, 0.5), (2, 5, 0.75), (10, 10, 1.0), (4, 2, 0.0), (0, 0, 0.5), (-1, 3, 0.25)]
SIGMAS = [14.6, 5.0, 1.0, 0.3, 0.03]
VP_TYPES = ["EPS", "V", "X0", "UNKNOWN"]
ALL_TYPES = [*VP_TYPES, "FLOW"]


def _make(sigmas):
    """Return (x, x0c, x0u, sigma) float64, shape [B,4,8,8]; sigma shape [B]."""
    g = torch.Generator().manual_seed(1234)
    sigma = torch.tensor(sigmas, dtype=torch.float64)
    shape = (len(sigmas), 4, 8, 8)
    x0c = torch.randn(shape, generator=g, dtype=torch.float64)
    x0u = x0c + 0.3 * torch.randn(shape, generator=g, dtype=torch.float64)
    x = x0c + sigma.view(-1, 1, 1, 1) * torch.randn(shape, generator=g, dtype=torch.float64)
    return x, x0c, x0u, sigma


def _guided_x0(x, x0c, x0u, sigma, knobs, pred):
    skew, stretch, squash = knobs
    pred_type = getattr(nrs_module.PredictionType, pred)
    out = nrs_module.NRS()._apply_guidance(x, x - x0c, x - x0u, sigma, skew, stretch, squash, pred_type)
    return x - out


def _lam(sigma, pred):
    return torch.ones_like(sigma) if pred == "FLOW" else 1 / (1 + sigma**2)


def _reference(x, x0c, x0u, sigma, knobs, pred):
    """Torch port of the review's lam_path(): guided x0 = (1-s)p + s(x0c + stretch*D_par + skew*D_perp)."""
    skew, stretch, squash = knobs
    p = _lam(sigma, pred).view(-1, 1, 1, 1) * x
    a = p - x0c  # axis vector
    delta = x0c - x0u
    e = a / a.norm(dim=1, keepdim=True)
    d_par = (delta * e).sum(1, keepdim=True) * e
    d_perp = delta - d_par
    pushed = a - stretch * d_par - skew * d_perp
    s = (1 - squash) + squash * a.norm(dim=1, keepdim=True) / pushed.norm(dim=1, keepdim=True)
    return (1 - s) * p + s * (x0c + stretch * d_par + skew * d_perp)


def _assert_close(actual, expected, rtol=1e-9, scale=None):
    scale = expected.abs().max().item() if scale is None else scale
    assert (actual - expected).abs().max().item() <= rtol * scale


@pytest.mark.parametrize("pred", ALL_TYPES)
@pytest.mark.parametrize("knobs", KNOBS)
@pytest.mark.parametrize("sigmas", [[14.6, 0.3], *[[s, s] for s in SIGMAS]])
def test_matches_reference_point_form(pred, knobs, sigmas):
    """x - _apply_guidance equals the reference-point guided x0 (lam=1/(1+s^2); lam=1 for FLOW)."""
    if pred == "FLOW":
        sigmas = [min(s, 0.99) for s in sigmas]  # flow time lives in (0, 1)
    x, x0c, x0u, sigma = _make(sigmas)
    _assert_close(_guided_x0(x, x0c, x0u, sigma, knobs, pred), _reference(x, x0c, x0u, sigma, knobs, pred))


@pytest.mark.parametrize("pred", ALL_TYPES)
@pytest.mark.parametrize("w", [1, 3, 7])
def test_squash_zero_balanced_is_cfg(pred, w):
    """At squash 0 and skew = stretch = w-1 the guided x0 is CFG: x0u + w*(x0c - x0u)."""
    sigmas = [0.9, 0.3] if pred == "FLOW" else [14.6, 0.3]
    x, x0c, x0u, sigma = _make(sigmas)
    guided = _guided_x0(x, x0c, x0u, sigma, (w - 1, w - 1, 0.0), pred)
    _assert_close(guided, x0u + w * (x0c - x0u), rtol=1e-10)


@pytest.mark.parametrize("pred", ["EPS", "FLOW"])
@pytest.mark.parametrize("squash", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("stretch", [0.5, 2.0])
def test_skew_zero_squash_scales_the_push(pred, squash, stretch):
    """With skew 0, guided x0 - x0c = (1 - squash) * stretch * D_par exactly (when the push does not overshoot).

    Derivation: with a = p - x0c = |a| e and D_par = d e (d = D.e, signed), skew 0 gives
    pushed = a - stretch*D_par = (|a| - stretch*d) e, parallel to a. If |a| > stretch*d
    (pushed keeps the direction of a), |pushed| = |a| - stretch*d, so
    s*pushed = (1-q)*pushed + q*a and guided x0 = p - s*pushed = x0c + (1-q)*stretch*D_par.
    Hence guided(q) - x0c = (1-q) * (guided(0) - x0c) per pixel. If the push overshoots
    (|a| < stretch*d), pushed flips sign, |pushed| = stretch*d - |a|, and the identity breaks;
    those pixels are excluded here. So the report's "effective stretch = stretch*(1-squash)"
    is exact per pixel, but only on the non-overshooting pixels, not unconditionally.
    """
    sigmas = [0.9, 0.5] if pred == "FLOW" else [14.6, 5.0]
    x, x0c, x0u, sigma = _make(sigmas)
    p = _lam(sigma, pred).view(-1, 1, 1, 1) * x
    a = p - x0c
    e = a / a.norm(dim=1, keepdim=True)
    d = ((x0c - x0u) * e).sum(1, keepdim=True)
    ok = (a.norm(dim=1, keepdim=True) - stretch * d > 0).expand_as(x)
    assert ok.float().mean() > 0.6  # the check must cover most pixels

    base = _guided_x0(x, x0c, x0u, sigma, (0.0, stretch, 0.0), pred) - x0c
    squashed = _guided_x0(x, x0c, x0u, sigma, (0.0, stretch, squash), pred) - x0c
    _assert_close(squashed[ok], (1 - squash) * base[ok], scale=base.abs().max().item())
    _assert_close(squashed[ok], (1 - squash) * stretch * (d * e)[ok], scale=base.abs().max().item())


def test_unknown_matches_eps_and_warns(caplog):
    """UNKNOWN logs a warning and is treated as VP, giving the same output as EPS."""
    x, x0c, x0u, sigma = _make([14.6, 0.3])
    knobs = (2, 4, 0.5)
    eps_out = _guided_x0(x, x0c, x0u, sigma, knobs, "EPS")
    with caplog.at_level(logging.WARNING):
        unknown_out = _guided_x0(x, x0c, x0u, sigma, knobs, "UNKNOWN")
    assert any("unknown prediction type" in r.message for r in caplog.records)
    assert torch.equal(eps_out, unknown_out)
