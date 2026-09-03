"""Regression tests for NRS._get_pred_type, _RAW_TO_ENUM mappings, and the
V/FLOW/EPS operation-space conversion helpers.

PR-3 reclassified the flow-matching family (flux, chroma, flow, wan, const)
from PredictionType.EPS onto a new PredictionType.FLOW, which is operated
natively (identity conversion, no VP ε<->v algebra). These tests pin that
post-reclassification behavior at both detection sites (the _RAW_TO_ENUM
dict and the enhanced-detection fallback in _get_pred_type).

FLOW is the sole native path; every VP parameterization (EPS, V, X0, and the
UNKNOWN fallback) shares one ε/v/x0 -> v-space conversion through
_convert_to_v_space / _finalize_from_v_space. These tests cover the FLOW
identity round-trip and confirm the VP branches actually transform their inputs.
"""

import enum
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from NRS.nodes_NRS import _RAW_TO_ENUM, NRS, PredictionType


def _make_model_sampling(class_name):
    """Build an instance whose type name matches class_name, for class-name fingerprinting."""
    return type(class_name, (object,), {})()


class _ModelType(enum.Enum):
    """Mirrors ComfyUI's real model_type.ModelType Enum, as exposed by MiniMax H3's
    BaseModel.model_type -- a genuine Enum member, not a raw string.
    """

    FLOW = enum.auto()


class _StubModel:
    """Minimal stand-in for a model object walked by _get_pred_type."""

    def __init__(self, model_type=None, model_sampling=None, inner_model_type=None):
        if model_type is not None:
            self.model_type = model_type
        if model_sampling is not None:
            self.model_sampling = model_sampling
        if inner_model_type is not None:
            # Emulates model.model.model_type used by the enhanced-detection fallback.
            self.model = _StubModel(model_type=inner_model_type)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("eps", PredictionType.EPS),
        ("epsilon", PredictionType.EPS),
        ("flux", PredictionType.FLOW),
        ("chroma", PredictionType.FLOW),
        ("flow", PredictionType.FLOW),
        ("wan", PredictionType.FLOW),
        ("const", PredictionType.FLOW),
        ("v", PredictionType.V),
        ("v_prediction", PredictionType.V),
        ("x0", PredictionType.X0),
        ("sample", PredictionType.X0),
    ],
)
def test_raw_to_enum_mapping(raw, expected):
    """Pin the current _RAW_TO_ENUM dict mappings."""
    assert _RAW_TO_ENUM[raw] == expected


def test_raw_to_enum_unknown_raw_not_present():
    """Unrecognized raw strings are not in the dict; callers fall back to UNKNOWN."""
    assert "totally-unrecognized" not in _RAW_TO_ENUM


class TestGetPredTypeDirectAttribute:
    """_get_pred_type's direct-hit path via model_type/prediction_type/parameterization."""

    def test_model_type_v_prediction(self):
        node = NRS()
        model = _StubModel(model_type="v_prediction")
        assert node._get_pred_type(model) == PredictionType.V

    def test_model_type_eps(self):
        node = NRS()
        model = _StubModel(model_type="eps")
        assert node._get_pred_type(model) == PredictionType.EPS

    def test_model_type_x0(self):
        node = NRS()
        model = _StubModel(model_type="x0")
        assert node._get_pred_type(model) == PredictionType.X0

    def test_model_type_flow_is_flow(self):
        """Flow-family models resolve to FLOW (native operation, no VP conversion)."""
        model = _StubModel(model_type="flow")
        assert NRS()._get_pred_type(model) == PredictionType.FLOW

    def test_model_type_wan_is_flow(self):
        model = _StubModel(model_type="wan")
        assert NRS()._get_pred_type(model) == PredictionType.FLOW

    def test_h3_enum_model_type_resolves_to_flow(self):
        """MiniMax H3 exposes model.model.model_type as a real Enum member
        (ModelType.FLOW), not a raw string. _canon's `isinstance(p, Enum)`
        branch reduces it to `p.name` ("FLOW" -> "flow") before the
        _RAW_TO_ENUM dict lookup, so this pins that Enum path -- as taken by
        H3's real model_type attribute -- resolves at the direct-hit site.
        """
        model = _StubModel(inner_model_type=_ModelType.FLOW)
        assert NRS()._get_pred_type(model) == PredictionType.FLOW


class TestGetPredTypeEnhancedDetectionFallback:
    """The model_sampling class-name and model.model.model_type fallback paths.

    Each stub below is deliberately built so the only detectable signal lives
    in the fallback (section 3) logic -- not an exact _RAW_TO_ENUM key hit
    during the BFS walk -- so these tests genuinely exercise the fallback
    branches rather than just re-testing the dict.
    """

    def test_model_sampling_const_class_is_flow(self):
        """A CONST-like model_sampling class name is the only flow signal here."""
        model = _StubModel(model_sampling=_make_model_sampling("ModelSamplingContinuousEDMConst"))
        assert NRS()._get_pred_type(model) == PredictionType.FLOW

    def test_model_sampling_v_prediction_class_is_v(self):
        model = _StubModel(model_sampling=_make_model_sampling("ModelSamplingV_Prediction"))
        assert NRS()._get_pred_type(model) == PredictionType.V

    def test_model_sampling_eps_class_is_eps(self):
        model = _StubModel(model_sampling=_make_model_sampling("ModelSamplingEps"))
        assert NRS()._get_pred_type(model) == PredictionType.EPS

    def test_inner_model_type_flow_string_is_flow(self):
        """A model.model.model_type whose str() merely *contains* 'flow' (e.g. an
        Enum repr like 'ModelType.FLOW') isn't an exact _RAW_TO_ENUM key, so the
        BFS direct-hit path can't resolve it -- only the model.model.model_type
        substring fallback can.
        """
        model = _StubModel(inner_model_type="ModelType.FLOW")
        assert NRS()._get_pred_type(model) == PredictionType.FLOW

    def test_inner_model_type_flux_string_is_flow(self):
        model = _StubModel(inner_model_type="ModelType.FLUX")
        assert NRS()._get_pred_type(model) == PredictionType.FLOW

    def test_unrecognized_model_defaults_to_eps(self):
        """Fully-unrecognized models fall back to EPS (documented default)."""
        model = _StubModel()
        assert NRS()._get_pred_type(model) == PredictionType.EPS


class TestConvertToVSpaceBranches:
    """FLOW is the only native (identity) parameterization; every VP type
    (EPS, V, X0, and the UNKNOWN fallback) now runs the shared ε/v/x0 -> v-space
    algebra. FLOW identity needs no tensor math, so sentinel objects prove it;
    the VP branches use MagicMock to confirm the algebra actually transforms.
    """

    def test_flow_convert_is_identity(self):
        node = NRS()
        cond, uncond = object(), object()
        v_cond, v_uncond = node._convert_to_v_space(object(), object(), object(), cond, uncond, PredictionType.FLOW)
        assert v_cond is cond
        assert v_uncond is uncond

    def test_flow_finalize_is_identity(self):
        node = NRS()
        x_final = object()
        result = node._finalize_from_v_space(object(), x_final, object(), object(), PredictionType.FLOW)
        assert result is x_final

    @pytest.mark.parametrize("pred_type", [PredictionType.EPS, PredictionType.V, PredictionType.X0])
    def test_vp_convert_performs_algebra(self, pred_type):
        """EPS/V/X0 all run the ε->v conversion (cond/uncond are transformed,
        not passed through)."""
        node = NRS()
        x_orig, sig_root, sigma = MagicMock(), MagicMock(), MagicMock()
        cond, uncond = MagicMock(), MagicMock()
        v_cond, v_uncond = node._convert_to_v_space(x_orig, sig_root, sigma, cond, uncond, pred_type)
        assert v_cond is not cond
        assert v_uncond is not uncond

    @pytest.mark.parametrize("pred_type", [PredictionType.EPS, PredictionType.V, PredictionType.X0])
    def test_vp_finalize_performs_algebra(self, pred_type):
        node = NRS()
        x_orig, x_final, sig_root, sigma = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        result = node._finalize_from_v_space(x_orig, x_final, sig_root, sigma, pred_type)
        assert result is not x_final

    def test_unknown_convert_falls_back_to_vp(self):
        """UNKNOWN (and any unhandled type) is treated as VP -> runs the algebra."""
        node = NRS()
        x_orig, sig_root, sigma = MagicMock(), MagicMock(), MagicMock()
        cond, uncond = MagicMock(), MagicMock()
        v_cond, v_uncond = node._convert_to_v_space(x_orig, sig_root, sigma, cond, uncond, PredictionType.UNKNOWN)
        assert v_cond is not cond
        assert v_uncond is not uncond
