"""Regression tests for NRS._get_pred_type and _RAW_TO_ENUM mappings.

These tests pin CURRENT behavior (flow-family names resolve to EPS) as a
safety net ahead of the FLOW reclassification planned for a later PR. If
this file needs updating because flow-family names now map to
PredictionType.FLOW, that is expected -- it means the reclassification
landed and this net did its job.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from NRS.nodes_NRS import PredictionType, _RAW_TO_ENUM, NRS


def _make_model_sampling(class_name):
    """Build an instance whose type name matches class_name, for class-name fingerprinting."""
    return type(class_name, (object,), {})()


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
        ("flux", PredictionType.EPS),
        ("chroma", PredictionType.EPS),
        ("flow", PredictionType.EPS),
        ("wan", PredictionType.EPS),
        ("const", PredictionType.EPS),
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

    def test_model_type_flow_family_is_currently_eps(self):
        """Flow-family models currently resolve to EPS (pre-reclassification)."""
        model = _StubModel(model_type="flow")
        assert NRS()._get_pred_type(model) == PredictionType.EPS

    def test_model_type_wan_is_currently_eps(self):
        model = _StubModel(model_type="wan")
        assert NRS()._get_pred_type(model) == PredictionType.EPS


class TestGetPredTypeEnhancedDetectionFallback:
    """The model_sampling class-name and model.model.model_type fallback paths."""

    def test_model_sampling_const_class_is_eps(self):
        model = _StubModel(model_sampling=_make_model_sampling("ModelSamplingContinuousEDMConst"))
        assert NRS()._get_pred_type(model) == PredictionType.EPS

    def test_model_sampling_v_prediction_class_is_v(self):
        model = _StubModel(model_sampling=_make_model_sampling("ModelSamplingV_Prediction"))
        assert NRS()._get_pred_type(model) == PredictionType.V

    def test_model_sampling_eps_class_is_eps(self):
        model = _StubModel(model_sampling=_make_model_sampling("ModelSamplingEps"))
        assert NRS()._get_pred_type(model) == PredictionType.EPS

    def test_unrecognized_model_defaults_to_eps(self):
        """Fully-unrecognized models fall back to EPS (documented default)."""
        model = _StubModel()
        assert NRS()._get_pred_type(model) == PredictionType.EPS
