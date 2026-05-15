"""Functional tests for NRS ComfyUI node and WebUI script."""

import inspect
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_nrs_node_comfyui_interface():
    """Test that NRS node has required ComfyUI interface."""
    from NRS.nodes_NRS import NRS

    # Test class can be instantiated
    node = NRS()
    assert node is not None

    # Test INPUT_TYPES classmethod exists and returns proper structure
    assert hasattr(NRS, "INPUT_TYPES")
    assert callable(NRS.INPUT_TYPES)
    input_types = NRS.INPUT_TYPES()
    assert isinstance(input_types, dict)
    assert "required" in input_types
    assert "model" in input_types["required"]
    assert "skew" in input_types["required"]
    assert "stretch" in input_types["required"]
    assert "squash" in input_types["required"]

    # Test patch method exists with correct signature
    assert hasattr(node, "patch")
    assert callable(node.patch)
    sig = inspect.signature(node.patch)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "skew" in params
    assert "stretch" in params
    assert "squash" in params

    # Test required class attributes
    assert hasattr(NRS, "RETURN_TYPES")
    assert NRS.RETURN_TYPES == ("MODEL",)
    assert hasattr(NRS, "FUNCTION")
    assert NRS.FUNCTION == "patch"
    assert hasattr(NRS, "CATEGORY")
    assert NRS.CATEGORY == "advanced/model"


def test_nrs_script_webui_interface():
    """Test that NRSScript has required WebUI/Gradio interface."""
    from scripts.negative_rejection_steering_script import NRSScript

    # Test class can be instantiated
    script = NRSScript()
    assert script is not None

    # Test required methods exist
    assert hasattr(script, "title")
    assert callable(script.title)
    assert isinstance(script.title(), str)

    assert hasattr(script, "show")
    assert callable(script.show)

    assert hasattr(script, "ui")
    assert callable(script.ui)

    assert hasattr(script, "process_before_every_sampling")
    assert callable(script.process_before_every_sampling)

    # Test process_before_every_sampling has correct signature
    sig = inspect.signature(script.process_before_every_sampling)
    params = list(sig.parameters.keys())
    # Note: 'self' is not included in signature, only other parameters
    assert "p" in params


def test_prediction_type_enum():
    """Test PredictionType enum has required values."""
    from NRS.nodes_NRS import PredictionType

    # Test enum has required prediction types
    assert hasattr(PredictionType, "EPS")
    assert hasattr(PredictionType, "V")
    assert hasattr(PredictionType, "X0")
    assert hasattr(PredictionType, "UNKNOWN")

    # Test enum values are distinct
    assert PredictionType.EPS != PredictionType.V
    assert PredictionType.V != PredictionType.X0
    assert PredictionType.X0 != PredictionType.UNKNOWN
