"""Pytest configuration - mock torch and gradio since they're provided by ComfyUI/WebUI at runtime."""

import sys
from unittest.mock import MagicMock


# Create mock base class for WebUI Script
class MockScript:
    """Mock base class for WebUI scripts."""

    AlwaysVisible = "AlwaysVisible"  # Mock the AlwaysVisible constant


# Mock torch, gradio, and WebUI modules before any imports happen
# This allows pytest to discover and import tests without these heavy dependencies
sys.modules["torch"] = MagicMock()

# Mock gradio with return_value configured for common patterns
mock_gradio = MagicMock()
mock_gradio.Accordion = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
sys.modules["gradio"] = mock_gradio

# Mock WebUI modules with proper base class
mock_modules = MagicMock()
mock_modules.scripts = MagicMock()
mock_modules.scripts.Script = MockScript
mock_modules.script_callbacks = MagicMock()
sys.modules["modules"] = mock_modules
sys.modules["modules.scripts"] = mock_modules.scripts
sys.modules["modules.script_callbacks"] = mock_modules.script_callbacks
