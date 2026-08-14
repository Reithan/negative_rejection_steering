"""Tests for NRS package version metadata and the patch()-time version/pred-type log line."""

import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import NRS  # noqa: E402
import NRS.nodes_NRS as nodes_NRS  # noqa: E402, N812

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def test_version_is_nonempty_semver_string():
    """NRS.__version__ must be importable and look like a X.Y.Z version."""
    assert isinstance(NRS.__version__, str)
    assert NRS.__version__
    assert _SEMVER_RE.match(NRS.__version__), f"__version__ {NRS.__version__!r} is not X.Y.Z"


def test_version_matches_pyproject():
    """__version__ must be kept in lock-step with pyproject.toml's version field."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject_path.read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match, "Could not find version in pyproject.toml"
    assert NRS.__version__ == match.group(1)


class _StubModel:
    """Minimal stand-in that resolves to PredictionType.EPS via the direct-hit path."""

    def __init__(self):
        self.model_type = "eps"

    def clone(self):
        return self

    def set_model_sampler_cfg_function(self, fn, flag):
        self._captured_fn = fn


def test_patch_logs_version_and_pred_type(caplog):
    """patch() must announce the NRS version and detected prediction type."""
    node = nodes_NRS.NRS()
    model = _StubModel()

    with caplog.at_level("INFO"):
        node.patch(model, skew=2.0, stretch=5.0, squash=0.75)

    expected = f"NRS v{NRS.__version__}: prediction type detected -> {nodes_NRS.PredictionType.EPS.name}"
    assert any(expected in rec.message for rec in caplog.records)
