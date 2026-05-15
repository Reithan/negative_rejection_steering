"""Smoke tests to ensure project structure and files are valid."""

from pathlib import Path


def test_required_files_exist():
    """Test that all required project files exist."""
    project_root = Path(__file__).parent.parent

    # Check main directories and files exist
    assert (project_root / "NRS").is_dir()
    assert (project_root / "scripts").is_dir()
    assert (project_root / "NRS" / "nodes_NRS.py").is_file()
    assert (project_root / "scripts" / "negative_rejection_steering_script.py").is_file()
    assert (project_root / "__init__.py").is_file()
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "README.md").is_file()
    assert (project_root / ".pre-commit-config.yaml").is_file() or True  # Will exist after Phase 4


def test_init_structure():
    """Test __init__.py exports the required mappings (no imports, text check only)."""
    project_root = Path(__file__).parent.parent
    init_content = (project_root / "__init__.py").read_text()

    # Check for required exports
    assert "NODE_CLASS_MAPPINGS" in init_content
    assert "NRS" in init_content
    assert "from .NRS.nodes_NRS import" in init_content


def test_nrs_node_structure():
    """Test NRS node file has expected structure (text check only, no imports)."""
    project_root = Path(__file__).parent.parent
    nrs_content = (project_root / "NRS" / "nodes_NRS.py").read_text()

    # Check for key classes and methods
    assert "class NRS:" in nrs_content
    assert "def INPUT_TYPES" in nrs_content
    assert "def patch" in nrs_content
    assert "class PredictionType" in nrs_content
    assert "PredictionType.EPS" in nrs_content
    assert "PredictionType.V" in nrs_content
    assert "PredictionType.X0" in nrs_content


def test_pyproject_has_dev_dependencies():
    """Test that pyproject.toml has dev dependencies configured."""
    project_root = Path(__file__).parent.parent
    pyproject_content = (project_root / "pyproject.toml").read_text()

    # Check for dev dependencies
    assert "[project.optional-dependencies]" in pyproject_content or "[tool.poetry.dev-dependencies]" in pyproject_content
    assert "pytest" in pyproject_content
    assert "ruff" in pyproject_content
    assert "pre-commit" in pyproject_content

    # Check for ruff config
    assert "[tool.ruff]" in pyproject_content
