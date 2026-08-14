# Contributing to Negative Rejection Steering

Thank you for your interest in contributing! This document provides guidelines for setting up your development environment and contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Git

### Initial Setup

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone https://github.com/Reithan/negative_rejection_steering.git
   cd negative_rejection_steering
   ```

2. **Create virtual environment with uv**:

   ```bash
   uv venv
   ```

3. **Activate virtual environment**:

   ```bash
   # Git Bash (Windows)
   source .venv/Scripts/activate

   # Linux/Mac
   source .venv/bin/activate

   # Windows CMD
   .venv\Scripts\activate.bat

   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

4. **Install development dependencies**:

   ```bash
   uv pip install -e ".[dev]"
   ```

5. **Install git hooks**:

   ```bash
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

   The pre-push hook runs the full test suite with branch coverage and blocks
   the push if changed code drops below 90% branch coverage (via pytest-cov +
   diff-cover, mirroring CI). It needs `uv` installed — if `uv` isn't found,
   the check is skipped with a warning — and `origin/main` fetched locally so
   there's something to diff against.

6. **Verify setup**:

   ```bash
   # Run hooks manually on all files
   pre-commit run --all-files

   # Check that ruff works
   ruff check .
   ```

## Git Workflow

### Protected Branches

- **Direct commits to `main` are blocked** by git hooks
- **Direct pushes to `main` are blocked** by git hooks
- All changes must go through feature branches and pull requests

### Recommended Workflow

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

   Or for bug fixes:

   ```bash
   git checkout -b fix/bug-description
   ```

2. **Make your changes and commit**:

   ```bash
   git add <files>
   git commit -m "Your commit message"
   ```

   The pre-commit hook will automatically:
   - Run ruff linting and auto-fix issues
   - Check for trailing whitespace, missing final newlines, etc.
   - Block the commit if you're on the main branch

   If the linter auto-fixes files, you'll need to re-stage and commit again.

3. **Push your branch**:

   ```bash
   git push origin feature/your-feature-name
   ```

   The pre-push hook will:
   - Run tests (if pytest is available)
   - Block the push if you're on the main branch

4. **Create a pull request** on GitHub

5. **Merge after review**

### Bypassing Hooks (Emergency Only)

If you must bypass hooks (NOT recommended):

```bash
git commit --no-verify    # Skip pre-commit hooks
git push --no-verify      # Skip pre-push hooks
```

**Warning**: Only use `--no-verify` in emergencies. Bypassing hooks may:

- Introduce linting issues
- Break Continuous Integration/Continuous Deployment pipelines
- Allow untested code to be pushed

## Development Commands

### Linting

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Check a specific file
ruff check NRS/nodes_NRS.py
```

### Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_smoke.py

# Run specific test function
pytest tests/test_smoke.py::test_file_structure
```

**Note**: Tests use mocked versions of torch and gradio (via conftest.py) since these dependencies are provided by ComfyUI/WebUI at runtime.

### Pre-commit Hooks

```bash
# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files

# Update hook versions
pre-commit autoupdate
```

## Code Style Guidelines

This project uses **ruff** for linting and formatting with the following configuration:

- **Line length**: 120 characters
- **Target Python version**: 3.10+
- **Enabled checks**: pycodestyle (E/W), pyflakes (F), isort (I), pep8-naming (N), pyupgrade (UP)

### Special Cases

- **ComfyUI API conventions**: The `INPUT_TYPES` method and `s` parameter naming are required by ComfyUI's API and are exempted from normal naming rules
- **Star imports in `__init__.py`**: Required for ComfyUI node discovery

## Commit Message Guidelines

Write clear, concise commit messages:

- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Add detailed description in the body if needed

Good examples:

```
Add support for XYZ model type
Fix crash when prediction type is unknown
Update README with installation instructions
```

Bad examples:

```
fixed stuff
WIP
Updated code
```

## Pull Request Guidelines

When submitting a pull request:

1. **Keep PRs focused**: One feature or fix per PR
2. **Update documentation**: If you add features, update README.md
3. **Test your changes**: Ensure the extension works in ComfyUI/reForge
4. **Run pre-commit hooks**: Make sure all checks pass
5. **Describe your changes**: Explain what and why in the PR description

## Project Structure

```
negative_rejection_steering/
├── NRS/
│   └── nodes_NRS.py          # Main NRS node implementation
├── scripts/
│   └── negative_rejection_steering_script.py  # Gradio UI for reForge
├── tests/
│   └── test_smoke.py         # Smoke tests
├── __init__.py               # ComfyUI node exports
├── pyproject.toml            # Project config, dependencies, tool config
├── .pre-commit-config.yaml   # Git hooks configuration
├── .gitignore
├── README.md
├── LICENSE
└── CONTRIBUTING.md           # This file
```

## Getting Help

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/Reithan/negative_rejection_steering/issues)
- **Discussions**: For questions or general discussion
- **Pull Requests**: Review the PR guidelines above before submitting

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to Negative Rejection Steering!
