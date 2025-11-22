# Contributing to nadoo-flow-core

Thank you for your interest in contributing to nadoo-flow-core! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment
4. Create a new branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip
- git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/nadoo-flow-core.git
cd nadoo-flow-core

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-node-type` - for new features
- `fix/streaming-bug` - for bug fixes
- `docs/update-readme` - for documentation
- `test/add-integration-tests` - for tests

### Commit Messages

Follow conventional commits:
```
feat: add parallel execution support
fix: resolve streaming timeout issue
docs: update quick start guide
test: add tests for chaining API
refactor: simplify workflow executor
```

## Code Style

We use automated tools to maintain code quality:

### Formatting

```bash
# Format code with Black
black .

# Check formatting
black --check .
```

### Linting

```bash
# Lint with Ruff
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking

```bash
# Type check with mypy
mypy src/nadoo_flow
```

### Run All Checks

```bash
# Using Makefile
make lint

# Or manually
black --check .
ruff check .
mypy src/nadoo_flow
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nadoo_flow --cov-report=html

# Run specific test file
pytest tests/test_chaining.py

# Run specific test
pytest tests/test_chaining.py::test_pipe_operator

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use `pytest` fixtures for setup/teardown
- Aim for high coverage of new code

Example:
```python
import pytest
from nadoo_flow import ChainableNode, NodeResult

@pytest.mark.asyncio
async def test_my_feature():
    node = MyNode()
    result = await node.run({"input": "test"})
    assert result["output"] == "expected"
```

## Submitting Changes

### Pull Request Process

1. **Update your branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all checks**
   ```bash
   make lint
   pytest
   ```

3. **Push to your fork**
   ```bash
   git push origin your-branch
   ```

4. **Create Pull Request**
   - Go to GitHub and create a PR
   - Fill out the PR template
   - Link any related issues

### PR Requirements

- ✅ All tests pass
- ✅ Code is formatted (Black)
- ✅ No linting errors (Ruff)
- ✅ Type checks pass (mypy)
- ✅ New code has tests
- ✅ Documentation is updated
- ✅ CHANGELOG.md is updated (for significant changes)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## Reporting Issues

### Bug Reports

Use the bug report template and include:
- Clear description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version
- nadoo-flow-core version
- Code example (if applicable)

### Feature Requests

Use the feature request template and include:
- Clear description of the feature
- Use case / motivation
- Proposed API (if applicable)
- Alternatives considered

## Development Workflow

### Making a Bug Fix

```bash
# Create branch
git checkout -b fix/describe-the-bug

# Make changes
# ... edit files ...

# Add tests
# ... create test_bug_fix.py ...

# Run tests
pytest

# Format and lint
black .
ruff check .

# Commit
git add .
git commit -m "fix: resolve issue with X"

# Push and create PR
git push origin fix/describe-the-bug
```

### Adding a New Feature

```bash
# Create branch
git checkout -b feature/my-new-feature

# Implement feature
# ... edit src/nadoo_flow/... ...

# Add tests
# ... edit tests/test_my_feature.py ...

# Add documentation
# ... update README.md or docs/ ...

# Update CHANGELOG
# ... edit CHANGELOG.md ...

# Run all checks
make lint
pytest

# Commit and push
git add .
git commit -m "feat: add new feature X"
git push origin feature/my-new-feature
```

## Questions?

- Open a [GitHub Discussion](https://github.com/nadoo-ai/nadoo-flow-core/discussions)
- Check existing [Issues](https://github.com/nadoo-ai/nadoo-flow-core/issues)
- Email: dev@nadoo.ai

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
