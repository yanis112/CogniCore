# Complete Guide: Transform Python Project to PyPI Module

This guide explains how to transform any Python project into a deployable PyPI package.

## 1. Project Structure Requirements

Your final project structure should look like this:
```
your_project/
├── pyproject.toml          # Main configuration file (required)
├── README.md               # Project description (recommended)
├── LICENSE                 # License file (recommended)
├── .gitignore             # Git ignore file (recommended)
├── your_package_name/     # Main package directory
│   ├── __init__.py        # Package initialization (required)
│   ├── main.py           # Main module code
│   └── submodule.py      # Additional modules
├── tests/                 # Test directory (recommended)
│   ├── __init__.py
│   └── test_main.py
└── dist/                  # Build artifacts (auto-generated)
```

## 2. Required Files and Their Contents

### 2.1 pyproject.toml (REQUIRED)

This is the most important file that defines your package metadata and dependencies:

```toml
[build-system]
requires = ["hatchling"]  # or "setuptools>=61.0" or "flit_core >=3.2,<4"
build-backend = "hatchling.build"  # or "setuptools.build_meta" or "flit_core.buildapi"

[project]
name = "your-package-name"  # Must be unique on PyPI
version = "0.1.0"
description = "A brief description of your package"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}  # or {file = "LICENSE"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["keyword1", "keyword2", "keyword3"]
dependencies = [
    "requests>=2.25.0",
    "numpy>=1.20.0",
    # Add your dependencies here
]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "isort>=5.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/your-package-name"
Repository = "https://github.com/yourusername/your-package-name"
Documentation = "https://your-package-name.readthedocs.io"
"Bug Reports" = "https://github.com/yourusername/your-package-name/issues"

[project.scripts]
# Optional: CLI commands
your-command = "your_package_name.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["your_package_name"]
```

### 2.2 your_package_name/__init__.py (REQUIRED)

This file makes your directory a Python package:

```python
"""
Your Package Name

A brief description of what your package does.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes/functions for easy access
from .main import YourMainClass, your_main_function

# Define what gets imported with "from your_package_name import *"
__all__ = [
    "YourMainClass",
    "your_main_function",
]
```

### 2.3 README.md (STRONGLY RECOMMENDED)

```markdown
# Your Package Name

Brief description of your package.

## Installation

```bash
pip install your-package-name
```

## Quick Start

```python
from your_package_name import YourMainClass

# Example usage
instance = YourMainClass()
result = instance.do_something()
print(result)
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Documentation

Full documentation available at [link to docs]

## Contributing

Contributions are welcome! Please read our contributing guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

### 2.4 LICENSE (STRONGLY RECOMMENDED)

For MIT License:
```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 2.5 .gitignore (RECOMMENDED)

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/
```

## 3. Development Setup

### 3.1 Install Build Tools

Using UV (recommended):
```bash
uv add --dev build twine pytest pytest-cov black isort
```

Using pip:
```bash
pip install build twine pytest pytest-cov black isort
```

### 3.2 Create Virtual Environment

Using UV:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## 4. Testing Your Package

### 4.1 Install in Development Mode

```bash
uv pip install -e .
# or
pip install -e .
```

### 4.2 Run Tests

```bash
pytest
# or with coverage
pytest --cov=your_package_name
```

## 5. Building the Package

### 5.1 Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info/
```

### 5.2 Build

```bash
python -m build
```

This creates:
- `dist/your_package_name-0.1.0.tar.gz` (source distribution)
- `dist/your_package_name-0.1.0-py3-none-any.whl` (wheel distribution)

## 6. Publishing to PyPI

### 6.1 Test on Test PyPI First

1. Create account on https://test.pypi.org/
2. Generate API token
3. Upload:

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

4. Test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ your-package-name
```

### 6.2 Publish to Real PyPI

1. Create account on https://pypi.org/
2. Generate API token
3. Upload:

```bash
python -m twine upload dist/*
```

## 7. Version Management

### 7.1 Semantic Versioning

Follow semantic versioning (semver.org):
- `1.0.0` - Major version (breaking changes)
- `1.1.0` - Minor version (new features, backward compatible)
- `1.1.1` - Patch version (bug fixes)

### 7.2 Update Version

1. Update version in `pyproject.toml`
2. Update version in `your_package_name/__init__.py`
3. Create git tag:

```bash
git tag v1.0.1
git push origin v1.0.1
```

4. Rebuild and republish

## 8. Complete File Checklist

### Required Files:
- [ ] `pyproject.toml` - Package metadata and configuration
- [ ] `your_package_name/__init__.py` - Package initialization
- [ ] `your_package_name/main.py` - Your main code

### Strongly Recommended:
- [ ] `README.md` - Package description and usage
- [ ] `LICENSE` - License file
- [ ] `.gitignore` - Git ignore rules

### Recommended:
- [ ] `tests/` - Test directory with test files
- [ ] `CHANGELOG.md` - Version history
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `requirements.txt` or `requirements-dev.txt` - Dependencies

### Optional:
- [ ] `docs/` - Documentation directory
- [ ] `examples/` - Example scripts
- [ ] `.github/workflows/` - CI/CD workflows
- [ ] `Makefile` - Build automation
- [ ] `tox.ini` - Testing across Python versions

## 9. Common Issues and Solutions

### 9.1 Package Name Already Exists
- Check availability on PyPI before choosing name
- Use unique, descriptive names
- Consider adding your username as prefix

### 9.2 Import Errors
- Ensure `__init__.py` exists in all package directories
- Check your import statements
- Verify package structure

### 9.3 Version Conflicts
- Always increment version before publishing
- Can't overwrite existing versions on PyPI
- Use test PyPI for testing

### 9.4 Missing Dependencies
- List all dependencies in `pyproject.toml`
- Specify minimum versions
- Test in clean environment

## 10. Best Practices

1. **Start Simple**: Begin with minimal required files
2. **Test Thoroughly**: Test on multiple Python versions
3. **Document Well**: Good README and docstrings
4. **Version Properly**: Follow semantic versioning
5. **Use CI/CD**: Automate testing and publishing
6. **Security**: Use API tokens, not passwords
7. **Backup**: Keep source code in version control

This guide provides everything needed to transform any Python project into a professional PyPI package.
