# A Guide to a Professional & Modern GitHub Project

This guide provides a comprehensive walkthrough for setting up a professional, modern, and automated Python project on GitHub. It uses best practices observed in successful open-source projects, including automated packaging and publishing.

---

## 1. The README: Your Project's Front Door

Your `README.md` is the first thing users see. It should be clear, concise, and visually appealing.

### Key Sections
A strong README includes:
- **Logo/Banner**: A hero image that gives your project a brand.
- **Badges**: Quick, visual indicators of project status.
- **Project Title & Pitch**: A clear, one-sentence description of what your project does.
- **Core Principles**: What makes your project unique? (e.g., fast, modular, free).
- **Features**: A bulleted list of key capabilities.
- **Installation**: A simple, copy-pasteable command.
- **Configuration**: How to set up necessary environment variables or API keys.
- **Quick Start**: A minimal, working example to get users started immediately.
- **Advanced Usage**: More complex examples showing the full power of your library.
- **API/Model Reference**: A table or list of available classes, functions, or models.

### Adding Professional Badges
Badges provide at-a-glance information. Use [Shields.io](https://shields.io/) to generate them.

**Example Badge Markdown:**
```markdown
<!-- PyPI Version -->
<a href="https://pypi.org/project/your-package-name/"><img src="https://img.shields.io/pypi/v/your-package-name?color=blue&label=PyPI&logo=pypi" alt="PyPI"></a>

<!-- GitHub Actions CI/CD Status -->
<a href="https://github.com/your-username/your-repo/actions"><img src="https://github.com/your-username/your-repo/actions/workflows/your-workflow-file.yml/badge.svg" alt="CI/CD Status"></a>

<!-- License -->
<a href="LICENSE"><img src="https://img.shields.io/github/license/your-username/your-repo" alt="License"></a>

<!-- Python Version Support -->
<a href="https://pypi.org/project/your-package-name/"><img src="https://img.shields.io/pypi/pyversions/your-package-name" alt="Python Versions"></a>
```

---

## 2. Versioning: Semantic & Automated

Stop hardcoding versions. Modern Python projects derive the version number directly from your Git history. This ensures your package version always matches your codebase.

### Semantic Versioning
Use the `MAJOR.MINOR.PATCH` format (e.g., `v1.2.1`):
- **MAJOR**: Incompatible API changes.
- **MINOR**: New, backward-compatible features.
- **PATCH**: Backward-compatible bug fixes.

### Dynamic Versioning with `pyproject.toml`
Use `hatch-vcs` to automatically version your package from Git tags. When you create a tag like `v0.2.0`, `hatch` will use that as the package version.

**Setup in `pyproject.toml`:**

```toml
[build-system]
# Require hatchling and the vcs plugin
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "your-package-name"
# Tell hatch the version is dynamic
dynamic = ["version"]
# ... other project metadata

# Configure hatch to use git tags for versioning
[tool.hatch.version]
source = "vcs"

# (Optional but recommended) Write the version to a file
# This makes it accessible in your code at runtime (e.g., from my_package import __version__)
[tool.hatch.build.hooks.vcs]
version-file = "your_package_name/__version__.py"
```

---

## 3. CI/CD: Automate Publishing with GitHub Actions

Automate your release process. This workflow will automatically build and publish your package to PyPI whenever you push a new Git tag.

### Workflow File
Create a file at `.github/workflows/publish.yml`:

```yaml
name: Publish Python Package to PyPI

# Trigger the workflow when a tag starting with 'v' is pushed
on:
  push:
    tags:
      - 'v*'
  # Allow manual runs for testing
  workflow_dispatch:

# Use trusted publishing (OIDC) for security - no API tokens needed!
permissions:
  id-token: write

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      # 1. Check out the repository code
      - name: Check out the repository
        uses: actions/checkout@v4
        with:
          # Fetch all history so hatch-vcs can determine the version
          fetch-depth: 0

      # 2. Set up a specific Python version
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      # 3. Install uv, the fast Python package manager
      - name: Set up uv
        uses: astral-sh/setup-uv@v1

      # 4. Install build dependencies
      - name: Install dependencies
        run: uv pip install build

      # 5. Build the package
      - name: Build package
        run: python -m build

      # 6. Publish to PyPI using trusted publishing
      - name: Publish package to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # No API token needed here, it's more secure!
```

### How It Works
1.  **Finalize Code**: Merge all your changes into your main branch.
2.  **Commit**: `git commit -m "feat: Add new feature"`
3.  **Tag**: `git tag v0.2.0` (This version number MUST match semantic versioning).
4.  **Push**: `git push && git push --tags`

Pushing the tag automatically triggers the GitHub Action, and your new version goes live on PyPI. You must also configure PyPI to trust your GitHub repository. You can find instructions in the PyPI documentation for "Trusted Publishers".
