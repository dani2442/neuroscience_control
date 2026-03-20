# Publishing

This repository includes automated workflows for CI, docs, and release publishing.

## Local preflight

Run these before creating a release:

```bash
uv sync --group dev --group docs
uv run pytest --cov=src --cov-report=term-missing
uv run mkdocs build --strict
uv run python -m build
uv run twine check dist/*
```

## TestPyPI publish (manual)

Use **Actions -> Publish -> Run workflow**.
This triggers the `publish-testpypi` job.

## PyPI publish (release)

Create a GitHub Release tag (for example `v0.2.0`).
On `release: published`, workflow uploads `dist/*` to PyPI.

## Required repository setup

1. Configure PyPI and TestPyPI trusted publishing for this repo.
2. Create GitHub Environments: `testpypi` and `pypi`.
3. Configure Codecov for coverage badge/reporting.
4. Enable GitHub Pages to serve `gh-pages` (see `docs/website.md`).
5. Ensure `main` is your default release branch.
