# Website Deployment

This project publishes docs to GitHub Pages from the `main` branch via the Docs workflow.

## Live URL

Default GitHub Pages URL for this repository:

```text
https://dani2442.github.io/neuroscience_control/
```

## How deployment works

1. Push changes to `main` that touch `docs/**`, `mkdocs.yml`, `README.md`, or `pyproject.toml`.
2. GitHub Action `.github/workflows/docs.yml` runs:
   - `uv sync --group docs`
   - `uv run mkdocs build --strict`
   - deploys `site/` to `gh-pages`
3. GitHub Pages serves the `gh-pages` branch.

## One-time GitHub setup

1. Open repository settings.
2. Go to **Pages**.
3. Set **Source** to **Deploy from a branch**.
4. Select branch `gh-pages` and folder `/ (root)`.
5. Save.

## Local preview

```bash
uv sync --group docs
uv run mkdocs serve
```

Then open:

```text
http://127.0.0.1:8000/
```

## Troubleshooting

- If Pages is blank, verify `gh-pages` exists and has `index.html`.
- If docs action fails, run `uv run mkdocs build --strict` locally and fix warnings/errors.
- If workflow updates fail to push, ensure your GitHub token has `workflow` scope.
