"""Shared runtime helpers for CLI scripts."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import wandb

DEFAULT_PROXY_URL = "http://proxy.nhr.fau.de:80"


def ensure_proxy_env(proxy_url: str = DEFAULT_PROXY_URL) -> None:
    """Configure proxy environment variables when they are not set."""
    os.environ.setdefault("HTTP_PROXY", proxy_url)
    os.environ.setdefault("HTTPS_PROXY", proxy_url)


def resolve_device(device: str = "auto") -> str:
    """Resolve runtime device from an explicit value or auto-detection."""
    if device != "auto":
        return device

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"

    print("Using CPU")
    return "cpu"


def seed_all(seed: int) -> None:
    """Seed supported random number generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def print_section(title: str) -> None:
    """Print a consistent section banner."""
    separator = "=" * 60
    print(f"\n{separator}")
    print(title)
    print(separator)


def wandb_active(use_wandb: bool = True) -> bool:
    """Return whether wandb logging should happen now."""
    return use_wandb and wandb.run is not None


def init_wandb_run(
    *,
    use_wandb: bool,
    project: str,
    run_name: str,
    entity: str | None = None,
    config: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
):
    """Initialize a wandb run when enabled."""
    if not use_wandb:
        return None

    ensure_proxy_env()
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=dict(config) if config is not None else None,
        settings=wandb.Settings(_service_transport="http"),
        tags=list(tags) if tags is not None else None,
    )
    print(f"Wandb initialized: {project}/{run_name}")
    return run


def wandb_log(data: Mapping[str, Any], *, use_wandb: bool = True) -> None:
    """Log a payload to wandb when active."""
    if wandb_active(use_wandb):
        wandb.log(dict(data))


def wandb_summary_update(data: Mapping[str, Any], *, use_wandb: bool = True) -> None:
    """Update wandb summary when active."""
    if wandb_active(use_wandb):
        wandb.summary.update(dict(data))


def wandb_log_figure(key: str, fig: Any, *, use_wandb: bool = True) -> None:
    """Log a matplotlib figure as wandb image when active."""
    if wandb_active(use_wandb):
        wandb.log({key: wandb.Image(fig)})


def wandb_log_artifact(
    artifact_name: str,
    file_path: str | Path,
    *,
    artifact_type: str = "model",
    use_wandb: bool = True,
) -> None:
    """Log a file-backed artifact when wandb is active."""
    if not wandb_active(use_wandb):
        return

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(file_path))
    wandb.log_artifact(artifact)


def finish_wandb_run() -> None:
    """Finish wandb run if one is currently active."""
    if wandb.run is not None:
        wandb.finish()


@contextmanager
def managed_wandb_run(**init_kwargs):
    """Initialize and always finish a wandb run."""
    init_wandb_run(**init_kwargs)
    try:
        yield
    finally:
        finish_wandb_run()
