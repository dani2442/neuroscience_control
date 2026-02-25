"""Model checkpoint serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base_model import BaseNeuroscienceModel
from .hopf_model import CoupledHopfModel
from .hybrid_hopf_model import HybridHopfModel
from .neural_sde import NeuralSDE


MODEL_REGISTRY = {
    "CoupledHopfModel": CoupledHopfModel,
    "HybridHopfModel": HybridHopfModel,
    "NeuralSDE": NeuralSDE,
}


def _read_checkpoint(checkpoint_path: Path | str, device: str) -> dict[str, Any]:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {checkpoint_path}")

    missing = [key for key in BaseNeuroscienceModel.REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Legacy or invalid checkpoint (missing: {missing_str}): {checkpoint_path}"
        )

    checkpoint_version = checkpoint.get("checkpoint_version")
    if not isinstance(checkpoint_version, int) or checkpoint_version < 2:
        raise ValueError(
            f"Unsupported checkpoint_version={checkpoint_version} at {checkpoint_path}"
        )

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint model_config must be a dictionary.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict must be a dictionary.")

    return checkpoint


def _build_model(
    model_class: str,
    model_config: dict[str, Any],
    device: str,
) -> BaseNeuroscienceModel:
    if model_class not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported checkpoint model class: {model_class}")

    config = dict(model_config)
    if model_class == "NeuralSDE":
        has_sc = bool(config.pop("has_structural_connectivity", False))
        if has_sc and "structural_connectivity" not in config:
            n_rois = int(config["n_rois"])
            config["structural_connectivity"] = torch.eye(n_rois, device=device)
    else:
        config.pop("has_structural_connectivity", None)

    config["device"] = device
    try:
        model = MODEL_REGISTRY[model_class](**config)
    except TypeError as exc:
        raise ValueError(f"Invalid model_config for {model_class}: {exc}") from exc
    return model


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    device: str = "cpu",
    strict: bool = True,
) -> tuple[BaseNeuroscienceModel, str, dict[str, Any]]:
    """
    Load a model from checkpoint and reconstruct the architecture.

    Returns:
        Tuple of (model, model_name, metadata).
    """
    checkpoint = _read_checkpoint(checkpoint_path, device=device)
    model_name = str(checkpoint["model_class"])
    model = _build_model(model_name, checkpoint["model_config"], device=device)
    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    metadata = checkpoint.get("metadata", {})
    if metadata is None:
        return model, model_name, {}
    if not isinstance(metadata, dict):
        return model, model_name, {"value": metadata}
    return model, model_name, metadata
