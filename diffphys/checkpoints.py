"""Strict, self-describing checkpoints; no automatic model selection."""

import torch
from pathlib import Path

from .models import build_model
from .profiles import get_profile


def save_checkpoint(path, model, profile, iteration, *, run_config, optimizer=None, scheduler=None):
    payload = dict(
        format_version=1,
        profile=profile.name,
        input_config=profile.metadata(),
        model_state_dict=model.state_dict(),
        iteration=iteration,
        run_config=run_config,
    )
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, profile, device="cpu"):
    spec = get_profile(profile) if isinstance(profile, str) else profile
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA checkpoint inference requested but CUDA is unavailable")
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Checkpoint must contain versioned profile/input metadata; raw legacy weights are ambiguous")
    if payload.get("profile") != spec.name or payload.get("input_config") != spec.metadata():
        raise ValueError("Checkpoint profile/input contract does not match the requested profile")
    weights = payload.get("model_state_dict")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Checkpoint has no model_state_dict")
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Checkpoint weights must be finite tensors")
    model = build_model(spec).to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.inference_mode():
        image = torch.zeros((2, 1, spec.image_height, spec.image_width), device=device)
        state = torch.zeros((2, spec.state_dim), device=device)
        action, _, hidden = model(image, state)
        if action.shape != (2, 6) or hidden.shape != (2, 192):
            raise ValueError("Checkpoint has incompatible output shapes")
        if not torch.isfinite(action).all() or not torch.isfinite(hidden).all():
            raise ValueError("Checkpoint failed finite forward validation")
    return model, payload
