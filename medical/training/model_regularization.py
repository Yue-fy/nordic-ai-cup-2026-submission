"""Small, explicit regularization helpers for extractive QA fine-tuning."""

from __future__ import annotations


def freeze_lower_encoder_layers(model, count: int) -> dict[str, int]:
    """Freeze embeddings and the first ``count`` transformer encoder layers."""
    if count < 0:
        raise ValueError("freeze layer count must be nonnegative")
    base_name = getattr(model, "base_model_prefix", "")
    base = getattr(model, base_name, None)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        if count:
            raise ValueError(f"model {type(model).__name__} has no supported encoder.layer stack")
        layers = []
    if count > len(layers):
        raise ValueError(f"cannot freeze {count} of {len(layers)} encoder layers")
    if count:
        embeddings = getattr(base, "embeddings", None)
        if embeddings is None:
            raise ValueError(f"model {type(model).__name__} has no supported embeddings module")
        for parameter in embeddings.parameters():
            parameter.requires_grad = False
        for layer in layers[:count]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    return {
        "encoder_layers": len(layers),
        "frozen_lower_layers": count,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
