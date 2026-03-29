"""
PyTorch forward hooks for capturing residual stream activations.

Hooks are registered on ``model.model.layers[i]`` (Qwen2 decoder layers).
Each hook captures the hidden state at the decision token position and
stores it in a shared dict keyed by layer index.

Usage:
    from src.activations.hooks import register_hooks, remove_hooks
    storage, handles = register_hooks(model, layers=range(28))
    model.model(**enc)            # forward pass fills storage
    layer_0_acts = storage[0]     # (batch, hidden_dim) float16
    remove_hooks(handles)
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.logging import get_logger

log = get_logger(__name__)


def _make_hook(
    layer_idx: int,
    storage: dict[int, torch.Tensor],
    token_pos: int = -1,
) -> callable:
    """Create a forward hook that captures the decision token hidden state."""
    def hook(module: nn.Module, input, output):
        # Qwen2DecoderLayer returns a plain Tensor (batch, seq_len, hidden_dim)
        # (older versions returned a tuple; handle both for safety)
        hidden = output[0] if isinstance(output, tuple) else output
        # Immediately slice to decision token, move to CPU as float16
        storage[layer_idx] = hidden[:, token_pos, :].half().detach().cpu()
    return hook


def register_hooks(
    model,
    layers: range | list[int] | None = None,
    token_pos: int = -1,
) -> tuple[dict[int, torch.Tensor], list]:
    """Register forward hooks on transformer decoder layers.

    Args:
        model: Qwen2ForCausalLM. Hooks placed on ``model.model.layers[i]``.
        layers: Layer indices to hook. None = all layers.
        token_pos: Sequence position to extract (-1 = last / decision token).

    Returns:
        storage: Dict populated with {layer_idx: Tensor (batch, hidden_dim) float16}
                 after each forward pass.
        handles: Hook handles — pass to ``remove_hooks()`` when done.
    """
    decoder_layers = model.model.layers
    n_layers = len(decoder_layers)

    if layers is None:
        layers = range(n_layers)

    storage: dict[int, torch.Tensor] = {}
    handles = []

    for i in layers:
        if i < 0 or i >= n_layers:
            raise IndexError(f"Layer {i} out of range [0, {n_layers})")
        h = decoder_layers[i].register_forward_hook(
            _make_hook(i, storage, token_pos=token_pos)
        )
        handles.append(h)

    log.info(f"Registered {len(handles)} hooks on layers {list(layers)[:5]}{'...' if len(list(layers)) > 5 else ''}")
    return storage, handles


def remove_hooks(handles: list) -> None:
    """Remove all registered hooks."""
    for h in handles:
        h.remove()
    log.info(f"Removed {len(handles)} hooks")
