"""
Utilities for extracting hidden states from frozen transformer models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from collections import defaultdict


class HiddenStateExtractor:
    """
    Extract hidden states from specific layers of a transformer model.

    This class registers forward hooks to capture intermediate hidden states
    during a forward pass, allowing for probing of different layers.

    Args:
        model: The transformer model to extract hidden states from
        layer_ids: List of layer indices to extract from (default: all layers)
        extract_attention: Whether to also extract attention weights (default: False)

    Example:
        ```python
        extractor = HiddenStateExtractor(model, layer_ids=[0, 6, 11])
        hidden_states = extractor.extract_states(input_ids)
        # hidden_states = {0: tensor, 6: tensor, 11: tensor}
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        layer_ids: Optional[List[int]] = None,
        extract_attention: bool = False
    ):
        self.model = model
        self.extract_attention = extract_attention
        self.hidden_states = {}
        self.attention_weights = {}
        self.hooks = []

        # Determine which layers to extract from
        if layer_ids is None:
            # Extract from all decoder layers
            if hasattr(model, 'decoder_layers'):
                self.layer_ids = list(range(len(model.decoder_layers)))
            else:
                raise ValueError("Model must have 'decoder_layers' attribute")
        else:
            self.layer_ids = layer_ids

        # Freeze the model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        if not hasattr(self.model, 'decoder_layers'):
            raise ValueError("Model must have 'decoder_layers' attribute")

        for layer_id in self.layer_ids:
            if layer_id >= len(self.model.decoder_layers):
                raise ValueError(f"Layer {layer_id} does not exist in model")

            layer = self.model.decoder_layers[layer_id]

            # Hook to capture hidden states after each layer
            def make_hook(lid):
                def hook(module, input, output):
                    # Output is typically the hidden states
                    if isinstance(output, tuple):
                        hidden = output[0]
                        if self.extract_attention and len(output) > 1:
                            self.attention_weights[lid] = output[1]
                    else:
                        hidden = output
                    self.hidden_states[lid] = hidden.detach()
                return hook

            hook_handle = layer.register_forward_hook(make_hook(layer_id))
            self.hooks.append(hook_handle)

    def extract_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from the model for given inputs.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            hidden_states: Dictionary mapping layer_id -> hidden states tensor
                          Each tensor has shape [batch_size, seq_len, hidden_dim]
        """
        # Clear previous states
        self.hidden_states = {}
        self.attention_weights = {}

        with torch.no_grad():
            # Forward pass through model
            if attention_mask is not None:
                _ = self.model(input_ids, attention_mask=attention_mask)
            else:
                _ = self.model(input_ids)

        return self.hidden_states

    def extract_states_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Extract hidden states for an entire dataset.

        Args:
            dataloader: DataLoader providing input batches
            max_batches: Maximum number of batches to process (default: all)
            device: Device to run extraction on (default: model's device)

        Returns:
            all_states: Dictionary mapping layer_id -> list of hidden state tensors
        """
        if device is None:
            device = next(self.model.parameters()).device

        all_states = defaultdict(list)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch[0].to(device)
                attention_mask = None

            # Extract states
            states = self.extract_states(input_ids, attention_mask)

            # Store states
            for layer_id, hidden in states.items():
                all_states[layer_id].append(hidden.cpu())

        return dict(all_states)

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """
        Get attention weights from the last forward pass.

        Returns:
            attention_weights: Dictionary mapping layer_id -> attention weights
        """
        if not self.extract_attention:
            raise ValueError("extract_attention must be True to get attention weights")
        return self.attention_weights

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Clean up hooks when extractor is deleted."""
        self.remove_hooks()


def extract_hidden_states_for_probing(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_ids: List[int],
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[int, torch.Tensor]:
    """
    Convenience function to extract and concatenate hidden states for probing.

    Args:
        model: Transformer model
        dataloader: DataLoader with input data
        layer_ids: List of layer indices to extract
        max_samples: Maximum number of samples to extract
        device: Device to use for extraction

    Returns:
        states: Dictionary mapping layer_id -> concatenated hidden states
               Each tensor has shape [total_tokens, hidden_dim]
    """
    extractor = HiddenStateExtractor(model, layer_ids=layer_ids)

    if device is None:
        device = next(model.parameters()).device

    all_states = defaultdict(list)
    total_samples = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = batch[0].to(device)
            attention_mask = None

        # Extract states
        states = extractor.extract_states(input_ids, attention_mask)

        # Flatten and store
        batch_size, seq_len = input_ids.shape
        for layer_id, hidden in states.items():
            # Reshape to [batch_size * seq_len, hidden_dim]
            hidden_flat = hidden.reshape(-1, hidden.size(-1))
            all_states[layer_id].append(hidden_flat.cpu())

        total_samples += batch_size * seq_len
        if max_samples is not None and total_samples >= max_samples:
            break

    # Concatenate all batches
    concatenated_states = {
        layer_id: torch.cat(tensors, dim=0)[:max_samples] if max_samples else torch.cat(tensors, dim=0)
        for layer_id, tensors in all_states.items()
    }

    extractor.remove_hooks()
    return concatenated_states
