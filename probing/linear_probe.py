"""
Linear probe models for analyzing hidden representations.
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearProbe(nn.Module):
    """
    Simple linear probe for classification tasks on hidden states.

    This probe takes hidden states from a frozen transformer model and applies
    a linear classifier to predict syntactic or semantic properties.

    Args:
        input_dim: Dimension of input hidden states (e.g., 768 for GPT-2)
        num_classes: Number of output classes for classification
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layer (default: True)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_classes, bias=bias)

        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        if bias:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the probe.

        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_len, hidden_dim]
                          or [batch_size, hidden_dim] for single positions

        Returns:
            logits: Classification logits [batch_size, seq_len, num_classes]
                   or [batch_size, num_classes] for single positions
        """
        x = self.dropout(hidden_states)
        logits = self.classifier(x)
        return logits


class MultiLayerProbe(nn.Module):
    """
    Multi-layer probe with one or more hidden layers before classification.

    This probe allows for non-linear transformations of hidden states before
    classification, useful for more complex probing tasks.

    Args:
        input_dim: Dimension of input hidden states
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions (default: [256])
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: 'relu')
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [256]

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.probe = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-layer probe.

        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_len, hidden_dim]
                          or [batch_size, hidden_dim] for single positions

        Returns:
            logits: Classification logits [batch_size, seq_len, num_classes]
                   or [batch_size, num_classes] for single positions
        """
        return self.probe(hidden_states)


class StructuralProbe(nn.Module):
    """
    Structural probe for learning parse tree distances and depths.

    Based on "A Structural Probe for Finding Syntax in Word Representations"
    (Hewitt & Manning, 2019). Projects hidden states to a lower-dimensional
    space where distances correspond to parse tree distances.

    Args:
        input_dim: Dimension of input hidden states
        probe_rank: Rank of the probe matrix (typically 128-512)
    """

    def __init__(self, input_dim: int, probe_rank: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.probe_rank = probe_rank

        # Projection matrix for structural probing
        self.proj = nn.Linear(input_dim, probe_rank, bias=False)

        # Initialize
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to probe space.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            projections: [batch_size, seq_len, probe_rank]
        """
        return self.proj(hidden_states)

    def compute_distance(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared distances in probe space.

        Args:
            projections: [batch_size, seq_len, probe_rank]

        Returns:
            distances: [batch_size, seq_len, seq_len] pairwise squared distances
        """
        # ||h_i - h_j||^2 = ||h_i||^2 + ||h_j||^2 - 2<h_i, h_j>
        norms = (projections ** 2).sum(dim=-1, keepdim=True)  # [B, L, 1]
        dots = torch.bmm(projections, projections.transpose(1, 2))  # [B, L, L]
        distances = norms + norms.transpose(1, 2) - 2 * dots
        return distances

    def compute_depth(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Compute depths (norms) in probe space.

        Args:
            projections: [batch_size, seq_len, probe_rank]

        Returns:
            depths: [batch_size, seq_len] squared norms
        """
        return (projections ** 2).sum(dim=-1)
