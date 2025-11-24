"""
Dataset classes for probing tasks on CFG data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add parent directory to path to import from data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.grammars import GRAMMAR_CFG3b, GRAMMAR_SIMPLE
    from data.annotate_tree import parse_grammar, next_token_prob
except ImportError:
    # If imports fail, user needs to ensure data module is accessible
    pass


class ProbingDataset(Dataset):
    """
    Base dataset for probing tasks.

    Args:
        hidden_states: Hidden states from model [num_samples, hidden_dim]
        labels: Target labels for classification [num_samples]
        mask: Optional mask for valid positions [num_samples]
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        self.hidden_states = hidden_states
        self.labels = labels
        self.mask = mask

        if len(self.hidden_states) != len(self.labels):
            raise ValueError("hidden_states and labels must have same length")

        if mask is not None and len(self.mask) != len(self.labels):
            raise ValueError("mask and labels must have same length")

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx]
        }

        if self.mask is not None:
            item['mask'] = self.mask[idx]

        return item


class CFGProbingDataset(Dataset):
    """
    Probing dataset for CFG sequences with syntactic labels.

    This dataset creates probing tasks from CFG-generated sequences, such as:
    - Next token prediction based on grammatical constraints
    - Depth in parse tree
    - Non-terminal type at current position
    - Whether current position starts/ends a constituent

    Args:
        sequences: Token sequences [num_sequences, seq_len]
        hidden_states: Pre-extracted hidden states [num_sequences, seq_len, hidden_dim]
        grammar: The CFG grammar used to generate sequences
        task: Type of probing task ('next_token', 'depth', 'nonterminal', 'constituent')
        parse_annotations: Optional pre-computed parse annotations
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        hidden_states: torch.Tensor,
        grammar: str = 'cfg3b',
        task: str = 'next_token',
        parse_annotations: Optional[List] = None
    ):
        self.sequences = sequences
        self.hidden_states = hidden_states
        self.task = task

        if sequences.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("sequences and hidden_states must have same shape in first two dims")

        # Load grammar
        if grammar == 'cfg3b':
            self.grammar = GRAMMAR_CFG3b
        elif grammar == 'simple':
            self.grammar = GRAMMAR_SIMPLE
        else:
            raise ValueError(f"Unknown grammar: {grammar}")

        # Parse grammar if needed for certain tasks
        self.parsed_grammar = None
        if task in ['next_token', 'nonterminal']:
            try:
                self.parsed_grammar = parse_grammar(self.grammar)
            except:
                print("Warning: Could not parse grammar for task")

        self.parse_annotations = parse_annotations

        # Create labels based on task
        self.labels = self._create_labels()

    def _create_labels(self) -> torch.Tensor:
        """
        Create labels for the probing task.

        Returns:
            labels: Target labels [num_sequences, seq_len]
        """
        num_sequences, seq_len = self.sequences.shape

        if self.task == 'next_token':
            # Predict next token (shifted by 1)
            # Labels are the next token in sequence
            labels = torch.zeros_like(self.sequences)
            labels[:, :-1] = self.sequences[:, 1:]
            labels[:, -1] = -100  # Ignore last position
            return labels

        elif self.task == 'depth':
            # Predict depth in parse tree (requires parse annotations)
            if self.parse_annotations is None:
                raise ValueError("parse_annotations required for depth task")
            # TODO: Implement depth extraction from parse trees
            labels = torch.zeros(num_sequences, seq_len, dtype=torch.long)
            return labels

        elif self.task == 'nonterminal':
            # Predict which non-terminal generated this position
            # This requires parsing each sequence
            if self.parsed_grammar is None:
                raise ValueError("Could not parse grammar for nonterminal task")
            # TODO: Implement non-terminal labeling
            labels = torch.zeros(num_sequences, seq_len, dtype=torch.long)
            return labels

        elif self.task == 'constituent':
            # Binary classification: does a constituent start/end here?
            # TODO: Implement constituent boundary detection
            labels = torch.zeros(num_sequences, seq_len, dtype=torch.long)
            return labels

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example for probing.

        Returns:
            Dictionary with:
                - hidden_states: [seq_len, hidden_dim]
                - labels: [seq_len]
                - sequence: [seq_len] (original tokens)
        """
        return {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx],
            'sequence': self.sequences[idx]
        }


class PositionwiseProbingDataset(Dataset):
    """
    Flattened probing dataset where each position is a separate example.

    This is useful when you want to treat each token position independently
    rather than processing entire sequences.

    Args:
        hidden_states: Hidden states [num_sequences, seq_len, hidden_dim]
        labels: Labels [num_sequences, seq_len]
        mask: Optional mask for valid positions [num_sequences, seq_len]
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        # Flatten sequence dimension
        num_sequences, seq_len, hidden_dim = hidden_states.shape
        self.hidden_states = hidden_states.reshape(-1, hidden_dim)
        self.labels = labels.reshape(-1)

        if mask is not None:
            mask = mask.reshape(-1)
            # Only keep valid positions
            valid_indices = mask.bool()
            self.hidden_states = self.hidden_states[valid_indices]
            self.labels = self.labels[valid_indices]

    def __len__(self) -> int:
        return len(self.hidden_states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx]
        }


def create_next_token_probing_data(
    sequences: Union[torch.Tensor, np.ndarray],
    hidden_states: Union[torch.Tensor, np.ndarray],
    ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create next-token prediction probing data from sequences and hidden states.

    Args:
        sequences: Token sequences [num_sequences, seq_len]
        hidden_states: Hidden states [num_sequences, seq_len, hidden_dim]
        ignore_index: Label to use for positions to ignore (default: -100)

    Returns:
        (features, labels):
            - features: [num_sequences, seq_len, hidden_dim]
            - labels: [num_sequences, seq_len] next token labels
    """
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences)
    if isinstance(hidden_states, np.ndarray):
        hidden_states = torch.from_numpy(hidden_states)

    num_sequences, seq_len = sequences.shape

    # Labels are shifted by 1 (predict next token)
    labels = torch.full((num_sequences, seq_len), ignore_index, dtype=torch.long)
    labels[:, :-1] = sequences[:, 1:]

    return hidden_states, labels


def create_depth_probing_data(
    sequences: Union[torch.Tensor, np.ndarray],
    hidden_states: Union[torch.Tensor, np.ndarray],
    grammar: str = 'cfg3b',
    max_depth: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create parse tree depth probing data (placeholder for future implementation).

    Args:
        sequences: Token sequences [num_sequences, seq_len]
        hidden_states: Hidden states [num_sequences, seq_len, hidden_dim]
        grammar: Grammar name to use for parsing
        max_depth: Maximum depth to consider

    Returns:
        (features, labels):
            - features: [num_sequences, seq_len, hidden_dim]
            - labels: [num_sequences, seq_len] depth labels
    """
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences)
    if isinstance(hidden_states, np.ndarray):
        hidden_states = torch.from_numpy(hidden_states)

    # Placeholder: return zeros for depths
    # TODO: Implement actual depth computation from parse trees
    num_sequences, seq_len = sequences.shape
    labels = torch.zeros(num_sequences, seq_len, dtype=torch.long)

    print("Warning: Depth computation not yet implemented, returning placeholder labels")

    return hidden_states, labels
