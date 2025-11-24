"""
Probing utilities for analyzing learned representations in transformer models.

This package provides tools for linear probing of hidden states to understand
what linguistic or syntactic properties are captured at different layers.
"""

from .linear_probe import LinearProbe, MultiLayerProbe
from .hidden_state_extractor import HiddenStateExtractor
from .probe_trainer import ProbeTrainer
from .probing_datasets import ProbingDataset, CFGProbingDataset
from .eval_probe import evaluate_probe, compute_probe_metrics

__all__ = [
    'LinearProbe',
    'MultiLayerProbe',
    'HiddenStateExtractor',
    'ProbeTrainer',
    'ProbingDataset',
    'CFGProbingDataset',
    'evaluate_probe',
    'compute_probe_metrics',
]
