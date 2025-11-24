import torch
from torch import nn
import math

"""
Positional Encoding Module.

Specification:
- Module adds positional information to input embeddings
- Uses sinusoidal position encodings as described in "Attention Is All You Need"
- Positional encoding matrix has shape (1, max_len, d_model)
- Even indices use sine functions, odd indices use cosine functions
- Wavelengths form geometric progression from 2π to 10000·2π
- Encoding values are on same device as input tensor
- Handles any sequence length up to max_len
- Raises error if input sequence length exceeds max_len

Note: This module is currently not used in the main model (RoPE is used instead).
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Steps:
        1. Call parent class constructor using super().__init__()
        2. Call create_pe_table to initialize positional encoding matrix
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Side Effects:
            - Initializes the positional encoding buffer 'pe' 
              of shape (1, max_len, d_model) (in order to broadcast with input tensor)
        """
        
        pe = torch.zeros((max_len, d_model))
        
        pos = torch.arange(max_len).unsqueeze(1) #(max_len, 1)
        
        div = (10000.0) ** (torch.arange(0, d_model, 2)/d_model) #(d_model/2)
        
        
        pe[:, 0::2] = torch.sin(pos / div) 
        pe[:, 1::2] = torch.cos(pos / div)
        
        
        pe = pe.unsqueeze(0) #(1, max_len, d_model)
        #self.pe = pe
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length
        """
        # Get sequence length from input tensor
        seq_len = x.size(1)
        # Verify sequence length doesn't exceed maximum length
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Add positional encodings to input
        out = self.pe[:, :x.size(1), :] + x
        return out
