import torch.nn as nn
import torch.nn.functional as F
import torch 
from typing import Tuple, Optional
from .rope import RotaryEmbedding
from torch.nn.functional import multi_head_attention_forward
from torch.nn import Transformer

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    ''' 
    def __init__(self, d_model: int, num_heads: int, seq_len:int, dropout: float = 0.0):
        '''
        Initialize the SelfAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.seq_len = seq_len
        assert d_model % num_heads == 0

        #self.rotary = RotaryEmbedding(self.head_dim, 2048) # Initialize rotary embedding
        
        # TODO: Initialize the multi-head attention mechanism (use nn.MultiheadAttention)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        # TODO: Initialize the normalization layer (use nn.LayerNorm)
        self.norm = nn.LayerNorm(d_model)
        
        # TODO: Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)
        


    def forward(self,
                x: torch.Tensor,                     # [B, T, D]
                attn_mask: torch.Tensor = None,      # optional [T, T] or [B*T, T]
                key_padding_mask: torch.Tensor = None # optional [B, T]
               ):
        
        B, T, D = x.shape
        # 1) Pre-norm + residual
        self.input = x
        x = self.norm(x)

        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device) # [T, T]


        x, mha_attn_weights = self.mha.forward(x, x, x, attn_mask = causal_mask)

        x = self.dropout(x) 
        
        x = x + self.input
        
        return x, mha_attn_weights
 
def apply_rotary(x, cos, sin):
    """Apply rotary position embeddings to input tensors."""
    # Split the last dimension in half
    x1, x2 = x.chunk(2, dim=-1)  # Each half becomes head_dim/2
    
    # Make sure cos, sin have the right shape
    # They should be (seq_len, head_dim/2) not (seq_len, head_dim)
    if cos.size(-1) == x.size(-1):  # If cos has full head_dim
        cos = cos[..., :x1.size(-1)]  # Take only first half
        sin = sin[..., :x1.size(-1)]  # Take only first half
    
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # 1, 1, seq_len, head_dim/2
    sin = sin.unsqueeze(0).unsqueeze(0)  # 1, 1, seq_len, head_dim/2
    
    # Apply RoPE via complex-number multiplication
    result = torch.cat([
        x1 * cos - x2 * sin,  # real component
        x2 * cos + x1 * sin   # imaginary component
    ], dim=-1)
    
    return result.type_as(x)

## -------------------------------------------------------------------------------------------------  
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer. 
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # TODO: Implement __init__

        # TODO: Initialize the feed-forward network (use nn.Sequential)
        # See writeup for what layers to use
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # TODO: Initialize the normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # TODO: Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)
       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
        ''' 
        input = x
        
        x = self.norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = input + x
        
        return x
    
