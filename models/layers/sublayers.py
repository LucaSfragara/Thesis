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
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
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
        assert d_model % num_heads == 0

        self.rotary = RotaryEmbedding(self.head_dim, 2048) # Initialize rotary embedding
        
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
        
        B, T, D = x.size()
        H, hD   = self.num_heads, self.head_dim

        # 1) Pre-norm + residual
        residual = x
        x = self.norm(x)

        # 2) Manually split in_proj_weight/bias into Q, K, V
        #    in_proj_weight is [3*D, D], bias is [3*D]
        w_q, w_k, w_v = self.mha.in_proj_weight.chunk(3, dim=0)
        if self.mha.in_proj_bias is not None:
            b_q = self.mha.in_proj_bias[      :   D]
            b_k = self.mha.in_proj_bias[  D  : 2*D]
            b_v = self.mha.in_proj_bias[2*D : 3*D]
        else:
            b_q = b_k = b_v = None

        # 3) Project x → Q, K, V each [B, T, D]
        q = F.linear(x, w_q, b_q)
        k = F.linear(x, w_k, b_k)
        v = F.linear(x, w_v, b_v)

        # 4) Reshape into (B, H, T, hD) for rotary
        #    Then apply your rotary embeddings
        q = q.view(B, T, H, hD).transpose(1,2)  # → [B, H, T, hD]
        k = k.view(B, T, H, hD).transpose(1,2)  # → [B, H, T, hD]
        cos, sin = self.rotary(T, x.device)     # each [T, hD]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # 5) Merge heads back → [B, T, D]
        q = q.transpose(1,2).reshape(B, T, D)
        k = k.transpose(1,2).reshape(B, T, D)
        # v stays [B, T, D]

        # 6) change to [T, B, D] for multi-head attention
        q = q.transpose(0, 1)  # [T, B, D]
        k = k.transpose(0, 1)  # [T, B, D]
        v = v.transpose(0, 1)  # [T, B, D]
        
        # 6) Call the functional attention
        attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn_out, attn_weights = multi_head_attention_forward(
            query=q,                 # [B, T, D]
            key=k,                   # [B, T, D]
            value=v,                 # [B, T, D]
            embed_dim_to_check=D,
            num_heads=H,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.mha.dropout,
            out_proj_weight=self.mha.out_proj.weight,
            out_proj_bias=self.mha.out_proj.bias,
            training=self.training,
            attn_mask=attn_mask,  # [T, T]
          
            need_weights=True,       # or False if you don’t need attn maps
            use_separate_proj_weight=True,
            q_proj_weight=w_q,
            k_proj_weight=w_k,
            v_proj_weight=w_v,
            average_attn_weights=False,
        )

        # 7) Reshape back to [B, T, D]
        attn_out = attn_out.transpose(0, 1)
        # 7) Dropout + residual
        return self.dropout(attn_out) + residual, attn_weights
 
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
    
