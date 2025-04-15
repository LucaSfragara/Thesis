import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.  
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        ''' 
        super().__init__()
        # TODO: Implement __init__
       
        # TODO: Initialize the sublayers      
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout) # Masked self-attention layer
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout) # Feed-forward network
        

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the DecoderLayer1.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (torch.Tensor): The padding mask for the decoder. shape: (batch_size, seq_len)
            attn_mask (torch.Tensor): The self-attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, num_classes)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # TODO: Implement forward: Follow the figure in the writeup

        x, mha_attn_weights = self.self_attn.forward(x, key_padding_mask=key_padding_mask, attn_mask =attn_mask )
        x = self.ffn(x)
        
        # TODO: Return the output tensor and attention weights
        return x, mha_attn_weights
