import torch
import torch.nn as nn
import random
from typing import Tuple, Optional
from models.layers.positional_embedding import PositionalEncoding
from models.masks import PadMask, CausalMask
from models.layers.decoder_layers import SelfAttentionDecoderLayer

class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        '''
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        '''
        super().__init__()
        
        # TODO: Implement __init__

        # Initialize the decoder
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        
        # TODO: Create a ModuleList of decoder layers based on the number of layers
        self.dec_layers     = nn.ModuleList(
            [(SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)) for _ in range(num_layers)]
        ) # ModuleList of decoder layers

        # TODO: Create target embedding and other layers
        self.target_embedding       = nn.Embedding(num_classes, d_model)
        self.positional_encoding    = PositionalEncoding(d_model, max_len) # Positional encoding
        self.final_linear           = nn.Linear(d_model, num_classes) # Final linear layer
        self.dropout                = nn.Dropout(dropout) # Dropout
        self.norm                   = nn.LayerNorm(d_model) # Layer norm

        # Weight tying (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight


    def forward(self, padded_targets: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): The lengths of the target sequences. shape: (batch_size,)
        Returns:
            seq_out (torch.Tensor): The output sequence. shape: (batch_size, seq_len, d_model)
            runnint_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        '''
        # DO NOT MODIFY 
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        
        # TODO: Implement forward

        # TODO: Create padding mask for padded_targets on the same device as the input (use PadMask)
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets,target_lengths).to(padded_targets.device)
        
        # TODO: Create causal mask to prevent attending to future tokens on the same device as the input (use CausalMask)
        causal_mask = CausalMask(padded_targets).to(padded_targets.device)

        # TODO: Apply the embedding
        x = self.target_embedding(padded_targets)
        # TODO: Apply positional encoding
        x = self.positional_encoding(x)
        # TODO: Apply dropout 
        x = self.dropout(x)

        # TODO: Pass through all decoder layers, save attention masks
        runnint_att = {}
        for i in range(self.num_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            
            # TODO: Pass through decoder layer
            x, attention = self.dec_layers[i](x, key_padding_mask = pad_mask_dec, attn_mask = causal_mask)
            
            # TODO: Save attention weights  
            runnint_att['layer{}_dec_self'.format(i + 1)] = attention #shape (batch_size, seq_len, seq_len) 

        # TODO: Apply normalization
        x = self.norm(x)
        # TODO: Linear layer (Final Projection) for next character prediction
        seq_out = self.final_linear(x)
        
        # TODO: Return the output sequence and running attention weights
        return seq_out, runnint_att
    
    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder. 
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded. 
        Can only handle batch_size = 1 or batch with same lengths and no padding. 
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")
        # Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # Return the last token's logits for next token prediction    
        logits     = seq_out[:, -1, :]
        return logits
    
