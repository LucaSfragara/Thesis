import torch, math
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]  # both (1, seq_len, dim)