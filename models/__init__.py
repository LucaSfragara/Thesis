from .LSTM_basic import LSTMBasicModel
from .LSTM_large1 import LSTMLarge1
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_basic = LSTMBasicModel(vocab_size=3, embed_dim=32, hidden_dim=16, device=device).to(device)   

model_large1 = LSTMLarge1(vocab_size=3, embed_dim=128, hidden_dim=512, device=device).to(device)