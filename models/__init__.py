from .LSTM_basic import LSTMBasicModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = LSTMBasicModel(vocab_size=3, embed_dim=32, hidden_dim=16, device=device).to(device)   