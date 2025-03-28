from torch import nn
import torch

class LSTMBasicModel(nn.Module):

    #Autoregressise model to predict the next token in the sequence    
    def __init__(self, vocab_size, embed_dim, hidden_dim, device):
        
        super(LSTMBasicModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm =  nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.device = device
    
    def init_hidden(self, x):
        
        h0 =  torch.zeros(2, x.size(0), self.hidden_dim).to(self.device)
        c0 =  torch.zeros(2, x.size(0), self.hidden_dim).to(self.device)
        hidden = (h0, c0)
    
    def forward(self, x, hidden = None):
        
        if hidden is None: 
            hidden = self.init_hidden(x)
            
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        
        logits = self.fc(x)
        
        return logits, hidden