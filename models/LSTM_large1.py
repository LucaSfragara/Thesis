from torch import nn
import torch


class LSTMLarge1(nn.Module):

    #Autoregressise model to predict the next token in the sequence    
    def __init__(self, vocab_size, embed_dim, hidden_dim, device):
        
        super(LSTMLarge1, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=0)
        self.dropotut = nn.Dropout(0.2)
        self.lstm =  nn.LSTM(embed_dim, hidden_dim, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            
            nn.GELU(),
        
            nn.Linear(hidden_dim, hidden_dim),
            
            nn.GELU(),
  
            nn.Linear(hidden_dim, vocab_size)
        )
        self.device = device
    
    def init_hidden(self, x):
        
        h0 =  torch.zeros(3, x.size(0), self.hidden_dim).to(self.device)
        c0 =  torch.zeros(3, x.size(0), self.hidden_dim).to(self.device)
        hidden = (h0, c0)
        return hidden
    
    def forward(self, x, hidden = None):
        
        #if hidden is None: 
        #    hidden = self.init_hidden(x)
        #print(x[0])
        x = self.embedding(x)
        x = self.dropout(x)
        #x = pack_padded_sequence(x, len_x, batch_first=True, enforce_sorted=False)
        #print(x.shape)
        x, hidden = self.lstm(x, hidden)
        #x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        
        logits = self.fc(x)
        
        return logits, hidden