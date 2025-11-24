import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import model_large1
from tqdm import tqdm
from data.datasets import CFGDataset
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data.CFG_parsers import CFGParser
from data.grammars import GRAMMAR_CFG3b


model = model_large1

def completion_accuracy(model, data_loader, device, prefix_length=20):
    """
    Evaluate sequence completion accuracy for a language model.

    Given a prefix of tokens, the model generates the remaining sequence
    and compares it against the ground truth to compute accuracy.

    Args:
        model: The language model to evaluate
        data_loader: DataLoader providing evaluation sequences
        device: Device to run evaluation on (cuda/cpu)
        prefix_length: Number of tokens to use as prefix (default: 20)

    Returns:
        float: Average completion accuracy across all sequences
    """
    model.eval()
    
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')
    
    total_accuracy = 0
    parser = CFGParser(GRAMMAR_CFG3b)

    for i, data in enumerate(data_loader):
        x, y = data 
        x, y = x.to(device), y.to(device)
        prefix = x[:, :prefix_length]

        hidden = None
        
        for i in range(prefix_length):
            logits, hidden = model(prefix[:, i].unsqueeze(1), hidden)
        cur_tok = prefix[:, -1].unsqueeze(1)
        
        total_accuracy = 0
        
        sentences = prefix
        
        for i in range(x.shape[1] - prefix_length):
            logits, hidden = model(cur_tok, hidden)
            cur_tok = torch.argmax(logits, dim=2)
            sentences = torch.cat((sentences, cur_tok), dim=1)
            acc = torch.sum(cur_tok == x[:, prefix_length + i].unsqueeze(1)).item() / cur_tok.size(0)
            print(acc)
        

if __name__ == "__main__":
    

    checkpoint_state_dict = torch.load('checkpoints/lstm_e30_e128_h512-cfg3b.pth')
    model.load_state_dict(checkpoint_state_dict['model_state_dict'])
    val_data = CFGDataset('cfg_sentences_train_cfg3b.pkl', subset = 0.1)
    data_loader = DataLoader(val_data, batch_size=1024, shuffle=True, pin_memory=True, num_workers=12, collate_fn=val_data.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    completion_accuracy(model, data_loader, device)