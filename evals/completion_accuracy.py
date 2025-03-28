import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import model
from tqdm import tqdm
from datasets import CFGDataset
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from CFG_parsers import CFGParser
from grammars import GRAMMAR_CFG3b

def completion_accuracy(model, data_loader, device, prefix_length = 10):
    
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
            #calculate accuracy
            #acc = torch.sum(cur_tok == x[:, prefix_length + i].unsqueeze(1)).item() / cur_tok.size(0)
            #total_accuracy += acc
        print(sentences[0])
        print("")
        print(x[0])
        
        k=0
        for i, j in zip(sentences[0], x[0]):
            if i != j:
                print(f"diff at pos: {k}, expected {j}, got {i}")
            k += 1
        
        #print(total_accuracy/30)
        for sentence in sentences:
            sentence = sentence.cpu().numpy()
            sentence = ''.join([str(i) for i in sentence])
            #print(sentence)
            print(parser.is_valid(sentence))
        
        
if __name__ == "__main__":
    

    checkpoint_state_dict = torch.load('checkpoints/lstm-cfg3b.pth')
    model.load_state_dict(checkpoint_state_dict['model_state_dict'])
    val_data = CFGDataset('cfg_sentences_val.pkl', subset = 1)
    data_loader = DataLoader(val_data, batch_size=512, shuffle=False, pin_memory=True, num_workers=12, collate_fn=val_data.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    completion_accuracy(model, data_loader, device)