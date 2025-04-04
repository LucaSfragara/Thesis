import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import model_large1
from tqdm import tqdm
from datasets import CFGDataset
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from CFG_parsers import CFGParser
from grammars import GRAMMAR_CFG3b
from collections import Counter


model = model_large1

def completion_accuracy(model, data_loader, device, prefix_length = 20):
    
    model.eval()
    
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')
    
    total_accuracy = 0
    parser = CFGParser(GRAMMAR_CFG3b)

            
    #true_sentences = torch.arange(0,200)
    
    #generated_sentences = torch.arange(0,200)

    #true_sentences = true_sentences.unsqueeze(0).repeat(512, 1)
    #generated_sentences = generated_sentences.unsqueeze(0).repeat(512, 1)
    #print(true_sentences.shape)
    for i, data in enumerate(data_loader):
        x, y = data 
        x, y = x.to(device), y.to(device)
        prefix = x[:, :prefix_length]

        hidden = None
        
        for i in range(prefix_length):
            logits, hidden = model(prefix[:, i].unsqueeze(1), hidden) 
        #prefix = true_sentences[:, :prefix_length]
        #print(prefix)
        #break
        cur_tok = prefix[:, -1].unsqueeze(1)
        
        total_accuracy = 0
        
        sentences = prefix
        
        for i in range(x.shape[1] - prefix_length):
            
            #print(cur_tok)
            #break
            logits, hidden = model(cur_tok, hidden)
            
            #print(logits)
            #break
            #print(logits.shape)
            cur_tok = torch.argmax(logits, dim=2)  
            
            #print(i, cur_tok)
            #cur_tok = generated_sentences[:, prefix_length+i].unsqueeze(1)
            sentences = torch.cat((sentences, cur_tok), dim=1)
            #calculate accuracy
            #acc = torch.sum(cur_tok == x[:, prefix_length + i].unsqueeze(1)).item() / cur_tok.size(0)
            acc = torch.sum(cur_tok == x[:, prefix_length + i].unsqueeze(1)).item() / cur_tok.size(0)
            print(acc)
            #print(cur_tok)
            #print(true_sentences[:, prefix_length + i].unsqueeze(1))
            #break
        #break
            #total_accuracy += acc
        #print(sentences[0])
        #print("")
        #print([0])
        """
        k=0
        for i, j in zip(sentences[0], x[0]):
            if i != j:
                print(f"diff at pos: {k}, expected {j}, got {i}")
            k += 1
        """
        #print(total_accuracy/30)
        #for sentence in sentences:
        #    sentence = sentence.cpu().numpy()
        #    sentence = ''.join([str(i) for i in sentence])
            #print(sentence)
        #  print(parser.is_valid(sentence))
        

if __name__ == "__main__":
    

    checkpoint_state_dict = torch.load('checkpoints/lstm_e30_e128_h512-cfg3b.pth')
    model.load_state_dict(checkpoint_state_dict['model_state_dict'])
    #print(checkpoint_state_dict['epoch'])
    val_data = CFGDataset('cfg_sentences_train_cfg3b.pkl', subset = 0.1)
    data_loader = DataLoader(val_data, batch_size=1024, shuffle=True, pin_memory=True, num_workers=12, collate_fn=val_data.collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    completion_accuracy(model, data_loader, device)