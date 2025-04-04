from struct import pack
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from nltk import CFG
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle

class CFGDataset(Dataset):
    
    def __init__(self, data_file:str, subset:int):
    

        self.sentences = []
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f) 

        length = int(len(data) * subset)
        data = data[:length]
        
        for sentence in tqdm(data, total = length):
            #shuffle sentence
            
            self.sentences.append(torch.tensor(sentence, dtype=torch.long))
    
        self.length = length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        x = self.sentences[index][:-1]
        y = self.sentences[index][1:]
        
        return x,y

    def collate_fn(self, batch):
        
        batch_x = [sample[0] for sample in batch]
        batch_y = [sample[1] for sample in batch]

        # Pad the sequences to the same length
        batch_x_padded = pad_sequence(batch_x, batch_first=True, padding_value=0)
        batch_y_padded = pad_sequence(batch_y, batch_first=True, padding_value=0)
        
        
        #TODO: do not pad with 0 as it is a valid token. apply packing
        
        return batch_x_padded, batch_y_padded
        

