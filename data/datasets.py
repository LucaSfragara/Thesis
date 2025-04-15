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
    
    def __init__(self, data_file:str, subset:int, pad_value:int):
    

        self.sentences = []
        self.pad_value = pad_value
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f) 

        length = int(len(data) * subset)
        data = data[:length]
        
        for sentence in tqdm(data, total = length):
            #shuffle sentence
            
            self.sentences.append(torch.tensor(sentence, dtype=torch.long))
            
            # Track max length (add 1 for the sos/eos tokens)
            self.text_max_len = max(self.text_max_len, len(sentence)+1)

            
        self.length = length
      

        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        x = self.sentences[index][:-1]
        y = self.sentences[index][1:]
        
        assert (len(x) == len(y)), f"Length mismatch: {len(x)} vs {len(y)}"
        
        return x,y

    def collate_fn(self, batch):
        
        """
        Custom collate function to pad the sequences to the same length.
        Args:
            - batch (list): List[(x, y)] 
        Returns:
            tuple where: 
            - batch_x_padded (torch.Tensor): Padded input sequences. shape: (batch_size, max_seq_len)
            - batch_y_padded (torch.Tensor): Padded target sequences. shape: (batch_size, max_seq_len)
            - lens (torch.Tensor): Lengths of the input sequences. shape: (batch_size,)
            """
        
        batch_x, batch_y = zip(*batch)

        # Pad the sequences to the same length
        batch_x_padded = pad_sequence(batch_x, batch_first=True, padding_value=self.pad_value)
        batch_y_padded = pad_sequence(batch_y, batch_first=True, padding_value=self.pad_value)
        
        lens = [len(x) for x in batch_x]
    
        
        return batch_x_padded, batch_y_padded, torch.tensor(lens, dtype=torch.long)
        

