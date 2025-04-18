
from random import seed
from wsgiref.util import setup_testing_defaults
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, get_worker_info
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np


class CFGDataset(IterableDataset):
    
    """ 
    Flattens a list of tokenized numpy arrays into one long stream,
    chops it into (batch_size x seq_len) and yields (x, y) pairs.
    """
    
    def __init__(self, 
                 data_file:str, 
                 subset:int,
                 batch_size:int, 
                 seq_len:int, 
                 sos_token:int, 
                 eos_token:int):
        
        if sos_token in [1,2,3] or eos_token in [1,2,3] or sos_token == eos_token:
            raise ValueError("sos_token and eos_token cannot be a part of the vocabulary or be the same")
          
        with open(data_file, 'rb') as f:
            data = pickle.load(f)  # loads list of sequences
        
        self.subset = subset
        length = int(len(data) * subset)
        data = data[:length] # Subset the data
        
        #ADD SOS and EOS tokens to each sequence
        self.eos_token = eos_token
        self.sos_token = sos_token
        
        # Add SOS and EOS tokens to each sequence
        data = [np.concatenate((np.array([sos_token]), seq, np.array([eos_token]))) for seq in data]        

        
        flattened_seq= np.concatenate(data, axis = 0)
        flattened_seq_tensor = torch.from_numpy(flattened_seq).long()
        
        self.total_tokens = flattened_seq_tensor.numel() # total number of tokens in the dataset
        
        self.n_batches = flattened_seq_tensor.numel() // (batch_size * seq_len) # drop extra so divisible by (batch size*seq_len)

        flattened_seq_tensor = flattened_seq_tensor[:self.n_batches * batch_size * seq_len]        
        
        self.sequences = flattened_seq_tensor.view(batch_size, -1) # (batch_size, n_batches * seq_len)
        self.seq_len = seq_len
      
        
    def __len__(self):
        
        """
        Returns the number of batches in the dataset.
        """
        return self.n_batches
    
    
    def __iter__(self):
        worker_info = get_worker_info()
        total_batches = self.n_batches
        L = self.seq_len
        seqs = self.sequences  # shape [batch_size, total_batches * L]

        if worker_info is None:
            # single‐process: cover the whole range
            start_batch, end_batch = 0, total_batches
        else:
            # split the *batch index* range among workers
            per_worker = total_batches // worker_info.num_workers
            remainder  = total_batches % worker_info.num_workers
            wid = worker_info.id

            # the first `remainder` workers get (per_worker+1) batches
            if wid < remainder:
                start_batch = wid * (per_worker + 1)
                end_batch   = start_batch + (per_worker + 1)
            else:
                start_batch = wid * per_worker + remainder
                end_batch   = start_batch + per_worker

        # now each worker only walks its own [start_batch, end_batch) slice
        max_batch = (seqs.size(1) - (L + 1)) // L

        for b in range(start_batch, min(end_batch, max_batch+1)):
            i = b * L
            x = seqs[:, i : i + L]           # shape [batch_size, seq_len]
            y = seqs[:, i + 1 : i + 1 + L]   # shifted targets
            yield x, y
        
    def __getitem__(self, index):
        return self.sequences[index]
 

def verify_dataloader(dataloader: DataLoader):
    
    """
    Verify the dataloader by checking the shapes of the batches.
    Args:
        dataloader (DataLoader): The dataloader to verify.
    """
    print("Verifying dataloader...")
    #print("Example sequence: ", dataloader.dataset[0])
    #print("Example batch: ", next(iter(dataloader)))
    print("data subset: ", dataloader.dataset.subset)
    print("Number of batches: ", len(dataloader))
    print(f"Total number of tokens: { dataloader.dataset.total_tokens: 2e}") #in scientific notation
    print("Example batch shapes (shifted, golden): ", 
          next(iter(dataloader))[0].shape, 
          next(iter(dataloader))[1].shape)
    
if __name__ == "__main__":
    # Example usage
    data_file = "/ocean/projects/cis250019p/sfragara/lstm/cfg_sentences_train_cfg3b.pkl"
    subset = 1
    batch_size = 96
    seq_len = 512
    sos_token = 0
    eos_token = 4
    
    dataset = CFGDataset(data_file, subset, batch_size, seq_len, sos_token, eos_token)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
    verify_dataloader(dataloader)