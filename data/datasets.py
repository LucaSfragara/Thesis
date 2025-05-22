
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
                 batch_size:int, 
                 seq_len:int, 
                 sos_token:int, 
                 eos_token:int):
        
        if sos_token in [1,2,3] or eos_token in [1,2,3] or sos_token == eos_token:
            raise ValueError("sos_token and eos_token cannot be a part of the vocabulary or be the same")
          
        self.flattened_sequences = np.load(data_file, mmap_mode="r")

        #ADD SOS and EOS tokens to each sequence
        self.eos_token = eos_token
        self.sos_token = sos_token
        
        #print("len data",  len(sequences))
        assert len(self.flattened_sequences) > 0, "No sequences in the dataset"
        #assert data[0][0] == sos_token, "SOS token not added to first element"
        #assert data[0][-1] == eos_token, "EOS token not added to last element"
        
    
        #print(flattened_seq_tensor.shape)
        self.total_tokens = len(self.flattened_sequences)
        
        self.n_batches = self.total_tokens // (batch_size * seq_len) # drop extra so divisible by (batch size*seq_len)

        self.flattened_sequences = self.flattened_sequences[:self.n_batches * batch_size * seq_len]        

        self.flattened_sequences = self.flattened_sequences.reshape(batch_size, -1) # (batch_size, n_batches * seq_len)
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
        seqs = self.flattened_sequences  # shape [batch_size, total_batches * L]
        
        offset = torch.randint(0, L, ()).item()
        
        total_batches = (self.total_tokens - (L + 1) - offset) // L + 1
        if total_batches <= 0:
            return  # nothing to yield

        
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
        max_batch = (seqs.shape[1] - (L + 1)) // L

        for b in range(start_batch, min(end_batch, max_batch+1)):
            i = b * L
            x = torch.from_numpy(seqs[:, i : i + L].copy()).long()         # shape [batch_size, seq_len]
            y = torch.from_numpy(seqs[:, i + 1 : i + 1 + L].copy()).long()  # shifted targets
            yield x, y
        
    def __getitem__(self, index):
        return self.sequences[index]
    
    def sample_prompts(self, num_prompts:int, prompt_len:int, seed:int):
        """
        Sample a batch of prompts from the dataset.
        Args:
            num_prompts (int): The number of prompts to sample.
            prompt_len (int): The length of each prompt.
        Returns:
            torch.Tensor: A tensor of shape (num_prompts, prompt_len) containing the sampled prompts.
            torch.Tensor: A tensor of shape (num_prompts, seq_len) containing the continuation
        """
        # Sample random indices
        np.random.seed(seed)
        #indices = np.random.randint(0, self.n_batches, size=num_prompts)
        
        # Get the prompts
        #split every time you encounter a eos token
        
        full_sentences = []
        i= 0
        
        #save flattened sequences to a txt file
      
        while len(full_sentences) < num_prompts:
            
            full_sentences.extend(np.split(self.flattened_sequences[1+i], np.where(self.flattened_sequences[1+i]==0)[0]))
            i +=1
        
        full_sentences = full_sentences[1:num_prompts+1]
        if full_sentences[0][0] != self.sos_token:
            
            full_sentences[0] = np.insert(full_sentences[0], 0, self.sos_token)
            
         
        for i in range(len(full_sentences)):
            assert full_sentences[i][-1] == self.eos_token, "EOS token not added to last element"

        full_sentences_padded = pad_sequence([torch.from_numpy(x.copy()) for x in full_sentences], batch_first=True, padding_value=5)
        
        prompts = full_sentences_padded[:, :prompt_len]
        originals = full_sentences_padded
        
        #check each prompt starts with sos_token
        assert (prompts[:, 0] == self.sos_token).all(), "SOS token not added to first element"
        
        #check each originals starts with sos_token
        assert (originals[:, 0] == self.sos_token).all(), "SOS token not added to first element"
        #check each original ends with eos_token
        #print(np.where(originals == 0))
        
        
        return prompts.long(), originals.long()
    
    
 

def verify_dataloader(dataloader: DataLoader):
    
    """
    Verify the dataloader by checking the shapes of the batches.
    Args:
        dataloader (DataLoader): The dataloader to verify.
    """
    print("Verifying dataloader...")
    #print("Example sequence: ", dataloader.dataset[0])
    #print("Example batch: ", next(iter(dataloader)))
    print("Number of batches: ", len(dataloader))
    print(f"Total number of tokens: { dataloader.dataset.total_tokens: 6e}") #in scientific notation
    print("Example batch shapes (shifted, golden): ", 
          next(iter(dataloader))[0].shape, 
          next(iter(dataloader))[1].shape)
    
if __name__ == "__main__":
    # Example usage
    data_file = "/workspace/Thesis/cfg_sentences_val_cfg3b.npy"

    batch_size = 1
    seq_len = 200
    sos_token = 0
    eos_token = 4
    
    dataset = CFGDataset(data_file, batch_size, seq_len, sos_token, eos_token)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=12)
    
    #save all the batches to a txt file to check if they are correct
    #dump flattened sequences to a txt file
    print(dataset.flattened_sequences[:, :200])
    with open("cfg_sentences_train_cfg3b.txt", "w") as f:
        f.write(dataset.flattened_sequences)
            
    """with open("cfg_sentences_train_cfg3b.txt", "w") as f:
        for x, y in dataloader:
            f.write("".join(x[0].numpy().astype(str)))
            f.write("\n")
            f.write("".join(y[0].numpy().astype(str)))
            f.write("\n")
    
    verify_dataloader(dataloader)"""