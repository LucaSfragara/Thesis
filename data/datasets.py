
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, get_worker_info
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np


class CFGDataset(Dataset):
    
    """ 
   
    """
    
    def __init__(self, 
                 data_file:str,  
                 seq_len:int, 
                 sos_token:int, 
                 eos_token:int):
        
        if sos_token in [1,2,3] or eos_token in [1,2,3] or sos_token == eos_token:
            raise ValueError("sos_token and eos_token cannot be a part of the vocabulary or be the same")
          
        self.sequences = np.load(data_file, allow_pickle=True)#, mmap_mode="r")

        #ADD SOS and EOS tokens to each sequence
        self.eos_token = eos_token
        self.sos_token = sos_token
        
        #print("len data",  len(sequences))
        assert len(self.sequences) > 0, "No sequences in the dataset"
        #assert data[0][0] == sos_token, "SOS token not added to first element"
        #assert data[0][-1] == eos_token, "EOS token not added to last element"
        
    
        #print(flattened_seq_tensor.shape)
        self.total_tokens = self.sequences.size
        
    
        self.seq_len = seq_len

        
    def __len__(self):
        
        """
        Returns the number of sentences in the dataset.
        """
        return len(self.sequences)
    
        
    def __getitem__(self, index):
        
        
        x = self.sequences[index][:-1]
        y = self.sequences[index][1:]

        assert (len(x) == len(y)), f"Length mismatch: {len(x)} vs {len(y)} between golden and shifted"

        return torch.from_numpy(x).long(),torch.from_numpy(y).long()
    
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
        indices = np.random.randint(0, len(self.sequences), size=num_prompts)
       
        
        prompts = self.sequences[indices, :prompt_len]
        originals = self.sequences[indices, :]
        
        #check each prompt starts with sos_token
        assert (prompts[:, 0] == self.sos_token).all(), "SOS token not added to first element"
        
        #check each originals starts with sos_token
        assert (originals[:, 0] == self.sos_token).all(), "SOS token not added to first element"
        #check each original ends with eos_token
        
        assert (originals[:, -1] == self.eos_token).all(), "EOS token not added to last element"
        
        
        return torch.from_numpy(prompts).long(), torch.from_numpy(originals).long()
    
    
 

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
    data_file = "/workspace/Thesis/cfg_sentences_val_cfg3b_samelen.npy"

    batch_size = 96
    seq_len = 326
    sos_token = 0
    eos_token = 4
    
    dataset = CFGDataset(data_file, seq_len, sos_token, eos_token)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12)
    
    #save all the batches to a txt file to check if they are correct
    #dump flattened sequences to a txt file
   
    """with open("cfg_sentences_train_cfg3b.txt", "w") as f:
        for x, y in dataloader:
            f.write("".join(x[0].numpy().astype(str)))
            f.write("\n")
            f.write("".join(y[0].numpy().astype(str)))
            f.write("\n")
    """
    verify_dataloader(dataloader)
    
    print(dataset.sample_prompts(10, 10, 0))