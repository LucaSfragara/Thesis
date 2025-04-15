import torch

def PadMask(padded_input: torch.tensor, input_lengths: torch.tensor):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    
    mask = torch.ones((padded_input.shape[0], padded_input.shape[1]))
    
    max_len = padded_input.shape[1]
    range_row = torch.arange(max_len).unsqueeze(0) #shape (1, T)
    range_row = range_row.to(input_lengths.device)
    
    
    #input_lengths = input_lengths.to('cpu')
    #range_row = range_row.to('cpu')
    
    #shape (1, T) #shape (1, N)
    mask = range_row >= input_lengths.unsqueeze(1)
    
    return mask.to(torch.bool)


def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    T = padded_input.shape[1]
    mask = torch.ones((T, T))
    
    return torch.triu(mask, diagonal = 1).to(torch.bool)

