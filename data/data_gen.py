import nltk
from nltk import CFG
from nltk.parse.generate import generate
from tqdm import tqdm
import pickle
from grammars import GRAMMAR_SIMPLE#GRAMMAR_CFG3b
import numpy as np
import random
from multiprocessing import Pool, cpu_count


terminals_to_idx = {'a':1, 'b':2, 'c':3, "sos":0, "eos":4}
data = []

def random_derivation(grammar, symbol=None):
   
    """
    Generate one random derivation (string) from a CFG, 
    by uniformly sampling from the expansions at each step.
    """
    
    if symbol is None:
        # start from the root symbol
        symbol = grammar.start()
        
    # If 'symbol' is actually a terminal (string), return it directly
    # In NLTK CFG, terminals are just python strings in the productions
    if isinstance(symbol, str):
        return [symbol]  # single token
    
    # Otherwise 'symbol' is a Nonterminal
    # Get all productions A -> alpha where A = symbol
    possible_prods = [prod for prod in grammar.productions(lhs=symbol)]
    if not possible_prods:
        # No expansions (shouldn't happen if grammar is well-defined)
        return []
    
    # Pick one production at random
    chosen_prod = random.choice(possible_prods)
    
    # Recursively expand each symbol on the RHS
    result = []
    for rhs_sym in chosen_prod.rhs():
        result.extend(random_derivation(grammar, rhs_sym))
    return result

    
def gen_sentence(_):
    # generate one sentence, convert to idx array
    sent = random_derivation(GRAMMAR_SIMPLE)
    sent.append("eos")  # add EOS token
    sent.insert(0, "sos")  # add SOS token
    return np.array([terminals_to_idx[t] for t in sent], dtype=np.uint8)

if __name__ == "__main__":

    length = 8_000_000  # number of sentences to generate
    n_procs = min(cpu_count(), 64)    # or whatever cap you want
    print("Generating sentences with", n_procs, "processes")
    with Pool(n_procs) as pool:
        # imap is lazy; tqdm will show progress
        data = list(tqdm(pool.imap(gen_sentence, range(length)),
                         total=length,
                         desc="Generating CFG sentences"))
    
    print("Lenghts Quartiles: ")
    print(np.percentile([len(d) for d in data], [25, 50, 75]))
    # split train/val
    flat = np.concatenate(data, axis=0)
    print("total tokens", flat.shape[0])
    split = int(0.99 * len(flat))
    #save to text for debugging
    print("Fist 200 tokens")
    print(flat[:200])
    np.save("cfg_sentences_train_cfg_simple.npy", flat[:split])
    np.save("cfg_sentences_val_cfg_simple.npy", flat[split:])
    print(f"Wrote {flat.nbytes/1e9:.2f} GB of both train and val")
    
    with open("cfg_sentences_train_simple.txt", "w") as f:
        f.write("".join(flat.astype(str)))
  
    
    