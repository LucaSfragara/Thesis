import nltk
from nltk import CFG
from nltk.parse.generate import generate
from tqdm import tqdm
import pickle
from grammars import GRAMMAR_CFG3b 
import numpy as np
import random

length = 800000
data = []

terminals_to_idx = {'a':1, 'b':2, 'c':3}


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

    
    
if __name__ == "__main__":
    
    for sentence in tqdm(range(length), desc="Generating sentences"):
            #print(sentence)
            
            sentence = random_derivation(GRAMMAR_CFG3b)
            
            sentence_integers = [terminals_to_idx[term] for term in sentence]
    
            # Store as numpy uint8 array (much more memory efficient)
            data.append(np.array(sentence_integers, dtype=np.uint8))
     
    #write data to jsonl

    with open('cfg_sentences_train_cfg3b.pkl', 'wb') as f:
        pickle.dump(data[:length//2], f, protocol=4)
        
    with open('cfg_sentences_val_cfg3b.pkl', 'wb') as f:
        pickle.dump(data[length//2:], f, protocol=4)

    print("Data saved to cfg_sentences_train.pkl")
    print("Data saved to cfg_sentences_val.pkl")
    
