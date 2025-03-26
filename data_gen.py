import nltk
from nltk import CFG
from nltk.parse.generate import generate
from tqdm import tqdm
import pickle
from grammars import GRAMMAR
import numpy as np

length = 20000000
data = []
i = 0
terminals_to_idx = {'a':0, 'b':1, 'c':2}

if __name__ == "__main__":
    
    for sentence in tqdm(generate(GRAMMAR), total = length):
                #print(sentence)
                sentence_integers = [terminals_to_idx[term] for term in sentence]
        
                # Store as numpy uint8 array (much more memory efficient)
                data.append(np.array(sentence_integers, dtype=np.uint8))
                i += 1
                if i == length:
                    break
                
    #write data to jsonl

    with open('cfg_sentences_train.pkl', 'wb') as f:
        pickle.dump(data[:length//2], f, protocol=4)
        
    with open('cfg_sentences_val.pkl', 'wb') as f:
        pickle.dump(data[length//2:], f, protocol=4)

    print("Data saved to cfg_sentences_train.pkl")
    print("Data saved to cfg_sentences_val.pkl")