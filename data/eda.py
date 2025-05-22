import numpy as np
from pprint import pprint

with open("/workspace/Thesis/cfg_sentences_train_cfg_simple.npy", "rb") as f:
    #val_data = pickle.load(f)
    #pprint(val_data[:10])
    #read each sentence and add 1 to every token and then save it to the same file
    data = np.load(f)
    print(data[:1000])
    #for i in range(len(data)):
    #    print(data[i])
    #save to the same file
    #with open("/ocean/projects/cis250019p/sfragara/lstm/cfg_sentences_train_cfg3b.pkl", "wb") as f:
    #    pickle.dump(data, f, protocol=4)
    