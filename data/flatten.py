# make_flat.py
import numpy as np
import pickle
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--in",   dest="in_path",  required=True)
parser.add_argument("--out",  dest="out_path", required=True)
parser.add_argument("--sos",  type=int, required=True)
parser.add_argument("--eos",  type=int, required=True)
args = parser.parse_args()

# 1) load your pickle of List[np.ndarray]
with open(args.in_path, "rb") as f:
    sequences = pickle.load(f)

# 2) optionally subset, if you want a fraction
# sequences = sequences[: int(len(sequences)*0.9)]

# 3) add SOS/EOS to each sequence
extended = []
for seq in tqdm(sequences, desc="Adding SOS/EOS", total=len(sequences)):
    # seq is a 1‑D numpy array of token IDs
    new = np.concatenate(([args.sos], seq, [args.eos]))
    extended.append(new)

# 4) flatten them all
flat = np.concatenate(extended, axis=0)

# 5) save as .npy
np.save(args.out_path, flat)
print(f"Wrote {flat.nbytes/1e9:.2f} GB to {args.out_path}")
