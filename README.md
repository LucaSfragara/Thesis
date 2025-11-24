# Learning Hierarchical Structures with Autoregressive Language Modelling

This repository contains the code for my thesis on training GPT-2 transformer models on context-free grammars (CFGs) and probing their learned representations to understand what syntactic and linguistic properties are captured.

## Overview

The project explores how transformer language models learn grammatical structure by:
1. **Generating** synthetic data from formal context-free grammars
2. **Training** decoder-only transformer models (GPT-2 architecture) on CFG-generated sequences
3. **Probing** the learned hidden representations to understand what syntactic properties are encoded at different layers

## Project Structure

```
Thesis/
├── config.yaml                  # Central configuration for training
├── main.py                      # Legacy LSTM training script
├── train_utils.py               # Checkpoint save/load utilities
├── gpt_train.ipynb             # Main training notebook for GPT-2
│
├── data/                        # Data generation and processing
│   ├── grammars.py             # CFG definitions (GRAMMAR_CFG3b, GRAMMAR_SIMPLE)
│   ├── data_gen.py             # CFG sentence generation with multiprocessing
│   ├── datasets.py             # CFGDataset (IterableDataset for training)
│   ├── annotate_tree.py        # Inside algorithm for CFG parsing
│   ├── CFG_parsers.py          # CFG validation parser
│   └── eda.py                  # Exploratory data analysis
│
├── models/                      # Model architectures
│   ├── transformers.py         # DecoderOnlyTransformer (GPT-2)
│   ├── masks.py                # PadMask and CausalMask utilities
│   └── layers/
│       ├── decoder_layers.py   # SelfAttentionDecoderLayer
│       ├── sublayers.py        # SelfAttentionLayer, FeedForwardLayer
│       ├── rope.py             # Rotary Position Embedding (RoPE)
│       └── positional_embedding.py  # Sinusoidal positional encoding (unused)
│
├── trainers/                    # Training infrastructure
│   ├── base_trainer.py         # BaseTrainer with experiment management
│   ├── GPT_trainer.py          # GPT_Trainer for language model training
│   ├── sequence_generator.py   # Generation strategies (greedy, beam, sampling)
│   └── utils/
│       ├── create_optimizer.py # Flexible optimizer with layer-wise LR
│       └── create_scheduler.py # LR scheduler with warmup
│
├── probing/                     # Linear probing utilities 
│   ├── linear_probe.py         # LinearProbe, MultiLayerProbe, StructuralProbe
│   ├── hidden_state_extractor.py  # Extract hidden states from frozen models
│   ├── probing_datasets.py     # CFGProbingDataset for syntactic tasks
│   ├── probe_trainer.py        # Training infrastructure for probes
│   ├── eval_probe.py           # Evaluation and analysis utilities
│   ├── example_probing.py      # Usage examples
│   └── README.md               # Detailed probing documentation
│
└── evals/                       # Evaluation metrics
    └── completion_accuracy.py   # Sequence completion accuracy
```

## Key Components

### 1. Context-Free Grammars

The project uses custom CFGs to generate synthetic language data:

- **GRAMMAR_CFG3b**: Complex 3-level grammar with 22 non-terminals, inspired by Allen Zhou et al.
- **GRAMMAR_SIMPLE**: Simplified version for debugging
- Generates strings from alphabet {a, b, c} with special tokens (sos=0, eos=4)

### 2. Model Architecture

**DecoderOnlyTransformer** (GPT-2 style):
- Pre-LayerNorm decoder-only architecture
- 12 layers, 768 d_model, 12 heads (configurable)
- Rotary Position Embeddings (RoPE) instead of sinusoidal
- Layer dropout for regularization
- Weight tying support

Key features:
- Causal self-attention masking
- Mixed precision training (AMP)
- Gradient accumulation
- Checkpoint management

### 3. Training Infrastructure

**GPT_Trainer** provides:
- Cross-entropy loss with label smoothing
- Validation with text generation
- Weights & Biases integration
- Automatic checkpointing
- Learning rate scheduling with warmup

Training configuration in `config.yaml`:
```yaml
model:
  num_layers: 12
  d_model: 768
  num_heads: 12
  d_ff: 3072

training:
  batch_size: 96
  learning_rate: 6e-4
  warmup_steps: 2000
  max_epochs: 100
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Thesis

# Create conda environment
conda env create -f myenv.yml
conda activate thesis-env

# Or install dependencies manually
pip install torch numpy nltk tqdm wandb matplotlib seaborn scikit-learn torchinfo
```

### Generate CFG Data

```bash
cd data
python data_gen.py
```

This generates:
- `cfg_sentences_train_cfg_simple.npy`: Training data
- `cfg_sentences_val_cfg_simple.npy`: Validation data

### Train a Model

**Option 1: Using Jupyter Notebook**
```bash
jupyter notebook gpt_train.ipynb
```

**Option 2: Using Python Script**
```python
from trainers.GPT_trainer import GPT_Trainer
from models.transformers import DecoderOnlyTransformer
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create model
model = DecoderOnlyTransformer(
    num_layers=config['model']['num_layers'],
    d_model=config['model']['d_model'],
    # ... other parameters
)

# Create trainer
trainer = GPT_Trainer(
    model=model,
    config=config,
    config_file='config.yaml',
    run_name='gpt2_cfg_experiment'
)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)
```

### Probe the Model

```python
from probing import LinearProbe, HiddenStateExtractor, ProbeTrainer

# Load trained model
model = DecoderOnlyTransformer(...)
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
model.eval()

# Extract hidden states from layer 6
extractor = HiddenStateExtractor(model, layer_ids=[6])
hidden_states = extractor.extract_states(input_ids)

# Train a linear probe
probe = LinearProbe(input_dim=768, num_classes=5)
trainer = ProbeTrainer(probe, train_loader, val_loader)
history = trainer.train(num_epochs=10)

# Evaluate
from probing import evaluate_probe
metrics = evaluate_probe(probe, test_loader)
print(f"Probe accuracy: {metrics['accuracy']:.4f}")
```

For more probing examples, see [`probing/example_probing.py`](probing/example_probing.py).

## Configuration

All hyperparameters are centralized in `config.yaml`:

```yaml
data:
  batch_size: 96
  seq_len: 512
  vocab_size: 5
  eos_token: 4

model:
  num_layers: 12
  d_model: 768
  num_heads: 12
  d_ff: 3072
  dropout: 0.1
  weight_tying: true

optimizer:
  type: "adamw"
  learning_rate: 6e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]

scheduler:
  type: "cosine"
  warmup_steps: 2000
  min_lr: 6e-5

training:
  max_epochs: 100
  gradient_accumulation_steps: 4
  mixed_precision: true
  val_interval: 1000

wandb:
  project: "thesis-cfg-gpt2"
  entity: null
```

## Experiments

### Data Generation
- CFG sentences generated with uniform random derivations
- Multiprocessing for parallel generation
- Configurable grammar complexity

### Model Training
- Decoder-only transformers (GPT-2 architecture)
- Trained on CFG-generated sequences
- RoPE positional embeddings
- Mixed precision training
- Gradient accumulation for large effective batch sizes

## Development

### Code Style
- Python 3.8+
- Type hints encouraged
- Docstrings in Google style
- Clean, documented code

### Testing
```bash
# Test data generation
python data/data_gen.py

# Test model forward pass
python -c "from models.transformers import DecoderOnlyTransformer; print('Model import OK')"

# Test probing
python probing/example_probing.py
```

## References

1. Vaswani et al. (2017). "Attention Is All You Need." 
2. Allen Zhou et al. (2019). "Physics of Large Language Model."
