"""
Example script demonstrating how to use the probing utilities.

This script shows a complete workflow for probing GPT-2 hidden representations:
1. Load trained model
2. Extract hidden states from specific layers
3. Create probing datasets
4. Train linear probes
5. Evaluate and analyze results
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformers import DecoderOnlyTransformer
from data.datasets import CFGDataset
from probing import (
    LinearProbe,
    HiddenStateExtractor,
    ProbeTrainer,
    ProbingDataset,
    evaluate_probe,
    layer_wise_accuracy_curve
)


def example_basic_probing():
    """
    Basic example: Extract hidden states and train a simple linear probe.
    """
    print("=== Basic Probing Example ===\n")

    # 1. Load your trained GPT-2 model
    print("Loading model...")
    model = DecoderOnlyTransformer(
        vocab_size=5,  # Adjust based on your vocab
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=0.1
    )

    # Load checkpoint
    checkpoint_path = 'path/to/your/checkpoint.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from checkpoint")
    else:
        print("Warning: Checkpoint not found, using untrained model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 2. Prepare data
    print("\nPreparing data...")
    dataset = CFGDataset(data_file="cfg_sentences_val_cfg3b.pkl", subset=0.1)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    # 3. Extract hidden states from layer 6 (middle layer)
    print("\nExtracting hidden states from layer 6...")
    extractor = HiddenStateExtractor(model, layer_ids=[6])

    all_hidden_states = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch.to(device)

        # Extract hidden states
        hidden_states = extractor.extract_states(input_ids)
        layer_6_hidden = hidden_states[6]  # Shape: [batch, seq_len, 768]

        # Create labels (next token prediction)
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]

        # Flatten sequence dimension
        batch_size, seq_len, hidden_dim = layer_6_hidden.shape
        hidden_flat = layer_6_hidden.reshape(-1, hidden_dim)
        labels_flat = labels.reshape(-1)

        # Filter out padding
        valid_mask = labels_flat != -100
        all_hidden_states.append(hidden_flat[valid_mask].cpu())
        all_labels.append(labels_flat[valid_mask].cpu())

        if len(all_hidden_states) >= 100:  # Limit for example
            break

    # Concatenate
    hidden_states_tensor = torch.cat(all_hidden_states, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    print(f"Extracted {len(hidden_states_tensor)} token representations")
    print(f"Hidden state shape: {hidden_states_tensor.shape}")

    # 4. Create probing dataset
    print("\nCreating probing dataset...")
    probe_dataset = ProbingDataset(hidden_states_tensor, labels_tensor)

    # Split into train/val
    train_size = int(0.8 * len(probe_dataset))
    val_size = len(probe_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        probe_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 5. Create and train linear probe
    print("\nTraining linear probe...")
    vocab_size = 5  # Adjust based on your vocab
    probe = LinearProbe(input_dim=768, num_classes=vocab_size, dropout=0.1)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)

    trainer = ProbeTrainer(
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        experiment_dir='./probe_experiments/layer6_next_token',
        use_wandb=False
    )

    # Train for 10 epochs
    history = trainer.train(num_epochs=10)

    print(f"\nBest validation accuracy: {trainer.best_val_acc:.4f}")

    # 6. Evaluate
    print("\nEvaluating probe...")
    metrics = evaluate_probe(probe, val_loader, device=device)
    print(f"Final metrics: {metrics}")

    extractor.remove_hooks()


def example_multi_layer_probing():
    """
    Advanced example: Compare probing performance across multiple layers.
    """
    print("\n=== Multi-Layer Probing Example ===\n")

    # Setup model and data (similar to above)
    model = DecoderOnlyTransformer(
        vocab_size=5,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=0.1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Extract from layers 0, 3, 6, 9, 11
    layer_ids = [0, 3, 6, 9, 11]
    print(f"Extracting hidden states from layers: {layer_ids}")

    extractor = HiddenStateExtractor(model, layer_ids=layer_ids)

    # (Similar data loading and extraction code as above)
    # For each layer, create separate probing datasets and train probes

    layer_probes = {}
    layer_accuracies = {}

    for layer_id in layer_ids:
        print(f"\n--- Layer {layer_id} ---")

        # Extract hidden states for this layer
        # (code similar to basic example)

        # Create and train probe
        probe = LinearProbe(input_dim=768, num_classes=5, dropout=0.1)

        # Train probe (abbreviated)
        # trainer = ProbeTrainer(...)
        # history = trainer.train(num_epochs=10)

        # Store probe and accuracy
        layer_probes[layer_id] = probe
        # layer_accuracies[layer_id] = trainer.best_val_acc

    # Compare across layers
    print("\n=== Layer-wise Accuracy ===")
    for layer_id, acc in sorted(layer_accuracies.items()):
        print(f"Layer {layer_id}: {acc:.4f}")

    extractor.remove_hooks()


def example_structural_probing():
    """
    Example of structural probing for parse tree distances.
    """
    print("\n=== Structural Probing Example ===\n")

    from probing.linear_probe import StructuralProbe

    # Load model and extract hidden states
    model = DecoderOnlyTransformer(vocab_size=5, d_model=768, n_heads=12, n_layers=12)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Create structural probe
    struct_probe = StructuralProbe(input_dim=768, probe_rank=128)
    struct_probe.to(device)

    print("Structural probe initialized")
    print(f"Projects from {struct_probe.input_dim}D to {struct_probe.probe_rank}D space")

    # Extract hidden states
    extractor = HiddenStateExtractor(model, layer_ids=[11])  # Last layer

    # (Get some sequences and their parse tree distances)
    # For now, this is a placeholder showing the API

    with torch.no_grad():
        # Example sequence
        input_ids = torch.randint(0, 5, (1, 20)).to(device)
        hidden_states = extractor.extract_states(input_ids)
        layer_11_hidden = hidden_states[11]

        # Project to probe space
        projections = struct_probe(layer_11_hidden)

        # Compute distances
        distances = struct_probe.compute_distance(projections)

        print(f"\nProjections shape: {projections.shape}")
        print(f"Distances shape: {distances.shape}")

        # Compare with true parse tree distances
        # (This would require implementing parse tree distance computation)

    extractor.remove_hooks()


if __name__ == '__main__':
    print("Probing Utilities Examples\n")
    print("This script demonstrates how to use the probing package.")
    print("Uncomment the examples you want to run:\n")

    # Uncomment to run examples:
    # example_basic_probing()
    # example_multi_layer_probing()
    # example_structural_probing()

    print("\nTo run these examples:")
    print("1. Ensure you have a trained model checkpoint")
    print("2. Update paths to your data files")
    print("3. Adjust vocab_size and model dimensions to match your setup")
    print("4. Uncomment the example functions above")
