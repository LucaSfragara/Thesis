"""
Training infrastructure for linear probes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import wandb
from tqdm import tqdm
import os


class ProbeTrainer:
    """
    Trainer for linear probes on frozen model representations.

    This trainer handles:
    - Training linear classifiers on extracted hidden states
    - Evaluation and metric tracking
    - Checkpointing and logging
    - Support for multiple probing tasks

    Args:
        probe: The probe model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        optimizer: Optimizer for probe parameters
        criterion: Loss function (default: CrossEntropyLoss)
        device: Device to train on
        experiment_dir: Directory to save checkpoints and logs
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
    """

    def __init__(
        self,
        probe: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        experiment_dir: str = './probe_experiments',
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None
    ):
        self.probe = probe
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.probe.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.probe.parameters(), lr=1e-3, weight_decay=0.01)
        else:
            self.optimizer = optimizer

        # Setup loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = criterion

        # Create experiment directory
        os.makedirs(experiment_dir, exist_ok=True)

        # Setup wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project or 'probing',
                name=wandb_run_name or 'linear_probe',
                config={
                    'probe_type': type(probe).__name__,
                    'optimizer': type(optimizer).__name__ if optimizer else 'AdamW',
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                }
            )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            metrics: Dictionary of training metrics
        """
        self.probe.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            hidden_states = batch['hidden_states'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.probe(hidden_states)

            # Compute loss
            if logits.dim() == 3:
                # Sequence output: [batch, seq_len, num_classes]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                # Compute accuracy
                pred = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (pred == labels).masked_select(mask).sum().item()
                n_samples = mask.sum().item()
            else:
                # Single position output: [batch, num_classes]
                loss = self.criterion(logits, labels)
                # Compute accuracy
                pred = logits.argmax(dim=-1)
                correct = (pred == labels).sum().item()
                n_samples = len(labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += n_samples
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / n_samples:.4f}'
            })

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/step_accuracy': correct / n_samples,
                    'global_step': self.global_step
                })

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            metrics: Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.probe.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc='Evaluating'):
            # Move to device
            hidden_states = batch['hidden_states'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.probe(hidden_states)

            # Compute loss
            if logits.dim() == 3:
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                pred = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (pred == labels).masked_select(mask).sum().item()
                n_samples = mask.sum().item()
            else:
                loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=-1)
                correct = (pred == labels).sum().item()
                n_samples = len(labels)

            total_loss += loss.item()
            total_correct += correct
            total_samples += n_samples

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }

    def train(self, num_epochs: int, save_every: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the probe for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs (default: save only best)

        Returns:
            training_history: Dictionary with training metrics over time
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])

            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")

            # Evaluate
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])

                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_metrics['loss'],
                        'train/epoch_accuracy': train_metrics['accuracy'],
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy'],
                    })

                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_probe.pt', epoch, val_metrics)
                    print(f"Saved best model with val accuracy: {self.best_val_acc:.4f}")

            else:
                # No validation set, log only training
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/epoch_loss': train_metrics['loss'],
                        'train/epoch_accuracy': train_metrics['accuracy'],
                    })

            # Periodic checkpoint
            if save_every is not None and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'probe_epoch_{epoch + 1}.pt', epoch, train_metrics)

        # Save final model
        final_metrics = history
        self.save_checkpoint('final_probe.pt', num_epochs - 1, final_metrics)

        return history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]):
        """
        Save probe checkpoint.

        Args:
            filename: Name of checkpoint file
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
        """
        checkpoint_path = os.path.join(self.experiment_dir, filename)
        checkpoint = {
            'epoch': epoch,
            'probe_state_dict': self.probe.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load probe checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.probe.load_state_dict(checkpoint['probe_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        return checkpoint.get('metrics', {})
