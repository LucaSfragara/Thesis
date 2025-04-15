
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from base_trainer import BaseTrainer

#from ..utils import create_scheduler
#from ..decoding.sequence_generator import SequenceGenerator

class LMTrainer(BaseTrainer):
   
    def __init__(self, model, pad_id, config, config_file, run_name, device=None):
       
        super().__init__(model, config, run_name, config_file, device)
       
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad_id, label_smoothing = config["loss"]["label_smoothing"])
       

    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Training metrics and attention weights
        """
        
        # Initialize training variables
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Training LM]")
        running_ce_loss = 0.0
        total_tokens = 0


        # Only zero gradients when starting a new accumulation cycle
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
        
            targets_shifted, targets_golden, lengths = batch
        
            targets_shifted, targets_golden, lengths = targets_shifted.to(self.device), targets_golden.to(self.device), lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):

                raw_preds, attn_weights = self.model(targets_shifted, lengths) 

                
                raw_preds = raw_preds.view(-1, raw_preds.size(-1))        # (batch_size * seq_len, vocab_size)
                targets = targets_golden.view(-1)                         # (batch_size * seq_len)

                raw_loss = self.criterion(raw_preds, targets)
                
            # Calculate metrics with raw loss 
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += raw_loss.item() * batch_tokens

            # Normalize loss for gradient accumulation
            loss = raw_loss / self.config['training']['gradient_accumulation_steps']
            
            # Backpropagate the loss
            self.scaler.scale(loss).backward()
        
            # Only update weights after accumulating enough gradients
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                # Only step scheduler here if it's not ReduceLROnPlateau
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                    
                self.scaler.update()
                self.optimizer.zero_grad()  # Reset gradients after update

            # Calculate metrics
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Clean up
            del targets_shifted, targets_golden, lengths, raw_preds, loss
            torch.cuda.empty_cache()

        # Handle any remaining gradients at the end of the epoch
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            # Only step scheduler here if it's not ReduceLROnPlateau
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
       
        batch_bar.close()

        return {
            'ce_loss_token': avg_ce_loss,
            'perplexity_token': avg_perplexity_token.item(),
        }, attn_weights
            
            
    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        
        Args:
            dataloader: DataLoader for validation data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Validation metrics and attention weights
        """

        
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Validating LM]")
        running_ce_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(dataloader):
            
      
            targets_shifted, targets_golden, lengths = batch
            targets_shifted, targets_golden, lengths = targets_shifted.to(self.device), targets_golden.to(self.device), lengths.to(self.device)

            # Forward pass
            with torch.inference_mode():
               
                raw_preds, attn_weights = self.model.forward(targets_shifted)

                raw_preds = raw_preds.view(-1, raw_preds.size(-1))        # (batch_size * seq_len, vocab_size)
                targets = targets_golden.view(-1)                         # (batch_size * seq_len)

                loss = self.criterion(raw_preds, targets)

            # Calculate metrics
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += loss.item() * batch_tokens

            # Update the progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
            )
            batch_bar.update()

            # Clean up
            del targets_shifted, targets_golden, lengths, raw_preds, loss
            torch.cuda.empty_cache()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        batch_bar.close()

        return {
            'ce_loss_token': avg_ce_loss,
            'perplexity_token': avg_perplexity_token.item(),
        }, attn_weights
        

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for language model training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: int, number of epochs to train
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")
        

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            
          
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            
            
            val_metrics, val_attn = self._validate_epoch(val_dataloader)

          
            #gen_results = self.generate(val_dataloader)
            
            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['ce_loss_char'])

            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)
            
            # Save attention plots
            train_attn_keys = list(train_attn.keys())
            val_attn_keys = list(val_attn.keys())
            
            self._save_attention_plot(train_attn[train_attn_keys[0]][0], epoch, "train_self")
            self._save_attention_plot(val_attn[val_attn_keys[0]][0], epoch, "val_self")

            # Save generated text
            #self._save_generated_text(gen_results, f'val_epoch_{epoch}')

            # Save checkpoints
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Check if this is the best model
            val_loss = val_metrics['ce_loss_char']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metric = val_loss
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def generate(self, dataloader, generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating sequences from prompts.
        
        Args:
            dataloader: DataLoader containing the evaluation data
            generation_config: Optional dictionary containing generation parameters:
                - num_samples: int, number of samples to generate
                - prompt_length: int, length of prompts
                - seed: int, random seed
                - max_length: int, maximum sequence length
                - temperature: float, sampling temperature
                - beam_width: int, beam search width
                - repeat_penalty: float, penalty for repeated tokens
                - top_k: int, top-k filtering value
                - top_p: float, nucleus sampling threshold
        Returns:
            Dict containing generation results with prompts, originals, and generated sequences
        """
        #TODO: Implement the generation logic, i.e. multinomial sampling
        pass

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of generation configurations for evaluation.
        
        Returns:
            Dictionary containing generation configurations
        """
        common_config = {
            'num_samples': 50,
            'prompt_length': 10,
            'seed': 11785,
            'max_length': self.model.max_len,
        }
        
        greedy_config = common_config.copy()
        
        greedy_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 0,
            'top_p': 0.0
        })
        
        #TODO: add multinomial sampling config
        
        return {
            'greedy': greedy_config,
            'multinomial': NotImplementedError
        }
    
    
    