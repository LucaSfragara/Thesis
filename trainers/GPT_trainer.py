
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from trainers.base_trainer import BaseTrainer
from itertools import islice
from trainers.sequence_generator import SequenceGenerator
import numpy as np
class GPT_Trainer(BaseTrainer):
   
    def __init__(self, model, config, config_file, run_name, device=None):
       
        super().__init__(model, config, run_name, config_file, device)
       
        self.criterion = nn.CrossEntropyLoss(label_smoothing = config["loss"]["label_smoothing"])
       

    def train_epoch(self, train_dataloader, val_dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Training metrics and attention weights
        """
        
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")
        if self.scaler is None:
            raise ValueError("GradScaler is not initialized, initialize it first!")
        
        # Initialize training variables
        self.model.train()
        batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=True, position=0, initial = self.current_batch, desc=f"[Training LM]")
        running_ce_loss = 0.0
        total_tokens = 0
        best_val_loss = float('inf')

        # Only zero gradients when starting a new accumulation cycle
        self.optimizer.zero_grad()
        
        log_interval = self.config['training']['log_interval']
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        val_interval = self.config['training']['val_interval']
        
        
        for i, batch in enumerate(islice(train_dataloader, self.current_batch, None), start=self.current_batch):

            
            targets_shifted, targets_golden = batch
            
            targets_shifted, targets_golden= targets_shifted.to(self.device), targets_golden.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                
                raw_preds, attn_weights = self.model(targets_shifted) 

                raw_preds = raw_preds.view(-1, raw_preds.size(-1))        # (batch_size * seq_len, vocab_size)
                targets = targets_golden.view(-1)                         # (batch_size * seq_len)

                raw_loss = self.criterion(raw_preds, targets)
                
            # Calculate metrics with raw loss 
            
            batch_tokens = targets_shifted.size(0) * targets_shifted.size(1)
            total_tokens += batch_tokens
            running_ce_loss += raw_loss.item() * batch_tokens

            # Normalize loss for gradient accumulation
            loss = raw_loss / gradient_accumulation_steps
            
            # Backpropagate the loss
            self.scaler.scale(loss).backward()

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            
            # Only update weights after accumulating enough gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                # Only step scheduler here if it's not ReduceLROnPlateau
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                    
                self.scaler.update()
                self.optimizer.zero_grad()  # Reset gradients after update

            
            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss:.4f}",
                perplexity_token=f"{perplexity_token:.4f}",
                lr = f"{self.scheduler.get_last_lr()[0]:.6f}",
                acc_step=f"{(i % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}"
            )
            batch_bar.update()
            
            if i % val_interval == 0 and i  > 0:
                
                avg_ce_loss = running_ce_loss / total_tokens
                avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
                
                val_metrics, val_attn = self._validate_epoch(val_dataloader)
                
                train_metrics = {
                    'ce_loss_token': avg_ce_loss,
                    'perplexity_token': avg_perplexity_token.item(),
                }
                
                # Log metrics
                metrics = {
                    'train': train_metrics,
                    'val': val_metrics
                }
                self._log_metrics(metrics, i)
                
                # Save attention plots
                train_attn_keys = list(attn_weights.keys())
                val_attn_keys = list(val_attn.keys())
                
                self._save_attention_plot(attn_weights[train_attn_keys[0]][0], self.current_batch, "train_self")
                self._save_attention_plot(val_attn[val_attn_keys[0]][0], self.current_batch, "val_self")

                self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
                # Check if this is the best model
                val_loss = val_metrics['ce_loss_token']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_metric = val_loss
                    self.save_checkpoint('checkpoint-best-metric-model.pth')

                #set model back to training
                self.model.train()
             
            self.current_batch += 1
            # Clean up
            del targets_shifted, targets_golden, raw_preds, loss

        # Handle any remaining gradients at the end of the epoch
        if (len(train_dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
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
        val_batches = self.config['training']['val_batches']
        batch_bar = tqdm(val_batches, dynamic_ncols=True, leave=False, position=0, desc=f"[Validating LM]")
        running_ce_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(dataloader):
            
      
            targets_shifted, targets_golden = batch #shape (batch_size, seq_len)
            targets_shifted, targets_golden = targets_shifted.to(self.device), targets_golden.to(self.device)

            # Forward pass
            with torch.inference_mode():
               
                raw_preds, attn_weights = self.model.forward(targets_shifted)

                raw_preds = raw_preds.view(-1, raw_preds.size(-1))        # (batch_size * seq_len, vocab_size)
                targets = targets_golden.view(-1)                         # (batch_size * seq_len)
                loss = self.criterion(raw_preds, targets)

            # Calculate metrics
            batch_tokens = targets_shifted.size(0) * targets_shifted.size(1)
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
        
            if i > val_batches:
                break

            # Clean up
            del targets_shifted, targets_golden, raw_preds, loss
            

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        batch_bar.close()
        print("generating")
        generation_result = self.generate(dataloader)
        
   
        self._save_generated_text(generation_result, f"val_{self.current_batch}")

        return {
            'ce_loss_token': avg_ce_loss,
            'perplexity_token': avg_perplexity_token.item(),
        }, attn_weights
        
        
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
        if generation_config is None:
            print("Using default generation config...")
            # Greedy search (default)
            generation_config = {
                'num_samples': 5,
                'prompt_length': 20,
                'seed': 11785,
                #'max_length': self.model.max_len,
                'temperature': 1.0,
                'beam_width': 1,
                'repeat_penalty': 1.0,
                'top_k': 0,
                'top_p': 0.0    
            }

        # Create sequence generator
        generator = SequenceGenerator(
            score_fn=lambda x: self.model.score(x),
            max_length=320,
            device=self.device,
            config = self.config
        )

        # Sample prompts and get original sequences
        prompts, originals = dataloader.dataset.sample_prompts(
            num_prompts=generation_config.get('num_samples', 1),
            prompt_len=generation_config.get('prompt_length', 1),
            seed=generation_config.get('seed', 11785)
        )
        prompts = prompts.to(self.device)

        # Generate sequences based on method
        self.model.eval()
      
        with torch.inference_mode():
            if generation_config.get('top_k', 0) > 0 or generation_config.get('top_p', 0) > 0:
                print("Generating with sampling...")
                seqs, scores = NotImplementedError, NotImplementedError
                raise NotImplementedError # Remove if you implemented the sampling method
            elif generation_config.get('beam_width', 1) > 1:
                print("Generating with beam search...")
                seqs, scores = NotImplementedError, NotImplementedError
                raise NotImplementedError # Remove if you implemented the beam search method
                # Take best beam and score
                seqs = seqs[:, 0]
                scores = scores[:, 0]
            else:
                # Use greedy search for generation
                print("Generating with greedy search...")
                seqs, scores = generator.generate_greedy(
                    prompts,     
                    generation_config.get("temperature"),
                    generation_config.get("repeat_penalty")
                )

        # Post-process sequences (trim upto EOS token)
        #processed_seqs = generator.post_process_sequence(seqs, self.tokenizer)

        # Compile results
        # results is a dictionary with the following keys:
        # - prompt: the decoded prompt
        # - generated: the decoded generated sequence after the prompt
        # - original: the decoded original sequence after the prompt
        # - score: the score of the generated sequence
        processed_seqs = generator.post_process_sequence(seqs, self.config["data"]["eos_token"])
        results = []
        for _, (prompt, seq, score, original) in enumerate(zip(prompts, processed_seqs, scores, originals)):
         
            results.append({
                'prompt': "".join(np.array(prompt.to('cpu')).astype(str)),
                'original': "".join(np.array(original[len(prompt):].to('cpu')).astype(str)),
                'generated': "".join(np.array(seq[len(prompt):].to('cpu')).astype(str)),
                'score': score.item()
            })

        return results

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

        # Multinomial sampling config not yet implemented

        return {
            'greedy': greedy_config,
            'multinomial': NotImplementedError
        }
    
    
    