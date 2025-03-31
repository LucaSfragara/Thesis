from math import log
from re import T
from turtle import forward
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary
from datasets import CFGDataset
from grammars import GRAMMAR_CFG3b
from models import model_basic, model_large1
import wandb
import numpy as np
import os
from train_utils import save_model, load_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

model = model_large1

CHECKPOINT_PATH = "checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
MODEL_NAME = "lstm_e30_e128_h512-cfg3b.pth"

def train(model, train_loader, optimizer, criterion, epoch, total_epochs):
    
    model.train()
    
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0
    total_accuracy = 0
    
    start_prob = 0.2
    end_prob = 0.7
    teacher_forcing_prob = start_prob + (end_prob - start_prob) * (epoch / total_epochs)
    
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = data
        x = x.to(device)
        y = y.to(device)
        
        batch_size, seq_len  = x.size()
        optimizer.zero_grad()
        
        hidden = None
        outputs = []
        
        
        
        with torch.cuda.amp.autocast():
            
            input_token = x[:,0].unsqueeze(1) #shape (batch_size,1)
            
            for j in range(seq_len):
                
                logits, hidden = model(input_token, hidden)
                logits = logits.squeeze(1) #shape (batch_size, vocab_size)
                outputs.append(logits)
                
                #do teacher forcing for the first 5 epoch
                if epoch <= 5:
                    if j < seq_len - 1:
                        input_token = x[:, j+1].unsqueeze(1)
                elif epoch > 5 and epoch <= 25:
                    
                    if j  < seq_len - 1:
                        #add teacher forcing with probability 0.5
                        if np.random.rand() < teacher_forcing_prob:
                            _, predicted = torch.max(logits, dim=1)
                            input_token = predicted.unsqueeze(1)
                        else:
                            input_token = x[:, j+1].unsqueeze(1)
                else:
                    _, predicted = torch.max(logits, dim=1)
                    input_token = predicted.unsqueeze(1)

            outputs = torch.stack(outputs, dim=1) #shape (batch_size, seq_len, vocab_size)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            total_loss += loss.item()
            
            total_accuracy += (torch.argmax(outputs, dim=2) == y).sum().item() / (batch_size * seq_len)
            
            batch_bar.set_postfix(
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                acc = "{:.04f}".format(float(total_accuracy)/(i+1)),
                lr="{:.06f}".format(float(optimizer.param_groups[0]['lr']))
                )

            batch_bar.update() # Update tqdm bar

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            del x, y, outputs, loss, hidden, data, batch_size, seq_len, input_token,logits
            torch.cuda.empty_cache()
            
    batch_bar.close()
            
    return total_loss/len(train_loader), total_accuracy/len(train_loader)
                
def validate_model(model:nn.Module, val_loader, criterion):
    
    model.eval()
    
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')
    
    total_loss = 0
    total_accuracy = 0
    
    for i, data in enumerate(val_loader):
        
        x, y = data
        x = x.to(device)
        y = y.to(device)
        hidden = None
        
        batch_size, seq_len = x.size()
        outputs = []
        
        input_token = x[:,0].unsqueeze(1)

        with torch.inference_mode():
            
            for j in range(seq_len):
                
                logits, hidden = model(input_token, hidden)
                logits = logits.squeeze(1) #shape (batch_size, vocab_size)
                
                outputs.append(logits)
            
                if j < seq_len - 1:
                    input_token = x[:, j+1].unsqueeze(1)
            
            outputs = torch.stack(outputs, dim=1) #shape (batch_size, seq_len, vocab_size)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            acc = (torch.argmax(outputs, dim=2) == y).sum().item() / (batch_size * seq_len)
            total_loss += loss.item()
            total_accuracy += acc
            
            batch_bar.set_postfix(
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                acc="{:.04f}".format(float(total_accuracy) / (i + 1))
            )
            batch_bar.update()
            
            # Free up memory
            del x, y, outputs, loss, hidden, batch_size, seq_len, input_token, logits
            torch.cuda.empty_cache()
            
    return total_loss/len(val_loader), total_accuracy/len(val_loader)
            
    

train_data =  CFGDataset(data_file="cfg_sentences_train_cfg3b.pkl", subset = .2)
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, collate_fn=train_data.collate_fn, num_workers=12, pin_memory=True)

val_data = CFGDataset(data_file="cfg_sentences_val_cfg3b.pkl", subset = .2)
val_loader = DataLoader(val_data, batch_size=512, shuffle=True, collate_fn=val_data.collate_fn, num_workers=12, pin_memory=True)


     
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x, y = next(iter(train_loader))

print(summary(model, input_data = [x.to(device)]))


"""
wandb.login(key="11902c0c8e2c6840d72bf65f04894b432d85f019")

wandb.init(
        name = "LSTM-cfg3b"
        project="thesis",
        reinit = True,
        )
"""
num_epochs = 30

last_epoch_completed = 0

RESUME_TRAINING = False

if RESUME_TRAINING:
    
    best_model_path = 'checkpoints/lstm_e30_e32_h16-cfg3b.pth'
    model, optimizer, scheduler, last_epoch_completed, metric = load_model(model, optimizer, None, scaler, best_model_path)

print("Last epoch completed: ", last_epoch_completed)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-5)

#wandb.watch(model, log="all")

for epoch in range(last_epoch_completed, num_epochs):
    
    print("\nEpoch: {}/{}".format(epoch + 1, num_epochs))
    curr_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, num_epochs)
    val_los, val_acc = validate_model(model, val_loader, criterion)
    
    print("Train Loss: {:.04f}, Train Acc: {:.04f}".format(train_loss, train_acc))
    print("Val Loss: {:.04f}, Val Acc: {:.04f}".format(val_los, val_acc))
    
    #wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_los, "val_acc": val_acc})
    scheduler.step()
    checkpoint_path = os.path.join(CHECKPOINT_PATH, MODEL_NAME)
    save_model(model, optimizer, None, ("val_loss", val_los), scaler, epoch, checkpoint_path)
    #artifact = wandb.Artifact("LSTM-cfg3b", type="model")
    #artifact.add_file(check_point_path)
    #wandb.log_artifact(artifact)


    