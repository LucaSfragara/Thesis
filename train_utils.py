import torch


def save_model(model, optimizer, scheduler, metric, scaler, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         'scaler_state_dict'        : scaler.state_dict() if scaler is not None else {},
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )
    
def load_model(model, optimizer, scheduler, scaler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    
    metric = checkpoint['val_loss']
    
    return [model, optimizer, scheduler, epoch, metric]
