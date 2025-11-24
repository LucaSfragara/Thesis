import torch


def save_model(model, optimizer, scheduler, metric, scaler, epoch, path):
    """
    Save model checkpoint with training state.

    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: Learning rate scheduler (can be None)
        metric: Tuple of (metric_name, metric_value) e.g., ('val_loss', 0.5)
        scaler: GradScaler for mixed precision training (can be None)
        epoch: Current epoch number
        path: Path to save the checkpoint
    """
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
    """
    Load model checkpoint and restore training state.

    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: Learning rate scheduler to load state into (can be None)
        scaler: GradScaler to load state into (can be None)
        path: Path to the checkpoint file

    Returns:
        list: [model, optimizer, scheduler, epoch, metric_value]
              Where metric_value is the validation loss from the checkpoint
    """
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
