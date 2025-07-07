"""
This is the main training script for the SEDD Research Framework.

It orchestrates the entire training process, including:
- Loading the configuration from a YAML file.
- Setting up the device (GPU/CPU).
- Initializing Weights & Biases for logging.
- Loading the dataset and tokenizer.
- Initializing the model, optimizer, and diffusion process.
- Running the main training and validation loop.
- Handling checkpointing, including resuming from a checkpoint,
  saving periodic checkpoints, and saving the best model based on
  validation loss.
- Implementing early stopping to prevent overfitting.
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import wandb
from typing import Any

# Local module imports
from src.utils.config_loader import get_config
from src.data import get_dataloader
from src.model import TransformerModel
from src.diffusion.diffusion_process import DiffusionProcess
from src.diffusion.noise_schedule import get_noise_schedule
from src.diffusion.graph import UniformGraph, AbsorbingGraph
from src.losses import get_loss_fn

def main():
    """
    The main entry point for the training script.
    """
    # --- 1. Load Configuration and Setup ---
    config = get_config()
    
    # Setup device
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(config.logging.checkpoint_dir, config.exp_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 2. Initialize Weights & Biases ---
    wandb.init(
        project="sedd-research",
        name=config.exp_id,
        config=config
    )

    # --- 3. Load Data ---
    print("Loading data...")
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('validation', config)
    print("Data loaded.")

    # --- 4. Initialize Model and Optimizer ---
    print("Initializing model...")
    model = TransformerModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.training.learning_rate))
    scaler = GradScaler()
    print("Model initialized.")

    # --- 5. Initialize Diffusion Process ---
    print("Initializing diffusion process...")
    noise_schedule = get_noise_schedule(config.diffusion.noise_schedule)
    
    if config.diffusion.graph_type == "uniform":
        graph = UniformGraph(config.vocab.size)
    elif config.diffusion.graph_type == "absorbing":
        graph = AbsorbingGraph(config.vocab.size, config.vocab.mask_token_id)
    else:
        raise ValueError(f"Unknown graph type: {config.diffusion.graph_type}")
        
    diffusion_process = DiffusionProcess(noise_schedule, graph, config.diffusion)
    loss_fn = get_loss_fn(config)
    print("Diffusion process initialized.")

    # --- 6. Resume from Checkpoint (if applicable) ---
    start_epoch = 0
    # Note: The config can be extended to include a resume_from path
    # if hasattr(config, 'resume_from') and config.resume_from:
    #     print(f"Resuming from checkpoint: {config.resume_from}")
    #     checkpoint = torch.load(config.resume_from)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Resuming from epoch {start_epoch}")

    # --- 7. Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            with autocast():
                loss = loss_fn(model, batch, diffusion_process)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % config.logging.log_interval == 0:
                print(f"Epoch [{epoch+1}/{config.training.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                wandb.log({"train_loss": loss.item()})

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = loss_fn(model, batch, diffusion_process)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config.training.num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config
            }, best_checkpoint_path)
            print(f"New best model saved to {best_checkpoint_path}")
            wandb.save(best_checkpoint_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.training.patience:
            print(f"Early stopping triggered after {config.training.patience} epochs with no improvement.")
            break

        if (epoch + 1) % config.logging.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            wandb.save(checkpoint_path)

if __name__ == "__main__":
    main()
