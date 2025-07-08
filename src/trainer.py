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
import argparse

# Local module imports
from src.utils.config_loader import get_config
from src.data import get_dataloader, get_or_create_token_distribution
from src.model import TransformerModel
from src.diffusion.diffusion_process import DiffusionProcess
from src.diffusion.noise_schedule import get_noise_schedule
from src.diffusion.graph import UniformGraph, AbsorbingGraph, HybridGraph, FrequencyGraph, FrequencyHybridGraph
from src.losses import get_loss_fn

def main():
    """
    The main entry point for the training script.
    """
    # --- 1. Load Configuration and Setup ---
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode.")
    args = parser.parse_args()

    config = get_config(args.config)
    config.debug = args.debug # Add debug flag to config
    
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
        config=vars(config) # Log the config dictionary
    )

    # --- 3. Load Data ---
    print("Loading data...")
    train_loader, tokenizer = get_dataloader('train', config, debug=config.debug)
    val_loader, _ = get_dataloader('validation', config, debug=config.debug)
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
    
    graph_args = {
        "vocab_size": config.vocab.size,
        "mask_token_id": tokenizer.mask_token_id,
        "mask_ratio": getattr(config.diffusion, 'mask_ratio', None),
        "device": device
    }

    if config.diffusion.graph_type == "uniform":
        graph = UniformGraph(**graph_args)
    elif config.diffusion.graph_type == "absorbing":
        graph = AbsorbingGraph(**graph_args)
    elif config.diffusion.graph_type == "hybrid":
        graph = HybridGraph(**graph_args)
    elif config.diffusion.graph_type == "frequency":
        token_dist = get_or_create_token_distribution(config)
        graph_args["token_distribution"] = token_dist
        graph = FrequencyGraph(**graph_args)
    elif config.diffusion.graph_type == "frequency_hybrid":
        token_dist = get_or_create_token_distribution(config)
        graph_args["token_distribution"] = token_dist
        graph = FrequencyHybridGraph(**graph_args)
    else:
        raise ValueError(f"Unknown graph type: {config.diffusion.graph_type}")
        
    diffusion_process = DiffusionProcess(noise_schedule, graph, config.diffusion, config.vocab)
    loss_fn = get_loss_fn(config)
    print("Diffusion process initialized.")

    # --- 6. Resume from Checkpoint (if applicable) ---
    start_epoch = 0
    # ... (resume logic can be added here) ...

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
                loss, _ = loss_fn(model, batch, diffusion_process)
            
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
                loss, _ = loss_fn(model, batch, diffusion_process, analysis_mode=True)
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
                'config': vars(config) # Save config dict
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
                'config': vars(config) # Save config dict
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            wandb.save(checkpoint_path)

if __name__ == "__main__":
    main()
