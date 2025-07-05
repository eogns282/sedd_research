import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import argparse
import wandb
from typing import Any
from configs import base_config as config
from data import get_dataloader
from model import TransformerModel
from diffusion.diffusion_process import DiffusionProcess
from diffusion.noise_schedule import get_noise_schedule
from diffusion.graph import UniformGraph
from losses import get_loss_fn

def main(args):
    """
    The main training script for the SEDD research baseline.
    This script handles the entire training process, including data loading,
    model initialization, training, validation, and checkpointing.
    """
    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Wandb ---
    wandb.init(
        project="sedd-research",
        config={k: v for k, v in config.__dict__.items() if not k.startswith('__')}
    )

    # --- Data ---
    print("Loading data...")
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('validation', config)
    print("Data loaded.")

    # --- Model ---
    print("Initializing model...")
    model = TransformerModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()
    print("Model initialized.")

    # --- Diffusion ---
    print("Initializing diffusion process...")
    noise_schedule = get_noise_schedule(config)
    graph = UniformGraph(config.VOCAB_SIZE)
    diffusion_process = DiffusionProcess(noise_schedule, graph, config)
    loss_fn = get_loss_fn(config)
    print("Diffusion process initialized.")

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            with autocast():
                loss = loss_fn(model, batch, diffusion_process)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % config.LOG_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                wandb.log({"train_loss": loss.item()})

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = loss_fn(model, batch, diffusion_process)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        # --- Save Checkpoint ---
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            wandb.save(checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use for training.")
    args = parser.parse_args()
    main(args)