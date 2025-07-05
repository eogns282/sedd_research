import torch
import os
from typing import Any
from configs import base_config as config
from model import TransformerModel
from diffusion.diffusion_process import DiffusionProcess
from diffusion.noise_schedule import get_noise_schedule
from diffusion.graph import UniformGraph
from transformers import BertTokenizer

def main():
    """
    Generates text from a trained SEDD model checkpoint.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # --- Model ---
    print("Initializing model...")
    model = TransformerModel(config).to(device)
    
    # Find the latest checkpoint
    latest_checkpoint_path = None
    if os.path.exists(config.CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pt')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, latest_checkpoint)

    if latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found. Please train the model first.")
        return
        
    model.eval()
    print("Model initialized.")

    # --- Diffusion ---
    print("Initializing diffusion process...")
    noise_schedule = get_noise_schedule(config)
    graph = UniformGraph(config.VOCAB_SIZE)
    diffusion_process = DiffusionProcess(noise_schedule, graph, config)
    print("Diffusion process initialized.")

    # --- Generation ---
    print("Generating text...")
    # Start with a sequence of [MASK] tokens
    noisy_tokens = torch.full((1, config.MAX_SEQ_LEN), diffusion_process.mask_token_id, device=device, dtype=torch.long)
    
    for i in reversed(range(config.NUM_TIMESTEPS)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        noisy_tokens = diffusion_process.remove_noise(model, noisy_tokens, t)
        
    generated_text = tokenizer.decode(noisy_tokens.squeeze(), skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(generated_text)

if __name__ == "__main__":
    main()