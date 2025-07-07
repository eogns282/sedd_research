import torch
import os
import argparse
from tqdm import tqdm
from src.configs import base_config as config
from src.data import get_dataloader
from src.model import TransformerModel
from src.diffusion.diffusion_process import DiffusionProcess
from src.diffusion.noise_schedule import get_noise_schedule
from src.diffusion.graph import UniformGraph
from src.losses import get_loss_fn

def find_latest_experiment(checkpoint_dir):
    """Finds the latest experiment directory."""
    exp_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not exp_dirs:
        return None
    return max(exp_dirs, key=lambda d: os.path.getmtime(os.path.join(checkpoint_dir, d)))

def main(args):
    """
    Evaluates a trained SEDD model by calculating the perplexity on the test set.
    """
    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading test data...")
    test_loader = get_dataloader('test', config)
    print("Data loaded.")

    # --- Model ---
    print("Initializing model...")
    model = TransformerModel(config).to(device)
    
    # Find the latest experiment and checkpoint
    if args.exp_id:
        exp_dir = os.path.join(config.CHECKPOINT_DIR, args.exp_id)
    else:
        latest_exp = find_latest_experiment(config.CHECKPOINT_DIR)
        if not latest_exp:
            print("No experiments found in the checkpoint directory.")
            return
        exp_dir = os.path.join(config.CHECKPOINT_DIR, latest_exp)

    latest_checkpoint_path = None
    if os.path.exists(exp_dir):
        checkpoints = [f for f in os.listdir(exp_dir) if f.endswith('.pt')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            latest_checkpoint_path = os.path.join(exp_dir, latest_checkpoint)

    if latest_checkpoint_path:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"No checkpoint found in {exp_dir}. Please train the model first.")
        return
        
    model.eval()
    print("Model initialized.")

    # --- Diffusion ---
    print("Initializing diffusion process...")
    noise_schedule = get_noise_schedule(config)
    graph = UniformGraph(config.VOCAB_SIZE)
    diffusion_process = DiffusionProcess(noise_schedule, graph, config)
    loss_fn = get_loss_fn(config)
    print("Diffusion process initialized.")

    # --- Evaluation ---
    print("Calculating perplexity on the test set...")
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            loss = loss_fn(model, batch, diffusion_process)
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)
            
    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss))
    
    print("\n--- Evaluation Results ---")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Perplexity:     {perplexity.item():.4f}")

    # --- Save Results ---
    eval_dir = os.path.join("evaluation", os.path.basename(exp_dir))
    os.makedirs(eval_dir, exist_ok=True)
    results_path = os.path.join(eval_dir, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("--- Evaluation Results ---\n")
        f.write(f"Checkpoint: {latest_checkpoint_path}\n")
        f.write(f"Average Loss: {average_loss:.4f}\n")
        f.write(f"Perplexity:     {perplexity.item():.4f}\n")
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use for evaluation.")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID to evaluate. If not provided, the latest experiment will be used.")
    args = parser.parse_args()
    main(args)

