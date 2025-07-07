"""
This is the main analysis script for the SEDD Research Framework.

It provides a comprehensive toolkit for evaluating and understanding the
behavior of trained diffusion models. It can perform:
- Quantitative evaluation (Perplexity).
- Qualitative analysis of generation diversity (Self-BLEU, Distinct-N).
- Qualitative analysis of contextual understanding (Infilling).
- Visualization of the forward and reverse diffusion processes.
"""

import torch
import os
import argparse
import yaml
from tqdm import tqdm
from typing import Any, Dict, Tuple
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
from transformers import BertTokenizer

# Local module imports
from src.utils.config_loader import dict_to_namespace
from src.data import get_dataloader
from src.model import TransformerModel
from src.diffusion.diffusion_process import DiffusionProcess
from src.diffusion.noise_schedule import get_noise_schedule
from src.diffusion.graph import UniformGraph, AbsorbingGraph
from src.losses import get_loss_fn

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[TransformerModel, Any]:
    """
    Loads a model and its configuration from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file.
        device (torch.device): The device to load the model onto.

    Returns:
        A tuple containing:
        - model (TransformerModel): The loaded model.
        - config (Any): The configuration object stored in the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if the config is in the checkpoint, otherwise load from default file
    if 'config' in checkpoint:
        print("Loading configuration from checkpoint.")
        # The config in the checkpoint is a dict, convert it to a namespace
        if isinstance(checkpoint['config'], dict):
            config = dict_to_namespace(checkpoint['config'])
        else: # It's already a namespace
            config = checkpoint['config']
    else:
        print("Warning: 'config' not found in checkpoint. Loading from default 'config.yaml'.")
        from src.utils.config_loader import load_config
        config_dict = load_config("config.yaml")
        config = dict_to_namespace(config_dict)

    model = TransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Successfully loaded model from {checkpoint_path}")
    return model, config

def calculate_perplexity(model, config, device):
    """Calculates perplexity on the test set."""
    print("\n--- Calculating Perplexity ---")
    test_loader = get_dataloader('test', config)
    loss_fn = get_loss_fn(config)
    
    # We need a diffusion process for the loss function
    noise_schedule = get_noise_schedule(config.diffusion.noise_schedule)
    if config.diffusion.graph_type == "uniform":
        graph = UniformGraph(config.vocab.size)
    else:
        graph = AbsorbingGraph(config.vocab.size, config.vocab.mask_token_id)
    diffusion_process = DiffusionProcess(noise_schedule, graph, config.diffusion, config.vocab)

    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Perplexity"):
            batch = batch.to(device)
            # The loss function expects a batch of clean data
            loss = loss_fn(model, batch, diffusion_process)
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)
            
    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss))
    
    print(f"  Average Loss: {average_loss:.4f}")
    print(f"  Perplexity: {perplexity.item():.4f}")
    return perplexity.item()

def analyze_diversity(model, config, diffusion_process, tokenizer, device, num_samples, guidance_scale):
    """Generates samples and calculates diversity metrics."""
    print(f"\n--- Analyzing Generation Diversity (Guidance Scale: {guidance_scale}) ---")
    
    # Generate samples
    samples = []
    if isinstance(diffusion_process.graph, AbsorbingGraph):
        start_tokens = torch.full((1, config.dataset.max_seq_len), config.vocab.mask_token_id, device=device, dtype=torch.long)
    else: # Uniform
        start_tokens = torch.randint(0, config.vocab.size, (1, config.dataset.max_seq_len), device=device, dtype=torch.long)

    for _ in tqdm(range(num_samples), desc="Generating Samples"):
        noisy_tokens = start_tokens.clone()
        for i in reversed(range(config.diffusion.num_timesteps)):
            t = torch.full((1,), (i + 1) / config.diffusion.num_timesteps, device=device)
            noisy_tokens = diffusion_process.remove_noise(model, noisy_tokens, t, guidance_scale)
        
        generated_text = tokenizer.decode(noisy_tokens.squeeze(), skip_special_tokens=True)
        samples.append(generated_text)

    # Calculate metrics
    hypotheses = [s.split() for s in samples]
    references = [[hypotheses[j] for j in range(len(hypotheses)) if i != j] for i in range(len(hypotheses))]
    # Use smoothing function to avoid issues with short sentences
    chencherry = SmoothingFunction()
    self_bleu = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)

    unigrams, bigrams = [], []
    for s in samples:
        tokens = s.split()
        unigrams.extend(tokens)
        bigrams.extend(zip(tokens, tokens[1:]))
        
    dist_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
    dist_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

    print(f"  Self-BLEU: {self_bleu:.4f} (Lower is more diverse)")
    print(f"  Distinct-1: {dist_1:.4f} (Higher is more diverse)")
    print(f"  Distinct-2: {dist_2:.4f} (Higher is more diverse)")
    
    return samples, {"self_bleu": self_bleu, "distinct_1": dist_1, "distinct_2": dist_2}

def analyze_infilling(model, config, diffusion_process, tokenizer, device, guidance_scale):
    """Analyzes the model's ability to fill in masked text."""
    print(f"\n--- Analyzing Infilling (Guidance Scale: {guidance_scale}) ---")
    sentence = "The quick brown fox jumps over the lazy dog near the river."
    tokens = tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=config.dataset.max_seq_len)
    
    masked_tokens = list(tokens)
    mask_start, mask_end = 5, 8 # "fox jumps over"
    for i in range(mask_start, mask_end):
        if i < len(masked_tokens):
            masked_tokens[i] = config.vocab.mask_token_id
            
    x_t_start = torch.tensor([masked_tokens], device=device)
    
    start_timestep = int(config.diffusion.num_timesteps * 0.5)
    noisy_tokens = x_t_start.clone()
    for i in reversed(range(start_timestep)):
        t = torch.full((1,), (i + 1) / config.diffusion.num_timesteps, device=device)
        noisy_tokens = diffusion_process.remove_noise(model, noisy_tokens, t, guidance_scale)
        
    infilled_text = tokenizer.decode(noisy_tokens.squeeze(), skip_special_tokens=True)
    
    print(f"  Original: {sentence}")
    print(f"  Masked:   {tokenizer.decode(x_t_start.squeeze())}")
    print(f"  Infilled: {infilled_text}")
    
    return infilled_text

def main(args):
    """Main entry point for the analysis script."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # --- Load Model and Config ---
    checkpoint_path = os.path.join(args.checkpoint_dir, args.exp_id, "best_checkpoint.pt")
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    
    # --- Setup ---
    tokenizer = BertTokenizer.from_pretrained(config.dataset.tokenizer)
    
    # Create a diffusion process based on the loaded config
    noise_schedule = get_noise_schedule(config.diffusion.noise_schedule)
    if config.diffusion.graph_type == "uniform":
        graph = UniformGraph(config.vocab.size)
    else:
        graph = AbsorbingGraph(config.vocab.size, config.vocab.mask_token_id)
    diffusion_process = DiffusionProcess(noise_schedule, graph, config.diffusion, config.vocab)

    # --- Run Analyses ---
    results = {}
    if "perplexity" in args.parts:
        results["perplexity"] = calculate_perplexity(model, config, device)
        
    if "diversity" in args.parts:
        samples, diversity_metrics = analyze_diversity(model, config, diffusion_process, tokenizer, device, args.num_samples, args.guidance_scale)
        results["diversity"] = diversity_metrics
        results["generated_samples"] = samples
        
    if "infilling" in args.parts:
        results["infilled_text"] = analyze_infilling(model, config, diffusion_process, tokenizer, device, args.guidance_scale)

    # --- Save Results ---
    output_dir = os.path.join("analysis_results", args.exp_id)
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f"summary_gs_{args.guidance_scale}.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nAnalysis summary saved to {results_path}")
    
    if "generated_samples" in results:
        samples_path = os.path.join(output_dir, f"generated_samples_gs_{args.guidance_scale}.txt")
        with open(samples_path, "w") as f:
            for sample in results["generated_samples"]:
                f.write(sample + "\n")
        print(f"Generated samples saved to {samples_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Analysis of SEDD Models")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID to analyze.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where checkpoints are saved.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use for analysis.")
    parser.add_argument("--parts", type=str, nargs='+', default=["perplexity", "diversity", "infilling"], help="Which analysis parts to run.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for diversity analysis.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Scale for Classifier-Free Guidance.")
    args = parser.parse_args()
    main(args)
