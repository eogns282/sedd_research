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
from typing import Any, Dict, Tuple, List
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
from transformers import BertTokenizer
import torch.nn.functional as F

# Local module imports
from src.utils.config_loader import dict_to_namespace
from src.data import get_dataloader, get_or_create_token_distribution
from src.model import TransformerModel
from src.diffusion.diffusion_process import DiffusionProcess
from src.diffusion.noise_schedule import get_noise_schedule
from src.diffusion.graph import UniformGraph, AbsorbingGraph, HybridGraph, FrequencyGraph, FrequencyHybridGraph
from src.losses import get_loss_fn

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[TransformerModel, Any]:
    """
    Loads a model and its configuration from a checkpoint file.
    Handles both old (Namespace) and new (dict) config formats.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    # Allow loading of legacy checkpoints that saved argparse.Namespace
    import argparse
    import torch.serialization
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' not in checkpoint:
        raise ValueError("'config' not found in checkpoint.")

    config_data = checkpoint['config']
    if isinstance(config_data, dict):
        # New format: config is saved as a dictionary
        config = dict_to_namespace(config_data)
    else:
        # Old format: config is an argparse.Namespace
        config = config_data

    model = TransformerModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Successfully loaded model from {checkpoint_path}")
    return model, config

def generate_basic(model: TransformerModel, diffusion_process: DiffusionProcess, config: Any, device: torch.device, num_samples: int, guidance_scale: float) -> torch.Tensor:
    """Performs a standard, single-pass generation."""
    model.eval()
    
    # Start with pure noise
    start_tokens = torch.randint(0, config.vocab.size, (num_samples, config.dataset.max_seq_len), device=device, dtype=torch.long)
    
    x_t = start_tokens
    for i in tqdm(reversed(range(config.diffusion.num_timesteps)), desc="Basic Generation", total=config.diffusion.num_timesteps):
        t = torch.full((x_t.shape[0],), i / config.diffusion.num_timesteps, device=device)
        x_t = diffusion_process.remove_noise(model, x_t, t, guidance_scale)
        
    return x_t

def generate_refined(model: TransformerModel, diffusion_process: DiffusionProcess, config: Any, device: torch.device, num_samples: int, guidance_scale: float) -> torch.Tensor:
    """Performs a two-pass, self-conditioned refinement generation."""
    model.eval()
    
    # Pass 1: Generate draft
    draft_tokens = generate_basic(model, diffusion_process, config, device, num_samples, guidance_scale)
    
    # Pass 2: Refine draft
    refine_start_t = getattr(config.analysis, 'refine_start_t', 250)
    t_float = torch.full((draft_tokens.shape[0],), refine_start_t / config.diffusion.num_timesteps, device=device)
    
    noisy_draft, _, _ = diffusion_process.add_noise(draft_tokens, t_float)

    x_t = noisy_draft
    for i in tqdm(reversed(range(refine_start_t)), desc="Refining Draft", total=refine_start_t):
        t_float_loop = torch.full((x_t.shape[0],), i / config.diffusion.num_timesteps, device=device)
        x_t = diffusion_process.remove_noise(model, x_t, t_float_loop, guidance_scale, draft_tokens=draft_tokens)
            
    return x_t

def calculate_perplexity(model, config, device):
    """Calculates perplexity on the test set."""
    print("\n--- Calculating Perplexity ---")
    test_loader, _ = get_dataloader('test', config)
    loss_fn = get_loss_fn(config)
    
    noise_schedule = get_noise_schedule(config.diffusion.noise_schedule)
    graph = UniformGraph(vocab_size=config.vocab.size) # Dummy graph for loss
    diffusion_process = DiffusionProcess(noise_schedule, graph, config.diffusion, config.vocab)

    total_nll = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Perplexity"):
            batch = batch.to(device)
            loss, mask = loss_fn(model, batch, diffusion_process, analysis_mode=True)
            # Perplexity is the exponential of the negative log-likelihood per token
            total_nll += loss.item() * batch.numel()
            total_tokens += batch.numel()
            
    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll))
    
    print(f"  Average NLL: {avg_nll:.4f}")
    print(f"  Perplexity: {perplexity.item():.4f}")
    return perplexity.item()

def analyze_diversity(samples: List[str]) -> Dict[str, float]:
    """Calculates diversity metrics for a list of generated samples."""
    hypotheses = [s.split() for s in samples]
    references = [[hypotheses[j] for j in range(len(hypotheses)) if i != j] for i in range(len(hypotheses))]
    chencherry = SmoothingFunction()
    self_bleu = corpus_bleu(references, hypotheses, smoothing_function=chencherry.method1)

    unigrams, bigrams = [], []
    for s in samples:
        tokens = s.split()
        unigrams.extend(tokens)
        bigrams.extend(zip(tokens, tokens[1:]))
        
    dist_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
    dist_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
    
    metrics = {"self_bleu": self_bleu, "distinct_1": dist_1, "distinct_2": dist_2}
    print(f"  Self-BLEU: {self_bleu:.4f} (Lower is more diverse)")
    print(f"  Distinct-1: {dist_1:.4f} (Higher is more diverse)")
    print(f"  Distinct-2: {dist_2:.4f} (Higher is more diverse)")
    return metrics

def main(args):
    """Main entry point for the analysis script."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = os.path.join(args.checkpoint_dir, args.exp_id, "best_checkpoint.pt")
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    
    tokenizer = BertTokenizer.from_pretrained(config.dataset.tokenizer)
    
    graph_args = {
        "vocab_size": config.vocab.size,
        "mask_token_id": tokenizer.mask_token_id,
        "mask_ratio": getattr(config.diffusion, 'mask_ratio', None),
        "device": device
    }
    if config.diffusion.graph_type == "frequency":
        graph_args["token_distribution"] = get_or_create_token_distribution(config)
        graph = FrequencyGraph(**graph_args)
    elif config.diffusion.graph_type == "uniform":
        graph = UniformGraph(**graph_args)
    elif config.diffusion.graph_type == "absorbing":
        graph = AbsorbingGraph(**graph_args)
    elif config.diffusion.graph_type == "hybrid":
        graph = HybridGraph(**graph_args)
    elif config.diffusion.graph_type == "frequency_hybrid":
        graph_args["token_distribution"] = get_or_create_token_distribution(config)
        graph = FrequencyHybridGraph(**graph_args)
    else:
        raise ValueError(f"Unknown graph type: {config.diffusion.graph_type}")
        
    noise_schedule = get_noise_schedule(config.diffusion.noise_schedule)
    diffusion_process = DiffusionProcess(noise_schedule, graph, config.diffusion, config.vocab)

    results = {}
    if "perplexity" in args.parts:
        results["perplexity"] = calculate_perplexity(model, config, device)
        
    if "diversity" in args.parts:
        print(f"\n--- Analyzing Basic Generation Diversity (Guidance Scale: {args.guidance_scale}) ---")
        generated_tokens = generate_basic(model, diffusion_process, config, device, args.num_samples, args.guidance_scale)
        samples = [tokenizer.decode(s, skip_special_tokens=True) for s in generated_tokens]
        results["diversity"] = analyze_diversity(samples)
        results["generated_samples"] = samples

    if "refined_diversity" in args.parts:
        print(f"\n--- Analyzing Refined Generation Diversity (Guidance Scale: {args.guidance_scale}) ---")
        generated_tokens = generate_refined(model, diffusion_process, config, device, args.num_samples, args.guidance_scale)
        samples = [tokenizer.decode(s, skip_special_tokens=True) for s in generated_tokens]
        results["refined_diversity"] = analyze_diversity(samples)
        results["refined_generated_samples"] = samples

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

    if "refined_generated_samples" in results:
        samples_path = os.path.join(output_dir, f"refined_generated_samples_gs_{args.guidance_scale}.txt")
        with open(samples_path, "w") as f:
            for sample in results["refined_generated_samples"]:
                f.write(sample + "\n")
        print(f"Refined generated samples saved to {samples_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Analysis of SEDD Models")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID to analyze.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where checkpoints are saved.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use for analysis.")
    parser.add_argument("--parts", type=str, nargs='+', default=["perplexity", "diversity"], help="Which analysis parts to run.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for diversity analysis.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Scale for Classifier-Free Guidance.")
    args = parser.parse_args()
    main(args)
