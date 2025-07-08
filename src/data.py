import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from typing import Dict, Any, Tuple
import os
from collections import Counter
import pickle

def get_or_create_token_distribution(config: Any) -> torch.Tensor:
    """
    Calculates and caches the token frequency distribution from the training dataset.
    If a cached version exists, it loads it. Otherwise, it computes the distribution
    and saves it to a file for future use.

    Args:
        config (Any): The configuration object.

    Returns:
        torch.Tensor: A tensor representing the probability distribution of tokens.
    """
    dist_cache_path = os.path.join(config.dataset.cache_dir, f"{config.dataset.name}_token_dist.pkl")
    
    if os.path.exists(dist_cache_path):
        print(f"Loading cached token distribution from {dist_cache_path}")
        with open(dist_cache_path, 'rb') as f:
            return pickle.load(f)

    print("Calculating token distribution from scratch...")
    # Load the full training dataset
    train_dataset = load_dataset(config.dataset.name, config.dataset.config, split='train')
    tokenizer = BertTokenizer.from_pretrained(config.dataset.tokenizer)
    
    # Count all tokens in the dataset
    token_counts = Counter()
    for example in train_dataset:
        text = example['text']
        if text:
            # Do not pad or truncate, we want the raw token counts
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_counts.update(token_ids)

    # Create the probability distribution
    vocab_size = tokenizer.vocab_size
    distribution = torch.zeros(vocab_size, dtype=torch.float32)
    for token_id, count in token_counts.items():
        distribution[token_id] = count
    
    # Normalize to get probabilities
    distribution /= distribution.sum()

    # Cache the result
    os.makedirs(os.path.dirname(dist_cache_path), exist_ok=True)
    with open(dist_cache_path, 'wb') as f:
        pickle.dump(distribution, f)
    print(f"Saved token distribution to {dist_cache_path}")

    return distribution

class WikiTextDataset(Dataset):
    """
    A PyTorch Dataset for the WikiText-2 dataset.
    This class handles loading the data, tokenizing it, and preparing it for the model.
    """
    def __init__(self, split: str, config: Any, debug: bool = False):
        """
        Initializes the dataset.

        Args:
            split (str): The dataset split to load (e.g., 'train', 'validation', 'test').
            config (Any): The configuration object with dataset settings.
            debug (bool): If True, loads only a small subset of the data.
        """
        self.dataset = load_dataset(config.dataset.name, config.dataset.config, split=split)
        if debug:
            self.dataset = self.dataset.select(range(100)) # Use only 100 samples for debugging
            
        self.tokenizer = BertTokenizer.from_pretrained(config.dataset.tokenizer)
        self.max_len = config.dataset.max_seq_len

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The tokenized input IDs for the sample.
        """
        text: str = self.dataset[idx]['text']
        if not text:  # Handle empty strings
            text = " "
        
        tokens: Dict[str, torch.Tensor] = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokens['input_ids'].squeeze()

def get_dataloader(split: str, config: Any, debug: bool = False) -> Tuple[DataLoader, BertTokenizer]:
    """
    Creates a PyTorch DataLoader for a given dataset split.

    Args:
        split (str): The dataset split to create the DataLoader for.
        config (Any): The configuration object with dataloader settings.
        debug (bool): If True, uses a small subset of the data for quick testing.

    Returns:
        Tuple[DataLoader, BertTokenizer]: The PyTorch DataLoader and the tokenizer.
    """
    dataset = WikiTextDataset(split, config, debug=debug)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    return dataloader, dataset.tokenizer
