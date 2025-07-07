import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from typing import Dict, Any

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

def get_dataloader(split: str, config: Any, debug: bool = False) -> DataLoader:
    """
    Creates a PyTorch DataLoader for a given dataset split.

    Args:
        split (str): The dataset split to create the DataLoader for.
        config (Any): The configuration object with dataloader settings.
        debug (bool): If True, uses a small subset of the data for quick testing.

    Returns:
        DataLoader: The PyTorch DataLoader.
    """
    dataset = WikiTextDataset(split, config, debug=debug)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )