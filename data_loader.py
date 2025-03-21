import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class HandwritingDataset(Dataset):
    """Dataset class for loading handwriting data"""
    
    def __init__(self, data_dir, split='training', max_seq_length=300, transform=None):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Directory containing the dataset
            split (str): 'training' or 'validation'
            max_seq_length (int): Maximum sequence length
            transform (callable, optional): Optional transform to apply to samples
        """
        self.data_dir = os.path.join(data_dir, split)
        self.max_seq_length = max_seq_length
        self.transform = transform
        
        # Load the necessary data files
        self.strokes = np.load(os.path.join(self.data_dir, 'strokes.npy'), allow_pickle=True)
        self.texts = np.load(os.path.join(self.data_dir, 'texts.npy'), allow_pickle=True)
        self.mean = np.load(os.path.join(self.data_dir, 'mean.npy'), allow_pickle=True)
        self.std = np.load(os.path.join(self.data_dir, 'std.npy'), allow_pickle=True)
        self.char_labels = np.load(os.path.join(self.data_dir, 'char_labels.npy'), allow_pickle=True)
        self.alphabet = np.load(os.path.join(self.data_dir, 'alphabet.npy'), allow_pickle=True)
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the stroke data"""
        # Normalize strokes
        self.normalized_strokes = []
        for stroke in self.strokes:
            # Ensure sequence is not longer than max_seq_length
            if len(stroke) > self.max_seq_length:
                stroke = stroke[:self.max_seq_length]
            
            # Normalize stroke data (x, y coordinates)
            normalized_stroke = (stroke - self.mean) / self.std
            self.normalized_strokes.append(normalized_stroke)
        
        # Convert texts to character indices
        self.text_indices = []
        for text in self.texts:
            # Convert each character to its index
            indices = [self.char_to_idx.get(char, 0) for char in text]  # Use 0 for unknown chars
            self.text_indices.append(indices)
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.normalized_strokes)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset"""
        # Get stroke data and corresponding text
        stroke = self.normalized_strokes[idx]
        text = self.text_indices[idx]
        
        # Convert to tensors
        stroke_tensor = torch.tensor(stroke, dtype=torch.float32)
        text_tensor = torch.tensor(text, dtype=torch.long)
        
        # Apply transforms if specified
        if self.transform:
            stroke_tensor = self.transform(stroke_tensor)
        
        return {
            'stroke': stroke_tensor,
            'text': text_tensor,
            'raw_text': self.texts[idx]
        }

def get_data_loader(data_dir, split='training', batch_size=64, shuffle=True, num_workers=4):
    """
    Create a data loader for the handwriting dataset
    
    Args:
        data_dir (str): Directory containing the dataset
        split (str): 'training' or 'validation'
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for loading data
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = HandwritingDataset(data_dir, split)
    
    # Create a collate function to handle variable length sequences
    def collate_fn(batch):
        # Sort batch by stroke sequence length (descending)
        batch.sort(key=lambda x: len(x['stroke']), reverse=True)
        
        # Get stroke sequences and their lengths
        strokes = [item['stroke'] for item in batch]
        texts = [item['text'] for item in batch]
        raw_texts = [item['raw_text'] for item in batch]
        
        # Get lengths of each sequence
        stroke_lengths = [len(s) for s in strokes]
        text_lengths = [len(t) for t in texts]
        
        # Pad sequences to the same length
        padded_strokes = torch.nn.utils.rnn.pad_sequence(strokes, batch_first=True)
        padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
        
        return {
            'stroke': padded_strokes,
            'text': padded_texts,
            'stroke_lengths': torch.tensor(stroke_lengths),
            'text_lengths': torch.tensor(text_lengths),
            'raw_text': raw_texts
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
