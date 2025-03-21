import os
import argparse
import numpy as np
import torch
from model import HandwritingGenerator
import utils
import matplotlib.pyplot as plt
from PIL import Image

def generate_samples(generator, texts, char_to_idx, mean, std, args):
    """
    Generate handwritten samples for the given texts
    
    Args:
        generator: The handwriting generator model
        texts: List of texts to generate
        char_to_idx: Character to index mapping
        mean: Mean values for denormalization
        std: Standard deviation values for denormalization
        args: Generation arguments
        
    Returns:
        List of generated stroke data
    """
    generated_samples = []
    
    for i, text in enumerate(texts):
        print(f"Generating sample {i+1}/{len(texts)}: '{text}'")
        
        # Generate stroke data
        stroke = generator.generate(
            text=text,
            char_to_idx=char_to_idx,
            mean=mean,
            std=std,
            max_length=args.max_length
        )
        
        generated_samples.append(stroke)
        
        # Visualize and save result
        if args.visualize:
            fig = utils.visualize_stroke(stroke, text)
            
            # Save visualization if output directory is specified
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f"sample_{i+1}_{text[:10]}.png")
                fig.savefig(output_path, bbox_inches='tight')
                print(f"Saved visualization to {output_path}")
            
            plt.close(fig)
    
    return generated_samples

def main():
    parser = argparse.ArgumentParser(description='Generate handwritten text samples')
    
    # Input parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Directory containing dataset (for alphabet and normalization)')
    parser.add_argument('--texts', type=str, nargs='+',
                        help='Texts to generate')
    parser.add_argument('--text_file', type=str,
                        help='File containing texts to generate (one per line)')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Maximum length of generated sequence')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate (if not specifying texts)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated samples')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save generated samples')
    parser.add_argument('--save_strokes', action='store_true',
                        help='Save raw stroke data as numpy arrays')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if device is available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load alphabet and normalization parameters
    alphabet = np.load(os.path.join(args.data_dir, 'training', 'alphabet.npy'), allow_pickle=True)
    mean = np.load(os.path.join(args.data_dir, 'training', 'mean.npy'), allow_pickle=True)
    std = np.load(os.path.join(args.data_dir, 'training', 'std.npy'), allow_pickle=True)
    
    # Create character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    # Determine texts to generate
    texts = []
    if args.texts:
        texts.extend(args.texts)
    elif args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    else:
        # Generate random texts as a fallback
        sample_texts = [
            "Hello world",
            "Handwritten text generation",
            "Deep learning is amazing",
            "This is a sample of generated text",
            "Using neural networks for writing"
        ]
        texts.extend(sample_texts[:args.num_samples])
    
    # Initialize generator
    generator = HandwritingGenerator(
        model_path=args.model_path,
        vocab_size=len(alphabet),
        device=device
    )
    
    # Generate samples
    generated_samples = generate_samples(
        generator=generator,
        texts=texts,
        char_to_idx=char_to_idx,
        mean=mean,
        std=std,
        args=args
    )
    
    # Save raw stroke data if requested
    if args.save_strokes and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for i, (stroke, text) in enumerate(zip(generated_samples, texts)):
            output_path = os.path.join(args.output_dir, f"stroke_{i+1}_{text[:10]}.npy")
            np.save(output_path, stroke)
            print(f"Saved stroke data to {output_path}")

if __name__ == '__main__':
    main()
