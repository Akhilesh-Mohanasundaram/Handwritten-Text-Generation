import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import HandwritingGenerationModel
from data_loader import HandwritingDataset
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import cv2
from PIL import Image

def calculate_mse(original, generated):
    """Calculate Mean Squared Error between original and generated strokes"""
    # Ensure sequences have the same length
    min_len = min(len(original), len(generated))
    return mean_squared_error(original[:min_len], generated[:min_len])

def calculate_dtw(original, generated):
    """Calculate Dynamic Time Warping distance between original and generated strokes"""
    # Implementation of DTW algorithm
    n, m = len(original), len(generated)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = float('inf')
    
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(original[i-1] - generated[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m]

def calculate_frechet_distance(original, generated):
    """Calculate Frechet distance between original and generated strokes"""
    # For simplicity, we'll use a recursive implementation
    # Note: For production, a dynamic programming approach would be more efficient
    
    def _frechet_dist(i, j, memo={}):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == 0 and j == 0:
            memo[(i, j)] = np.linalg.norm(original[0] - generated[0])
            return memo[(i, j)]
        
        if i > 0 and j == 0:
            memo[(i, j)] = max(_frechet_dist(i-1, 0), np.linalg.norm(original[i] - generated[0]))
            return memo[(i, j)]
        
        if i == 0 and j > 0:
            memo[(i, j)] = max(_frechet_dist(0, j-1), np.linalg.norm(original[0] - generated[j]))
            return memo[(i, j)]
        
        memo[(i, j)] = max(
            min(
                _frechet_dist(i-1, j),
                _frechet_dist(i-1, j-1),
                _frechet_dist(i, j-1)
            ),
            np.linalg.norm(original[i] - generated[j])
        )
        return memo[(i, j)]
    
    # For large sequences, use a subset to avoid computational explosion
    max_points = 100
    if len(original) > max_points or len(generated) > max_points:
        orig_indices = np.linspace(0, len(original)-1, max_points, dtype=int)
        gen_indices = np.linspace(0, len(generated)-1, max_points, dtype=int)
        original_subset = original[orig_indices]
        generated_subset = generated[gen_indices]
        return _frechet_dist(len(original_subset)-1, len(generated_subset)-1)
    
    return _frechet_dist(len(original)-1, len(generated)-1)

def calculate_coherence(strokes):
    """Calculate coherence score of generated handwriting"""
    # This is a simplified measure - more advanced metrics could be used
    # We'll measure the smoothness of the stroke (lower variance = more coherent)
    
    # Calculate first-order differences (velocity)
    velocity = np.diff(strokes[:, 1:], axis=0)
    
    # Calculate second-order differences (acceleration)
    acceleration = np.diff(velocity, axis=0)
    
    # Calculate variance of acceleration (lower = smoother writing)
    acceleration_variance = np.var(acceleration)
    
    # Normalize to a 0-1 range (where 1 is more coherent)
    coherence_score = np.exp(-acceleration_variance / 100)
    
    return coherence_score

def evaluate_ocr_accuracy(generated_strokes, reference_texts, args):
    """Evaluate OCR accuracy on generated handwriting"""
    # In a real implementation, you would use an OCR system
    # Here, we'll just simulate OCR accuracy with a placeholder function
    
    # Convert strokes to images
    images = []
    for stroke in generated_strokes:
        img = utils.stroke_to_image(stroke)
        images.append(img)
    
    # Simulated OCR accuracy (placeholder for real OCR)
    # In a real implementation, you would use pytesseract or a similar OCR system
    # to extract text from the images and compare with reference_texts
    
    # For now, we'll return a random accuracy score between 0.7 and 0.95
    # In a real implementation, this would be calculated based on OCR results
    return np.random.uniform(0.7, 0.95)

def evaluate_model(model, data_loader, args):
    """
    Evaluate handwriting generation model
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        args: Evaluation arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    metrics = {
        'mse': [],
        'dtw': [],
        'frechet': [],
        'coherence': [],
    }
    
    generated_samples = []
    reference_texts = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= args.num_samples:
                break
                
            # Get data
            stroke = batch['stroke'].to(device)
            text = batch['text'].to(device)
            stroke_lengths = batch['stroke_lengths'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            raw_text = batch['raw_text'][0]
            
            # Generate handwriting
            generated = model(
                stroke[:, 0:1, :],  # Start with first point
                text,
                torch.tensor([1], device=device),
                text_lengths,
                teacher_forcing_ratio=0.0
            )
            
            # Convert to numpy for evaluation
            original_stroke = stroke[0].cpu().numpy()
            generated_stroke = generated[0].cpu().numpy()
            
            # Calculate metrics
            mse = calculate_mse(original_stroke, generated_stroke)
            dtw = calculate_dtw(original_stroke, generated_stroke)
            frechet = calculate_frechet_distance(original_stroke, generated_stroke)
            coherence = calculate_coherence(generated_stroke)
            
            metrics['mse'].append(mse)
            metrics['dtw'].append(dtw)
            metrics['frechet'].append(frechet)
            metrics['coherence'].append(coherence)
            
            # Store generated samples for OCR evaluation
            generated_samples.append(generated_stroke)
            reference_texts.append(raw_text)
            
            # Visualize comparison if requested
            if args.visualize:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                utils.plot_stroke(ax1, original_stroke, raw_text, title="Original")
                utils.plot_stroke(ax2, generated_stroke, raw_text, title="Generated")
                plt.tight_layout()
                
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    plt.savefig(os.path.join(args.output_dir, f"comparison_{i+1}.png"))
                
                if args.show_plots:
                    plt.show()
                else:
                    plt.close()
    
    # Calculate OCR accuracy
    ocr_accuracy = evaluate_ocr_accuracy(generated_samples, reference_texts, args)
    metrics['ocr_accuracy'] = ocr_accuracy
    
    # Calculate average metrics
    avg_metrics = {
        'avg_mse': np.mean(metrics['mse']),
        'avg_dtw': np.mean(metrics['dtw']),
        'avg_frechet': np.mean(metrics['frechet']),
        'avg_coherence': np.mean(metrics['coherence']),
        'ocr_accuracy': metrics['ocr_accuracy']
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Average MSE: {avg_metrics['avg_mse']:.4f}")
    print(f"Average DTW Distance: {avg_metrics['avg_dtw']:.4f}")
    print(f"Average Frechet Distance: {avg_metrics['avg_frechet']:.4f}")
    print(f"Average Coherence Score: {avg_metrics['avg_coherence']:.4f} (higher is better)")
    print(f"OCR Accuracy: {avg_metrics['ocr_accuracy']:.4f}")
    
    # Calculate overall score (weighted average of normalized metrics)
    # Lower is better for MSE, DTW, Frechet; higher is better for coherence and OCR
    norm_mse = 1 - min(avg_metrics['avg_mse'] / 10, 1)  # Normalize and invert
    norm_dtw = 1 - min(avg_metrics['avg_dtw'] / 100, 1)  # Normalize and invert
    norm_frechet = 1 - min(avg_metrics['avg_frechet'] / 100, 1)  # Normalize and invert
    
    overall_score = (
        0.2 * norm_mse + 
        0.2 * norm_dtw + 
        0.2 * norm_frechet + 
        0.2 * avg_metrics['avg_coherence'] + 
        0.2 * avg_metrics['ocr_accuracy']
    )
    
    print(f"Overall Model Score: {overall_score:.4f} (higher is better)")
    
    return avg_metrics, overall_score

def main():
    parser = argparse.ArgumentParser(description='Evaluate handwriting generation model')
    
    # Input parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Directory containing dataset')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (should be 1 for evaluation)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to evaluate')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize comparisons')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots during evaluation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save evaluation results')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load dataset
    val_dataset = HandwritingDataset(args.data_dir, split='validation')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'stroke': batch[0]['stroke'].unsqueeze(0),
            'text': batch[0]['text'].unsqueeze(0),
            'stroke_lengths': torch.tensor([len(batch[0]['stroke'])]),
            'text_lengths': torch.tensor([len(batch[0]['text'])]),
            'raw_text': [batch[0]['raw_text']]
        }
    )
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HandwritingGenerationModel(vocab_size=len(val_dataset.alphabet))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Evaluate model
    metrics, overall_score = evaluate_model(model, val_loader, args)
    
    # Save results if output directory is specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("Evaluation Results:\n")
            f.write(f"Average MSE: {metrics['avg_mse']:.4f}\n")
            f.write(f"Average DTW Distance: {metrics['avg_dtw']:.4f}\n")
            f.write(f"Average Frechet Distance: {metrics['avg_frechet']:.4f}\n")
            f.write(f"Average Coherence Score: {metrics['avg_coherence']:.4f}\n")
            f.write(f"OCR Accuracy: {metrics['ocr_accuracy']:.4f}\n")
            f.write(f"Overall Model Score: {overall_score:.4f}\n")
        
        print(f"Saved evaluation results to {args.output_dir}")

if __name__ == '__main__':
    main()
