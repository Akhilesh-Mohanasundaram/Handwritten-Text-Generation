import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import HandwritingGenerationModel
from data_loader import get_data_loader
import utils

def train(model, train_loader, val_loader, args):
    """
    Train the handwriting generation model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        args: Training arguments
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Set up TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # Get data
            stroke = batch['stroke'].to(device)
            text = batch['text'].to(device)
            stroke_lengths = batch['stroke_lengths'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(stroke, text, stroke_lengths, text_lengths, 
                           teacher_forcing_ratio=args.teacher_forcing_ratio)
            
            # Compute loss
            loss = criterion(outputs, stroke)
            
            # Backward pass and optimize
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            
            # Print progress
            if (i + 1) % args.print_every == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s')
                start_time = time.time()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                stroke = batch['stroke'].to(device)
                text = batch['text'].to(device)
                stroke_lengths = batch['stroke_lengths'].to(device)
                text_lengths = batch['text_lengths'].to(device)
                
                # Forward pass
                outputs = model(stroke, text, stroke_lengths, text_lengths, 
                               teacher_forcing_ratio=0.0)  # No teacher forcing during validation
                
                # Compute loss
                loss = criterion(outputs, stroke)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print validation results
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')
        
        # Generate and visualize sample
        if (epoch + 1) % args.sample_every == 0:
            sample_batch = next(iter(val_loader))
            sample_text = sample_batch['raw_text'][0]
            sample_stroke = sample_batch['stroke'][0].unsqueeze(0).to(device)
            sample_text_tensor = sample_batch['text'][0].unsqueeze(0).to(device)
            sample_stroke_length = sample_batch['stroke_lengths'][0].unsqueeze(0).to(device)
            sample_text_length = sample_batch['text_lengths'][0].unsqueeze(0).to(device)
            
            with torch.no_grad():
                generated_stroke = model(
                    sample_stroke[:, 0:1, :],
                    sample_text_tensor,
                    torch.tensor([1], device=device),
                    sample_text_length,
                    teacher_forcing_ratio=0.0
                )
            
            # Visualize result
            fig = utils.visualize_stroke(generated_stroke[0].cpu().numpy(), sample_text)
            writer.add_figure(f'Sample/{sample_text}', fig, epoch)
        
        # Save model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f'Model saved at epoch {epoch+1} with validation loss {val_loss:.4f}')
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Close TensorBoard writer
    writer.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'final_model.pth'))
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train handwriting generation model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Directory containing dataset')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Size of hidden layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of character embeddings')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.8,
                        help='Probability of using teacher forcing')
    parser.add_argument('--clip_grad', type=float, default=5.0,
                        help='Gradient clipping value')
    
    # Output parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--print_every', type=int, default=100,
                        help='Print loss every N steps')
    parser.add_argument('--sample_every', type=int, default=5,
                        help='Generate sample every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load dataset
    train_loader = get_data_loader(
        args.data_dir, 
        split='training', 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = get_data_loader(
        args.data_dir, 
        split='validation', 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Get vocabulary size from dataset
    vocab_size = len(train_loader.dataset.alphabet)
    
    # Create model
    model = HandwritingGenerationModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    )
    
    # Train model
    train(model, train_loader, val_loader, args)

if __name__ == '__main__':
    main()
