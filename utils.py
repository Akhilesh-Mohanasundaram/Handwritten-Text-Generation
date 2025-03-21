import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
from PIL import Image, ImageDraw

def plot_stroke(ax, stroke, text=None, title=None):
    """
    Plot a single handwriting stroke
    
    Args:
        ax: Matplotlib axis
        stroke: Stroke data with shape [seq_len, 3]
        text: Optional text label
        title: Optional plot title
    """
    # Extract pen states and coordinates
    pen_up = stroke[:, 0]
    x_coords = stroke[:, 1]
    y_coords = stroke[:, 2]
    
    # Initialize starting position
    pos_x, pos_y = 0, 0
    prev_pen_up = 0
    
    # Lists to store line segments
    line_x = []
    line_y = []
    lines = []
    
    # Process stroke points
    for i in range(len(stroke)):
        if prev_pen_up == 0 and pen_up[i] == 0:
            # Continue current line
            pos_x += x_coords[i]
            pos_y += y_coords[i]
            line_x.append(pos_x)
            line_y.append(pos_y)
        elif prev_pen_up == 0 and pen_up[i] == 1:
            # End current line
            pos_x += x_coords[i]
            pos_y += y_coords[i]
            line_x.append(pos_x)
            line_y.append(pos_y)
            lines.append((line_x, line_y))
            line_x = []
            line_y = []
        elif prev_pen_up == 1 and pen_up[i] == 0:
            # Start new line
            pos_x += x_coords[i]
            pos_y += y_coords[i]
            line_x.append(pos_x)
            line_y.append(pos_y)
        
        prev_pen_up = pen_up[i]
    
    # Add any remaining line
    if line_x:
        lines.append((line_x, line_y))
    
    # Plot lines
    for line_x, line_y in lines:
        ax.plot(line_x, line_y, 'k-')
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Add text label if provided
    if text:
        ax.text(0.5, -0.1, f"Text: {text}", transform=ax.transAxes, 
                horizontalalignment='center')

def visualize_stroke(stroke, text=None):
    """
    Visualize a handwriting stroke
    
    Args:
        stroke: Stroke data with shape [seq_len, 3]
        text: Optional text label
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_stroke(ax, stroke, text)
    plt.tight_layout()
    return fig

def stroke_to_image(stroke, image_size=(256, 64), line_width=2):
    """
    Convert stroke data to an image
    
    Args:
        stroke: Stroke data with shape [seq_len, 3]
        image_size: Size of the output image (width, height)
        line_width: Width of the stroke lines
        
    Returns:
        PIL Image
    """
    width, height = image_size
    
    # Create blank image
    img = Image.new('L', image_size, color=255)
    draw = ImageDraw.Draw(img)
    
    # Extract pen states and coordinates
    pen_up = stroke[:, 0]
    x_coords = stroke[:, 1]
    y_coords = stroke[:, 2]
    
    # Scale coordinates to fit in image
    x_min, x_max = np.min(np.cumsum(x_coords)), np.max(np.cumsum(x_coords))
    y_min, y_max = np.min(np.cumsum(y_coords)), np.max(np.cumsum(y_coords))
    
    # Add margin
    margin = 10
    x_scale = (width - 2 * margin) / max(1, x_max - x_min)
    y_scale = (height - 2 * margin) / max(1, y_max - y_min)
    scale = min(x_scale, y_scale)
    
    # Initialize starting position
    pos_x, pos_y = 0, 0
    prev_pen_up = 0
    
    # Scale to fit in image and center
    x_offset = margin - x_min * scale + (width - 2 * margin - (x_max - x_min) * scale) / 2
    y_offset = margin - y_min * scale + (height - 2 * margin - (y_max - y_min) * scale) / 2
    
    # Previous point
    prev_x, prev_y = None, None
    
    # Process stroke points
    for i in range(len(stroke)):
        # Update position
        pos_x += x_coords[i]
        pos_y += y_coords[i]
        
        # Scale and translate
        img_x = pos_x * scale + x_offset
        img_y = pos_y * scale + y_offset
        
        if prev_x is not None and prev_pen_up == 0 and pen_up[i] == 0:
            # Draw line
            draw.line([(prev_x, prev_y), (img_x, img_y)], fill=0, width=line_width)
        
        # Update previous point
        prev_x, prev_y = img_x, img_y
        prev_pen_up = pen_up[i]
    
    return img

def image_to_stroke(image, threshold=200):
    """
    Convert an image to stroke data (for demonstration purposes)
    Note: This is a simplistic approach and does not create accurate strokes
    
    Args:
        image: PIL Image or numpy array
        threshold: Threshold for binarization
        
    Returns:
        Stroke data with shape [seq_len, 3]
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Binarize the image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to strokes
    strokes = []
    for contour in contours:
        # Skip small contours
        if len(contour) < 5:
            continue
        
        # First point with pen down
        point = contour[0][0]
        strokes.append([0, point[0], point[1]])
        
        # Middle points
        for i in range(1, len(contour)):
            point = contour[i][0]
            prev_point = contour[i-1][0]
            dx = point[0] - prev_point[0]
            dy = point[1] - prev_point[1]
            strokes.append([0, dx, dy])
        
        # Last point with pen up
        strokes.append([1, 0, 0])
    
    # If no strokes, return an empty array with the right shape
    if not strokes:
        return np.zeros((0, 3))
    
    return np.array(strokes)

def normalize_stroke(stroke, mean=None, std=None):
    """
    Normalize stroke data
    
    Args:
        stroke: Stroke data with shape [seq_len, 3]
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized stroke data
    """
    if mean is None or std is None:
        # Calculate mean and std from the data
        # Don't include pen up/down in normalization
        coords = stroke[:, 1:]
        mean = np.mean(coords, axis=0)
        std = np.std(coords, axis=0)
        std[std == 0] = 1  # Avoid division by zero
    
    # Clone stroke data
    normalized = stroke.copy()
    
    # Normalize only coordinates (not pen state)
    normalized[:, 1:] = (stroke[:, 1:] - mean) / std
    
    return normalized, mean, std

def denormalize_stroke(stroke, mean, std):
    """
    Denormalize stroke data
    
    Args:
        stroke: Normalized stroke data with shape [seq_len, 3]
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized stroke data
    """
    # Clone stroke data
    denormalized = stroke.copy()
    
    # Denormalize only coordinates (not pen state)
    denormalized[:, 1:] = stroke[:, 1:] * std + mean
    
    return denormalized

def prepare_batch(batch, device):
    """
    Prepare batch data for training/evaluation
    
    Args:
        batch: Batch data from data loader
        device: Device to transfer data to
        
    Returns:
        Processed batch data
    """
    return {
        'stroke': batch['stroke'].to(device),
        'text': batch['text'].to(device),
        'stroke_lengths': batch['stroke_lengths'].to(device),
        'text_lengths': batch['text_lengths'].to(device)
    }
