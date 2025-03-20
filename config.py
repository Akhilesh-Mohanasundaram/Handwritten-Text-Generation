import os

# Dataset paths
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "training")
VALIDATION_PATH = os.path.join(DATASET_PATH, "validation")

# Model parameters
HIDDEN_UNITS = 512
NUM_LAYERS = 3
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
SEQUENCE_LENGTH = 400
GRADIENT_CLIP = 5.0

# Training parameters
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
SAMPLE_EVERY = 5  # Generate samples every N epochs
SAVE_EVERY = 5  # Save model every N epochs

# Inference parameters
TEMPERATURE = 0.8  # Controls randomness in generation
MAX_GEN_LENGTH = 800  # Maximum length of generated sequence
