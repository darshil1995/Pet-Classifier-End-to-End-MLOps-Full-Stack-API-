import os

# Store all constants, paths, and hyperparameters here to avoid "magic numbers" in your logic.
# Paths
DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw')
EXTRACTED_PATH = os.path.join(DATA_DIR, 'train')

# Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE= 3

# Path for the saved model
MODEL_SAVE_DIR = 'models'
MODEL_NAME = 'cat_dog_classifier.keras'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#MODEL_PATH = os.path.join(BASE_DIR, "models", "cat_dog_classifier.keras")

