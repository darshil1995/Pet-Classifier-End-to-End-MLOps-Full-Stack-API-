import os
import pytest
import tensorflow as tf
from configs import config

"""
These tests verify that our organize_data.py logic worked and that 
the TensorFlow dataset is yielding the correct shapes.
"""

def test_folder_structure():
    """Check if organize_data.py created the expected subfolders."""
    cat_path = os.path.join(config.EXTRACTED_PATH, 'cat')
    dog_path = os.path.join(config.EXTRACTED_PATH, 'dog')
    assert os.path.isdir(cat_path), "Cat subfolder missing!"
    assert os.path.isdir(dog_path), "Dog subfolder missing!"

def test_image_count():
    """Verify images exist in the folders (Assuming at least some were moved)."""
    cat_images = os.listdir(os.path.join(config.EXTRACTED_PATH, 'cat'))
    dog_images = os.listdir(os.path.join(config.EXTRACTED_PATH, 'dog'))
    assert len(cat_images) > 0, "No cat images found!"
    assert len(dog_images) > 0, "No dog images found!"

def test_input_shape():
    """Verify the tf.data pipeline outputs correct tensor shapes."""
    # We load a tiny batch to test shape
    dataset = tf.keras.utils.image_dataset_from_directory(
        config.EXTRACTED_PATH,
        image_size=config.IMAGE_SIZE,
        batch_size=2
    )
    for images, labels in dataset.take(1):
        expected_shape = (2, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
        assert images.shape == expected_shape, f"Expected {expected_shape}, got {images.shape}"