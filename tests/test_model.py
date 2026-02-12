import numpy as np
import tensorflow as tf
import os
import pytest
from src.model_builder import build_mobilenet_model

"""This tests the model_builder.py which is brain of the project"""

def test_output_range():
    """Verify sigmoid output is between 0 and 1."""
    model = build_mobilenet_model()
    # Create a random image tensor
    random_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    prediction = model.predict(random_input)
    assert 0 <= prediction[0][0] <= 1, "Prediction outside 0-1 range!"


def test_model_loading():
    """Ensure a saved model can be reloaded (if it exists)."""
    from configs import config
    if os.path.exists(config.MODEL_PATH):
        model = tf.keras.models.load_model(config.MODEL_PATH)
        assert isinstance(model, tf.keras.Model)
    else:
        pytest.skip("Model file not found; skipping loading test.")


def test_overfit_small_batch():
    """Test if model can learn 2 images perfectly (Sanity Check)."""
    model = build_mobilenet_model()
    # Create 2 fake images (1 cat, 1 dog)
    x = np.random.rand(2, 224, 224, 3).astype(np.float32)
    y = np.array([[0.0], [1.0]])  # Labels

    # Train for 20 epochs on just these 2 images
    history = model.fit(x, y, epochs=20, verbose=0)
    final_acc = history.history['accuracy'][-1]

    # If the model is built correctly, it should easily memorize 2 images
    assert final_acc > 0.8, f"Model failed to overfit small batch. Acc: {final_acc}"