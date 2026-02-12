import os
import sys
import tensorflow as tf
from configs import config
from src.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(self, model_path=config.MODEL_PATH):
        """Initialize the predictor by loading the trained model."""
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}. Please train the model first.")
            raise FileNotFoundError(f"Model file {model_path} missing.")

        logger.info(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")

    def preprocess_image(self, image_path):
        """Load and scale a single image to match model input requirements."""
        try:
            # 1. Load image
            img = tf.keras.utils.load_img(image_path, target_size=config.IMAGE_SIZE)

            # 2. Convert to array and scale (0 to 1)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = img_array / 255.0  # Same scaling as training

            # 3. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            img_array = tf.expand_dims(img_array, 0)

            return img_array
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def predict(self, image_path):
        """Run inference on a single image."""
        processed_img = self.preprocess_image(image_path)

        if processed_img is None:
            return None

        # Run prediction
        prediction = self.model.predict(processed_img, verbose=0)[0][0]

        # Classification logic (Sigmoid output: 0 to 1)
        # Assuming Cat = 0 and Dog = 1 based on default directory sorting
        label = "DOG" if prediction > 0.5 else "CAT"
        confidence = prediction if label == "DOG" else (1 - prediction)

        return label, confidence


def main():
    """Command-line interface for the predictor."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        predictor = Predictor()
        result = predictor.predict(image_path)

        if result:
            label, confidence = result
            print("-" * 30)
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 30)

    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()