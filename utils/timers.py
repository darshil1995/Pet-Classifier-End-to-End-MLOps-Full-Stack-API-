import time
from datetime import timedelta
import tensorflow as tf
from src.logger import get_logger

logger = get_logger(__name__)


class TimeHistory(tf.keras.callbacks.Callback):
    """Custom Keras callback to log the duration of each epoch and total training."""

    def on_train_begin(self, logs=None):
        self.total_start_time = time.time()
        logger.info(">>> Training started. Clock is running...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        logger.info(f"Epoch {epoch + 1} finished in {epoch_duration:.2f} seconds.")

    def on_train_end(self, logs=None):
        total_seconds = time.time() - self.total_start_time
        formatted_time = str(timedelta(seconds=int(total_seconds)))
        logger.info(f">>> Training Complete!")
        logger.info(f">>> Total Execution Time: {formatted_time} (HH:MM:SS)")