import os
import tensorflow as tf
from configs import config
from src.model_builder import build_mobilenet_model
from src.logger import get_logger
from utils.timers import TimeHistory

logger = get_logger(__name__)


def train():
    logger.info("Starting data pipeline initialization...")

    # 1. Load datasets from directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.EXTRACTED_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.EXTRACTED_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='binary'
    )

    # 2. Optimize for Performance
    AUTOTUNE = tf.data.AUTOTUNE

    # shuffle(1000) ensures the model doesn't see images in the same order every time
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Scaling Layer
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 4. Build Model
    model = build_mobilenet_model()

    # --- CALLBACKS SECTION ---

    # Early Stopping: Stops training if validation loss doesn't improve
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True
    )

    # Learning Rate Reducer: Slows down learning for "fine-tuning"
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Custom Timer: Logs the timing of each epoch
    time_callback = TimeHistory()

    # 5. Model Fitting
    logger.info("Beginning model training with optimized callbacks...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=[early_stopping, lr_reducer, time_callback]
    )

    # 6. Save Our Model
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)

    model.save(config.MODEL_PATH)
    logger.info(f"Model saved successfully at: {config.MODEL_PATH}")

    return history


if __name__ == "__main__":
    train()