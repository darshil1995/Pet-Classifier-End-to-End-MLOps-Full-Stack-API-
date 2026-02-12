import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from configs import config


def build_mobilenet_model():
    """Builds a MobileNetV2 base model with custom classification layers."""
    # Pre-trained base
    base_model = MobileNetV2(
        input_shape=(*config.IMAGE_SIZE, config.CHANNELS),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model to preserve pre-trained weights
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model