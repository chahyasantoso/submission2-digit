"""
Transform module
"""

import tensorflow as tf
from modules.utils import (
    transformed_name,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_KEY,
    FEATURE_KEY
)


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """
    outputs = {}

    # tf.io.decode_png function cannot be applied on a batch of data.
    # We have to use tf.map_fn
    image_features = tf.map_fn(
        lambda x: tf.io.decode_png(x[0], channels=3),
        inputs[FEATURE_KEY],
        fn_output_signature=tf.uint8)
    image_features = tf.image.resize(image_features, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image_features = tf.keras.applications.inception_v3.preprocess_input(
        image_features)

    outputs[transformed_name(FEATURE_KEY)] = image_features
    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

    return outputs
