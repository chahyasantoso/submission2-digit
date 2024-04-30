"""
Transform module
"""

import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from utils import (
    transformed_name,
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
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    # lowercase inputs
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])

    # remove <br />
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r"(?:<br />)", " ")

    # remove non alphanum characters
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r"\W+", " ")

    # remove stopwords
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r"\b(" + r"|".join(stop_words) + r")\b\s*", "")

    # change label negative, positive to numeric 0/1
    @tf.function
    def num_label(label):
        return tf.constant(
            0,
            dtype=tf.int64) if (
            label == 'negative') else tf.constant(
            1,
            dtype=tf.int64)

    outputs[transformed_name(LABEL_KEY)] = tf.map_fn(
        fn=num_label,
        elems=inputs[LABEL_KEY],
        fn_output_signature=tf.int64)

    return outputs
