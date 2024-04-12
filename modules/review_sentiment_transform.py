'''
Transform module
'''

import nltk
from nltk.corpus import stopwords

import tensorflow as tf
import tensorflow_transform as tft
from utils import (
    transformed_name,
    LABEL_KEY,
    FEATURE_KEY
)

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocessing_fn(inputs):
    '''
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    '''

    outputs = {}

    # lowercase inputs
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])

    # remove <br />
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r'(?:<br />)', ' ')

    # remove non alphanum characters
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r'\W+', ' ')

    # remove stopwords
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)],
        r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*', '')

    # change label negative, positive to numeric 0/1
    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY], top_k=2
    )

    return outputs
