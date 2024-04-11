'''Transform module
'''

import tensorflow as tf
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

LABEL_KEY = 'sentiment'
FEATURE_KEY = 'review'

STOPWORDS = set(stopwords.words('english'))
    
def transformed_name(key):
    '''Renaming transformed features'''
    return key + "_xf"
    
    
def preprocessing_fn(inputs):
    '''
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    '''
    
    outputs = {}
    
    '''for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )
    
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])
    
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)'''
    # lowercase inputs
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY]) 

    # remove <br />
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], 
        r'(?:<br />)',' ')

    # remove non alphanum characters
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], 
        r'\W+',' ')
    
    # remove stopwords
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(
        outputs[transformed_name(FEATURE_KEY)], 
        r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*','')

    # change label negative, positive to numeric 0/1
    # bikin lookup table dulu
    table_keys = ['negative', 'positive']
    table_vals = [0, 1]
    with tf.init_scope():
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=table_keys,
            values=table_vals,
            key_dtype=tf.string,
            value_dtype=tf.int64)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    
    outputs[transformed_name(LABEL_KEY)] = table.lookup(inputs[LABEL_KEY])
    
    return outputs