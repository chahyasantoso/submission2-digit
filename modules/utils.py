'''
Utilily module
'''

import os
import base64
from pprint import PrettyPrinter
import tensorflow as tf
import kerastuner as kt
from keras import layers


TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
LSTM_SIZE = 64
LEARNING_RATE = 1e-4

FEATURE_KEY = 'review'
LABEL_KEY = 'sentiment'


def transformed_name(key):
    '''Renaming transformed features'''
    return key + '_xf'


def gzip_reader_fn(filenames):
    '''Loads compressed data'''
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64) -> tf.data.Dataset:
    '''Generates features and labels for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    '''
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset


# layer untuk vectorization (standardize, tokenize, vectorize)
vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)


def get_hyperparameters() -> kt.HyperParameters:
    '''Returns HyperParameters for keras model'''
    hp = kt.HyperParameters()
    hp.Int(
        'hidden_size',
        min_value=256,
        max_value=512,
        step=128,
        default=256,
    )
    hp.Float(
        'dropout_rate',
        min_value=0.2,
        max_value=0.5,
        step=0.1,
        default=0.2,
    )
    return hp


def model_builder(hp: kt.HyperParameters, show_summary=True):
    '''
    Defines a Keras model and returns the model as a Keras object.
    Args:
        hp: HyperParameter from kerastuner

    Returns:
        Keras object
    '''
    inputs = tf.keras.Input(
        shape=(1,),
        name=transformed_name(FEATURE_KEY),
        dtype=tf.string)

    x = vectorize_layer(tf.reshape(inputs, [-1]))
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding')(x)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_SIZE, dropout=float(hp.get('dropout_rate'))))(x)
    x = layers.Dense(int(hp.get('hidden_size')), activation='relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    if show_summary:
        model.summary()

    return model


def get_example_from_uri(uri, count=-1):
    '''Returns list of tf Example dari lokasi uri'''
    tfrecord_filenames = [os.path.join(uri, name) for name in os.listdir(uri)]
    dataset = tf.data.TFRecordDataset(
        tfrecord_filenames, compression_type="GZIP")
    return [tfrecord.numpy() for tfrecord in dataset.take(count)]


def print_example_from_uri(uri, count=5):
    '''print tf Example di lokasi uri'''
    examples = get_example_from_uri(uri, count)
    for serialized_example in examples:
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        pp = PrettyPrinter()
        pp.pprint(example)


def decode_example(serialized_example, feature_dict):
    '''Return a dict of Tensors from a serialized tf Example.'''
    decoded = tf.io.parse_example(
        serialized_example,
        features=feature_dict
    )
    return [[value.numpy() for value in decoded[key].values]
            for key in decoded]


def _bytes_feature(value):
    '''Returns a bytes_list from a string / byte.'''
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    '''Returns a float_list from a float / double.'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_serialize_example(data_dict):
    '''
    membuat serialized tf.Example

    inputan berupa data yang merupakan dictionary
    contoh: data_dict = {'review':'ini review movie','sentiment':'positive'}
    '''
    feature = {}
    for key in data_dict:
        dtype = type(data_dict[key])
        if dtype == str:
            feature[key] = _bytes_feature(data_dict[key].encode())
        elif dtype == int:
            feature[key] = _int64_feature(data_dict[key])
        elif dtype == float:
            feature[key] = _float_feature(data_dict[key])

    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature))

    return example.SerializeToString()


def make_base64_example(serialized_example):
    '''
    konversi serialized tf.Example ke base64 encode
    '''
    return {
        'examples': {
            'b64': base64.b64encode(serialized_example).decode()
        }
    }
