'''
Utilily module
'''

import os
import base64
from pprint import PrettyPrinter

import tensorflow as tf
import kerastuner as kt
from keras import layers

LABEL_KEY = 'sentiment'
FEATURE_KEY = 'review'

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

# layer yang dipakai untuk vectorization (standardize, tokenize, vectorize)
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250
vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

def vectorize_dataset(train_set):
    '''Vectorize train dataset'''
    train_text = [j.numpy()[0] for i in list(train_set)
                for j in i[0][transformed_name(FEATURE_KEY)]]
    vectorize_layer.adapt(train_text)

def get_hyperparameters() -> kt.HyperParameters:
    '''Returns HyperParameters for keras model'''

    hp = kt.HyperParameters()
    hp.Int(
        'hidden_layers',
        min_value=1,
        max_value=3,
        default=2)
    for i in range(3):
        hp.Int(
            'unit' + str(i),
            min_value=64,
            max_value=256,
            step=64)
    hp.Choice(
        'learning_rate',
        [1e-2, 1e-3],
        default=1e-2)
    return hp


def model_builder(hp: kt.HyperParameters, show_summary=True):
    '''
    Defines a Keras model and returns the model as a Keras object.
    Args:
        hp: HyperParameter from kerastuner

    Returns:
        Keras object
    '''

    embedding_dim = 16
    num_hidden_layers = int(hp.get('hidden_layers'))
    learning_rate = hp.get('learning_rate')

    inputs = tf.keras.Input(shape=(1,),
                            name=transformed_name(FEATURE_KEY),
                            dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name='embedding')(x)
    x = layers.GlobalAveragePooling1D()(x)
    for i in range(num_hidden_layers):
        num_nodes = int(hp.get('unit' + str(i)))
        x = layers.Dense(num_nodes, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    if show_summary:
        model.summary()

    return model


def print_example_from_uri(uri, num_records):
    '''print tf Example di lokasi uri'''

    # Get the list of files in this directory (all compressed TFRecord files)
    tfrecord_filenames = [os.path.join(uri, name) for name in os.listdir(uri)]
    # Create TFRecordDataset to read these files
    dataset = tf.data.TFRecordDataset(
        tfrecord_filenames, compression_type="GZIP")
    # Iterate over the first num_records and decode them.
    for tfrecord in dataset.take(num_records):
        serialized_example = tfrecord.numpy()
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        pp = PrettyPrinter()
        pp.pprint(example)


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


def serialize_example(feature_dict):
    '''
    membuat serialized tf.Example yang berasal dari feature

    inputan berupa feature yang merupakan dictionary dengan value type text
    contoh: feature_dict = {'sentence':'ini review movie'}
    '''
    for key in feature_dict:
        feature_dict[key] = _bytes_feature(feature_dict[key].encode())

    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature_dict))
    return example.SerializeToString()


def make_base64_examples(serialized_tf_example):
    '''
    konversi serialized tf.Example ke base64 encode
    '''
    return {
        'examples': {
            'b64': base64.b64encode(serialized_tf_example).decode()
        }
    }
