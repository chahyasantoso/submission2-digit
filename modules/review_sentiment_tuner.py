'''Tuner module
'''

import tensorflow as tf
import tensorflow_transform as tft
import kerastuner as kt
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner.engine import base_tuner
from typing import (
    NamedTuple, 
    Dict, 
    Text, 
    Any
)

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
            batch_size=64)->tf.data.Dataset:
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
        label_key = transformed_name(LABEL_KEY))
    return dataset

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250

# layer yang dipakai untuk vectorization (standardize, tokenize, vectorize)
vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

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
            'unit'+str(i), 
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

    EMBEDDING_DIM=16
    NUM_HIDDEN_LAYERS = int(hp.get('hidden_layers'))
    HP_LEARNING_RATE = hp.get('learning_rate')

    inputs = tf.keras.Input(shape=(1,), 
                            name=transformed_name(FEATURE_KEY), 
                            dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding')(x)
    x = layers.GlobalAveragePooling1D()(x)
    for i in range(NUM_HIDDEN_LAYERS):
        num_nodes = int(hp.get('unit' + str(i)))
        x = layers.Dense(num_nodes, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    model.compile(
        loss = 'binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=HP_LEARNING_RATE),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    if show_summary:
        model.summary()

    return model 


TunerFnResult = NamedTuple(
    'TunerFnResult', 
    [('tuner', base_tuner.BaseTuner), 
     ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    '''fungsi ini dipanggil oleh Tuner
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
        
    Returns:
        TunerFnResult
    '''

    # Strategi hyperparameter tuning
    tuner = kt.RandomSearch(
        model_builder,
        max_trials=20,
        hyperparameters=get_hyperparameters(),
        objective=kt.Objective('val_binary_accuracy', 'max'),
        directory=fn_args.working_dir,
        project_name='my_tuning')

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', 
        mode='max', 
        verbose=1, 
        patience=10)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    # Vectorize batch of training text
    train_text = [j.numpy()[0] for i in list(train_set) 
                                    for j in i[0][transformed_name(FEATURE_KEY)]]
    vectorize_layer.adapt(train_text)

    return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          'callbacks':[es],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
    )
