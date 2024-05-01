'''
Trainer module
'''

import os
import tensorflow as tf
import tensorflow_transform as tft
import kerastuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.utils import (
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LABEL_KEY,
    input_fn,
    get_hyperparameters,
    model_builder,
)


def run_fn(fn_args: FnArgs) -> None:
    '''Train the model based on given args. dipanggil oleh TFX Trainer

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    '''
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE)
    val_set = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE)

    if fn_args.hyperparameters is None:
        hp = get_hyperparameters()
    else:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)

    model = model_builder(hp)

    # Callbacks
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            verbose=1,
            patience=10)

    # Train model
    model.fit(
        x=train_set,
        validation_data=val_set,
        epochs=50,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, es])

    signatures = make_serving_signatures(model, tf_transform_output)

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures)


def make_serving_signatures(
    model,
    tf_transform_output: tft.TFTransformOutput):
    '''Returns the serving signatures.

    Args:
    model: the model function to apply to the transformed features.
    tf_transform_output: The transformation to apply to the serialized
        tf.Example.

    Returns:
    The signatures to use for saving the mode. The 'serving_default' signature
    will be a concrete function that takes a batch of unspecified length of
    serialized tf.Example, parses them, transformes the features and
    then applies the model. The 'transform_features' signature will parses the
    example and transforms the features.
    '''

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_example, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        # Predict
        pred = model(transformed_features)
        # kebetulan label sama dengan index nya
        # misal argmax kan cari index dengan value paling besar
        # misal ketemu index=7. maka labelnya ya 7 itu
        label = tf.math.argmax(pred, 1)

        return {
            'output': pred,
            'label': label
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        '''Returns the transformed_features to be fed as input to evaluator.'''
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        return transformed_features

    return {
        'serving_default': serve_tf_examples_fn,
        'transform_features': transform_features_fn
    }
