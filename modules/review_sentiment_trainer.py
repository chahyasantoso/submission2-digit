'''
Trainer module
'''

import os
import tensorflow as tf
import tensorflow_transform as tft
import kerastuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from utils import (
    input_fn,
    LABEL_KEY,
    vectorize_dataset,
    model_builder,
)


def run_fn(fn_args: FnArgs) -> None:
    '''Train the model based on given args. dipanggil oleh TFX Trainer

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    '''

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    vectorize_dataset(train_set)

    # Build the model, dengan hyperparameter yang dikirim dari dari args
    hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    model = model_builder(hparams)

    # Train the model
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, mc],
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=20
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))}
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    '''Returns a function that parses a serialized tf.Example. dipanggil untuk TFX Serving.'''

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        '''Returns the output to be used in the serving signature.
        Args:
            serialized_tf_examples: serialized tf.Example bukan text biasa
        '''

        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn
