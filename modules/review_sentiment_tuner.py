'''
Tuner module
'''

from typing import (
    NamedTuple,
    Dict,
    Text,
    Any
)
import tensorflow as tf
import tensorflow_transform as tft
import kerastuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner.engine import base_tuner
from utils import (
    input_fn,
    get_hyperparameters,
    vectorize_dataset,
    model_builder,
)

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

    vectorize_dataset(train_set)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'callbacks': [es],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
