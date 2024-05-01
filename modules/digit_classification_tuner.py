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
from kerastuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.utils import (
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    input_fn,
    get_hyperparameters,
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
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_dataset = input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      tf_transform_output,
      is_train=True,
      batch_size=TRAIN_BATCH_SIZE)

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE)

    # Strategi hyperparameter tuning
    tuner = kt.RandomSearch(
        model_builder,
        max_trials=10,
        hyperparameters=get_hyperparameters(),
        objective=kt.Objective('val_sparse_categorical_accuracy', 'max'),
        directory=fn_args.working_dir,
        project_name='my_tuning')

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        verbose=1,
        patience=10)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'callbacks': [es],
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
