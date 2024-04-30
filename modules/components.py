'''Initiate tfx pipeline components
'''

import os
import tensorflow_model_analysis as tfma
from utils import (
    LABEL_KEY,
    transformed_name,
)

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)


def init_components(
    data_dir,
    module_files,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    '''Initiate tfx pipeline components

    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the transform_module
        training_steps (int): number of training steps
        eval_steps (int): number of eval steps
        serving_model_dir (str): a path to the serving model directory

    Returns:
        TFX components
    '''

    components = {}

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
        ])
    )

    components['example_gen'] = CsvExampleGen(
        input_base=data_dir,
        output_config=output
    )

    components['statistics_gen'] = StatisticsGen(
        examples=components['example_gen'].outputs['examples']
    )

    components['schema_gen'] = SchemaGen(
        statistics=components['statistics_gen'].outputs['statistics']
    )

    components['example_validator'] = ExampleValidator(
        statistics=components['statistics_gen'].outputs['statistics'],
        schema=components['schema_gen'].outputs['schema']
    )

    components['transform'] = Transform(
        examples=components['example_gen'].outputs['examples'],
        schema=components['schema_gen'].outputs['schema'],
        module_file=os.path.abspath(module_files['transform'])
    )

    components['tuner'] = Tuner(
        module_file=os.path.abspath(module_files['tuner']),
        examples=components['transform'].outputs['transformed_examples'],
        transform_graph=components['transform'].outputs['transform_graph'],
        schema=components['schema_gen'].outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

    components['trainer'] = Trainer(
        module_file=os.path.abspath(module_files['trainer']),
        examples=components['transform'].outputs['transformed_examples'],
        transform_graph=components['transform'].outputs['transform_graph'],
        schema=components['schema_gen'].outputs['schema'],
        hyperparameters=components['tuner'].outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

    components['resolver'] = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    slicing_specs = [
        tfma.SlicingSpec()
    ]

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name="ExampleCount"),
            tfma.MetricConfig(class_name='Precision'),
            tfma.MetricConfig(class_name='Recall'),
            tfma.MetricConfig(class_name='F1Score'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.85}),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={'value': 0.0001})
                              )
                              )
        ])
    ]
    eval_config = tfma.EvalConfig(
        # Label key pakai label key yang sudah ditransform
        # karena output modelnya numeric bukan 'negative/positive'
        model_specs=[tfma.ModelSpec(
            signature_name='serving_default',
            label_key=transformed_name(LABEL_KEY),
            preprocessing_function_names=['transform_features'],
            prediction_key='output'
        )],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    components['evaluator'] = Evaluator(
        examples=components['example_gen'].outputs['examples'],
        model=components['trainer'].outputs['model'],
        baseline_model=components['resolver'].outputs['model'],
        eval_config=eval_config)

    components['pusher'] = Pusher(
        model=components['trainer'].outputs['model'],
        model_blessing=components['evaluator'].outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    return list(components.values())
