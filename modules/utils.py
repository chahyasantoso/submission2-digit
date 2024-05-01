'''
Utilily module
'''

import os
import io
import numpy as np
import base64

import cv2
from PIL import Image, ImageOps
from typing import List
from pprint import PrettyPrinter

import tensorflow as tf
import kerastuner as kt
from keras import layers
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

IMAGE_WIDTH = 75
IMAGE_HEIGHT = 75

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-4

FEATURE_KEY = 'image'
LABEL_KEY = 'label'


def transformed_name(key):
    '''Renaming transformed features'''
    return key + '_xf'


def image_augmentation(image_features):
    """Perform image augmentation on batches of images .

    Args:
    image_features: a batch of image features

    Returns:
    The augmented image features
    """
    batch_size = tf.shape(image_features)[0]
    #image_features = tf.image.random_flip_left_right(image_features)
    image_features = tf.image.resize_with_crop_or_pad(
        image_features, IMAGE_HEIGHT+15, IMAGE_WIDTH+15)
    image_features = tf.image.random_crop(image_features,
        (batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    return image_features


def data_augmentation(feature_dict):
    """Perform data augmentation on batches of data.

    Args:
    feature_dict: a dict containing features of samples

    Returns:
    The feature dict with augmented features
    """
    image_features = feature_dict[transformed_name(FEATURE_KEY)]
    image_features = image_augmentation(image_features)
    feature_dict[transformed_name(FEATURE_KEY)] = image_features
    return feature_dict


def input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              is_train: bool = False,
              batch_size: int = 64) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    is_train: Whether the input dataset is train split or not.
    batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
    A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=transformed_name(LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema).repeat()

    # Apply data augmentation. We have to do data augmentation here because
    # we need to apply data agumentation on-the-fly during training. If we put
    # it in Transform, it will only be applied once on the whole dataset, which
    # will lose the point of data augmentation.
    if is_train:
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y))

    return dataset


def get_hyperparameters() -> kt.HyperParameters:
    '''Returns HyperParameters for keras model'''
    hp = kt.HyperParameters()
    hp.Int(
        'hidden_size',
        min_value=64,
        max_value=256,
        step=64,
        default=256
    )
    hp.Float(
        'dropout_rate',
        min_value=0.2,
        max_value=0.5,
        step=0.1,
        default=0.2
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
    base_model = tf.keras.applications.InceptionV3 (
      input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
      include_top=False,
      weights='imagenet',
      pooling='avg'
      )
    base_model.input_spec = None
    base_model.trainable = False

    inputs = tf.keras.Input(
          shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
          name=transformed_name(FEATURE_KEY))
    x = base_model(inputs, training=False)

    hidden_size = int(hp.get('hidden_size'))
    dropout_rate = float(hp.get('dropout_rate'))
    x = layers.Dense(hidden_size, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_size, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        metrics=['sparse_categorical_accuracy'])

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
        elif dtype == bytes:
            feature[key] = _bytes_feature(data_dict[key])
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

def segment_image(image_filename):
    '''
    men-segment/memecah image menjadi bagian2
    misal 1 image berisi 4 digit angka, 
    maka image akan dipecah 4 image dengan masing2 berisi 1 digit
    '''
    # load the image, convert it to grayscale, and blur it to remove noise
    image = cv2.imread(image_filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # threshold the image
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilate the white portions
    dilate = cv2.dilate(thresh1, None, iterations=2)

    # find contours in the image
    contours = cv2.findContours(
        dilate.copy(), 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[0]
    
    def get_ROI(contour):
        # Filtered countours are detected
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]

    # only get the segment when area of contour is big enough
    segments = [get_ROI(contour) for contour in contours if(cv2.contourArea(contour) >= 100)]
    return segments


def array_to_png(image_array):

    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    PIL_image = Image.fromarray(
        np.array(image_array).astype('uint8'), 'RGB')
    PIL_image = PIL_image.resize((75, 75))
    PIL_image = add_margin(PIL_image,30,30,30,30, (255, 255, 255))
    PIL_image = PIL_image.convert('L') #.point(fn, mode='1')
    PIL_image = ImageOps.invert(PIL_image)
    
    output = io.BytesIO()
    PIL_image.save(output, format="png")
    
    return output.getvalue()
