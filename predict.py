"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import random
import sys

import pandas as pd
import tensorflow as tf
import numpy as np

import resnet_model
import vgg_preprocessing
from prepare_dataset import get_feature_label

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,
                    default='/home/hugo/datasets/celebA',
                    help='The path to the celebA data directory.')

parser.add_argument('--model_dir', type=str,
                    default='/home/hugo/datasets/celebA/Male_models',
                    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size', type=int, default=18, choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=5,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=25,
                    help='The number of images per batch.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 32

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_TRAIN_NUM = 160000
_CHOOSE_NUM = 50

_FILE_SHUFFLE_BUFFER = 200
_SHUFFLE_BUFFER = 200


def predict_record_parser(filename):
    image = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(tf.reshape(image, shape=[]), channels=_NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=False)
    return image


def predict_input_fn(name):
    """Input function which provides batches for train or eval."""
    filenames = os.path.join(FLAGS.data_dir, 'img_align_celeba', name)
    filenames = tf.constant(filenames)
    image = predict_record_parser(filenames)
    image = tf.expand_dims(image, 3)
    # with tf.Session() as sess:
    #     iiii = image
    #     aqwe = 321
    return image, 0


def resnet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""

    network = resnet_model.imagenet_resnet_v2(
        params['resnet_size'], _NUM_CLASSES, params['data_format'])
    logits = network(
        inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def main(unused_argv):
    path = ''

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'batch_size': 1,
        })

    print('Starting to predict.')
    predict_results = resnet_classifier.predict(
        input_fn=lambda: predict_input_fn('000651.jpg'))
    print(list(predict_results))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
