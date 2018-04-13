"""Runs a ResNet model on the CelebA dataset."""

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
from faceDetect import saveFaces, drawFaces

parser = argparse.ArgumentParser()

# parser.add_argument('--data_dir', type=str,
#                     default='/home/hugo/Pictures',
#                     help='The path to the celebA data directory.')
DATA_DIR = '/home/hugo/Pictures'

# parser.add_argument('--model_dir', type=str,
#                     default='/home/hugo/datasets/celebA',
#                     help='The directory where the model will be stored.')
MODEL_DIR = '/home/hugo/datasets/celebA'

# parser.add_argument(
#     '--resnet_size', type=int, default=18, choices=[18, 34, 50, 101, 152, 200],
#     help='The size of the ResNet model to use.')
RESNET_SIZE = 18

# parser.add_argument('--batch_size', type=int, default=25,
#                     help='The number of images per batch.')

BATCH_SIZE = 25

# parser.add_argument(
#     '--data_format', type=str, default=None,
#     choices=['channels_first', 'channels_last'],
#     help='A flag to override the data format used in the model. channels_first '
#          'provides a performance boost on GPU but is not always compatible '
#          'with CPU. If left unspecified, the data format will be chosen '
#          'automatically based on whether TensorFlow was built for CPU or GPU.')
DATA_FORMAT = None

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 2

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_TRAIN_NUM = 199999
# _TRAIN_NUM = 160000
_CHOOSE_NUM = 50

_FILE_SHUFFLE_BUFFER = 200
_SHUFFLE_BUFFER = 200


def get_filenames(is_training, feature_label):
    """Returns a list of filenames and labels."""

    features, labels = zip(*feature_label)

    if is_training:

        return features[:_TRAIN_NUM], labels[:_TRAIN_NUM]
    else:
        # return [os.path.join(data_dir, 'test_batch.bin')]
        return features[_TRAIN_NUM:], labels[_TRAIN_NUM:]


def _parse_func(filename, label):
    """Convert features(filenames) into images
       Convert label into one-hot encoding
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)

    # image_decoded = tf.image.resize_images(image_decoded, [32, 32])
    label = tf.one_hot(label, _NUM_CLASSES)
    return image_decoded, label


def record_parser(filename, label, is_training):
    image = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(tf.reshape(image, shape=[]), channels=_NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=is_training)

    tf.cast(label, tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)

    return image, label


def input_fn(is_training, filename, batch_size, num_epochs=1):
    """Input function which provides batches for train or eval."""
    # filenames, labels = get_filenames(is_training, feature_label)
    filenames, labels = [filename], [0]
    # filenames = [os.path.join(DATA_DIR, 'face', filename) for filename in filenames]
    filenames = [os.path.join('./', 'detected_face', filename) for filename in filenames]
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=_TRAIN_NUM)

    # dataset = dataset.map(_parse_func)
    dataset = dataset.map(lambda filename, label: record_parser(filename, label, is_training),
                          num_parallel_calls=1)
    dataset = dataset.prefetch(batch_size)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    # with tf.Session() as sess:
    #     qwe = sess.run(images)
    #     rty = sess.run(labels)
    #     aaa = 32131
    return images, labels


def resnet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    tf.summary.image('images', features, max_outputs=6)

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

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 256, the learning rate should be 0.1.
        initial_learning_rate = 0.1 * params['batch_size'] / 25
        batches_per_epoch = _TRAIN_NUM / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        values = [
            initial_learning_rate * decay for decay in [1, 0.5, 0.3, 0.2, 0.1]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes.
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes.
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def run_predict(image_name):
    #  face detect
    drawFaces(image_name)
    saveFaces(image_name)
    res = {}
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # image_name = '100_2431_01.png'
    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    ls_attr = ['Male', 'Smiling', 'Eyeglasses', 'Young', 'Attractive']
    for attr in ls_attr:
        model_path = os.path.join(MODEL_DIR, attr + '_models')
        classifier = tf.estimator.Estimator(
            model_fn=resnet_model_fn, model_dir=model_path, config=run_config,
            params={
                'resnet_size': RESNET_SIZE,
                'data_format': DATA_FORMAT,
                'batch_size': BATCH_SIZE,
            })
        results = classifier.predict(input_fn=lambda: input_fn(False, image_name, BATCH_SIZE)[0])
        # print(attr + '\'s value:')
        # print(list(results)[0])
        res[attr] = list(results)[0]
    print(res)


if __name__ == '__main__':
    run_predict('delrey.jpg')
