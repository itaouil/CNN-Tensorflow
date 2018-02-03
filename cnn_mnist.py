from __future__ import division
from __future__ import print_function
from __future__ import absolute_division

# Imports
import numpy as np
import tensorflow as tf

# CNN model
def cnn_model_fn(features, labels, mode):
    """ CNN model """

    # CNN input where the input is
    # [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolution layer #1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu
    )

    # Pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    # Convolution layer #2
    # Convolution layer #1
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu
    )

    # Pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

    # Flatten pool2 output (dense layer)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense layer
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)

    # Dropout regularisation (avoid overfitting)
    droput = tf.layers.dropout(inputs = dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

# Logging flag
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
    tf.app.run()
