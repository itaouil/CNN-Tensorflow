from __future__ import division
from __future__ import print_function
from __future__ import absolute_division

# Imports
import numpy as np
import tensorflow as tf

# Logging flag
tf.logging.set_verbosity(tf.logging.INFO)

# CNN model
def cnn_model_fn(features, labels, mode):
    """
        CNN model

        Arguments:
            param1: Input vector
            param2: Batche's labels (i.e. targets)
            param3: Network mode (Train, Eval, etc)
    """

    # CNN input where the input is
    # [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    """ Convolution and pooling layers """
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

    """ Fully connected layer (a.k.a dense layer) """
    # Dense layer
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)

    """ Loss function + training process """
    # Dropout regularisation (avoid overfitting)
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    # Output layer
    logits = tf.layers.dense(inputs = dropout, units = 10)

    # Predictions
    predictions = {
        "classes": tf.argmax(inputs = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # One-hot encoding
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 10)

    # Loss as softmax cross-entropy
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)

    """ Loss function optimisation """
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    """ Evaluation metrics """
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels = labels,
            predictions = predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        eval_metric_ops = eval_metric_ops
    )

def main():
    # Split dataset in training and test sets
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype = np.int32)

    # Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn,
        model_dir = "./mnist_convnet_model"
    )

    # Loggin for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LogginTensorHook(
        tensors = tensors_to_log,
        every_n_iter = 50
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )

    mnist_classifier.train(
        input_fn = train_input_fn,
        steps = 20000,
        hooks = [logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
