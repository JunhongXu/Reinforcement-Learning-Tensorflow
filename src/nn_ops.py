import tensorflow as tf


def dense_layer(x, output_dim, initializer, scope, use_bias):
    """
    A convenient function for constructing fully connected layers
    """

    shape = x.get_shape().as_list()
    if len(shape) == 2:     # if the previous layer is fully connected, the shape of X is (N, D)
        D = shape[1]
    else:                   # if the previous layer is convolutional, the shape of X is (N, H, W, C)
        N, H, W, C = shape
        D = H * W * C
        x = tf.reshape(x, (-1, D))

    with tf.variable_scope(scope):
        w = tf.get_variable("W", shape=(D, output_dim), initializer=initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        # calculate
        x = tf.matmul(x, w)

        if use_bias:
            b = tf.get_variable("b", shape=output_dim, initializer=tf.constant_initializer(.0, dtype=tf.float32))
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
            x = tf.nn.bias_add(x, b)

    return x


def conv2d(x, filter_size, stride, output_size, initializer, scope, use_bias, padding="VALID"):
    """
    A convenient function for constructing convolutional layer
    """

    # input x should be (N, H, W, C)
    N, H, W, C = x.get_shape().as_list()
    stride = (1, stride, stride, 1)

    with tf.variable_scope(scope):
        w = tf.get_variable("W", shape=(filter_size, filter_size, C, output_size), initializer=initializer,
                            dtype=tf.float32)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        x = tf.nn.conv2d(x, w, strides=stride, padding=padding)

        if use_bias:
            b = tf.get_variable("b", shape=output_size, initializer=tf.constant_initializer(0.01), dtype=tf.float32)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
            x = tf.nn.bias_add(x, b)

    return x


def batch_norm(x, is_train, scope):
    """
    A wrapper for batch normalization layer
    """
    train_time = tf.contrib.layers.batch_norm(x, decay=0.9, scope="%s/bn" % scope, center=True, scale=True,
                                              updates_collections=None, is_training=True, reuse=None)
    test_time = tf.contrib.layers.batch_norm(x, decay=0.9, scope="%s/bn" % scope, center=True, scale=True,
                                             updates_collections=None, is_training=False, reuse=True)

    x = tf.cond(is_train, lambda: train_time, lambda: test_time)
    return x
