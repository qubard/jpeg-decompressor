import tensorflow as tf


def conv2d(x, kernel_size, out_channels, name, stride=[1,1], padding="VALID", activation=tf.nn.elu):
    with tf.variable_scope(name):
        return tf.layers.conv2d(
            x,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=True,
            trainable=True,
            reuse=tf.AUTO_REUSE,
            activation=activation
        )


def fully_connected(x, n_outputs, activation=tf.nn.sigmoid):
    with tf.variable_scope('fc%s' % n_outputs):
        return tf.contrib.layers.fully_connected(
            x,
            n_outputs,
            activation,
            biases_initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )


def conv_transpose(x, kernel_size, output_shape, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w",[kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1,2,2,1])