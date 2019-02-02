import tensorflow as tf


def conv2d(x, kernel_size, out_channels, stride=[1,1], activation=tf.nn.elu):
    with tf.variable_scope('kernel%s/%s' % (out_channels, kernel_size)):
        return tf.layers.conv2d(
            x,
            out_channels,
            kernel_size,
            stride,
            "VALID",
            bias_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=True,
            trainable=True,
            reuse=tf.AUTO_REUSE,
            activation=activation
        )


def fully_connected(x, n_outputs, activation_fn=tf.nn.sigmoid):
    with tf.variable_scope('fc%s' % n_outputs):
        return tf.contrib.layers.fully_connected(
            x,
            n_outputs,
            activation_fn,
            biases_initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )