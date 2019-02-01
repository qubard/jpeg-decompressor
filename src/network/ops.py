import tensorflow as tf


def conv2d(x, kernel_size, in_channels, out_channels, name, stride=[1,1,1,1]):
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape=[kernel_size, kernel_size, in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(x, w, stride, "VALID")
