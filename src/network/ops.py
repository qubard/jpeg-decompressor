import tensorflow as tf


def conv2d(x, kernel_size, out_channels, stride=[1,1,1,1]):
    w = tf.get_variable("w", shape=[kernel_size, kernel_size, 3, out_channels],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(x, w, stride, "SAME")
