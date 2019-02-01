import tensorflow as tf
import numpy as np

from ops import conv2d


class DecompressorNetwork():

    def __init__(self):
        self.sess = tf.Session()

    # Build the network architecture
    def build(self):
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')  # batch_size x 32 x 32 x 3 input (rgb)
        cl_1 = conv2d(x, 3, 3, 64, name='c1')  # batch size x 30 x 30 x 64 feature maps\
        bn_1 = tf.nn.elu(tf.layers.batch_normalization(cl_1, trainable=True))

        pool = tf.nn.max_pool(bn_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        cl_2 = conv2d(pool, 2, 64, 128, name='c2', stride=[1, 2, 2, 1])
        bn_2 = tf.nn.elu(tf.layers.batch_normalization(cl_2, trainable=True))

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        actual_out = self.sess.run(bn_2, feed_dict={x: self.sess.run(tf.random.normal([1, 32, 32, 3]))})

        print(actual_out, actual_out.shape)


test = DecompressorNetwork()
test.build()