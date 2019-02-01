import tensorflow as tf
import numpy as np

from ops import conv2d


class DecompressorNetwork():

    def __init__(self):
        self.sess = tf.Session()

    # Build the network architecture
    def build(self):
        self.c1 = conv2d(tf.random.normal([5, 32, 32, 3]), 3, 3)
        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

    def test(self):
        actual_out = self.sess.run(self.c1)
        print(actual_out.shape)

test = DecompressorNetwork()
test.build()

test.test()