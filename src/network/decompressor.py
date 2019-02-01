import tensorflow as tf

from .ops import conv2d, fully_connected


class DecompressorNetwork():

    def __init__(self, batch_size=1):
        self.batch_size = 1
        self.sess = tf.Session()
        self.model = self.build()

    # Build the network architecture
    def build(self):
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3], name='x')  # batch_size x 32 x 32 x 3 input (rgb)
        cl_1 = conv2d(self.x, 3, 64)  # batch size x 30 x 30 x 64 feature maps\
        bn_1 = tf.nn.elu(tf.layers.batch_normalization(cl_1, trainable=True))

        pool = tf.nn.max_pool(bn_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        cl_2 = conv2d(pool, 2, 128)
        bn_2 = tf.nn.elu(tf.layers.batch_normalization(cl_2, trainable=True))

        # (batch_size, 14, 14, 128) reshape to (self.batch_size, 14*14*128)
        d1 = fully_connected(tf.reshape(bn_2, [self.batch_size, 14*14*128]), 32*32*3)
        #d2 = fully_connected(d1, 32*32*3, 'd2')

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        return d1

    def eval(self):
        return self.sess.run(self.model, feed_dict={self.x: self.sess.run(tf.random.normal([self.batch_size, 32, 32, 3]))})
