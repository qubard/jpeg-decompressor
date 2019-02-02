import tensorflow as tf

from .ops import conv2d, fully_connected


class DecompressorNetwork():

    def __init__(self, batch_size, learning_rate=1e-3):
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.learning_rate = learning_rate

        # Generate the model
        self.model = self.build()
        self.saver = tf.train.Saver()

    # Evaluate the tensor x on the model
    def eval(self, x):
        return self.sess.run(self.model, feed_dict={self.x: x})

    # Build the network architecture
    def build(self):
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, 32 * 32 * 3], name='y')
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3], name='x')  # batch_size x 32 x 32 x 3 input (rgb)
        cl_1 = conv2d(self.x, 3, 64)  # batch size x 30 x 30 x 64 feature maps\

        bn_1 = tf.nn.elu(tf.layers.batch_normalization(cl_1, trainable=True))

        pool = tf.nn.max_pool(bn_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        cl_2 = conv2d(pool, 2, 128)
        bn_2 = tf.nn.elu(tf.layers.batch_normalization(cl_2, trainable=True))

        # (batch_size, 14, 14, 128) reshape to (self.batch_size, 14*14*128)
        x_hat = fully_connected(tf.reshape(bn_2, [self.batch_size, 14*14*128]), 32*32*3)

        self.loss = tf.losses.mean_squared_error(self.y, x_hat)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        return x_hat

    def train(self, x, y, checkpoint=-1, n_epochs=1000):
        if x.shape[0] % self.batch_size != 0 or y.shape[0] % self.batch_size != 0:
            raise ValueError("Assigned batch size must be some multiple of the input's batch size!")

        if checkpoint >= 0:
            self.saver.restore(self.sess, "checkpoints/model%s.ckpt" % checkpoint)

        for i in range(checkpoint + 1, n_epochs):
            for v in range(1, int(x.shape[0] / self.batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=
                {
                    self.x : x[(v-1) * self.batch_size : v * self.batch_size],
                    self.y: y[(v-1) * self.batch_size : v * self.batch_size]
                })
                print("Epoch %s Batch %s Loss: %s" % (i, v * self.batch_size, loss))
                if i % 10 == 0:
                    save_path = self.saver.save(self.sess, "checkpoints/model%s.ckpt" % i)
                    print("Model saved in path: %s" % save_path)
