import tensorflow as tf
import numpy as np

from .ops import conv2d


class DecompressorNetwork():

    def __init__(self, batch_size, learning_rate=1e-3):
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.learning_rate = learning_rate

        # Generate the model
        self.model = self.build()
        self.saver = tf.train.Saver()

    def restore_checkpoint(self, checkpoint):
        self.saver.restore(self.sess, "checkpoints/model%s.ckpt" % checkpoint)

    # Evaluate the tensor x on the model
    def eval(self, x):
        return self.sess.run(self.model, feed_dict={self.x: x})

    # Build the network architecture
    def build(self):
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3], name='y')
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3], name='x')  # batch_size x 32 x 32 x 3 input (rgb)

        cl_0 = conv2d(self.x, 3, 128, padding="SAME", name='cl0')
        bn_0 = tf.nn.elu(tf.layers.batch_normalization(cl_0, trainable=True))
        cl_1 = conv2d(bn_0, 3, 64, padding="SAME", name='cl1')

        bn_1 = tf.nn.elu(tf.layers.batch_normalization(cl_1, trainable=True))

        cl_2 = conv2d(bn_1, 3, 64, padding="SAME", name='cl2')
        bn_2 = tf.nn.elu(tf.layers.batch_normalization(cl_2, trainable=True))

        cl_3 = conv2d(bn_2, 2, 32, padding="SAME", name='cl3')
        bn_3 = tf.layers.batch_normalization(cl_3, trainable=True)

        cl_4 = conv2d(bn_3, 2, 3, padding="SAME", name='cl4', activation=tf.nn.sigmoid)
        bn_4 = tf.nn.relu(self.x + tf.layers.batch_normalization(cl_4, trainable=True)) # residual connection

        self.loss = tf.losses.mean_squared_error(self.y, bn_4)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        return bn_4

    def train(self, x, y, checkpoint=-1, n_epochs=1000):
        if x.shape[0] % self.batch_size != 0 or y.shape[0] % self.batch_size != 0:
            raise ValueError("Assigned batch size must be some multiple of the input's batch size!")

        if checkpoint >= 0:
            self.restore_checkpoint(checkpoint)

        n_pairs = int(x.shape[0] / self.batch_size)
        indices = np.zeros((n_pairs - 1, 2), dtype=np.int32)

        for i in range(0, indices.shape[0]):
            indices[i] = [i * self.batch_size, (i + 1) * self.batch_size]

        for i in range(checkpoint + 1, n_epochs):
            np.random.shuffle(indices)
            total_loss = 0
            for index in range(0, indices.shape[0]):
                indice = indices[index]
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=
                {
                    self.x: x[indice[0]:indice[1]],
                    self.y: y[indice[0]:indice[1]]
                })
                total_loss += loss
            avg_loss = (self.batch_size * (total_loss / indices.shape[0]))
            print("Epoch %s Batch %s AvgLoss: %s" % (i, indice, avg_loss))
            if i % 5 == 0:
                save_path = self.saver.save(self.sess, "checkpoints/model%s.ckpt" % i)
                print("Model saved in path: %s" % save_path)
