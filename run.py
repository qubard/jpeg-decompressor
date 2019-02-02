from src.cifar import get_training_data

batch = get_training_data('dataset', 1)

print(batch.shape)

import matplotlib.pyplot as plt
import numpy as np

from src.network.decompressor import DecompressorNetwork

arr = np.array([np.transpose(batch[i]) / 255 for i in range(0, batch.shape[0])])

#test = DecompressorNetwork(batch_size=500)
#test.train(arr[:500])

import tensorflow as tf

a = tf.constant([[1,1,1], [2,1,1]])
b = tf.constant([[1,0,1,], [2,1,1]]) # ground truth

c = tf.losses.mean_squared_error(b, a)

with tf.Session() as sess:
    print(sess.run(c))
