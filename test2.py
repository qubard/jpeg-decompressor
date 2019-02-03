import numpy as np

from src.network.decompressor import DecompressorNetwork
from src.helpers import compress_numpy_image

from src.cifar import get_training_data

n_samples = 1
batch = get_training_data('dataset', 1)
for i in range(1, n_samples):
    batch = np.append(get_training_data('dataset', i), batch, axis=0)

arr = np.array([np.transpose(batch[i]) / 255 for i in range(0, batch.shape[0])])

compressed_arr = np.array([compress_numpy_image(arr[i], quality=50) for i in range(0, 1)])

import matplotlib.pyplot as plt

n_rows = 2
f, ax = plt.subplots(n_rows, 2)
[axi.set_axis_off() for axi in ax.ravel()]

start = np.random.randint(low=0, high=499)

for index in range(0, n_rows):
    ax[index][0].imshow(arr[0])
    ax[index][0].set_title('Original')

    ax[index][1].imshow(compressed_arr[0])
    ax[index][1].set_title('Reconstruction')

plt.show()