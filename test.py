import numpy as np
from src.network import DecompressorNetwork

test = DecompressorNetwork(batch_size=1)
test.restore_checkpoint(20)

from src.helpers import image_to_numpy_array, compress_numpy_image

from src.cifar import get_training_data

batch = get_training_data('dataset', 1)

arr = np.array([np.transpose(batch[i]) / 255 for i in range(0, batch.shape[0])])

compressed_arr = np.array([compress_numpy_image(arr[i], quality=10) for i in range(0, arr.shape[0])])

#arr = image_to_numpy_array("test-in.jpg")
output = test.eval(np.reshape(compressed_arr[0], [1, 32, 32, 3]))
output = np.reshape(output, (32, 32, 3))

import matplotlib.pyplot as plt

n_rows = 2
f, ax = plt.subplots(n_rows, 2)
[axi.set_axis_off() for axi in ax.ravel()]

start = np.random.randint(low=0, high=499)

for index in range(0, n_rows):
    ax[index][0].imshow(arr[0])
    ax[index][0].set_title('Original')

    ax[index][1].imshow(output)
    ax[index][1].set_title('Reconstruction')

plt.show()