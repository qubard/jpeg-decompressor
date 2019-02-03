import numpy as np
from src.network import DecompressorNetwork

test = DecompressorNetwork(batch_size=1)
test.restore_checkpoint(25)

from src.helpers import image_to_numpy_array

arr = image_to_numpy_array("test-in.jpg")
output = test.eval(np.reshape(arr, [1, 32, 32, 3]))
output = np.reshape(output, (32, 32, 3))

import matplotlib.pyplot as plt

n_rows = 2
f, ax = plt.subplots(n_rows, 2)
[axi.set_axis_off() for axi in ax.ravel()]

start = np.random.randint(low=0, high=499)

for index in range(0, n_rows):
    ax[index][0].imshow(arr)
    ax[index][0].set_title('Original')

    ax[index][1].imshow(output)
    ax[index][1].set_title('Reconstruction')

plt.show()