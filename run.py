from src.cifar import get_training_data

batch = get_training_data('dataset', 1)

print(batch.shape)

import matplotlib.pyplot as plt
import numpy as np

from src.network.decompressor import DecompressorNetwork

test = DecompressorNetwork()
arr = test.eval()

plt.imshow(np.reshape(arr, (32, 32, 3))) # transpose to 32x32x3
plt.show()