from src.cifar import get_training_data

batch = get_training_data('dataset', 1)

print(batch.shape)

import matplotlib.pyplot as plt
import numpy as np

for i in range(0, 10):
    plt.imshow(np.transpose(batch[i])) # transpose to 32x32x3
    plt.show()