import numpy as np

from src.network.decompressor import DecompressorNetwork
from src.helpers import compress_numpy_image

from src.cifar import get_training_data

n_samples = 5
batch = get_training_data('dataset', 1)
for i in range(1, n_samples):
    batch = np.append(get_training_data('dataset', i), batch, axis=0)

arr = np.array([np.transpose(batch[i]) / 255 for i in range(0, batch.shape[0])])

compressed_arr = np.array([compress_numpy_image(arr[i], quality=10) for i in range(0, arr.shape[0])])

test = DecompressorNetwork(batch_size=500)

checkpoint = 25
test.restore_checkpoint(checkpoint=checkpoint)
test.train(x=compressed_arr, y=np.reshape(arr, (arr.shape[0], 32 * 32 * 3)), checkpoint=checkpoint)