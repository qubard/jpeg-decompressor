from src.cifar import get_training_data

batch = get_training_data('dataset', 1)

import numpy as np

from src.network.decompressor import DecompressorNetwork

arr = np.array([np.transpose(batch[i]) / 255 for i in range(0, batch.shape[0])])

from PIL import Image
import io

# Compress the numpy array as an image to a JPEG
def compress_numpy_image(arr, quality=50):
    im = Image.fromarray(np.uint8(255 * arr))
    out = io.BytesIO()
    im.save(out, format='JPEG', quality=quality)
    obj = Image.open(out)
    compressed = np.zeros(arr.shape)

    # Copy each pixel over (no idea if PIL supports np.array copies)
    for x in range(0, arr.shape[0]):
        for y in range(0, arr.shape[1]):
            pixel = obj.getpixel((x, y))
            for channel in range(0, len(pixel)):
                compressed[y][x][channel] = pixel[channel] / 255

    return compressed


compressed_arr = np.array([compress_numpy_image(arr[i], quality=10) for i in range(0, arr.shape[0])])


test = DecompressorNetwork(batch_size=1000)
test.train(x=compressed_arr, y=np.reshape(arr, (arr.shape[0], 32 * 32 * 3)))