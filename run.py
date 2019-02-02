from src.cifar import get_training_data

batch = get_training_data('dataset', 1)

print(batch.shape)

import matplotlib.pyplot as plt
import numpy as np

from src.network.decompressor import DecompressorNetwork

arr = np.array([np.transpose(batch[i]) / 255 for i in range(0, batch.shape[0])])

test = DecompressorNetwork(batch_size=500)
#test.train(arr[:500], checkpoint=510)

test.saver.restore(test.sess, "checkpoints/model510.ckpt")
recovered_xes = test.eval(arr[:500])


n_rows = 3
f, ax = plt.subplots(n_rows, 2)
[axi.set_axis_off() for axi in ax.ravel()]

start = np.random.randint(low=0, high=499)

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

for index in range(0, n_rows):
    output = np.reshape(recovered_xes[start + index], [32, 32, 3])

    ax[index][0].imshow(arr[start + index])
    ax[index][0].set_title('Original')

    compressed = compress_numpy_image(arr[start + index], quality=30)

    ax[index][1].imshow(compressed)
    ax[index][1].set_title('Compressed')

plt.show()