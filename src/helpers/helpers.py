import numpy as np
import io

from PIL import Image


# Compress the numpy array as an image to a JPEG
def compress_numpy_image(arr, quality=50, grayscale=False):
    im = Image.fromarray(np.uint8(255 * arr))
    out = io.BytesIO()
    im.save(out, format='JPEG', quality=quality)
    obj = Image.open(out)
    compressed = np.zeros(arr.shape, dtype=np.float32)

    # Copy each pixel over (no idea if PIL supports np.array copies)
    for x in range(0, arr.shape[0]):
        for y in range(0, arr.shape[1]):
            pixel = obj.getpixel((x, y))
            if grayscale:
                compressed[y][x] = pixel / 255
            else:
                for channel in range(0, len(pixel)):
                    compressed[y][x][channel] = pixel[channel] / 255

    return compressed


def image_to_numpy_array(filename):
    im = Image.open(filename)
    width, height = im.size

    arr = np.zeros((width, height, 3))

    for x in range(0, width):
        for y in range(0, height):
            pixel = im.getpixel((x, y))
            for channel in range(0, len(pixel)):
                arr[y][x][channel] = pixel[channel] / 255

    return arr