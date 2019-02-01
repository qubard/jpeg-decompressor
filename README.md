# jpeg-decompressor
jpeg decompressor written in tensorflow

# How does it work?

It's basically an autoencoder (convnet + batchnorm + residual connections) trained on noisy images and their denoised counterparts.

# Dependencies

- tensorflow
- numpy
- matplotlib (optional)