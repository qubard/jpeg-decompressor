# jpeg-decompressor
jpeg decompressor written in tensorflow

# How does it work?

It's basically an autoencoder (convnet + batchnorm + residual connections) trained on noisy images generated using CIFAR-10 and their denoised counterparts.

# Setup

Firstly run
```
mkdir dataset/
mkdir checkpoints/
```

Then install `CIFAR-10` dataset and then drag all the `data_batch_#` files into the `dataset/` directory.

Start the virtual environment

`
virtualenv venv
`

And install the requirements

`pip install -r requirements.txt`

# Training
To train simply run the training script `python train.py`.

# Dependencies

See `requirements.txt`.