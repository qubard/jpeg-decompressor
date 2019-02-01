def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')
    return None


def get_training_data(data_directory, index):
    dict_ = unpickle(data_directory + '/' + 'data_batch_%i' % index)
    data_key = b'data'
    import numpy as np
    if dict_ and data_key in dict_:
        return np.reshape(dict_[data_key], (10000, 3, 32, 32))
    return None