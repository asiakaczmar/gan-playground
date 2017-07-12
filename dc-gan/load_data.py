import numpy as np
import os

from settings import PROJECT_PATH

DATA_DIR = os.path.join(PROJECT_PATH, 'data', 'mnist')


def load_one_file(filename, offset, shape, cast_float=False):
    with open(os.path.join(DATA_DIR, filename)) as f:
        data = np.fromfile(file=f, dtype=np.uint8)
        data = data[offset:].reshape(shape)
        return data.astype(float) if cast_float else data


def load_mnist():
    train_x = load_one_file('train-images-idx3-ubyte', 16, (60000, 28*28), True)
    test_x = load_one_file('t10k-images-idx3-ubyte', 16, (10000, 28 * 28), True)

    train_y = np.asarray(load_one_file('train-labels-idx1-ubyte', 8, (60000,)))
    test_y = np.asarray(load_one_file('t10k-labels-idx1-ubyte', 8, (10000,)))

    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    x = load_mnist()