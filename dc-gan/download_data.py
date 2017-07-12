import os
import gzip
import requests

from settings import PROJECT_PATH

filenames = ['t10k-labels-idx1-ubyte',
             't10k-images-idx3-ubyte',
             'train-labels-idx1-ubyte',
             'train-images-idx3-ubyte']

main_url = 'http://yann.lecun.com/exdb/mnist/'


def extract_zipfile(filepath):
    with gzip.open(filepath + '.gz', 'rb') as f:
        file_content = f.read()
    with open(filepath, 'w') as f:
        f.write(file_content)
    os.remove(filepath + '.gz')


def download_mnist(destination_dir, filename):
    whole_url = main_url + filename + '.gz'
    gzipped_filepath = os.path.join(destination_dir, filename + '.gz')
    with open(gzipped_filepath, "wb") as f:
        r = requests.get(whole_url)
        f.write(r.content)
    extract_zipfile(os.path.join(PROJECT_PATH, 'data', 'mnist', filename))


def download_if_not_present():
    for filename in filenames:
        destination_dir = os.path.join(PROJECT_PATH, 'data', 'mnist')
        if filename not in os.listdir(destination_dir):
            download_mnist(destination_dir, filename)

if __name__ == '__main__':
    download_if_not_present()