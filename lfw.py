from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2
from data_loader import *
from neg_generator import *



def generate_positive(output_path):
    # download LFW
    lfw_people = fetch_lfw_people(funneled=False, slice_=(slice(70, 195), slice(78 - 1, 172 + 1)), resize=1, download_if_missing=False)



    data = lfw_people.images
    shape = data.shape
    # crop
    w = min(shape[1], shape[2])
    data = data[:, int((shape[1] - w) / 2) : w + int((shape[1] - w) / 2), int((shape[2] - w) / 2) : w + int((shape[2] - w) / 2)]
    data = np.array([cv2.resize(d, (24, 24)) for d in data])
    # normalize
    data = normalize(data)
    # add flipped copy
    data_fliped = data[:, :, ::-1]
    data = np.concatenate((data, data_fliped), axis=0)

    # dump dataset
    X = data
    y = np.ones((X.shape[0]))
    print(X.shape, y.shape)
    dump_dataset(output_path, X, y)

generate_positive('./data/all_data_pos.pkl')