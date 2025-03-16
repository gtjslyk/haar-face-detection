#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: neg_generator
Author: lyk
Date: 2025-02-13
Description: generator of negative data. run load_from_dir to load pictures from dir and generate .pkl file. run generate_negetive to generate negative dataset
"""
import cv2, os, random, pickle
import numpy as np
import data_loader
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people

i = 0
def normalize(imgs: np.ndarray):
    means = np.mean(imgs, axis=(1, 2), keepdims=True)
    stds = np.std(imgs, axis=(1, 2), keepdims=True)
    stds[stds == 0] = 1
    normalized_imgs = (imgs - means) / stds
    mins = np.min(normalized_imgs, axis=(1, 2), keepdims=True)
    maxs = np.max(normalized_imgs, axis=(1, 2), keepdims=True)
    
    return normalized_imgs




def random_crop_and_resize(img, crop_size, target_size):

    img_height, img_width = img.shape[:2]

    if crop_size[0] > img_width or crop_size[1] > img_height:
        raise ValueError("out of range")

    left = random.randint(0, img_width - crop_size[0])
    top = random.randint(0, img_height - crop_size[1])

    right = left + crop_size[0]
    bottom = top + crop_size[1]

    cropped_img = img[top:bottom, left:right]

    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
    if len(resized_img.shape) == 3:
        if resized_img.shape[2] == 3:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    return resized_img


def load_negatives_from_dir(dir='none_face', output_file='./data/negative_list.pkl', num = None):
    """load images from dir, convert them into gray and dump them into files for further use"""
    img_list = []
    for p, dl, fl in os.walk(dir):
        random.shuffle(fl)
        for file in tqdm(fl[:num]):
            img = cv2.imread(os.path.join(p, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_list.append(img)
    with open(output_file, 'wb') as f:
        pickle.dump(img_list, f)
        f.close()
    print(f'negative imgs dumped into {output_file}')
    return img_list
    

def generate_negative(num=10000, pickle_file = './data/negative_list.pkl'):
    """read images from input file, randomly cropand  normalize them then return negative dataset"""
    with open(pickle_file, 'rb') as f:
        img_list = pickle.load(f)
        f.close()
    data_list = []
    for i in range(num):
        img = random.choice(img_list)
        a = random.randint(12, min(img.shape[0], img.shape[1]))
        croped = random_crop_and_resize(img, (a, a), (24, 24))
        data_list.append(croped)
    
    X = np.array(data_list)
    X = normalize(X)
    y = np.zeros(X.shape[0])
    
    return X, y


def generate_normalized_dataset(input_pickle, output_pickle):
    X, y = data_loader.load_dataset(input_pickle)
    X = normalize(X)
    X, y = data_loader.shuffle_dataset(X, y)
    data_loader.dump_dataset(output_pickle, X, y)
    return 



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
    data_loader.dump_dataset(output_path, X, y)



if __name__ == "__main__":
    generate_positive('./data/pos_data_normalized.pkl')
    load_negatives_from_dir()