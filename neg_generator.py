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


def load_negatives_from_dir(dir='none_face', output_file='negative_list.pkl', num = None):
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
    

def generate_negative(img_list, num=10000):
    """read images from pickle file, normalize them and generate negative dataset and save it into pickle file"""
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


if __name__ == "__main__":
    generate_normalized_dataset('./data/all_pos_data_uint8.pkl', './data/pos_data_normalized.pkl')