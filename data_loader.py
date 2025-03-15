#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: data_utils
Author: lyk
Date: 2025-02-12
Description: A utility module for loading and processing image datasets. 
             It loads positive and negative image samples, resizes them to 24x24 pixels, 
             and saves the data as a pickle file.
"""
import cv2
import os
import numpy as np
import pickle
import random
from tqdm import tqdm

def dump_dataset(output_file, X, y):
    # export as pickle file
    with open(output_file, 'wb') as f:
        pickle.dump((X, y), f)
    print(f"dataset dumped to {output_file}")


def load_dataset(path: str = 'training_data_pos.pkl'):
    with open(path, 'rb+') as f:
        X, y = pickle.load(f)
        print(f'load dataset from {path}')
        return X, y


def merge_dataset(XA: np.ndarray, yA: np.ndarray, XB: np.ndarray, yB: np.ndarray):
    X = np.concatenate((XA, XB), axis=0)
    y = np.concatenate((yA, yB), axis=0)
    return X, y


def shuffle_dataset(X, y):
    tmp = [(xx, yy) for xx, yy in zip(X, y)]
    random.shuffle(tmp)
    X = np.array([xx for xx, yy in tmp])
    y = np.array([yy for xx, yy in tmp])
    return X, y