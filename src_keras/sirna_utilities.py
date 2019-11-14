#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:52:30 2019
Utility functions
@author: mzhao
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from keras import backend as K

from scipy.ndimage import affine_transform

import sirna_config as cfg

def load_site(image_info, site, nchannels=6):
    image_path = os.path.join(cfg.TRAIN_DATA_PATH, image_info.experiment, 'Plate'+str(int(image_info.plate)))
    if not os.path.exists(image_path):
        image_path = os.path.join(cfg.TEST_DATA_PATH, image_info.experiment, 'Plate'+str(int(image_info.plate)))

    filename_prefix = image_info['well'] + '_s' + str(site) + '_w'

    image = np.zeros(shape=(512, 512, nchannels), dtype=np.uint8)
    for index in range(nchannels):
        image_name = filename_prefix + str(int(index+1)) + '.png'
        image[:, :, index] = imread(os.path.join(image_path, image_name))

    return image

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def load_image_with_crop(image_info, site, target_shape, x_aug_range, y_aug_range):
    """
    Crop a random region on the image and add image augmentation
    """
    image = load_site(image_info, site)
    x0 = random.randint(0, x_aug_range)     # x origin coordinates
    y0 = random.randint(0, y_aug_range)     # y origin coordinates
    x1 = x0 + target_shape[1]
    y1 = y0 + target_shape[0]

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * target_shape[0]], [0, 1, -0.5 * target_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / target_shape[0], 0, 0], [0, (x1 - x0) / target_shape[1], 0],
                             [0, 0, 1]]), trans)
    # Add augmentation
    trans = np.dot(build_transform(
                random.uniform(-45, 45),
                random.uniform(-5, 5),
                1.0,
                1.0,
                random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
                random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
                ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    new_image = np.zeros(target_shape)
    for channel in range(image.shape[-1]):
        img = image[:, :, channel].reshape(image.shape[:-1])
        img = affine_transform(img, matrix, offset, output_shape=target_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
        new_image[:, :, channel] = img
    new_image = new_image.reshape(target_shape)

    # Normalize to zero mean and unit variance
    new_image -= np.mean(new_image, keepdims=True)
    new_image /= np.std(new_image, keepdims=True) + K.epsilon()
    return new_image

def load_full_image(image_info, site):
    image = load_site(image_info, site)
    image = image.astype(np.float32)
    image -= 128.0
    image /= 8.0
    return image

def get_onehot_label(label, num_classes, start_class):
    index = label - start_class # [int(i) for i in label.split()]
    labels = np.zeros(num_classes, dtype='float32')
#    for index in indices:
    labels[int(index)] = 1.0

    return labels

def show_image(image, image_sirna):
    _, axes = plt.subplots(1, 6, figsize=(40, 16))
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        ax.set_title('channel {}'.format(i + 1) + ', sirna ' + str(image_sirna))
        _ = ax.imshow(image[:, :, i], cmap='gray')
    plt.show()

def lap_random_approximation(score, num_iterations=1):
    """
    Use random sampling to approximate best solution to lap problem.
    Not exact solution, but much lower running time.
    """
    lowest_score = 1000000.
    data_length = score.shape[0]
    for _ in range(num_iterations):
        x_temp = np.arange(data_length, dtype=np.int32)
        y_temp = np.zeros(data_length, dtype=np.int32)
        np.random.shuffle(x_temp)
        count = 0
        yindices = np.arange(data_length, dtype=np.int32)
        for index in x_temp:
            y_index = yindices[np.argmin(score[index, yindices])]
            y_temp[count] = y_index
            y_delete = np.argmax(yindices == y_index)
            yindices = np.delete(yindices, y_delete, 0)
            count += 1

        score_rand = np.sum(score[x_temp, y_temp])
        if score_rand < lowest_score:
            target_x = x_temp
            target_y = y_temp
            lowest_score = score_rand
    
    x = np.zeros(data_length, dtype=np.int32)
    y = np.arange(data_length, dtype=np.int32)
    count = 0
    for index in target_y:
        x[index] = target_x[count]
        count += 1

    score_final = np.sum(score[x, y])
    return score_final, x, y
