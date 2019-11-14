# -*- coding: utf-8 -*-
"""
Created on Sat Dec 8 23:34:36 2018
ref:
https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
@author: mzhao
"""

import tensorflow as tf
import sirna_config as cfg

def augment(images):
    """
    Runs image augmentation on the GPU (random rotate 90, random flip, random crop)
    """
    if cfg.DTYPE == 'float32':
        dtype = tf.float32
    else:
        dtype = tf.float16

    # Experiments showed that casting on GPU improves training performance
    if images.dtype != dtype:
        images = tf.cast(images, dtype=dtype)
        images = tf.subtract(images, 128.0)
        images = tf.multiply(images, 0.125)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]

    width = tf.cast(width, dtype)
    height = tf.cast(height, dtype)

    if cfg.HORIZONTAL_FLIP:
        images = tf.image.random_flip_left_right(images)

    if cfg.VERTICAL_FLIP:
        images = tf.image.random_flip_up_down(images)

    if cfg.ROTATE:
        images = tf.image.rot90(images,
                    tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    images_cropped = tf.image.random_crop(images, size=[batch_size, cfg.RESIZE[0],
                                                 cfg.RESIZE[1], cfg.RESIZE[2]])

    images_cropped = tf.reshape(images_cropped, [-1, cfg.RESIZE[0], cfg.RESIZE[1],
                                                 cfg.RESIZE[2]])
    return images_cropped
