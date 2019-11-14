#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:18:05 2019
Adapted from
https://github.com/4uiiurz1/keras-arcface/blob/master/metrics.py
With ideas from @bestfitting
https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109#latest-632983
@author: mzhao
"""

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=10.0, m=0.50, regularizer=None, **kwargs):
        self.n_classes = n_classes
        self.s = tf.dtypes.cast(s, dtype=K.floatx())
        self.m = tf.dtypes.cast(m, dtype=K.floatx())
        self.cos_m = tf.dtypes.cast(math.cos(m), dtype=K.floatx())
        self.sin_m = tf.dtypes.cast(math.sin(m), dtype=K.floatx())
        self.threshold = tf.dtypes.cast(math.cos(math.pi - m), dtype=K.floatx())
        self.mm = tf.dtypes.cast(math.sin(m)*m, dtype=K.floatx())

        self.regularizer = regularizers.get(regularizer)
        super(ArcFace, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True, dtype=K.floatx(),
                                 regularizer=self.regularizer)
        super(ArcFace, self).build(input_shape[0])

    def call(self, inputs):
        x, y = inputs

        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        cos_t = tf.matmul(x, W, name='cos_t')

        cos_t2 = tf.square(cos_t, name='cos_2')
        one = tf.constant(1.0, dtype=K.floatx())
        sin_t2 = tf.subtract(one, cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')

        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, self.cos_m),
                                      tf.multiply(sin_t, self.sin_m), name='cos_mt')

        cond_v = cos_t - self.threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = self.s*(cos_t - self.mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        mask = y
        inv_mask = tf.subtract(one, mask, name='inverse_mask')
        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')
        logits = tf.add(tf.multiply(s_cos_t, inv_mask),
                        tf.multiply(cos_mt_temp, mask), name='arcface_logits')

        out_softmax = tf.nn.softmax(cos_t)
        out_arcface = tf.nn.softmax(logits)

        return tf.concat([out_softmax, out_arcface], axis=1)

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes * 2)

def arcface_loss(n_classes, class_weights=None, softmax_weights=1.0, arcface_weights=1.0):
    """
    Arcface loss function with adjustable ratio between softmax loss and arcface loss
    """
    def loss_function(y_true, y_pred):
#        logits = tf.reshape(y_pred, [-1, 2, n_classes])
#        logit_softmax = logits[:, 0, :]
#        logit_arcface = logits[:, 1, :]
        if class_weights is None:
            loss_softmax = categorical_crossentropy(y_true[:, :n_classes], y_pred[:, :n_classes])
            loss_arcface = categorical_crossentropy(y_true[:, n_classes:], y_pred[:, n_classes:])

            loss = loss_softmax * softmax_weights + loss_arcface * arcface_weights
            return loss
        else:
            weights = K.variable(class_weights)

            y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
            loss_softmax = y_true[:, :n_classes] * K.log(y_pred[:, :n_classes]) * weights
            loss_softmax = -K.sum(loss_softmax, -1)
            loss_arcface = y_true[:, n_classes:] * K.log(y_pred[:, n_classes:]) * weights
            loss_arcface = -K.sum(loss_arcface, -1)

            loss = loss_softmax * softmax_weights + loss_arcface * arcface_weights
            return loss

    return loss_function
