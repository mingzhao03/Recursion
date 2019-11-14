#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:18:05 2019
Adapted from
https://github.com/4uiiurz1/keras-cosface/blob/master/metrics.py
With ideas from @bestfitting
https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109#latest-632983
@author: mzhao
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy

class CosFace(Layer):
    def __init__(self, n_classes=10, s=10.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = tf.dtypes.cast(s, dtype=K.floatx())
        self.m = tf.dtypes.cast(m, dtype=K.floatx())
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True, dtype=K.floatx(),
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs

        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)

        cos_t = tf.matmul(x, W, name='cos_t')

        logits = tf.multiply(self.s, tf.subtract(cos_t, tf.multiply(y, self.m)),
                             name='cosface_logits')

        out_softmax = tf.nn.softmax(cos_t)
        out_cosface = tf.nn.softmax(logits)

        return tf.concat([out_softmax, out_cosface], axis=1)

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes * 2)

def cosface_loss(n_classes, class_weights=None, softmax_weights=1.0, cosface_weights=1.0):
    """
    Cosface loss function with adjustable ratio between softmax loss and cosface loss
    """

    def loss_function(y_true, y_pred):
#        logits = tf.reshape(y_pred, [-1, 2, n_classes])
#        logit_softmax = logits[:, 0, :]
#        logit_cosface = logits[:, 1, :]
        if class_weights is None:
            loss_softmax = categorical_crossentropy(y_true[:, :n_classes], y_pred[:, :n_classes])
            loss_cosface = categorical_crossentropy(y_true[:, n_classes:], y_pred[:, n_classes:])

            loss = loss_softmax * softmax_weights + loss_cosface * cosface_weights
            return loss
        else:
            weights = K.variable(class_weights)

            y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
            loss_softmax = y_true[:, :n_classes] * K.log(y_pred[:, :n_classes]) * weights
            loss_softmax = -K.sum(loss_softmax, -1)
            loss_cosface = y_true[:, n_classes:] * K.log(y_pred[:, n_classes:]) * weights
            loss_cosface = -K.sum(loss_cosface, -1)

            loss = loss_softmax * softmax_weights + loss_cosface * cosface_weights
            return loss

    return loss_function
