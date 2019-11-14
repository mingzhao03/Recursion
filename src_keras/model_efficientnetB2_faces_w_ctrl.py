#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 23:17:24 2019

@author: mzhao
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dropout, BatchNormalization
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Flatten
from tensorflow.keras.layers import GlobalMaxPooling2D, Lambda, Reshape
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam

import efficientnet.tfkeras as efn

from arcface import ArcFace
from cosface import CosFace
from image_augmentation_tf import augment

#%% model constants
input_shape = (512, 512, 6)
network_shape = (384, 384, 6)
num_features = 512

model_path_head = '../models/all_faces/efnB2_w_ctrl_6chan_'
validation_path_head = '../validations/all_faces/efnB2_w_ctrl_6chan_'
result_path_head = '../results/all_faces/efnB2_w_ctrl_6chan_'

batch_size = 12

#%%
def create_model(l2, num_classes, num_ctrl_classes):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
#    optim = Adam(lr=lr)
#    kwargs = {'kernel_regularizer': regul}

    base_model = efn.EfficientNetB2(input_shape=(network_shape[0], network_shape[1], 3),
                                    weights='imagenet', include_top=False)

    input_tensor = Input(shape=network_shape, dtype=K.floatx())
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                   use_bias=False, padding='same',
                   input_shape=(input_shape[0]+6, input_shape[1]+6, input_shape[2]))
    layers = []
    layers.append(input_tensor)
    layers.append(conv1)
    layers[2:] = base_model.layers[2:]

    new_model = copy_model_graph(layers, base_model, input_tensor)

    weights = base_model.layers[1].get_weights()
    weight0 = weights[0]
    w = np.concatenate((weight0, weight0), axis=2)
    w = w / 2.0
    weights[0] = w
#    weights.append(np.zeros((64),dtype='float32'))

    new_model.layers[1].set_weights(weights)

    inp = Input(shape=input_shape, dtype='uint8')  # 384x384x6
    x = Lambda(augment)(inp)

    for layer in new_model.layers:
        if type(layer) is Conv2D:
            layer.kernel_regularizer = regul
    x = new_model(x)
    x = GlobalMaxPooling2D()(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(512, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    encoder_model = Model(inputs=inp, outputs=x)

    # softmax model for training encoder
    output_softmax = Dense(num_classes, use_bias=False, activation='softmax')(x)
    softmax_model = Model(inputs=inp, outputs=output_softmax)

    #################
    # COMPARE MODEL #
    #################
    mid = 32
    xa_inp = Input(shape=encoder_model.output_shape[1:])
    xb_inp = Input(shape=encoder_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: x[0] - x[1])([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    head = Concatenate()([x1, x2, x3, x4])
    head = Reshape((4, encoder_model.output_shape[1], 1), name='reshape1')(head)
    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    head = Conv2D(mid, (4, 1), activation='relu', padding='valid')(head)
    head = Reshape((encoder_model.output_shape[1], mid, 1))(head)
    head = Conv2D(1, (1, mid), activation='linear', padding='valid')(head)
    head = Flatten()(head)

    compare_model = Model([xa_inp, xb_inp], head)

    # process encoding from control
    # compare the current features to all controls
    features_controls = Input(shape=[num_ctrl_classes, encoder_model.output_shape[1]])
    fs = Lambda(lambda x: tf.unstack(x, axis=1))(features_controls)
#    def create_mask(features_controls):
#        # Use a function with a Keras Lambda layer wrapper to resolve a tensorflow issue.
#        # https://stackoverflow.com/questions/50715928/valueerror-output-tensors-to-a-model-must-be-the-output-of-a-tensorflow-layer
#        max_abs_features = K.max(K.abs(features_controls), axis=2)
#        mask = tf.greater(max_abs_features, K.epsilon())
#        mask = tf.expand_dims(tf.expand_dims(tf.dtypes.cast(mask, K.floatx()), axis=-1), axis=-1)
#        return mask
#    mask = Lambda(create_mask)(features_controls)
    comps = []
    for f in fs:
        comp = compare_model([x, f])
        comps.append(comp)
    c = Concatenate()(comps)
    c = Reshape((num_ctrl_classes, encoder_model.output_shape[1], 1))(c)
#    c = Lambda(lambda x: tf.math.multiply(x[0], x[1]))([c, mask])

#    compare = Lambda(compare_features)([x, features_controls])
    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    compare = Conv2D(mid, (num_ctrl_classes, 1), activation='relu', padding='valid')(c)
    compare = Reshape((encoder_model.output_shape[1], mid, 1))(compare)
    compare = Conv2D(1, (1, mid), activation='linear', padding='valid')(compare)
    compare = Flatten(name='flatten2')(compare)

    feature_model = Model(inputs=[inp, features_controls], outputs=compare)

    label = Input(shape=(num_classes,))

    output_arcface = ArcFace(num_classes, regularizer=regul)([compare, label])
    arcface_model = Model([inp, features_controls, label], output_arcface)

    output_cosface = CosFace(num_classes, regularizer=regul)([compare, label])
    cosface_model = Model([inp, features_controls, label], output_cosface)

    return encoder_model, softmax_model, feature_model, arcface_model, cosface_model

def copy_model_graph(layers, base_model, input_tensor):
    if not len(layers) == len(base_model.layers):
        raise ValueError('The two models does not have the same number of layers.')

    base_model_outputs = [l.output.name for l in base_model.layers]

    xs = []
    for index in range(len(base_model.layers)):
        if index == 0:
            # input layer
            x = input_tensor
            xs.append(x)
            continue

        old_layer_inputs = base_model.layers[index].input
        if type(old_layer_inputs) == list:
            activation = []
            for layer_input in old_layer_inputs:
                input_index = base_model_outputs.index(layer_input.name)
                activation.append(xs[input_index])
            x = layers[index](activation)
            xs.append(x)
        else:
            input_index = base_model_outputs.index(old_layer_inputs.name)
            x = layers[index](xs[input_index])
            xs.append(x)

    model = Model(input_tensor, x)

    return model

#%%
if __name__ == "__main__":
    print('Creating models...')
    lr = 0.001
    l2 = 0.0
    num_classes = 1108
    num_ctrl_classes = 31
    enc, soft, fea, arc, cos, sph = create_model(l2, num_classes, num_ctrl_classes)
