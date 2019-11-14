#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:52:03 2019

@author: mzhao
"""

#%% Library imports
import numpy as np
import pandas as pd
import os
import random
#from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
#from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
#from tensorflow.keras.callbacks import ModelCheckpoint

from rectified_adam import RAdam

from tqdm import tqdm

from sirna_utilities import load_site, show_image, get_onehot_label

import model_efficientnetB2_faces_w_ctrl as m

import sirna_config as cfg

from arcface import arcface_loss
from cosface import cosface_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#dtype = 'float16'
#K.set_floatx(dtype)

#K.set_epsilon(1e-4)

#%% Constants
N_FOLDS = 3
NUM_WORKERS = 12

FOLD = 0

CELL_LINE_INDEX = 3

IS_SOFTMAX_TRAINED = False
IS_ARCFACE_TRAINED = False
IS_COSFACE_TRAINED = False

#%% Load train and test data and split into folds along experiments.
train_df = pd.read_csv('../data/train.csv')
train_ctrl_df = pd.read_csv('../data/train_controls.csv')

test_df = pd.read_csv('../data/test.csv')
test_ctrl_df = pd.read_csv('../data/test_controls.csv')

train_cellline = []
train_ctrl_cellline = []
test_cellline = []
test_ctrl_cellline = []
for cell_line in cfg.CELL_LINES:
    train = train_df.loc[(train_df['experiment'].str.contains(cell_line))].reset_index(drop=True)
    train_cellline.append(train)
    train_ctrl = train_ctrl_df.loc[(train_ctrl_df['experiment'].str.contains(cell_line))].drop(['well_type'], axis=1).reset_index(drop=True)
    train_ctrl_cellline.append(train_ctrl)
    test = test_df.loc[(test_df['experiment'].str.contains(cell_line))].reset_index(drop=True)
    test_cellline.append(test)
    test_ctrl = test_ctrl_df.loc[(test_ctrl_df['experiment'].str.contains(cell_line))].drop(['well_type'], axis=1).reset_index(drop=True)
    test_ctrl_cellline.append(test_ctrl)


# split along experiments, 3 fold.
# fold 0 HEPG 5536, HUVEC 11078, RPE 5539, U2OS 2216
# fold 1 HEPG 3320, HUVEC 5540, RPE 2216, U2OS 1108

train_all = []
val_all = []

if FOLD == 0:
    train_all.append(train_cellline[0].iloc[:5536, :])
    val_all.append(train_cellline[0].iloc[5536:, :])
    train_all.append(train_cellline[1].iloc[:11078, :])
    val_all.append(train_cellline[1].iloc[11078:, :])
    train_all.append(train_cellline[2].iloc[:5539, :])
    val_all.append(train_cellline[2].iloc[5539:, :])
    train_all.append(train_cellline[3].iloc[:2216, :])
    val_all.append(train_cellline[3].iloc[2216:, :])
elif FOLD == 1:
    train_all.append(train_cellline[0].iloc[3320:, :])
    val_all.append(train_cellline[0].iloc[:3320, :])
    train_all.append(train_cellline[1].iloc[5540:, :])
    val_all.append(train_cellline[1].iloc[:5540, :])
    train_all.append(train_cellline[2].iloc[2216:, :])
    val_all.append(train_cellline[2].iloc[:2216, :])
    train_all.append(train_cellline[3].iloc[1108:, :])
    val_all.append(train_cellline[3].iloc[:1108, :])
else:
    val = train_cellline[0].iloc[3320:5536, :]
    val_all.append(val)
    train = train_cellline[0][~train_cellline[0].isin(val)].dropna()
    train_all.append(train)

    val = train_cellline[1].iloc[5540:11078, :]
    val_all.append(val)
    train = train_cellline[1][~train_cellline[1].isin(val)].dropna()
    train_all.append(train)

    val = train_cellline[2].iloc[2216:5539, :]
    val_all.append(val)
    train = train_cellline[2][~train_cellline[2].isin(val)].dropna()
    train_all.append(train)

    val = train_cellline[3].iloc[1108:2216, :]
    val_all.append(val)
    train = train_cellline[3][~train_cellline[3].isin(val)].dropna()
    train_all.append(train)

#%% Select out the data for the cell line to be trained
CELL_LINE = cfg.CELL_LINES[CELL_LINE_INDEX]
print('Training models for cell line: ' + CELL_LINE)

test = test_cellline[CELL_LINE_INDEX]
val = val_all[CELL_LINE_INDEX]
train = train_all[CELL_LINE_INDEX]
train_ctrl = train_ctrl_cellline[CELL_LINE_INDEX]
test_ctrl = test_ctrl_cellline[CELL_LINE_INDEX]
train = pd.concat([train, train_ctrl, test_ctrl])

train_all.append(train_ctrl_df.drop(['well_type'], axis=1))
train_all.append(test_ctrl_df.drop(['well_type'], axis=1))
train_all = pd.concat(train_all)
ctrl_all = pd.concat([train_ctrl, test_ctrl]).reset_index(drop=True)

#%% Build dictionary for control experiments
exp_to_ctrl = {}
# Add test experiments
for index in range(len(test)):
    exp_id = test.iloc[index, 0][:-4]
    if exp_id not in exp_to_ctrl:
        image_info = test.iloc[index, :]
        info_neg_ctrl = ctrl_all.loc[(ctrl_all['experiment'] == image_info['experiment']) &
                                     (ctrl_all['plate'] == image_info['plate'])]
        exp_to_ctrl[exp_id] = info_neg_ctrl
# Add train experiments
for index in range(len(train)):
    exp_id = train.iloc[index, 0][:-4]
    if exp_id not in exp_to_ctrl:
        image_info = train.iloc[index, :]
        info_neg_ctrl = ctrl_all.loc[(ctrl_all['experiment'] == image_info['experiment']) &
                                     (ctrl_all['plate'] == image_info['plate'])]
        exp_to_ctrl[exp_id] = info_neg_ctrl


#%% Show a sample image
image_info = train.iloc[0, :]
image = load_site(image_info, 1)
y_aug_range, x_aug_range = tuple(x - y for x, y in zip(image.shape[:2], m.input_shape[:2]))

sirna = image_info.iloc[-1]
show_image(image, sirna)


#%% Generator for calculating image embedding.
class EncoderGen(Sequence):
    def __init__(self, data_df, call_index, batch_size=64, verbose=1):
        super(EncoderGen, self).__init__()
        self.data_df = data_df
        self.call_index = call_index
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data_df) - start, self.batch_size)
        site = self.call_index % 2 + 1
        a = np.zeros((size,) + m.input_shape, dtype=K.floatx())
        for i in range(size):
            image_info = self.data_df.iloc[start + i, :]
            img_a = load_site(image_info.iloc[:4], site)
            a[i, :, :, :] = img_a
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data_df) + self.batch_size - 1) // self.batch_size

#%% Generator for training softmax. Return image only
class SoftmaxTrainGen(Sequence):
    def __init__(self, data_df, num_classes, start_class,
                 steps=10, batch_size=32):
        """
        @param steps
        @param batch_size
        """
        super(SoftmaxTrainGen, self).__init__()
        self.data_df = data_df
        self.steps = steps
        self.num_classes = num_classes
        self.start_class = start_class
        self.batch_size = batch_size
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.data_df)*2)
        size = end - start
        assert size > 0
        a = np.zeros((size,) + m.input_shape, dtype=K.floatx())
        b = np.zeros((size, self.num_classes), dtype=K.floatx())

        for i in range(0, size):
            df_index = self.train_indices[start + i] // 2
            site = (self.train_indices[start + i] % 2) + 1
            image_info = self.data_df.iloc[df_index, :]
            img_a = load_site(image_info.iloc[:4], site)
            a[i, :, :, :] = img_a
            b[i, :] = get_onehot_label(image_info['sirna'], self.num_classes,
                                       self.start_class)

        return a, b

    def on_epoch_end(self):
#        if self.steps <= 0:
#            return
        self.steps -= 1
        indices = np.arange(len(self.data_df)*2)
        random.shuffle(indices)
        self.train_indices = indices

    def __len__(self):
        return (len(self.data_df) * 2 + self.batch_size - 1) // self.batch_size

#%% Generator for training arcface/cosface. Returns images with control embedding
def calculate_ctrl_feature_avg(ctrl_feature, feature_length, ctrl_all):
    # calculate the average of control well features to replace missing controls
    ctrl_feature_avg = np.zeros((cfg.NUM_CTRL_CLASSES, feature_length), dtype=K.floatx())
    ctrl_count = np.zeros(cfg.NUM_CTRL_CLASSES)
    for item_index in range(len(ctrl_all)):
        feature = ctrl_feature[item_index, :]
        sirna = ctrl_all.sirna.values[item_index]
        ctrl_index = sirna - cfg.NUM_CLASSES
        ctrl_feature_avg[ctrl_index, :] += feature
        ctrl_count[ctrl_index] += 1
    for class_index in range(cfg.NUM_CTRL_CLASSES):
        ctrl_feature_avg[class_index, :] /= ctrl_count[class_index]
    return ctrl_feature_avg

class TrainingFaceGen(Sequence):
    def __init__(self, data_df, num_classes, start_class, ctrl_all,
                 is_arcface=False, steps=10, batch_size=32):
        """
        @param steps
        @param batch_size
        """
        super(TrainingFaceGen, self).__init__()
        self.data_df = data_df
        self.feature_length = encoder_model.output_shape[-1]
        self.ctrl_all = ctrl_all
        self.steps = steps
        self.num_classes = num_classes
        self.start_class = start_class
        self.is_arcface = is_arcface
        self.batch_size = batch_size
        self.on_epoch_end()
        
    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.data_df)*2)
        size = end - start
        assert size > 0
        a = np.zeros((size,) + m.input_shape, dtype=K.floatx())
        b = np.zeros((size, cfg.NUM_CTRL_CLASSES, self.feature_length), dtype=K.floatx())
        c = np.zeros((size, self.num_classes), dtype=K.floatx())

        for i in range(0, size):
            df_index = self.train_indices[start + i] // 2
            site = (self.train_indices[start + i] % 2) + 1
            image_info = self.data_df.iloc[df_index, :]
            img_a = load_site(image_info.iloc[:4], site)
            a[i, :, :, :] = img_a

            # find corresponding control features
            exp_id = image_info['id_code'][:-4]
            info_ctrl = exp_to_ctrl[exp_id]
            sirna_count = np.zeros(cfg.NUM_CTRL_CLASSES)
            for item_index in range(len(info_ctrl)):
                feature = self.ctrl_feature[info_ctrl.index[item_index], :]
                sirna = info_ctrl.sirna.values[item_index]
                ctrl_index = sirna - cfg.NUM_CLASSES
                b[i, ctrl_index, :] += feature
                sirna_count[ctrl_index] += 1
            # normalize duplicate sirna
            for class_index in range(cfg.NUM_CTRL_CLASSES):
                if sirna_count[class_index] > 1:
                    b[i, class_index, :] /= sirna_count[class_index]
                else:
                    # use the average control features in case a control is missing
                    b[i, class_index, :] = self.ctrl_feature_avg[class_index, :]

            c[i, :] = get_onehot_label(image_info['sirna'], self.num_classes, self.start_class)

        if self.is_arcface:
            l = np.tile(c, 2)
        else:
            l = c

        return [a, b, c], l

    def on_epoch_end(self):
        if self.steps <= 0:
            return
        self.steps -= 1
        indices = np.arange(len(self.data_df)*2)
        random.shuffle(indices)
        self.train_indices = indices
        ctrl_features = []
        for index in tqdm(range(4)):
            ctrl_feature = encoder_model.predict_generator(EncoderGen(self.ctrl_all, call_index=index, verbose=0),
                                            max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                            verbose=0)
            ctrl_features.append(ctrl_feature)
        self.ctrl_feature = np.mean(np.array(ctrl_features), axis=0)
        self.ctrl_feature_avg = calculate_ctrl_feature_avg(self.ctrl_feature,
                                        self.feature_length, self.ctrl_all)

    def __len__(self):
        return (len(self.data_df) * 2 + self.batch_size - 1) // self.batch_size


#%% Generator to compute final feature embedding with control embedding input
class FeatureGen(Sequence):
    def __init__(self, data_df, ctrl_feature, ctrl_feature_avg, call_index,
                 batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data_df = data_df
        self.call_index = call_index
        self.batch_size = batch_size
        self.verbose = verbose
        self.ctrl_feature = ctrl_feature
        self.ctrl_feature_avg = ctrl_feature_avg
        self.feature_length = ctrl_feature.shape[-1]
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data_df) - start, self.batch_size)
        site = self.call_index % 2 + 1
        a = np.zeros((size,) + m.input_shape, dtype=K.floatx())
        b = np.zeros((size, cfg.NUM_CTRL_CLASSES, self.feature_length), dtype=K.floatx())
        for i in range(size): 
            image_info = self.data_df.iloc[start + i, :]
            img_a = load_site(image_info.iloc[:4], site)
            a[i, :, :, :] = img_a
            
            # find corresponding control features
            exp_id = image_info['id_code'][:-4]
            info_ctrl = exp_to_ctrl[exp_id]
            sirna_count = np.zeros(cfg.NUM_CTRL_CLASSES)
            for item_index in range(len(info_ctrl)):
                feature = self.ctrl_feature[info_ctrl.index[item_index], :]
                sirna = info_ctrl.sirna.values[item_index]
                ctrl_index = sirna - cfg.NUM_CLASSES
                b[i, ctrl_index, :] += feature
                sirna_count[ctrl_index] += 1
            # normalize duplicate sirna
            for class_index in range(cfg.NUM_CTRL_CLASSES):
                if (sirna_count[class_index] > 1):
                    b[i, class_index, :] /= sirna_count[class_index]
                else:
                    # use the average control features in case a control is missing
                    b[i, class_index, :] = self.ctrl_feature_avg[class_index, :]
                    
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.data_df) + self.batch_size - 1) // self.batch_size

#%%
total_classes = cfg.NUM_CLASSES + cfg.NUM_CTRL_CLASSES

class_weights = np.ones(total_classes)
class_weights[cfg.NUM_CLASSES:] *= 0.1

print('Constructing models......')
encoder_model, softmax_model, feature_model, arcface_model, cosface_model = m.create_model(l2 = 0.0004, 
                            num_classes=total_classes, num_ctrl_classes=cfg.NUM_CTRL_CLASSES)

#%%
def set_lr(model, lr):
    K.set_value(model.optimizer._lr, float(lr))

def get_lr(model):
    return K.get_value(model.optimizer._lr)

#%% Train softmax model
def compute_val_acc_softmax(val_df, num_classes):
    results_val = []
    for index in tqdm(range(8)):
        result_val = softmax_model.predict_generator(EncoderGen(val_df, 
                        call_index=index, verbose=0), 
                        max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                              verbose=0)
        results_val.append(result_val[:,:num_classes])
    result_avg = np.mean(np.array(results_val), axis=0)
    preds = np.argmax(result_avg, axis=1)
    
    correct_preds = preds == val_df['sirna']
    acc = np.sum(correct_preds) / len(correct_preds)
    return acc, preds, result_avg
    
def make_steps_softmax(data_df, val_df, step):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global steps, histories, best_acc_softmax
    
    # Train the model for 'step' epochs
    for _ in range(step):
        history = softmax_model.fit_generator(SoftmaxTrainGen(data_df, steps=1, 
            num_classes=total_classes, start_class=0, batch_size=m.batch_size), 
            class_weight=class_weights, initial_epoch=steps, epochs=steps + 1, 
            max_queue_size=m.batch_size * 4, workers=NUM_WORKERS, verbose=1).history
        steps += 1

    # Collect history data
    history['epochs'] = steps
    history['lr'] = get_lr(softmax_model)
    print(history['epochs'], history['lr'])
    histories.append(history)
    # save the work in progress model
    model_temp_name = m.model_path_head + CELL_LINE + '_softmax_temp.h5'
    softmax_model.save_weights(model_temp_name)
    
    if val_df is not None:
        acc, _, _ = compute_val_acc_softmax(val_df, total_classes)
        print('Accuracy on validation set:', acc)
        if (acc > best_acc_softmax):
            print('Accuracy is better than previous best of ', best_acc_softmax, ', saving model.')
            best_acc_softmax = acc
            model_best_name = m.model_path_head + CELL_LINE + '_softmax_best_' + str(FOLD) + '.h5'
            softmax_model.save_weights(model_best_name)

#%% Calculate validation accuracy with arcface/cosface models
def compute_val_acc_faces(train, val, ctrl_all):
    # calculate encode vectors on contrl data
    encodes_ctrl = []
    for index in tqdm(range(8)):
        encode_ctrl = encoder_model.predict_generator(EncoderGen(ctrl_all, call_index=index, verbose=0), 
                                            max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                            verbose=0)
        encodes_ctrl.append(encode_ctrl)
    encode_ctrl = np.mean(np.array(encodes_ctrl), axis=0)
    feature_length = encode_ctrl.shape[-1]
    encode_ctrl_avg = calculate_ctrl_feature_avg(encode_ctrl,
                                        feature_length, ctrl_all)
    
    # calculate feature vectors
    features_val = []
    for index in tqdm(range(4)):
        feature_val = feature_model.predict_generator(FeatureGen(val, encode_ctrl, encode_ctrl_avg, call_index=index, verbose=0), 
                                            max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                            verbose=0)
        features_val.append(feature_val)
    features_val_avg = np.mean(np.array(features_val), axis=0)
    # L2 norm
    features_val_avg = features_val_avg / np.linalg.norm(features_val_avg, ord=2, axis=1, keepdims=True)
    
    features_train = []
    for index in tqdm(range(4)):
        feature_train = feature_model.predict_generator(FeatureGen(train, encode_ctrl, encode_ctrl_avg, call_index=index, verbose=0), 
                                            max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                            verbose=0)
        features_train.append(feature_train)
    features_train_avg = np.mean(np.array(features_train), axis=0)
    # L2 norm
    features_train_avg = features_train_avg / np.linalg.norm(features_train_avg, ord=2, axis=1, keepdims=True)

    # calculate the average features for sirnas.
    features_sirna = np.zeros([cfg.NUM_CLASSES, features_train_avg.shape[1]])
    for index in range(len(train)):
        sirna = train.iloc[index, 4]
        if sirna < cfg.NUM_CLASSES:
            features_sirna[sirna, :] += features_train_avg[index, :]
    # L2 norm
    features_sirna = features_sirna / np.linalg.norm(features_sirna, ord=2, axis=1, keepdims=True)
    features_sirna = np.nan_to_num(features_sirna)

    # compare similarities.
    score = np.dot(features_val_avg, features_sirna.T)
    max_index = np.argmax(score, axis=1)
#    preds = train['sirna'].values[max_index]
    preds = max_index

    correct_preds = preds == val['sirna']
    acc = np.sum(correct_preds) / len(correct_preds)

    return acc, preds, score

#%% Predict on test data with arcface/cosface models
def compute_test_faces(encoder_model, feature_model, suffix, train, test, val, ctrl_all):
    # run on test data
    # calculate encode vectors on contrl data
    encodes_ctrl_all = []
    for index in tqdm(range(8)):
        encode_ctrl = encoder_model.predict_generator(EncoderGen(ctrl_all, call_index=index, verbose=0), 
                                                max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                                verbose=0)
        encodes_ctrl_all.append(encode_ctrl)
    encode_ctrl = np.mean(np.array(encodes_ctrl_all), axis=0)
    feature_length = encode_ctrl.shape[-1]
    encode_ctrl_avg = calculate_ctrl_feature_avg(encode_ctrl, 
                                            feature_length, ctrl_all)
        
    # calculate feature vectors
    features_test = []
    for index in tqdm(range(16)):
        feature_test = encoder_model.predict_generator(FeatureGen(test, encode_ctrl, encode_ctrl_avg, call_index=index, verbose=0), 
                                                max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                                verbose=0)
        features_test.append(feature_test)
    features_test_avg = np.mean(np.array(features_test), axis=0)
    # L2 norm
    features_test_avg = features_test_avg / np.linalg.norm(features_test_avg, 
                                                           ord=2, axis=1, keepdims=True) 
    
    known = pd.concat([train, val])    
    features_known = []
    for index in tqdm(range(8)):
        feature_known = encoder_model.predict_generator(FeatureGen(known, encode_ctrl, encode_ctrl_avg, call_index=index, verbose=0), 
                                            max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                                            verbose=0)
        features_known.append(feature_known)
    features_known_avg = np.mean(np.array(features_known), axis=0)
    # L2 norm
    features_known_avg = features_known_avg / np.linalg.norm(features_known_avg, ord=2, axis=1, keepdims=True)
    
    features_sirna = np.zeros([cfg.NUM_CLASSES, features_known_avg.shape[1]])
    for index in range(len(known)):
        sirna = known.iloc[index, 4]
        if sirna < cfg.NUM_CLASSES:
            features_sirna[sirna, :] += features_known_avg[index, :]
    # L2 norm
    features_sirna = features_sirna / np.linalg.norm(features_sirna, ord=2, axis=1, keepdims=True)
    features_sirna = np.nan_to_num(features_sirna)
    
    score = np.dot(features_test_avg, features_sirna.T)
    max_index = np.argmax(score, axis=1)
    #preds = known['sirna'].values[max_index]
    preds = max_index
            
    test['sirna'] = preds
    result_name = m.result_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.csv'
    test.to_csv(result_name, index=False)
    result_score_name = m.result_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.npy'
    np.save(result_score_name, score)


#%% Train arcface model
def make_steps_arcface(data_df, val_df, ctrl_all, suffix, step):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global steps, histories, best_acc_arcface
    
    # Train the model for 'step' epochs
    for _ in range(step):
        history = arcface_model.fit_generator(
            TrainingFaceGen(data_df, steps=1, num_classes=total_classes, start_class=0,  
                         ctrl_all=ctrl_all, is_arcface=True, batch_size=m.batch_size),
            initial_epoch=steps, epochs=steps + 1, max_queue_size=m.batch_size * 4, 
            workers=NUM_WORKERS, verbose=1).history
        steps += 1

    # Collect history data
    history['epochs'] = steps
    history['lr'] = get_lr(arcface_model)
    print(history['epochs'], history['lr'])
    histories.append(history)
    # save the work in progress model
    model_temp_name = m.model_path_head + CELL_LINE + suffix + '_temp.h5'
    arcface_model.save_weights(model_temp_name)

    if val_df is not None:
        acc, _, _ = compute_val_acc_faces(data_df, val_df, ctrl_all)
        print('Accuracy on validation set:', acc)
        if (acc > best_acc_arcface):
            print('Accuracy is better than previous best of ', best_acc_arcface, ', saving model.')
            best_acc_arcface = acc
            model_best_name = m.model_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.h5'
            arcface_model.save_weights(model_best_name)


#%% Train cosface model
def make_steps_cosface(data_df, val_df, ctrl_all, suffix, step):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global steps, histories, best_acc_cosface
    
    # Train the model for 'step' epochs
    for _ in range(step):
        history = cosface_model.fit_generator(
            TrainingFaceGen(data_df, steps=1, num_classes=total_classes, start_class=0,  
                         ctrl_all=ctrl_all, is_arcface=True, batch_size=m.batch_size),
            initial_epoch=steps, epochs=steps + 1, max_queue_size=m.batch_size * 4, 
            workers=NUM_WORKERS, verbose=1).history
        steps += 1

    # Collect history data
    history['epochs'] = steps
    history['lr'] = get_lr(cosface_model)
    print(history['epochs'], history['lr'])
    histories.append(history)
    # save the work in progress model
    model_temp_name = m.model_path_head + CELL_LINE + suffix + '_temp.h5'
    cosface_model.save_weights(model_temp_name)

    if val_df is not None:
        acc, _, _ = compute_val_acc_faces(data_df, val_df, ctrl_all)
        print('Accuracy on validation set:', acc)
        if (acc > best_acc_cosface):
            print('Accuracy is better than previous best of ', best_acc_cosface, ', saving model.')
            best_acc_cosface = acc
            model_best_name = m.model_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.h5'
            cosface_model.save_weights(model_best_name)


#%% start training

histories = []
steps = 0
best_acc_softmax = 0.0
best_acc_arcface = 0.0
best_acc_cosface = 0.0

print('Start training softmax model...')

# train softmax model
# train last layer
for layer in softmax_model.layers:
    layer.trainable = True
softmax_model.layers[2].trainable = False # freeze the underlying model
softmax_model.compile(RAdam(lr=0.01), loss='categorical_crossentropy', 
                      metrics=['acc'])
                
make_steps_softmax(train, None, 2)
set_lr(softmax_model, 512e-5)
make_steps_softmax(train, None, 2)
set_lr(softmax_model, 256e-5)
make_steps_softmax(train, None, 2)
set_lr(softmax_model, 128e-5)
make_steps_softmax(train, None, 2)
        
# train all layers
for layer in softmax_model.layers:
    layer.trainable = True
softmax_model.layers[2].trainable = True # unfreeze the underlying model
            
softmax_model.compile(RAdam(lr=0.005), loss='categorical_crossentropy', 
                          metrics=['acc'])
                
make_steps_softmax(train, None, 4)
set_lr(softmax_model, 256e-5)
make_steps_softmax(train, None, 8)               
                
set_lr(softmax_model, 128e-5)
for _ in range(5):
    make_steps_softmax(train, val, 5)
set_lr(softmax_model, 64e-5)
for _ in range(5):
    make_steps_softmax(train, val, 5)
set_lr(softmax_model, 32e-5)
for _ in range(5):
    make_steps_softmax(train, val, 5)
set_lr(softmax_model, 16e-5)
for _ in range(5): make_steps_softmax(train, val, 2)
set_lr(softmax_model, 4e-5)
for _ in range(5): make_steps_softmax(train, val, 2)

#%% evaluate on cv data
# load best softmax model
print('Calculating validation and test data on Softmax model...')
model_best_name = m.model_path_head + CELL_LINE + '_softmax_best_' + str(FOLD) + '.h5'
print('Loading model weights from ' + model_best_name)

softmax_model.load_weights(model_best_name)
acc, val_preds, val_score = compute_val_acc_softmax(val, cfg.NUM_CLASSES)
print('Softmax accuracy on validation set:', acc)
val_result = val.copy()
val_result['pred'] = val_preds
val_name = m.validation_path_head + CELL_LINE + '_softmax_best_' + str(FOLD) + '.csv'
val_result.to_csv(val_name, index=False)
val_score_name = m.validation_path_head + CELL_LINE + '_softmax_best_' + str(FOLD) + '.npy'
np.save(val_score_name, val_score)
        
encoder_model_best_name = m.model_path_head + CELL_LINE + '_encoder_softmax_best_' + str(FOLD) + '.h5'
encoder_model.save_weights(encoder_model_best_name)

# run on test data
results_val = []
for index in tqdm(range(16)):
    result_val = softmax_model.predict_generator(EncoderGen(test, call_index=index, verbose=0), 
                          max_queue_size=NUM_WORKERS * 4, workers=NUM_WORKERS,
                          verbose=0)
    results_val.append(result_val[:,:cfg.NUM_CLASSES])
result_avg = np.mean(np.array(results_val), axis=0)
preds = np.argmax(result_avg, axis=1)
        
test['sirna'] = preds
result_name = m.result_path_head + CELL_LINE + '_softmax_best_' + str(FOLD) + '.csv'
test.to_csv(result_name, index=False)
result_score_name = m.result_path_head + CELL_LINE + '_softmax_best_' + str(FOLD) + '.npy'
np.save(result_score_name, result_avg)
        

#%% Train arcface model with 0 softmax weight
print('Start training arcface model...')
suffix = '_arcface'

# Restore encoder model weights
encoder_model.load_weights(encoder_model_best_name)

# Train last layers
for layer in arcface_model.layers:
    layer.trainable = True
arcface_model.layers[0].trainable = False
arcface_model.layers[1].trainable = False
arcface_model.layers[2].trainable = False

arcface_model.compile(RAdam(lr=0.005), loss=arcface_loss(n_classes=total_classes, 
                            class_weights=class_weights, softmax_weights=0.0, 
                            arcface_weights=1.0), metrics=['acc'])
make_steps_arcface(train, None, ctrl_all, suffix, 4)

set_lr(arcface_model, 0.002)
make_steps_arcface(train, None, ctrl_all, suffix, 2)
set_lr(arcface_model, 0.001)
make_steps_arcface(train, None, ctrl_all, suffix, 2)
set_lr(arcface_model, 0.0005)
make_steps_arcface(train, None, ctrl_all, suffix, 2)        

for layer in arcface_model.layers:
    layer.trainable = True
arcface_model.compile(RAdam(lr=256e-5), loss=arcface_loss(n_classes=total_classes, 
                          class_weights=class_weights, softmax_weights=0.0, 
                          arcface_weights=1.0), 
                          metrics=['acc'])
for _ in range(2):
    make_steps_arcface(train, None, ctrl_all, suffix, 5)
set_lr(arcface_model, 128e-5)
for _ in range(4):
    make_steps_arcface(train, val, ctrl_all, suffix, 5)
set_lr(arcface_model, 64e-5)
for _ in range(5):
    make_steps_arcface(train, val, ctrl_all, suffix, 4)
set_lr(arcface_model, 32e-5)
for _ in range(5):
    make_steps_arcface(train, val, ctrl_all, suffix, 4)
set_lr(arcface_model, 16e-5)
for _ in range(5):
    make_steps_arcface(train, val, ctrl_all, suffix, 4)
    
set_lr(arcface_model, 4e-5)
for _ in range(3):
    make_steps_arcface(train, val, ctrl_all, suffix, 5)
set_lr(arcface_model, 1e-5)
for _ in range(3):
    make_steps_arcface(train, val, ctrl_all, suffix, 5)
        
#%% evaluate on cv data
# load bast arcface model
print('Calculating validation and test data on Arcface model...')
suffix = '_arcface'
model_best_name = m.model_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.h5'
print('Loading model weights from ' + model_best_name)

arcface_model.load_weights(model_best_name)
acc, val_preds, val_score = compute_val_acc_faces(train, val, ctrl_all)
print('Arcface accuracy on validation set:', acc)
val_result = val.copy()
val_result['pred'] = val_preds
val_name = m.validation_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.csv'
val_result.to_csv(val_name, index=False)
val_score_name = m.validation_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.npy'
np.save(val_score_name, val_score)
        
compute_test_faces(encoder_model, feature_model, suffix, train, test, val, ctrl_all)

#%% train cosface model
print('Start training cosface model...')
suffix = '_cosface'

# Restore encoder model weights
encoder_model.load_weights(encoder_model_best_name)

# Train last layers
for layer in cosface_model.layers:
    layer.trainable = True
cosface_model.layers[0].trainable = False
cosface_model.layers[1].trainable = False
cosface_model.layers[2].trainable = False

cosface_model.compile(RAdam(lr=0.005), loss=cosface_loss(n_classes=total_classes, 
                            class_weights=class_weights, softmax_weights=0.0, 
                            arcface_weights=1.0), metrics=['acc'])
make_steps_cosface(train, None, ctrl_all, suffix, 4)

set_lr(cosface_model, 0.002)
make_steps_cosface(train, None, ctrl_all, suffix, 2)
set_lr(cosface_model, 0.001)
make_steps_cosface(train, None, ctrl_all, suffix, 2)
set_lr(cosface_model, 0.0005)
make_steps_cosface(train, None, ctrl_all, suffix, 2)

for layer in cosface_model.layers:
    layer.trainable = True
cosface_model.compile(RAdam(lr=256e-5), loss=cosface_loss(n_classes=total_classes,
                      class_weights=class_weights, softmax_weights=0.0, cosface_weights=1.0), 
              metrics=['acc'])
for _ in range(2):
    make_steps_cosface(train, None, ctrl_all, suffix, 5)
set_lr(cosface_model, 128e-5)
for _ in range(4):
    make_steps_cosface(train, val, ctrl_all, suffix, 5)
set_lr(cosface_model, 64e-5)
for _ in range(5):
    make_steps_cosface(train, val, ctrl_all, suffix, 4)
set_lr(cosface_model, 32e-5)
for _ in range(5):
    make_steps_cosface(train, val, ctrl_all, suffix, 4)
set_lr(cosface_model, 16e-5)
for _ in range(5):
    make_steps_cosface(train, val, ctrl_all, suffix, 4)
set_lr(cosface_model, 4e-5)
for _ in range(3):
    make_steps_cosface(train, val, ctrl_all, suffix, 5)
set_lr(cosface_model, 1e-5)
for _ in range(3):
    make_steps_cosface(train, val, ctrl_all, suffix, 5)
        
#%% evaluate on cv data
# load bast cosface model
print('Calculating validation and test data on Cosface model...')
suffix = '_cosface'
model_best_name = m.model_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.h5'
print('Loading model weights from ' + model_best_name)
cosface_model.load_weights(model_best_name)
acc, val_preds, val_score = compute_val_acc_faces(train, val, ctrl_all)
print('cosface accuracy on validation set:', acc)
val_result = val.copy()
val_result['pred'] = val_preds
val_name = m.validation_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.csv'
val_result.to_csv(val_name, index=False)
val_score_name = m.validation_path_head + CELL_LINE + suffix + '_best_' + str(FOLD) + '.npy'
np.save(val_score_name, val_score)
    
compute_test_faces(encoder_model, feature_model, suffix, train, test, val, ctrl_all)
