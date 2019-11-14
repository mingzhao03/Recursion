# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:16:09 2018

@author: Ming Zhao
"""

#%% Parameters for image augmentation
RESIZE = (384, 384, 6) # (width, height) tuple or None
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
ROTATE = True
DTYPE = 'float32'

#%% General parameters
DATA_PATH = '../data/'
TRAIN_DATA_PATH = '../data/train/'
TEST_DATA_PATH = '../data/test/'

CELL_LINES = ['HEPG', 'HUVEC', 'RPE', 'U2OS']
NUM_CLASSES = 1108
NUM_CTRL_CLASSES = 31
