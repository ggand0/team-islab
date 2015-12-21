from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from six.moves import range
from batch_iterator import BatchIterator
from keras.preprocessing.image import *

import numpy as np
import os
import json
from random import shuffle
import cPickle as pickle
import collections
import csv
import cv2
import pandas as pd

# modified from keras.preprocessing.image.ImageDataGenerator
def random_transform(x, rotation_range, width_shift_range, height_shift_range,
  do_horizontal_flip, do_vertical_flip):
  if rotation_range:
      x = random_rotation(x, rotation_range)
  if width_shift_range or height_shift_range:
      x = random_shift(x, width_shift_range, height_shift_range)
  if do_horizontal_flip:
      if random.random() < 0.5:
          x = horizontal_flip(x)
  if do_vertical_flip:
      if random.random() < 0.5:
          x = vertical_flip(x)

  # TODO:
  # zoom
  # barrel/fisheye
  # shearing
  # channel shifting
  return x


def augment(image):
  '''
  perform data augmentation
  '''
  return random_transform(image, 180, 0.2, 0.2, True, True)


# debug
if __name__ == '__main__':
  DATA_DIR_PATH ='imgs_processed'
  image_path = 'w_0.jpg'
  img_arr = cv2.imread(DATA_DIR_PATH + '/' + image_path)
  img_arr = augment(img_arr)
  cv2.imshow('dst_rt', img_arr)