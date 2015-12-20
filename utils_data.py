from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from six.moves import range
from batch_iterator import BatchIterator


#from __future__ import absolute_import
#from keras.datasets.cifar import load_batch
#from keras.datasets.data_utils import get_file
import numpy as np
import os
import json
from random import shuffle
import cPickle as pickle
import collections
import csv
import cv2
import pandas as pd

DATA_DIR_PATH ='imgs_processed'
IMAGE_SIZE = 128

def mean(array):
  mean=0.0
  for val in array:
    mean += val
  mean /= (len(array)*1.0)
  return mean

class Validator(object):
  def __init__(self, X_val, Y_val, batch_size=32, image_size=128, patience=5, patience_increase=2):
    self.X_val = X_val
    self.Y_val = Y_val
    self.batch_size = batch_size
    self.image_size = image_size
    self.patience = patience
    self.patience_increase = patience_increase
    self.prev_val_score = 0
    self.cur_val_score = 0
    self.best_val_score = 0
    self.tracking_score = 0
    self.being_patient = False
    self.patience_increase_count = 0

  def validate(self, epoch, model):
    '''
    returns True when overfitting (early stopping)
    '''

    print("Validating...")
    self.prev_val_score = self.cur_val_score
    progbar = generic_utils.Progbar(self.X_val.shape[0])
    t=[]

    val_batches = list(BatchIterator(self.X_val, self.Y_val, self.batch_size, self.image_size))
    for X_batch, Y_batch in val_batches:#zip(self.X_val, self.Y_val):
      X_batch_image = []
      for image_path in X_batch:
        # load pre-processed val images from filenames
        processed_img_arr = cv2.imread(DATA_DIR_PATH + '/' + image_path)
        X_batch_image.append(processed_img_arr.reshape(3, self.image_size, self.image_size))
      # convert to ndarray
      X_batch_image = np.array(X_batch_image)
      X_batch_image =  X_batch_image.astype("float32")
      X_batch_image /= 255

      score = model.test_on_batch(X_batch_image, Y_batch)
      valid_accuracy = model.test_on_batch(X_batch_image, Y_batch,accuracy=True) # calc valid accuracy
      progbar.add(X_batch.shape[0], values=[("val loss", score), ("val accuracy", valid_accuracy[1])])
      t.append(score)

    # track the last validation score of the validation
    self.cur_val_score = mean(t)
    if self.cur_val_score < self.best_val_score:
      self.best_val_score = self.cur_val_score
    print ('cur_val_score: %f' % self.cur_val_score)
    print ('prev_val_score: %f' % self.prev_val_score)

    # detect worsening and perform early stopping if needed
    if epoch > self.patience:
      if not self.being_patient and self.cur_val_score > self.prev_val_score or self.being_patient and self.cur_val_score > self.tracking_score:
        if not self.being_patient: # first time
          self.being_patient = True
          self.tracking_score = cur_val_score
        self.patience_increase_count += 1
        print('early stopping: being patient %d / %d' % (self.patience_increase_count, self.patience_increase))
        if self.patience_increase_count > self.patience_increase:
          print('EARLY STOPPING')
          return True
      elif self.being_patient and self.cur_val_score < self.tracking_score:
        self.being_patient = False
        self.patience_increase_count = 0
        print('patience_increase initialized')
    return False


def get_count_dict():
  """
  returns a dictionary of image counts of classes.
  """
  whale_ids = pd.read_csv("data/train.csv")
  count_dict = {}
  for idx, whale_id in enumerate(whale_ids['whaleID']):
    #print whale_id
    count_dict[whale_id] = 0

  for idx, whale_id in enumerate(whale_ids['whaleID']):
    #print whale_id
    count_dict[whale_id] += 1
  return count_dict


def get_val_classes(nb_images=10):
  """
  returns a list of classes for creating a validation set.
  """
  count_dict = get_count_dict()

  # Filter the dict
  count_dict = {k: v for k, v in count_dict.iteritems() if v >= nb_images}
  return count_dict.keys()


def split_validation(train_X, train_Y, nb_images=10):
  """
  splits the validation set from the train set.

  train_X:    image array or filenames
  train_Y:    labels
  nb_images:  split from classes that have more than n images.
  returns:    set of train and val samples
  """

  val_X = []
  val_Y = []
  label_map = pickle.load(open("bin/label_map.bin", "rb"))            # class_name => label(integer)
  reverse_map = pickle.load(open("bin/label_map_reverse.bin", "rb"))  # label(integer) => class_name

  val_classes = get_val_classes(nb_images)
  val_labels = [label_map[val_class] for val_class in val_classes]
  #print val_labels

  val_samples_dict = {}
  for label in val_labels:
    val_samples_dict[label] = []

  """
  # get actual samples and store their index in val_sampes_dict
  # note that this is the indices of train_X and train_Y
  for idx, Y in enumerate(train_Y):
    if Y in val_labels:
      val_samples_dict[Y].append(idx)

  # move the half of samples to val list
  val_indices = []
  for label in val_samples_dict.keys():
    for i in range(0, nb_images/2)
      val_indices.append(val_samples_dict[label][i])

  # create validation set based on indices
  for index in val_indices:
    val_X
  """

  val_samples_moved = {}
  for label in val_labels:
    val_samples_moved[label] = 0
  #print val_samples_moved
  #print type(val_samples_moved)

  # move [nb_images/2] many samples that have val_labels from train list to val list
  # http://stackoverflow.com/questions/6022764/python-removing-list-element-while-iterating-over-list
  for i in xrange(len(train_Y)-1, -1, -1): # from 4543 to -1, step=-1
    label = train_Y[i][0] # train_Y[0] => [141], train_Y[0][0] => 141
    if label in val_labels and val_samples_moved[label] < nb_images/2:
      val_X.append(train_X[i])
      val_Y.append(train_Y[i])
      del train_X[i]
      del train_Y[i]
      val_samples_moved[label] += 1

  print(len(train_X))
  print (len(train_Y))
  print (len(val_X))
  print (len(val_Y))

  return (train_X, train_Y), (val_X, val_Y)