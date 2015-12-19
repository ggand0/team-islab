from __future__ import absolute_import
from keras.datasets.cifar import load_batch
from keras.datasets.data_utils import get_file
import numpy as np
import os
import json
from random import shuffle
import cPickle as pickle
import collections
import csv
import pandas as pd


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

  train_X:    image array
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

  print len(train_X)
  print len(train_Y)
  print len(val_X)
  print len(val_Y)

  return (train_X, train_Y), (val_X, val_Y)