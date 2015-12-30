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
from utils_data import split_validation

IMAGE_SIZE = 64


# returns imgs array or filename array
def load_file(test=False, load_filename=False):
  if not test:
    # load train dataset

    if load_filename:
      with open("filenames.json","r") as f:
        org = json.load(f)
      return org['train'], org['label']
    else:
      with open("nn_train_datav3.json","r") as f:
        org = json.load(f)
      train = np.array(org['X'])
      train = train.reshape(len(train), 3, 64, 64)
      labels = org['Y']
      return train, labels
  else:
    # load test dataset

    if load_filename:
      with open("filenames.json","r") as f:
        org = json.load(f)
      return org['test'], org['test']
    else:
      with open("nn_train_datav3_test.json","r") as f:
        org = json.load(f)
      test = np.array(org['X'])
      test = test.reshape(len(test), 3, 64, 64)
      return test, org['filenames']


# for batchiterator training
def load_annotations(test=False):
  if not test:
    # load train dataset
    print 'Loading train data...'
    with open("nn_train_datav3.json","r") as f:
      org = json.load(f)
    filenames = org['filenames']
    labels = org['Y']

    with open("master_annotations.json","r") as f:
      annotations = json.load(f)
    return (filenames, labels), annotations, filenames
  else:
    # load test dataset
    print 'Loading test data...'
    with open("nn_train_datav3_test.json","r") as f:
      org = json.load(f)
    filenames = org['filenames']
    with open("master_annotations_test.json","r") as f:
      annotations = json.load(f)
    return filenames, annotations['annotations'], filenames



def load_data(create_validation=False, load_filename=False):
  print("Loading train data...")
  samples, labels = load_file(False, load_filename)

  # In the new version, labels are already actual whale_ids.
  data = zip(samples, labels) # => ( [(...img array..., label), (...), ...] )
  shuffle(data)

  # unzip it
  samples, labels = zip(*data)

  # load test set
  print 'Loading test data...'
  test_samples, filenames = load_file(True, load_filename)

  # initialize vars
  X_train = samples
  y_train = labels
  X_test = test_samples

  # codes for testing with a part of train dataset
  if create_validation:
    print 'Creating validation data...'
    X_train = list(X_train)
    y_train = list(y_train)

    # create validation set from train
    (X_train, y_train), (X_val, y_val) = split_validation(X_train, y_train, 10)

    # Convert to ndarray
    X_train = np.array(list(X_train))
    y_train = np.array(list(y_train))
    X_val = np.array(list(X_val))
    y_val = np.array(list(y_val))
    X_test = np.array(list(X_test))

    return (X_train, y_train), (X_val, y_val), X_test, filenames
  else:
    # Convert to ndarray
    X_train = np.array(list(X_train))
    y_train = np.array(list(y_train))
    X_test = np.array(list(X_test))

    return (X_train, y_train), X_test, filenames
