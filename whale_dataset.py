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
    with open("nn_train_datav3.json","r") as f:
      org = json.load(f)
    train = np.array(org['X'])
    train = train.reshape(len(train), 3, 64, 64)
    labels = org['Y']
    if load_filename:
      return org['filenames'], labels
    else:
      return train, labels
  else:
    # load test dataset
    with open("nn_train_datav3_test.json","r") as f:
      org = json.load(f)
    test = np.array(org['X'])
    test = test.reshape(len(test), 3, 64, 64)
    if load_filename:
      return org['filenames'], org['filenames']
    else:
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
    #print type(annotations)
    #print annotations[0]
    return (filenames, labels), annotations
  else:
    # load test dataset
    print 'Loading test data...'
    with open("nn_train_datav3_test.json","r") as f:
      org = json.load(f)
    filenames = org['filenames']
    with open("master_annotations_test.json","r") as f:
      annotations = json.load(f)
    #print type(annotations)
    #print annotations['annotations']
    return filenames, annotations['annotations']


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
  test_samples, filenames = load_file(test=True)

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


# [OLD]Not used
def get_all_labels():
  with open('./train.csv','r') as f:
    reader = csv.reader(f)
    labels=collections.defaultdict(int)
    index=0
    for i,x in enumerate(reader):
      if index==0:# Skip the header
        index+=1
        continue
      #print i # => 1
      #print x # => ['w_7812.jpg', 'whale_48813']

      labels[x[1]] += 1
    return labels

# Not used for now
def load_test_images(img_paths, ids):
  imgs=[]
  # Load img arrays
  for idx, img_path in enumerate(img_paths):
    if ids[idx] == 0:
      original = Image.open(img_path)
      resized = original.resize((IMG_SIZE, IMG_SIZE))
      final_img_arr = np.asarray(resized).flatten().tolist()
      imgs.append(final_img_arr)
    elif ids[idx] == 1:
      original = Image.open('sketch/'+img_path)
      resized = original.resize((IMG_SIZE, IMG_SIZE))
      # Convert grayscale to RGB, since it's read as a grayscale image
      rgbimg = Image.new("RGB", resized.size)
      rgbimg.paste(resized)
      ### transform image to a list of numbers for easy storage.
      final_img_arr = np.asarray(rgbimg).flatten().tolist()
      imgs.append(final_img_arr)
  return  change_img_array_shape(imgs)
