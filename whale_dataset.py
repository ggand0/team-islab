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

IMAGE_SIZE=64

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


# returns imgs array or filename array
def load_file(test=False, load_filename=False):
    if not test:
        # train dataset
        with open("nn_train_datav3.json","r") as f:
            org = json.load(f)
        train = np.array(org['X'])
        print train.shape
        #train = change_img_array_shape(train)
        train = train.reshape(len(train), 3, 64, 64)

        print train.shape
        labels = org['Y']  # is this the cause of the bug?

        if load_filename:
            return org['filenames'], labels
        else:
            return train, labels
    else:
        # test dataset
        with open("nn_train_datav3_test.json","r") as f:
            org = json.load(f)
        test = np.array(org['X'])
        print test.shape
        test = test.reshape(len(test), 3, 64, 64)
        print test.shape
        return test, org['filenames']


def load_data(test_with_train=False, load_filename=False):
    print("loading data...")

    samples, labels = load_file(test_with_train, load_filename)
    print len(samples)
    print labels[0]
    print labels[1]
    print type(labels[0])

    # In the new version, labels are already actual whale_ids.
    data = zip(samples, labels) # => ([(...img array..., label), (...), ...])
    shuffle(data)

    # unzip it
    samples, labels = zip(*data)

    # load test set
    if not test_with_train:
        print 'Loading test data...'
        test_samples, filenames = load_file(test=True)

    # codes for testing with a part of train dataset
    if test_with_train:
        nb_test_samples = 1000
        nb_train_samples = len(data)-nb_test_samples
        #X_train = np.zeros((nb_train_samples, 64, 64, 3), dtype="uint8")
        X_test = samples[:nb_test_samples]
        y_test = labels[:nb_test_samples]
        X_train = samples[nb_test_samples+1:]
        y_train = labels[nb_test_samples+1:]#np.zeros((nb_train_samples,), dtype="uint8")
        X_train = np.array(list(X_train))
        y_train = np.array(list(y_train))
        X_test = np.array(list(X_test))
        y_test = np.array(list(y_test))

        return (X_train, y_train), (X_test, y_test)
    else:
        X_train = samples
        X_test = test_samples
        y_train = labels

        # Convert to ndarray
        X_train = np.array(list(X_train))
        y_train = np.array(list(y_train))
        X_test = np.array(list(X_test))
        print 'dataset debug'
        print len(X_train)
        print len(X_test)
        print type(X_train)
        print type(y_train)
        print type(X_test)

        return (X_train, y_train), X_test, filenames


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
