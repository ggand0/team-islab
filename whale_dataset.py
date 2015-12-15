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

# reshape img array to (rgb, location(1D))
def change_img_array_shape(imgs):
    result=[]
    i=0

    for img in imgs:
        print img.shape
        print img[0].shape

        # img = 64,64,3
        index=0
        new_img = [[],[],[]]
        #new_img = np.array([np.array([]),np.array([]),np.array([])])
        for col in img:
            if index % 3 == 0:#R
                print col
                fds
                new_img[0].append(col)
                #print index
                #print new_img[0]
                #np.append(new_img[0], col)

            elif index % 3 == 1:#G
                new_img[1].append(col)
                #np.append(new_img[1], col)
            elif index % 3 == 2:#B
                new_img[2].append(col)
                #np.append(new_img[2], col)
            index += 1
        
        new_img = np.array(new_img)
        print new_img[0][0]
        print new_img.shape
        print new_img.size
        print len(new_img[0])
        print len(new_img[1])
        print len(new_img[2])
        result.append(new_img.reshape(3,64,64))
        #print new_img.shape# => (3, 4096)
        #print new_img.reshape(3,64,64)
        #np.append(result, new_img.reshape(3,64,64))
        i+=1
        if i % 100==0: print i
    return np.array(result)


def load_file(test=False):
    if not test:
        with open("nn_train_datav3.json","r") as f:
            org = json.load(f)
        train = np.array(org['X'])
        print train.shape
        #print train[0][0]
        #print train[0][1]
        #train = change_img_array_shape(train)
        train = train.reshape(len(train), 3, 64, 64)
        print train.shape
        labels = org['Y']
        return train, labels
    else:
        with open("nn_train_datav3_test.json","r") as f:
            org = json.load(f)
        train = np.array(org['X'])
        print train.shape
        train = train.reshape(len(train), 3, 64, 64)
        print train.shape
        return train


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


def load_data():
    print("loading data...")

    """
    print("processing data...")
    #print data
    X = np.array(data["X"])
    X = X.astype("float32")
    X /= 255.
    X = X.reshape(X.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)

    # take the transpose of the matrix
    print np.array(data["Y"])
    print X.shape
    print Y.shape
    print X[0]
    Y = np.array(data["Y"]).transpose()[0]
    nb_classes=np.max(Y)+1
    """

    imgs, labels = load_file()
    print len(imgs)
    print labels[0]
    print labels[1]
    print type(labels[0])

    # In the new version, labels are already actual whale_ids.
    """
    # Ceate a dict for labels
    classes = get_all_labels()
    print(len(classes))
    print(classes['whale_70138'])
    print(classes[u'whale_70138'])
    classMap = dict(zip(classes.iterkeys(),xrange(len(classes))))
    index = 0
    labels = [ classMap[label] for label in labels ]
    with open('label_map.bin','w') as fid:
        pickle.dump(labels, fid)
    """
    data = zip(imgs, labels) # => ([(...img array..., label), (...), ...])
    shuffle(data)

    # unzip it
    imgs, labels = zip(*data)
    
    #nb_test_samples = 1000#10000
    #nb_train_samples = len(data)-nb_test_samples
    # load test set
    print 'Loading test data...'
    test_imgs = load_file(test=True)

    #X_train = np.zeros((nb_train_samples, 50, 50, 3), dtype="uint8")
    #X_test = imgs[:nb_test_samples]
    #y_test = labels[:nb_test_samples]
    #X_train = imgs[nb_test_samples+1:]
    #y_train = labels[nb_test_samples+1:]#np.zeros((nb_train_samples,), dtype="uint8")
    X_train = imgs
    X_test = test_imgs
    y_train = labels


    # Convert to ndarray
    X_train = np.array(list(X_train))
    y_train = np.array(list(y_train))
    X_test = np.array(list(X_test))
    #y_test = np.array(list(y_test))
    print 'dataset debug'
    print len(X_train)
    print len(X_test)
    print type(X_train)
    print type(y_train)
    print type(X_test)
    

    return (X_train, y_train), X_test#(X_test, y_test)



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
