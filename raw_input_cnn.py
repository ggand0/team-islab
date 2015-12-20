# Modified from cifar10_cnn.py

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

import whale_dataset as wd
import sys
import csv
import pandas as pd
import cPickle as pickle
import cv2
import numpy as np
from utils_csv import export_to_csv
from batch_iterator import BatchIterator

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''


DATA_DIR_PATH ='imgs_processed'
batch_size = 32
nb_classes = 448
nb_epoch = 1
data_augmentation = False
#data_augmentation = True
use_validation = False      # use manually splited validation set
use_batch_iterator = True   # load image arrays batch by batch.

# input image dimensions
img_rows, img_cols = 128,128
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between tran and test sets
if use_batch_iterator:
  # X_train is a list of filenames
  # annotations list contains data for cropping
  (X_train, y_train), annotations = wd.load_annotations(False)
  X_test, annotations_test = wd.load_annotations(True)
else:
  if data_augmentation and use_validation:
    (X_train, y_train), (X_val, y_val), X_test, filenames = wd.load_data(use_validation, use_batch_iterator)
  else:
    (X_train, y_train), X_test, filenames = wd.load_data()

if not use_batch_iterator:
  print('X_train shape:', X_train.shape)
  print(y_train.shape)
  print(X_train.shape[0], 'train samples')
  if use_validation:
    print(X_val.shape[0], 'val samples')
    print(X_train.shape)
    print(X_val.shape)
  print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
if use_validation:
  Y_val = np_utils.to_categorical(y_val, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='full',
                        #input_shape=(img_rows, img_cols, img_channels)))
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adadelta = Adadelta(decay=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

if not use_batch_iterator:
  X_train = X_train.astype("float32")
  X_test = X_test.astype("float32")
  X_train /= 255
  X_test /= 255
  if use_validation:
    X_val = X_val.astype("float32")
    X_val /= 255


if not data_augmentation:
  print("Not using data augmentation or normalization")


  # load BatchIterator
  if use_batch_iterator:

    # train batch by batch
    batches = list(BatchIterator(X_train, Y_train, annotations, batch_size, 128))
    progbar = generic_utils.Progbar(len(X_train))
    total = 0
    for X_batch, Y_batch, A_batch in batches: # X_batch: filenames, A_batch: annotations
      total += len(X_batch) # check the total sample size for debug
      X_batch_image = []
      image_size = 128
      for image_path, annotation in zip(X_batch, A_batch):
        # load pre-processed train images from filenames
        processed_img_arr = cv2.imread(DATA_DIR_PATH + '/' + image_path)
        X_batch_image.append(processed_img_arr.reshape(3, image_size, image_size))

      # convert to ndarray
      X_batch_image = np.array(X_batch_image)
      X_batch_image =  X_batch_image.astype("float32")
      X_batch_image /= 255
      loss, acc = model.train_on_batch(X_batch_image, Y_batch, accuracy=True)
      progbar.add(batch_size, values=[("train loss", loss), ("train acc", acc)])
    print("Saving the trained model...")
    json_string = model.to_json()
    open('noda_model_architecture.json', 'w').write(json_string)


    # predict batch by batch
    print('Predicting...')
    test_batches = list(BatchIterator(X_test, Y_train, annotations_test, batch_size, 128))  # we only use X_test and Y_train, Y_train is a  dummy arg
    progbar = generic_utils.Progbar(len(X_test))                                            # add progress bar since it takes a while
    preds = []
    for X_batch, Y_batch, A_batch in test_batches: # X_test:filenames, A_batch: annotation
      X_batch_image = []
      image_size = 128
      for image_path, annotation in zip(X_batch, A_batch):
        # load pre-processed test images from filenames
        processed_img_arr = cv2.imread(DATA_DIR_PATH + '/' + image_path)
        X_batch_image.append(processed_img_arr.reshape(3, image_size, image_size))
      # convert to ndarray
      X_batch_image = np.array(X_batch_image)
      X_batch_image =  X_batch_image.astype("float32")
      X_batch_image /= 255
      preds_batch = model.predict_on_batch(X_batch_image)
      progbar.add(batch_size, values=[])
      #print(len(preds_batch))  # => batch_size
      #print(preds_batch.shape) # => (batch_size, 448)
      preds += list(preds_batch)
    preds = np.array(preds)
    print(preds.shape)
    print('Saving prediction result...')
    with open('bin/head_%dx%d_noda_preds.bin' % (128, 128),'w') as fid:
      pickle.dump(preds, fid)



  else:
    early_stopping =  EarlyStopping(monitor='val_loss', patience=2)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split = 0.1, callbacks=[early_stopping])

    print('Saving prediction result...')
    preds = model.predict_proba(X_test, verbose=0)
    with open('bin/head_64x64_noda_preds.bin','w') as fid:
      pickle.dump(preds, fid)

    print("Saving the trained model...")
    json_string = model.to_json()
    open('noda_model_architecture.json', 'w').write(json_string)
    #score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    #print('Test score:', score)

  export_to_csv(preds, filenames, 'data/head_64x64_noda.csv')

else:
  print("Using real time data augmentation")

  # this will do preprocessing and realtime data augmentation
  datagen = ImageDataGenerator(
      featurewise_center=True,  # set input mean to 0 over the dataset
      samplewise_center=False,  # set each sample mean to 0
      featurewise_std_normalization=True,  # divide inputs by std of the dataset
      samplewise_std_normalization=False,  # divide each input by its std
      zca_whitening=False,  # apply ZCA whitening
      rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180) def:20
      width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
      height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
      horizontal_flip=True,  # randomly flip images
      vertical_flip=False)  # randomly flip images

  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(X_train)

  # tracks validation loss history for early stopping
  # ref: http://deeplearning.net/tutorial/gettingstarted.html#early-stopping
  val_history = []
  patience = 5         # minimal epochs
  patience_incrase = 2  # if loss increases this many times, we stop training
  patience_incrase_count = 0
  being_patient = False
  cur_val_score = 0
  prev_val_score = 0
  best_val_score = 0
  tracking_score = 0

  for e in range(nb_epoch):
    print('-'*40)
    print('Epoch', e)
    print('-'*40)
    print("Training...")

    # batch train with realtime data augmentation
    progbar = generic_utils.Progbar(X_train.shape[0])
    for X_batch, Y_batch in datagen.flow(X_train, Y_train):
      loss = model.train_on_batch(X_batch, Y_batch)
      progbar.add(X_batch.shape[0], values=[("train loss", loss)])

    print('Saving prediction result...')
    preds = model.predict_proba(X_test, verbose=0)
    with open('bin/head_64x64_preds_tmp.bin','w') as fid:
      pickle.dump(preds, fid)

    print('Saving the trained model...')
    json_string = model.to_json()
    open('da_model_architecture.json', 'w').write(json_string)

    if use_validation:
      # Uncomment for testing with a part of train set
      print("Validating...")
      prev_val_score = cur_val_score
      progbar = generic_utils.Progbar(X_val.shape[0])
      t=[]
      for X_batch, Y_batch in datagen.flow(X_val, Y_val):
        score = model.test_on_batch(X_batch, Y_batch)
        valid_accuracy = model.test_on_batch(X_batch, Y_batch,accuracy=True) # calc valid accuracy
        progbar.add(X_batch.shape[0], values=[("val loss", score), ("val accuracy", valid_accuracy[1])])
        t.append(score)
      mean=0.0
      for score in t:
        mean += score
      mean /= (len(t)*1.0)

      # track the last validation score of the validation
      cur_val_score = mean
      if cur_val_score < best_val_score:
        best_val_score = cur_val_score
      print ('cur_val_score: %f' % cur_val_score)
      print ('prev_val_score: %f' % prev_val_score)

      # detect worsening and perform early stopping if needed
      if e > patience:
        if not being_patient and cur_val_score > prev_val_score or being_patient and cur_val_score > tracking_score:
          if not being_patient: # first time
            being_patient = True
            tracking_score = cur_val_score
          patience_incrase_count += 1
          print('early stopping: being patient %d / %d' % (patience_incrase_count, patience_incrase))
          if patience_incrase_count > patience_incrase:
            print('EARLY STOPPING')
            break
        elif being_patient and cur_val_score < tracking_score:
          being_patient = False
          patience_incrase_count = 0
          print('patience_incraese initialized')


  print('Predicting on the test dataset...')
  preds = model.predict(X_test, verbose=0)
  with open('bin/head_64x64_da_preds.bin','w') as fid:
    pickle.dump(preds, fid)
  export_to_csv(preds, filenames, 'data/head_64x64_da.csv')
