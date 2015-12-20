from PIL import Image
import numpy as np
import cv2

DATA_DIR_PATH ='imgs'

# Receives all image paths, and yield each mini batch images.
class BatchIterator(object):
  def __init__(self, all_data, all_target, annotations, batch_size=32, image_size=128):
    self._data = all_data
    self._target = all_target
    self._annotations = annotations
    self._n_samples = len(self._data) # use length of data because all_target could be a dummy data
    self._i = 0
    self._batch_size = batch_size
    self._image_size = image_size

    if batch_size > self._n_samples:
      raise Exception("Invalid batch_size !")

    if self._n_samples % batch_size == 0:
      self._batch_num = self._n_samples / batch_size
      #print 'debug1', self._batch_num
    else:
      self._batch_num = self._n_samples / batch_size + 1
      #print 'debug2', self._batch_num

  def __iter__(self):
    return self

  def next(self):
    if self._i > (self._batch_num -1):
      raise StopIteration

    if self._i == (self._batch_num -1):
      # get batch data
      x = self._data[self._i*self._batch_size:]
      y = self._target[self._i*self._batch_size:]
      a = self._annotations[self._i*self._batch_size:]
    else:
      #print self._batch_size
      #print self._i*self._batch_size:(self._i+1)*self._batch_size

      # get batch data
      x = self._data[self._i*self._batch_size:(self._i+1)*self._batch_size]
      y = self._target[self._i*self._batch_size:(self._i+1)*self._batch_size]
      a = self._annotations[self._i*self._batch_size:(self._i+1)*self._batch_size]
    self._i += 1

    """
    X = []
    #print zip(x, a)
    idx=0
    for image_path, annotation in zip(x, a):
      #print image_path
      #print annotation
      print idx
      idx+=1

      # load images from filenames
      original = cv2.imread(DATA_DIR_PATH + '/' + image_path)

      # crop and resize
      head_annotation = annotation['annotations'][0]
      xx = int(head_annotation['x'])
      yy = int(head_annotation['y'])
      w = int(head_annotation['width'])
      h = int(head_annotation['height'])

      cropped = original[yy:yy+h, xx:xx+w]
      resized_img_arr = cv2.resize(cropped, (self._image_size, self._image_size))

      # change img array shape
      resized_img_arr = resized_img_arr.reshape(self._image_size, self._image_size, 3)
      X.append(resized_img_arr)
    """

    return x, y, a
