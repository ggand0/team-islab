from PIL import Image
import numpy as np
import cv2


class BatchIterator(object):
  '''
  Receives all image paths, and yield each mini batch images.
  '''
  def __init__(self, all_data, all_target, batch_size=32, image_size=128):
    self._data = all_data
    self._target = all_target
    #self._annotations = annotations
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
      #a = self._annotations[self._i*self._batch_size:]
    else:
      #print self._batch_size
      #print self._i*self._batch_size:(self._i+1)*self._batch_size

      # get batch data
      x = self._data[self._i*self._batch_size:(self._i+1)*self._batch_size]
      y = self._target[self._i*self._batch_size:(self._i+1)*self._batch_size]
      #a = self._annotations[self._i*self._batch_size:(self._i+1)*self._batch_size]
    self._i += 1

    return x, y
