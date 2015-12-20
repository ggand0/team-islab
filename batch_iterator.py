from PIL import Image
import numpy as np
import cv2

DATA_DIR_PATH ='imgs'

# reshape img array to (rgb, location(1D))
# rect: dict
def change_img_array_shape(img_paths, ids, annotation):
    result=[]
    imgs=[]

    # Load img arrays
    for idx, img_path in enumerate(img_paths):
        if ids[idx] == 0:
            original = Image.open(DATA_DIR_PATH+'/'+img_path)
            cropped=original[y:y+h,x:x+w]
            resized = original.resize((IMG_SIZE, IMG_SIZE))
            final_img_arr = np.asarray(resized).flatten().tolist()
            imgs.append(final_img_arr)
        elif ids[idx] == 1:

            original = Image.open(DATA_DIR_PATH+'/'+img_path)
            cropped=original[y:y+h,x:x+w]
            resized = original.resize((IMG_SIZE, IMG_SIZE))
            # Convert grayscale to RGB, since it's read as a grayscale image
            rgbimg = Image.new("RGB", resized.size)
            rgbimg.paste(resized)
            ### transform image to a list of numbers for easy storage.
            final_img_arr = np.asarray(rgbimg).flatten().tolist()
            imgs.append(final_img_arr)

    i=0
    for img in imgs:
        # img = 64,64,3
        index=0
        new_img = [[],[],[]]
        for col in img:
            if index % 3 == 0:#R
                new_img[0].append(col)
            elif index % 3 == 1:#G
                new_img[1].append(col)
            elif index % 3 == 2:#B
                new_img[2].append(col)
            index += 1

        new_img = np.array(new_img)
        result.append(new_img.reshape(3,IMG_SIZE,IMG_SIZE))
        i+=1
        if i % 100==0: print i
    return np.array(result)


# Receives all image paths, and yield each mini batch images.
class BatchIterator(object):
  def __init__(self, all_data, all_target, annotations, batch_size=32, image_size=128):
    self._data = all_data
    self._target = all_target
    self._annotations = annotations
    self._n_samples = len(self._target)
    self._i = 0
    self._batch_size = batch_size
    self._image_size = image_size

    if batch_size > self._n_samples:
      raise Exception("Invalid batch_size !")

    if self._n_samples % batch_size == 0:
      self._batch_num = self._n_samples / batch_size
      print 'debug1', self._batch_num
    else:
      self._batch_num = self._n_samples / batch_size + 1
      print 'debug2', self._batch_num

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
