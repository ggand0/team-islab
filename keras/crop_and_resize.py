# create dataset that is already cropped and resized
import numpy as np
import json
import os
import cv2

SOURCE_DIR_PATH ='imgs_anil'
TARGET_DIR_PATH ='imgs_processed64'
IMAGE_SIZE = 64
if not os.path.exists(TARGET_DIR_PATH):
  os.makedirs(TARGET_DIR_PATH)

train_annotation_file = "master_annotations.json"
test_annotation_file = "master_annotations_test.json"


# process train images
print 'Processing train images...'
with open(train_annotation_file,"r") as f:
  train_annotations = json.load(f)
with open('nn_train_datav3.json',"r") as f:
  org = json.load(f)
  train_image_paths = org['filenames']
c=0
for image_path, annotation in zip(train_image_paths, train_annotations):
  # load images from filenames
  original = cv2.imread(SOURCE_DIR_PATH + '/' + image_path)
  # crop and resize
  head_annotation = annotation['annotations'][0]
  x = int(head_annotation['x'])
  y = int(head_annotation['y'])
  w = int(head_annotation['width'])
  h = int(head_annotation['height'])
  cropped = original#[y:y+h, x:x+w]
  resized_img_arr = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
  # save image
  cv2.imwrite(TARGET_DIR_PATH + '/' + image_path, resized_img_arr)
  c+=1
  if c % 1000 == 0:
    print c


# process test images
print 'Processing test images...'
with open(test_annotation_file,"r") as f:
  test_annotations = json.load(f)
with open('nn_train_datav3_test.json',"r") as f:
  org = json.load(f)
  test_image_paths = org['filenames']
c=0
for image_path, annotation in zip(test_image_paths, test_annotations['annotations']):
  # load images from filenames
  original = cv2.imread(SOURCE_DIR_PATH + '/' + image_path)
  # crop and resize
  head_annotation = annotation['annotations'][0]
  x = int(head_annotation['x'])
  y = int(head_annotation['y'])
  w = int(head_annotation['width'])
  h = int(head_annotation['height'])
  cropped = original#[y:y+h, x:x+w]
  resized_img_arr = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
  # save image
  cv2.imwrite(TARGET_DIR_PATH + '/' + image_path, resized_img_arr)
  c+=1
  if c % 1000 == 0:
    print c

print 'DONE.'