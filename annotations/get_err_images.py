import os
from os import listdir
from os.path import isfile, join
import shutil

# edit these dir names according to your assign
dir_path='../5_err'
output_path='../5_err_org_images'
image_path='imgs_test'

# get file names
image_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

# create output dir if necessary
if not os.path.exists(output_path):
    os.makedirs(output_path)

# copy images
for image_file in image_files:
  if image_file == '.DS_Store':
    continue
  print image_file
  shutil.copyfile(image_path+'/'+image_file, output_path+'/'+image_file)

print 'DONE.'