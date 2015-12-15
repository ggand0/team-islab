import os
import json
from matplotlib import pyplot
import numpy as np
from PIL import Image
import pandas as pd
import time
import sys
import cv2

whale_ids_ref=pd.read_csv("train.csv")
## returns the whale id
def get_id(image,whale_ids):
    return whale_ids[whale_ids["Image"]==image].iloc[0]["whaleID"]


cleaned_imgs=[]
#whale_ids=[]
whale_names=[]
filenames = []

annotation_file="master_annotations_test.json"
with open(annotation_file, "r") as f:
    annotations = json.load(f)
    
### iterate through annotations in each file extracting 
### the cleaned image and the whale_id and storing in arrays 
count=0.
print len(annotations['annotations'])
for i in annotations['annotations']:
	count+=1.
	print count/len(annotations['annotations'])

	### get filename and load image
	filename = i["filename"]
	filenames.append(filename)

	#original = Image.open("imgs_test/"+filename)
	#print sys.getsizeof(original)
	original = cv2.imread("imgs_test/"+filename)

	head_annotation = i['annotations']
	#print head_annotation
	#print 'debug0'

	"""
	length = min(head_annotation["width"],head_annotation["height"])
	top = int(head_annotation["y"])
	left = int(head_annotation["x"])
	bottom = int(head_annotation["y"]+length)
	right = int(head_annotation["x"]+length)
	print 'debug1'
	head_img = original.crop((left,top,right,bottom))
	print sys.getsizeof(head_img)
	print 'debug2'
	###resize image to fit to 50x50px, this can be changed later
	#though all images should be the same size...
	img_arr=head_img.resize((64,64))
	print 'debug3'
	"""
	try:
		# Crop and resize the image array
		x=int(head_annotation["x"])
		y=int(head_annotation["y"])
		w=int(head_annotation["width"])
		h=int(head_annotation["height"])
		#print original.shape
		cropped=original[y:y+h,x:x+w]
		#print cropped.shape
		img_arr = cv2.resize(cropped, (64, 64))
	except:
		print filename

	##uncomment next line to convert to 2D greyscale
	#img_arr = img_arr.convert("L")

	###rotate and flip images
	"""for flip in [True,False]:
		for angle in [0,90,180,360]:

			### flip image array
			if flip:
				img_arr = img_arr.transpose(Image.FLIP_LEFT_RIGHT)

			### rotate image array
			img_arr = img_arr.rotate(angle)

			### transform image to a list of numbers to append to list.
			final_img_arr=np.asarray(img_arr).tolist()

			###append data to lists
			cleaned_imgs.append(final_img_arr)
			whale_ids.append(whale_id)
			whale_names.append(filename)
	"""
	### transform image to a list of numbers to append to list.
	final_img_arr=np.asarray(img_arr).tolist()
	cleaned_imgs.append(final_img_arr)
	#whale_ids.append(whale_id)
	whale_names.append(filename)

      
#### transform whale ids into rows with a 1 and the rest 0:
id_dict={}
ind=0
"""for i in list(set(whale_ids)):
	id_dict[i]=ind
	ind+=1
Y = [0]*len(whale_ids)
for i in range(len(whale_ids)):
	Y[i]=[id_dict[whale_ids[i]]];
"""
  
train_data={
    "X":cleaned_imgs,
    'filenames':filenames
    #"Y":Y,
    #"ids":whale_ids
}
with open("nn_train_datav3_test.json","w") as f:
    json.dump(train_data,f)
