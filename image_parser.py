import os
import json
from matplotlib import pyplot
import numpy as np
from PIL import Image
import pandas as pd
import cPickle as pickle

whale_ids_ref=pd.read_csv("data/train.csv")

## returns the whale id
def get_id(image,whale_ids):
    return whale_ids[whale_ids["Image"]==image].iloc[0]["whaleID"]


cleaned_imgs=[]
whale_ids=[]
whale_names=[]
filenames=[]


annotation_file="master_annotations.json"
with open(annotation_file,"r") as f:
	annotations = json.load(f)


### iterate through annotations in each file extracting
### the cleaned image and the whale_id and storing in arrays
count=0.
for i in annotations:
	count+=1.
	print count/len(annotations)
	### get filename and load image
	filename=i["filename"]
	fileparts = filename.split('/')
	filename = fileparts[len(fileparts)-1]
	filenames.append(filename)
	whale_id=get_id(filename,whale_ids_ref)

	# Modify this line so that 'imgs' path matches yours.
	original = Image.open("imgs/"+whale_id+"/"+filename)


	###crop image to contain just the head:
	for annotation in i["annotations"]:
	    if annotation["class"]=="Head":
		head_annotation=annotation
	length = min(head_annotation["width"],head_annotation["height"])
	top = int(head_annotation["y"])
	left = int(head_annotation["x"])
	bottom = int(head_annotation["y"]+length)
	right = int(head_annotation["x"]+length)

	head_img = original.crop((left,top,right,bottom))

	###resize image to fit to 50x50px, this can be changed later
	#though all images should be the same size...
	img_arr=head_img.resize((64,64))

	##uncomment next line to convert to 2D greyscale
	#img_arr = img_arr.convert("L")

	###rotate and flip images
	"""
	for flip in [True,False]:
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
	whale_ids.append(whale_id)
	whale_names.append(filename)


#### transform whale ids into rows with a 1 and the rest 0:
id_dict={}
ind=0

# Order classes from the order of train.csv
for i in list(set(whale_ids)):
	id_dict[i]=ind
	ind+=1
print id_dict
with open('bin/label_map.bin','w') as fid:
	pickle.dump(id_dict, fid)

# create a reverse map for testing time
inv_map = {v: k for k, v in id_dict.items()}
print inv_map
with open('bin/label_map_reverse.bin','w') as fid:
	pickle.dump(inv_map, fid)

# id_dict => whale_id-label
Y = [0]*len(whale_ids)
for i in range(len(whale_ids)):
	Y[i]=[id_dict[whale_ids[i]]];
with open('bin/Y.bin','w') as fid:
	pickle.dump(Y, fid)

train_data={
    "X":cleaned_imgs,
    "Y":Y,
    "ids":whale_ids,
    'filenames':filenames
}
with open("nn_train_datav3.json","w") as f:
    json.dump(train_data,f)
