# -*- coding: utf-8 -*-
import os
import json
from matplotlib import pyplot
import numpy as np
from PIL import Image
import pandas as pd

whale_ids_ref=pd.read_csv("train.csv")
## returns the whale id
def get_id(image,whale_ids):
    return whale_ids[whale_ids["Image"]==image].iloc[0]["whaleID"]

cleaned_imgs=[]
whale_ids=[]
whale_names=[]

annotation_files=os.listdir("annotations")
print len(annotation_files)

for idx,filepath in enumerate(annotation_files):
    #if idx == 1:
	#break

    #print 'debug1:',idx
    print filepath
    with open("annotations/"+ filepath) as f:
        annotations = json.load(f)

    ### iterate through annotations in each file extracting
    ### the cleaned image and the whale_id and storing in arrays
    for idx2,i in enumerate(annotations):
    	#if idx2 == 1:
        #break
        ### get filename and load image
        filename=i["filename"]
        #print 'debug2:'

        fileparts = filename.split('/')
        filename = fileparts[len(fileparts)-1]
        whale_id=get_id(filename,whale_ids_ref)
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
		resized_head_img=head_img.resize((50,50))
		### transform image to a list of numbers for easy storage.
		final_img_arr = np.asarray(resized_head_img).flatten().tolist()

		###append data to lists
		cleaned_imgs.append(final_img_arr)
		whale_ids.append(whale_id)
		whale_names.append(filename)

print 'cleaned_imgs: %s' % len(cleaned_imgs)
train_data={
    "imgs":cleaned_imgs,
    "ids":whale_ids,
    "names":whale_names
}
with open("./nn_train_data.json","w") as f:
    json.dump(train_data,f)
