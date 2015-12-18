import numpy as np
import csv
import os
import shutil

with open('./train.csv','r') as f:
    reader = csv.reader(f)

    a = []
    for i,x in enumerate(reader):
        imgname = './imgs/'+x[0]
        dirname = './imgs/'+x[1]
        if i > 0:
            if os.path.exists(dirname)==False:
                os.mkdir(dirname)
            if os.path.exists(imgname)==True:
                shutil.move(imgname,dirname)