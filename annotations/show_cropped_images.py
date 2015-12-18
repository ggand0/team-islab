import json
import numpy as np
import matplotlib.pyplot as plt
#import cv2
import math

def convert(color):
  #return math.floor(color == 1.0 ? 255 : color * 256.0)
  return math.floor(255 if color == 1.0 else color * 256.0)

with open('nn_train_data_sample.json') as data_file:
    org = json.load(data_file)
    #print data['imgs']
    org = np.array(org['imgs'][0])
    #print org.shape
    org = org.reshape((50,50,3))
    data = org.view('float64')
    data[:]=org
    #print data[0][0]
    #print data.dtype

    # Covert 0-1 RGB to 0-255.
    i=0
    j=0
    for line in data:
      j=0
      for row in data[i]:
        data[i][j] = data[i][j]/255.0#[ float(col/255.0) for col in data[i][j] ]
        j+=1
      i+=1

    plt.imshow(data)
    #plt.imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    plt.show()